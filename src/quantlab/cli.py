"""Command line entry point.

    quantlab generate --days 10 --out data/ticks.csv
    quantlab run --ticks data/ticks.csv --strategy all
    quantlab sweep --ticks data/ticks.csv --strategy mean_reversion
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import TickGenConfig, generate_ticks, load_ticks, periods_per_year, ticks_to_bars, write_ticks_csv
from .engine import CapitalConstraints, CostModel, run_backtest
from .report import _table, comparison_report, strategy_report
from .strategies import STRATEGIES


def _add_dataclass_args(p: argparse.ArgumentParser, cls, prefix: str = "") -> None:
    g = p.add_argument_group(cls.__name__)
    for f in fields(cls):
        default = f.default
        flag = f"--{prefix}{f.name.replace('_', '-')}"
        if isinstance(default, bool):
            g.add_argument(flag, type=lambda s: s.lower() in {"1", "true", "yes"}, default=default,
                           metavar="BOOL", help=f"(default {default})")
        else:
            typ = float if default is None or isinstance(default, float) else type(default)
            g.add_argument(flag, type=typ, default=default, help=f"(default {default})")


def _from_args(cls, ns: argparse.Namespace, prefix: str = ""):
    return cls(**{f.name: getattr(ns, prefix.replace("-", "_") + f.name) for f in fields(cls)})


def _load(ns):
    ticks, ingest = load_ticks(ns.ticks)
    bars = ticks_to_bars(ticks, ns.freq)
    return bars, ingest, periods_per_year(bars)


def cmd_generate(ns) -> None:
    cfg = TickGenConfig(n_days=ns.days, ticks_per_second=ns.rate, seed=ns.seed, s0=ns.s0)
    t0 = time.perf_counter()
    ticks = generate_ticks(cfg)
    path = write_ticks_csv(ticks, ns.out)
    print(f"wrote {len(ticks):,} ticks over {cfg.n_days} sessions to {path} "
          f"({path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.2f}s)")


def cmd_run(ns) -> None:
    bars, ingest, ppy = _load(ns)
    costs, capital = _from_args(CostModel, ns), _from_args(CapitalConstraints, ns)
    names = list(STRATEGIES) if ns.strategy == "all" else [ns.strategy]
    outdir = Path(ns.reports) / datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in names:
        strat = STRATEGIES[name]()
        t0 = time.perf_counter()
        res = run_backtest(bars, strat.generate(bars, ppy), ppy, costs, capital, name=name)
        elapsed = time.perf_counter() - t0
        results.append(res)
        text = strategy_report(res, strat, bars, ingest, ns.freq, ns.rolling_window)
        (outdir / f"{name}.txt").write_text(text)
        res.frame.to_csv(outdir / f"{name}_ledger.csv")
        print(f"{name:<15} net {res.stats['total_return']:+.3%}  Sharpe {res.stats['sharpe']:+.2f}  "
              f"MDD {res.stats['max_drawdown']:.3%}  ({elapsed * 1e3:.1f} ms)")
        if ns.print:
            print(text)

    if len(results) > 1:
        cmp_text = comparison_report(results)
        (outdir / "comparison.txt").write_text(cmp_text)
        print("\n" + cmp_text)
    print(f"reports written to {outdir}/")


SWEEP_GRIDS = {
    "mean_reversion": {"lookback": [10, 20, 30, 60], "entry_z": [1.5, 2.0, 2.5], "exit_z": [0.0, 0.2, 0.5]},
    "momentum": {"fast": [10, 20, 30], "slow": [60, 120, 240], "deadband": [0.1, 0.3, 0.5]},
}


def cmd_sweep(ns) -> None:
    bars, _, ppy = _load(ns)
    costs, capital = _from_args(CostModel, ns), _from_args(CapitalConstraints, ns)
    grid = SWEEP_GRIDS[ns.strategy]
    keys = list(grid)
    rows, t0 = [], time.perf_counter()
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        if params.get("fast", 0) >= params.get("slow", np.inf):
            continue
        st = run_backtest(bars, STRATEGIES[ns.strategy](**params).generate(bars, ppy), ppy, costs, capital).stats
        rows.append((params, st))
    elapsed = time.perf_counter() - t0
    rows.sort(key=lambda x: -np.nan_to_num(x[1]["sharpe"], nan=-1e9))
    print(f"{len(rows)} backtests in {elapsed:.2f}s ({elapsed / len(rows) * 1e3:.1f} ms each)\n")
    print(_table(keys + ["net", "gross", "Sharpe", "Sortino", "MDD", "trips"],
                 [[str(p[k]) for k in keys] + [f"{s['total_return']:+.3%}", f"{s['gross_total_return']:+.3%}",
                   f"{s['sharpe']:+.2f}", f"{s['sortino']:+.2f}", f"{s['max_drawdown']:.3%}",
                   str(s["trade_n_trades"])] for p, s in rows[: ns.top]]))
    print("\n  In-sample ranking on one synthetic path: treat as sensitivity analysis, not selection.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="quantlab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="simulate a mock tick tape to CSV")
    g.add_argument("--days", type=int, default=10)
    g.add_argument("--rate", type=float, default=3.0, help="mean ticks per second")
    g.add_argument("--s0", type=float, default=100.0)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/ticks.csv")
    g.set_defaults(fn=cmd_generate)

    for name, fn, help_ in [("run", cmd_run, "backtest and write reports"),
                            ("sweep", cmd_sweep, "parameter sensitivity grid")]:
        r = sub.add_parser(name, help=help_)
        r.add_argument("--ticks", default="data/ticks.csv")
        r.add_argument("--freq", default="1min", help="bar size, pandas offset alias")
        choices = (["all"] + list(STRATEGIES)) if name == "run" else list(SWEEP_GRIDS)
        r.add_argument("--strategy", choices=choices, default=choices[0])
        if name == "run":
            r.add_argument("--reports", default="reports")
            r.add_argument("--rolling-window", type=int, default=60)
            r.add_argument("--print", action="store_true", help="also print full reports")
        else:
            r.add_argument("--top", type=int, default=15)
        _add_dataclass_args(r, CostModel)
        _add_dataclass_args(r, CapitalConstraints)
        r.set_defaults(fn=fn)

    ns = p.parse_args(argv)
    ns.fn(ns)


if __name__ == "__main__":
    main()
