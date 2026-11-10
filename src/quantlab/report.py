"""Plain-text analytical reports (fixed-width, diff-able, terminal friendly)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from . import metrics
from .data import IngestReport
from .engine import BacktestResult

W = 78


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _pct(x: float, d: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:,.{d}f}%"


def _num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


def _money(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"${x:,.0f}"


def _section(title: str) -> str:
    return f"\n{title.upper()}\n{'-' * W}"


def _kv(rows: list[tuple[str, str]], indent: int = 2) -> str:
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{' ' * indent}{k:<{width}}  {v}" for k, v in rows)


def _table(header: list[str], rows: list[list[str]], indent: int = 2) -> str:
    cols = [header] + rows
    widths = [max(len(r[i]) for r in cols) for i in range(len(header))]
    fmt = lambda r: " " * indent + "  ".join(  # noqa: E731
        c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
    rule = " " * indent + "  ".join("-" * w for w in widths)
    return "\n".join([fmt(header), rule] + [fmt(r) for r in rows])


def ascii_chart(values: np.ndarray, height: int = 12, width: int = 66,
                fmt=lambda v: f"{v:,.0f}") -> str:
    """Line chart: each column shows the bucket's min..max range, '*' at its close."""
    v = np.asarray(values, dtype=float)
    edges = np.linspace(0, v.size, min(width, v.size) + 1).astype(int)
    lo = np.minimum.reduceat(v, edges[:-1])
    hi = np.maximum.reduceat(v, edges[:-1])
    last = v[edges[1:] - 1]
    vmin, vmax = lo.min(), hi.max()
    span = vmax - vmin or 1.0
    to_row = lambda x: np.round((x - vmin) / span * (height - 1)).astype(int)  # noqa: E731
    r_lo, r_hi, r_last = to_row(lo), to_row(hi), to_row(last)
    grid = np.full((height, lo.size), " ")
    for c in range(lo.size):
        grid[r_lo[c]:r_hi[c] + 1, c] = "|"
        grid[r_last[c], c] = "*"
    labels = [fmt(vmax - i * span / (height - 1)) for i in range(height)]
    lw = max(map(len, labels))
    lines = [f"  {labels[i]:>{lw}} +{''.join(grid[height - 1 - i])}" for i in range(height)]
    lines.append(f"  {'':>{lw}} +{'-' * lo.size}")
    return "\n".join(lines)


def _ts(idx: pd.DatetimeIndex, i: int | None) -> str:
    return "open (not recovered)" if i is None else idx[min(i, len(idx) - 1)].strftime("%Y-%m-%d %H:%M")


def _params(obj) -> str:
    d = asdict(obj) if is_dataclass(obj) else dict(obj)
    return ", ".join(f"{k}={v}" for k, v in d.items())


# --------------------------------------------------------------------------- #
# Single-strategy report
# --------------------------------------------------------------------------- #
def strategy_report(res: BacktestResult, strategy, bars: pd.DataFrame,
                    ingest: IngestReport | None = None, freq: str = "1min",
                    rolling_window: int = 60) -> str:
    s, f, ppy = res.stats, res.frame, res.ppy
    idx = f.index
    E0 = res.capital.initial_capital
    out = ["=" * W, f"BACKTEST REPORT :: {res.name}".center(W),
           f"generated {datetime.now():%Y-%m-%d %H:%M:%S}".center(W), "=" * W]

    # ---- data ---------------------------------------------------------------
    out.append(_section("1. Data"))
    rows = []
    if ingest is not None:
        rows += [("source", ingest.path),
                 ("ticks read / kept", f"{ingest.rows_read:,} / {ingest.rows_kept:,}"),
                 ("ticks rejected", f"{ingest.rows_dropped:,}  ("
                  + ", ".join(f"{k}={v}" for k, v in ingest.dropped.items() if v) + ")"),
                 ("ingest time", f"{ingest.seconds:.2f}s ({ingest.rows_read / max(ingest.seconds, 1e-9):,.0f} rows/s)")]
    n_sessions = bars["session"].nunique()
    rows += [("bar frequency", f"{freq}  ({len(bars):,} bars, {n_sessions} sessions)"),
             ("period", f"{idx[0]:%Y-%m-%d %H:%M} -> {idx[-1]:%Y-%m-%d %H:%M}"),
             ("annualisation", f"{ppy:,.0f} bars/year"),
             ("price range", f"{bars['low'].min():.2f} - {bars['high'].max():.2f}"),
             ("mean quoted spread", f"{bars['spread_bps'].mean():.2f} bps"),
             ("median bar $ volume", _money(bars["dollar_volume"].median()))]
    out.append(_kv(rows))

    # ---- configuration ------------------------------------------------------
    out.append(_section("2. Configuration"))
    out.append(_kv([("strategy", f"{res.name}({_params(strategy)})"),
                    ("costs", _params(res.costs)),
                    ("capital", _params(res.capital))]))

    # ---- performance --------------------------------------------------------
    out.append(_section("3. Return & risk-adjusted performance"))
    out.append(_kv([
        ("starting / final equity", f"{_money(E0)} -> {_money(s['final_equity'])}"),
        ("total return (net)", _pct(s["total_return"], 3)),
        ("total return (gross)", _pct(s["gross_total_return"], 3) + "   before all friction"),
        ("CAGR", _pct(s["cagr"])),
        ("annualised mean return", _pct(s["ann_return_arith"])),
        ("annualised volatility", _pct(s["ann_volatility"])),
        ("Sharpe ratio", _num(s["sharpe"])),
        ("Sortino ratio", _num(s["sortino"])),
        ("Calmar ratio", _num(s["calmar"])),
        ("bar hit rate", _pct(s["hit_rate"], 1)),
        ("profit factor", _num(s["profit_factor"])),
    ]))

    # ---- risk ---------------------------------------------------------------
    out.append(_section("4. Drawdown & tail risk"))
    eq = f["equity"].to_numpy()
    mdd = metrics.max_drawdown(eq)
    out.append(_kv([
        ("maximum drawdown", _pct(mdd.depth, 3)),
        ("  peak", _ts(idx, mdd.peak)),
        ("  trough", f"{_ts(idx, mdd.trough)}  ({mdd.length} bars down)"),
        ("  recovery", _ts(idx, mdd.recovery) + (f"  ({mdd.duration} bars peak->recovery)" if mdd.duration else "")),
        ("ulcer index", _pct(s["ulcer_index"], 3)),
        ("VaR 95 / CVaR 95 (per bar)", f"{_pct(s['var_95'], 4)} / {_pct(s['cvar_95'], 4)}"),
        ("VaR 99 / CVaR 99 (per bar)", f"{_pct(s['var_99'], 4)} / {_pct(s['cvar_99'], 4)}"),
        ("worst / best bar", f"{_pct(s['worst_bar'], 4)} / {_pct(s['best_bar'], 4)}"),
        ("skewness / excess kurtosis", f"{_num(s['skew'])} / {_num(s['excess_kurtosis'])}"),
    ]))
    dds = metrics.drawdown_table(eq, top=5)
    if dds:
        out.append("\n  Top drawdown episodes")
        out.append(_table(["#", "depth", "peak", "trough", "recovered", "bars"],
                          [[str(i + 1), _pct(d.depth, 3), _ts(idx, d.peak), _ts(idx, d.trough),
                            _ts(idx, d.recovery) if d.recovery is not None else "open",
                            str(d.duration if d.duration is not None else len(idx) - d.peak)]
                           for i, d in enumerate(dds)], indent=4))

    # ---- rolling ------------------------------------------------------------
    out.append(_section(f"5. Rolling risk ({rolling_window}-bar window)"))
    r = f["net_ret"].to_numpy()
    rv = metrics.rolling_volatility(r, rolling_window, ppy)
    rdd = metrics.rolling_downside_deviation(r, rolling_window, ppy)
    rsh = metrics.rolling_sharpe(r, rolling_window, ppy)
    mkt_rv = metrics.rolling_volatility(f["ret"].to_numpy(), rolling_window, ppy)

    def dist(x, fmt=_pct):
        x = x[np.isfinite(x)]
        return [fmt(np.min(x)), fmt(np.median(x)), fmt(np.max(x)), fmt(x[-1])] if x.size else ["n/a"] * 4

    # Sharpe is undefined while the book is (nearly) flat; only report invested windows.
    rsh = np.where(rv > 0.25 * np.nanmedian(mkt_rv) * s["avg_gross_exposure"], rsh, np.nan)

    out.append(_table(["series", "min", "median", "max", "latest"], [
        ["strategy realised vol"] + dist(rv),
        ["strategy downside dev"] + dist(rdd),
        ["underlying realised vol"] + dist(mkt_rv),
        ["rolling Sharpe (invested)"] + dist(rsh, _num),
    ]))

    # ---- trading ------------------------------------------------------------
    out.append(_section("6. Trading activity, friction & capacity"))
    tc = s["total_cost"] or np.nan
    out.append(_kv([
        ("rebalances / round trips", f"{s['n_rebalances']:,} / {s['trade_n_trades']:,}"),
        ("trade win rate", _pct(s["trade_win_rate"], 1)),
        ("avg trade / win / loss", f"{_pct(s['trade_avg_trade'], 3)} / {_pct(s['trade_avg_win'], 3)} / {_pct(s['trade_avg_loss'], 3)}"),
        ("payoff ratio", _num(s["trade_payoff"])),
        ("avg holding period", f"{_num(s['trade_avg_bars_held'], 1)} bars"),
        ("time in market", _pct(s["time_in_market"], 1)),
        ("avg gross exposure", _pct(s["avg_gross_exposure"], 1)),
        ("traded notional", _money(s["traded_notional"])),
        ("daily turnover", f"{_num(s['daily_turnover'], 2)}x equity"),
        ("total friction", f"{_money(s['total_cost'])}  ({_pct(s['total_cost'] / E0, 3)} of initial capital)"),
        ("  commission", f"{_money(s['cost_commission'])}  ({_pct(s['cost_commission'] / tc, 1)})"),
        ("  spread", f"{_money(s['cost_spread'])}  ({_pct(s['cost_spread'] / tc, 1)})"),
        ("  market impact", f"{_money(s['cost_impact'])}  ({_pct(s['cost_impact'] / tc, 1)})"),
        ("  borrow / financing", f"{_money(s['cost_carry'])}  ({_pct(s['cost_carry'] / tc, 1)})"),
        ("all-in cost per $ traded", f"{_num(s['cost_per_traded_bps'])} bps"),
        ("gross edge per $ traded", f"{_num(s['edge_per_traded_bps'])} bps"
         + ("   <- edge does not cover friction" if s["edge_per_traded_bps"] < s["cost_per_traded_bps"] else "")),
        ("bar participation p95 / max", f"{_pct(s['p95_participation'], 1)} / {_pct(s['max_participation'], 1)}"
         + ("   <- capacity-constrained" if s["p95_participation"] > 0.10 else "")),
        ("equity solver", f"{s['solver_iterations']} fixed-point passes, "
         + ("converged" if s["solver_converged"] else "NOT converged")),
    ]))

    # ---- daily --------------------------------------------------------------
    out.append(_section("7. Session-by-session P&L"))
    day = idx.normalize()
    g = f.assign(log_net=np.log1p(f["net_ret"]), log_gross=np.log1p(f["gross_ret"])).groupby(day)
    daily = pd.DataFrame({
        "net": np.expm1(g["log_net"].sum()), "gross": np.expm1(g["log_gross"].sum()),
        "cost": g["cost"].sum(), "equity": g["equity"].last(),
        "vol": g["net_ret"].apply(lambda x: metrics.annualized_volatility(x.to_numpy(), ppy)),
        "trades": g["trade_notional"].apply(lambda x: int((x > 1e-9).sum())),
    })
    out.append(_table(["session", "net", "gross", "friction", "rebal", "ann. vol", "end equity"],
                      [[d.strftime("%a %Y-%m-%d"), _pct(row.net, 3), _pct(row.gross, 3), _money(row.cost),
                        str(int(row.trades)), _pct(row.vol, 1), _money(row.equity)] for d, row in daily.iterrows()]))
    dr = daily["net"].to_numpy()
    out.append("\n" + _kv([("positive sessions", f"{(dr > 0).sum()} of {dr.size}"),
                           ("daily Sharpe (x sqrt 252)", _num(metrics.sharpe_ratio(dr, 252)))]))

    # ---- charts ---------------------------------------------------------------
    out.append(_section("8. Equity curve (net, $)"))
    out.append(ascii_chart(eq))
    out.append(_section("9. Underwater curve (drawdown)"))
    out.append(ascii_chart(f["drawdown"].to_numpy(), height=8, fmt=lambda v: f"{v * 100:.2f}%"))

    out.append(_section("Notes"))
    out.append(
        "  * Mock data, in-sample. Annualised figures extrapolate a short sample and\n"
        "    are dominated by noise; prefer the per-session table and friction split.\n"
        "  * Fills at bar close mid +/- half-spread + sqrt impact; signals use only\n"
        "    data up to that close. Overnight positions are flattened when\n"
        "    flat_overnight=True.\n"
        "  * Risk metrics are computed in quantlab.metrics from first principles.")
    out.append("=" * W)
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Cross-strategy comparison
# --------------------------------------------------------------------------- #
COMPARE_ROWS = [
    ("total return (net)", "total_return", lambda v: _pct(v, 3)),
    ("total return (gross)", "gross_total_return", lambda v: _pct(v, 3)),
    ("annualised vol", "ann_volatility", _pct),
    ("Sharpe", "sharpe", _num),
    ("Sortino", "sortino", _num),
    ("Calmar", "calmar", _num),
    ("max drawdown", "max_drawdown", lambda v: _pct(v, 3)),
    ("CVaR 95 (bar)", "cvar_95", lambda v: _pct(v, 4)),
    ("round trips", "trade_n_trades", lambda v: f"{v:,}"),
    ("daily turnover", "daily_turnover", lambda v: f"{_num(v)}x"),
    ("friction ($)", "total_cost", _money),
    ("edge / $ traded (bps)", "edge_per_traded_bps", _num),
    ("cost / $ traded (bps)", "cost_per_traded_bps", _num),
    ("p95 participation", "p95_participation", lambda v: _pct(v, 1)),
]


def comparison_report(results: list[BacktestResult]) -> str:
    out = ["=" * W, "STRATEGY COMPARISON".center(W),
           f"generated {datetime.now():%Y-%m-%d %H:%M:%S}".center(W), "=" * W, ""]
    out.append(_table(["metric"] + [r.name for r in results],
                      [[label] + [fn(r.stats[key]) for r in results] for label, key, fn in COMPARE_ROWS]))
    best = max(results, key=lambda r: r.stats["sharpe"] if np.isfinite(r.stats["sharpe"]) else -np.inf)
    out.append("")
    out.append(f"  Highest net Sharpe: {best.name} ({_num(best.stats['sharpe'])}).")
    for r in results:
        st = r.stats
        drag = st["gross_total_return"] - st["total_return"]
        out.append(f"  {r.name}: friction consumed {_pct(drag, 3)} of return"
                   + (f" -- {_pct(drag / st['gross_total_return'], 0)} of gross P&L." if st["gross_total_return"] > 0
                      else " on top of a gross loss."))
    out.append("=" * W)
    return "\n".join(out) + "\n"
