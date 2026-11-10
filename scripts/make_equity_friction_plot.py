"""Two-panel validation figure: equity before/after costs + cost attribution.

    .venv/bin/python scripts/make_equity_friction_plot.py [--run reports/<ts>] [--out assets/equity_friction.png]

Reads the per-bar ledgers written by ``quantlab run`` (``<strategy>_ledger.csv``)
from the newest run directory unless ``--run`` is given. Nothing is re-simulated:
every plotted number comes from the telemetry on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

STRATEGIES = [  # fixed categorical order -> fixed colour (validated dark-surface steps)
    ("mean_reversion", "Mean reversion", "#3987e5"),
    ("momentum", "Momentum", "#d95926"),
    ("buy_and_hold", "Buy & hold", "#199e70"),
]
COSTS = [("commission", "Commission"), ("spread_cost", "Bid/ask spread"), ("impact_cost", "Market impact")]

SURFACE = "#000000"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#7d7c76"
GRID = "#262624"


def latest_run(reports: Path) -> Path:
    runs = sorted(p for p in reports.iterdir() if p.is_dir() and any(p.glob("*_ledger.csv")))
    if not runs:
        raise SystemExit(f"no run directories with ledgers under {reports}/ - run `quantlab run` first")
    return runs[-1]


def load(run: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for key, *_ in STRATEGIES:
        path = run / f"{key}_ledger.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        out[key] = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    return out


def style() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": INK_2,
    })


def plot(ledgers: dict[str, pd.DataFrame], run: Path, out: Path) -> dict:
    style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9.2), height_ratios=[1.55, 1],
                                   gridspec_kw={"hspace": 0.5})
    E0 = None
    summary = {}

    # ---- top: indexed equity, gross (dashed) vs net (solid) ------------------
    for key, label, color in STRATEGIES:
        f = ledgers[key]
        E0 = E0 or f["equity"].iloc[0] + f["cost"].iloc[0]  # equity before the first fill
        x = np.arange(len(f))
        net = (f["equity"].to_numpy() / E0 - 1) * 100
        gross = (f["gross_equity"].to_numpy() / E0 - 1) * 100
        ax1.plot(x, gross, color=color, lw=1.4, ls=(0, (4, 3)), alpha=0.85)
        ax1.plot(x, net, color=color, lw=2.0)
        summary[key] = {"net": net[-1], "gross": gross[-1],
                        **{c: f[c].sum() / E0 * 100 for c, _ in COSTS},
                        "carry": f["carry_cost"].sum() / E0 * 100}

    # direct end labels (net only), nudged apart so they never collide
    ends = sorted(((summary[k]["net"], summary[k]["gross"], lab, col) for k, lab, col in STRATEGIES), reverse=True)
    lo, hi = ax1.get_ylim()
    min_gap = (hi - lo) * 0.11
    placed = []
    for y, g, lab, col in ends:
        y_l = min(y, placed[-1] - min_gap) if placed else y
        placed.append(y_l)
        ax1.annotate(f"{lab}  {y:+.2f}%\n(gross {g:+.2f}%)", xy=(len(x) - 1, y), xytext=(len(x) + 25, y_l),
                     color=INK, fontsize=9.5, va="center", annotation_clip=False,
                     arrowprops=dict(arrowstyle="-", color=col, lw=1))

    sessions = ledgers[STRATEGIES[0][0]].index.normalize()
    starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    for s in starts[1:]:
        ax1.axvline(s, color=GRID, lw=0.8, zorder=0)
    ax1.set_xticks(starts + np.diff(np.r_[starts, len(sessions)]) / 2,
                   [d.strftime("%a %d %b") for d in sessions[starts]], fontsize=8.5)
    ax1.tick_params(axis="x", length=0)
    ax1.grid(axis="x", visible=False)
    ax1.axhline(0, color=MUTED, lw=0.9, zorder=1)
    ax1.set_xlim(0, len(sessions) - 1)
    ax1.set_ylabel("Cumulative return on initial capital (%)")
    ax1.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax1.set_title("Cumulative equity, before vs. after transaction costs", loc="left",
                  fontsize=13, fontweight="bold", pad=26)
    ax1.text(0, 1.035, f"1-minute bars · {len(starts)} sessions · ${E0:,.0f} initial capital · "
             "overnight flat (buy & hold included)", transform=ax1.transAxes, color=INK_2, fontsize=9)

    handles = [Line2D([], [], color=c, lw=2.4, label=l) for _, l, c in STRATEGIES]
    handles += [Line2D([], [], color=INK_2, lw=2.0, label="After costs (net)"),
                Line2D([], [], color=INK_2, lw=1.4, ls=(0, (4, 3)), label="Before costs (gross)")]
    ax1.legend(handles=handles, loc="lower left", ncol=5, fontsize=8.8, handlelength=2.6,
               bbox_to_anchor=(0, -0.2), columnspacing=1.6)

    # ---- bottom: grouped bars, cost component x strategy --------------------
    n_s = len(STRATEGIES)
    group_w = 0.78
    bar_w = group_w / n_s
    xs = np.arange(len(COSTS))
    for i, (key, label, color) in enumerate(STRATEGIES):
        vals = np.array([summary[key][c] for c, _ in COSTS])
        pos = xs - group_w / 2 + bar_w * (i + 0.5)
        bars = ax2.bar(pos, vals, width=bar_w, color=color, edgecolor=SURFACE, linewidth=2,
                       label=label, zorder=3)
        for b, v in zip(bars, vals):
            ax2.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax2.set_xticks(xs, [lab for _, lab in COSTS], fontsize=10, color=INK)
    ax2.tick_params(axis="x", length=0)
    ax2.grid(axis="x", visible=False)
    ax2.set_ylim(0, max(summary[k][c] for k, *_ in STRATEGIES for c, _ in COSTS) * 1.18)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%" if v else "0%")
    ax2.set_ylabel("Drag (% of initial capital)")
    ax2.set_title("Cost attribution by component", loc="left", fontsize=13, fontweight="bold", pad=26)
    totals = " · ".join(
        f"{lab}: {sum(summary[k][c] for c, _ in COSTS):.2f}% total" for k, lab, _ in STRATEGIES)
    ax2.text(0, 1.035, totals, transform=ax2.transAxes, color=INK_2, fontsize=9)
    ax2.legend(loc="upper right", fontsize=8.8, handlelength=1.2)

    carry = max(summary[k]["carry"] for k, *_ in STRATEGIES)
    fig.text(0.125, 0.025,
             f"Source: {run.as_posix()}/*_ledger.csv (synthetic in-sample data). Fills at bar-close mid ± half-spread; "
             f"impact = η·σ·N·√(N/$vol). Borrow/financing carry omitted (≤ {carry:.2f}% per strategy).",
             color=MUTED, fontsize=7.8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="report run directory (default: newest under reports/)")
    ap.add_argument("--out", type=Path, default=Path("assets/equity_friction.png"))
    ns = ap.parse_args()
    run = ns.run or latest_run(Path("reports"))
    summary = plot(load(run), run, ns.out)
    from PIL import Image

    with Image.open(ns.out) as im:
        dpi = im.info.get("dpi", (0, 0))[0]
        print(f"wrote {ns.out} ({im.width}x{im.height} px, {dpi:.0f} dpi, "
              f"{ns.out.stat().st_size / 1024:.0f} KB) from {run}")
    for key, s in summary.items():
        print(f"  {key:<15} gross {s['gross']:+.3f}%  net {s['net']:+.3f}%  "
              f"comm {s['commission']:.3f}%  spread {s['spread_cost']:.3f}%  impact {s['impact_cost']:.3f}%")


if __name__ == "__main__":
    main()
