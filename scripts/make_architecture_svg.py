"""Render the quantlab execution-architecture blueprint as a standalone SVG.

    .venv/bin/python scripts/make_architecture_svg.py [--out assets/quant_architecture.svg]

Pure standard library: the SVG is assembled from primitives so every label,
formula and arrow is explicit and diff-able. A monospace width estimate guards
against text overflowing its block; the script fails rather than ship clipping.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

W, H = 1600, 1040

BG = "#000000"
PANEL = "#07110b"
GREEN = "#39ff88"        # data vectors
GREEN_DIM = "#1d7a45"
CRIMSON = "#ff2d55"      # risk thresholds
WHITE = "#f4f4f2"
GREY = "#9a9a96"
MONO = "'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace"
SANS = "'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif"

CHAR_W = {MONO: 0.61, SANS: 0.56}  # average advance / font-size, conservative
TITLE_W = 0.64                       # bold upper-case sans is wider


@dataclass
class Block:
    num: str
    title: str
    x: float
    y: float
    w: float
    h: float
    lines: list[tuple[str, str]] = field(default_factory=list)  # (kind, text)

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


# kind -> (font, size, colour, line height)
STYLES = {
    "body": (SANS, 15, GREY, 22),
    "math": (MONO, 15, WHITE, 24),
    "risk": (MONO, 14, CRIMSON, 22),
    "head": (SANS, 12, GREEN, 24),
    "gap": (SANS, 8, GREY, 8),
}

BLOCKS = [
    Block("01", "RAW HIGH-FREQUENCY TICK INGESTION ENGINE", 60, 150, 460, 210, [
        ("body", "Chunked CSV reader · typed dtypes, no inference"),
        ("body", "250k-row chunks · bounded memory"),
        ("gap", ""),
        ("math", "tick = (t, bid, ask, px, size, side)"),
        ("math", "≈ 2.0M rows/s  ·  4.2M ticks in 2.2 s"),
    ]),
    Block("02", "DATA CLEANING FILTER", 570, 150, 460, 210, [
        ("body", "Vectorised masks · first-failure attribution"),
        ("gap", ""),
        ("risk", "✕ reject  bid ≥ ask        (crossed quote)"),
        ("risk", "✕ reject  px ≤ 0 · size ≤ 0 · NaN"),
        ("risk", "✕ reject  exact duplicate rows"),
        ("math", "sort(t) → monotone time index"),
    ]),
    Block("03", "1-MINUTE BAR AGGREGATOR", 1080, 150, 460, 210, [
        ("body", "Right-labelled bars: no intra-bar leakage"),
        ("gap", ""),
        ("math", "mid = (bid + ask) / 2  →  OHLC"),
        ("math", "VWAP = Σ px·q / Σ q"),
        ("math", "spread_bps · $vol · imbalance"),
        ("math", "N = 390 bars/day × 252 = 98,280"),
    ]),
    Block("04", "VECTORISED STRATEGY MODULE", 60, 430, 640, 300, [
        ("head", "MEAN REVERSION  ·  hysteresis z-score fade"),
        ("math", "zₜ = (ln Pₜ − μ₂₀) / σ₂₀"),
        ("math", "sₜ = −1 if z > 2 · +1 if z < −2 · 0 if |z| < 0.2"),
        ("head", "MOMENTUM  ·  vol-normalised EMA crossover"),
        ("math", "sₜ = tanh( (EMA₃₀ − EMA₂₄₀) / (0.5 · σ_bar · √240) )"),
        ("head", "VOLATILITY-SCALING LOOP"),
        ("math", "σ̂ₜ = √N · std(r[t−59 : t])"),
        ("math", "wₜ = sₜ · clip( σ* / σ̂ₜ , 0, L ),   σ* = 20%"),
        ("risk", "|wₜ| ≤ L = 1.0x   ·   w quantised to 0.1"),
    ]),
    Block("05", "4–6 PASS CONVERGENT SOLVER", 760, 430, 780, 300, [
        ("body", "Equity, sizing and costs are mutually dependent → fixed point"),
        ("math", "w⁻ₜ = wₜ₋₁ (1 + rₜ) / (1 + wₜ₋₁ rₜ)  # pre-trade drift"),
        ("math", "Nₜ = |wₜ − w⁻ₜ| · Eₜ₋₁ (1 + wₜ₋₁ rₜ)  # $ traded"),
        ("math", "cₜ = max(fee_min, κ·Nₜ) + ½ spreadₜ·Nₜ  # fees + spread"),
        ("math", "     + η σₜ Nₜ √(Nₜ / $volₜ) + carryₜ  # sqrt impact"),
        ("math", "E⁽ᵏ⁺¹⁾ₜ = E₀ ∏ₛ≤ₜ [ 1 + wₛ₋₁rₛ − cₛ(E⁽ᵏ⁾) / Eₛ₋₁ ]  # fixed point"),
        ("risk", "‖E⁽ᵏ⁺¹⁾ − E⁽ᵏ⁾‖∞ / E₀ < 1e−12  ⇒  converged"),
        ("risk", "kill-switch: E⁻ ≤ (1 − DD_max)·HWM ⇒ w ≡ 0"),
        ("risk", "notional cap: |w|·E⁻ ≤ $cap  ·  flat overnight"),
    ]),
    Block("06", "NUMPY RISK METRICS PIPELINE", 60, 800, 960, 200, [
        ("math", "Sharpe  = √N · mean(r − r_f) / std(r − r_f)"),
        ("math", "Sortino = √N · mean(r − T) / DD_T,   DD_T = √( (1/n) Σ min(r − T, 0)² )"),
        ("math", "MDD = minₜ ( Eₜ / maxₛ≤ₜ Eₛ − 1 )   ·   rolling σ via O(n) prefix sums"),
        ("risk", "VaR₉₅ / CVaR₉₅ · Calmar · Ulcer · skew · kurtosis · trade stats"),
    ]),
    Block("07", "AUTOMATED REPORT GENERATOR", 1080, 800, 460, 200, [
        ("body", "Fixed-width text reports per strategy"),
        ("math", "perf · drawdowns · rolling risk"),
        ("math", "friction split · capacity · per-session"),
        ("math", "→ reports/<ts>/*.txt + ledgers.csv"),
    ]),
]


def text_width(s: str, font: str, size: float) -> float:
    return len(s) * size * CHAR_W[font]


def t(x, y, s, font, size, fill, anchor="start", weight=400, spacing=0.0):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{font}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'letter-spacing="{spacing}">{escape(s)}</text>')


def draw_block(b: Block, problems: list[str]) -> list[str]:
    out = [
        f'<g id="block-{b.num}">',
        f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="6" fill="{PANEL}" '
        f'stroke="{GREEN_DIM}" stroke-width="1.5"/>',
        # corner ticks, blueprint style
        *[f'<path d="M{px} {py + dy * 14} L{px} {py} L{px + dx * 14} {py}" stroke="{GREEN}" '
          f'stroke-width="2" fill="none"/>'
          for px, py, dx, dy in [(b.x, b.y, 1, 1), (b.x + b.w, b.y, -1, 1),
                                 (b.x, b.y + b.h, 1, -1), (b.x + b.w, b.y + b.h, -1, -1)]],
        t(b.x + 20, b.y + 32, b.num, MONO, 14, GREEN, weight=700),
        t(b.x + 52, b.y + 32, b.title, SANS, 14, WHITE, weight=700, spacing=0.4),
        f'<line x1="{b.x + 20}" y1="{b.y + 46}" x2="{b.x + b.w - 20}" y2="{b.y + 46}" '
        f'stroke="{GREEN_DIM}" stroke-width="1" stroke-dasharray="2 4"/>',
    ]
    if 52 + len(b.title) * 14 * TITLE_W + 0.4 * len(b.title) > b.w - 20:
        problems.append(f"block {b.num} title overflows")
    y = b.y + 72
    for kind, s in b.lines:
        font, size, fill, lh = STYLES[kind]
        if kind == "gap":
            y += lh
            continue
        if kind == "head":
            out.append(t(b.x + 20, y, s, SANS, size, fill, weight=700, spacing=1.0))
        elif "  # " in s:  # formula + right-aligned annotation
            s, note = s.split("  # ")
            out.append(t(b.x + 20, y, s, font, size, fill))
            out.append(t(b.x + b.w - 20, y, note, SANS, 13, GREY, anchor="end"))
            if 20 + text_width(s, font, size) + 24 + text_width(note, SANS, 13) > b.w - 20:
                problems.append(f"block {b.num} annotated line overflows: {s!r}")
        else:
            out.append(t(b.x + 20, y, s, font, size, fill))
        if 20 + text_width(s, font, size) > b.w - 16:
            problems.append(f"block {b.num} line overflows: {s!r}")
        y += lh
    if y - STYLES["body"][3] + 8 > b.y + b.h:
        problems.append(f"block {b.num} content overflows bottom ({y:.0f} > {b.y + b.h})")
    out.append("</g>")
    return out


def arrow(points: list[tuple[float, float]], label: str | None = None, color=GREEN,
          label_at: tuple[float, float] | None = None, dashed=False) -> str:
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in points)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    s = (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{dash} '
         f'marker-end="url(#{"head-red" if color == CRIMSON else "head"})" filter="url(#glow)"/>')
    if label:
        lx, ly = label_at or points[len(points) // 2]
        s += t(lx, ly, label, MONO, 12, color, anchor="middle")
    return s


def build() -> tuple[str, list[str]]:
    problems: list[str] = []
    b = {blk.num: blk for blk in BLOCKS}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-labelledby="title desc">',
        '<title id="title">quantlab execution architecture</title>',
        '<desc id="desc">Pipeline from raw tick ingestion through cleaning, bar aggregation, '
        'vectorised strategies, a fixed-point equity solver, numpy risk metrics and report '
        'generation, annotated with the governing formulas and risk thresholds.</desc>',
        "<defs>",
        f'<pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">'
        f'<path d="M40 0 L0 0 0 40" fill="none" stroke="#0d1f14" stroke-width="1"/></pattern>',
        f'<filter id="glow" filterUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
        '<feGaussianBlur stdDeviation="2.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        f'<marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{GREEN}"/></marker>',
        f'<marker id="head-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{CRIMSON}"/></marker>',
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="{BG}"/>',
        f'<rect width="{W}" height="{H}" fill="url(#grid)"/>',
        t(60, 62, "QUANTLAB", MONO, 30, GREEN, weight=700, spacing=6),
        t(262, 62, "// EXECUTION ARCHITECTURE BLUEPRINT", MONO, 30, WHITE, weight=400, spacing=1),
        t(60, 96, "Vectorised tick-to-report backtesting engine · NumPy / pandas · "
          "risk analytics implemented from first principles", SANS, 16, GREY),
        # legend
        f'<line x1="1150" y1="90" x2="1190" y2="90" stroke="{GREEN}" stroke-width="2.2"/>',
        t(1198, 95, "data vector", MONO, 13, WHITE),
        f'<line x1="1320" y1="90" x2="1360" y2="90" stroke="{CRIMSON}" stroke-width="2.2"/>',
        t(1368, 95, "risk threshold", MONO, 13, WHITE),
    ]
    for blk in BLOCKS:
        svg += draw_block(blk, problems)

    # ---- data-flow vectors --------------------------------------------------
    svg.append(arrow([(b["01"].x + b["01"].w, b["01"].cy), (b["02"].x, b["02"].cy)],
                     "ticks", label_at=(545, b["01"].cy - 10)))
    svg.append(arrow([(b["02"].x + b["02"].w, b["02"].cy), (b["03"].x, b["03"].cy)],
                     "clean", label_at=(1055, b["02"].cy - 10)))
    # bars: down from 03, across the gutter, into 04
    y_g = 395
    svg.append(arrow([(b["03"].cx, b["03"].y + b["03"].h), (b["03"].cx, y_g),
                      (b["04"].cx, y_g), (b["04"].cx, b["04"].y)],
                     "bars[t]  OHLC · $vol · spread", label_at=(760, y_g - 9)))
    svg.append(arrow([(b["04"].x + b["04"].w, b["04"].cy), (b["05"].x, b["05"].cy)],
                     "wₜ", label_at=(730, b["04"].cy - 10)))
    # solver feedback loop on the right edge
    lx = b["05"].x + b["05"].w + 22
    svg.append(arrow([(b["05"].x + b["05"].w, b["05"].y + 250), (lx, b["05"].y + 250),
                      (lx, b["05"].y + 60), (b["05"].x + b["05"].w, b["05"].y + 60)],
                     color=GREEN, dashed=True))
    svg.append(f'<text x="{lx + 14}" y="{b["05"].cy}" font-family="{MONO}" font-size="12" '
               f'fill="{GREEN}" text-anchor="middle" transform="rotate(90 {lx + 14} {b["05"].cy})">'
               f'E⁽ᵏ⁾ → E⁽ᵏ⁺¹⁾</text>')
    # solver -> metrics, metrics -> report
    y_g2 = 765
    svg.append(arrow([(b["05"].cx, b["05"].y + b["05"].h), (b["05"].cx, y_g2),
                      (b["06"].cx, y_g2), (b["06"].cx, b["06"].y)],
                     "equity · ledger · costs", label_at=(910, y_g2 - 9)))
    svg.append(arrow([(b["06"].x + b["06"].w, b["06"].cy), (b["07"].x, b["07"].cy)],
                     "stats", label_at=(1050, b["06"].cy - 10)))

    svg.append(t(W - 60, H - 14, "quantlab · single-asset, bar-close fills, "
                 "in-sample synthetic data", MONO, 11, "#5c5c58", anchor="end"))
    svg.append("</svg>")
    return "\n".join(svg), problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/quant_architecture.svg")
    out = Path(ap.parse_args().out)
    svg, problems = build()
    ET.fromstring(svg)  # must be well-formed XML
    if problems:
        raise SystemExit("layout check failed:\n  " + "\n  ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {W}x{H}, {len(BLOCKS)} blocks)")


if __name__ == "__main__":
    main()
