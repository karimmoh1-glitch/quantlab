# quantlab

A vectorized tick-to-bar backtester with risk analytics written from scratch. It generates mock high-frequency tick data, cleans it, aggregates it into bars, runs mean-reversion and momentum strategies through a cost-aware engine, and writes plain-text performance reports.

```bash
python3.13 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/quantlab generate --days 10            # ~700k ticks -> data/ticks.csv
.venv/bin/quantlab run --strategy all            # reports/<timestamp>/*.txt + ledgers
.venv/bin/quantlab sweep --strategy momentum     # parameter sensitivity grid
.venv/bin/pytest
```

Every field of `CostModel` and `CapitalConstraints` is also a CLI flag, for example `--impact-coef 0.5 --max-drawdown-stop 0.05 --initial-capital 5e6 --allow-short false`. Run `quantlab run -h` for the full list.

## Layout

| module | role |
|---|---|
| `data.py` | Poisson-clock tick simulator (regime-switching trend + exact OU mean-reversion, jumps, spreads, feed defects); chunked typed CSV ingestion with validation; tick→OHLC/VWAP/spread/imbalance bars |
| `metrics.py` | Max drawdown (with peak/trough/recovery and an episode table), Sharpe, Sortino, Calmar, Ulcer, VaR/CVaR, moments, and O(n) rolling volatility, downside deviation, Sharpe and Sortino, all on raw numpy |
| `strategies.py` | `MeanReversion` (z-score with hysteresis), `Momentum` (vol-normalized EMA crossover), `BuyAndHold`; all vol-targeted |
| `engine.py` | Vectorized execution and accounting with friction and capital constraints |
| `report.py` | Text reports: data quality, performance, drawdowns, rolling risk, friction breakdown, capacity, per-session P&L, ASCII equity and underwater charts |

## Engine design

The decision at the close of bar *t* is filled at that close. The book first drifts with the bar's return, then trades to the target and pays:

- **commission**: bps of notional, with a minimum ticket charge
- **spread**: the bar's mean quoted half-spread, scaled by `spread_capture`
- **market impact**: square root, `coef · σ_bar · √(notional / bar $volume)`
- **carry**: borrow on shorts and financing on gross exposure above 1×

Capital constraints:

- leverage cap and long-only switch
- weight quantization, which acts as a no-trade band
- dollar position cap
- drawdown kill-switch
- flat overnight
- execution lag

Impact, minimum commissions, the dollar cap and the kill-switch all depend on the size of the book in dollars, so equity depends on its own path. The engine does not fall back to a per-bar loop. It treats equity as a **fixed point**: guess the equity path, compute every bar's costs from that guess in one vectorized pass, rebuild equity with a cumulative product, and repeat. It usually converges to 1e-12 in 4–6 passes, and `tests/test_engine.py` checks the result against a plain loop implementation.

Measured on an M-series Mac with 10 sessions: 700k ticks ingest in ≈0.3 s, and a backtest over 3,900 bars takes ≈2 ms. With 60 sessions, 4.2M ticks ingest in 2.2 s and a backtest takes ≈7 ms.

## Correctness guarantees (tested)

- Every metric matches an independent reference: pandas moments, the closed-form Sharpe and Sortino, and a brute-force drawdown walk.
- Rolling standard deviation is centered causally. A global-mean centering leaked future data into early windows at the floating-point level, and the look-ahead test caught it. It is also more accurate than `pandas.rolling().std()` against an exact two-pass calculation.
- **No look-ahead:** rewriting the second half of the price path leaves every earlier target weight bit-for-bit identical.
- The engine's equity path reproduces itself from the costs it reports, and impact cost in bps rises with capital.

## Caveats

The data is synthetic and every result is in-sample. Annualized ratios from 10 sessions of 1-minute bars are mostly noise. The per-session table and the "gross edge vs. cost per $ traded" line in each report are the more honest diagnostics.
