"""Risk and performance metrics implemented directly on numpy arrays.

No pandas rolling/ewm, scipy or empyrical: every statistic below is written
out from its definition so the arithmetic is auditable. Rolling statistics are
O(n) via prefix sums rather than O(n * window) via sliding windows.

Conventions
-----------
* ``returns`` are simple per-period returns, ``equity`` is a wealth path.
* ``ppy`` is periods per year (e.g. 390 one-minute bars * 252 days).
* Sample statistics use the n-1 (Bessel) denominator unless stated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("expected a 1-D series")
    return a


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #
def mean(x) -> float:
    x = _arr(x)
    return float(x.sum() / x.size) if x.size else np.nan


def std(x, ddof: int = 1) -> float:
    x = _arr(x)
    n = x.size
    if n <= ddof:
        return np.nan
    d = x - x.sum() / n
    return float(np.sqrt(np.dot(d, d) / (n - ddof)))


def skewness(x) -> float:
    """Adjusted Fisher-Pearson sample skewness (G1)."""
    x = _arr(x)
    n = x.size
    if n < 3:
        return np.nan
    d = x - x.sum() / n
    m2, m3 = np.dot(d, d) / n, np.dot(d * d, d) / n
    if m2 == 0:
        return 0.0
    g1 = m3 / m2**1.5
    return float(g1 * np.sqrt(n * (n - 1)) / (n - 2))


def excess_kurtosis(x) -> float:
    """Bias-corrected sample excess kurtosis (G2)."""
    x = _arr(x)
    n = x.size
    if n < 4:
        return np.nan
    d = x - x.sum() / n
    d2 = d * d
    m2, m4 = d2.sum() / n, np.dot(d2, d2) / n
    if m2 == 0:
        return 0.0
    g2 = m4 / m2**2 - 3.0
    return float((n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6))


# --------------------------------------------------------------------------- #
# Return / risk ratios
# --------------------------------------------------------------------------- #
def total_return(equity) -> float:
    e = _arr(equity)
    return float(e[-1] / e[0] - 1.0)


def cagr(equity, ppy: float) -> float:
    e = _arr(equity)
    years = (e.size - 1) / ppy
    if years <= 0 or e[-1] <= 0:
        return np.nan
    return float((e[-1] / e[0]) ** (1.0 / years) - 1.0)


def annualized_volatility(returns, ppy: float) -> float:
    return std(returns) * np.sqrt(ppy)


def sharpe_ratio(returns, ppy: float, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe: mean(r - rf) / std(r - rf) * sqrt(ppy)."""
    r = _arr(returns) - risk_free_annual / ppy
    s = std(r)
    if not np.isfinite(s) or s == 0:
        return np.nan
    return float(mean(r) / s * np.sqrt(ppy))


def downside_deviation(returns, target: float = 0.0) -> float:
    """Target semideviation sqrt(mean(min(r - target, 0)^2)) over *all* periods.

    Using the full sample size (not only the losing periods) as the
    denominator is the Sortino & Price definition; dividing by the count of
    negative periods overstates downside risk.
    """
    r = _arr(returns)
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt(np.dot(shortfall, shortfall) / r.size)) if r.size else np.nan


def sortino_ratio(returns, ppy: float, target_annual: float = 0.0) -> float:
    """Annualised Sortino: mean(r - T) / DD_T * sqrt(ppy)."""
    t = target_annual / ppy
    dd = downside_deviation(returns, t)
    if not np.isfinite(dd) or dd == 0:
        return np.nan
    return float((mean(returns) - t) / dd * np.sqrt(ppy))


# --------------------------------------------------------------------------- #
# Drawdowns
# --------------------------------------------------------------------------- #
def drawdown_series(equity) -> np.ndarray:
    """Fractional distance below the running high-water mark (<= 0)."""
    e = _arr(equity)
    return e / np.maximum.accumulate(e) - 1.0


@dataclass(frozen=True)
class Drawdown:
    depth: float            # negative fraction, e.g. -0.12
    peak: int               # index of the high-water mark
    trough: int             # index of the lowest point
    recovery: int | None    # first index back at/above the peak, None if open

    @property
    def length(self) -> int:
        """Peak-to-trough periods."""
        return self.trough - self.peak

    @property
    def duration(self) -> int | None:
        """Peak-to-recovery periods (None while underwater)."""
        return None if self.recovery is None else self.recovery - self.peak


def max_drawdown(equity) -> Drawdown:
    e = _arr(equity)
    hwm = np.maximum.accumulate(e)
    dd = e / hwm - 1.0
    trough = int(np.argmin(dd))
    # The peak is the last time the HWM was set before the trough.
    peak = int(np.flatnonzero(e[: trough + 1] == hwm[trough])[-1])
    after = np.flatnonzero(e[trough:] >= hwm[trough])
    recovery = int(trough + after[0]) if after.size else None
    return Drawdown(float(dd[trough]), peak, trough, recovery)


def drawdown_table(equity, top: int = 5) -> list[Drawdown]:
    """The ``top`` deepest non-overlapping drawdown episodes.

    Episodes are found without a Python loop over periods: each underwater
    run is labelled by a prefix-sum segment id, and the per-episode minimum is
    taken with ``np.minimum.reduceat``.
    """
    e = _arr(equity)
    hwm = np.maximum.accumulate(e)
    dd = e / hwm - 1.0
    under = dd < 0
    if not under.any():
        return []
    starts = np.flatnonzero(under & ~np.r_[False, under[:-1]])
    ends = np.flatnonzero(under & ~np.r_[under[1:], False])  # inclusive
    seg_min = np.minimum.reduceat(dd, starts)
    # Trough of each episode: the earliest underwater index equal to its minimum.
    pos = np.flatnonzero(under)
    seg_id = np.searchsorted(starts, pos, side="right") - 1
    is_min = dd[pos] == seg_min[seg_id]
    first_min = np.full(starts.size, -1)
    first_min[seg_id[is_min][::-1]] = pos[is_min][::-1]  # reversed so the earliest wins

    order = np.argsort(seg_min)[:top]
    out = []
    for k in order:
        recovery = int(ends[k] + 1) if ends[k] + 1 < e.size else None
        out.append(Drawdown(float(seg_min[k]), int(starts[k] - 1), int(first_min[k]), recovery))
    return out


def calmar_ratio(equity, ppy: float) -> float:
    mdd = max_drawdown(equity).depth
    return float(cagr(equity, ppy) / -mdd) if mdd < 0 else np.nan


def ulcer_index(equity) -> float:
    """Root-mean-square drawdown (penalises depth *and* duration)."""
    dd = drawdown_series(equity)
    return float(np.sqrt(np.dot(dd, dd) / dd.size))


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def _empirical_quantile(x: np.ndarray, q: float) -> float:
    """Linear-interpolated quantile (Hyndman & Fan type 7) via partial sort."""
    n = x.size
    h = (n - 1) * q
    lo = int(np.floor(h))
    hi = min(lo + 1, n - 1)
    part = np.partition(x, (lo, hi))
    return float(part[lo] + (h - lo) * (part[hi] - part[lo]))


def value_at_risk(returns, level: float = 0.95) -> float:
    """Historical VaR as a positive loss fraction at the given confidence."""
    return -_empirical_quantile(_arr(returns), 1.0 - level)


def conditional_var(returns, level: float = 0.95) -> float:
    """Expected shortfall: mean loss in the worst (1 - level) tail."""
    r = _arr(returns)
    k = max(1, int(np.ceil(r.size * (1.0 - level) - 1e-9)))  # tolerate 1-0.95 != 0.05
    return float(-np.partition(r, k - 1)[:k].mean())


# --------------------------------------------------------------------------- #
# Rolling statistics (O(n) prefix sums)
# --------------------------------------------------------------------------- #
def _window_sums(x: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(x)))
    return c[w:] - c[:-w]


def rolling_mean(x, window: int) -> np.ndarray:
    x = _arr(x)
    out = np.full(x.size, np.nan)
    if x.size >= window:
        out[window - 1:] = _window_sums(x, window) / window
    return out


def rolling_std(x, window: int) -> np.ndarray:
    """Rolling sample std in O(n).

    The naive E[x^2] - E[x]^2 form suffers catastrophic cancellation when
    the mean is large relative to the dispersion, so the series is centred
    on its global mean first (variance is shift-invariant).
    """
    x = _arr(x)
    out = np.full(x.size, np.nan)
    if window < 2 or x.size < window:
        return out
    d = x - x[0]  # causal shift: a global mean would leak future data into early windows
    s1 = _window_sums(d, window)
    s2 = _window_sums(d * d, window)
    var = (s2 - s1 * s1 / window) / (window - 1)
    out[window - 1:] = np.sqrt(np.maximum(var, 0.0))
    return out


def rolling_volatility(returns, window: int, ppy: float) -> np.ndarray:
    """Annualised rolling realised volatility."""
    return rolling_std(returns, window) * np.sqrt(ppy)


def rolling_downside_deviation(returns, window: int, ppy: float, target: float = 0.0) -> np.ndarray:
    r = _arr(returns)
    out = np.full(r.size, np.nan)
    if r.size >= window:
        sf = np.minimum(r - target, 0.0)
        out[window - 1:] = np.sqrt(_window_sums(sf * sf, window) / window * ppy)
    return out


def rolling_sharpe(returns, window: int, ppy: float) -> np.ndarray:
    m = rolling_mean(returns, window) * ppy
    v = rolling_volatility(returns, window, ppy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, m / v, np.nan)


def rolling_sortino(returns, window: int, ppy: float) -> np.ndarray:
    m = rolling_mean(returns, window) * ppy
    d = rolling_downside_deviation(returns, window, ppy)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, m / d, np.nan)


# --------------------------------------------------------------------------- #
# Trade-level statistics
# --------------------------------------------------------------------------- #
def trade_stats(weights, returns) -> dict[str, float]:
    """Round-trip statistics from a position path.

    A trade is a maximal run of bars holding the same position sign.
    ``returns[t]`` is the strategy return earned by ``weights[t-1]``.
    """
    w = np.sign(_arr(weights))
    r = _arr(returns)
    held = np.r_[0.0, w[:-1]]                         # sign responsible for r[t]
    new_run = np.r_[True, held[1:] != held[:-1]]
    starts = np.flatnonzero(new_run)
    run_sign = held[starts]
    log_r = np.log1p(r)
    run_ret = np.expm1(np.add.reduceat(log_r, starts))
    run_len = np.diff(np.r_[starts, r.size])
    active = run_sign != 0
    tr, tl = run_ret[active], run_len[active]
    if tr.size == 0:
        return {"n_trades": 0, "win_rate": np.nan, "avg_trade": np.nan,
                "avg_win": np.nan, "avg_loss": np.nan, "payoff": np.nan, "avg_bars_held": np.nan}
    wins, losses = tr[tr > 0], tr[tr <= 0]
    return {
        "n_trades": int(tr.size),
        "win_rate": float(wins.size / tr.size),
        "avg_trade": float(tr.mean()),
        "avg_win": float(wins.mean()) if wins.size else np.nan,
        "avg_loss": float(losses.mean()) if losses.size else np.nan,
        "payoff": float(wins.mean() / -losses.mean()) if wins.size and losses.size and losses.mean() < 0 else np.nan,
        "avg_bars_held": float(tl.mean()),
    }


def profit_factor(returns) -> float:
    r = _arr(returns)
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


# --------------------------------------------------------------------------- #
# One-shot summary
# --------------------------------------------------------------------------- #
def summarize(equity, ppy: float, risk_free_annual: float = 0.0) -> dict[str, float]:
    e = _arr(equity)
    r = e[1:] / e[:-1] - 1.0
    mdd = max_drawdown(e)
    active = r[r != 0]
    return {
        "total_return": total_return(e),
        "cagr": cagr(e, ppy),
        "ann_return_arith": mean(r) * ppy,
        "ann_volatility": annualized_volatility(r, ppy),
        "sharpe": sharpe_ratio(r, ppy, risk_free_annual),
        "sortino": sortino_ratio(r, ppy, risk_free_annual),
        "max_drawdown": mdd.depth,
        "mdd_length_bars": mdd.length,
        "mdd_duration_bars": mdd.duration if mdd.duration is not None else np.nan,
        "calmar": calmar_ratio(e, ppy),
        "ulcer_index": ulcer_index(e),
        "var_95": value_at_risk(r, 0.95),
        "cvar_95": conditional_var(r, 0.95),
        "var_99": value_at_risk(r, 0.99),
        "cvar_99": conditional_var(r, 0.99),
        "skew": skewness(r),
        "excess_kurtosis": excess_kurtosis(r),
        "hit_rate": float((active > 0).mean()) if active.size else np.nan,
        "profit_factor": profit_factor(r),
        "best_bar": float(r.max()),
        "worst_bar": float(r.min()),
    }
