import numpy as np
import pandas as pd
import pytest

from quantlab import metrics as m

RNG = np.random.default_rng(0)
R = RNG.normal(0.0004, 0.01, 2_000)


def test_moments_match_pandas():
    s = pd.Series(R)
    assert m.std(R) == pytest.approx(s.std(), rel=1e-12)
    assert m.skewness(R) == pytest.approx(s.skew(), rel=1e-9)
    assert m.excess_kurtosis(R) == pytest.approx(s.kurt(), rel=1e-9)


def test_sharpe_definition():
    ppy = 252
    expected = R.mean() / R.std(ddof=1) * np.sqrt(ppy)
    assert m.sharpe_ratio(R, ppy) == pytest.approx(expected, rel=1e-12)
    rf = 0.03
    ex = R - rf / ppy
    assert m.sharpe_ratio(R, ppy, rf) == pytest.approx(ex.mean() / ex.std(ddof=1) * np.sqrt(ppy))


def test_sortino_uses_full_sample_denominator():
    r = np.array([0.02, -0.01, 0.03, -0.02, 0.0])
    dd = np.sqrt((0.01**2 + 0.02**2) / 5)
    assert m.downside_deviation(r) == pytest.approx(dd)
    assert m.sortino_ratio(r, 1) == pytest.approx(r.mean() / dd)
    assert np.isnan(m.sortino_ratio(np.array([0.01, 0.02]), 252))  # no downside


def test_max_drawdown_known_path():
    eq = np.array([100, 120, 90, 110, 130, 65, 70, 140], float)
    dd = m.max_drawdown(eq)
    assert dd.depth == pytest.approx(65 / 130 - 1)
    assert (dd.peak, dd.trough, dd.recovery) == (4, 5, 7)
    assert dd.duration == 3


def test_max_drawdown_open_and_flat():
    dd = m.max_drawdown(np.array([1.0, 2.0, 1.5, 1.2]))
    assert dd.recovery is None and dd.duration is None
    assert m.max_drawdown(np.ones(5)).depth == 0


def test_drawdown_table_matches_bruteforce():
    eq = np.cumprod(1 + RNG.normal(0, 0.01, 3_000))
    table = m.drawdown_table(eq, top=50)
    # brute force: walk the path
    hwm, episodes, cur = eq[0], [], None
    for i, v in enumerate(eq):
        if v >= hwm:
            if cur:
                episodes.append(cur)
                cur = None
            hwm = v
        else:
            d = v / hwm - 1
            if cur is None:
                cur = [d, i]
            elif d < cur[0]:
                cur = [d, i]
    if cur:
        episodes.append(cur)
    episodes.sort()
    assert len(table) == min(50, len(episodes))
    for got, (depth, trough) in zip(table, episodes):
        assert got.depth == pytest.approx(depth)
        assert got.trough == trough
    assert table[0].depth == pytest.approx(m.max_drawdown(eq).depth)


def test_rolling_std_matches_exact_two_pass_and_is_stable():
    from numpy.lib.stride_tricks import sliding_window_view

    x = 1e6 + RNG.normal(0, 1e-3, 5_000)  # large offset: naive sum-of-squares would fail
    ours = m.rolling_std(x, 50)
    exact = sliding_window_view(x, 50).std(axis=1, ddof=1)
    # Tighter than pandas' own rolling().std(), which drifts ~1e-6 relative here.
    np.testing.assert_allclose(ours[49:], exact, rtol=1e-10)
    assert np.isnan(ours[:49]).all()


def test_rolling_volatility_and_sharpe():
    ppy, w = 98_280, 60
    vol = m.rolling_volatility(R, w, ppy)
    ref = pd.Series(R).rolling(w).std().to_numpy() * np.sqrt(ppy)
    np.testing.assert_allclose(vol[w - 1:], ref[w - 1:], rtol=1e-9)
    sh = m.rolling_sharpe(R, w, ppy)
    i = 500
    win = R[i - w + 1: i + 1]
    assert sh[i] == pytest.approx(m.sharpe_ratio(win, ppy), rel=1e-9)


def test_var_cvar():
    r = np.arange(-50, 50) / 1000.0  # uniform -5%..4.9%
    assert m.value_at_risk(r, 0.95) == pytest.approx(-np.quantile(r, 0.05))
    assert m.conditional_var(r, 0.95) == pytest.approx(-r[:5].mean())


def test_cagr_and_calmar():
    eq = np.array([100.0, 110.0, 121.0])
    assert m.cagr(eq, ppy=2) == pytest.approx(0.21)
    assert np.isnan(m.calmar_ratio(eq, 2))  # no drawdown


def test_trade_stats_segments_runs():
    w = np.array([0, 1, 1, 1, 0, -1, -1, 0, 0])
    r = np.array([0, 0, 0.01, 0.02, -0.01, 0, -0.03, 0.01, 0])  # r[t] earned by w[t-1]
    ts = m.trade_stats(w, r)
    assert ts["n_trades"] == 2
    long_ret = (1 + 0.01) * (1 + 0.02) * (1 - 0.01) - 1
    short_ret = (1 - 0.03) * (1 + 0.01) - 1
    assert ts["avg_trade"] == pytest.approx((long_ret + short_ret) / 2)
    assert ts["win_rate"] == 0.5
