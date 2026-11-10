import numpy as np
import pandas as pd
import pytest

from quantlab.data import TickGenConfig, generate_ticks, periods_per_year, ticks_to_bars
from quantlab.engine import CapitalConstraints, CostModel, run_backtest
from quantlab.strategies import MeanReversion

FREE = CostModel(commission_bps=0, min_commission=0, spread_capture=0, impact_coef=0,
                 borrow_rate_annual=0, financing_rate_annual=0)
RAW = CapitalConstraints(weight_step=0, flat_overnight=False)


@pytest.fixture(scope="module")
def bars():
    t = generate_ticks(TickGenConfig(n_days=3, ticks_per_second=1.0, dirty_fraction=0, seed=11))
    return ticks_to_bars(t.set_index("timestamp"))


@pytest.fixture(scope="module")
def ppy(bars):
    return periods_per_year(bars)


def _loop_reference(bars, w, costs, E0):
    """Straightforward per-bar loop implementation of the same accounting."""
    close = bars["close"].to_numpy()
    hs = bars["spread_bps"].to_numpy() * 0.5e-4 * costs.spread_capture
    E, w_prev, out = E0, 0.0, []
    for t in range(len(close)):
        r = 0.0 if t == 0 else close[t] / close[t - 1] - 1
        E_pre = E * (1 + w_prev * r)
        w_pre = w_prev * (1 + r) / (1 + w_prev * r)
        n = abs(w[t] - w_pre) * E_pre
        fee = max(costs.min_commission, costs.commission_bps * 1e-4 * n) if n > 1e-9 else 0.0
        E = E_pre - fee - n * hs[t]
        w_prev = w[t]
        out.append(E)
    return np.array(out)


def test_frictionless_buy_and_hold_tracks_price(bars, ppy):
    res = run_backtest(bars, pd.Series(1.0, bars.index), ppy, FREE, RAW)
    expected = 1e6 * bars["close"] / bars["close"].iloc[0]
    np.testing.assert_allclose(res.equity, expected, rtol=1e-10)
    assert res.converged


def test_flat_book_is_flat(bars, ppy):
    res = run_backtest(bars, pd.Series(0.0, bars.index), ppy, CostModel(), RAW)
    assert (res.equity == 1e6).all()
    assert res.stats["total_cost"] == 0


def test_vectorized_engine_matches_loop(bars, ppy):
    rng = np.random.default_rng(3)
    w = np.round(rng.uniform(-1, 1, len(bars)), 1)
    costs = CostModel(commission_bps=1.0, min_commission=5.0, impact_coef=0,
                      borrow_rate_annual=0, financing_rate_annual=0)
    res = run_backtest(bars, pd.Series(w, bars.index), ppy, costs, RAW)
    np.testing.assert_allclose(res.equity.to_numpy(), _loop_reference(bars, w, costs, 1e6), rtol=1e-10)


def test_costs_are_monotone(bars, ppy):
    sig = MeanReversion().generate(bars, ppy)
    finals = [run_backtest(bars, sig, ppy, CostModel(commission_bps=c), RAW).stats["final_equity"]
              for c in (0.0, 1.0, 5.0)]
    assert finals[0] > finals[1] > finals[2]
    gross = run_backtest(bars, sig, ppy, FREE, RAW)
    assert gross.stats["total_return"] == pytest.approx(gross.stats["gross_total_return"], abs=1e-12)


def test_impact_fixed_point_is_self_consistent(bars, ppy):
    sig = MeanReversion().generate(bars, ppy)
    res = run_backtest(bars, sig, ppy, CostModel(impact_coef=2.0), RAW)
    f = res.frame
    assert res.converged and res.iterations > 1
    # Re-derive equity from the reported per-bar costs: must reproduce itself.
    E_prev = np.r_[1e6, f["equity"].to_numpy()[:-1]]
    rebuilt = E_prev * (1 + f["gross_ret"].to_numpy()) - f["cost"].to_numpy()
    np.testing.assert_allclose(rebuilt, f["equity"].to_numpy(), rtol=1e-9)
    # Impact must scale with capital (bigger book -> more bps of cost).
    big = run_backtest(bars, sig, ppy, CostModel(impact_coef=2.0),
                       CapitalConstraints(initial_capital=2e7, weight_step=0, flat_overnight=False))
    assert big.stats["cost_per_traded_bps"] > res.stats["cost_per_traded_bps"]


def test_position_notional_cap_binds(bars, ppy):
    cap = CapitalConstraints(max_position_notional=250_000, weight_step=0, flat_overnight=False)
    res = run_backtest(bars, pd.Series(1.0, bars.index), ppy, FREE, cap)
    held = res.frame["weight"] * res.frame["equity"]
    assert held.max() <= 250_000 * (1 + 1e-6)


def test_drawdown_stop_flattens_permanently(bars, ppy):
    cap = CapitalConstraints(max_drawdown_stop=0.002, weight_step=0, flat_overnight=False)
    res = run_backtest(bars, pd.Series(-1.0 if bars["close"].iloc[-1] > bars["close"].iloc[0] else 1.0,
                                       bars.index), ppy, FREE, cap)
    w = res.frame["weight"].to_numpy()
    first_flat = np.flatnonzero(w == 0)
    assert first_flat.size and (w[first_flat[0]:] == 0).all()
    assert res.stats["max_drawdown"] > -0.01


def test_flat_overnight_and_constraints(bars, ppy):
    cap = CapitalConstraints(allow_short=False, max_leverage=0.5, weight_step=0.25)
    res = run_backtest(bars, pd.Series(np.linspace(-2, 2, len(bars)), bars.index), ppy, FREE, cap)
    w = res.frame["weight"]
    assert w.min() >= 0 and w.max() <= 0.5
    assert set(np.round(w.unique(), 10)) <= {0.0, 0.25, 0.5}
    last_of_day = bars.groupby("session").tail(1).index
    assert (w.loc[last_of_day] == 0).all()


def test_execution_lag_shifts_positions(bars, ppy):
    sig = MeanReversion().generate(bars, ppy)
    a = run_backtest(bars, sig, ppy, FREE, RAW).frame["weight"].to_numpy()
    b = run_backtest(bars, sig, ppy, FREE, CapitalConstraints(weight_step=0, flat_overnight=False,
                                                              execution_lag=2)).frame["weight"].to_numpy()
    np.testing.assert_array_equal(b[2:], a[:-2])
