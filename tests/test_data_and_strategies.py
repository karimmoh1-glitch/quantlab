import numpy as np
import pandas as pd
import pytest

from quantlab import metrics
from quantlab.data import TickGenConfig, generate_ticks, load_ticks, periods_per_year, ticks_to_bars, write_ticks_csv
from quantlab.strategies import STRATEGIES, MeanReversion, Momentum


@pytest.fixture(scope="module")
def ticks():
    return generate_ticks(TickGenConfig(n_days=2, ticks_per_second=1.0, dirty_fraction=0.002, seed=5))


def test_generator_is_deterministic(ticks):
    again = generate_ticks(TickGenConfig(n_days=2, ticks_per_second=1.0, dirty_fraction=0.002, seed=5))
    pd.testing.assert_frame_equal(ticks, again)


def test_csv_roundtrip_cleans_feed_defects(ticks, tmp_path):
    path = write_ticks_csv(ticks, tmp_path / "t.csv")
    df, rep = load_ticks(path, chunksize=10_000)
    assert rep.rows_read == len(ticks)
    assert rep.rows_dropped > 0 and rep.rows_kept == len(df)
    assert (df["bid"] < df["ask"]).all() and (df["price"] > 0).all()
    assert df.index.is_monotonic_increasing
    assert not df.reset_index().duplicated().any()


def test_bars_are_session_only_and_consistent(ticks):
    bars = ticks_to_bars(ticks.dropna().query("price > 0 and bid < ask").set_index("timestamp"))
    assert bars.index.time.min() > pd.Timestamp("09:30").time()
    assert bars.index.time.max() <= pd.Timestamp("16:00").time()
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert periods_per_year(bars) == 390 * 252


def test_ar1_filter_matches_recursion():
    from quantlab.data import _ar1_filter

    rng = np.random.default_rng(1)
    a, e = rng.uniform(0.9, 1.0, 2_000), rng.normal(size=2_000)
    x, ref = 0.5, []
    for ai, ei in zip(a, e):
        x = ai * x + ei
        ref.append(x)
    np.testing.assert_allclose(_ar1_filter(a, e, 0.5), ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_have_no_lookahead(ticks, name):
    bars = ticks_to_bars(ticks.dropna().query("price > 0 and bid < ask").set_index("timestamp"))
    ppy = periods_per_year(bars)
    strat = STRATEGIES[name]()
    base = strat.generate(bars, ppy).to_numpy()
    cut = len(bars) // 2
    shocked = bars.copy()
    shocked.iloc[cut:, shocked.columns.get_loc("close")] *= 1.25  # rewrite the future
    after = strat.generate(shocked, ppy).to_numpy()
    np.testing.assert_array_equal(base[:cut], after[:cut])


def test_mean_reversion_hysteresis():
    close = np.exp(np.r_[np.zeros(30), [0.02], np.full(5, 0.01), np.zeros(30)] + 1e-4 * np.arange(66) % 3e-4)
    bars = pd.DataFrame({"close": close}, index=pd.date_range("2026-01-05 09:31", periods=66, freq="min"))
    w = MeanReversion(lookback=20, entry_z=2.0, exit_z=0.2, target_vol=None).generate(bars, 98_280).to_numpy()
    assert w[30] == -1.0          # spike above the band -> short
    assert (w[31:36] == -1.0).all()  # held while z sits between exit and entry
    assert w[-1] == 0.0           # eventually reverts and exits


def test_momentum_follows_trend():
    lp = np.r_[np.linspace(0, 0.05, 300), np.linspace(0.05, 0.0, 300)]
    lp = lp + np.random.default_rng(2).normal(0, 2e-4, lp.size)
    bars = pd.DataFrame({"close": np.exp(lp)}, index=pd.date_range("2026-01-05", periods=600, freq="min"))
    w = Momentum(target_vol=None).generate(bars, 98_280).to_numpy()
    assert w[280] > 0.5 and w[580] < -0.5
    assert metrics.mean(np.abs(w)) <= 1.0
