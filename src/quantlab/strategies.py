"""Signal generators: bars in, target portfolio weight (fraction of equity) out.

Every strategy is a pure function of information available at each bar's
close; the engine trades at that close and earns the *next* bar's return, so
no shifting is needed here (``tests/test_strategies.py`` checks this by
perturbing the future and confirming past weights are unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics


def _vol_scale(close: np.ndarray, window: int, ppy: float, target_vol: float | None, cap: float) -> np.ndarray:
    """Inverse-volatility position scale so each strategy runs at ~target_vol."""
    if target_vol is None:
        return np.ones_like(close)
    r = np.r_[0.0, np.diff(np.log(close))]
    vol = metrics.rolling_volatility(r, window, ppy)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(vol > 0, target_vol / vol, 0.0)
    return np.clip(np.nan_to_num(scale), 0.0, cap)


@dataclass(frozen=True)
class MeanReversion:
    """Z-score fade of log-price vs. its rolling mean, with hysteresis.

    Enter short above +entry_z, long below -entry_z, exit once |z| < exit_z.
    The entry/exit state machine is vectorised with the forward-fill trick:
    bars that are neither an entry nor an exit carry the previous state.
    """

    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.2
    target_vol: float | None = 0.20
    vol_window: int = 60
    max_leverage: float = 1.0

    name = "mean_reversion"

    def generate(self, bars: pd.DataFrame, ppy: float) -> pd.Series:
        close = bars["close"].to_numpy(np.float64)
        lp = np.log(close)
        mu = metrics.rolling_mean(lp, self.lookback)
        sd = metrics.rolling_std(lp, self.lookback)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (lp - mu) / sd
        state = np.full(close.size, np.nan)
        state[np.abs(z) < self.exit_z] = 0.0
        state[z > self.entry_z] = -1.0
        state[z < -self.entry_z] = 1.0
        state = pd.Series(state).ffill().fillna(0.0).to_numpy()
        w = state * _vol_scale(close, self.vol_window, ppy, self.target_vol, self.max_leverage)
        return pd.Series(w, index=bars.index, name=self.name)


@dataclass(frozen=True)
class Momentum:
    """Volatility-normalised EMA crossover (time-series momentum).

    signal = (EMA_fast - EMA_slow) of log-price, divided by the bar volatility
    scaled to the slow horizon; mapped smoothly to [-1, 1] with tanh so the
    position grows with trend conviction instead of flipping on noise.
    """

    fast: int = 30
    slow: int = 240
    conviction: float = 0.5   # signal level that maps to ~76% of full size
    deadband: float = 0.3     # |signal| below this -> flat
    target_vol: float | None = 0.20
    vol_window: int = 60
    max_leverage: float = 1.0

    name = "momentum"

    def generate(self, bars: pd.DataFrame, ppy: float) -> pd.Series:
        close = bars["close"].to_numpy(np.float64)
        lp = pd.Series(np.log(close))
        ema_f = lp.ewm(span=self.fast, adjust=False).mean().to_numpy()
        ema_s = lp.ewm(span=self.slow, adjust=False).mean().to_numpy()
        bar_vol = metrics.rolling_std(np.r_[0.0, np.diff(lp.to_numpy())], self.vol_window)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig = (ema_f - ema_s) / (bar_vol * np.sqrt(self.slow))
        sig = np.nan_to_num(sig)
        sig[np.abs(sig) < self.deadband] = 0.0
        w = np.tanh(sig / self.conviction)
        w *= _vol_scale(close, self.vol_window, ppy, self.target_vol, self.max_leverage)
        return pd.Series(w, index=bars.index, name=self.name)


@dataclass(frozen=True)
class BuyAndHold:
    weight: float = 1.0
    name = "buy_and_hold"

    def generate(self, bars: pd.DataFrame, ppy: float) -> pd.Series:
        return pd.Series(self.weight, index=bars.index, name=self.name)


STRATEGIES = {"mean_reversion": MeanReversion, "momentum": Momentum, "buy_and_hold": BuyAndHold}
