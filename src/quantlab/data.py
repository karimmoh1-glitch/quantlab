"""Mock tick generation, CSV ingestion and tick -> bar aggregation.

The synthetic price process is deliberately a mixture of two components so
that both strategy families have something to find:

* a regime-switching drift (trend) component  -> rewards momentum
* an Ornstein-Uhlenbeck deviation component    -> rewards mean reversion

Both are simulated exactly on the irregular (Poisson) tick clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
SESSION_SECONDS = 6.5 * 3600
TRADING_DAYS_PER_YEAR = 252

TICK_COLUMNS = ["timestamp", "symbol", "bid", "ask", "price", "size", "side"]
TICK_DTYPES = {
    "symbol": "category",
    "bid": "float64",
    "ask": "float64",
    "price": "float64",
    "size": "float64",
    "side": "int8",
}


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TickGenConfig:
    symbol: str = "MOCK"
    start_date: str = "2026-01-05"
    n_days: int = 10
    ticks_per_second: float = 3.0
    s0: float = 100.0
    tick_size: float = 0.01
    # trend component (annualized)
    trend_vol: float = 0.18
    regime_drift: float = 3.00          # |annualized drift| in a trending regime
    regime_mean_minutes: float = 90.0   # mean regime duration
    # mean-reverting component
    ou_vol: float = 0.12                # annualized diffusion of the OU leg
    ou_half_life_minutes: float = 8.0
    # microstructure
    half_spread_bps: float = 1.5
    jump_intensity_per_day: float = 2.0
    jump_scale: float = 0.004
    overnight_gap_vol: float = 0.006
    dirty_fraction: float = 0.0005      # share of corrupted rows (exercise cleaning)
    seed: int = 7


def generate_ticks(cfg: TickGenConfig = TickGenConfig()) -> pd.DataFrame:
    """Simulate a multi-day tick tape (quotes + prints) for one symbol."""
    rng = np.random.default_rng(cfg.seed)
    sec_per_year = TRADING_DAYS_PER_YEAR * SESSION_SECONDS
    theta = np.log(2) / (cfg.ou_half_life_minutes * 60.0)  # per second
    sig_tr = cfg.trend_vol / np.sqrt(sec_per_year)
    sig_ou = cfg.ou_vol / np.sqrt(sec_per_year)
    mu_reg = cfg.regime_drift / sec_per_year

    days = pd.bdate_range(cfg.start_date, periods=cfg.n_days)
    frames = []
    log_trend, ou = np.log(cfg.s0), 0.0

    for day in days:
        n = rng.poisson(cfg.ticks_per_second * SESSION_SECONDS)
        t = np.sort(rng.uniform(0.0, SESSION_SECONDS, n))
        dt = np.diff(t, prepend=0.0)
        dt[0] = max(dt[0], 1e-3)

        # Regime chain {-1, 0, +1}, switching as a Poisson process in time.
        switch = rng.random(n) < 1.0 - np.exp(-dt / (cfg.regime_mean_minutes * 60.0))
        seg = np.cumsum(switch)
        regime = rng.choice([-1.0, 0.0, 1.0], size=seg[-1] + 1, p=[0.3, 0.4, 0.3])[seg]

        # Trend leg: arithmetic Brownian motion in log space with regime drift.
        jumps = (rng.random(n) < cfg.jump_intensity_per_day / n) * rng.laplace(0, cfg.jump_scale, n)
        d_trend = mu_reg * regime * dt + sig_tr * np.sqrt(dt) * rng.standard_normal(n) + jumps
        trend = log_trend + rng.normal(0, cfg.overnight_gap_vol) + np.cumsum(d_trend)

        # OU leg, exact discretisation x' = a x + s z  ->  closed form as a linear filter.
        a = np.exp(-theta * dt)
        s = sig_ou * np.sqrt((1 - a**2) / (2 * theta))
        ou_path = _ar1_filter(a, s * rng.standard_normal(n), ou)

        mid = np.exp(trend + ou_path)
        hs = mid * cfg.half_spread_bps * 1e-4 * rng.lognormal(0.0, 0.35, n)
        bid = np.floor((mid - hs) / cfg.tick_size) * cfg.tick_size
        ask = np.ceil((mid + hs) / cfg.tick_size) * cfg.tick_size
        ask = np.maximum(ask, bid + cfg.tick_size)
        side = np.where(rng.random(n) < 0.5, 1, -1).astype(np.int8)
        price = np.where(side > 0, ask, bid)
        size = np.maximum(1, np.round(rng.lognormal(5.0, 1.0, n)))

        frames.append(pd.DataFrame({
            "timestamp": day + SESSION_OPEN + pd.to_timedelta(t, unit="s"),
            "symbol": cfg.symbol, "bid": np.round(bid, 4), "ask": np.round(ask, 4),
            "price": np.round(price, 4), "size": size, "side": side,
        }))
        log_trend, ou = trend[-1], ou_path[-1]

    ticks = pd.concat(frames, ignore_index=True)
    return _corrupt(ticks, cfg.dirty_fraction, rng)


def _ar1_filter(a: np.ndarray, eps: np.ndarray, x0: float) -> np.ndarray:
    """Vectorised x_k = a_k x_{k-1} + eps_k with time-varying a_k.

    Uses x_k = P_k (x0 + sum_{j<=k} eps_j / P_j), P_k = prod a, computed in
    blocks so that 1/P_j never under/overflows.
    """
    out = np.empty_like(eps)
    block = 512
    for i in range(0, len(eps), block):
        aa, ee = a[i:i + block], eps[i:i + block]
        p = np.cumprod(aa)
        out[i:i + block] = p * (x0 + np.cumsum(ee / p))
        x0 = out[i + len(aa) - 1]
    return out


def _corrupt(ticks: pd.DataFrame, frac: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic feed defects: crossed quotes, NaNs, zero prices, dupes."""
    k = int(len(ticks) * frac)
    if k == 0:
        return ticks
    idx = rng.choice(len(ticks), size=k, replace=False)
    kind = rng.integers(0, 4, k)
    ticks.loc[idx[kind == 0], "bid"] = ticks.loc[idx[kind == 0], "ask"] + 0.5
    ticks.loc[idx[kind == 1], "price"] = np.nan
    ticks.loc[idx[kind == 2], "price"] = 0.0
    dupes = ticks.loc[idx[kind == 3]]
    return pd.concat([ticks, dupes]).sort_values("timestamp", kind="stable").reset_index(drop=True)


def write_ticks_csv(ticks: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ticks.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f", float_format="%.4f")
    return path


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
@dataclass
class IngestReport:
    path: str
    rows_read: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    rows_kept: int = 0
    seconds: float = 0.0

    @property
    def rows_dropped(self) -> int:
        return sum(self.dropped.values())


def load_ticks(path: str | Path, chunksize: int | None = 250_000) -> tuple[pd.DataFrame, IngestReport]:
    """Stream a tick CSV into a validated, time-indexed DataFrame.

    Typed columns are declared up front (no dtype inference) and the file is
    read in chunks so memory stays bounded for large tapes.
    """
    import time

    t0 = time.perf_counter()
    rep = IngestReport(path=str(path))
    reader = pd.read_csv(path, usecols=TICK_COLUMNS, dtype=TICK_DTYPES, chunksize=chunksize)
    chunks = [reader] if isinstance(reader, pd.DataFrame) else reader
    parts = []
    for chunk in chunks:
        rep.rows_read += len(chunk)
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="ISO8601")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)

    checks = {
        "missing_values": df[["bid", "ask", "price", "size"]].isna().any(axis=1),
        "non_positive_price": (df["price"] <= 0) | (df["bid"] <= 0) | (df["ask"] <= 0),
        "crossed_quote": df["bid"] >= df["ask"],
        "non_positive_size": df["size"] <= 0,
    }
    bad = np.zeros(len(df), dtype=bool)
    for name, mask in checks.items():
        m = mask.to_numpy() & ~bad
        rep.dropped[name] = int(m.sum())
        bad |= m
    df = df.loc[~bad]

    dup = df.duplicated(keep="first").to_numpy()
    rep.dropped["duplicate_row"] = int(dup.sum())
    df = df.loc[~dup]

    df = df.sort_values("timestamp", kind="stable").set_index("timestamp")
    rep.rows_kept = len(df)
    rep.seconds = time.perf_counter() - t0
    return df, rep


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def ticks_to_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Aggregate ticks into OHLC mid-price bars with liquidity statistics.

    Bars are labelled by their *close* time (``label="right"``) so that a bar
    stamped 09:31 contains only information available at 09:31.
    """
    mid = (ticks["bid"].to_numpy() + ticks["ask"].to_numpy()) * 0.5
    spread_bps = (ticks["ask"].to_numpy() - ticks["bid"].to_numpy()) / mid * 1e4
    notional = ticks["price"].to_numpy() * ticks["size"].to_numpy()
    f = pd.DataFrame(
        {"mid": mid, "spread_bps": spread_bps, "size": ticks["size"].to_numpy(),
         "notional": notional, "signed": ticks["side"].to_numpy() * ticks["size"].to_numpy()},
        index=ticks.index,
    )
    g = f.resample(freq, label="right", closed="right")
    bars = g["mid"].ohlc()
    bars["volume"] = g["size"].sum()
    bars["dollar_volume"] = g["notional"].sum()
    bars["vwap"] = bars["dollar_volume"] / bars["volume"]
    bars["spread_bps"] = g["spread_bps"].mean()
    bars["order_imbalance"] = g["signed"].sum() / bars["volume"]
    bars["n_ticks"] = g["mid"].count()
    bars = bars[bars["n_ticks"] > 0].copy()  # drops overnight / weekend buckets
    bars["session"] = bars.index.normalize()
    return bars


def periods_per_year(bars: pd.DataFrame) -> float:
    """Annualisation factor inferred from the observed bar density."""
    per_day = bars.groupby("session").size()
    return float(per_day.median() * TRADING_DAYS_PER_YEAR)
