"""Vectorised single-asset backtest engine with path-dependent friction.

Timing convention (bar ``t`` closes at ``index[t]``)
----------------------------------------------------
* ``r[t]``  = close[t] / close[t-1] - 1, the return *into* bar t.
* ``w[t]``  = the weight chosen at close t, after trading, held over (t, t+1].
* At close t the book first drifts:  w_pre[t] = w[t-1] (1 + r[t]) / (1 + w[t-1] r[t]),
  then trades |w[t] - w_pre[t]| x pre-trade equity, paying costs.
* E[t] = E[t-1] (1 + w[t-1] r[t]) - cost[t].

Costs and capital constraints that depend on the *dollar* size of the book
(square-root market impact, minimum commissions, a notional position cap and
a drawdown kill-switch) make the equity path depend on itself. Rather than
falling back to a per-bar Python loop, the engine solves the path as a fixed
point: guess E, compute every bar's costs/constraints from that guess in one
vectorised pass, rebuild E with a cumulative product, and repeat. Costs are a
small fraction of equity so the map is a contraction and typically converges
to 1e-12 in 3-6 passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import metrics


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5        # per side, on traded notional
    min_commission: float = 1.0        # $ floor per rebalance that trades
    spread_capture: float = 1.0        # share of the quoted half-spread paid (1 = cross)
    impact_coef: float = 0.3           # sqrt impact: coef * sigma_bar * sqrt(notional / bar $vol)
    vol_window: int = 30               # bars, for sigma_bar in the impact model
    borrow_rate_annual: float = 0.01   # on short notional
    financing_rate_annual: float = 0.05  # on gross exposure above 1x equity


@dataclass(frozen=True)
class CapitalConstraints:
    initial_capital: float = 1_000_000.0
    max_leverage: float = 1.0                    # cap on |weight|
    max_position_notional: float | None = None   # $ cap, binds as equity grows
    allow_short: bool = True
    weight_step: float = 0.1                     # quantise weights (lot / no-trade granularity)
    max_drawdown_stop: float | None = None       # e.g. 0.15 -> flatten for good at -15%
    flat_overnight: bool = True                  # close out on each session's last bar
    execution_lag: int = 0                       # extra bars between signal and fill


@dataclass
class BacktestResult:
    name: str
    frame: pd.DataFrame
    ppy: float
    costs: CostModel
    capital: CapitalConstraints
    iterations: int
    converged: bool
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]

    @property
    def returns(self) -> pd.Series:
        return self.frame["net_ret"]


def run_backtest(
    bars: pd.DataFrame,
    target_weight: pd.Series,
    ppy: float,
    costs: CostModel = CostModel(),
    capital: CapitalConstraints = CapitalConstraints(),
    name: str = "strategy",
    max_iter: int = 50,
    tol: float = 1e-12,
) -> BacktestResult:
    close = bars["close"].to_numpy(np.float64)
    n = close.size
    r = np.r_[0.0, close[1:] / close[:-1] - 1.0]
    E0 = capital.initial_capital

    # ---- static (equity-independent) constraints --------------------------
    tgt = target_weight.reindex(bars.index).fillna(0.0).to_numpy(np.float64)
    if capital.execution_lag:
        k = capital.execution_lag
        tgt = np.r_[np.zeros(k), tgt[:-k]]
    if not capital.allow_short:
        tgt = np.maximum(tgt, 0.0)
    if capital.weight_step > 0:
        tgt = np.round(tgt / capital.weight_step) * capital.weight_step
    tgt = np.clip(tgt, -capital.max_leverage, capital.max_leverage)
    if capital.flat_overnight and "session" in bars:
        s = bars["session"].to_numpy()
        tgt[np.r_[s[1:] != s[:-1], True]] = 0.0

    # ---- per-bar market inputs for the cost model --------------------------
    half_spread = bars["spread_bps"].to_numpy(np.float64) * 0.5e-4 * costs.spread_capture
    dvol = bars["dollar_volume"].to_numpy(np.float64)
    sigma = metrics.rolling_std(r, costs.vol_window)
    first = np.flatnonzero(np.isfinite(sigma))
    sigma[: first[0] if first.size else n] = sigma[first[0]] if first.size else 0.0
    comm_rate = costs.commission_bps * 1e-4
    carry_rate = np.array([costs.borrow_rate_annual, costs.financing_rate_annual]) / ppy

    # ---- fixed-point solve for the self-consistent equity path ------------
    E = np.full(n, E0)
    w = tgt.copy()
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        E_prev = np.r_[E0, E[:-1]]
        # Pre-trade equity at each decision, from the previous pass's book.
        E_dec = E_prev * (1.0 + np.r_[0.0, w[:-1]] * r)
        w = tgt.copy()

        # Dollar position cap, sized off equity available at the decision.
        if capital.max_position_notional is not None:
            with np.errstate(divide="ignore"):
                cap = np.where(E_dec > 0, capital.max_position_notional / E_dec, 0.0)
            w = np.clip(w, -cap, cap)

        # Kill-switch: once pre-trade equity breaches the stop, stay flat.
        w_held = np.r_[0.0, w[:-1]]
        g = 1.0 + w_held * r
        if capital.max_drawdown_stop is not None:
            hwm = np.maximum.accumulate(E_prev)
            breached = np.maximum.accumulate(E_prev * g <= hwm * (1.0 - capital.max_drawdown_stop))
            w[breached] = 0.0
            w_held = np.r_[0.0, w[:-1]]
            g = 1.0 + w_held * r

        with np.errstate(divide="ignore", invalid="ignore"):
            w_pre = np.where(g > 0, w_held * (1.0 + r) / g, 0.0)
        E_pre = E_prev * g
        notional = np.abs(w - w_pre) * np.maximum(E_pre, 0.0)
        trades = notional > 1e-9

        commission = np.where(trades, np.maximum(costs.min_commission, comm_rate * notional), 0.0)
        spread_cost = notional * half_spread
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(dvol > 0, notional / dvol, 0.0)
        impact_cost = notional * costs.impact_coef * sigma * np.sqrt(participation)
        carry_cost = np.maximum(E_prev, 0.0) * (
            carry_rate[0] * np.maximum(-w_held, 0.0)
            + carry_rate[1] * np.maximum(np.abs(w_held) - 1.0, 0.0)
        )
        cost = commission + spread_cost + impact_cost + carry_cost

        with np.errstate(divide="ignore", invalid="ignore"):
            growth = np.where(E_prev > 0, g - cost / E_prev, 0.0)
        E_new = E0 * np.cumprod(np.maximum(growth, 0.0))  # ruin is absorbing
        err = np.max(np.abs(E_new - E)) / E0
        E = E_new
        if err < tol:
            converged = True
            break

    net_ret = np.r_[0.0, E[1:] / np.where(E[:-1] > 0, E[:-1], np.nan) - 1.0]
    frame = pd.DataFrame(
        {
            "close": close, "ret": r, "target": tgt, "weight": w, "weight_pre_trade": w_pre,
            "trade_notional": notional, "participation": participation,
            "commission": commission, "spread_cost": spread_cost,
            "impact_cost": impact_cost, "carry_cost": carry_cost, "cost": cost,
            "gross_ret": w_held * r, "net_ret": np.nan_to_num(net_ret),
            "equity": E, "gross_equity": E0 * np.cumprod(g),
            "drawdown": metrics.drawdown_series(E),
        },
        index=bars.index,
    )
    res = BacktestResult(name, frame, ppy, costs, capital, it, converged)
    res.stats = compute_stats(res)
    return res


def compute_stats(res: BacktestResult) -> dict[str, float]:
    f, ppy = res.frame, res.ppy
    E0 = res.capital.initial_capital
    eq = np.r_[E0, f["equity"].to_numpy()]  # include the pre-trade starting point
    st = metrics.summarize(eq, ppy)
    st.update({f"trade_{k}": v for k, v in metrics.trade_stats(f["weight"], f["net_ret"]).items()})

    total_cost = f["cost"].sum()
    traded = f["trade_notional"].sum()
    avg_eq = f["equity"].mean()
    n_days = f.index.normalize().nunique()
    st.update({
        "gross_total_return": f["gross_equity"].iloc[-1] / E0 - 1.0,
        "total_cost": total_cost,
        "cost_commission": f["commission"].sum(),
        "cost_spread": f["spread_cost"].sum(),
        "cost_impact": f["impact_cost"].sum(),
        "cost_carry": f["carry_cost"].sum(),
        "cost_per_traded_bps": total_cost / traded * 1e4 if traded else np.nan,
        "traded_notional": traded,
        # Gross P&L per dollar traded: the cost level at which the edge breaks even.
        "edge_per_traded_bps": (f["gross_equity"].iloc[-1] - E0) / traded * 1e4 if traded else np.nan,
        "daily_turnover": traded / avg_eq / max(n_days, 1),
        "n_rebalances": int((f["trade_notional"] > 1e-9).sum()),
        "avg_gross_exposure": f["weight"].abs().mean(),
        "time_in_market": float((f["weight"] != 0).mean()),
        "max_participation": f["participation"].max(),
        "p95_participation": float(np.quantile(f["participation"][f["participation"] > 0], 0.95))
        if (f["participation"] > 0).any() else 0.0,
        "final_equity": f["equity"].iloc[-1],
        "solver_iterations": res.iterations,
        "solver_converged": res.converged,
    })
    return st


def config_dict(res: BacktestResult) -> dict:
    return {"costs": asdict(res.costs), "capital": asdict(res.capital)}
