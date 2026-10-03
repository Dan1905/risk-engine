"""Risk-based daily P&L attribution.

Actual P&L is full revaluation of the (static) portfolio between two market
states, plus any coupon cash received::

    Actual = V(state_t, t) - V(state_{t-1}, t-1) + coupons in (t-1, t]

Explained P&L is built from start-of-day sensitivities:

    Equity price   q * (P_t - P_{t-1}) * FX_{t-1}
    FX             q * P_{t-1} * (FX_t - FX_{t-1})          (translation / spot)
    Bond yield     MV * (-D_mod * dy + 0.5 * C * dy^2)       (dy = change in YTM)
    Carry          MV * ((1 + y/f)^{f dt} - 1)               (accrual at yield)
    Option delta   Delta * dS
    Option gamma   0.5 * Gamma * dS^2
    Option vega    Vega * dsigma
    Option theta   Theta * dt

    Residual = Actual - Explained

By construction ``Explained + Residual = Actual``; the residual captures
higher-order and cross terms (e.g. vanna, rho, FX x price) and model
approximations. A small residual relative to actual P&L is evidence that the
sensitivities describe the book well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.bonds import convexity, modified_duration, yield_to_maturity
from analytics.greeks import all_greeks
from analytics.revaluation import MarketState, PortfolioPricer, _tau
from models.positions import AssetType

COMPONENTS: tuple[str, ...] = (
    "equity",
    "bond_yield",
    "fx",
    "option_delta",
    "option_gamma",
    "option_vega",
    "option_theta",
    "carry",
)
COMPONENT_LABELS: dict[str, str] = {
    "equity": "Equity Price P&L",
    "bond_yield": "Bond Yield P&L",
    "fx": "FX P&L",
    "option_delta": "Option Delta P&L",
    "option_gamma": "Option Gamma P&L",
    "option_vega": "Option Vega P&L",
    "option_theta": "Option Theta P&L",
    "carry": "Carry",
    "residual": "Residual",
}


@dataclass
class AttributionResult:
    """Attribution for one day (from ``date_from`` to ``date_to``)."""

    date_from: pd.Timestamp
    date_to: pd.Timestamp
    by_position: pd.DataFrame  # positions x (components, explained, residual, actual)
    nav_start: float
    nav_end: float

    @property
    def totals(self) -> pd.Series:
        return self.by_position.sum(axis=0)

    @property
    def actual(self) -> float:
        return float(self.totals["actual"])

    @property
    def explained(self) -> float:
        return float(self.totals["explained"])

    @property
    def residual(self) -> float:
        return float(self.totals["residual"])

    @property
    def explained_pct(self) -> float:
        """``1 - |residual| / |actual|`` (NaN when actual P&L is ~0)."""
        a = self.actual
        return float("nan") if abs(a) < 1e-9 else 1.0 - abs(self.residual) / abs(a)

    def reconciles(self, tolerance: float) -> bool:
        """Check ``Explained + Residual ≈ Actual`` within ``tolerance``."""
        return abs(self.explained + self.residual - self.actual) <= tolerance


def attribute_pnl(pricer: PortfolioPricer, state0: MarketState, state1: MarketState) -> AttributionResult:
    """Decompose the P&L of ``pricer``'s portfolio between two market states.

    Args:
        pricer: calibrated portfolio pricer.
        state0: start-of-period market state (t-1).
        state1: end-of-period market state (t).

    Returns:
        AttributionResult with per-position components.

    Edge cases:
        Matured bonds (no cash flows left) are attributed entirely to carry
        (their redemption cash) and residual.
    """
    if state1.date <= state0.date:
        raise ValueError("Attribution end date must be after start date")
    v0 = pricer.value(state0)
    v1 = pricer.value(state1)
    cash = pricer.coupon_cash(state0.date, state1.date)
    dt = (state1.date - state0.date).days / 365.0

    comp = pd.DataFrame(0.0, index=pricer.ids, columns=list(COMPONENTS))
    for p in pricer.positions:
        pid = p.position_id
        fx0, fx1 = pricer.fx_rate(p, state0), pricer.fx_rate(p, state1)
        if p.asset_type == AssetType.EQUITY:
            px0, px1 = state0.price(p.ticker), state1.price(p.ticker)
            comp.at[pid, "equity"] = p.quantity * (px1 - px0) * fx0
            comp.at[pid, "fx"] = p.quantity * px0 * (fx1 - fx0)
        elif p.asset_type in (AssetType.FX, AssetType.CASH):
            comp.at[pid, "fx"] = p.quantity * (fx1 - fx0)
        elif p.asset_type == AssetType.BOND:
            px0, t0, a0 = pricer.bond_price(pid, state0)
            px1, t1, a1 = pricer.bond_price(pid, state1)
            if t0.size == 0:
                continue
            f = pricer.bond_spec(pid).schedule.frequency
            mv0 = p.quantity / 100.0 * px0 * fx0
            y0 = yield_to_maturity(px0, t0, a0, f)
            comp.at[pid, "carry"] = mv0 * ((1.0 + y0 / f) ** (f * dt) - 1.0)
            if t1.size:
                y1 = yield_to_maturity(px1, t1, a1, f)
                dy = y1 - y0
                d_mod, cx = modified_duration(y0, t0, a0, f), convexity(y0, t0, a0, f)
                comp.at[pid, "bond_yield"] = mv0 * (-d_mod * dy + 0.5 * cx * dy**2)
        elif p.asset_type == AssetType.OPTION:
            spec = pricer.option_spec(pid)
            s0, s1 = state0.price(spec.underlying), state1.price(spec.underlying)
            tau0, tau1 = _tau(spec.expiry, state0.date), _tau(spec.expiry, state1.date)
            sig0 = pricer.option_vol(pid, state0, s0, tau0)
            sig1 = pricer.option_vol(pid, state1, s1, tau1)
            r0 = float(state0.curve.continuous_rate(max(tau0, 1e-6))[0])
            g = all_greeks(s0, spec.strike, tau0, r0, sig0, spec.option_type)
            units = p.quantity * spec.multiplier * fx0
            ds, dsig = s1 - s0, sig1 - sig0
            comp.at[pid, "option_delta"] = units * g["delta"] * ds
            comp.at[pid, "option_gamma"] = units * 0.5 * g["gamma"] * ds**2
            comp.at[pid, "option_vega"] = units * g["vega"] * dsig
            comp.at[pid, "option_theta"] = units * g["theta"] * dt
            if fx0 != fx1:
                comp.at[pid, "fx"] = v0[pid] / fx0 * (fx1 - fx0)

    actual = v1 - v0 + cash
    out = comp.copy()
    out["explained"] = comp.sum(axis=1)
    out["actual"] = actual
    out["residual"] = out["actual"] - out["explained"]
    return AttributionResult(
        date_from=pd.Timestamp(state0.date),
        date_to=pd.Timestamp(state1.date),
        by_position=out,
        nav_start=float(v0.sum()),
        nav_end=float(v1.sum() + cash.sum()),
    )


def attribution_by_asset_class(result: AttributionResult, asset_classes: pd.Series) -> pd.DataFrame:
    """Aggregate a position-level attribution by asset class."""
    return result.by_position.groupby(asset_classes.reindex(result.by_position.index)).sum()


def summarise_components(result: AttributionResult) -> pd.DataFrame:
    """Portfolio-level component table with display labels."""
    totals = result.totals
    rows = [(COMPONENT_LABELS[c], float(totals[c])) for c in COMPONENTS]
    rows.append(("Explained P&L", float(totals["explained"])))
    rows.append(("Residual (Unexplained)", float(totals["residual"])))
    rows.append(("Actual P&L", float(totals["actual"])))
    return pd.DataFrame(rows, columns=["component", "pnl"])


def attribution_series(results: list[AttributionResult]) -> pd.DataFrame:
    """Stack daily attribution totals into a date-indexed frame."""
    if not results:
        return pd.DataFrame(columns=[*COMPONENTS, "explained", "residual", "actual"])
    frame = pd.DataFrame([r.totals for r in results], index=[r.date_to for r in results])
    frame.index.name = "date"
    return frame[[*COMPONENTS, "explained", "residual", "actual"]]


def explained_ratio(frame: pd.DataFrame) -> float:
    """Aggregate explained share over a period: ``1 - sum|residual| / sum|actual|``."""
    denom = float(np.abs(frame["actual"]).sum())
    return float("nan") if denom == 0 else 1.0 - float(np.abs(frame["residual"]).sum()) / denom
