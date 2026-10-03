"""Full revaluation of a portfolio under market states and factor scenarios.

This module is the single pricing path used by historical VaR, stress testing,
P&L attribution and backtesting, so every figure in the application comes from
the same valuation logic.

Market factors
--------------
* equity prices (per ticker, local currency)
* FX spot, expressed as USD per unit of each foreign currency
* government zero curve nodes (3M ... 30Y)
* credit spread curve nodes (per credit curve)
* option implied volatility (per underlying: ATM level, skew and term-structure factors)

A ``ScenarioSet`` holds S joint factor moves relative to a base
``MarketState``: equity/FX simple returns, absolute curve/spread node shifts
and absolute vol shifts. Historical VaR uses S = lookback daily moves; a stress
test uses S = 1.

Position valuation (USD)
------------------------
EQUITY  q * P * FX
FX      q * FX                         (q = foreign-currency amount)
CASH    q * FX
BOND    q/100 * sum_j a_j (1 + z(t_j) + s_credit(t_j) + s_cal + s_override)^{-t_j}
OPTION  q * multiplier * BlackScholes(S, K, T, r(T), sigma) * FX

``s_cal`` is a constant residual spread calibrated so the model reproduces the
position's input price on the valuation date; option vols are anchored to the
input implied vol and move with the underlying's surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from analytics.bonds import (
    BP,
    BondSchedule,
    calibrate_spread_to_rates,
    convexity,
    dv01,
    effective_duration_convexity,
    macaulay_duration,
    modified_duration,
    price_from_zero_matrix,
    yield_to_maturity,
)
from analytics import risk_factors as rf
from analytics.greeks import all_greeks, position_greeks
from analytics.key_rate_duration import key_rate_durations
from analytics.options import black_scholes_price
from analytics.volatility_surface import VolatilitySurface, skew_loading, sticky_delta_vol, sticky_strike_vol, term_loading
from analytics.yield_curve import YieldCurve, interpolation_weights
from models.portfolio import Portfolio
from models.positions import AssetType, Position, fx_usd_per_unit

logger = logging.getLogger(__name__)

VolDynamics = Literal["STICKY_STRIKE", "STICKY_DELTA"]
BondMethod = Literal["FULL_REPRICING", "SENSITIVITY"]


class MarketDataMissingError(KeyError):
    """Raised when a required market observation is unavailable."""

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable
        return str(self.args[0]) if self.args else "Market data missing"


@dataclass(frozen=True)
class MarketState:
    """Snapshot of every market factor on one date."""

    date: date
    equity_prices: dict[str, float]
    fx_rates: dict[str, float]  # currency -> USD per unit (USD -> 1.0)
    curve: YieldCurve
    spread_curves: dict[str, YieldCurve] = field(default_factory=dict)
    atm_vols: dict[str, float] = field(default_factory=dict)
    vol_surfaces: dict[str, VolatilitySurface] = field(default_factory=dict)

    def fx(self, currency: str) -> float:
        if currency == "USD":
            return 1.0
        try:
            return self.fx_rates[currency]
        except KeyError:
            raise MarketDataMissingError(f"No FX rate available for {currency} on {self.date:%d %b %Y}.") from None

    def price(self, ticker: str) -> float:
        try:
            return self.equity_prices[ticker]
        except KeyError:
            raise MarketDataMissingError(f"No market data available for {ticker} on {self.date:%d %b %Y}.") from None

    def credit_curve(self, curve_id: str) -> YieldCurve:
        try:
            return self.spread_curves[curve_id]
        except KeyError:
            raise MarketDataMissingError(f"Spread curve {curve_id} unavailable on {self.date:%d %b %Y}.") from None


@dataclass
class ScenarioSet:
    """S joint market-factor moves relative to a base state.

    Backed by a *factor frame* (see :mod:`analytics.risk_factors`): one array
    per named factor. Factors that are absent are treated as unchanged.
    """

    labels: pd.Index
    factors: dict[str, np.ndarray]
    horizon_years: float = 0.0  # time roll applied to options (0 = instantaneous)
    _curve_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, horizon_years: float = 0.0) -> "ScenarioSet":
        values = frame.to_numpy(dtype=float)
        return cls(frame.index, {str(c): values[:, i] for i, c in enumerate(frame.columns)}, horizon_years)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.factors, index=self.labels)

    @property
    def n(self) -> int:
        return len(self.labels)

    def get(self, name: str) -> np.ndarray:
        arr = self.factors.get(name)
        return np.zeros(self.n) if arr is None else arr

    def equity(self, ticker: str) -> np.ndarray:
        return self.get(rf.factor(rf.EQ, ticker))

    def fx(self, currency: str) -> np.ndarray:
        return np.zeros(self.n) if currency == "USD" else self.get(rf.factor(rf.FX, currency))

    def vol(self, underlying: str) -> np.ndarray:
        return self.get(rf.factor(rf.VOL, underlying))

    def vol_skew(self, underlying: str) -> np.ndarray:
        return self.get(rf.factor(rf.VOLSKEW, underlying))

    def vol_term(self, underlying: str) -> np.ndarray:
        return self.get(rf.factor(rf.VOLTERM, underlying))

    def yield_override(self, ticker: str) -> np.ndarray:
        return self.get(rf.factor(rf.YIELD, ticker))

    def curve(self, curve_id: str | None, tenor_labels: tuple[str, ...]) -> np.ndarray:
        """(S, n_nodes) node shifts of a curve, ordered like ``tenor_labels``."""
        if curve_id is None:
            return np.zeros((self.n, len(tenor_labels)))
        key = (curve_id, tuple(tenor_labels))
        if key not in self._curve_cache:
            self._curve_cache[key] = np.column_stack([self.get(rf.factor(rf.CURVE, curve_id, t)) for t in tenor_labels])
        return self._curve_cache[key]


@dataclass(frozen=True)
class _BondSpec:
    schedule: BondSchedule
    credit_curve: str | None
    spread: float  # calibrated residual spread


@dataclass(frozen=True)
class _OptionSpec:
    underlying: str
    strike: float
    expiry: date
    option_type: str
    multiplier: float
    anchor_vol: float
    anchor_surface_vol: float | None
    anchor_atm: float | None


def _tau(expiry: date, d: date) -> float:
    return max((expiry - d).days, 0) / 365.0


class PortfolioPricer:
    """Values a static portfolio under any ``MarketState`` and scenario set.

    The constructor calibrates bond spreads and option volatility anchors on
    the base (valuation-date) state. Instruments are then revalued with
    today's quantities for any other state - this is the "static portfolio"
    assumption used for hypothetical P&L, VaR and backtesting.
    """

    def __init__(self, portfolio: Portfolio, base_state: MarketState, min_vol: float = 0.01):
        self.portfolio = portfolio
        self.base_state = base_state
        self.min_vol = min_vol
        self.positions: list[Position] = list(portfolio.positions)
        self.ids: list[str] = [p.position_id for p in self.positions]
        self._bonds: dict[str, _BondSpec] = {}
        self._options: dict[str, _OptionSpec] = {}
        self._calibrate()

    # ------------------------------------------------------------------ setup
    def _calibrate(self) -> None:
        state = self.base_state
        for p in self.positions:
            if p.asset_type == AssetType.BOND and p.bond is not None:
                if p.currency != "USD":
                    raise MarketDataMissingError(f"Yield curve data missing for {p.currency} bonds ({p.position_id}).")
                sched = BondSchedule.build(p.bond.maturity_date, p.bond.coupon_rate, p.bond.coupon_frequency)
                times, amounts = sched.flows_after(state.date)
                base = self._all_in_zero(state, p.bond.credit_curve, times)
                try:
                    spread = calibrate_spread_to_rates(p.price, base, times, amounts)
                except ValueError as exc:
                    logger.error("Failed to price bond %s: %s", p.position_id, exc)
                    raise ValueError(f"Failed to price bond {p.position_id}: price inconsistent with curve.") from exc
                self._bonds[p.position_id] = _BondSpec(sched, p.bond.credit_curve, spread)
            elif p.asset_type == AssetType.OPTION and p.option is not None:
                o = p.option
                spot = state.price(o.underlying)
                tau = _tau(o.expiry_date, state.date)
                surf = state.vol_surfaces.get(o.underlying)
                surf_vol = float(surf.vol(o.strike / spot, tau)) if surf is not None else None
                if surf is None:
                    logger.warning("Volatility surface unavailable for %s; using flat vol", o.underlying)
                self._options[p.position_id] = _OptionSpec(
                    underlying=o.underlying,
                    strike=o.strike,
                    expiry=o.expiry_date,
                    option_type=o.option_type.value,
                    multiplier=o.contract_multiplier,
                    anchor_vol=o.volatility,
                    anchor_surface_vol=surf_vol,
                    anchor_atm=state.atm_vols.get(o.underlying),
                )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _all_in_curve(state: MarketState, credit_curve: str | None) -> YieldCurve:
        """Government curve plus the credit spread curve (same tenor grid) if any."""
        if not credit_curve:
            return state.curve
        spread = state.credit_curve(credit_curve)
        return YieldCurve(state.curve.tenors, state.curve.rates + spread.rates, state.curve.labels)

    def _all_in_zero(self, state: MarketState, credit_curve: str | None, times: np.ndarray) -> np.ndarray:
        return self._all_in_curve(state, credit_curve).zero_rate(times)

    def option_vol(self, pid: str, state: MarketState, spot: float, tau: float) -> float:
        """Implied vol of an option position in ``state`` (anchored to the input vol)."""
        spec = self._options[pid]
        sigma = spec.anchor_vol
        surf = state.vol_surfaces.get(spec.underlying)
        if spec.anchor_surface_vol is not None and surf is not None:
            sigma += float(surf.vol(spec.strike / spot, max(tau, 1e-6))) - spec.anchor_surface_vol
        elif spec.anchor_atm is not None and spec.underlying in state.atm_vols:
            sigma += state.atm_vols[spec.underlying] - spec.anchor_atm
        return max(sigma, self.min_vol)

    def exposure_currency(self, p: Position) -> str:
        return p.exposure_currency

    def fx_rate(self, p: Position, state: MarketState) -> float:
        if p.asset_type == AssetType.FX:
            return state.fx(p.exposure_currency)
        return state.fx(p.currency)

    def bond_price(self, pid: str, state: MarketState) -> tuple[float, np.ndarray, np.ndarray]:
        """(dirty price per 100, times, amounts) of a bond position in ``state``."""
        spec = self._bonds[pid]
        times, amounts = spec.schedule.flows_after(state.date)
        if times.size == 0:
            return 0.0, times, amounts
        z = self._all_in_zero(state, spec.credit_curve, times) + spec.spread
        return float(np.sum(amounts * (1.0 + z) ** (-times))), times, amounts

    # -------------------------------------------------------------- valuation
    def value(self, state: MarketState) -> pd.Series:
        """USD value of every position in ``state`` (instruments aged to ``state.date``)."""
        out = np.zeros(len(self.positions))
        for j, p in enumerate(self.positions):
            fx = self.fx_rate(p, state)
            if p.asset_type == AssetType.EQUITY:
                out[j] = p.quantity * state.price(p.ticker) * fx
            elif p.asset_type in (AssetType.FX, AssetType.CASH):
                out[j] = p.quantity * fx
            elif p.asset_type == AssetType.BOND:
                out[j] = p.quantity / 100.0 * self.bond_price(p.position_id, state)[0] * fx
            elif p.asset_type == AssetType.OPTION:
                spec = self._options[p.position_id]
                spot = state.price(spec.underlying)
                tau = _tau(spec.expiry, state.date)
                sigma = self.option_vol(p.position_id, state, spot, tau)
                r = float(state.curve.continuous_rate(max(tau, 1e-6))[0])
                px = black_scholes_price(spot, spec.strike, tau, r, sigma, spec.option_type)
                out[j] = p.quantity * spec.multiplier * px * fx
        return pd.Series(out, index=self.ids, name="market_value")

    def coupon_cash(self, start: date, end: date) -> pd.Series:
        """USD coupon/principal cash received in (start, end] per position."""
        out = pd.Series(0.0, index=self.ids)
        for p in self.positions:
            if p.position_id in self._bonds:
                out[p.position_id] = p.quantity / 100.0 * self._bonds[p.position_id].schedule.cash_between(start, end)
        return out

    # -------------------------------------------------------------- scenarios
    def scenario_pnl(
        self,
        state: MarketState,
        scenarios: ScenarioSet,
        vol_dynamics: VolDynamics = "STICKY_STRIKE",
        bond_method: BondMethod = "FULL_REPRICING",
    ) -> pd.DataFrame:
        """Hypothetical P&L (USD) of every position under every scenario.

        Returns:
            (S x positions) frame indexed by the scenario labels.
        """
        S = scenarios.n
        n_nodes = state.curve.tenors.size
        out = np.zeros((S, len(self.positions)))
        for j, p in enumerate(self.positions):
            fx0 = self.fx_rate(p, state)
            fx_ret = scenarios.fx(p.exposure_currency if p.asset_type == AssetType.FX else p.currency)
            if p.asset_type == AssetType.EQUITY:
                v0 = p.quantity * state.price(p.ticker) * fx0
                out[:, j] = v0 * ((1.0 + scenarios.equity(p.ticker)) * (1.0 + fx_ret) - 1.0)
            elif p.asset_type in (AssetType.FX, AssetType.CASH):
                out[:, j] = p.quantity * fx0 * fx_ret
            elif p.asset_type == AssetType.BOND:
                out[:, j] = self._bond_scenarios(p, state, scenarios, n_nodes, bond_method) * (1.0 + fx_ret) + (
                    self.bond_price(p.position_id, state)[0] * p.quantity / 100.0 * fx0 * fx_ret
                )
            elif p.asset_type == AssetType.OPTION:
                out[:, j] = self._option_scenarios(p, state, scenarios, vol_dynamics, fx0, fx_ret)
        return pd.DataFrame(out, index=scenarios.labels, columns=self.ids)

    def _bond_scenarios(
        self, p: Position, state: MarketState, sc: ScenarioSet, n_nodes: int, method: BondMethod
    ) -> np.ndarray:
        """Local-currency bond P&L per scenario (before FX translation)."""
        spec = self._bonds[p.position_id]
        p0, times, amounts = self.bond_price(p.position_id, state)
        if times.size == 0:
            return np.zeros(sc.n)
        fx0 = self.fx_rate(p, state)
        notional = p.quantity / 100.0 * fx0
        w = interpolation_weights(state.curve.tenors, times)  # (n_cf, n_nodes)
        labels = state.curve.labels
        node_shift = sc.curve(rf.GOVT_CURVE, labels) + sc.curve(spec.credit_curve, labels)  # (S, n_nodes)
        override = sc.yield_override(p.ticker)
        if method == "FULL_REPRICING":
            z0 = self._all_in_zero(state, spec.credit_curve, times) + spec.spread
            z = z0[None, :] + node_shift @ w.T + override[:, None]
            return notional * (price_from_zero_matrix(z, times, amounts) - p0)
        # Sensitivity approximation: KRD-weighted yield move with effective duration / convexity.
        z_curve = self._all_in_curve(state, spec.credit_curve)
        krd = key_rate_durations(z_curve, times, amounts, spec.spread).to_numpy()
        eff_d, eff_c = effective_duration_convexity(z_curve, times, amounts, spec.spread)
        dy = (node_shift @ krd) / eff_d + override if eff_d else override
        return notional * p0 * (-eff_d * dy + 0.5 * eff_c * dy**2)

    def _option_scenarios(
        self, p: Position, state: MarketState, sc: ScenarioSet, vol_dynamics: VolDynamics, fx0: float, fx_ret: np.ndarray
    ) -> np.ndarray:
        spec = self._options[p.position_id]
        s0 = state.price(spec.underlying)
        tau0 = _tau(spec.expiry, state.date)
        sigma0 = self.option_vol(p.position_id, state, s0, tau0)
        z0 = float(state.curve.zero_rate(max(tau0, 1e-6))[0])
        r0 = np.log1p(z0)
        px0 = black_scholes_price(s0, spec.strike, tau0, r0, sigma0, spec.option_type)

        s1 = s0 * (1.0 + sc.equity(spec.underlying))
        tau1 = np.full(sc.n, max(tau0 - sc.horizon_years, 0.0))
        surf = state.vol_surfaces.get(spec.underlying)
        # Vol move at the option's own strike and maturity: level + skew + term-structure factors.
        vol_shift = (
            sc.vol(spec.underlying)
            + sc.vol_skew(spec.underlying) * float(skew_loading(spec.strike / s0))
            + sc.vol_term(spec.underlying) * float(term_loading(tau0))
        )
        if vol_dynamics == "STICKY_DELTA" and surf is not None:
            sigma1 = sticky_delta_vol(sigma0, surf, spec.strike, s0, s1, tau0, vol_shift, self.min_vol)
        else:
            if vol_dynamics == "STICKY_DELTA":
                logger.warning("Sticky-delta requested but no surface for %s; using sticky strike", spec.underlying)
            sigma1 = sticky_strike_vol(sigma0, vol_shift, self.min_vol)
        w = interpolation_weights(state.curve.tenors, max(tau0, 1e-6))[0]
        r1 = np.log1p(z0 + sc.curve(rf.GOVT_CURVE, state.curve.labels) @ w)
        px1 = black_scholes_price(s1, spec.strike, tau1, r1, sigma1, spec.option_type)
        units = p.quantity * spec.multiplier
        return units * (px1 * fx0 * (1.0 + fx_ret) - px0 * fx0)

    # ------------------------------------------------------------ analytics
    def option_analytics(self, state: MarketState | None = None) -> pd.DataFrame:
        """Pricing inputs, per-unit Greeks and position Greeks for option positions."""
        state = state or self.base_state
        rows = []
        for p in self.positions:
            if p.position_id not in self._options:
                continue
            spec = self._options[p.position_id]
            spot = state.price(spec.underlying)
            tau = _tau(spec.expiry, state.date)
            sigma = self.option_vol(p.position_id, state, spot, tau)
            r = float(state.curve.continuous_rate(max(tau, 1e-6))[0])
            fx = self.fx_rate(p, state)
            px = black_scholes_price(spot, spec.strike, tau, r, sigma, spec.option_type)
            raw = all_greeks(spot, spec.strike, tau, r, sigma, spec.option_type)
            pos = position_greeks(p.quantity, spec.multiplier, spot, raw, fx)
            rows.append(
                {
                    "position_id": p.position_id,
                    "underlying": spec.underlying,
                    "option_type": spec.option_type,
                    "quantity": p.quantity,
                    "spot": spot,
                    "strike": spec.strike,
                    "expiry": spec.expiry,
                    "time_to_expiry": tau,
                    "rate": r,
                    "implied_vol": sigma,
                    "price": px,
                    "market_value": p.quantity * spec.multiplier * px * fx,
                    **{f"unit_{k}": v for k, v in raw.items()},
                    **pos,
                }
            )
        return pd.DataFrame(rows)

    def bond_analytics(self, state: MarketState | None = None) -> pd.DataFrame:
        """Yield, durations, convexity and DV01 for bond positions."""
        state = state or self.base_state
        rows = []
        for p in self.positions:
            if p.position_id not in self._bonds:
                continue
            spec = self._bonds[p.position_id]
            px, times, amounts = self.bond_price(p.position_id, state)
            if times.size == 0:
                continue
            f = spec.schedule.frequency
            fx = self.fx_rate(p, state)
            mv = p.quantity / 100.0 * px * fx
            y = yield_to_maturity(px, times, amounts, f)
            z_curve = self._all_in_curve(state, spec.credit_curve)
            eff_d, eff_c = effective_duration_convexity(z_curve, times, amounts, spec.spread)
            rows.append(
                {
                    "position_id": p.position_id,
                    "ticker": p.ticker,
                    "face": p.quantity,
                    "coupon": spec.schedule.coupon_rate,
                    "maturity": spec.schedule.maturity,
                    "price": px,
                    "market_value": mv,
                    "ytm": y,
                    "macaulay_duration": macaulay_duration(y, times, amounts, f),
                    "modified_duration": modified_duration(y, times, amounts, f),
                    "convexity": convexity(y, times, amounts, f),
                    "dv01": dv01(y, times, amounts, f) * p.quantity / 100.0 * fx,
                    "effective_duration": eff_d,
                    "effective_convexity": eff_c,
                    "spread_bps": spec.spread / BP,
                    "credit_curve": spec.credit_curve or "GOVT",
                }
            )
        return pd.DataFrame(rows)

    def key_rate_durations(self, state: MarketState | None = None) -> pd.DataFrame:
        """Bonds x tenors KRD matrix."""
        state = state or self.base_state
        rows = {}
        for p in self.positions:
            if p.position_id not in self._bonds:
                continue
            spec = self._bonds[p.position_id]
            _, times, amounts = self.bond_price(p.position_id, state)
            if times.size == 0:
                continue
            z_curve = self._all_in_curve(state, spec.credit_curve)
            rows[p.position_id] = key_rate_durations(z_curve, times, amounts, spec.spread)
        return pd.DataFrame(rows).T if rows else pd.DataFrame(columns=list(state.curve.labels))

    def bond_spec(self, pid: str) -> _BondSpec:
        return self._bonds[pid]

    def option_spec(self, pid: str) -> _OptionSpec:
        return self._options[pid]

    def is_credit(self, pid: str) -> bool:
        return pid in self._bonds and self._bonds[pid].credit_curve is not None


def fx_quote_to_usd(ticker: str, quoted: float) -> float:
    """Convenience re-export for services that hold quoted FX rates."""
    return fx_usd_per_unit(ticker, quoted)
