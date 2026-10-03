"""Fixed-rate bond pricing and risk measures.

Yield-based pricing (per 100 face, coupon frequency f)::

    P = sum_j a_j (1 + y/f)^{-f t_j}

With f = 1 this is exactly ``sum C/(1+y)^t + F/(1+y)^T``.

    Macaulay duration  D_mac = sum_j t_j a_j (1+y/f)^{-f t_j} / P
    Modified duration  D_mod = D_mac / (1 + y/f)
    Convexity          C     = sum_j a_j t_j (t_j + 1/f) (1+y/f)^{-f t_j - 2} / P
    DV01               = D_mod * P * 0.0001   (price change for a 1 bp yield move)

Curve-based pricing discounts each cash flow on the zero curve plus a
constant spread: ``P = sum_j a_j (1 + z(t_j) + s)^{-t_j}``. The spread is
calibrated so that the model reproduces the position's market price.

Assumptions: bullet bonds, fixed coupons, ACT/365 year fractions, prices are
dirty (full) prices, coupon dates roll back from maturity by 12/f months.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from analytics.yield_curve import YieldCurve, interpolation_weights

DAY_COUNT = 365.0
BP = 1e-4


def year_fractions(valuation_date: date, dates: np.ndarray) -> np.ndarray:
    """ACT/365 year fractions from ``valuation_date`` to ``dates`` (datetime64[D])."""
    vd = np.datetime64(valuation_date, "D")
    return (np.asarray(dates, dtype="datetime64[D]") - vd).astype(float) / DAY_COUNT


@dataclass(frozen=True)
class BondSchedule:
    """Full coupon schedule of a bullet bond, per 100 face.

    The schedule is generated once (back to ``start``) and then sliced for any
    valuation date, which keeps historical revaluation fast.
    """

    maturity: date
    coupon_rate: float
    frequency: int
    dates: np.ndarray  # datetime64[D], ascending
    amounts: np.ndarray  # per 100 face

    @classmethod
    def build(cls, maturity: date, coupon_rate: float, frequency: int = 2, start: date = date(2000, 1, 1)) -> "BondSchedule":
        if frequency not in (1, 2, 4, 12):
            raise ValueError("frequency must be 1, 2, 4 or 12")
        if coupon_rate < 0:
            raise ValueError("coupon rate must be non-negative")
        months = 12 // frequency
        mat = pd.Timestamp(maturity)
        if coupon_rate == 0:
            dates = [mat]
        else:
            dates, k = [], 0
            while True:
                d = mat - pd.DateOffset(months=months * k)
                if d.date() <= start:
                    break
                dates.append(d)
                k += 1
            dates = dates[::-1]
        arr = np.array([d.to_datetime64() for d in dates], dtype="datetime64[D]")
        amounts = np.full(arr.size, 100.0 * coupon_rate / frequency)
        amounts[-1] += 100.0
        return cls(maturity, coupon_rate, frequency, arr, amounts)

    def flows_after(self, valuation_date: date) -> tuple[np.ndarray, np.ndarray]:
        """(times in years, amounts) of cash flows strictly after ``valuation_date``."""
        vd = np.datetime64(valuation_date, "D")
        mask = self.dates > vd
        return year_fractions(valuation_date, self.dates[mask]), self.amounts[mask]

    def cash_between(self, start: date, end: date) -> float:
        """Cash paid per 100 face in the interval (start, end]."""
        s, e = np.datetime64(start, "D"), np.datetime64(end, "D")
        mask = (self.dates > s) & (self.dates <= e)
        return float(self.amounts[mask].sum())


def generate_cash_flows(
    valuation_date: date, maturity_date: date, coupon_rate: float, frequency: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Future cash-flow times (years) and amounts (per 100 face)."""
    if maturity_date <= valuation_date:
        raise ValueError("Maturity date is earlier than valuation date.")
    return BondSchedule.build(maturity_date, coupon_rate, frequency, start=valuation_date).flows_after(valuation_date)


def _check(times: np.ndarray, amounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    a = np.asarray(amounts, dtype=float)
    if t.size == 0 or t.shape != a.shape:
        raise ValueError("Bond has no future cash flows")
    return t, a


def price_from_yield(ytm: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    """Dirty price per 100 face from a yield to maturity."""
    t, a = _check(times, amounts)
    base = 1.0 + ytm / frequency
    if base <= 0:
        raise ValueError("Yield too negative for the compounding convention")
    return float(np.sum(a * base ** (-frequency * t)))


def yield_to_maturity(price: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    """Solve ``price_from_yield(y) = price`` with Brent's method.

    Raises:
        ValueError: for non-positive prices or if no root exists in [-50%, 100%].
    """
    if price <= 0:
        raise ValueError("Bond price must be positive")
    t, a = _check(times, amounts)
    f = lambda y: price_from_yield(y, t, a, frequency) - price  # noqa: E731
    lo, hi = -0.5 * frequency + 1e-6, 1.0
    if f(lo) * f(hi) > 0:
        raise ValueError("Yield to maturity could not be bracketed")
    return float(brentq(f, lo, hi, xtol=1e-12, maxiter=200))


def macaulay_duration(ytm: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    t, a = _check(times, amounts)
    pv = a * (1.0 + ytm / frequency) ** (-frequency * t)
    return float(np.sum(t * pv) / np.sum(pv))


def modified_duration(ytm: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    return macaulay_duration(ytm, times, amounts, frequency) / (1.0 + ytm / frequency)


def convexity(ytm: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    t, a = _check(times, amounts)
    base = 1.0 + ytm / frequency
    price = np.sum(a * base ** (-frequency * t))
    return float(np.sum(a * t * (t + 1.0 / frequency) * base ** (-frequency * t - 2)) / price)


def dv01(ytm: float, times: np.ndarray, amounts: np.ndarray, frequency: int = 1) -> float:
    """Price change per 100 face for a 1 bp fall in yield (positive for long bonds)."""
    price = price_from_yield(ytm, times, amounts, frequency)
    return modified_duration(ytm, times, amounts, frequency) * price * BP


def price_from_curve(curve: YieldCurve, times: np.ndarray, amounts: np.ndarray, spread: float = 0.0) -> float:
    """Dirty price per 100 face discounting on ``curve`` plus a constant ``spread``."""
    t, a = _check(times, amounts)
    z = curve.zero_rate(t) + spread
    return float(np.sum(a * (1.0 + z) ** (-t)))


def price_from_zero_matrix(zero_rates: np.ndarray, times: np.ndarray, amounts: np.ndarray) -> np.ndarray:
    """Vectorised pricing: ``zero_rates`` is (n_scenarios, n_cf) of all-in discount rates."""
    z = np.asarray(zero_rates, dtype=float)
    if np.any(z <= -0.99):
        raise ValueError("Shocked discount rates below -99%")
    return np.sum(amounts[None, :] * (1.0 + z) ** (-times[None, :]), axis=1)


def calibrate_spread_to_rates(price: float, base_rates: np.ndarray, times: np.ndarray, amounts: np.ndarray) -> float:
    """Constant spread ``s`` such that ``sum a (1 + base + s)^{-t} = price``."""
    t, a = _check(times, amounts)
    z = np.asarray(base_rates, dtype=float)
    if price <= 0:
        raise ValueError("Bond price must be positive")
    f = lambda s: float(np.sum(a * (1.0 + z + s) ** (-t))) - price  # noqa: E731
    lo, hi = -0.9 - float(z.min()) + 1e-6, 1.0
    lo = max(lo, -0.2)
    if f(lo) * f(hi) > 0:
        raise ValueError("Spread could not be calibrated to the bond price")
    return float(brentq(f, lo, hi, xtol=1e-12))


def calibrate_spread(price: float, curve: YieldCurve, times: np.ndarray, amounts: np.ndarray) -> float:
    """Constant spread over the zero curve that reproduces ``price`` (z-spread)."""
    t, _ = _check(times, amounts)
    return calibrate_spread_to_rates(price, curve.zero_rate(t), times, amounts)


def effective_duration_convexity(
    curve: YieldCurve, times: np.ndarray, amounts: np.ndarray, spread: float = 0.0, bump: float = BP
) -> tuple[float, float]:
    """Effective duration and convexity from parallel +/- ``bump`` curve shifts."""
    p0 = price_from_curve(curve, times, amounts, spread)
    up = price_from_curve(curve, times, amounts, spread + bump)
    dn = price_from_curve(curve, times, amounts, spread - bump)
    return float((dn - up) / (2 * bump * p0)), float((up + dn - 2 * p0) / (p0 * bump**2))


def price_change_approximation(price: float, duration: float, convexity_: float, dy: np.ndarray | float) -> np.ndarray | float:
    """Second-order approximation: ``dP = P * (-D dy + 0.5 C dy^2)``."""
    dy = np.asarray(dy, dtype=float)
    out = price * (-duration * dy + 0.5 * convexity_ * dy**2)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class BondAnalytics:
    price: float  # per 100 face
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float  # per 100 face
    effective_duration: float
    effective_convexity: float


def bond_analytics(curve: YieldCurve, times: np.ndarray, amounts: np.ndarray, spread: float, frequency: int) -> BondAnalytics:
    """Full analytics set for one bond priced off the curve plus spread."""
    price = price_from_curve(curve, times, amounts, spread)
    y = yield_to_maturity(price, times, amounts, frequency)
    eff_d, eff_c = effective_duration_convexity(curve, times, amounts, spread)
    return BondAnalytics(
        price=price,
        ytm=y,
        macaulay_duration=macaulay_duration(y, times, amounts, frequency),
        modified_duration=modified_duration(y, times, amounts, frequency),
        convexity=convexity(y, times, amounts, frequency),
        dv01=dv01(y, times, amounts, frequency),
        effective_duration=eff_d,
        effective_convexity=eff_c,
    )


def zero_rates_at(curve: YieldCurve, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Base zero rates at ``times`` and the node->cash-flow interpolation matrix."""
    w = interpolation_weights(curve.tenors, times)
    return w @ curve.rates, w
