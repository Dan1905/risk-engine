from datetime import date

import numpy as np
import pytest

from analytics.bonds import (
    BondSchedule,
    bond_analytics,
    calibrate_spread,
    convexity,
    dv01,
    effective_duration_convexity,
    generate_cash_flows,
    macaulay_duration,
    modified_duration,
    price_change_approximation,
    price_from_curve,
    price_from_yield,
    yield_to_maturity,
)
from analytics.yield_curve import TENOR_LABELS, TENOR_YEARS, YieldCurve

# Annual 5-year 5% bond with integer cash-flow times.
T5 = np.arange(1.0, 6.0)
A5 = np.array([5.0, 5.0, 5.0, 5.0, 105.0])


def test_price_matches_spec_formula():
    y = 0.06
    expected = sum(5.0 / (1 + y) ** t for t in range(1, 6)) + 100 / (1 + y) ** 5
    assert price_from_yield(y, T5, A5, 1) == pytest.approx(expected)


def test_par_bond():
    assert price_from_yield(0.05, T5, A5, 1) == pytest.approx(100.0)


def test_zero_coupon_duration_equals_maturity():
    t, a = np.array([7.0]), np.array([100.0])
    assert macaulay_duration(0.04, t, a, 1) == pytest.approx(7.0)
    assert modified_duration(0.04, t, a, 1) == pytest.approx(7.0 / 1.04)
    assert convexity(0.04, t, a, 1) == pytest.approx(7 * 8 / 1.04**2)


def test_macaulay_duration_par_bond():
    # Known closed form for a par bond: (1+y)/y * (1 - 1/(1+y)^n)
    y = 0.05
    assert macaulay_duration(y, T5, A5, 1) == pytest.approx((1 + y) / y * (1 - 1 / (1 + y) ** 5))


def test_modified_duration_and_convexity_vs_finite_difference():
    y, h = 0.045, 1e-5
    p0 = price_from_yield(y, T5, A5, 1)
    up, dn = price_from_yield(y + h, T5, A5, 1), price_from_yield(y - h, T5, A5, 1)
    assert modified_duration(y, T5, A5, 1) == pytest.approx(-(up - dn) / (2 * h * p0), rel=1e-7)
    assert convexity(y, T5, A5, 1) == pytest.approx((up + dn - 2 * p0) / (p0 * h**2), rel=1e-4)


def test_semiannual_measures_vs_finite_difference():
    t, a = generate_cash_flows(date(2026, 9, 30), date(2036, 8, 15), 0.04, 2)
    y, h = 0.041, 1e-5
    p0 = price_from_yield(y, t, a, 2)
    up, dn = price_from_yield(y + h, t, a, 2), price_from_yield(y - h, t, a, 2)
    assert modified_duration(y, t, a, 2) == pytest.approx(-(up - dn) / (2 * h * p0), rel=1e-6)
    assert convexity(y, t, a, 2) == pytest.approx((up + dn - 2 * p0) / (p0 * h**2), rel=1e-3)


def test_dv01():
    y = 0.05
    p0 = price_from_yield(y, T5, A5, 1)
    numeric = (price_from_yield(y - 1e-4, T5, A5, 1) - price_from_yield(y + 1e-4, T5, A5, 1)) / 2
    assert dv01(y, T5, A5, 1) == pytest.approx(numeric, rel=1e-4)
    assert dv01(y, T5, A5, 1) == pytest.approx(modified_duration(y, T5, A5, 1) * p0 * 1e-4)


def test_ytm_roundtrip():
    for y in (-0.005, 0.0, 0.03, 0.12):
        p = price_from_yield(y, T5, A5, 1)
        assert yield_to_maturity(p, T5, A5, 1) == pytest.approx(y, abs=1e-10)


def test_ytm_invalid_price():
    with pytest.raises(ValueError):
        yield_to_maturity(-5.0, T5, A5, 1)


def test_schedule_generation():
    t, a = generate_cash_flows(date(2026, 9, 30), date(2028, 9, 15), 0.05, 2)
    assert len(t) == 4
    np.testing.assert_allclose(a, [2.5, 2.5, 2.5, 102.5])
    assert t[-1] == pytest.approx((date(2028, 9, 15) - date(2026, 9, 30)).days / 365.0)
    sched = BondSchedule.build(date(2028, 9, 15), 0.05, 2)
    assert sched.cash_between(date(2027, 3, 14), date(2027, 3, 16)) == pytest.approx(2.5)
    with pytest.raises(ValueError):
        generate_cash_flows(date(2026, 9, 30), date(2026, 1, 1), 0.05, 2)


def test_flat_curve_pricing_equals_yield_pricing():
    curve = YieldCurve(TENOR_YEARS, np.full(10, 0.05), TENOR_LABELS)
    assert price_from_curve(curve, T5, A5) == pytest.approx(price_from_yield(0.05, T5, A5, 1))


def test_spread_calibration_roundtrip():
    curve = YieldCurve(TENOR_YEARS, np.linspace(0.035, 0.045, 10), TENOR_LABELS)
    s = calibrate_spread(97.3, curve, T5, A5)
    assert price_from_curve(curve, T5, A5, s) == pytest.approx(97.3, abs=1e-9)


def test_duration_convexity_approximation_close_to_full_reprice():
    curve = YieldCurve(TENOR_YEARS, np.full(10, 0.04), TENOR_LABELS)
    d, c = effective_duration_convexity(curve, T5, A5)
    p0 = price_from_curve(curve, T5, A5)
    for dy in (0.0025, 0.01, -0.01):
        full = price_from_curve(curve, T5, A5, dy) - p0
        approx = price_change_approximation(p0, d, c, dy)
        assert approx == pytest.approx(full, rel=2e-3)
    # Convexity term improves on duration-only
    full = price_from_curve(curve, T5, A5, 0.02) - p0
    assert abs(price_change_approximation(p0, d, c, 0.02) - full) < abs(-d * p0 * 0.02 - full)


def test_bond_analytics_bundle():
    curve = YieldCurve(TENOR_YEARS, np.full(10, 0.04), TENOR_LABELS)
    t, a = generate_cash_flows(date(2026, 9, 30), date(2031, 9, 30), 0.04, 2)
    ba = bond_analytics(curve, t, a, 0.0, 2)
    assert ba.macaulay_duration > ba.modified_duration > 0
    assert ba.dv01 > 0 and ba.convexity > 0
    assert ba.effective_duration == pytest.approx(ba.modified_duration, rel=0.02)
