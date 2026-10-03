from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.bonds import effective_duration_convexity, generate_cash_flows, price_from_curve
from analytics.key_rate_duration import aggregate_krd, key_rate_dv01, key_rate_durations, krd_pnl_approximation
from analytics.yield_curve import TENOR_LABELS, TENOR_YEARS, YieldCurve, interpolation_weights

VD = date(2026, 9, 30)


@pytest.fixture
def curve() -> YieldCurve:
    return YieldCurve(TENOR_YEARS, np.array([0.039, 0.0385, 0.037, 0.0355, 0.0355, 0.037, 0.039, 0.041, 0.0465, 0.047]))


@pytest.fixture
def ten_year():
    return generate_cash_flows(VD, date(2036, 8, 15), 0.04, 2)


def test_interpolation_weights_sum_to_one():
    w = interpolation_weights(TENOR_YEARS, np.array([0.1, 0.25, 1.5, 12.0, 40.0]))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    assert w[0, 0] == 1.0 and w[-1, -1] == 1.0  # flat extrapolation


def test_key_rate_shock_is_localised(curve, ten_year):
    t, a = ten_year
    krd = key_rate_durations(curve, t, a)
    assert krd["20Y"] == pytest.approx(0.0, abs=1e-10)
    assert krd["30Y"] == pytest.approx(0.0, abs=1e-10)
    assert krd["10Y"] > krd["7Y"] > 0  # principal sits near the 10Y node
    assert krd.idxmax() == "10Y"


def test_krd_sum_equals_effective_duration(curve, ten_year):
    t, a = ten_year
    krd = key_rate_durations(curve, t, a)
    eff_d, _ = effective_duration_convexity(curve, t, a)
    assert krd.sum() == pytest.approx(eff_d, rel=1e-6)


def test_short_bond_has_no_long_end_risk(curve):
    t, a = generate_cash_flows(VD, date(2028, 9, 30), 0.035, 2)
    krd = key_rate_durations(curve, t, a)
    assert krd[["5Y", "7Y", "10Y", "20Y", "30Y"]].abs().sum() == pytest.approx(0.0, abs=1e-12)
    assert krd["2Y"] > 0


def test_repricing_consistency(curve, ten_year):
    t, a = ten_year
    p0 = price_from_curve(curve, t, a)
    krd = key_rate_durations(curve, t, a)
    shifts = pd.Series(0.0, index=list(TENOR_LABELS))
    shifts["10Y"] = 0.0005  # +5 bp at the 10Y node only
    full = price_from_curve(curve.shifted(shifts.to_dict()), t, a) - p0
    approx = krd_pnl_approximation(krd, p0, shifts)
    # First-order error ~ 0.5 * C * dy / D ~ 0.25% of the P&L for a 5 bp node shock
    assert approx == pytest.approx(full, rel=5e-3)


def test_twist_repricing_consistency(curve, ten_year):
    t, a = ten_year
    p0 = price_from_curve(curve, t, a)
    krd = key_rate_durations(curve, t, a)
    shifts = pd.Series(np.linspace(0.0010, -0.0010, 10), index=list(TENOR_LABELS))
    full = price_from_curve(curve.shifted(shifts.to_numpy()), t, a) - p0
    assert krd_pnl_approximation(krd, p0, shifts) == pytest.approx(full, rel=2e-2, abs=1e-4)


def test_aggregation(curve, ten_year):
    t10, a10 = ten_year
    t2, a2 = generate_cash_flows(VD, date(2028, 9, 30), 0.035, 2)
    krd = pd.DataFrame({"B10": key_rate_durations(curve, t10, a10), "B2": key_rate_durations(curve, t2, a2)}).T
    mv = pd.Series({"B10": 1_000_000.0, "B2": 3_000_000.0})
    agg = aggregate_krd(krd, mv)
    # KR01 is additive across positions
    expected = key_rate_dv01(krd.loc["B10"], 1_000_000.0) + key_rate_dv01(krd.loc["B2"], 3_000_000.0)
    np.testing.assert_allclose(agg["KR01"].to_numpy(), expected.to_numpy())
    # MV-weighted KRD sums to MV-weighted duration
    assert agg["KRD"].sum() == pytest.approx((krd.sum(axis=1) * mv).sum() / mv.sum())
