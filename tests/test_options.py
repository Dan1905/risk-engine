import numpy as np
import pytest

from analytics.greeks import all_greeks, delta, gamma, position_greeks, rho, theta, vega
from analytics.options import black_scholes_price, implied_volatility

# Hull-style reference: S=100, K=100, T=1, r=5%, sigma=20%
S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.20


def test_call_price():
    assert black_scholes_price(S, K, T, R, SIG, "CALL") == pytest.approx(10.4506, abs=1e-4)


def test_put_price():
    assert black_scholes_price(S, K, T, R, SIG, "PUT") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    for k in (80.0, 100.0, 125.0):
        c = black_scholes_price(S, k, T, R, SIG, "CALL")
        p = black_scholes_price(S, k, T, R, SIG, "PUT")
        assert c - p == pytest.approx(S - k * np.exp(-R * T), abs=1e-10)


def test_delta():
    assert delta(S, K, T, R, SIG, "CALL") == pytest.approx(0.636831, abs=1e-6)
    assert delta(S, K, T, R, SIG, "PUT") == pytest.approx(-0.363169, abs=1e-6)


def test_gamma():
    assert gamma(S, K, T, R, SIG) == pytest.approx(0.018762, abs=1e-6)


def test_vega():
    assert vega(S, K, T, R, SIG) == pytest.approx(37.5240, abs=1e-3)


def test_theta():
    assert theta(S, K, T, R, SIG, "CALL") == pytest.approx(-6.4140, abs=1e-3)
    assert theta(S, K, T, R, SIG, "PUT") == pytest.approx(-1.6579, abs=1e-3)


def test_rho():
    assert rho(S, K, T, R, SIG, "CALL") == pytest.approx(53.2325, abs=1e-3)
    assert rho(S, K, T, R, SIG, "PUT") == pytest.approx(-41.8905, abs=1e-3)


@pytest.mark.parametrize("opt", ["CALL", "PUT"])
def test_greeks_match_finite_differences(opt):
    px = lambda **kw: black_scholes_price(  # noqa: E731
        kw.get("s", S), K, kw.get("t", T), kw.get("r", R), kw.get("v", SIG), opt
    )
    h = 1e-4
    g = all_greeks(S, K, T, R, SIG, opt)
    assert g["delta"] == pytest.approx((px(s=S + h) - px(s=S - h)) / (2 * h), rel=1e-6)
    assert g["gamma"] == pytest.approx((px(s=S + 1e-2) - 2 * px() + px(s=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((px(v=SIG + h) - px(v=SIG - h)) / (2 * h), rel=1e-6)
    assert g["rho"] == pytest.approx((px(r=R + h) - px(r=R - h)) / (2 * h), rel=1e-6)
    assert g["theta"] == pytest.approx(-(px(t=T + h) - px(t=T - h)) / (2 * h), rel=1e-5)


def test_vectorised():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = black_scholes_price(S, strikes, T, R, SIG, np.array(["CALL", "PUT", "CALL"]))
    assert prices.shape == (3,)
    assert prices[1] == pytest.approx(5.5735, abs=1e-4)


def test_near_zero_maturity_is_intrinsic():
    assert black_scholes_price(110.0, 100.0, 0.0, R, SIG, "CALL") == pytest.approx(10.0)
    assert black_scholes_price(90.0, 100.0, 1e-9, R, SIG, "PUT") == pytest.approx(10.0)
    assert delta(110.0, 100.0, 0.0, R, SIG, "CALL") == 1.0
    assert gamma(110.0, 100.0, 0.0, R, SIG) == 0.0


def test_zero_volatility_is_discounted_forward_intrinsic():
    assert black_scholes_price(S, K, T, R, 0.0, "CALL") == pytest.approx(S - K * np.exp(-R * T))
    assert black_scholes_price(S, K, T, R, 0.0, "PUT") == pytest.approx(0.0)


def test_extreme_volatility_bounded():
    c = black_scholes_price(S, K, T, R, 4.0, "CALL")
    assert 0 < c < S


@pytest.mark.parametrize("bad", [dict(s=-1.0), dict(k=0.0), dict(t=-0.5), dict(v=-0.1)])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        black_scholes_price(bad.get("s", S), bad.get("k", K), bad.get("t", T), R, bad.get("v", SIG), "CALL")


def test_invalid_option_type():
    with pytest.raises(ValueError):
        black_scholes_price(S, K, T, R, SIG, "STRADDLE")


def test_implied_vol_roundtrip():
    px = black_scholes_price(S, 110.0, 0.5, R, 0.33, "PUT")
    assert implied_volatility(px, S, 110.0, 0.5, R, "PUT") == pytest.approx(0.33, abs=1e-8)


def test_position_greeks_scaling():
    raw = all_greeks(S, K, T, R, SIG, "CALL")
    pos = position_greeks(-10, 100, S, raw)
    assert pos["delta"] == pytest.approx(-1000 * raw["delta"])
    assert pos["vega"] == pytest.approx(-1000 * raw["vega"] / 100)
    assert pos["theta"] == pytest.approx(-1000 * raw["theta"] / 365)
