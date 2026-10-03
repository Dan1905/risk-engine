"""Factor frame, filtered historical simulation and Monte Carlo."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis, norm

from analytics import risk_factors as rf
from analytics.filtered_simulation import ewma_vol_forecasts, filter_moves, scaling_ratios
from analytics.monte_carlo import covariance_root, simulate_factor_moves, standard_draws
from analytics.revaluation import PortfolioPricer, ScenarioSet
from analytics.var import historical_var
from models.portfolio import Portfolio
from tests.conftest import VAL_DATE, make_positions


def _frame(values: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=pd.bdate_range("2026-01-01", periods=n))


# ------------------------------------------------------------------ factor frame


def test_factor_naming():
    assert rf.factor(rf.CURVE, "USD_TSY", "10Y") == "curve|USD_TSY|10Y"
    assert rf.kind_of("eq|AAPL") == "eq"
    assert rf.is_return("fx|EUR") and not rf.is_return("vol|AAPL")
    assert rf.return_mask(["eq|A", "curve|X|1Y", "fx|EUR"]).tolist() == [True, False, True]


def test_aggregate_horizon_compounds_returns_and_sums_changes():
    f = _frame({"eq|A": [0.10, -0.05, 0.02, 0.03], "curve|USD_TSY|10Y": [0.001, 0.002, -0.001, 0.0005]})
    agg = rf.aggregate_horizon(f, 2)
    assert len(agg) == 3 and agg.index[0] == f.index[1]
    assert agg["eq|A"].iloc[0] == pytest.approx(1.10 * 0.95 - 1)
    assert agg["eq|A"].iloc[-1] == pytest.approx(1.02 * 1.03 - 1)
    assert agg["curve|USD_TSY|10Y"].tolist() == pytest.approx([0.003, 0.001, -0.0005])
    pd.testing.assert_frame_equal(rf.aggregate_horizon(f, 1), f)
    full = rf.aggregate_horizon(f, 4)
    assert full["eq|A"].iloc[0] == pytest.approx(1.10 * 0.95 * 1.02 * 1.03 - 1)
    with pytest.raises(ValueError):
        rf.aggregate_horizon(f, 5)


def test_clip_returns_only_touches_return_factors():
    f = _frame({"eq|A": [-1.5, 0.1], "vol|A": [-1.5, 0.1]})
    out = rf.clip_returns(f)
    assert out["eq|A"].iloc[0] == rf.RETURN_FLOOR and out["vol|A"].iloc[0] == -1.5


def test_scenario_set_round_trip():
    f = _frame({"eq|AAPL": [0.01, -0.02], "fx|EUR": [0.0, 0.01], "curve|USD_TSY|2Y": [0.0001, 0.0], "curve|USD_TSY|10Y": [0.0, 0.0002]})
    sc = ScenarioSet.from_frame(f)
    assert sc.n == 2 and sc.equity("AAPL").tolist() == [0.01, -0.02]
    assert sc.equity("MISSING").tolist() == [0.0, 0.0]  # absent factor = unchanged
    assert sc.fx("USD").tolist() == [0.0, 0.0]
    np.testing.assert_allclose(sc.curve("USD_TSY", ("2Y", "5Y", "10Y")), [[0.0001, 0, 0], [0, 0, 0.0002]])
    assert sc.curve(None, ("2Y",)).shape == (2, 1)
    pd.testing.assert_frame_equal(sc.to_frame(), f)


# ------------------------------------------------------- filtered historical simulation


def test_vol_forecast_recursion_and_no_look_ahead():
    rng = np.random.default_rng(50)
    f = _frame({"eq|A": rng.normal(0, 0.01, 120).tolist(), "vol|A": rng.normal(0, 0.005, 120).tolist()})
    lam = 0.94
    vols = ewma_vol_forecasts(f, lam, seed_obs=20)
    assert vols.shape == (121, 2)
    x = f["eq|A"].to_numpy()
    expected = np.empty(121)
    expected[0] = np.mean(x[:20] ** 2)
    for t in range(1, 121):
        expected[t] = lam * expected[t - 1] + (1 - lam) * x[t - 1] ** 2
    np.testing.assert_allclose(vols[:, 0] ** 2, expected, rtol=1e-12)
    # Changing the last observation only changes the forecast for the day after it.
    g = f.copy()
    g.iloc[-1] = 0.5
    vols2 = ewma_vol_forecasts(g, lam, seed_obs=20)
    np.testing.assert_allclose(vols2[:-1], vols[:-1])
    assert vols2[-1, 0] > vols[-1, 0]


def test_filter_scales_calm_history_up_to_current_volatility():
    rng = np.random.default_rng(51)
    calm, stressed = rng.normal(0, 0.005, 200), rng.normal(0, 0.02, 50)
    f = _frame({"eq|A": np.concatenate([calm, stressed]).tolist()})
    out = filter_moves(f, lookback=250, lam=0.94, bounds=(0.2, 5.0))
    ratio = (out["eq|A"] / f["eq|A"]).to_numpy()
    assert ratio[30:150].mean() > 2.5  # calm-period moves scaled up towards the stressed regime
    assert out["eq|A"].std() > f["eq|A"].std()
    # Filtered scenarios have roughly today's volatility throughout.
    assert out["eq|A"].iloc[30:150].std() == pytest.approx(0.02, rel=0.35)


def test_filter_bounds_and_degenerate_factors():
    f = _frame({"eq|A": ([0.001] * 100 + [0.2] * 5), "curve|USD_TSY|10Y": [0.0] * 105})
    out = filter_moves(f, lookback=100, bounds=(1 / 3, 3.0))
    ratio = out["eq|A"].iloc[:50] / f["eq|A"].iloc[-100:].iloc[:50]
    assert ratio.max() <= 3.0 + 1e-12  # capped
    assert (out["curve|USD_TSY|10Y"] == 0).all()  # zero-volatility factor is left alone
    vols = ewma_vol_forecasts(f)
    assert np.all(scaling_ratios(vols, 0, 105)[:, 1] == 1.0)
    with pytest.raises(ValueError):
        filter_moves(f, lookback=500)


def test_filter_is_identity_when_volatility_is_constant():
    f = _frame({"eq|A": [0.01, -0.01] * 60})
    out = filter_moves(f, lookback=60)
    np.testing.assert_allclose(out["eq|A"], f["eq|A"].iloc[-60:], rtol=1e-12)


# ------------------------------------------------------------------- Monte Carlo

COV = pd.DataFrame(
    [[0.0004, 0.00012, 0.0], [0.00012, 0.0001, 0.0], [0.0, 0.0, 0.000001]],
    index=["eq|A", "fx|EUR", "curve|USD_TSY|10Y"], columns=["eq|A", "fx|EUR", "curve|USD_TSY|10Y"],
)


def test_covariance_root_handles_singular_matrices():
    a = covariance_root(COV.to_numpy())
    np.testing.assert_allclose(a @ a.T, COV.to_numpy(), atol=1e-15)
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    r = covariance_root(singular)
    np.testing.assert_allclose(r @ r.T, singular, atol=1e-12)
    with pytest.raises(ValueError):
        covariance_root(np.array([[np.nan]]))


@pytest.mark.parametrize("dist", ["NORMAL", "STUDENT_T"])
def test_simulated_covariance_matches_target(dist):
    sim = simulate_factor_moves(COV, n_paths=400_000, seed=1, distribution=dist, dof=6)
    np.testing.assert_allclose(sim.cov().to_numpy(), COV.to_numpy(), rtol=0.05, atol=5e-7)  # atol covers noise on the zero entries
    assert abs(sim.mean()).max() < 1e-12  # antithetic pairs cancel exactly


def test_student_t_has_fatter_tails_than_normal():
    n = simulate_factor_moves(COV, 200_000, seed=2, distribution="NORMAL")["eq|A"]
    t = simulate_factor_moves(COV, 200_000, seed=2, distribution="STUDENT_T", dof=5)["eq|A"]
    assert kurtosis(t) > kurtosis(n) + 1.0
    assert np.quantile(t, 0.001) < np.quantile(n, 0.001)


def test_simulation_is_deterministic_and_seed_dependent():
    a = simulate_factor_moves(COV, 1000, seed=7)
    b = simulate_factor_moves(COV, 1000, seed=7)
    c = simulate_factor_moves(COV, 1000, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_horizon_scales_covariance_and_inputs_are_validated():
    one = simulate_factor_moves(COV, 100_000, seed=3, distribution="NORMAL", horizon=1)
    ten = simulate_factor_moves(COV, 100_000, seed=3, distribution="NORMAL", horizon=10)
    assert ten["eq|A"].std() == pytest.approx(one["eq|A"].std() * np.sqrt(10), rel=1e-9)
    with pytest.raises(ValueError):
        standard_draws(100, 3, 1, "STUDENT_T", dof=2.0)
    with pytest.raises(ValueError):
        standard_draws(100, 3, 1, "LOGNORMAL")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        simulate_factor_moves(COV, 100, horizon=0)


def test_returns_are_floored():
    wild = pd.DataFrame([[4.0]], index=["eq|A"], columns=["eq|A"])
    sim = simulate_factor_moves(wild, 10_000, seed=4)
    assert sim["eq|A"].min() >= rf.RETURN_FLOOR


def test_monte_carlo_var_matches_parametric_for_a_linear_position(base_state):
    equity = Portfolio(portfolio_id="E", name="E", valuation_date=VAL_DATE, positions=[p for p in make_positions() if p.position_id == "EQ1"])
    pricer = PortfolioPricer(equity, base_state)
    cov = pd.DataFrame([[0.0004]], index=["eq|AAPL"], columns=["eq|AAPL"])
    sim = simulate_factor_moves(cov, 400_000, seed=5, distribution="NORMAL")
    pnl = pricer.scenario_pnl(base_state, ScenarioSet.from_frame(sim)).sum(axis=1)
    analytic = norm.ppf(0.99) * 0.02 * 200_000  # z * sigma * market value
    assert historical_var(pnl, 0.99) == pytest.approx(analytic, rel=0.02)


def test_monte_carlo_captures_option_convexity(base_state):
    """Long option: full revaluation loses less than the delta-linear approximation in the tail."""
    opt = Portfolio(portfolio_id="O", name="O", valuation_date=VAL_DATE, positions=[p for p in make_positions() if p.position_id == "OP1"])
    pricer = PortfolioPricer(opt, base_state)
    cov = pd.DataFrame([[0.0009]], index=["eq|AAPL"], columns=["eq|AAPL"])
    sim = simulate_factor_moves(cov, 200_000, seed=6, distribution="NORMAL")
    pnl = pricer.scenario_pnl(base_state, ScenarioSet.from_frame(sim)).sum(axis=1)
    delta_usd = float(pricer.option_analytics()["delta_usd"].iloc[0])
    linear_var = norm.ppf(0.99) * 0.03 * delta_usd
    assert historical_var(pnl, 0.99) < linear_var


# ------------------------------------------------------ implied-vol skew / term factors
from analytics.volatility_surface import (  # noqa: E402
    GridVolSurface,
    skew_loading,
    surface_factors,
    term_loading,
)


def test_factor_loadings():
    assert float(skew_loading(0.9)) == pytest.approx(0.5) and float(skew_loading(1.1)) == pytest.approx(-0.5)
    assert float(skew_loading(1.0)) == 0.0
    assert float(skew_loading(0.5)) == float(skew_loading(0.8))  # clipped to the surface grid
    assert float(term_loading(0.25)) == 0.0
    assert float(term_loading(1.0)) - float(term_loading(1 / 12)) == pytest.approx(1.0)
    assert float(term_loading(3.0)) == float(term_loading(1.0))


def _grid(level=0.25, skew=0.03, term=0.02) -> GridVolSurface:
    mats, mny = np.array([1 / 12, 0.25, 0.5, 1.0]), np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    vols = level + skew * skew_loading(mny)[None, :] + term * term_loading(mats)[:, None]
    return GridVolSurface(mats, mny, vols)


def test_surface_factors_recover_level_skew_term():
    assert surface_factors(_grid(0.25, 0.03, 0.02)) == pytest.approx((0.25, 0.03, 0.02))
    a, b = _grid(0.25, 0.03, 0.02), _grid(0.27, 0.045, 0.012)
    d_level, d_skew, d_term = (y - x for x, y in zip(surface_factors(a), surface_factors(b)))
    for m, tau in ((1.06, 0.47), (0.93, 0.15), (1.15, 0.8)):  # arbitrary options inside the grid
        actual = float(b.vol(m, tau) - a.vol(m, tau))
        modelled = d_level + d_skew * float(skew_loading(m)) + d_term * float(term_loading(tau))
        assert modelled == pytest.approx(actual, abs=1e-12)
        assert abs(actual - d_level) > 1e-4  # a level-only model would miss the shape change


def test_select_vol_factors():
    f = _frame({"eq|A": [0.01], "vol|A": [0.01], "volskew|A": [0.002], "volterm|A": [0.003]})
    assert list(rf.select_vol_factors(f, "LEVEL").columns) == ["eq|A", "vol|A"]
    assert list(rf.select_vol_factors(f, "LEVEL_SKEW_TERM").columns) == list(f.columns)
    with pytest.raises(ValueError):
        rf.select_vol_factors(f, "SURFACE")


def test_pricer_applies_skew_and_term_shifts(base_state):
    opt = Portfolio(portfolio_id="O", name="O", valuation_date=VAL_DATE, positions=[p for p in make_positions() if p.position_id == "OP1"])
    pricer = PortfolioPricer(opt, base_state)
    row = pricer.option_analytics().iloc[0]
    m, tau, vega_per_point = row["strike"] / row["spot"], row["time_to_expiry"], row["vega"]
    for factor_name, loading in (("volskew|AAPL", float(skew_loading(m))), ("volterm|AAPL", float(term_loading(tau)))):
        shock = 0.01
        pnl = float(pricer.scenario_pnl(base_state, ScenarioSet.from_frame(_frame({factor_name: [shock]}))).iloc[0, 0])
        assert pnl == pytest.approx(vega_per_point * shock * 100 * loading, rel=0.02)
    # an OTM call (m > 1) loses when the put-skew steepens, and gains when the term structure steepens (tau > 3M)
    assert float(skew_loading(m)) < 0 < float(term_loading(tau))
