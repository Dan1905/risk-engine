import numpy as np
import pytest
from scipy.stats import norm

from analytics.ewma import ewma_weights
from analytics.var import (
    InsufficientDataError,
    component_var,
    historical_var,
    historical_var_rows,
    marginal_var,
    parametric_var,
    parametric_var_rows,
    pnl_summary,
    portfolio_volatility,
    scale_to_horizon,
    weighted_historical_var,
    weighted_historical_var_rows,
    weighted_quantile,
)

UNIFORM = np.arange(-50.0, 50.0)  # 100 equally likely outcomes: -50 ... 49


class TestHistoricalVaR:
    def test_known_distribution_99(self):
        # k = ceil(100 * 1%) = 1 -> worst outcome
        assert historical_var(UNIFORM, 0.99) == pytest.approx(50.0)

    def test_known_distribution_95(self):
        # k = ceil(100 * 5%) = 5 -> 5th worst outcome (-46)
        assert historical_var(UNIFORM, 0.95) == pytest.approx(46.0)

    def test_250_observations_uses_third_worst(self):
        pnl = np.arange(250.0)
        # ceil(250 * 0.01) = 3 -> 3rd smallest = 2
        assert historical_var(pnl, 0.99) == pytest.approx(-2.0)

    def test_normal_sample_converges(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(400_000)
        assert historical_var(x, 0.99) == pytest.approx(norm.ppf(0.99), abs=0.02)
        assert historical_var(x, 0.95) == pytest.approx(norm.ppf(0.95), abs=0.01)

    def test_order_invariant(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(250)
        assert historical_var(x, 0.99) == pytest.approx(historical_var(rng.permutation(x), 0.99))

    def test_99_exceeds_95(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(500)
        assert historical_var(x, 0.99) > historical_var(x, 0.95)

    def test_nan_values_ignored(self):
        x = np.append(UNIFORM, [np.nan, np.inf])
        assert historical_var(x, 0.99) == pytest.approx(50.0)

    def test_insufficient_observations(self):
        with pytest.raises(InsufficientDataError):
            historical_var(np.arange(50.0), 0.99)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            historical_var(UNIFORM, 1.2)

    def test_summary(self):
        s = pnl_summary(UNIFORM)
        assert s["worst_loss"] == 50.0 and s["best_gain"] == 49.0 and s["mean_pnl"] == pytest.approx(-0.5)


class TestWeightedVaR:
    def test_weights_sum_to_one_and_decay(self):
        w = ewma_weights(250, 0.94)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(np.diff(w) > 0)  # oldest -> newest increasing
        assert w[-1] / w[-2] == pytest.approx(1 / 0.94)

    def test_equal_weights_match_standard_quantile(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(250)
        q = weighted_quantile(x, np.ones_like(x), 0.01)
        assert -q == pytest.approx(historical_var(x, 0.99))

    def test_reacts_to_recent_volatility(self):
        rng = np.random.default_rng(4)
        calm = np.clip(rng.normal(0, 1, 240), -3, 3)
        stressed = np.array([-6.0, 3.0, -5.0, 4.0, -7.0, 2.0, -4.5, 5.0, -5.5, 3.0])
        recent = np.concatenate([calm, stressed])  # stress in the most recent 10 days
        old = np.concatenate([stressed, calm])  # same stress 240+ days ago
        # Equal weighting cannot tell the two histories apart (3rd worst = -5.5)...
        assert historical_var(recent, 0.99) == pytest.approx(5.5)
        assert historical_var(old, 0.99) == pytest.approx(5.5)
        # ...exponential weighting reacts to where the stress sits in time.
        assert weighted_historical_var(recent, 0.99, 0.94) == pytest.approx(7.0)
        assert weighted_historical_var(old, 0.99, 0.94) <= 3.0

    def test_higher_lambda_approaches_standard(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(250)
        diffs = [abs(weighted_historical_var(x, 0.99, lam) - historical_var(x, 0.99)) for lam in (0.90, 0.999999)]
        assert diffs[1] <= diffs[0] + 1e-12

    def test_rows_match_scalar(self):
        rng = np.random.default_rng(6)
        m = rng.standard_normal((5, 250))
        np.testing.assert_allclose(historical_var_rows(m, 0.99), [historical_var(r, 0.99) for r in m])
        np.testing.assert_allclose(
            weighted_historical_var_rows(m, 0.99, 0.94), [weighted_historical_var(r, 0.99, 0.94) for r in m]
        )


class TestParametricVaR:
    def test_single_asset(self):
        var = parametric_var([1.0], [[0.0001]], 0.99, portfolio_value=1_000_000)
        assert var == pytest.approx(norm.ppf(0.99) * 0.01 * 1_000_000)
        assert norm.ppf(0.99) == pytest.approx(2.326, abs=1e-3)

    def test_holding_period_scaling(self):
        v1 = parametric_var([1.0], [[0.0001]], 0.99, 1.0, 1)
        v10 = parametric_var([1.0], [[0.0001]], 0.99, 1.0, 10)
        assert v10 == pytest.approx(v1 * np.sqrt(10))
        assert scale_to_horizon(2.0, 4) == pytest.approx(4.0)

    def test_diversification(self):
        cov = np.array([[0.0004, 0.0], [0.0, 0.0004]])
        w = np.array([0.5, 0.5])
        assert portfolio_volatility(w, cov) == pytest.approx(np.sqrt(0.5 * 0.0004))

    def test_component_var_sums_to_total(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((5, 5))
        cov = a @ a.T / 100
        w = np.array([0.3, 0.2, -0.1, 0.4, 0.2])
        total = parametric_var(w, cov, 0.99, 5_000_000, 1)
        comp = component_var(w, cov, 0.99, 5_000_000, 1)
        assert comp.sum() == pytest.approx(total)
        # Marginal VaR is the gradient: check by finite difference.
        eps = 1e-6
        mv = marginal_var(w, cov, 0.99)
        for i in range(5):
            bump = w.copy()
            bump[i] += eps
            fd = (parametric_var(bump, cov, 0.99) - parametric_var(w, cov, 0.99)) / eps
            assert mv[i] == pytest.approx(fd, rel=1e-4)

    def test_zero_variance_asset(self):
        cov = np.diag([0.0001, 0.0])
        assert np.all(np.isfinite(marginal_var([0.5, 0.5], cov)))
        assert parametric_var_rows(np.array([1.0, 2.0]), 0.99) == pytest.approx(norm.ppf(0.99) * np.array([1.0, 2.0]))
