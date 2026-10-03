import numpy as np
import pandas as pd
import pytest

from analytics.covariance import covariance_matrix, ensure_psd
from analytics.correlation import cov_to_corr, correlation_summary
from analytics.ewma import (
    ewma_covariance,
    ewma_covariance_recursive,
    ewma_latest_variance,
    ewma_variance_series,
    ewma_volatility_series,
    ewma_weights,
    validate_lambda,
)


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(20)
    idx = pd.bdate_range("2025-01-01", periods=300)
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.1], [-0.2, 0.1, 1.0]])
    x = rng.multivariate_normal(np.zeros(3), corr * 0.0001, size=300)
    return pd.DataFrame(x, index=idx, columns=["A", "B", "C"])


def test_variance_recursion_matches_loop(returns):
    r = returns["A"]
    lam = 0.94
    got = ewma_variance_series(r, lam)
    expected = np.empty(len(r))
    expected[0] = np.var(r.to_numpy(), ddof=1)
    for t in range(1, len(r)):
        expected[t] = lam * expected[t - 1] + (1 - lam) * r.iloc[t - 1] ** 2
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-12)


def test_variance_uses_only_past_returns(returns):
    r = returns["A"].copy()
    base = ewma_variance_series(r, 0.94)
    r.iloc[-1] = 1.0  # a huge final return must not change the final forecast
    bumped = ewma_variance_series(r, 0.94, initial_variance=float(base.iloc[0]))
    assert bumped.iloc[-1] == pytest.approx(base.iloc[-1])


def test_covariance_closed_form_matches_recursion(returns):
    for lam in (0.90, 0.94, 0.97, 0.99):
        np.testing.assert_allclose(ewma_covariance(returns, lam), ewma_covariance_recursive(returns, lam), rtol=1e-10)


def test_covariance_recursion_single_step():
    r = pd.DataFrame({"x": [0.01, -0.02], "y": [0.03, 0.01]})
    lam = 0.94
    s0 = np.cov(r.to_numpy(), rowvar=False, ddof=1)
    s1 = lam * s0 + (1 - lam) * np.outer(r.iloc[0], r.iloc[0])
    s2 = lam * s1 + (1 - lam) * np.outer(r.iloc[1], r.iloc[1])
    np.testing.assert_allclose(ewma_covariance(r, lam).to_numpy(), s2)


def test_covariance_diagonal_matches_variance(returns):
    cov = ewma_covariance(returns, 0.94)
    assert cov.loc["A", "A"] == pytest.approx(ewma_latest_variance(returns["A"], 0.94))
    np.testing.assert_allclose(cov.to_numpy(), cov.to_numpy().T)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, float("nan")])
def test_lambda_validation(bad):
    with pytest.raises(ValueError):
        validate_lambda(bad)
    with pytest.raises(ValueError):
        ewma_weights(10, bad)


def test_ewma_reacts_faster_than_sample():
    rng = np.random.default_rng(21)
    r = pd.Series(np.concatenate([rng.normal(0, 0.01, 250), rng.normal(0, 0.03, 30)]))
    ewma_vol = float(ewma_volatility_series(r, 0.94).iloc[-1])
    sample_vol = float(r.iloc[-250:].std())
    assert abs(ewma_vol - 0.03) < abs(sample_vol - 0.03)


def test_psd_enforcement():
    bad = pd.DataFrame([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    fixed = ensure_psd(bad)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() >= -1e-12


def test_correlation_helpers(returns):
    cov = covariance_matrix(returns, "SAMPLE")
    corr = cov_to_corr(cov)
    assert np.allclose(np.diag(corr), 1.0)
    summary = correlation_summary(corr)
    assert summary.highest_positive[:2] == ("A", "B")
    assert summary.highest_negative[:2] == ("A", "C")
    # zero-variance asset -> zero correlation instead of NaN
    cov2 = cov.copy()
    cov2.loc["C", :] = 0.0
    cov2.loc[:, "C"] = 0.0
    assert np.isfinite(cov_to_corr(cov2).to_numpy()).all()
