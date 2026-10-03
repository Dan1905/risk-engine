import numpy as np
import pytest
from scipy.stats import norm

from analytics.expected_shortfall import historical_es, parametric_es, weighted_historical_es
from analytics.var import historical_var, weighted_historical_var

UNIFORM = np.arange(-50.0, 50.0)


def test_tail_selection_99():
    # VaR99 = 50, only the worst outcome is in the tail
    assert historical_es(UNIFORM, 0.99) == pytest.approx(50.0)


def test_tail_selection_95():
    # VaR95 = 46 -> tail losses {50, 49, 48, 47, 46}
    assert historical_es(UNIFORM, 0.95) == pytest.approx(48.0)


def test_es_at_least_var():
    rng = np.random.default_rng(10)
    for _ in range(20):
        x = rng.standard_t(4, 250)
        assert historical_es(x, 0.99) >= historical_var(x, 0.99) - 1e-12
        assert weighted_historical_es(x, 0.99, 0.94) >= weighted_historical_var(x, 0.99, 0.94) - 1e-12


def test_parametric_es_normal():
    z = norm.ppf(0.99)
    assert parametric_es(1.0, 0.99) == pytest.approx(norm.pdf(z) / 0.01)
    assert parametric_es(1.0, 0.99) == pytest.approx(2.665, abs=1e-3)
    assert parametric_es(1.0, 0.99, 100.0, 4) == pytest.approx(2 * 100 * norm.pdf(z) / 0.01)


def test_historical_es_converges_to_normal():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(500_000)
    assert historical_es(x, 0.99) == pytest.approx(parametric_es(1.0, 0.99), abs=0.03)


def test_weighted_es_emphasises_recent_tail():
    rng = np.random.default_rng(12)
    calm = rng.normal(0, 1, 245)
    shock = np.array([-8.0, -7.0, -9.0, -6.5, -7.5])
    recent = np.concatenate([calm, shock])
    old = np.concatenate([shock, calm])
    # Same observations, same equal-weighted ES...
    assert historical_es(recent, 0.99) == pytest.approx(historical_es(old, 0.99))
    # ...but weighting makes the recent-shock ES larger.
    assert weighted_historical_es(recent, 0.99, 0.94) > weighted_historical_es(old, 0.99, 0.94)


def test_weighted_es_tail_mean_is_weighted():
    pnl = np.array([-10.0] + [1.0] * 98 + [-5.0])  # oldest loss -10, newest loss -5
    lam = 0.94
    var = weighted_historical_var(pnl, 0.99, lam)
    es = weighted_historical_es(pnl, 0.99, lam)
    assert es >= var
    assert es <= 10.0
