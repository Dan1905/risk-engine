import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from analytics.backtesting import (
    NO_EVIDENCE_MSG,
    REJECT_MSG,
    count_exceptions,
    exceptions_by_month,
    kupiec_lr_statistic,
    kupiec_pof_test,
    rolling_exception_rate,
    run_backtest,
)


def _series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_exception_counting():
    pnl = _series([-5.0, -10.0, -10.0001, 3.0, -20.0])
    var = _series([10.0, 10.0, 10.0, 10.0, 10.0])
    exc = count_exceptions(pnl, var)
    # loss equal to VaR is not an exception; strictly greater is
    assert exc.tolist() == [False, False, True, False, True]
    assert exc.mean() == pytest.approx(0.4)


def test_exception_alignment_drops_missing():
    pnl = _series([-5.0, -50.0, 1.0])
    var = _series([10.0, np.nan, 10.0])
    assert len(count_exceptions(pnl, var)) == 2


def test_kupiec_statistic_known_value():
    # N = 250, x = 5, p = 1%
    n, x, p = 250, 5, 0.01
    expected = -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - x / n) - x * np.log(x / n)
    )
    assert kupiec_lr_statistic(n, x, p) == pytest.approx(expected)
    assert expected == pytest.approx(1.9568, abs=1e-3)


def test_kupiec_p_value():
    res = kupiec_pof_test(250, 5, 0.01, 0.05)
    assert res.p_value == pytest.approx(chi2.sf(res.lr_statistic, 1))
    assert res.p_value == pytest.approx(0.1619, abs=1e-3)
    assert not res.reject_null
    assert res.message == NO_EVIDENCE_MSG
    assert res.exception_rate == pytest.approx(0.02)
    assert res.expected_exceptions == pytest.approx(2.5)


def test_kupiec_zero_exceptions():
    res = kupiec_pof_test(250, 0, 0.01)
    assert res.lr_statistic == pytest.approx(-2 * 250 * np.log(0.99))
    assert res.reject_null  # 0 exceptions in 250 days is unlikely under p = 1%


def test_kupiec_too_many_exceptions():
    res = kupiec_pof_test(250, 12, 0.01)
    assert res.reject_null
    assert res.message == REJECT_MSG


def test_kupiec_all_exceptions_and_expected():
    assert np.isfinite(kupiec_lr_statistic(10, 10, 0.01))
    assert kupiec_lr_statistic(1000, 10, 0.01) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("args", [(0, 0, 0.01), (10, 11, 0.01), (10, 1, 0.0)])
def test_kupiec_invalid(args):
    with pytest.raises(ValueError):
        kupiec_lr_statistic(*args)


def test_correct_model_passes():
    rng = np.random.default_rng(30)
    pnl = _series(rng.standard_normal(2000))
    var = _series(np.full(2000, norm.ppf(0.99)))
    bt = run_backtest(pnl, var, 0.99, 250, "PARAMETRIC")
    assert bt.kupiec.exception_rate == pytest.approx(0.01, abs=0.006)
    assert set(bt.exceptions_table.columns) == {"Actual P&L", "Predicted VaR", "Excess Loss"}
    assert (bt.exceptions_table["Excess Loss"] > 0).all()


def test_understated_var_rejected():
    rng = np.random.default_rng(31)
    pnl = _series(rng.standard_normal(1000) * 2.0)
    var = _series(np.full(1000, norm.ppf(0.99)))
    assert run_backtest(pnl, var, 0.99, 250, "HISTORICAL").kupiec.reject_null


def test_rolling_and_monthly():
    exc = pd.Series([False] * 40 + [True] * 2 + [False] * 18, index=pd.bdate_range("2025-01-01", periods=60))
    roll = rolling_exception_rate(exc, 20)
    assert roll.iloc[-1] == pytest.approx(2 / 20)
    monthly = exceptions_by_month(exc)
    assert monthly["exceptions"].sum() == 2
    assert monthly["observations"].sum() == 60


# ----------------------------------------------------------------- Christoffersen / Basel
from analytics.backtesting import (  # noqa: E402
    CLUSTERED_MSG,
    INDEPENDENT_MSG,
    basel_traffic_light,
    christoffersen_test,
    transition_counts,
)


def test_transition_counts():
    e = np.array([0, 1, 1, 0, 0, 1, 0], dtype=bool)
    assert transition_counts(e) == (1, 2, 2, 1)  # 00:1, 01:2, 10:2, 11:1
    assert sum(transition_counts(e)) == len(e) - 1


def test_christoffersen_statistic_by_hand():
    e = np.zeros(100, dtype=bool)
    e[[10, 11, 50, 51, 52]] = True  # two clusters
    n00, n01, n10, n11 = transition_counts(e)
    assert (n01, n10, n11) == (2, 2, 3)
    pi01, pi11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 99
    expected = -2 * (
        (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
        - n00 * np.log(1 - pi01) - n01 * np.log(pi01) - n10 * np.log(1 - pi11) - n11 * np.log(pi11)
    )
    res = christoffersen_test(e, 0.05)
    assert res.lr_independence == pytest.approx(expected)
    assert res.p_value_independence == pytest.approx(chi2.sf(expected, 1))
    assert res.lr_conditional_coverage == pytest.approx(kupiec_lr_statistic(100, 5, 0.05) + expected)
    assert res.p_value_conditional_coverage == pytest.approx(chi2.sf(res.lr_conditional_coverage, 2))
    assert res.reject_independence and res.independence_message == CLUSTERED_MSG


def test_christoffersen_same_count_clustered_vs_spread():
    clustered = np.zeros(500, dtype=bool)
    clustered[100:105] = True
    spread = np.zeros(500, dtype=bool)
    spread[[50, 150, 250, 350, 450]] = True
    c, s = christoffersen_test(clustered, 0.01), christoffersen_test(spread, 0.01)
    # Kupiec cannot tell these apart (same count)...
    assert kupiec_pof_test(500, 5, 0.01).p_value == pytest.approx(kupiec_pof_test(500, int(spread.sum()), 0.01).p_value)
    # ...Christoffersen can.
    assert c.reject_independence and not s.reject_independence
    assert s.independence_message == INDEPENDENT_MSG
    assert c.pi_11 == pytest.approx(0.8) and s.pi_11 == 0.0


def test_christoffersen_edge_cases():
    none = christoffersen_test(np.zeros(250, dtype=bool), 0.01)
    assert none.lr_independence == 0.0 and not none.reject_independence
    assert none.lr_conditional_coverage == pytest.approx(kupiec_lr_statistic(250, 0, 0.01))
    assert np.isfinite(christoffersen_test(np.ones(10, dtype=bool), 0.01).lr_independence)
    assert christoffersen_test(np.array([True]), 0.01).lr_independence == 0.0


def test_independent_exceptions_usually_pass():
    rng = np.random.default_rng(40)
    rejections = sum(christoffersen_test(rng.random(1000) < 0.01, 0.01).reject_independence for _ in range(200))
    assert rejections / 200 < 0.12  # close to the 5% size


@pytest.mark.parametrize(
    "x,zone,plus", [(0, "GREEN", 0.0), (4, "GREEN", 0.0), (5, "YELLOW", 0.40), (6, "YELLOW", 0.50), (7, "YELLOW", 0.65),
                    (8, "YELLOW", 0.75), (9, "YELLOW", 0.85), (10, "RED", 1.0), (25, "RED", 1.0)]
)
def test_basel_zones_250_days(x, zone, plus):
    res = basel_traffic_light(250, x)
    assert res.zone == zone and res.plus_factor == plus
    assert res.multiplier == pytest.approx(3.0 + plus)
    assert (res.green_max, res.yellow_max) == (4, 9)


def test_basel_other_windows_and_validation():
    res = basel_traffic_light(500, 9)  # binomial rule generalises; no plus factor off the 250-day window
    assert res.zone in {"GREEN", "YELLOW"} and res.plus_factor is None and res.multiplier is None
    assert basel_traffic_light(100, 0).zone == "GREEN"
    with pytest.raises(ValueError):
        basel_traffic_light(0, 0)
    with pytest.raises(ValueError):
        basel_traffic_light(10, 11)


def test_run_backtest_includes_christoffersen():
    rng = np.random.default_rng(41)
    pnl = _series(rng.standard_normal(600))
    var = _series(np.full(600, norm.ppf(0.99)))
    bt = run_backtest(pnl, var, 0.99, 250, "PARAMETRIC")
    assert bt.christoffersen is not None
    assert bt.christoffersen.n00 + bt.christoffersen.n01 + bt.christoffersen.n10 + bt.christoffersen.n11 == 599
