"""VaR backtesting statistics and the Kupiec proportion-of-failures test.

An *exception* on day t occurs when the realised loss exceeds the VaR that was
forecast for day t::

    exception_t = 1{ -P&L_t > VaR_t }

VaR_t must be estimated using only information available before P&L_t is
realised (see ``services.backtest_service`` for the rolling estimation that
enforces this).

Kupiec POF test
---------------
With N observations, x exceptions and expected exception probability p::

    LR_POF = -2 ln[ (1-p)^{N-x} p^x / ((1 - x/N)^{N-x} (x/N)^x) ]  ~  chi^2(1)

Under the null hypothesis that the true exception probability equals p, LR
is asymptotically chi-square with one degree of freedom. The p-value is
``1 - F_chi2(LR)``; the null is rejected when p-value < significance level.
``0 * ln 0`` is taken as 0 so that x = 0 and x = N are handled.

Christoffersen independence and conditional coverage
----------------------------------------------------
Kupiec only checks *how many* exceptions occurred. A model can have the right
count and still be wrong if exceptions arrive in clusters (it fails to adapt
to volatility regimes). With n_ij the number of days in state j (1 =
exception) that follow a day in state i::

    pi_01 = n_01 / (n_00 + n_01)      P(exception | no exception yesterday)
    pi_11 = n_11 / (n_10 + n_11)      P(exception | exception yesterday)
    pi    = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    LR_ind = -2 ln[ (1-pi)^{n_00+n_10} pi^{n_01+n_11}
                    / ((1-pi_01)^{n_00} pi_01^{n_01} (1-pi_11)^{n_10} pi_11^{n_11}) ]  ~  chi^2(1)
    LR_cc  = LR_POF + LR_ind                                                          ~  chi^2(2)

``LR_ind`` tests independence (first-order Markov alternative); ``LR_cc`` is
the joint conditional-coverage test of correct frequency *and* independence.

Basel traffic light
-------------------
The Basel framework classifies a 99% VaR model by its exception count over
the most recent 250 trading days. Zones follow the cumulative binomial
probability P(X <= x) under the null: green below 95%, yellow from 95% up to
99.99%, red at or above 99.99%. For 250 days that is 0-4 / 5-9 / 10+
exceptions, with a capital-multiplier add-on ("plus factor") of 0.40, 0.50,
0.65, 0.75, 0.85 in the yellow zone and 1.00 in the red zone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

NO_EVIDENCE_MSG = (
    "No statistically significant evidence of incorrect exception frequency at the selected significance level."
)
REJECT_MSG = "Observed exception frequency differs significantly from the model's expected exception frequency."


@dataclass(frozen=True)
class KupiecResult:
    n_observations: int
    n_exceptions: int
    exception_rate: float
    expected_rate: float
    lr_statistic: float
    p_value: float
    significance_level: float
    reject_null: bool

    @property
    def message(self) -> str:
        return REJECT_MSG if self.reject_null else NO_EVIDENCE_MSG

    @property
    def expected_exceptions(self) -> float:
        return self.expected_rate * self.n_observations


def count_exceptions(actual_pnl: pd.Series, predicted_var: pd.Series) -> pd.Series:
    """Boolean series: True where the realised loss exceeds the forecast VaR.

    Both series are aligned on their index; dates missing from either side
    are dropped.
    """
    pnl, var = actual_pnl.align(predicted_var, join="inner")
    valid = pnl.notna() & var.notna()
    return ((-pnl) > var)[valid]


def kupiec_lr_statistic(n: int, x: int, p: float) -> float:
    """Kupiec likelihood-ratio statistic (see module docstring)."""
    if n <= 0:
        raise ValueError("Number of observations must be positive")
    if not 0 <= x <= n:
        raise ValueError("Exception count must be between 0 and N")
    if not 0 < p < 1:
        raise ValueError("Expected exception probability must be in (0, 1)")
    phat = x / n
    log_null = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    log_alt = xlogy(n - x, 1.0 - phat) + xlogy(x, phat)
    return float(max(-2.0 * (log_null - log_alt), 0.0))


def kupiec_pof_test(n: int, x: int, p: float, significance_level: float = 0.05) -> KupiecResult:
    """Run the Kupiec POF test.

    Args:
        n: number of backtest observations.
        x: number of exceptions.
        p: expected exception probability (1 - VaR confidence).
        significance_level: test size, e.g. 0.05.
    """
    lr = kupiec_lr_statistic(n, x, p)
    p_value = float(chi2.sf(lr, df=1))
    return KupiecResult(
        n_observations=n,
        n_exceptions=x,
        exception_rate=x / n,
        expected_rate=p,
        lr_statistic=lr,
        p_value=p_value,
        significance_level=significance_level,
        reject_null=p_value < significance_level,
    )


def rolling_exception_rate(exceptions: pd.Series, window: int = 60) -> pd.Series:
    """Trailing exception frequency over ``window`` observations."""
    return exceptions.astype(float).rolling(window, min_periods=max(5, window // 4)).mean()


def exceptions_by_month(exceptions: pd.Series) -> pd.DataFrame:
    """Exception count and observation count per calendar month."""
    s = exceptions.astype(int)
    idx = pd.DatetimeIndex(s.index)
    grouped = s.groupby(idx.to_period("M"))
    out = pd.DataFrame({"exceptions": grouped.sum(), "observations": grouped.count()})
    out.index = out.index.to_timestamp()
    out.index.name = "month"
    return out


INDEPENDENT_MSG = "No statistically significant evidence that exceptions cluster in time at the selected significance level."
CLUSTERED_MSG = "Exceptions are not independent: an exception is significantly more (or less) likely after a prior exception."
CC_PASS_MSG = "No statistically significant evidence against correct conditional coverage (frequency and independence jointly)."
CC_FAIL_MSG = "Conditional coverage is rejected: exception frequency and/or independence differs significantly from the model's assumptions."


@dataclass(frozen=True)
class ChristoffersenResult:
    """Independence and conditional-coverage tests (see module docstring)."""

    n00: int
    n01: int
    n10: int
    n11: int
    pi_01: float
    pi_11: float
    lr_independence: float
    p_value_independence: float
    lr_conditional_coverage: float
    p_value_conditional_coverage: float
    significance_level: float
    reject_independence: bool
    reject_conditional_coverage: bool

    @property
    def independence_message(self) -> str:
        return CLUSTERED_MSG if self.reject_independence else INDEPENDENT_MSG

    @property
    def conditional_coverage_message(self) -> str:
        return CC_FAIL_MSG if self.reject_conditional_coverage else CC_PASS_MSG


def transition_counts(exceptions: pd.Series | np.ndarray) -> tuple[int, int, int, int]:
    """(n00, n01, n10, n11): counts of consecutive-day state transitions."""
    e = np.asarray(exceptions, dtype=bool)
    if e.size < 2:
        return 0, 0, 0, 0
    prev, cur = e[:-1], e[1:]
    return (
        int(np.sum(~prev & ~cur)),
        int(np.sum(~prev & cur)),
        int(np.sum(prev & ~cur)),
        int(np.sum(prev & cur)),
    )


def christoffersen_test(
    exceptions: pd.Series | np.ndarray, expected_rate: float, significance_level: float = 0.05
) -> ChristoffersenResult:
    """Christoffersen (1998) independence and conditional-coverage tests.

    Args:
        exceptions: boolean exception indicator ordered by date.
        expected_rate: p = 1 - VaR confidence (used for the coverage component).
        significance_level: test size.

    Edge cases:
        With no exceptions, or no exception ever followed by a trading day,
        the independence statistic is 0 (the alternative is not identifiable);
        ``0 ln 0`` is treated as 0 throughout.
    """
    e = np.asarray(exceptions, dtype=bool)
    n00, n01, n10, n11 = transition_counts(e)
    total = n00 + n01 + n10 + n11
    pi_01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / total if total else 0.0
    log_null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    log_alt = xlogy(n00, 1.0 - pi_01) + xlogy(n01, pi_01) + xlogy(n10, 1.0 - pi_11) + xlogy(n11, pi_11)
    lr_ind = float(max(-2.0 * (log_null - log_alt), 0.0))
    lr_pof = kupiec_lr_statistic(e.size, int(e.sum()), expected_rate) if e.size else 0.0
    lr_cc = lr_pof + lr_ind
    p_ind, p_cc = float(chi2.sf(lr_ind, df=1)), float(chi2.sf(lr_cc, df=2))
    return ChristoffersenResult(
        n00=n00, n01=n01, n10=n10, n11=n11, pi_01=pi_01, pi_11=pi_11,
        lr_independence=lr_ind, p_value_independence=p_ind,
        lr_conditional_coverage=lr_cc, p_value_conditional_coverage=p_cc,
        significance_level=significance_level,
        reject_independence=p_ind < significance_level,
        reject_conditional_coverage=p_cc < significance_level,
    )


BASEL_WINDOW = 250
BASEL_CONFIDENCE = 0.99
BASEL_PLUS_FACTORS: dict[int, float] = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}
BASEL_GREEN_CUTOFF = 0.95
BASEL_RED_CUTOFF = 0.9999


@dataclass(frozen=True)
class BaselResult:
    """Basel traffic-light classification of a 99% VaR model."""

    n_observations: int
    n_exceptions: int
    zone: str  # GREEN | YELLOW | RED
    cumulative_probability: float  # P(X <= x) under the null
    plus_factor: float | None  # None when the window is not the regulatory 250 days
    green_max: int  # largest exception count still in the green zone
    yellow_max: int  # largest exception count still in the yellow zone

    @property
    def multiplier(self) -> float | None:
        return None if self.plus_factor is None else 3.0 + self.plus_factor

    @property
    def message(self) -> str:
        return {
            "GREEN": "Exception count is consistent with a 99% model; no capital add-on.",
            "YELLOW": "Exception count is elevated; it could arise from either an accurate or an inaccurate model. A capital add-on applies.",
            "RED": "Exception count is very unlikely under a correct 99% model; the model is presumed inaccurate.",
        }[self.zone]


def basel_traffic_light(n_observations: int, n_exceptions: int, expected_rate: float = 1.0 - BASEL_CONFIDENCE) -> BaselResult:
    """Classify an exception count into the Basel green / yellow / red zones.

    Zones use the cumulative binomial probability so the rule generalises to
    windows other than 250 days; the plus factor is only defined for the
    regulatory 250-day window at 99%.
    """
    if n_observations <= 0:
        raise ValueError("Number of observations must be positive")
    if not 0 <= n_exceptions <= n_observations:
        raise ValueError("Exception count must be between 0 and N")
    cdf = binom.cdf(np.arange(n_observations + 1), n_observations, expected_rate)
    green_max = int(np.searchsorted(cdf, BASEL_GREEN_CUTOFF, side="left")) - 1
    yellow_max = int(np.searchsorted(cdf, BASEL_RED_CUTOFF, side="left")) - 1
    zone = "GREEN" if n_exceptions <= green_max else ("YELLOW" if n_exceptions <= yellow_max else "RED")
    plus: float | None = None
    if n_observations == BASEL_WINDOW and abs(expected_rate - 0.01) < 1e-12:
        plus = 0.0 if zone == "GREEN" else (1.0 if zone == "RED" else BASEL_PLUS_FACTORS[n_exceptions])
    return BaselResult(
        n_observations=n_observations, n_exceptions=n_exceptions, zone=zone,
        cumulative_probability=float(cdf[n_exceptions]), plus_factor=plus, green_max=green_max, yellow_max=yellow_max,
    )


@dataclass
class BacktestResult:
    """Full output of a VaR backtest."""

    method: str
    confidence: float
    lookback: int
    frame: pd.DataFrame  # date-indexed: actual_pnl, var, exception, excess_loss
    kupiec: KupiecResult
    christoffersen: ChristoffersenResult | None = None

    @property
    def exceptions_table(self) -> pd.DataFrame:
        ex = self.frame[self.frame["exception"]]
        return ex[["actual_pnl", "var", "excess_loss"]].rename(
            columns={"actual_pnl": "Actual P&L", "var": "Predicted VaR", "excess_loss": "Excess Loss"}
        )


def run_backtest(
    actual_pnl: pd.Series,
    predicted_var: pd.Series,
    confidence: float,
    lookback: int,
    method: str,
    significance_level: float = 0.05,
) -> BacktestResult:
    """Combine realised P&L and forecast VaR into a backtest with Kupiec statistics."""
    exc = count_exceptions(actual_pnl, predicted_var)
    if exc.empty:
        raise ValueError("Insufficient historical observations for the selected backtest window.")
    pnl, var = actual_pnl.reindex(exc.index), predicted_var.reindex(exc.index)
    frame = pd.DataFrame(
        {
            "actual_pnl": pnl,
            "var": var,
            "exception": exc,
            "excess_loss": np.where(exc, -pnl - var, 0.0),
        }
    )
    kup = kupiec_pof_test(len(exc), int(exc.sum()), 1.0 - confidence, significance_level)
    chris = christoffersen_test(exc, 1.0 - confidence, significance_level)
    return BacktestResult(method=method, confidence=confidence, lookback=lookback, frame=frame, kupiec=kup, christoffersen=chris)
