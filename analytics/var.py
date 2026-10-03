"""Value-at-Risk estimators and Euler risk decomposition.

Sign convention: P&L is positive for gains; VaR and ES are reported as positive
loss amounts (a negative VaR means even the tail scenario is a gain).

Quantile convention
-------------------
``VaR_c = -Q_{1-c}(P&L)`` where ``Q`` is the *inverted empirical CDF*
(Hyndman-Fan type 1): the k-th worst P&L with ``k = ceil(N (1 - c))``. With
N = 250 and c = 99% that is the 3rd worst outcome. The same rule applied to
exponentially weighted observations (smallest P&L whose cumulative weight
reaches 1 - c) makes the weighted estimator collapse exactly to the standard
one when all weights are equal.

Holding period
--------------
1-day figures are scaled to h days with the square-root-of-time rule
``VaR_h = VaR_1 * sqrt(h)``, which assumes i.i.d. zero-mean daily P&L.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from analytics.ewma import ewma_weights

_EPS = 1e-12


class InsufficientDataError(ValueError):
    """Raised when there are too few observations for the requested estimate."""


def validate_confidence(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"Confidence must be in (0.5, 1), got {confidence}")
    return float(confidence)


def clean_pnl(pnl: pd.Series | np.ndarray | list[float], min_obs: int = 1) -> np.ndarray:
    """Return finite P&L values as a float array (order preserved: oldest first)."""
    arr = np.asarray(pnl, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < min_obs:
        raise InsufficientDataError(f"Insufficient observations: {arr.size} available, {min_obs} required")
    return arr


def minimum_observations(confidence: float) -> int:
    """Smallest sample in which at least one observation lies in the tail."""
    return int(np.ceil(1.0 / (1.0 - confidence) - 1e-9))


def tail_rank(n: int, confidence: float) -> int:
    """Rank k (1 = worst) of the VaR scenario: ``k = ceil(N (1 - c))``, at least 1.

    A small tolerance guards against floating-point noise (``1 - 0.99`` is
    ``0.010000000000000009`` in binary, which would otherwise round 100 * p
    up to 2 instead of 1).
    """
    return max(1, int(np.ceil(n * (1.0 - confidence) - 1e-9)))


def scale_to_horizon(value: float, holding_period: int) -> float:
    """Square-root-of-time scaling of a 1-day risk figure."""
    if holding_period < 1:
        raise ValueError("holding_period must be >= 1")
    return float(value) * float(np.sqrt(holding_period))


def historical_var(pnl: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Historical-simulation VaR.

    Args:
        pnl: hypothetical P&L per historical scenario.
        confidence: e.g. 0.99.

    Returns:
        ``-Q_{1-c}(P&L)`` using the inverted empirical CDF.

    Edge cases:
        Raises ``InsufficientDataError`` when fewer than ``ceil(1/(1-c))``
        observations are available (the tail would be empty).
    """
    c = validate_confidence(confidence)
    x = clean_pnl(pnl, minimum_observations(c))
    return float(-np.sort(x)[tail_rank(x.size, c) - 1])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, p: float) -> float:
    """Inverted-CDF quantile of a weighted empirical distribution.

    Sort values ascending, accumulate the (normalised) weights and return the
    first value whose cumulative weight reaches ``p``.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be non-negative and sum to a positive number")
    order = np.argsort(v, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    idx = int(np.searchsorted(cw, p - _EPS, side="left"))
    return float(v[order][min(idx, v.size - 1)])


def weighted_historical_var(pnl: pd.Series | np.ndarray, confidence: float = 0.99, lam: float = 0.94) -> float:
    """Exponentially weighted historical-simulation VaR (Boudoukh-Richardson-Whitelaw).

    Observation weights ``w_t ∝ lambda^age`` (age 0 = most recent), normalised
    to sum to one. Recent scenarios therefore dominate the tail, so the
    estimate reacts faster to a change in volatility regime than equal
    weighting, at the cost of a noisier estimate.

    Args:
        pnl: scenario P&L ordered oldest -> newest.
        confidence: e.g. 0.99.
        lam: decay factor in (0, 1).
    """
    c = validate_confidence(confidence)
    x = clean_pnl(pnl, 2)
    w = ewma_weights(x.size, lam)
    return float(-weighted_quantile(x, w, 1.0 - c))


def portfolio_volatility(weights: np.ndarray | pd.Series, cov: np.ndarray | pd.DataFrame) -> float:
    """``sigma_p = sqrt(w' Sigma w)`` (in the units of the returns)."""
    w = np.asarray(weights, dtype=float)
    s = np.asarray(cov, dtype=float)
    var = float(w @ s @ w)
    return float(np.sqrt(max(var, 0.0)))


def parametric_var(
    weights: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    confidence: float = 0.99,
    portfolio_value: float = 1.0,
    holding_period: int = 1,
) -> float:
    """Variance-covariance (delta-normal) VaR.

    Formula: ``VaR = z_c * sqrt(w' Sigma w) * V * sqrt(h)``; for 99%, z = 2.326.

    Assumptions: jointly normal returns with zero mean; exposures constant
    over the horizon.
    """
    c = validate_confidence(confidence)
    sigma = portfolio_volatility(weights, cov)
    return scale_to_horizon(norm.ppf(c) * sigma * portfolio_value, holding_period)


def marginal_var(weights: np.ndarray | pd.Series, cov: np.ndarray | pd.DataFrame, confidence: float = 0.99) -> np.ndarray:
    """Marginal VaR per unit of weight: ``dVaR/dw_i / V = z (Sigma w)_i / sigma_p``.

    Multiply by the portfolio value to obtain the change in VaR (USD) for a
    one-unit change in weight; equivalently it is the VaR added per $1 of
    additional exposure.
    """
    c = validate_confidence(confidence)
    w = np.asarray(weights, dtype=float)
    s = np.asarray(cov, dtype=float)
    sigma = portfolio_volatility(w, s)
    if sigma <= 0:
        return np.zeros_like(w)
    return norm.ppf(c) * (s @ w) / sigma


def component_var(
    weights: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    confidence: float = 0.99,
    portfolio_value: float = 1.0,
    holding_period: int = 1,
) -> np.ndarray:
    """Euler allocation: ``Component VaR_i = w_i * Marginal VaR_i * V``.

    The components sum exactly to the parametric VaR (VaR is homogeneous of
    degree one in the weights).
    """
    w = np.asarray(weights, dtype=float)
    mvar = marginal_var(w, cov, confidence)
    return w * mvar * portfolio_value * np.sqrt(holding_period)


def pnl_summary(pnl: pd.Series | np.ndarray) -> dict[str, float]:
    """Worst loss (positive number), best gain and average P&L."""
    x = clean_pnl(pnl, 1)
    return {"worst_loss": float(-x.min()), "best_gain": float(x.max()), "mean_pnl": float(x.mean())}


# ---------------------------------------------------------------------------
# Row-wise (vectorised) estimators used by rolling calculations / backtests.
# Each row of ``matrix`` is one estimation window ordered oldest -> newest.
# ---------------------------------------------------------------------------


def historical_var_rows(matrix: np.ndarray, confidence: float) -> np.ndarray:
    c = validate_confidence(confidence)
    k = tail_rank(matrix.shape[1], c)
    return -np.partition(matrix, k - 1, axis=1)[:, k - 1]


def weighted_historical_var_rows(matrix: np.ndarray, confidence: float, lam: float) -> np.ndarray:
    c = validate_confidence(confidence)
    n_rows, n = matrix.shape
    w = ewma_weights(n, lam)
    order = np.argsort(matrix, axis=1, kind="mergesort")
    sorted_vals = np.take_along_axis(matrix, order, axis=1)
    cw = np.cumsum(w[order], axis=1)
    idx = np.argmax(cw >= (1.0 - c) - _EPS, axis=1)
    return -sorted_vals[np.arange(n_rows), idx]


def parametric_var_rows(sigma: np.ndarray, confidence: float) -> np.ndarray:
    c = validate_confidence(confidence)
    return norm.ppf(c) * np.asarray(sigma, dtype=float)
