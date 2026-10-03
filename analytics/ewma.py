"""Exponentially weighted moving average (EWMA) volatility and covariance.

RiskMetrics recursion (default lambda = 0.94)::

    sigma^2_t      = lambda * sigma^2_{t-1}      + (1 - lambda) * r^2_{t-1}
    Cov_t(i, j)    = lambda * Cov_{t-1}(i, j)    + (1 - lambda) * r_{t-1,i} * r_{t-1,j}

``sigma^2_t`` is the forecast for day t made with information up to t-1, so it
never uses the return it is forecasting. Unrolling the recursion over n
observations gives the closed form used for the latest forecast::

    Sigma_{T+1} = (1 - lambda) * sum_{k=0}^{n-1} lambda^k r_{T-k} r_{T-k}^T + lambda^n Sigma_0

Assumptions: zero conditional mean (returns are not demeaned, standard for
daily data); ``Sigma_0`` is the sample covariance of the estimation window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter


def validate_lambda(lam: float) -> float:
    """Raise ``ValueError`` unless ``0 < lam < 1``."""
    if not isinstance(lam, (int, float)) or not np.isfinite(lam) or not 0.0 < float(lam) < 1.0:
        raise ValueError(f"EWMA decay factor must be strictly between 0 and 1, got {lam!r}")
    return float(lam)


def ewma_weights(n: int, lam: float, normalise: bool = True) -> np.ndarray:
    """Weights for ``n`` observations ordered oldest -> newest.

    ``w_t ∝ lambda^(age)`` with age 0 for the most recent observation. When
    ``normalise`` is True the weights sum to one.
    """
    lam = validate_lambda(lam)
    if n <= 0:
        raise ValueError("n must be positive")
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = lam**ages
    if normalise:
        w = w / w.sum()
    return w


def ewma_variance_series(returns: pd.Series, lam: float = 0.94, initial_variance: float | None = None) -> pd.Series:
    """EWMA variance forecasts ``sigma^2_t`` for every date in ``returns``.

    Args:
        returns: daily returns indexed by date.
        lam: decay factor.
        initial_variance: seed ``sigma^2`` for the first date; defaults to the
            sample variance of the series.

    Returns:
        Series aligned with ``returns`` where the value at t uses r up to t-1.

    Implementation: the recursion is a first-order IIR filter, evaluated with
    ``scipy.signal.lfilter`` (vectorised, no Python loop).
    """
    lam = validate_lambda(lam)
    r = returns.astype(float).fillna(0.0).to_numpy()
    if r.size == 0:
        return pd.Series(dtype=float, index=returns.index)
    if initial_variance is not None:
        seed = float(initial_variance)
    elif r.size > 1:
        seed = float(np.var(r, ddof=1))
    else:
        seed = float(r[0] ** 2)
    # sigma2[t] = lam*sigma2[t-1] + (1-lam)*r[t-1]^2 with sigma2[0] = seed.
    # lfilter computes y[n] = (1-lam)*x[n] + lam*y[n-1]; feeding x[0] = 0 and
    # initial state zi = seed yields y[0] = seed exactly.
    x = np.concatenate([[0.0], r[:-1] ** 2])
    y, _ = lfilter([1.0 - lam], [1.0, -lam], x, zi=[seed])
    return pd.Series(y, index=returns.index, name=returns.name)


def ewma_volatility_series(returns: pd.Series | pd.DataFrame, lam: float = 0.94) -> pd.Series | pd.DataFrame:
    """EWMA daily volatility forecasts (square root of :func:`ewma_variance_series`)."""
    if isinstance(returns, pd.DataFrame):
        return returns.apply(lambda col: np.sqrt(ewma_variance_series(col, lam)))
    return np.sqrt(ewma_variance_series(returns, lam))


def ewma_latest_variance(returns: pd.Series | np.ndarray, lam: float = 0.94) -> float:
    """Variance forecast for the day after the last observation (includes r_T)."""
    arr = np.asarray(returns, dtype=float)
    return float(ewma_covariance(pd.DataFrame(arr[:, None]), lam).iloc[0, 0])


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Latest EWMA covariance forecast ``Sigma_{T+1}`` (closed form).

    Args:
        returns: T x N frame of returns, oldest first. NaNs are treated as 0.
        lam: decay factor.

    Returns:
        N x N covariance matrix labelled by the columns of ``returns``.

    Edge cases:
        With a single observation the seed covariance is that observation's
        outer product. Constant (zero-variance) series produce zero rows/cols.
    """
    lam = validate_lambda(lam)
    r = returns.astype(float).fillna(0.0).to_numpy()
    n = r.shape[0]
    if n == 0:
        raise ValueError("Cannot estimate a covariance matrix from zero observations")
    seed = np.cov(r, rowvar=False, ddof=1) if n > 1 else np.outer(r[0], r[0])
    seed = np.atleast_2d(seed)
    w = (1.0 - lam) * lam ** np.arange(n - 1, -1, -1, dtype=float)
    cov = (r * w[:, None]).T @ r + lam**n * seed
    cov = 0.5 * (cov + cov.T)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_covariance_recursive(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """Reference implementation of the covariance recursion (explicit loop).

    Produces the same result as :func:`ewma_covariance`; kept for transparency
    and used by the unit tests to verify the closed form.
    """
    lam = validate_lambda(lam)
    r = returns.astype(float).fillna(0.0).to_numpy()
    n = r.shape[0]
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1) if n > 1 else np.outer(r[0], r[0]))
    for t in range(n):
        cov = lam * cov + (1.0 - lam) * np.outer(r[t], r[t])
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
