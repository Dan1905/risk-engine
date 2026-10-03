"""Covariance estimation and numerical safeguards.

Parametric VaR only needs the quadratic form ``w' Sigma w``, so the engine never
inverts a covariance matrix. What it does need is a positive semi-definite
matrix: EWMA estimates with few observations relative to the number of assets,
or series containing missing values, can produce tiny negative eigenvalues.
:func:`ensure_psd` repairs those by eigenvalue clipping (Higham-style nearest
PSD approximation). If a future feature needs ``Sigma^{-1}`` (e.g. a
minimum-variance hedge), ``np.linalg.pinv`` should be used on near-singular
matrices; :func:`condition_number` reports when that is the case.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from analytics.ewma import ewma_covariance

logger = logging.getLogger(__name__)

CovMethod = Literal["SAMPLE", "EWMA"]


class SingularCovarianceError(ValueError):
    """Raised when a covariance matrix cannot be used (e.g. all zeros)."""


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Equally weighted sample covariance (demeaned, ddof=1)."""
    if len(returns) < 2:
        raise ValueError("At least two observations are required for a sample covariance")
    return returns.astype(float).cov(ddof=1)


def covariance_matrix(returns: pd.DataFrame, method: CovMethod = "SAMPLE", lam: float = 0.94) -> pd.DataFrame:
    """Estimate a covariance matrix with the chosen method and make it PSD.

    Args:
        returns: T x N returns, oldest first.
        method: ``SAMPLE`` or ``EWMA``.
        lam: EWMA decay factor (ignored for SAMPLE).
    """
    if method == "SAMPLE":
        cov = sample_covariance(returns.fillna(0.0))
    elif method == "EWMA":
        cov = ewma_covariance(returns, lam)
    else:
        raise ValueError(f"Unknown covariance method '{method}'")
    return ensure_psd(cov)


def ensure_psd(cov: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """Return ``cov`` with negative eigenvalues clipped to zero.

    The matrix is symmetrised first. Clipping is only applied (and logged) when
    the smallest eigenvalue is materially negative.
    """
    a = cov.to_numpy(dtype=float)
    a = 0.5 * (a + a.T)
    if not np.all(np.isfinite(a)):
        raise SingularCovarianceError("Covariance matrix contains non-finite values")
    eigval, eigvec = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(eigval))), 1e-300)
    if eigval.min() < -tol * scale:
        logger.warning("Covariance matrix not PSD (min eigenvalue %.3e); clipping", eigval.min())
        eigval = np.clip(eigval, 0.0, None)
        a = (eigvec * eigval) @ eigvec.T
        a = 0.5 * (a + a.T)
    return pd.DataFrame(a, index=cov.index, columns=cov.columns)


def condition_number(cov: pd.DataFrame) -> float:
    """2-norm condition number; ``inf`` for singular matrices."""
    eig = np.linalg.eigvalsh(cov.to_numpy(dtype=float))
    lo = eig.min()
    return float("inf") if lo <= 0 else float(eig.max() / lo)


def is_singular(cov: pd.DataFrame, threshold: float = 1e12) -> bool:
    """True when the matrix is singular or numerically near-singular."""
    return condition_number(cov) > threshold


def annualise_covariance(cov: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    return cov * periods
