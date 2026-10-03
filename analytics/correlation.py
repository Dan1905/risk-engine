"""Correlation matrices and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.covariance import CovMethod, covariance_matrix


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix.

    Zero-variance instruments (e.g. cash) get zero correlation with everything
    and a unit diagonal instead of NaN from a 0/0 division.
    """
    a = cov.to_numpy(dtype=float)
    sd = np.sqrt(np.clip(np.diag(a), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = a / np.outer(sd, sd)
    corr[~np.isfinite(corr)] = 0.0
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def correlation_matrix(returns: pd.DataFrame, method: CovMethod = "SAMPLE", lam: float = 0.94) -> pd.DataFrame:
    """Correlation matrix from returns using the chosen covariance estimator."""
    return cov_to_corr(covariance_matrix(returns, method, lam))


@dataclass(frozen=True)
class CorrelationSummary:
    highest_positive: tuple[str, str, float] | None
    highest_negative: tuple[str, str, float] | None
    average: float


def correlation_summary(corr: pd.DataFrame, exclude: set[str] | None = None) -> CorrelationSummary:
    """Highest positive / negative pair and average off-diagonal correlation.

    Args:
        corr: correlation matrix.
        exclude: labels to ignore (e.g. zero-variance cash).
    """
    labels = [c for c in corr.columns if not exclude or c not in exclude]
    sub = corr.loc[labels, labels].to_numpy(dtype=float)
    n = len(labels)
    if n < 2:
        return CorrelationSummary(None, None, float("nan"))
    iu = np.triu_indices(n, k=1)
    vals = sub[iu]
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    hp = (labels[iu[0][i_max]], labels[iu[1][i_max]], float(vals[i_max]))
    hn = (labels[iu[0][i_min]], labels[iu[1][i_min]], float(vals[i_min]))
    return CorrelationSummary(
        highest_positive=hp if hp[2] > 0 else None,
        highest_negative=hn if hn[2] < 0 else None,
        average=float(vals.mean()),
    )
