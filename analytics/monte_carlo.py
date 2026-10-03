"""Monte Carlo simulation of market-factor moves.

Factor moves are drawn from a zero-mean multivariate distribution whose
covariance is estimated from history (sample or EWMA) and every path is then
*fully revalued*, so option convexity and bond convexity are captured -
unlike the variance-covariance method, which is linear in the factors.

Normal:      X = Z A',  Z ~ N(0, I),  A A' = Sigma
Student-t:   X = Z A' * sqrt((nu - 2) / W),  W ~ chi-square(nu)

The Student-t draw is rescaled so its covariance is still ``Sigma`` (requires
nu > 2); it adds joint fat tails - all factors share the same mixing variable
W, so extreme moves tend to occur together.

``A`` comes from an eigen-decomposition (``Sigma = V diag(l) V'``,
``A = V diag(sqrt(max(l, 0)))``), which works for singular or slightly
non-PSD covariance matrices where a Cholesky factorisation would fail.

A horizon of h days scales the covariance by h (i.i.d. daily shocks).
Antithetic variates (Z and -Z) halve the sampling noise of odd moments at no
cost. The generator is seeded, so identical inputs give identical results.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from analytics.risk_factors import clip_returns

Distribution = Literal["NORMAL", "STUDENT_T"]


def covariance_root(cov: np.ndarray) -> np.ndarray:
    """Matrix ``A`` with ``A A' = cov`` (negative eigenvalues clipped to zero)."""
    a = np.asarray(cov, dtype=float)
    a = 0.5 * (a + a.T)
    if not np.all(np.isfinite(a)):
        raise ValueError("Covariance matrix contains non-finite values")
    eigval, eigvec = np.linalg.eigh(a)
    return eigvec * np.sqrt(np.clip(eigval, 0.0, None))


def standard_draws(
    n_paths: int, n_factors: int, seed: int, distribution: Distribution = "STUDENT_T", dof: float = 6.0, antithetic: bool = True
) -> np.ndarray:
    """Unit-covariance draws of shape (n_paths, n_factors).

    Reusing the same draws across dates (common random numbers) makes a
    rolling Monte Carlo VaR series smooth: changes come from the covariance
    and the portfolio, not from sampling noise.
    """
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_factors))
    if distribution == "STUDENT_T":
        if dof <= 2:
            raise ValueError("Student-t degrees of freedom must exceed 2 for a finite covariance")
        w = rng.chisquare(dof, size=half)
        z = z * np.sqrt((dof - 2.0) / w)[:, None]
    elif distribution != "NORMAL":
        raise ValueError(f"Unknown distribution '{distribution}'")
    if antithetic:
        z = np.vstack([z, -z])[:n_paths]
    return z


def simulate_factor_moves(
    cov: pd.DataFrame,
    n_paths: int = 10_000,
    seed: int = 42,
    distribution: Distribution = "STUDENT_T",
    dof: float = 6.0,
    horizon: int = 1,
    antithetic: bool = True,
    draws: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate factor moves with covariance ``cov * horizon``.

    Args:
        cov: daily factor covariance (labelled by factor name).
        n_paths: number of simulated scenarios.
        seed: RNG seed (ignored when ``draws`` is supplied).
        distribution: ``NORMAL`` or ``STUDENT_T``.
        dof: Student-t degrees of freedom (> 2).
        horizon: holding period in days.
        draws: pre-generated unit-covariance draws (see :func:`standard_draws`).

    Returns:
        (n_paths x n_factors) factor frame; return factors floored at -95%.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    k = cov.shape[0]
    z = draws if draws is not None else standard_draws(n_paths, k, seed, distribution, dof, antithetic)
    if z.shape[1] != k:
        raise ValueError("draws do not match the number of factors")
    x = z @ covariance_root(cov.to_numpy()).T * np.sqrt(horizon)
    frame = pd.DataFrame(x, index=pd.RangeIndex(len(z), name="path"), columns=cov.columns)
    return clip_returns(frame)
