"""Filtered (volatility-scaled) historical simulation.

Plain historical simulation replays past factor moves at the volatility that
prevailed when they happened. If today's volatility is higher than it was
then, those scenarios understate risk (and vice versa). Filtered historical
simulation (Hull & White 1998; Barone-Adesi et al. 1999) rescales each
historical move to today's volatility before revaluing::

    move*_t = move_t * sigma_{T+1} / sigma_t

where ``sigma_t`` is the EWMA volatility forecast for day t made with data up
to t-1, and ``sigma_{T+1}`` is the current forecast. The empirical shape of
the standardised moves (fat tails, joint behaviour across factors) is kept;
only their scale is updated. VaR and ES are then read off the equally
weighted distribution of the revalued scenarios.

No look-ahead: ``sigma_t`` only uses moves before t, and the target
``sigma_{T+1}`` only uses moves up to T. The recursion is seeded with the
mean square of the first ``seed_obs`` observations, all of which precede any
date for which a VaR is produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from analytics.ewma import validate_lambda
from analytics.risk_factors import clip_returns

DEFAULT_SEED_OBS = 20
DEFAULT_RATIO_BOUNDS = (1.0 / 3.0, 3.0)
_MIN_VOL = 1e-12


def ewma_vol_forecasts(moves: pd.DataFrame, lam: float = 0.94, seed_obs: int = DEFAULT_SEED_OBS) -> np.ndarray:
    """EWMA volatility forecasts for every row of ``moves`` and for the next day.

    Returns:
        Array of shape (n + 1, n_factors). Row t (t < n) is the forecast for
        ``moves.iloc[t]`` using rows before t; row n is the forecast for the
        day after the last row.
    """
    lam = validate_lambda(lam)
    x = moves.to_numpy(dtype=float)
    n, k = x.shape
    if n == 0:
        raise ValueError("Cannot compute volatility forecasts from zero observations")
    seed = np.mean(x[: max(1, min(seed_obs, n))] ** 2, axis=0)
    # sigma2[t] = lam * sigma2[t-1] + (1 - lam) * x[t-1]^2 with sigma2[0] = seed
    drive = np.vstack([np.zeros((1, k)), x**2])
    sigma2, _ = lfilter([1.0 - lam], [1.0, -lam], drive, axis=0, zi=seed[None, :])
    return np.sqrt(np.maximum(sigma2, 0.0))


def scaling_ratios(
    vols: np.ndarray, start: int, stop: int, bounds: tuple[float, float] = DEFAULT_RATIO_BOUNDS
) -> np.ndarray:
    """Ratios ``sigma_target / sigma_t`` for rows ``[start, stop)``; the target is the forecast for row ``stop``.

    Ratios are clipped to ``bounds`` so one quiet (or wild) day cannot inflate
    a scenario without limit, and set to 1 for factors with no variability.
    """
    target = vols[stop]
    window = vols[start:stop]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = target[None, :] / window
    ratio = np.where((window > _MIN_VOL) & np.isfinite(ratio), ratio, 1.0)
    return np.clip(ratio, bounds[0], bounds[1])


def filter_moves(
    moves: pd.DataFrame,
    lookback: int,
    lam: float = 0.94,
    bounds: tuple[float, float] = DEFAULT_RATIO_BOUNDS,
    seed_obs: int = DEFAULT_SEED_OBS,
) -> pd.DataFrame:
    """Volatility-filter the last ``lookback`` rows of a daily factor frame.

    Args:
        moves: daily factor moves, oldest first. Rows before the last
            ``lookback`` are only used to warm up the volatility recursion.
        lookback: number of scenarios to return.

    Returns:
        ``lookback`` rescaled scenarios (same columns), returns floored at -95%.
    """
    n = len(moves)
    if n < lookback:
        raise ValueError(f"Need at least {lookback} observations, got {n}")
    vols = ewma_vol_forecasts(moves, lam, seed_obs)
    ratios = scaling_ratios(vols, n - lookback, n, bounds)
    window = moves.iloc[n - lookback :]
    return clip_returns(window * ratios)
