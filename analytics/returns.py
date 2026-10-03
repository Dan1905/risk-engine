"""Price cleaning and return calculation.

Formulas
--------
Simple return:  R_t = P_t / P_{t-1} - 1
Log return:     r_t = ln(P_t / P_{t-1})

The two are interchangeable for scenario generation (``P_t = P_{t-1} e^{r}`` or
``P_{t-1}(1+R)`` give the same price), but differ as inputs to volatility and
covariance estimation, which is why the choice is configurable.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ReturnType = Literal["log", "simple"]


def clean_price_frame(prices: pd.DataFrame, max_ffill: int = 5, business_days_only: bool = True) -> pd.DataFrame:
    """Clean a wide price frame (dates x instruments).

    Steps:
        1. Coerce the index to a sorted ``DatetimeIndex``.
        2. Drop duplicate dates (the last observation for a date wins).
        3. Drop weekend rows (non-trading days) when ``business_days_only``.
        4. Treat zero, negative and non-finite prices as missing.
        5. Forward-fill gaps of at most ``max_ffill`` observations
           (stale-price assumption); longer gaps stay missing.
        6. Drop rows where every instrument is missing.

    Args:
        prices: wide frame of prices.
        max_ffill: maximum consecutive observations to forward-fill.
        business_days_only: drop Saturday/Sunday observations.

    Returns:
        Cleaned copy of ``prices``.

    Edge cases:
        Leading missing values are never back-filled (that would leak future
        information into the past).
    """
    if prices.empty:
        return prices.copy()
    df = prices.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    n_dupes = int(df.index.duplicated(keep="last").sum())
    if n_dupes:
        logger.warning("Dropping %d duplicate date rows", n_dupes)
        df = df[~df.index.duplicated(keep="last")]
    if business_days_only:
        weekend = df.index.dayofweek >= 5
        if weekend.any():
            logger.info("Dropping %d non-trading (weekend) rows", int(weekend.sum()))
            df = df[~weekend]
    df = df.apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(df.to_numpy(dtype=float)) | (df.to_numpy(dtype=float) <= 0)
    n_invalid = int((invalid & df.notna().to_numpy()).sum())
    if n_invalid:
        logger.warning("Treating %d zero/negative/non-finite prices as missing", n_invalid)
    df = df.mask(invalid)
    if max_ffill > 0:
        df = df.ffill(limit=max_ffill)
    return df.dropna(how="all")


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Simple returns ``P_t / P_{t-1} - 1``; the first observation is dropped."""
    return (prices / prices.shift(1) - 1.0).iloc[1:]


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Log returns ``ln(P_t / P_{t-1})``; the first observation is dropped."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def compute_returns(prices: pd.DataFrame | pd.Series, method: ReturnType = "log") -> pd.DataFrame | pd.Series:
    """Dispatch to :func:`log_returns` or :func:`simple_returns`.

    Raises:
        ValueError: on an unknown method.
    """
    if method == "log":
        return log_returns(prices)
    if method == "simple":
        return simple_returns(prices)
    raise ValueError(f"Unknown return method '{method}' (expected 'log' or 'simple')")


def to_simple(returns: pd.DataFrame | pd.Series | np.ndarray, method: ReturnType) -> pd.DataFrame | pd.Series | np.ndarray:
    """Convert returns of type ``method`` to simple returns (``e^r - 1`` for log)."""
    return np.expm1(returns) if method == "log" else returns


def differences(levels: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Absolute first differences (used for yields, spreads and implied vols)."""
    return levels.diff().iloc[1:]


def annualise_volatility(daily_vol: float | pd.Series | np.ndarray, periods: int = 252) -> float | pd.Series | np.ndarray:
    """Square-root-of-time annualisation of a daily volatility."""
    return daily_vol * np.sqrt(periods)
