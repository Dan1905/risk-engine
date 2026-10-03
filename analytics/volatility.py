"""Historical (equally weighted) volatility estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def historical_volatility(
    returns: pd.Series | pd.DataFrame, annualise: bool = True, periods: int = 252
) -> float | pd.Series:
    """Sample standard deviation of returns.

    Formula: ``sigma = sqrt( sum (r_t - mean)^2 / (n - 1) )``, optionally
    multiplied by ``sqrt(periods)``.

    Edge cases:
        Fewer than two observations returns NaN.
    """
    vol = returns.std(ddof=1)
    return vol * np.sqrt(periods) if annualise else vol


def rolling_volatility(
    returns: pd.Series | pd.DataFrame, window: int, annualise: bool = True, periods: int = 252
) -> pd.Series | pd.DataFrame:
    """Rolling-window sample volatility; the value at t uses returns up to and including t."""
    if window < 2:
        raise ValueError("window must be at least 2")
    vol = returns.rolling(window, min_periods=window).std(ddof=1)
    return vol * np.sqrt(periods) if annualise else vol


def realised_volatility_from_prices(prices: pd.DataFrame, window: int = 21, periods: int = 252) -> pd.DataFrame:
    """Annualised rolling realised volatility from a price frame (log returns)."""
    rets = np.log(prices / prices.shift(1))
    return rets.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods)
