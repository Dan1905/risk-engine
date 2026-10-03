"""European option pricing with Black-Scholes(-Merton).

Formulas
--------
    d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Assumptions
-----------
* European exercise only (no early exercise premium).
* Constant volatility and risk-free rate over the option's life.
* Lognormal underlying, frictionless markets, no arbitrage.
* Continuous trading (continuous hedging) approximation.
* ``r`` and ``q`` are continuously compounded; dividends are ignored by
  default (q = 0).

Numerical safeguards
--------------------
* T below one hour  -> intrinsic value (avoids 0/0 in d1).
* sigma below 1e-6  -> discounted forward intrinsic value (deterministic limit).
* Non-positive S or K raise ``ValueError``.
All functions are vectorised over NumPy arrays (broadcasting applies).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, far cheaper than the scipy.stats wrapper

MIN_T = 1.0 / (365.0 * 24.0)
MIN_VOL = 1e-6

ArrayLike = float | np.ndarray


def is_call_flag(option_type: str | np.ndarray | bool) -> np.ndarray:
    """Map 'CALL'/'PUT' (or booleans) to a boolean array (True = call)."""
    if isinstance(option_type, (bool, np.bool_)):
        return np.asarray(option_type)
    arr = np.asarray(option_type)
    if arr.dtype == bool:
        return arr
    upper = np.char.upper(arr.astype(str))
    if not np.all(np.isin(upper, ["CALL", "PUT", "C", "P"])):
        raise ValueError(f"Invalid option type(s): {option_type!r}. Use CALL or PUT.")
    return np.isin(upper, ["CALL", "C"])


def validate_inputs(S: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike) -> None:
    """Raise ``ValueError`` for economically invalid option inputs."""
    if np.any(np.asarray(S) <= 0) or np.any(~np.isfinite(np.asarray(S, dtype=float))):
        raise ValueError("Underlying price must be positive and finite.")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("Strike must be positive.")
    if np.any(np.asarray(T) < 0):
        raise ValueError("Option maturity must be greater than valuation date.")
    if np.any(np.asarray(sigma) < 0) or np.any(~np.isfinite(np.asarray(sigma, dtype=float))):
        raise ValueError("Volatility must be non-negative and finite.")


def d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0):
    """Return (d1, d2); inputs are clamped to the safe region (T >= MIN_T, sigma >= MIN_VOL)."""
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    Ts = np.maximum(T, MIN_T)
    vs = np.maximum(sigma, MIN_VOL)
    sqrt_t = np.sqrt(Ts)
    d1 = (np.log(S / K) + (r - q + 0.5 * vs**2) * Ts) / (vs * sqrt_t)
    return d1, d1 - vs * sqrt_t


def black_scholes_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: str | np.ndarray | bool = "CALL",
    q: ArrayLike = 0.0,
) -> ArrayLike:
    """Black-Scholes price of a European option (per unit of underlying).

    Args:
        S: spot price.  K: strike.  T: time to expiry in years.
        r: continuously compounded risk-free rate.  sigma: implied volatility.
        option_type: 'CALL' / 'PUT' (or array thereof).  q: dividend yield.

    Returns:
        Price with the broadcast shape of the inputs (float for scalar inputs).
    """
    validate_inputs(S, K, T, sigma)
    call = is_call_flag(option_type)
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    Ts = np.maximum(T, MIN_T)
    df_r, df_q = np.exp(-r * Ts), np.exp(-q * Ts)
    call_px = S * df_q * ndtr(d1) - K * df_r * ndtr(d2)
    put_px = K * df_r * ndtr(-d2) - S * df_q * ndtr(-d1)
    price = np.where(call, call_px, put_px)

    # Degenerate limits.
    fwd_intrinsic = np.where(call, np.maximum(S * df_q - K * df_r, 0.0), np.maximum(K * df_r - S * df_q, 0.0))
    price = np.where(sigma < MIN_VOL, fwd_intrinsic, price)
    intrinsic = np.where(call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    price = np.where(T < MIN_T, intrinsic, price)
    price = np.maximum(price, 0.0)
    return float(price) if price.ndim == 0 else price


def implied_volatility(
    price: float, S: float, K: float, T: float, r: float, option_type: str = "CALL", q: float = 0.0
) -> float:
    """Invert Black-Scholes for sigma with Brent's method.

    Raises:
        ValueError: if the price violates no-arbitrage bounds.
    """
    from scipy.optimize import brentq

    lo_px = black_scholes_price(S, K, T, r, MIN_VOL, option_type, q)
    hi_px = black_scholes_price(S, K, T, r, 5.0, option_type, q)
    if not lo_px - 1e-10 <= price <= hi_px + 1e-10:
        raise ValueError("Option price is outside no-arbitrage bounds; implied volatility undefined.")
    return float(brentq(lambda s: black_scholes_price(S, K, T, r, s, option_type, q) - price, MIN_VOL, 5.0, xtol=1e-10))
