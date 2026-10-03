"""Black-Scholes Greeks (analytic) and position-level aggregation.

Per-unit formulas (q = dividend yield, n = standard normal pdf)::

    Delta_call = e^{-qT} N(d1)             Delta_put = e^{-qT} (N(d1) - 1)
    Gamma      = e^{-qT} n(d1) / (S sigma sqrt(T))
    Vega       = S e^{-qT} n(d1) sqrt(T)                 (per 1.00 change in sigma)
    Theta_call = -S e^{-qT} n(d1) sigma / (2 sqrt(T)) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)
    Theta_put  = -S e^{-qT} n(d1) sigma / (2 sqrt(T)) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)
                                                          (per year)
    Rho_call   =  K T e^{-rT} N(d2)          Rho_put = -K T e^{-rT} N(-d2)   (per 1.00 change in r)

Reporting units used across the application (see :func:`position_greeks`):
    Delta  shares-equivalent (x quantity x multiplier); Delta $ = Delta x S
    Gamma  change in Delta per $1 move in the underlying
    Vega   $ per 1 volatility point (sigma +0.01)
    Theta  $ per calendar day
    Rho    $ per 1% (100 bp) move in rates

At or past expiry (T < MIN_T) or with zero volatility the Greeks collapse to
their deterministic limits: Delta is the exercise indicator, the rest are 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from analytics.options import MIN_T, MIN_VOL, ArrayLike, d1_d2, is_call_flag, validate_inputs


_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal density."""
    return _INV_SQRT_2PI * np.exp(-0.5 * np.asarray(x, dtype=float) ** 2)


def _prep(S, K, T, r, sigma, q):
    validate_inputs(S, K, T, sigma)
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))
    degenerate = (T < MIN_T) | (sigma < MIN_VOL)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S, K, np.maximum(T, MIN_T), r, np.maximum(sigma, MIN_VOL), q, d1, d2, degenerate


def _out(x: np.ndarray) -> ArrayLike:
    return float(x) if np.ndim(x) == 0 else x


def delta(S, K, T, r, sigma, option_type="CALL", q=0.0) -> ArrayLike:
    call = is_call_flag(option_type)
    S, K, T, r, sigma, q, d1, _, degen = _prep(S, K, T, r, sigma, q)
    dq = np.exp(-q * T)
    val = np.where(call, dq * ndtr(d1), dq * (ndtr(d1) - 1.0))
    itm = np.where(call, (S > K).astype(float), -(S < K).astype(float))
    return _out(np.where(degen, itm, val))


def gamma(S, K, T, r, sigma, option_type="CALL", q=0.0) -> ArrayLike:
    S, K, T, r, sigma, q, d1, _, degen = _prep(S, K, T, r, sigma, q)
    val = np.exp(-q * T) * _pdf(d1) / (S * sigma * np.sqrt(T))
    return _out(np.where(degen, 0.0, val))


def vega(S, K, T, r, sigma, option_type="CALL", q=0.0) -> ArrayLike:
    S, K, T, r, sigma, q, d1, _, degen = _prep(S, K, T, r, sigma, q)
    val = S * np.exp(-q * T) * _pdf(d1) * np.sqrt(T)
    return _out(np.where(degen, 0.0, val))


def theta(S, K, T, r, sigma, option_type="CALL", q=0.0) -> ArrayLike:
    call = is_call_flag(option_type)
    S, K, T, r, sigma, q, d1, d2, degen = _prep(S, K, T, r, sigma, q)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    common = -S * dq * _pdf(d1) * sigma / (2.0 * np.sqrt(T))
    call_th = common - r * K * dr * ndtr(d2) + q * S * dq * ndtr(d1)
    put_th = common + r * K * dr * ndtr(-d2) - q * S * dq * ndtr(-d1)
    return _out(np.where(degen, 0.0, np.where(call, call_th, put_th)))


def rho(S, K, T, r, sigma, option_type="CALL", q=0.0) -> ArrayLike:
    call = is_call_flag(option_type)
    S, K, T, r, sigma, q, _, d2, degen = _prep(S, K, T, r, sigma, q)
    dr = np.exp(-r * T)
    val = np.where(call, K * T * dr * ndtr(d2), -K * T * dr * ndtr(-d2))
    return _out(np.where(degen, 0.0, val))


@dataclass(frozen=True)
class Greeks:
    """Per-unit Greeks in raw model units (vega/rho per 1.00, theta per year)."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def all_greeks(S, K, T, r, sigma, option_type="CALL", q=0.0) -> dict[str, ArrayLike]:
    """All five Greeks in raw model units (vectorised)."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, option_type, q),
        "vega": vega(S, K, T, r, sigma, option_type, q),
        "theta": theta(S, K, T, r, sigma, option_type, q),
        "rho": rho(S, K, T, r, sigma, option_type, q),
    }


def position_greeks(
    quantity: float,
    multiplier: float,
    spot: float,
    raw: dict[str, float],
    fx_rate: float = 1.0,
) -> dict[str, float]:
    """Scale per-unit Greeks to a position in reporting units (base currency).

    Args:
        quantity: number of contracts (negative = short).
        multiplier: shares per contract.
        spot: underlying price (local currency).
        raw: output of :func:`all_greeks` for one option.
        fx_rate: USD per unit of the option's currency.
    """
    units = quantity * multiplier
    return {
        "delta": units * raw["delta"],  # shares-equivalent
        "delta_usd": units * raw["delta"] * spot * fx_rate,
        "gamma": units * raw["gamma"],  # delta change per 1 unit of underlying
        "gamma_usd": 0.5 * units * raw["gamma"] * (0.01 * spot) ** 2 * fx_rate,  # P&L from gamma for a 1% move
        "vega": units * raw["vega"] * 0.01 * fx_rate,  # per vol point
        "theta": units * raw["theta"] / 365.0 * fx_rate,  # per calendar day
        "rho": units * raw["rho"] * 0.01 * fx_rate,  # per 1% rate move
    }
