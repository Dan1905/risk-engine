"""Implied-volatility surface abstraction: ``sigma = f(strike, maturity)``.

Surfaces are parameterised in *moneyness* ``m = K / S`` and maturity ``T`` so
that the same object can express both stress conventions:

* Sticky strike - the vol of a given strike is unchanged when spot moves:
  ``sigma_new(K) = sigma_old(K)``.
* Sticky delta (sticky moneyness) - the vol is attached to moneyness, so when
  spot moves an option with fixed strike slides along the smile:
  ``sigma_new(K) = f(K / S_new, T)``.

Three implementations:

``FlatVolSurface``    one volatility for every strike and maturity.
``SmileVolSurface``   quadratic smile ``atm + skew (m-1) + curvature (m-1)^2``;
                      can be built from ATM / 25-delta call / 25-delta put quotes.
``GridVolSurface``    maturity x moneyness grid; bilinear interpolation (or
                      piecewise-constant maturity buckets), flat extrapolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from analytics.yield_curve import interpolation_weights

MIN_SURFACE_VOL = 0.01


class VolatilitySurface(ABC):
    """Interface: implied volatility as a function of moneyness and maturity."""

    @abstractmethod
    def vol(self, moneyness: np.ndarray | float, maturity: np.ndarray | float) -> np.ndarray:
        """Implied vol at moneyness ``K/S`` and maturity (years)."""

    def vol_for_strike(self, strike: float | np.ndarray, spot: float | np.ndarray, maturity: float | np.ndarray) -> np.ndarray:
        return self.vol(np.asarray(strike, dtype=float) / np.asarray(spot, dtype=float), maturity)

    @abstractmethod
    def shifted(self, shift: float) -> "VolatilitySurface":
        """Parallel shift in absolute vol points (0.05 = +5 vols)."""


@dataclass(frozen=True)
class FlatVolSurface(VolatilitySurface):
    sigma: float

    def vol(self, moneyness, maturity):
        shape = np.broadcast(np.asarray(moneyness), np.asarray(maturity)).shape
        return np.full(shape, max(self.sigma, MIN_SURFACE_VOL))

    def shifted(self, shift: float) -> "FlatVolSurface":
        return FlatVolSurface(max(self.sigma + shift, MIN_SURFACE_VOL))


@dataclass(frozen=True)
class SmileVolSurface(VolatilitySurface):
    """Maturity-independent quadratic smile in moneyness."""

    atm: float
    skew: float = 0.0
    curvature: float = 0.0

    def vol(self, moneyness, maturity):
        m = np.asarray(moneyness, dtype=float) - 1.0
        v = self.atm + self.skew * m + self.curvature * m**2
        v = np.broadcast_to(v, np.broadcast(np.asarray(moneyness), np.asarray(maturity)).shape)
        return np.maximum(v, MIN_SURFACE_VOL)

    def shifted(self, shift: float) -> "SmileVolSurface":
        return SmileVolSurface(self.atm + shift, self.skew, self.curvature)

    @classmethod
    def from_delta_quotes(
        cls, atm: float, call_25d: float, put_25d: float, maturity: float, rate: float = 0.0
    ) -> "SmileVolSurface":
        """Fit the quadratic through ATM, 25-delta call and 25-delta put vols.

        The 25-delta strikes are found from ``N(d1) = 0.25`` (call) and
        ``N(d1) = 0.75`` (put, |delta| = 0.25) using each quote's own vol:
        ``K = S exp(-d1 sigma sqrt(T) + (r + sigma^2/2) T)``.
        """
        def moneyness(target_nd1: float, sigma: float) -> float:
            d1 = norm.ppf(target_nd1)
            return float(np.exp(-d1 * sigma * np.sqrt(maturity) + (rate + 0.5 * sigma**2) * maturity))

        m = np.array([moneyness(0.75, put_25d), moneyness(0.5, atm), moneyness(0.25, call_25d)]) - 1.0
        v = np.array([put_25d, atm, call_25d])
        a = np.column_stack([np.ones(3), m, m**2])
        c0, c1, c2 = np.linalg.solve(a, v)
        # Re-express around m = 1 exactly (c0 is the vol at m = 1).
        return cls(atm=float(c0), skew=float(c1), curvature=float(c2))


@dataclass(frozen=True)
class GridVolSurface(VolatilitySurface):
    """Implied vols on a maturity x moneyness grid."""

    maturities: np.ndarray  # (n_T,) years, increasing
    moneyness_grid: np.ndarray  # (n_m,), increasing
    vols: np.ndarray  # (n_T, n_m)
    maturity_buckets: bool = False  # True: piecewise-constant in maturity

    def __post_init__(self) -> None:
        mats = np.asarray(self.maturities, dtype=float)
        mny = np.asarray(self.moneyness_grid, dtype=float)
        v = np.asarray(self.vols, dtype=float)
        if v.shape != (mats.size, mny.size):
            raise ValueError("vols must have shape (n_maturities, n_moneyness)")
        if np.any(~np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("Volatility surface data unavailable or invalid for one or more points")
        object.__setattr__(self, "maturities", mats)
        object.__setattr__(self, "moneyness_grid", mny)
        object.__setattr__(self, "vols", v)

    def _maturity_weights(self, t: np.ndarray) -> np.ndarray:
        if self.maturities.size == 1:
            return np.ones((t.size, 1))
        if not self.maturity_buckets:
            return interpolation_weights(self.maturities, t)
        idx = np.clip(np.searchsorted(self.maturities, t, side="left"), 0, self.maturities.size - 1)
        w = np.zeros((t.size, self.maturities.size))
        w[np.arange(t.size), idx] = 1.0
        return w

    def vol(self, moneyness, maturity):
        m = np.asarray(moneyness, dtype=float)
        t = np.asarray(maturity, dtype=float)
        shape = np.broadcast(m, t).shape
        mf, tf = np.broadcast_to(m, shape).ravel(), np.broadcast_to(t, shape).ravel()
        wt = self._maturity_weights(tf)  # (n, n_T)
        wm = interpolation_weights(self.moneyness_grid, mf)  # (n, n_m)
        out = np.einsum("ni,ij,nj->n", wt, self.vols, wm)
        return np.maximum(out.reshape(shape), MIN_SURFACE_VOL)

    def shifted(self, shift: float) -> "GridVolSurface":
        return GridVolSurface(
            self.maturities, self.moneyness_grid, np.maximum(self.vols + shift, MIN_SURFACE_VOL), self.maturity_buckets
        )

    def atm_term_structure(self) -> pd.Series:
        return pd.Series(self.vol(np.ones_like(self.maturities), self.maturities), index=self.maturities)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, maturity_buckets: bool = False) -> "GridVolSurface":
        """Build from long-format rows with ``maturity_years``, ``moneyness``, ``implied_vol``."""
        if df.empty:
            raise ValueError("Volatility surface data unavailable for selected option.")
        grid = df.pivot_table(index="maturity_years", columns="moneyness", values="implied_vol", aggfunc="last")
        grid = grid.sort_index().sort_index(axis=1)
        return cls(grid.index.to_numpy(float), grid.columns.to_numpy(float), grid.to_numpy(float), maturity_buckets)


# ---------------------------------------------------------------------------
# Volatility dynamics used when spot moves in a scenario.
# ---------------------------------------------------------------------------


def sticky_strike_vol(sigma0: float | np.ndarray, vol_shift: float | np.ndarray = 0.0, floor: float = MIN_SURFACE_VOL) -> np.ndarray:
    """Sticky strike: the option keeps its own implied vol; only the parallel shift applies."""
    return np.maximum(np.asarray(sigma0, dtype=float) + vol_shift, floor)


def sticky_delta_vol(
    sigma0: float | np.ndarray,
    surface: VolatilitySurface,
    strike: float,
    spot0: float,
    spot1: float | np.ndarray,
    maturity: float,
    vol_shift: float | np.ndarray = 0.0,
    floor: float = MIN_SURFACE_VOL,
) -> np.ndarray:
    """Sticky delta (moneyness): vol follows the smile as spot moves.

    ``sigma1 = sigma0 + [f(K/S1, T) - f(K/S0, T)] + shift``. Anchoring to
    ``sigma0`` keeps the option's own vol level while importing the smile
    move, so the result is consistent even when the position's implied vol
    differs slightly from the surface.
    """
    t = max(float(maturity), 1e-6)
    smile_move = surface.vol(strike / np.asarray(spot1, dtype=float), t) - float(surface.vol(strike / spot0, t))
    return np.maximum(np.asarray(sigma0, dtype=float) + smile_move + vol_shift, floor)


# ---------------------------------------------------------------------------
# Three-factor description of surface moves (used as VaR risk factors).
# ---------------------------------------------------------------------------
#
#   level  L = sigma(100%, 3M)
#   skew   S = sigma(90%, 3M) - sigma(110%, 3M)          (positive = puts richer than calls)
#   term   T = sigma(100%, 1Y) - sigma(100%, 1M)          (positive = upward-sloping ATM curve)
#
# A move (dL, dS, dT) changes the vol of an option at moneyness m and maturity t by
#
#   d sigma(m, t) = dL + dS * skew_loading(m) + dT * term_loading(t)
#
# with loadings chosen so the three reference quantities are reproduced exactly:
# skew_loading is linear in moneyness (+0.5 at 90%, -0.5 at 110%, 0 at the money) and
# term_loading is linear in maturity, zero at 3M, with loading(1Y) - loading(1M) = 1.

LEVEL_MATURITY = 0.25
SKEW_LOW, SKEW_HIGH = 0.90, 1.10
TERM_SHORT, TERM_LONG = 1.0 / 12.0, 1.0
MONEYNESS_BOUNDS = (0.80, 1.20)


def surface_factors(surface: VolatilitySurface) -> tuple[float, float, float]:
    """(level, skew, term) of a surface as defined above."""
    level = float(surface.vol(1.0, LEVEL_MATURITY))
    skew = float(surface.vol(SKEW_LOW, LEVEL_MATURITY)) - float(surface.vol(SKEW_HIGH, LEVEL_MATURITY))
    term = float(surface.vol(1.0, TERM_LONG)) - float(surface.vol(1.0, TERM_SHORT))
    return level, skew, term


def skew_loading(moneyness: float | np.ndarray) -> np.ndarray:
    """Sensitivity of an option's vol to the skew factor: ``(1 - m) / (SKEW_HIGH - SKEW_LOW)``, m clipped to the grid."""
    m = np.clip(np.asarray(moneyness, dtype=float), *MONEYNESS_BOUNDS)
    return (1.0 - m) / (SKEW_HIGH - SKEW_LOW)


def term_loading(maturity: float | np.ndarray) -> np.ndarray:
    """Sensitivity of an option's vol to the term factor: linear in maturity, zero at 3M."""
    t = np.clip(np.asarray(maturity, dtype=float), TERM_SHORT, TERM_LONG)
    return (t - LEVEL_MATURITY) / (TERM_LONG - TERM_SHORT)
