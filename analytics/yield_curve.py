"""Zero-coupon yield curve with linear interpolation.

Conventions
-----------
* Node rates are annually compounded zero rates (decimals).
* Discount factor: ``DF(t) = (1 + z(t))^{-t}``.
* Interpolation is linear in maturity between nodes and flat beyond the first
  and last node. Because linear interpolation is linear in the node values,
  ``z(t) = sum_k W_k(t) z_k`` with "tent" weights ``W`` that sum to one. The
  same weights map node shocks to cash-flow shocks, which is what makes
  key-rate durations and non-parallel curve stresses consistent with each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

TENOR_LABELS: tuple[str, ...] = ("3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y")
TENOR_YEARS: np.ndarray = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([DWMY])\s*$", re.IGNORECASE)


def tenor_to_years(label: str) -> float:
    """Convert a tenor label such as ``3M`` or ``10Y`` to years."""
    m = _TENOR_RE.match(label)
    if not m:
        raise ValueError(f"Invalid tenor label '{label}'")
    n, unit = float(m.group(1)), m.group(2).upper()
    return {"D": n / 365.0, "W": n * 7 / 365.0, "M": n / 12.0, "Y": n}[unit]


def interpolation_weights(node_times: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Linear-interpolation weights mapping node values to values at ``t``.

    Args:
        node_times: increasing node maturities (n_nodes,).
        t: query maturities (n,) or scalar.

    Returns:
        (n, n_nodes) matrix whose rows sum to one (flat extrapolation).
    """
    nodes = np.asarray(node_times, dtype=float)
    tt = np.atleast_1d(np.asarray(t, dtype=float))
    w = np.zeros((tt.size, nodes.size))
    tc = np.clip(tt, nodes[0], nodes[-1])
    j = np.clip(np.searchsorted(nodes, tc, side="right") - 1, 0, nodes.size - 2)
    left, right = nodes[j], nodes[j + 1]
    frac = (tc - left) / (right - left)
    rows = np.arange(tt.size)
    w[rows, j] = 1.0 - frac
    w[rows, j + 1] += frac
    return w


@dataclass(frozen=True)
class YieldCurve:
    """Immutable zero curve defined on key-rate nodes."""

    tenors: np.ndarray
    rates: np.ndarray
    labels: tuple[str, ...] = TENOR_LABELS

    def __post_init__(self) -> None:
        t = np.asarray(self.tenors, dtype=float)
        r = np.asarray(self.rates, dtype=float)
        if t.shape != r.shape or t.ndim != 1 or t.size < 2:
            raise ValueError("Yield curve needs matching 1-D tenor and rate arrays with at least two nodes")
        if np.any(np.diff(t) <= 0):
            raise ValueError("Curve tenors must be strictly increasing")
        if not np.all(np.isfinite(r)):
            raise ValueError("Yield curve data missing for one or more tenors")
        if np.any(r <= -0.99):
            raise ValueError("Zero rates must be greater than -99%")
        object.__setattr__(self, "tenors", t)
        object.__setattr__(self, "rates", r)
        if len(self.labels) != t.size:
            object.__setattr__(self, "labels", tuple(f"{x:g}Y" for x in t))

    @classmethod
    def from_series(cls, series: pd.Series) -> "YieldCurve":
        """Build from a Series indexed by tenor labels (e.g. '3M', '10Y')."""
        years = np.array([tenor_to_years(str(lbl)) for lbl in series.index])
        order = np.argsort(years)
        return cls(years[order], series.to_numpy(dtype=float)[order], tuple(str(series.index[i]) for i in order))

    def zero_rate(self, t: np.ndarray | float) -> np.ndarray:
        return interpolation_weights(self.tenors, t) @ self.rates

    def discount_factor(self, t: np.ndarray | float) -> np.ndarray:
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        return (1.0 + self.zero_rate(tt)) ** (-tt)

    def continuous_rate(self, t: np.ndarray | float) -> np.ndarray:
        """Continuously compounded equivalent ``ln(1 + z(t))`` (used for option pricing)."""
        return np.log1p(self.zero_rate(t))

    def shifted(self, shifts: np.ndarray | float | dict[str, float]) -> "YieldCurve":
        """Return a new curve with node shifts applied (decimal, e.g. 0.01 = +100 bp)."""
        if isinstance(shifts, dict):
            vec = np.array([shifts.get(lbl, 0.0) for lbl in self.labels], dtype=float)
        else:
            vec = np.broadcast_to(np.asarray(shifts, dtype=float), self.rates.shape)
        return YieldCurve(self.tenors, self.rates + vec, self.labels)

    def to_series(self) -> pd.Series:
        return pd.Series(self.rates, index=list(self.labels), name="rate")
