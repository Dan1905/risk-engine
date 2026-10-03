"""Yield-curve shock builders defined on the key-rate nodes.

Non-parallel shocks are specified at two pillars and interpolated linearly in
maturity between them::

    shift(t) = short                          t <= short_pillar (2Y)
             = long                           t >= long_pillar  (10Y)
             = linear interpolation           in between

Because shocks are applied node by node, bonds are repriced with the full
shocked curve (or approximated with their key-rate durations) rather than with
a single yield shock.
"""

from __future__ import annotations

import numpy as np

from analytics.yield_curve import TENOR_LABELS, TENOR_YEARS, tenor_to_years

BP = 1e-4
SHORT_PILLAR = 2.0
LONG_PILLAR = 10.0


def parallel_shift(bps: float, labels: tuple[str, ...] = TENOR_LABELS) -> dict[str, float]:
    """Same shift (decimal) at every node."""
    return {lbl: bps * BP for lbl in labels}


def twist(
    short_bps: float,
    long_bps: float,
    labels: tuple[str, ...] = TENOR_LABELS,
    short_pillar: float = SHORT_PILLAR,
    long_pillar: float = LONG_PILLAR,
) -> dict[str, float]:
    """Two-pillar curve shock (steepener / flattener / inversion)."""
    years = np.array([tenor_to_years(lbl) for lbl in labels])
    frac = np.clip((years - short_pillar) / (long_pillar - short_pillar), 0.0, 1.0)
    shifts = (short_bps + (long_bps - short_bps) * frac) * BP
    return {lbl: float(v) for lbl, v in zip(labels, shifts)}


def steepener(short_bps: float = 50, long_bps: float = 150) -> dict[str, float]:
    return twist(short_bps, long_bps)


def flattener(short_bps: float = 150, long_bps: float = 50) -> dict[str, float]:
    return twist(short_bps, long_bps)


def inversion(short_bps: float = 150, long_bps: float = -50) -> dict[str, float]:
    return twist(short_bps, long_bps)


def key_rate_shock(tenor: str, bps: float, labels: tuple[str, ...] = TENOR_LABELS) -> dict[str, float]:
    """Shock a single node (used for KRD illustration)."""
    if tenor not in labels:
        raise ValueError(f"Unknown tenor {tenor}")
    return {lbl: (bps * BP if lbl == tenor else 0.0) for lbl in labels}


def shock_vector(rates_shock: float, curve_shocks: dict[str, float], labels: tuple[str, ...] = TENOR_LABELS) -> np.ndarray:
    """Combine a parallel shift and node shocks into a node vector (decimal).

    Raises:
        ValueError: if ``curve_shocks`` references a tenor not on the curve
            ("Yield curve data missing for selected scenario").
    """
    unknown = set(curve_shocks) - set(labels)
    if unknown:
        raise ValueError(f"Yield curve data missing for selected scenario (unknown tenors: {sorted(unknown)})")
    return np.array([rates_shock + curve_shocks.get(lbl, 0.0) for lbl in labels], dtype=float)


def describe(shifts: dict[str, float]) -> str:
    """Compact text description, e.g. '3M +50bp ... 30Y +150bp'."""
    return ", ".join(f"{k} {v / BP:+.0f}bp" for k, v in shifts.items())


__all__ = [
    "BP",
    "TENOR_YEARS",
    "describe",
    "flattener",
    "inversion",
    "key_rate_shock",
    "parallel_shift",
    "shock_vector",
    "steepener",
    "twist",
]
