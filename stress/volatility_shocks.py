"""Volatility stress conventions.

Sticky strike
    The implied vol of each strike is unchanged when spot moves, so an option
    is repriced with its own vol plus the scenario's vol shift.

Sticky delta
    Implied vol is attached to moneyness (approximately to delta). When spot
    moves, a fixed-strike option slides along the smile, so its vol becomes
    ``f(K / S_new, T)`` plus the shift. With an equity skew, a sell-off lifts
    the vol of every fixed strike under sticky strike only through the explicit
    shock, whereas under sticky delta a call's vol falls (it moves to higher
    moneyness where the skewed smile is lower) and a put's vol rises less.

The repricing itself lives in ``analytics.revaluation`` (single pricing path);
this module provides the scenario builders and the user-facing descriptions.
"""

from __future__ import annotations

from analytics.volatility_surface import sticky_delta_vol, sticky_strike_vol
from models.risk import ShockBasis, StressScenario

VOL_METHODS: dict[str, str] = {
    "STICKY_STRIKE": "Sticky strike: each option keeps its strike's implied vol; only the vol shock is applied.",
    "STICKY_DELTA": (
        "Sticky delta: implied vol follows moneyness, so options slide along the smile as spot moves; "
        "the vol shock is applied on top."
    ),
}


def vol_shock_scenario(points: float) -> StressScenario:
    """Parallel implied-vol shock of ``points`` (0.05 = +5 vols) on every underlying."""
    return StressScenario(
        scenario_id=f"VOL_{points * 100:+.0f}",
        name=f"Vol {points * 100:+.0f} pts",
        category="VOLATILITY",
        description=f"Parallel {points * 100:+.0f} vol-point shift of all implied volatility surfaces.",
        volatility_shock=points,
        shock_basis={"volatility": ShockBasis.ASSUMED},
    )


def describe_method(method: str) -> str:
    return VOL_METHODS.get(method, method)


__all__ = ["VOL_METHODS", "describe_method", "sticky_delta_vol", "sticky_strike_vol", "vol_shock_scenario"]
