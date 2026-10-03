"""Historical stress scenarios.

Two kinds are supported:

* **Library scenarios** (2008 GFC, 2020 COVID) defined in
  ``config/scenarios.yaml`` as rounded, index-level observations. Each input is
  tagged as a historical observation or a modelling assumption (for example
  mapping the VIX change onto single-name implied vols, or scaling the index
  move by each stock's beta).

* **Replay scenarios** built from the engine's own stored market history: the
  cumulative move of every factor between two dates (per-ticker equity
  returns, per-currency FX returns, per-node curve and spread changes,
  per-underlying implied-vol level, skew and term-structure changes) is applied to today's portfolio as one
  instantaneous shock. Because the moves are joint and instrument-specific,
  a replay preserves the dispersion and co-movement that actually occurred.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from analytics import risk_factors as rf
from models.risk import ShockBasis, StressScenario

if TYPE_CHECKING:  # avoid a runtime dependency of the stress package on the service layer
    from services.market_data_service import MarketHistory

HISTORICAL_SCENARIO_IDS: tuple[str, ...] = ("GFC_2008", "COVID_2020")
OBSERVED = ShockBasis.OBSERVED


def historical_scenarios(scenarios: list[StressScenario]) -> list[StressScenario]:
    """Filter the scenario library to historical-crisis scenarios."""
    return [s for s in scenarios if s.category == "HISTORICAL"]


def describe_window(scenario: StressScenario) -> str:
    return f"{scenario.name} ({scenario.window})" if scenario.window else scenario.name


def snap_to_history(history: "MarketHistory", d: date) -> pd.Timestamp:
    """Latest history date on or before ``d``.

    Raises:
        ValueError: if ``d`` precedes the first history date.
    """
    pos = history.dates.searchsorted(pd.Timestamp(d), side="right") - 1
    if pos < 0:
        raise ValueError("No market data available on or before the selected start date.")
    return history.dates[pos]


def replay_window_scenario(history: "MarketHistory", start: date, end: date, name: str | None = None) -> StressScenario:
    """Scenario replaying the stored market moves from the close of ``start`` to the close of ``end``.

    Args:
        history: market history (never contains data after its own end date).
        start, end: window bounds; each is snapped to the latest available
            trading day on or before it.

    Raises:
        ValueError: if the window is empty or extends beyond the history.
    """
    s, e = snap_to_history(history, start), snap_to_history(history, end)
    if e <= s:
        raise ValueError("Replay window end date must be after the start date.")
    i0, i1 = int(history.dates.get_loc(s)), int(history.dates.get_loc(e))
    moves = rf.aggregate_horizon(history.daily_moves(i1, i1 - i0), i1 - i0).iloc[-1]

    equity, fx, vol, curve, credit, skew, term = {}, {}, {}, {}, {}, {}, {}
    for col, value in moves.items():
        parts = str(col).split(rf.SEP)
        kind = parts[0]
        if kind == rf.EQ:
            equity[parts[1]] = float(value)
        elif kind == rf.FX:
            fx[parts[1]] = float(value)
        elif kind == rf.VOL:
            vol[parts[1]] = float(value)
        elif kind == rf.VOLSKEW:
            skew[parts[1]] = float(value)
        elif kind == rf.VOLTERM:
            term[parts[1]] = float(value)
        elif kind == rf.CURVE and parts[1] == rf.GOVT_CURVE:
            curve[parts[2]] = float(value)
        elif kind == rf.CURVE:
            credit[parts[2]] = credit.get(parts[2], 0.0) + float(value)
    n_days = i1 - i0
    label = name or f"Replay {s:%d %b %Y} to {e:%d %b %Y}"
    return StressScenario(
        scenario_id=f"REPLAY_{s:%Y%m%d}_{e:%Y%m%d}",
        name=label,
        category="HISTORICAL",
        window=f"{s:%d-%b-%Y} to {e:%d-%b-%Y} ({n_days} trading days)",
        description=(
            "Replay of the stored market history: the cumulative move of every equity, currency, curve node, "
            "credit-spread node and implied-vol level over the window, applied to today's portfolio as one shock."
        ),
        equity_shock=float(pd.Series(equity).mean()) if equity else 0.0,
        equity_overrides=equity,
        fx_shocks=fx,
        curve_shocks=curve,
        credit_curve_shocks=credit,
        vol_overrides=vol,
        vol_skew_shocks=skew,
        vol_term_shocks=term,
        volatility_shock=float(pd.Series(vol).mean()) if vol else 0.0,
        shock_basis={k: OBSERVED for k in ("equity", "rates", "fx", "volatility", "credit")},
        source_notes=(
            "Moves are observed in the stored market data for this window (synthetic in the sample dataset). "
            "The average equity and vol moves shown are summaries; each instrument receives its own observed move."
        ),
    )


def worst_window(daily_pnl: pd.Series, days: int) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """The ``days``-day window with the largest cumulative loss in a daily P&L series.

    Returns:
        (start, end, cumulative P&L) where ``start`` is the close *before* the
        first day of the window, so ``replay_window_scenario(start, end)``
        covers exactly those ``days`` moves.

    Raises:
        ValueError: if the series has fewer than ``days + 1`` observations.
    """
    if days < 1:
        raise ValueError("Window length must be at least one day.")
    if len(daily_pnl) < days + 1:
        raise ValueError("Insufficient historical observations for the selected window length.")
    rolling = daily_pnl.rolling(days).sum().iloc[days:]  # drop windows whose prior close is not in the series
    end = rolling.idxmin()
    pos = int(daily_pnl.index.get_loc(end))
    return pd.Timestamp(daily_pnl.index[pos - days]), pd.Timestamp(end), float(rolling.loc[end])
