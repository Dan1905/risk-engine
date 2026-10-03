"""Key Rate Duration (KRD).

A key-rate shift bumps a single curve node and, through linear interpolation,
the rates of neighbouring maturities by a "tent" that is 1 at the node and 0 at
the adjacent nodes. For node k::

    KRD_k = -(P(z_k + h) - P(z_k - h)) / (2 h P)          (central difference)

Properties:
* sum_k KRD_k equals the effective duration for a parallel shift (the tent
  weights sum to one everywhere), which the unit tests verify.
* Dollar key-rate DV01 ``KR01_k = KRD_k * MV * 0.0001`` is additive across
  bonds, so portfolio key-rate risk is the sum of position KR01s.
* A curve scenario ``dz`` has first-order P&L ``-MV * sum_k KRD_k dz_k``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.bonds import BP, price_from_zero_matrix
from analytics.yield_curve import YieldCurve, interpolation_weights


def key_rate_durations(
    curve: YieldCurve, times: np.ndarray, amounts: np.ndarray, spread: float = 0.0, bump: float = BP
) -> pd.Series:
    """KRD for every curve node (years), labelled by tenor.

    Args:
        curve: base zero curve.
        times, amounts: cash-flow times (years) and amounts.
        spread: constant spread added to the curve.
        bump: node shift used for the central difference (default 1 bp).
    """
    w = interpolation_weights(curve.tenors, times)  # (n_cf, n_nodes)
    z0 = w @ curve.rates + spread
    p0 = float(np.sum(amounts * (1.0 + z0) ** (-times)))
    up = price_from_zero_matrix(z0[None, :] + bump * w.T, times, amounts)
    dn = price_from_zero_matrix(z0[None, :] - bump * w.T, times, amounts)
    krd = -(up - dn) / (2.0 * bump * p0)
    return pd.Series(krd, index=list(curve.labels), name="KRD")


def key_rate_dv01(krd: pd.Series, market_value: float) -> pd.Series:
    """Dollar value of a 1 bp move at each key rate (positive = gains when rates fall)."""
    return krd * market_value * BP


def aggregate_krd(krd_by_position: pd.DataFrame, market_values: pd.Series) -> pd.DataFrame:
    """Portfolio-level key-rate risk.

    Args:
        krd_by_position: positions x tenors KRD matrix.
        market_values: position market values (USD), same index.

    Returns:
        Frame indexed by tenor with ``KR01`` (USD per bp, additive) and
        ``KRD`` (market-value-weighted KRD of the fixed-income book).
    """
    mv = market_values.reindex(krd_by_position.index).fillna(0.0)
    kr01 = krd_by_position.mul(mv, axis=0).sum(axis=0) * BP
    total = mv.sum()
    weighted = krd_by_position.mul(mv, axis=0).sum(axis=0) / total if total else kr01 * np.nan
    return pd.DataFrame({"KR01": kr01, "KRD": weighted})


def krd_pnl_approximation(krd: pd.Series, market_value: float, node_shifts: pd.Series | dict[str, float]) -> float:
    """First-order P&L from key-rate sensitivities: ``-MV * sum_k KRD_k dz_k``."""
    shifts = pd.Series(node_shifts, dtype=float).reindex(krd.index).fillna(0.0)
    return float(-market_value * float((krd * shifts).sum()))
