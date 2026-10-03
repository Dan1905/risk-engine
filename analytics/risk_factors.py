"""Market risk-factor naming and transformations.

Every methodology that works on market-factor moves (historical simulation,
volatility filtering, Monte Carlo, multi-day aggregation) shares one
representation: a *factor frame* - a DataFrame with one row per scenario and
one column per factor. Column names encode the factor::

    eq|AAPL              simple return of an equity / option underlying
    fx|EUR               simple return of USD per unit of the currency
    curve|USD_TSY|10Y    absolute change of a zero-curve node (decimal)
    curve|USD_IG|5Y      absolute change of a credit-spread curve node
    vol|AAPL             absolute change of 3M ATM implied vol (level)
    volskew|AAPL         absolute change of the 3M 90%-110% skew
    volterm|AAPL         absolute change of the 1Y-1M ATM term slope
    yield|US10Y          instrument-specific parallel yield shift (stress overrides)

Equity and FX factors are *returns* (they compound over time and cannot fall
below -100%); all other factors are *additive* changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEP = "|"
EQ, FX, CURVE, VOL, VOLSKEW, VOLTERM, YIELD = "eq", "fx", "curve", "vol", "volskew", "volterm", "yield"
RETURN_KINDS: frozenset[str] = frozenset({EQ, FX})
GOVT_CURVE = "USD_TSY"
RETURN_FLOOR = -0.95  # a simulated / scaled return may not wipe out more than 95%

FACTOR_GROUP_LABELS = {
    EQ: "Equity", FX: "FX", CURVE: "Rates & credit curves", VOL: "Implied vol level",
    VOLSKEW: "Implied vol skew", VOLTERM: "Implied vol term structure", YIELD: "Bond yield",
}


def factor(kind: str, *parts: str) -> str:
    """Build a factor column name, e.g. ``factor('curve', 'USD_TSY', '10Y')``."""
    return SEP.join((kind, *parts))


def kind_of(name: str) -> str:
    return name.split(SEP, 1)[0]


def is_return(name: str) -> bool:
    return kind_of(name) in RETURN_KINDS


def return_mask(columns: pd.Index | list[str]) -> np.ndarray:
    """Boolean mask of the columns that are multiplicative returns."""
    return np.array([is_return(c) for c in columns], dtype=bool)


def clip_returns(frame: pd.DataFrame, floor: float = RETURN_FLOOR) -> pd.DataFrame:
    """Floor return factors so that no scenario implies a price at or below zero."""
    mask = return_mask(frame.columns)
    if not mask.any():
        return frame
    out = frame.copy()
    cols = frame.columns[mask]
    out[cols] = out[cols].clip(lower=floor)
    return out


def aggregate_horizon(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Overlapping ``horizon``-day factor moves from daily moves.

    Row t of the result is the move over the ``horizon`` days ending on t:
    returns compound (``prod(1 + r) - 1``), additive factors sum. The first
    ``horizon - 1`` rows are dropped, so n daily rows give n - horizon + 1
    overlapping scenarios.

    Overlapping windows are serially dependent (adjacent scenarios share
    ``horizon - 1`` days), which reduces the effective sample size; they
    nevertheless capture multi-day compounding and non-linear repricing that
    square-root-of-time scaling of 1-day figures cannot.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if horizon == 1:
        return frame
    if len(frame) < horizon:
        raise ValueError(f"Need at least {horizon} daily observations for a {horizon}-day horizon")
    mask = return_mask(frame.columns)
    values = frame.to_numpy(dtype=float)
    out = np.empty((len(frame) - horizon + 1, values.shape[1]))
    if mask.any():
        log_growth = np.log1p(np.clip(values[:, mask], -0.999999, None))
        csum = np.vstack([np.zeros((1, mask.sum())), np.cumsum(log_growth, axis=0)])
        out[:, mask] = np.expm1(csum[horizon:] - csum[:-horizon])
    if (~mask).any():
        csum = np.vstack([np.zeros((1, (~mask).sum())), np.cumsum(values[:, ~mask], axis=0)])
        out[:, ~mask] = csum[horizon:] - csum[:-horizon]
    return pd.DataFrame(out, index=frame.index[horizon - 1 :], columns=frame.columns)


VOL_FACTOR_MODELS = ("LEVEL", "LEVEL_SKEW_TERM")


def select_vol_factors(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Keep only the implied-vol factors used by ``model``.

    ``LEVEL`` keeps the ATM level factor; ``LEVEL_SKEW_TERM`` also keeps the
    skew and term-structure factors.
    """
    if model == "LEVEL_SKEW_TERM":
        return frame
    if model != "LEVEL":
        raise ValueError(f"Unknown volatility factor model '{model}'")
    keep = [c for c in frame.columns if kind_of(c) not in (VOLSKEW, VOLTERM)]
    return frame[keep]
