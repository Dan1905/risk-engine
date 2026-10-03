"""Equity betas to a market index.

    beta_i = Cov(r_i, r_m) / Var(r_m)

The market return ``r_m`` is the equally weighted average of the supplied
equity returns (the engine holds no index series, so the cross-section of
available equities stands in for the market). By construction the equally
weighted average of the betas is exactly 1, so mapping an index-level shock
through betas preserves the average move while adding dispersion: high-beta
names fall further than low-beta, defensive ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_betas(returns: pd.DataFrame, weights: pd.Series | None = None) -> pd.Series:
    """Betas of each column of ``returns`` to the (weighted) cross-sectional market return.

    Args:
        returns: T x N daily returns.
        weights: optional index weights (default equal weights); normalised to sum to 1.

    Edge cases:
        Fewer than two observations, or a market with zero variance, gives
        beta 1 for every name (the shock is then applied uniformly).
    """
    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    r = returns.astype(float).fillna(0.0)
    if weights is None:
        w = np.full(r.shape[1], 1.0 / r.shape[1])
    else:
        w = weights.reindex(r.columns).fillna(0.0).to_numpy(dtype=float)
        w = w / w.sum() if w.sum() else np.full(r.shape[1], 1.0 / r.shape[1])
    market = r.to_numpy() @ w
    var = float(np.var(market, ddof=1)) if len(market) > 1 else 0.0
    if var <= 0:
        return pd.Series(1.0, index=r.columns, name="beta")
    centred = r.to_numpy() - r.to_numpy().mean(axis=0)
    cov = centred.T @ (market - market.mean()) / (len(market) - 1)
    return pd.Series(cov / var, index=r.columns, name="beta")
