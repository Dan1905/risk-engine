"""Expected Shortfall (conditional VaR).

VaR answers "what loss is exceeded in roughly (1-c) of scenarios?". Expected
Shortfall answers "given that we are in that worst (1-c) tail, what is the
average loss?". ES is sub-additive (coherent) and sensitive to the shape of the
tail beyond VaR, which VaR ignores.

Historical:    ES_c = mean{ L_i : L_i >= VaR_c }
Weighted:      ES_c = sum_{tail} w_i L_i / sum_{tail} w_i
Parametric:    ES_c = sigma * phi(z_c) / (1 - c)            (normal, zero mean)

where ``L = -P&L`` and the tail is defined with the same quantile convention as
:mod:`analytics.var`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from analytics.ewma import ewma_weights
from analytics.var import (
    clean_pnl,
    historical_var,
    minimum_observations,
    validate_confidence,
    weighted_historical_var,
)


def historical_es(pnl: pd.Series | np.ndarray, confidence: float = 0.99) -> float:
    """Average loss over scenarios whose loss is at least the historical VaR.

    Edge cases:
        The tail always contains at least the worst scenario, so the result is
        defined whenever VaR is.
    """
    c = validate_confidence(confidence)
    x = clean_pnl(pnl, minimum_observations(c))
    var = historical_var(x, c)
    losses = -x
    tail = losses[losses >= var - 1e-12]
    return float(tail.mean())


def weighted_historical_es(pnl: pd.Series | np.ndarray, confidence: float = 0.99, lam: float = 0.94) -> float:
    """Exponentially weighted ES: weight-averaged loss over the weighted tail.

    Args:
        pnl: scenario P&L ordered oldest -> newest.
    """
    c = validate_confidence(confidence)
    x = clean_pnl(pnl, 2)
    w = ewma_weights(x.size, lam)
    var = weighted_historical_var(x, c, lam)
    losses = -x
    mask = losses >= var - 1e-12
    return float(np.sum(w[mask] * losses[mask]) / np.sum(w[mask]))


def parametric_es(sigma: float, confidence: float = 0.99, portfolio_value: float = 1.0, holding_period: int = 1) -> float:
    """Normal-distribution ES: ``sigma * phi(z) / (1 - c) * V * sqrt(h)``."""
    c = validate_confidence(confidence)
    z = norm.ppf(c)
    return float(sigma * norm.pdf(z) / (1.0 - c) * portfolio_value * np.sqrt(holding_period))
