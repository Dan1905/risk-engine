"""Shared fixtures: small deterministic market states and portfolios."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from analytics.revaluation import MarketState
from analytics.volatility_surface import SmileVolSurface
from analytics.yield_curve import TENOR_LABELS, TENOR_YEARS, YieldCurve
from models.portfolio import Portfolio
from models.positions import Position

VAL_DATE = date(2026, 9, 30)


@pytest.fixture
def flat_curve() -> YieldCurve:
    return YieldCurve(TENOR_YEARS, np.full(TENOR_YEARS.size, 0.04), TENOR_LABELS)


@pytest.fixture
def base_state(flat_curve: YieldCurve) -> MarketState:
    return MarketState(
        date=VAL_DATE,
        equity_prices={"AAPL": 200.0, "MSFT": 400.0, "SAP": 150.0},
        fx_rates={"USD": 1.0, "EUR": 1.10, "INR": 1 / 88.0},
        curve=flat_curve,
        spread_curves={"USD_IG": YieldCurve(TENOR_YEARS, np.full(TENOR_YEARS.size, 0.01), TENOR_LABELS)},
        atm_vols={"AAPL": 0.25},
        vol_surfaces={"AAPL": SmileVolSurface(atm=0.25, skew=-0.15, curvature=0.8)},
    )


def make_positions() -> list[Position]:
    ctx = {"valuation_date": VAL_DATE}
    rows = [
        {"position_id": "EQ1", "asset_type": "EQUITY", "ticker": "AAPL", "quantity": 1000, "price": 200.0},
        {"position_id": "EQ2", "asset_type": "EQUITY", "ticker": "SAP", "quantity": 500, "price": 150.0, "currency": "EUR"},
        {
            "position_id": "BD1",
            "asset_type": "BOND",
            "ticker": "UST10",
            "quantity": 1_000_000,
            "price": 98.0,
            "maturity": "2036-08-15",
            "coupon": 0.04,
            "coupon_frequency": 2,
        },
        {
            "position_id": "BD2",
            "asset_type": "BOND",
            "ticker": "CORP5",
            "quantity": 500_000,
            "price": 101.0,
            "maturity": "2031-06-30",
            "coupon": 0.055,
            "coupon_frequency": 2,
            "credit_curve": "USD_IG",
        },
        {"position_id": "FX1", "asset_type": "FX", "ticker": "EURUSD", "quantity": 200_000, "price": 1.10},
        {"position_id": "FX2", "asset_type": "FX", "ticker": "USDINR", "quantity": 10_000_000, "price": 88.0},
        {
            "position_id": "OP1",
            "asset_type": "OPTION",
            "ticker": "AAPL",
            "quantity": 20,
            "price": 12.0,
            "maturity": "2027-03-19",
            "strike": 210.0,
            "option_type": "CALL",
            "volatility": 0.26,
        },
        {"position_id": "CASH", "asset_type": "CASH", "ticker": "USD", "quantity": 250_000},
    ]
    return [Position.model_validate(r, context=ctx) for r in rows]


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(portfolio_id="TEST", name="Test", valuation_date=VAL_DATE, positions=make_positions())
