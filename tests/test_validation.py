from datetime import date

import numpy as np
import pandas as pd
import pytest

from analytics.returns import clean_price_frame, compute_returns, log_returns, simple_returns
from models.market_data import MarketDataPoint
from models.portfolio import PortfolioValidationError, build_portfolio, validate_position_records
from models.positions import Position, fx_usd_per_unit
from models.risk import RiskConfiguration, StressScenario

VD = date(2026, 9, 30)


def _rec(**kw):
    base = {"position_id": "P1", "asset_type": "EQUITY", "ticker": "AAPL", "quantity": 10, "price": 100.0}
    base.update(kw)
    return base


def test_valid_equity():
    rep = validate_position_records([_rec()], VD)
    assert rep.is_valid and rep.positions[0].instrument_id == "AAPL"


def test_missing_option_volatility():
    rec = _rec(position_id="POS008", asset_type="OPTION", strike=225, option_type="CALL", maturity="2027-01-15")
    rep = validate_position_records([rec], VD)
    assert rep.errors == {"POS008": ["Missing volatility for OPTION position."]}


def test_bond_maturity_before_valuation():
    rec = _rec(position_id="POS014", asset_type="BOND", ticker="US10Y", maturity="2020-01-01", coupon=0.04)
    rep = validate_position_records([rec], VD)
    assert rep.errors["POS014"] == ["Maturity date is earlier than valuation date."]


def test_unsupported_asset_type():
    rep = validate_position_records([_rec(position_id="POS022", asset_type="CRYPTO")], VD)
    assert rep.errors["POS022"] == ["Unsupported asset type: CRYPTO."]


def test_option_expiry_before_valuation():
    rec = _rec(asset_type="OPTION", strike=100, option_type="PUT", maturity="2026-09-01", volatility=0.2)
    rep = validate_position_records([rec], VD)
    assert rep.errors["P1"] == ["Option maturity must be greater than valuation date."]


@pytest.mark.parametrize(
    "kw,fragment",
    [
        ({"quantity": 0}, "non-zero"),
        ({"price": -5}, "greater than 0"),
        ({"currency": "XYZ"}, "Unsupported currency"),
        ({"asset_type": "OPTION", "strike": -1, "option_type": "CALL", "maturity": "2027-01-01", "volatility": 0.2}, "greater than 0"),
        ({"asset_type": "OPTION", "strike": 100, "option_type": "CALL", "maturity": "2027-01-01", "volatility": 9.0}, "outside the supported range"),
        ({"asset_type": "OPTION", "strike": 100, "option_type": "SWAPTION", "maturity": "2027-01-01", "volatility": 0.2}, "Invalid option type"),
        ({"asset_type": "FX", "ticker": "EURGBP"}, "quoted against USD"),
        ({"position_id": "bad id!"}, "Invalid position ID"),
        ({"asset_type": "BOND", "maturity": "not-a-date", "coupon": 0.04}, "date"),
    ],
)
def test_field_validation(kw, fragment):
    rep = validate_position_records([_rec(**kw)], VD)
    assert not rep.is_valid
    assert any(fragment.lower() in m.lower() for msgs in rep.errors.values() for m in msgs)


def test_multiple_errors_aggregated_and_formatted():
    recs = [
        _rec(position_id="POS008", asset_type="OPTION", strike=225, option_type="CALL", maturity="2027-01-15"),
        _rec(position_id="POS014", asset_type="BOND", maturity="2020-01-01", coupon=0.04),
        _rec(position_id="POS022", asset_type="CRYPTO"),
        _rec(position_id="POS023"),
    ]
    with pytest.raises(PortfolioValidationError) as exc:
        build_portfolio(recs, portfolio_id="X", name="X", valuation_date=VD)
    text = str(exc.value)
    assert text.startswith("Portfolio validation failed")
    for fragment in ("POS008:", "POS014:", "POS022:", "Unsupported asset type: CRYPTO"):
        assert fragment in text
    assert "POS023" not in text


def test_duplicate_ids_rejected():
    rep = validate_position_records([_rec(), _rec()], VD)
    assert "Duplicate position ID" in rep.errors["P1"][0]


def test_nan_fields_treated_as_missing():
    rec = _rec(maturity=np.nan, coupon=np.nan, strike=np.nan)
    assert validate_position_records([rec], VD).is_valid


def test_fx_conventions():
    p = Position.model_validate({"position_id": "F", "asset_type": "FX", "ticker": "USDINR", "quantity": 1e6, "price": 88.0})
    assert p.exposure_currency == "INR"
    assert fx_usd_per_unit("USDINR", 88.0) == pytest.approx(1 / 88.0)
    assert fx_usd_per_unit("EURUSD", 1.17) == pytest.approx(1.17)


def test_market_data_point():
    pt = MarketDataPoint.model_validate({"date": "2026-09-30", "instrument_id": "aapl", "asset_type": "equity", "price": 10, "yield": np.nan})
    assert pt.instrument_id == "AAPL" and pt.yield_ is None
    with pytest.raises(ValueError):
        MarketDataPoint.model_validate({"date": "2026-09-30", "instrument_id": "X", "asset_type": "EQUITY", "price": 0})


def test_risk_configuration_and_scenario_validation():
    with pytest.raises(ValueError):
        RiskConfiguration(confidence=1.0)
    with pytest.raises(ValueError):
        RiskConfiguration(decay_factor=1.0)
    with pytest.raises(ValueError):
        StressScenario(scenario_id="X", name="X", equity_shock=-1.5)
    s = StressScenario(scenario_id="GRO", name="Global Risk-Off", equity_shock=-0.2, rates_shock=0.015, fx_shock=0.1, volatility_shock=0.15)
    assert s.basis_for("equity") == "HYPOTHETICAL_ASSUMPTION"


def test_price_cleaning_and_returns():
    idx = pd.to_datetime(["2025-01-06", "2025-01-07", "2025-01-07", "2025-01-08", "2025-01-11", "2025-01-13"])
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0, -1.0, 50.0, 104.0]}, index=idx)
    clean = clean_price_frame(prices, max_ffill=1)
    assert clean.index.is_unique and clean.index.is_monotonic_increasing
    assert pd.Timestamp("2025-01-11") not in clean.index  # Saturday dropped
    assert clean.loc["2025-01-07", "A"] == 102.0  # last duplicate wins
    assert clean.loc["2025-01-08", "A"] == 102.0  # invalid price forward-filled
    s = simple_returns(clean["A"])
    l = log_returns(clean["A"])
    np.testing.assert_allclose(np.expm1(l), s)
    with pytest.raises(ValueError):
        compute_returns(clean, "bogus")
