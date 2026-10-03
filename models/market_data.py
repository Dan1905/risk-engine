"""Pydantic schemas for market data records.

The normalised market-data schema is::

    date | instrument_id | asset_type | price | yield | volatility | currency

with optional equity OHLCV and bond analytics columns. Yield-curve nodes and
volatility-surface points have their own record types.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.positions import SUPPORTED_CURRENCIES, AssetType


def _finite_or_none(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    return v


class MarketDataPoint(BaseModel):
    """One instrument observation on one date."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    date: date
    instrument_id: str
    asset_type: AssetType
    price: float = Field(gt=0)
    yield_: float | None = Field(default=None, alias="yield")
    volatility: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    open: float | None = Field(default=None, gt=0)
    high: float | None = Field(default=None, gt=0)
    low: float | None = Field(default=None, gt=0)
    adj_close: float | None = Field(default=None, gt=0)
    volume: float | None = Field(default=None, ge=0)
    duration: float | None = None
    convexity: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {k: _finite_or_none(v) for k, v in data.items()}
        return data

    @field_validator("instrument_id", mode="before")
    @classmethod
    def _iid(cls, v: Any) -> str:
        if v is None or str(v).strip() == "":
            raise ValueError("instrument_id is required")
        return str(v).strip().upper()

    @field_validator("asset_type", mode="before")
    @classmethod
    def _atype(cls, v: Any) -> Any:
        v = str(v).strip().upper()
        if v not in AssetType.__members__:
            raise ValueError(f"Unsupported asset type: {v}")
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def _ccy(cls, v: Any) -> str:
        v = "USD" if v is None else str(v).strip().upper()
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {v}")
        return v

    @field_validator("volatility")
    @classmethod
    def _vol(cls, v: float | None) -> float | None:
        if v is not None and v > 5.0:
            raise ValueError("volatility above 500% is not plausible")
        return v

    @model_validator(mode="after")
    def _ohlc(self) -> "MarketDataPoint":
        if self.high is not None and self.low is not None and self.high < self.low:
            raise ValueError("high is below low")
        return self


class YieldCurvePoint(BaseModel):
    """One yield-curve node observation (rates as decimals)."""

    model_config = ConfigDict(frozen=True)

    date: date
    curve_id: str
    tenor: str
    tenor_years: float = Field(gt=0)
    rate: float = Field(gt=-0.05, lt=0.5)


class VolSurfacePoint(BaseModel):
    """One implied-volatility surface point."""

    model_config = ConfigDict(frozen=True)

    date: date
    underlying: str
    maturity: str
    maturity_years: float = Field(gt=0)
    moneyness: float = Field(gt=0)
    implied_vol: float = Field(gt=0, le=5.0)
