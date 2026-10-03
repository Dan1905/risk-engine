"""Pydantic schemas for positions and their instrument terms.

A ``Position`` always carries the common fields (id, asset type, ticker,
quantity, price, currency). Asset-specific terms live in nested ``Bond`` and
``Option`` models so the representation stays normalised instead of relying on
every flat CSV column being populated.

Price conventions
-----------------
EQUITY  local-currency price per share
BOND    dirty price per 100 face; ``quantity`` is the face amount held
FX      quoted pair rate (EURUSD = USD per EUR, USDINR = INR per USD);
        ``quantity`` is the foreign-currency amount held (positive = long)
OPTION  premium per share; ``quantity`` is the number of contracts
CASH    1.0; ``quantity`` is the cash amount in ``currency``

Validation that needs the valuation date (maturity / expiry checks) reads it
from the Pydantic validation context: ``Position.model_validate(row,
context={"valuation_date": date(...)})``.
"""

from __future__ import annotations

import math
import re
from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

SUPPORTED_CURRENCIES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "INR", "JPY", "CHF", "CAD", "AUD", "CNY", "HKD", "SGD"}
)
SUPPORTED_COUPON_FREQUENCIES: frozenset[int] = frozenset({1, 2, 4, 12})
MAX_VOLATILITY = 5.0  # 500% annualised; anything above is treated as a data error
POSITION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.]{1,40}$")


class AssetType(str, Enum):
    EQUITY = "EQUITY"
    BOND = "BOND"
    FX = "FX"
    OPTION = "OPTION"
    CASH = "CASH"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _valuation_date(info: ValidationInfo) -> date | None:
    ctx = info.context or {}
    val = ctx.get("valuation_date")
    return val if isinstance(val, date) else None


def parse_fx_pair(ticker: str) -> tuple[str, str]:
    """Split a six-letter FX ticker (e.g. ``EURUSD``) into (base, quote).

    Raises:
        ValueError: if the ticker is not six letters or neither leg is USD.
    """
    t = ticker.strip().upper()
    if len(t) != 6 or not t.isalpha():
        raise ValueError(f"Invalid FX pair '{ticker}': expected six letters such as EURUSD")
    base, quote = t[:3], t[3:]
    if "USD" not in (base, quote) or base == quote:
        raise ValueError(f"Unsupported FX pair '{ticker}': pairs must be quoted against USD")
    for ccy in (base, quote):
        if ccy not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency '{ccy}' in FX pair '{ticker}'")
    return base, quote


def fx_foreign_currency(ticker: str) -> str:
    """Return the non-USD leg of a USD-quoted FX pair."""
    base, quote = parse_fx_pair(ticker)
    return quote if base == "USD" else base


def fx_usd_per_unit(ticker: str, quoted_rate: float) -> float:
    """Convert a quoted pair rate into USD per unit of the foreign currency.

    EURUSD 1.17  -> 1.17 USD per EUR
    USDINR 88.5  -> 1/88.5 USD per INR
    """
    base, _ = parse_fx_pair(ticker)
    if quoted_rate <= 0:
        raise ValueError("FX rate must be positive")
    return 1.0 / quoted_rate if base == "USD" else quoted_rate


class Bond(BaseModel):
    """Fixed-coupon bullet bond terms."""

    model_config = ConfigDict(frozen=True)

    face_value: float = Field(gt=0, description="Face value per bond unit (price is quoted per 100 of this)")
    coupon_rate: float = Field(ge=0.0, lt=1.0, description="Annual coupon rate as a decimal")
    maturity_date: date
    coupon_frequency: int = 2
    credit_curve: str | None = Field(default=None, description="Spread curve id for credit bonds; None = government")

    @field_validator("coupon_frequency")
    @classmethod
    def _freq(cls, v: int) -> int:
        if v not in SUPPORTED_COUPON_FREQUENCIES:
            raise ValueError(f"Unsupported coupon frequency {v}; use one of {sorted(SUPPORTED_COUPON_FREQUENCIES)}")
        return v

    @model_validator(mode="after")
    def _maturity_after_valuation(self, info: ValidationInfo) -> "Bond":
        vd = _valuation_date(info)
        if vd is not None and self.maturity_date <= vd:
            raise ValueError("Maturity date is earlier than valuation date.")
        return self


class Option(BaseModel):
    """European equity option terms."""

    model_config = ConfigDict(frozen=True)

    underlying: str
    strike: float = Field(gt=0)
    expiry_date: date
    option_type: OptionType
    volatility: float = Field(gt=0, description="Annualised implied volatility (decimal)")
    contract_multiplier: float = Field(default=100.0, gt=0)
    exercise_style: Literal["EUROPEAN"] = "EUROPEAN"

    @field_validator("underlying")
    @classmethod
    def _underlying(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("Option underlying is required")
        return v

    @field_validator("option_type", mode="before")
    @classmethod
    def _option_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip().upper()
            if v in {"C", "P"}:
                v = "CALL" if v == "C" else "PUT"
            if v not in {"CALL", "PUT"}:
                raise ValueError(f"Invalid option type: {v}. Use CALL or PUT.")
        return v

    @field_validator("volatility")
    @classmethod
    def _vol(cls, v: float) -> float:
        if not math.isfinite(v) or v > MAX_VOLATILITY:
            raise ValueError(f"Implied volatility {v:.2f} is outside the supported range (0, {MAX_VOLATILITY:.0f}].")
        return v

    @model_validator(mode="after")
    def _expiry_after_valuation(self, info: ValidationInfo) -> "Option":
        vd = _valuation_date(info)
        if vd is not None and self.expiry_date <= vd:
            raise ValueError("Option maturity must be greater than valuation date.")
        return self


# Flat CSV columns that feed the nested models.
_BOND_FIELDS = ("maturity", "face_value", "coupon", "coupon_frequency", "credit_curve")
_OPTION_FIELDS = ("maturity", "strike", "option_type", "volatility", "underlying", "contract_multiplier")


class Position(BaseModel):
    """A validated portfolio position."""

    model_config = ConfigDict(frozen=True)

    position_id: str
    asset_type: AssetType
    ticker: str
    quantity: float
    price: float = Field(gt=0)
    currency: str = "USD"
    bond: Bond | None = None
    option: Option | None = None

    # ------------------------------------------------------------------ parsing
    @model_validator(mode="before")
    @classmethod
    def _from_flat_record(cls, data: Any) -> Any:
        """Accept flat CSV-style records and build the nested terms.

        Checks asset-type specific required fields up front so that every
        missing field is reported with a readable message.
        """
        if not isinstance(data, dict):
            return data
        rec = {k: (None if _is_missing(v) else v) for k, v in data.items()}
        raw_type = rec.get("asset_type")
        asset_type = str(raw_type).strip().upper() if raw_type is not None else None
        problems: list[str] = []

        if asset_type == AssetType.CASH.value and rec.get("price") is None:
            rec["price"] = 1.0

        if asset_type == AssetType.BOND.value and rec.get("bond") is None:
            if rec.get("maturity") is None:
                problems.append("Missing maturity for BOND position.")
            if rec.get("coupon") is None:
                problems.append("Missing coupon for BOND position.")
            if not problems:
                bond: dict[str, Any] = {
                    "maturity_date": rec.get("maturity"),
                    "coupon_rate": rec.get("coupon"),
                    "face_value": rec.get("face_value") or 100.0,
                }
                if rec.get("coupon_frequency") is not None:
                    bond["coupon_frequency"] = int(float(rec["coupon_frequency"]))
                if rec.get("credit_curve") is not None:
                    bond["credit_curve"] = rec["credit_curve"]
                rec["bond"] = bond

        if asset_type == AssetType.OPTION.value and rec.get("option") is None:
            if rec.get("volatility") is None:
                problems.append("Missing volatility for OPTION position.")
            if rec.get("strike") is None:
                problems.append("Missing strike for OPTION position.")
            if rec.get("option_type") is None:
                problems.append("Missing option type (CALL/PUT) for OPTION position.")
            if rec.get("maturity") is None:
                problems.append("Missing maturity (expiry date) for OPTION position.")
            if not problems:
                opt: dict[str, Any] = {
                    "underlying": rec.get("underlying") or rec.get("ticker"),
                    "strike": rec.get("strike"),
                    "expiry_date": rec.get("maturity"),
                    "option_type": rec.get("option_type"),
                    "volatility": rec.get("volatility"),
                }
                if rec.get("contract_multiplier") is not None:
                    opt["contract_multiplier"] = rec["contract_multiplier"]
                rec["option"] = opt

        if problems:
            raise ValueError(" | ".join(problems))
        for key in set(_BOND_FIELDS + _OPTION_FIELDS) | {"market_value", "weight"}:
            rec.pop(key, None)
        return rec

    # --------------------------------------------------------------- field checks
    @field_validator("position_id", mode="before")
    @classmethod
    def _position_id(cls, v: Any) -> str:
        if v is None:
            raise ValueError("Position ID is required.")
        v = str(v).strip()
        if not POSITION_ID_PATTERN.match(v):
            raise ValueError(f"Invalid position ID '{v}': use letters, digits, '_', '-' or '.' (max 40).")
        return v

    @field_validator("asset_type", mode="before")
    @classmethod
    def _asset_type(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("Asset type is required.")
        v = str(v).strip().upper()
        if v not in AssetType.__members__:
            raise ValueError(f"Unsupported asset type: {v}")
        return v

    @field_validator("ticker", mode="before")
    @classmethod
    def _ticker(cls, v: Any) -> str:
        if v is None or str(v).strip() == "":
            raise ValueError("Ticker is required.")
        return str(v).strip().upper()

    @field_validator("quantity")
    @classmethod
    def _quantity(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Quantity must be a finite number.")
        if v == 0:
            raise ValueError("Quantity must be non-zero.")
        return v

    @field_validator("price")
    @classmethod
    def _price(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("Price must be a finite number.")
        return v

    @field_validator("currency", mode="before")
    @classmethod
    def _currency(cls, v: Any) -> str:
        v = "USD" if v is None else str(v).strip().upper()
        if v not in SUPPORTED_CURRENCIES:
            raise ValueError(f"Unsupported currency: {v}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "Position":
        problems: list[str] = []
        if self.asset_type == AssetType.FX:
            try:
                parse_fx_pair(self.ticker)
            except ValueError as exc:
                problems.append(str(exc))
        if self.asset_type == AssetType.BOND and self.bond is None:
            problems.append("Bond terms are required for BOND positions.")
        if self.asset_type == AssetType.OPTION and self.option is None:
            problems.append("Option terms are required for OPTION positions.")
        if self.asset_type != AssetType.BOND and self.bond is not None:
            problems.append(f"Bond terms supplied for a {self.asset_type.value} position.")
        if self.asset_type != AssetType.OPTION and self.option is not None:
            problems.append(f"Option terms supplied for a {self.asset_type.value} position.")
        if problems:
            raise ValueError(" | ".join(problems))
        return self

    # ----------------------------------------------------------------- derived
    @property
    def instrument_id(self) -> str:
        """Stable instrument identifier used for persistence and market data lookups."""
        if self.asset_type == AssetType.OPTION and self.option is not None:
            o = self.option
            return f"{o.underlying}_{o.option_type.value[0]}{o.strike:g}_{o.expiry_date:%Y%m%d}"
        if self.asset_type == AssetType.CASH:
            return f"CASH_{self.currency}"
        return self.ticker

    @property
    def exposure_currency(self) -> str:
        """Currency whose moves against USD drive this position's FX risk."""
        if self.asset_type == AssetType.FX:
            return fx_foreign_currency(self.ticker)
        return self.currency

    @property
    def underlying(self) -> str | None:
        return self.option.underlying if self.option is not None else None
