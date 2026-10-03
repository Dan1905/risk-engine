"""Portfolio schema and aggregated validation.

``validate_position_records`` validates every row independently and collects
all problems, so a user sees the complete list of issues in one pass::

    Portfolio validation failed

    POS008:
    Missing volatility for OPTION position.

    POS014:
    Maturity date is earlier than valuation date.

Invalid portfolios never reach the risk engine: ``build_portfolio`` raises
``PortfolioValidationError`` if any row fails.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from models.positions import AssetType, Position


class Portfolio(BaseModel):
    """A validated portfolio as of a valuation date."""

    model_config = ConfigDict(frozen=True)

    portfolio_id: str
    name: str
    base_currency: str = "USD"
    valuation_date: date
    positions: list[Position] = Field(min_length=1)

    @field_validator("base_currency")
    @classmethod
    def _base_ccy(cls, v: str) -> str:
        if v.upper() != "USD":
            raise ValueError("Only USD base currency is supported in this version.")
        return v.upper()

    @model_validator(mode="after")
    def _unique_ids(self) -> "Portfolio":
        ids = [p.position_id for p in self.positions]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"Duplicate position IDs: {', '.join(dupes)}")
        return self

    def positions_by_type(self, asset_type: AssetType) -> list[Position]:
        return [p for p in self.positions if p.asset_type is asset_type]

    def get(self, position_id: str) -> Position:
        for p in self.positions:
            if p.position_id == position_id:
                return p
        raise KeyError(position_id)

    @property
    def key(self) -> str:
        """Deterministic content hash; used as a cache key."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def to_frame(self) -> pd.DataFrame:
        """Flat tabular view of the positions (one row per position)."""
        rows = []
        for p in self.positions:
            rows.append(
                {
                    "position_id": p.position_id,
                    "asset_type": p.asset_type.value,
                    "ticker": p.ticker,
                    "instrument_id": p.instrument_id,
                    "quantity": p.quantity,
                    "price": p.price,
                    "currency": p.currency,
                    "exposure_currency": p.exposure_currency,
                    "maturity": (p.bond.maturity_date if p.bond else p.option.expiry_date if p.option else None),
                    "coupon": p.bond.coupon_rate if p.bond else None,
                    "strike": p.option.strike if p.option else None,
                    "option_type": p.option.option_type.value if p.option else None,
                    "volatility": p.option.volatility if p.option else None,
                }
            )
        return pd.DataFrame(rows)


@dataclass
class ValidationReport:
    """Outcome of validating raw position records."""

    positions: list[Position] = field(default_factory=list)
    errors: dict[str, list[str]] = field(default_factory=dict)
    warnings: dict[str, list[str]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors and bool(self.positions)

    def add_error(self, position_id: str, message: str) -> None:
        self.errors.setdefault(position_id, []).append(message)

    def add_warning(self, position_id: str, message: str) -> None:
        self.warnings.setdefault(position_id, []).append(message)


class PortfolioValidationError(Exception):
    """Raised when a portfolio fails validation; carries per-position messages."""

    def __init__(self, errors: dict[str, list[str]], warnings: dict[str, list[str]] | None = None):
        self.errors = errors
        self.warnings = warnings or {}
        super().__init__(self.format())

    def format(self) -> str:
        lines = ["Portfolio validation failed", ""]
        for pid, msgs in self.errors.items():
            lines.append(f"{pid}:")
            lines.extend(msgs)
            lines.append("")
        return "\n".join(lines).rstrip()


def _friendly_messages(exc: ValidationError) -> list[str]:
    """Turn Pydantic errors into short user-facing messages."""
    messages: list[str] = []
    for err in exc.errors():
        msg = str(err.get("msg", ""))
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, ") :]
        loc = [str(x) for x in err.get("loc", ()) if not isinstance(x, int)]
        # Drop union/tag noise from nested locations.
        loc = [x for x in loc if x not in {"function-after[_consistency(), Position]"}]
        parts = [m.strip() for m in msg.split(" | ") if m.strip()]
        for part in parts:
            if loc and err.get("type") != "value_error":
                messages.append(f"{'.'.join(loc)}: {part}")
            else:
                messages.append(part if part.endswith(".") else f"{part}.")
    return messages


def validate_position_records(records: list[dict[str, Any]], valuation_date: date) -> ValidationReport:
    """Validate raw (flat) position records against the schema.

    Args:
        records: one dict per position, e.g. rows from a CSV upload.
        valuation_date: maturities/expiries must be after this date.

    Returns:
        ValidationReport with the valid ``Position`` objects plus any errors
        and warnings keyed by position id (or ``ROW n`` when the id is absent).
    """
    report = ValidationReport()
    seen: set[str] = set()
    for idx, record in enumerate(records, start=1):
        raw_id = record.get("position_id")
        pid = str(raw_id).strip() if raw_id is not None and str(raw_id).strip() not in {"", "nan"} else f"ROW {idx}"
        if pid in seen:
            report.add_error(pid, f"Duplicate position ID: {pid}.")
            continue
        seen.add(pid)
        try:
            position = Position.model_validate(record, context={"valuation_date": valuation_date})
        except ValidationError as exc:
            for message in _friendly_messages(exc):
                report.add_error(pid, message)
            continue
        report.positions.append(position)
    return report


def build_portfolio(
    records: list[dict[str, Any]],
    *,
    portfolio_id: str,
    name: str,
    valuation_date: date,
    base_currency: str = "USD",
) -> tuple[Portfolio, ValidationReport]:
    """Validate records and assemble a ``Portfolio``.

    Raises:
        PortfolioValidationError: if any record is invalid or no positions remain.
    """
    report = validate_position_records(records, valuation_date)
    if report.errors:
        raise PortfolioValidationError(report.errors, report.warnings)
    if not report.positions:
        raise PortfolioValidationError({"PORTFOLIO": ["Portfolio contains no positions."]})
    portfolio = Portfolio(
        portfolio_id=portfolio_id,
        name=name,
        base_currency=base_currency,
        valuation_date=valuation_date,
        positions=report.positions,
    )
    return portfolio, report
