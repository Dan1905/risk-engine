"""Data-access layer.

All SQL lives here; services receive pandas DataFrames or plain dicts. Writes
use ``INSERT ... ON CONFLICT`` on the natural keys so repeated seeding or
re-running a calculation never duplicates rows. Both the SQLite and the
PostgreSQL dialects support this syntax.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from database.connection import get_session_factory, session_scope
from database.models import (
    AppMetadataRecord,
    BacktestResultRecord,
    Base,
    BondTermsRecord,
    FxTermsRecord,
    InstrumentRecord,
    MarketDataRecord,
    OptionTermsRecord,
    PnLAttributionRecord,
    PortfolioRecord,
    PositionRecord,
    RiskContributionRecord,
    RiskMetricRecord,
    StressResultRecord,
    StressScenarioRecord,
    VolSurfaceRecord,
    YieldCurveRecord,
)

logger = logging.getLogger(__name__)

_CHUNK = 2000


def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.date()
    if v is pd.NaT:
        return None
    return v


def _records(df: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    """DataFrame -> list of dicts with NaN/NaT converted to None and numpy scalars unwrapped."""
    sub = df[list(columns)]
    return [{k: _clean_value(v) for k, v in zip(columns, row)} for row in sub.itertuples(index=False, name=None)]


class RiskRepository:
    """Repository over the risk-engine schema."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._factory: sessionmaker[Session] = get_session_factory(engine)

    # ---------------------------------------------------------------- schema
    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self):
        return session_scope(self._factory)

    # ----------------------------------------------------------------- upsert
    def _insert(self, table):
        dialect = self.engine.dialect.name
        if dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        elif dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:  # pragma: no cover
            raise NotImplementedError(f"Upsert not implemented for dialect {dialect}")
        return insert(table)

    def upsert(
        self,
        model: type[Base],
        rows: list[dict[str, Any]],
        conflict_cols: Sequence[str],
        update_cols: Sequence[str] | None = None,
    ) -> int:
        """Insert rows, skipping (or updating) rows that violate the natural key.

        Returns:
            Number of rows submitted.
        """
        if not rows:
            return 0
        # Unwrap numpy scalars / timestamps: PostgreSQL drivers do not adapt them (SQLite happens to).
        rows = [{k: _clean_value(v) for k, v in row.items()} for row in rows]
        table = model.__table__
        with self.engine.begin() as conn:
            for i in range(0, len(rows), _CHUNK):
                stmt = self._insert(table)
                if update_cols:
                    stmt = stmt.on_conflict_do_update(
                        index_elements=list(conflict_cols), set_={c: stmt.excluded[c] for c in update_cols}
                    )
                else:
                    stmt = stmt.on_conflict_do_nothing(index_elements=list(conflict_cols))
                conn.execute(stmt, rows[i : i + _CHUNK])
        return len(rows)

    # --------------------------------------------------------------- metadata
    def get_meta(self, key: str) -> str | None:
        with self.session() as s:
            rec = s.get(AppMetadataRecord, key)
            return rec.value if rec else None

    def set_meta(self, key: str, value: str) -> None:
        self.upsert(AppMetadataRecord, [{"key": key, "value": value}], ["key"], ["value"])

    def table_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self.session() as s:
            for table in Base.metadata.sorted_tables:
                out[table.name] = int(s.execute(select(func.count()).select_from(table)).scalar_one())
        return out

    # ------------------------------------------------------------ instruments
    def upsert_instruments(self, df: pd.DataFrame) -> int:
        """Upsert instruments and their asset-specific terms from a flat frame."""
        n = self.upsert(
            InstrumentRecord,
            _records(df, ["instrument_id", "ticker", "name", "asset_type", "currency", "sector"]),
            ["instrument_id"],
            ["ticker", "name", "asset_type", "currency", "sector"],
        )
        bonds = df[df["asset_type"] == "BOND"]
        if not bonds.empty:
            b = bonds.assign(maturity_date=pd.to_datetime(bonds["maturity_date"]).dt.date)
            self.upsert(
                BondTermsRecord,
                _records(b, ["instrument_id", "coupon_rate", "maturity_date", "coupon_frequency", "face_value", "credit_curve"]),
                ["instrument_id"],
                ["coupon_rate", "maturity_date", "coupon_frequency", "face_value", "credit_curve"],
            )
        opts = df[df["asset_type"] == "OPTION"]
        if not opts.empty:
            o = opts.assign(expiry_date=pd.to_datetime(opts["expiry_date"]).dt.date)
            self.upsert(
                OptionTermsRecord,
                _records(o, ["instrument_id", "underlying_id", "strike", "expiry_date", "option_type", "contract_multiplier"]),
                ["instrument_id"],
                ["underlying_id", "strike", "expiry_date", "option_type", "contract_multiplier"],
            )
        fx = df[df["asset_type"] == "FX"]
        if not fx.empty:
            self.upsert(
                FxTermsRecord,
                _records(fx, ["instrument_id", "base_currency", "quote_currency"]),
                ["instrument_id"],
                ["base_currency", "quote_currency"],
            )
        return n

    def load_instruments(self) -> pd.DataFrame:
        """Instruments joined with their terms (one row per instrument)."""
        stmt = (
            select(
                InstrumentRecord.instrument_id,
                InstrumentRecord.ticker,
                InstrumentRecord.name,
                InstrumentRecord.asset_type,
                InstrumentRecord.currency,
                InstrumentRecord.sector,
                BondTermsRecord.coupon_rate,
                BondTermsRecord.maturity_date,
                BondTermsRecord.coupon_frequency,
                BondTermsRecord.credit_curve,
                OptionTermsRecord.underlying_id,
                OptionTermsRecord.strike,
                OptionTermsRecord.expiry_date,
                OptionTermsRecord.option_type,
                OptionTermsRecord.contract_multiplier,
                FxTermsRecord.base_currency,
                FxTermsRecord.quote_currency,
            )
            .outerjoin(BondTermsRecord, BondTermsRecord.instrument_id == InstrumentRecord.instrument_id)
            .outerjoin(OptionTermsRecord, OptionTermsRecord.instrument_id == InstrumentRecord.instrument_id)
            .outerjoin(FxTermsRecord, FxTermsRecord.instrument_id == InstrumentRecord.instrument_id)
            .order_by(InstrumentRecord.asset_type, InstrumentRecord.instrument_id)
        )
        with self.engine.connect() as conn:
            return pd.read_sql(stmt, conn)

    # ------------------------------------------------------------- portfolios
    def upsert_portfolio(self, portfolio_id: str, name: str, base_currency: str, valuation_date: date, description: str = "") -> None:
        self.upsert(
            PortfolioRecord,
            [
                {
                    "portfolio_id": portfolio_id,
                    "name": name,
                    "base_currency": base_currency,
                    "valuation_date": valuation_date,
                    "description": description,
                }
            ],
            ["portfolio_id"],
            ["name", "base_currency", "valuation_date", "description"],
        )

    def list_portfolios(self) -> pd.DataFrame:
        stmt = select(
            PortfolioRecord.portfolio_id,
            PortfolioRecord.name,
            PortfolioRecord.base_currency,
            PortfolioRecord.valuation_date,
            PortfolioRecord.description,
        ).order_by(PortfolioRecord.portfolio_id)
        with self.engine.connect() as conn:
            return pd.read_sql(stmt, conn)

    def replace_positions(self, portfolio_id: str, valuation_date: date, positions: pd.DataFrame) -> int:
        """Replace the positions of a portfolio for one valuation date (transactional)."""
        rows = _records(
            positions.assign(portfolio_id=portfolio_id, valuation_date=valuation_date),
            ["position_id", "portfolio_id", "instrument_id", "quantity", "price", "market_value", "currency", "weight", "valuation_date"],
        )
        with self.session() as s:
            s.execute(
                delete(PositionRecord).where(
                    PositionRecord.portfolio_id == portfolio_id, PositionRecord.valuation_date == valuation_date
                )
            )
            s.execute(PositionRecord.__table__.insert(), rows)
        return len(rows)

    def upsert_positions(self, portfolio_id: str, valuation_date: date, positions: pd.DataFrame) -> int:
        rows = _records(
            positions.assign(portfolio_id=portfolio_id, valuation_date=valuation_date),
            ["position_id", "portfolio_id", "instrument_id", "quantity", "price", "market_value", "currency", "weight", "valuation_date"],
        )
        return self.upsert(PositionRecord, rows, ["portfolio_id", "position_id", "valuation_date"])

    def load_position_records(self, portfolio_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Portfolio header and flat position records (ready for Pydantic validation)."""
        with self.session() as s:
            header = s.get(PortfolioRecord, portfolio_id)
            if header is None:
                raise KeyError(f"Portfolio {portfolio_id} not found")
            vd = header.valuation_date
            info = {
                "portfolio_id": header.portfolio_id,
                "name": header.name,
                "base_currency": header.base_currency,
                "valuation_date": vd,
            }
        stmt = (
            select(
                PositionRecord.position_id,
                InstrumentRecord.asset_type,
                InstrumentRecord.ticker,
                PositionRecord.instrument_id,
                PositionRecord.quantity,
                PositionRecord.price,
                PositionRecord.market_value,
                PositionRecord.weight,
                PositionRecord.currency,
                BondTermsRecord.maturity_date.label("bond_maturity"),
                BondTermsRecord.coupon_rate.label("coupon"),
                BondTermsRecord.coupon_frequency,
                BondTermsRecord.credit_curve,
                BondTermsRecord.face_value,
                OptionTermsRecord.expiry_date.label("option_expiry"),
                OptionTermsRecord.strike,
                OptionTermsRecord.option_type,
                OptionTermsRecord.underlying_id.label("underlying"),
                OptionTermsRecord.contract_multiplier,
                MarketDataRecord.volatility,
            )
            .join(InstrumentRecord, InstrumentRecord.instrument_id == PositionRecord.instrument_id)
            .outerjoin(BondTermsRecord, BondTermsRecord.instrument_id == PositionRecord.instrument_id)
            .outerjoin(OptionTermsRecord, OptionTermsRecord.instrument_id == PositionRecord.instrument_id)
            .outerjoin(
                MarketDataRecord,
                (MarketDataRecord.instrument_id == PositionRecord.instrument_id)
                & (MarketDataRecord.date == PositionRecord.valuation_date)
                & (InstrumentRecord.asset_type == "OPTION"),
            )
            .where(PositionRecord.portfolio_id == portfolio_id, PositionRecord.valuation_date == vd)
            .order_by(PositionRecord.position_id)
        )
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["maturity"] = df["bond_maturity"].where(df["bond_maturity"].notna(), df["option_expiry"])
        df = df.drop(columns=["bond_maturity", "option_expiry"])
        records = [{k: _clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
        for rec in records:
            if rec["asset_type"] != "OPTION":
                for k in ("volatility", "strike", "option_type", "underlying", "contract_multiplier"):
                    rec.pop(k, None)
            if rec["asset_type"] != "BOND":
                for k in ("coupon", "coupon_frequency", "credit_curve", "face_value"):
                    rec.pop(k, None)
            if rec["asset_type"] not in ("BOND", "OPTION"):
                rec.pop("maturity", None)
        return info, records

    # ------------------------------------------------------------ market data
    MARKET_COLUMNS = (
        "instrument_id", "date", "asset_type", "price", "yield", "volatility", "currency",
        "open", "high", "low", "adj_close", "volume", "duration", "convexity",
    )

    def upsert_market_data(self, df: pd.DataFrame, update: bool = False) -> int:
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        for col in self.MARKET_COLUMNS:
            if col not in frame.columns:
                frame[col] = None
        update_cols = [c for c in self.MARKET_COLUMNS if c not in ("instrument_id", "date")] if update else None
        return self.upsert(MarketDataRecord, _records(frame, self.MARKET_COLUMNS), ["instrument_id", "date"], update_cols)

    def load_market_data(
        self, instrument_ids: Iterable[str] | None = None, end: date | None = None, start: date | None = None
    ) -> pd.DataFrame:
        stmt = select(MarketDataRecord.__table__).order_by(MarketDataRecord.date)
        if instrument_ids is not None:
            stmt = stmt.where(MarketDataRecord.instrument_id.in_(list(instrument_ids)))
        if end is not None:
            stmt = stmt.where(MarketDataRecord.date <= end)
        if start is not None:
            stmt = stmt.where(MarketDataRecord.date >= start)
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["date"] = pd.to_datetime(df["date"])
        return df.drop(columns=["id"])

    def market_data_dates(self) -> tuple[date | None, date | None]:
        with self.session() as s:
            lo, hi = s.execute(select(func.min(MarketDataRecord.date), func.max(MarketDataRecord.date))).one()
        return lo, hi

    def instruments_with_data_on(self, d: date) -> set[str]:
        with self.session() as s:
            rows = s.execute(select(MarketDataRecord.instrument_id).where(MarketDataRecord.date == d)).scalars().all()
        return set(rows)

    # ------------------------------------------------------------ yield curves
    def upsert_yield_curves(self, df: pd.DataFrame, update: bool = False) -> int:
        frame = df.assign(date=pd.to_datetime(df["date"]).dt.date)
        return self.upsert(
            YieldCurveRecord,
            _records(frame, ["curve_id", "date", "tenor", "tenor_years", "rate"]),
            ["curve_id", "date", "tenor"],
            ["tenor_years", "rate"] if update else None,
        )

    def load_yield_curves(self, end: date | None = None) -> pd.DataFrame:
        stmt = select(
            YieldCurveRecord.curve_id, YieldCurveRecord.date, YieldCurveRecord.tenor, YieldCurveRecord.tenor_years, YieldCurveRecord.rate
        ).order_by(YieldCurveRecord.date)
        if end is not None:
            stmt = stmt.where(YieldCurveRecord.date <= end)
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # -------------------------------------------------------- vol surfaces
    def upsert_vol_surface(self, df: pd.DataFrame, update: bool = False) -> int:
        frame = df.rename(columns={"underlying": "underlying_id"}).assign(date=pd.to_datetime(df["date"]).dt.date)
        return self.upsert(
            VolSurfaceRecord,
            _records(frame, ["underlying_id", "date", "maturity", "maturity_years", "moneyness", "implied_vol"]),
            ["underlying_id", "date", "maturity", "moneyness"],
            ["maturity_years", "implied_vol"] if update else None,
        )

    def load_vol_surface(self, end: date | None = None) -> pd.DataFrame:
        stmt = select(
            VolSurfaceRecord.underlying_id.label("underlying"),
            VolSurfaceRecord.date,
            VolSurfaceRecord.maturity,
            VolSurfaceRecord.maturity_years,
            VolSurfaceRecord.moneyness,
            VolSurfaceRecord.implied_vol,
        ).order_by(VolSurfaceRecord.date)
        if end is not None:
            stmt = stmt.where(VolSurfaceRecord.date <= end)
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------- scenarios
    def upsert_scenarios(self, scenarios: list[dict[str, Any]], update: bool = True) -> int:
        rows = [
            {
                "scenario_id": s["scenario_id"],
                "name": s["name"],
                "category": s.get("category", "HYPOTHETICAL"),
                "description": s.get("description", ""),
                "definition": json.loads(json.dumps(s, default=str)),
                "is_predefined": bool(s.get("is_predefined", True)),
            }
            for s in scenarios
        ]
        return self.upsert(
            StressScenarioRecord, rows, ["scenario_id"], ["name", "category", "description", "definition"] if update else None
        )

    def load_scenarios(self) -> list[dict[str, Any]]:
        with self.session() as s:
            recs = s.execute(select(StressScenarioRecord).order_by(StressScenarioRecord.created_at)).scalars().all()
            return [dict(r.definition) for r in recs]

    # --------------------------------------------------------------- results
    def save_risk_metrics(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert(
            RiskMetricRecord,
            rows,
            ["portfolio_id", "date", "method", "confidence", "lookback", "holding_period", "decay_factor"],
            ["var", "expected_shortfall", "volatility", "portfolio_value"],
        )

    def save_risk_contributions(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert(
            RiskContributionRecord,
            rows,
            ["portfolio_id", "date", "method", "position_id"],
            ["exposure", "marginal_var", "component_var", "pct_contribution"],
        )

    def save_stress_results(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert(
            StressResultRecord,
            rows,
            ["portfolio_id", "scenario_id", "run_date", "vol_method", "bond_method"],
            ["total_pnl", "pnl_pct", "stressed_nav", "worst_position", "worst_asset_class"],
        )

    def save_pnl_attribution(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert(PnLAttributionRecord, rows, ["portfolio_id", "date", "component"], ["amount"])

    def save_backtest_results(self, rows: list[dict[str, Any]]) -> int:
        return self.upsert(
            BacktestResultRecord,
            rows,
            ["portfolio_id", "method", "confidence", "lookback", "date"],
            ["predicted_var", "actual_pnl", "exception"],
        )

    def load_risk_metrics(self, portfolio_id: str) -> pd.DataFrame:
        stmt = select(RiskMetricRecord.__table__).where(RiskMetricRecord.portfolio_id == portfolio_id)
        with self.engine.connect() as conn:
            return pd.read_sql(stmt, conn)
