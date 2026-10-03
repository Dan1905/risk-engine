"""SQLAlchemy ORM schema (PostgreSQL-compatible).

Normalisation
-------------
``instruments`` holds attributes common to every instrument. Asset-specific
terms live in 1:1 extension tables (``bond_terms``, ``option_terms``,
``fx_terms``) so no table depends on sparsely populated columns. Positions
reference instruments; market data, curves and surfaces are keyed by
instrument / curve and date with unique constraints that make seeding
idempotent (``INSERT ... ON CONFLICT``).
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PortfolioRecord(Base):
    __tablename__ = "portfolios"

    portfolio_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    base_currency: Mapped[str] = mapped_column(String(3), default="USD")
    valuation_date: Mapped[dt.date] = mapped_column(Date)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    positions: Mapped[list[PositionRecord]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")


class InstrumentRecord(Base):
    __tablename__ = "instruments"

    instrument_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ticker: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(128))
    asset_type: Mapped[str] = mapped_column(String(16), index=True)
    currency: Mapped[str] = mapped_column(String(3))
    sector: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class BondTermsRecord(Base):
    __tablename__ = "bond_terms"

    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"), primary_key=True)
    coupon_rate: Mapped[float] = mapped_column(Float)
    maturity_date: Mapped[dt.date] = mapped_column(Date)
    coupon_frequency: Mapped[int] = mapped_column(Integer, default=2)
    face_value: Mapped[float] = mapped_column(Float, default=100.0)
    credit_curve: Mapped[str | None] = mapped_column(String(32))


class OptionTermsRecord(Base):
    __tablename__ = "option_terms"

    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"), primary_key=True)
    underlying_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"), index=True)
    strike: Mapped[float] = mapped_column(Float)
    expiry_date: Mapped[dt.date] = mapped_column(Date)
    option_type: Mapped[str] = mapped_column(String(4))
    contract_multiplier: Mapped[float] = mapped_column(Float, default=100.0)
    exercise_style: Mapped[str] = mapped_column(String(16), default="EUROPEAN")


class FxTermsRecord(Base):
    __tablename__ = "fx_terms"

    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"), primary_key=True)
    base_currency: Mapped[str] = mapped_column(String(3))
    quote_currency: Mapped[str] = mapped_column(String(3))


class PositionRecord(Base):
    __tablename__ = "positions"
    __table_args__ = (UniqueConstraint("portfolio_id", "position_id", "valuation_date", name="uq_position"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    position_id: Mapped[str] = mapped_column(String(40))
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"))
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    weight: Mapped[float] = mapped_column(Float)
    valuation_date: Mapped[dt.date] = mapped_column(Date)

    portfolio: Mapped[PortfolioRecord] = relationship(back_populates="positions")


class MarketDataRecord(Base):
    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint("instrument_id", "date", name="uq_market_data"),
        Index("ix_market_data_date", "date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"))
    date: Mapped[dt.date] = mapped_column(Date)
    asset_type: Mapped[str] = mapped_column(String(16))
    price: Mapped[float] = mapped_column(Float)
    yield_: Mapped[float | None] = mapped_column("yield", Float)
    volatility: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    duration: Mapped[float | None] = mapped_column(Float)
    convexity: Mapped[float | None] = mapped_column(Float)


class YieldCurveRecord(Base):
    __tablename__ = "yield_curve_data"
    __table_args__ = (UniqueConstraint("curve_id", "date", "tenor", name="uq_curve_point"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    curve_id: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    tenor: Mapped[str] = mapped_column(String(8))
    tenor_years: Mapped[float] = mapped_column(Float)
    rate: Mapped[float] = mapped_column(Float)


class VolSurfaceRecord(Base):
    __tablename__ = "volatility_surface"
    __table_args__ = (UniqueConstraint("underlying_id", "date", "maturity", "moneyness", name="uq_vol_point"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    underlying_id: Mapped[str] = mapped_column(ForeignKey("instruments.instrument_id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    maturity: Mapped[str] = mapped_column(String(8))
    maturity_years: Mapped[float] = mapped_column(Float)
    moneyness: Mapped[float] = mapped_column(Float)
    implied_vol: Mapped[float] = mapped_column(Float)


class RiskMetricRecord(Base):
    __tablename__ = "risk_metrics"
    __table_args__ = (
        UniqueConstraint(
            "portfolio_id", "date", "method", "confidence", "lookback", "holding_period", "decay_factor",
            name="uq_risk_metric",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    method: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[float] = mapped_column(Float)
    lookback: Mapped[int] = mapped_column(Integer)
    holding_period: Mapped[int] = mapped_column(Integer, default=1)
    decay_factor: Mapped[float] = mapped_column(Float)
    var: Mapped[float] = mapped_column(Float)
    expected_shortfall: Mapped[float] = mapped_column(Float)
    volatility: Mapped[float] = mapped_column(Float)
    portfolio_value: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class RiskContributionRecord(Base):
    __tablename__ = "risk_contributions"
    __table_args__ = (UniqueConstraint("portfolio_id", "date", "method", "position_id", name="uq_risk_contribution"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    method: Mapped[str] = mapped_column(String(24))
    position_id: Mapped[str] = mapped_column(String(40))
    exposure: Mapped[float] = mapped_column(Float)
    marginal_var: Mapped[float] = mapped_column(Float)
    component_var: Mapped[float] = mapped_column(Float)
    pct_contribution: Mapped[float] = mapped_column(Float)


class StressScenarioRecord(Base):
    __tablename__ = "stress_scenarios"

    scenario_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(24))
    description: Mapped[str | None] = mapped_column(Text)
    definition: Mapped[dict] = mapped_column(JSON)
    is_predefined: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class StressResultRecord(Base):
    __tablename__ = "stress_results"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "scenario_id", "run_date", "vol_method", "bond_method", name="uq_stress_result"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    scenario_id: Mapped[str] = mapped_column(String(48))
    run_date: Mapped[dt.date] = mapped_column(Date)
    vol_method: Mapped[str] = mapped_column(String(16))
    bond_method: Mapped[str] = mapped_column(String(16))
    total_pnl: Mapped[float] = mapped_column(Float)
    pnl_pct: Mapped[float] = mapped_column(Float)
    stressed_nav: Mapped[float] = mapped_column(Float)
    worst_position: Mapped[str | None] = mapped_column(String(40))
    worst_asset_class: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class PnLAttributionRecord(Base):
    __tablename__ = "pnl_attribution"
    __table_args__ = (UniqueConstraint("portfolio_id", "date", "component", name="uq_pnl_attribution"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    date: Mapped[dt.date] = mapped_column(Date)
    component: Mapped[str] = mapped_column(String(24))
    amount: Mapped[float] = mapped_column(Float)


class BacktestResultRecord(Base):
    __tablename__ = "backtest_results"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "method", "confidence", "lookback", "date", name="uq_backtest_result"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.portfolio_id"), index=True)
    method: Mapped[str] = mapped_column(String(24))
    confidence: Mapped[float] = mapped_column(Float)
    lookback: Mapped[int] = mapped_column(Integer)
    date: Mapped[dt.date] = mapped_column(Date)
    predicted_var: Mapped[float] = mapped_column(Float)
    actual_pnl: Mapped[float] = mapped_column(Float)
    exception: Mapped[bool] = mapped_column(Boolean)


class AppMetadataRecord(Base):
    """Key/value store for seed status and schema version."""

    __tablename__ = "app_metadata"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
