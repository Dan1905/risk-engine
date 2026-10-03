"""Portfolio ingestion, validation and valuation.

Responsibilities:
* parse CSV uploads into flat records (ingestion)
* validate records with the Pydantic schema plus market-data coverage checks
  (validation) - invalid portfolios never reach the risk engine
* load / save portfolios through the repository (persistence)
* compute market values, weights and exposures (valuation)
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date
from typing import IO, Any

import numpy as np
import pandas as pd

from analytics.revaluation import PortfolioPricer
from config.settings import AppSettings
from database.repository import RiskRepository
from models.portfolio import Portfolio, PortfolioValidationError, ValidationReport, validate_position_records
from models.positions import AssetType, Position, parse_fx_pair
from services.market_data_service import MarketHistory

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = (
    "position_id", "asset_type", "ticker", "quantity", "price", "market_value", "currency", "weight",
    "maturity", "face_value", "coupon", "strike", "option_type", "volatility",
)
OPTIONAL_COLUMNS = ("coupon_frequency", "credit_curve", "underlying", "contract_multiplier")
ASSET_CLASS_LABELS = {
    "EQUITY": "Equity",
    "BOND": "Fixed Income",
    "FX": "FX",
    "OPTION": "Options",
    "CASH": "Cash",
}


@dataclass
class UploadResult:
    portfolio: Portfolio | None
    report: ValidationReport

    @property
    def error_text(self) -> str:
        return PortfolioValidationError(self.report.errors).format() if self.report.errors else ""


def parse_portfolio_csv(source: str | bytes | IO[Any]) -> list[dict[str, Any]]:
    """Read a portfolio CSV into flat records.

    Tolerant of trailing delimiters (``index_col=False``), header case and
    whitespace, and blank lines. Missing optional columns are simply absent.

    Raises:
        ValueError: if required columns are missing.
    """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    df = pd.read_csv(source, index_col=False, dtype=str, skip_blank_lines=True, keep_default_na=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("unnamed")]]
    required = {"position_id", "asset_type", "ticker", "quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio file is missing required columns: {', '.join(sorted(missing))}")
    df = df.dropna(how="all")
    records = []
    for row in df.to_dict(orient="records"):
        rec = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        records.append({k: (None if (v is None or (isinstance(v, float) and np.isnan(v)) or v == "") else v) for k, v in rec.items()})
    return records


def coverage_check(portfolio_positions: list[Position], history: MarketHistory, report: ValidationReport) -> None:
    """Add errors for positions whose market data is unavailable."""
    tickers, pairs = history.known_tickers(), history.known_fx_pairs()
    ccys = set(history.fx_usd.columns) | {"USD"}
    for p in portfolio_positions:
        pid = p.position_id
        if p.asset_type == AssetType.EQUITY and p.ticker not in tickers:
            report.add_error(pid, f"No market data available for {p.ticker}.")
        if p.asset_type == AssetType.OPTION and p.option is not None and p.option.underlying not in tickers:
            report.add_error(pid, f"No market data available for option underlying {p.option.underlying}.")
        if p.asset_type == AssetType.FX and p.ticker not in pairs:
            report.add_error(pid, f"No market data available for FX pair {p.ticker}.")
        if p.asset_type == AssetType.BOND and p.currency != "USD":
            report.add_error(pid, f"Yield curve data missing for {p.currency} bonds.")
        if p.currency not in ccys:
            report.add_error(pid, f"No FX rate available for {p.currency}.")
        if p.asset_type == AssetType.OPTION and p.option is not None and p.option.underlying not in history.surfaces:
            report.add_warning(pid, "Volatility surface data unavailable for selected option; flat volatility used.")
        if p.asset_type == AssetType.OPTION and p.option is not None and p.option.volatility > 2.0:
            report.add_warning(pid, f"Implied volatility {p.option.volatility:.0%} is unusually high.")


def price_reconciliation(portfolio: Portfolio, pricer: PortfolioPricer, tolerance: float = 0.05) -> dict[str, list[str]]:
    """Warnings where the input price differs materially from the market/model price."""
    warnings: dict[str, list[str]] = {}
    state = pricer.base_state
    for p in portfolio.positions:
        model: float | None = None
        if p.asset_type == AssetType.EQUITY:
            model = state.equity_prices.get(p.ticker)
        elif p.asset_type == AssetType.FX:
            base, _ = parse_fx_pair(p.ticker)
            usd = state.fx_rates.get(p.exposure_currency)
            model = None if usd is None else (1 / usd if base == "USD" else usd)
        elif p.asset_type == AssetType.OPTION:
            opt = pricer.option_analytics()
            row = opt[opt["position_id"] == p.position_id]
            model = float(row["price"].iloc[0]) if not row.empty else None
        if model and abs(p.price / model - 1) > tolerance:
            warnings.setdefault(p.position_id, []).append(
                f"Input price {p.price:,.4f} differs from market/model price {model:,.4f}; the market/model price is used."
            )
    return warnings


class PortfolioService:
    """Loads, validates, values and persists portfolios."""

    def __init__(self, repo: RiskRepository, settings: AppSettings):
        self.repo = repo
        self.settings = settings

    # ------------------------------------------------------------------ load
    def list_portfolios(self) -> pd.DataFrame:
        return self.repo.list_portfolios()

    def load_portfolio(self, portfolio_id: str | None = None) -> Portfolio:
        pid = portfolio_id or self.settings.portfolio.portfolio_id
        header, records = self.repo.load_position_records(pid)
        report = validate_position_records(records, header["valuation_date"])
        if report.errors:
            logger.error("Stored portfolio %s failed validation: %s", pid, report.errors)
            raise PortfolioValidationError(report.errors, report.warnings)
        portfolio = Portfolio(positions=report.positions, **header)
        logger.info("Portfolio loaded: %s (%d positions validated)", pid, len(report.positions))
        return portfolio

    # ---------------------------------------------------------------- upload
    def validate_upload(
        self,
        source: str | bytes | IO[Any],
        history: MarketHistory,
        valuation_date: date,
        portfolio_id: str = "UPLOAD",
        name: str = "Uploaded Portfolio",
    ) -> UploadResult:
        """Parse and validate an uploaded CSV against schema and market-data coverage."""
        try:
            records = parse_portfolio_csv(source)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.warning("Portfolio CSV could not be parsed: %s", exc)
            rep = ValidationReport()
            rep.add_error("FILE", str(exc) if isinstance(exc, ValueError) else "File could not be parsed as CSV.")
            return UploadResult(None, rep)
        report = validate_position_records(records, valuation_date)
        coverage_check(report.positions, history, report)
        if not history.has_date(valuation_date):
            report.add_error("PORTFOLIO", "No market data available for selected valuation date.")
        if report.errors or not report.positions:
            if not report.positions and not report.errors:
                report.add_error("PORTFOLIO", "Portfolio contains no positions.")
            return UploadResult(None, report)
        portfolio = Portfolio(
            portfolio_id=portfolio_id,
            name=name,
            base_currency="USD",
            valuation_date=valuation_date,
            positions=report.positions,
        )
        logger.info("%d positions validated from upload", len(report.positions))
        return UploadResult(portfolio, report)

    def save_portfolio(self, portfolio: Portfolio, valuation: pd.DataFrame, instruments: pd.DataFrame) -> None:
        """Persist a validated portfolio (instruments, terms, positions, option marks)."""
        known = set(instruments["instrument_id"])
        new_rows = []
        for p in portfolio.positions:
            iid = p.instrument_id
            if iid in known:
                continue
            row: dict[str, Any] = {
                "instrument_id": iid, "ticker": p.ticker, "name": iid, "asset_type": p.asset_type.value,
                "currency": p.currency, "sector": None,
            }
            if p.bond is not None:
                row.update(
                    coupon_rate=p.bond.coupon_rate, maturity_date=p.bond.maturity_date, coupon_frequency=p.bond.coupon_frequency,
                    face_value=p.bond.face_value, credit_curve=p.bond.credit_curve,
                )
            if p.option is not None:
                row.update(
                    underlying_id=p.option.underlying, strike=p.option.strike, expiry_date=p.option.expiry_date,
                    option_type=p.option.option_type.value, contract_multiplier=p.option.contract_multiplier,
                )
            if p.asset_type == AssetType.FX:
                base, quote = parse_fx_pair(p.ticker)
                row.update(base_currency=base, quote_currency=quote)
            new_rows.append(row)
        if new_rows:
            self.repo.upsert_instruments(pd.DataFrame(new_rows))
        self.repo.upsert_portfolio(portfolio.portfolio_id, portfolio.name, portfolio.base_currency, portfolio.valuation_date)
        positions = pd.DataFrame(
            {
                "position_id": [p.position_id for p in portfolio.positions],
                "instrument_id": [p.instrument_id for p in portfolio.positions],
                "quantity": [p.quantity for p in portfolio.positions],
                "price": [p.price for p in portfolio.positions],
                "currency": [p.currency for p in portfolio.positions],
            }
        ).merge(valuation[["position_id", "market_value", "weight"]], on="position_id")
        self.repo.replace_positions(portfolio.portfolio_id, portfolio.valuation_date, positions)
        option_marks = [
            {
                "instrument_id": p.instrument_id, "date": portfolio.valuation_date, "asset_type": "OPTION",
                "price": p.price, "volatility": p.option.volatility, "currency": p.currency,
            }
            for p in portfolio.positions
            if p.option is not None
        ]
        if option_marks:
            self.repo.upsert_market_data(pd.DataFrame(option_marks), update=True)
        logger.info("Portfolio %s saved (%d positions)", portfolio.portfolio_id, len(positions))

    # ------------------------------------------------------------- valuation
    @staticmethod
    def valuation(portfolio: Portfolio, pricer: PortfolioPricer, instruments: pd.DataFrame | None = None) -> pd.DataFrame:
        """Position-level valuation table (USD)."""
        mv = pricer.value(pricer.base_state)
        nav = float(mv.sum())
        names = {}
        if instruments is not None and not instruments.empty:
            names = dict(zip(instruments["instrument_id"], instruments["name"]))
        rows = []
        for p in portfolio.positions:
            v = float(mv[p.position_id])
            rows.append(
                {
                    "position_id": p.position_id,
                    "ticker": p.ticker,
                    "instrument_id": p.instrument_id,
                    "description": names.get(p.instrument_id, p.instrument_id),
                    "asset_type": p.asset_type.value,
                    "asset_class": ASSET_CLASS_LABELS[p.asset_type.value],
                    "quantity": p.quantity,
                    "price": p.price,
                    "currency": p.currency,
                    "exposure_currency": p.exposure_currency,
                    "market_value": v,
                    "weight": v / nav if nav else np.nan,
                    "side": "Long" if v >= 0 else "Short",
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def exposures(valuation: pd.DataFrame) -> dict[str, float]:
        """Headline exposure figures (USD)."""
        mv = valuation.set_index("position_id")["market_value"]
        by_type = valuation.groupby("asset_type")["market_value"].sum()
        long_, short_ = float(mv[mv > 0].sum()), float(mv[mv < 0].sum())
        return {
            "nav": float(mv.sum()),
            "equity": float(by_type.get("EQUITY", 0.0)),
            "fixed_income": float(by_type.get("BOND", 0.0)),
            "fx": float(by_type.get("FX", 0.0)),
            "options": float(by_type.get("OPTION", 0.0)),
            "cash": float(by_type.get("CASH", 0.0)),
            "long": long_,
            "short": short_,
            "gross": long_ - short_,
            "net": long_ + short_,
        }

    @staticmethod
    def exposure_by(valuation: pd.DataFrame, key: str) -> pd.DataFrame:
        """Net / gross exposure and weight grouped by ``key`` (e.g. asset_class, exposure_currency)."""
        nav = valuation["market_value"].sum()
        g = valuation.groupby(key)["market_value"]
        out = pd.DataFrame({"net": g.sum(), "gross": g.apply(lambda s: s.abs().sum())})
        out["weight"] = out["net"] / nav if nav else np.nan
        out["gross_share"] = out["gross"] / out["gross"].sum()
        return out.sort_values("gross", ascending=False)
