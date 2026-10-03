"""Seed the database from the generated sample CSVs (idempotent).

Every insert uses ``ON CONFLICT DO NOTHING`` on the table's natural key
(instrument id, instrument/date, curve/date/tenor, ...), so running the seeder
any number of times never duplicates instruments, positions, market data or
scenarios. Scenario definitions are upserted so edits to
``config/scenarios.yaml`` propagate.

Usage::

    python scripts/seed_database.py [--data data/sample] [--force-generate]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.revaluation import PortfolioPricer  # noqa: E402
from scripts.generate_sample_data import DATASET_VERSION, VERSION_FILE  # noqa: E402
from config.settings import AppSettings, configure_logging, get_settings  # noqa: E402
from database.connection import get_engine  # noqa: E402
from database.repository import RiskRepository  # noqa: E402
from models.market_data import MarketDataPoint, VolSurfacePoint, YieldCurvePoint  # noqa: E402
from models.portfolio import build_portfolio  # noqa: E402
from services.market_data_service import CsvMarketDataProvider, MarketHistory  # noqa: E402
from services.portfolio_service import PortfolioService, parse_portfolio_csv  # noqa: E402
from stress.scenario_engine import load_scenarios, scenario_to_dict  # noqa: E402

logger = logging.getLogger(__name__)

SEED_VERSION = DATASET_VERSION  # the seed marker tracks the generated dataset version
REQUIRED_FILES = ("instruments.csv", "market_data.csv", "yield_curves.csv", "vol_surface.csv", "sample_portfolio.csv")


class SeedValidationError(ValueError):
    """Sample data failed validation; nothing was written."""


@dataclass
class SeedReport:
    counts: dict[str, int] = field(default_factory=dict)
    submitted: dict[str, int] = field(default_factory=dict)
    portfolio_id: str = ""
    positions: int = 0
    nav: float = 0.0


def sample_files_present(data_dir: Path) -> bool:
    """True when every sample CSV exists and was written by the current generator version."""
    version = data_dir / VERSION_FILE
    current = version.exists() and version.read_text(encoding="utf-8").strip() == DATASET_VERSION
    return current and all((data_dir / f).exists() for f in REQUIRED_FILES)


def _validate(adapter: TypeAdapter, df: pd.DataFrame, label: str) -> None:
    records = df.to_dict(orient="records")
    try:
        adapter.validate_python(records)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise SeedValidationError(f"{label}: {exc.error_count()} invalid rows (first: row {first['loc'][0]} {first['msg']})") from exc
    logger.info("%s validated (%d rows)", label, len(records))


def _dataset_hash(data_dir: Path) -> str:
    h = hashlib.sha256()
    for name in REQUIRED_FILES:
        h.update((data_dir / name).read_bytes())
    return h.hexdigest()[:16]


def seed_database(repo: RiskRepository, settings: AppSettings, data_dir: Path | None = None, refresh: bool = False) -> SeedReport:
    """Validate the sample CSVs and load them into the database.

    Args:
        refresh: False (first seed) inserts with ``ON CONFLICT DO NOTHING``. True
            (dataset upgrade) overwrites the seeded market data, curves, surfaces
            and sample-portfolio positions in place with ``ON CONFLICT DO UPDATE``;
            user-saved portfolios and custom scenarios are left untouched. Both
            modes are idempotent.

    Raises:
        FileNotFoundError: if sample files are missing.
        SeedValidationError: if any file fails schema validation.
    """
    data_dir = data_dir or settings.sample_data_dir
    if not sample_files_present(data_dir):
        raise FileNotFoundError(f"Sample data missing in {data_dir}; run scripts/generate_sample_data.py")
    repo.create_schema()
    provider = CsvMarketDataProvider(data_dir)
    report = SeedReport()

    instruments = provider.load_instruments()
    market = provider.load_market_data()
    curves = provider.load_yield_curves()
    surfaces = provider.load_vol_surfaces()

    # ---- validation (nothing is written unless everything passes) ------------------
    _validate(TypeAdapter(list[MarketDataPoint]), market, "Market data")
    _validate(TypeAdapter(list[YieldCurvePoint]), curves, "Yield curves")
    _validate(TypeAdapter(list[VolSurfacePoint]), surfaces, "Volatility surface")
    unknown = set(market["instrument_id"]) - set(instruments["instrument_id"])
    if unknown:
        raise SeedValidationError(f"Market data references unknown instruments: {sorted(unknown)}")

    valuation_date = pd.Timestamp(settings.market_data.end_date).date()
    if pd.Timestamp(valuation_date) not in set(market["date"]):
        valuation_date = market["date"].max().date()
    records = parse_portfolio_csv(data_dir / "sample_portfolio.csv")
    portfolio, _ = build_portfolio(
        records,
        portfolio_id=settings.portfolio.portfolio_id,
        name=settings.portfolio.name,
        valuation_date=valuation_date,
        base_currency=settings.portfolio.base_currency,
    )

    # ---- persistence (idempotent) ---------------------------------------------------
    report.submitted["instruments"] = repo.upsert_instruments(instruments)
    report.submitted["market_data"] = repo.upsert_market_data(market, update=refresh)
    report.submitted["yield_curve_data"] = repo.upsert_yield_curves(curves, update=refresh)
    report.submitted["volatility_surface"] = repo.upsert_vol_surface(surfaces, update=refresh)
    scenarios = load_scenarios(settings.scenarios_path)
    report.submitted["stress_scenarios"] = repo.upsert_scenarios([scenario_to_dict(s) for s in scenarios])

    history = MarketHistory.build(provider, end=valuation_date, max_ffill=settings.market_data.max_forward_fill_days)
    pricer = PortfolioPricer(portfolio, history.state_at(valuation_date), settings.stress.min_volatility)
    valuation = PortfolioService.valuation(portfolio, pricer)
    repo.upsert_portfolio(
        portfolio.portfolio_id, portfolio.name, portfolio.base_currency, valuation_date,
        "Synthetic multi-asset sample portfolio (seeded).",
    )
    positions = pd.DataFrame(
        {
            "position_id": [p.position_id for p in portfolio.positions],
            "instrument_id": [p.instrument_id for p in portfolio.positions],
            "quantity": [p.quantity for p in portfolio.positions],
            "price": [p.price for p in portfolio.positions],
            "currency": [p.currency for p in portfolio.positions],
        }
    ).merge(valuation[["position_id", "market_value", "weight"]], on="position_id")
    if refresh:
        report.submitted["positions"] = repo.replace_positions(portfolio.portfolio_id, valuation_date, positions)
    else:
        report.submitted["positions"] = repo.upsert_positions(portfolio.portfolio_id, valuation_date, positions)

    repo.set_meta("seed_version", SEED_VERSION)
    repo.set_meta("seed_dataset_hash", _dataset_hash(data_dir))
    repo.set_meta("seeded_at", datetime.now().isoformat(timespec="seconds"))
    report.counts = repo.table_counts()
    report.portfolio_id = portfolio.portfolio_id
    report.positions = len(portfolio.positions)
    report.nav = float(valuation["market_value"].sum())
    logger.info("Database seeded: %s", {k: v for k, v in report.counts.items() if v})
    return report


def needs_refresh(repo: RiskRepository) -> bool:
    """True when the database was seeded by an older dataset version."""
    try:
        version = repo.get_meta("seed_version")
    except Exception:  # noqa: BLE001 - no metadata table yet
        return False
    return version is not None and version != SEED_VERSION


def is_seeded(repo: RiskRepository) -> bool:
    """True when the seed marker is present and core tables are populated."""
    try:
        if repo.get_meta("seed_version") != SEED_VERSION:
            return False
        counts = repo.table_counts()
    except Exception:  # noqa: BLE001 - missing tables etc.
        return False
    return all(counts.get(t, 0) > 0 for t in ("instruments", "market_data", "positions", "yield_curve_data", "stress_scenarios"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed the risk-engine database from sample CSVs.")
    parser.add_argument("--data", type=Path, default=None, help="Directory with sample CSVs")
    parser.add_argument("--force-generate", action="store_true", help="Regenerate sample CSVs first")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings)
    data_dir = args.data or settings.sample_data_dir
    if args.force_generate or not sample_files_present(data_dir):
        from scripts.generate_sample_data import generate_sample_data

        md = settings.market_data
        generate_sample_data(data_dir, md.seed, md.n_days, md.end_date)
    repo = RiskRepository(get_engine(settings.database_url))
    report = seed_database(repo, settings, data_dir, refresh=needs_refresh(repo))
    print(f"Seeded portfolio {report.portfolio_id}: {report.positions} positions, NAV {report.nav:,.0f} USD")
    for table, n in report.counts.items():
        print(f"  {table:22s} {n:>8,d}")


if __name__ == "__main__":
    main()
