"""PostgreSQL integration tests.

Skipped unless ``TEST_DATABASE_URL`` points at a disposable PostgreSQL
database, e.g.::

    TEST_DATABASE_URL=postgresql+psycopg://risk:pwd@localhost:5432/risk_engine pytest tests/test_postgres.py

The tests drop and recreate the schema, so never point them at real data.
"""

from __future__ import annotations

import os

import pytest

from config.settings import get_settings
from database.connection import create_db_engine
from database.models import Base
from database.repository import RiskRepository
from scripts.generate_sample_data import generate_sample_data
from scripts.seed_database import is_seeded, seed_database
from services.eod_service import last_run, run_end_of_day
from services.portfolio_service import PortfolioService

URL = os.getenv("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not URL.startswith("postgresql"), reason="TEST_DATABASE_URL is not a PostgreSQL URL")

N_DAYS = 320


@pytest.fixture(scope="module")
def pg_settings(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("pg_sample")
    generate_sample_data(data_dir, seed=7, n_days=N_DAYS, end_date="2026-09-30")
    base = get_settings()
    md = base.market_data.model_copy(update={"n_days": N_DAYS, "seed": 7})
    return base.model_copy(update={"database_url": URL, "sample_data_dir": data_dir, "market_data": md})


@pytest.fixture(scope="module")
def pg_repo(pg_settings):
    engine = create_db_engine(URL)
    assert engine.dialect.name == "postgresql"
    Base.metadata.drop_all(engine)  # disposable test database
    repo = RiskRepository(engine)
    yield repo
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_seeding_is_idempotent_on_postgres(pg_repo, pg_settings):
    seed_database(pg_repo, pg_settings)
    first = pg_repo.table_counts()
    assert is_seeded(pg_repo)
    assert first["market_data"] == N_DAYS * 19 and first["positions"] == 20
    seed_database(pg_repo, pg_settings)
    assert pg_repo.table_counts() == first
    seed_database(pg_repo, pg_settings, refresh=True)  # ON CONFLICT DO UPDATE path
    assert pg_repo.table_counts() == first


def test_portfolio_round_trip_on_postgres(pg_repo, pg_settings):
    portfolio = PortfolioService(pg_repo, pg_settings).load_portfolio()
    assert len(portfolio.positions) == 20
    options = [p for p in portfolio.positions if p.option is not None]
    assert len(options) == 3 and all(o.option.volatility > 0 for o in options)  # vol joined from market data


def test_end_of_day_batch_on_postgres(pg_repo, pg_settings, tmp_path):
    risk = pg_settings.risk.model_copy(update={"lookback_days": 100})
    settings = pg_settings.model_copy(update={"risk": risk})
    result = run_end_of_day(pg_repo, settings, reports_dir=tmp_path)
    counts = pg_repo.table_counts()
    assert result.nav > 0 and result.var_99 > 0
    assert counts["risk_metrics"] == 12  # 6 methods x 2 confidence levels
    assert counts["risk_contributions"] == 20 and counts["stress_results"] == 8
    assert counts["backtest_results"] > 0 and counts["pnl_attribution"] > 0
    assert len(result.reports) == 2 and all(os.path.getsize(p) > 1000 for p in result.reports)
    # Re-running the batch upserts: no duplicate result rows.
    run_end_of_day(pg_repo, settings)
    assert pg_repo.table_counts() == counts
    assert last_run(pg_repo)["portfolio_id"] == result.portfolio_id
