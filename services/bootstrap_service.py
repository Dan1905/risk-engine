"""First-run initialisation.

    Check DB -> create schema -> seeded? -> (generate sample data -> seed) -> validate -> ready

The whole sequence is idempotent: on subsequent launches it finds the seed
marker and skips generation / seeding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import AppSettings
from database.connection import check_connection, database_exists, get_engine
from database.repository import RiskRepository
from scripts.generate_sample_data import generate_sample_data
from scripts.seed_database import is_seeded, needs_refresh, sample_files_present, seed_database
from stress.scenario_engine import load_scenarios, scenario_to_dict

logger = logging.getLogger(__name__)


@dataclass
class StartupStep:
    label: str
    ok: bool
    detail: str = ""


@dataclass
class StartupStatus:
    steps: list[StartupStep] = field(default_factory=list)
    seeded_now: bool = False

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def add(self, label: str, ok: bool, detail: str = "") -> None:
        self.steps.append(StartupStep(label, ok, detail))


class InitialisationError(RuntimeError):
    def __init__(self, status: StartupStatus, message: str):
        self.status = status
        super().__init__(message)


def initialise_database(settings: AppSettings) -> tuple[RiskRepository, StartupStatus]:
    """Ensure the database exists, has a schema and contains sample data."""
    status = StartupStatus()
    existed = database_exists(settings.database_url)
    engine = get_engine(settings.database_url)
    if not check_connection(engine):
        status.add("Database connected", False, "Could not connect to the configured database.")
        raise InitialisationError(status, "Database connection failed")
    repo = RiskRepository(engine)
    repo.create_schema()
    status.add("Database connected", True, "existing database" if existed else "new database created")

    if is_seeded(repo):
        logger.info("Seed marker found; skipping sample data seeding")
    else:
        data_dir = settings.sample_data_dir
        if not sample_files_present(data_dir):
            md = settings.market_data
            logger.info("Generating sample data (seed=%d, days=%d)", md.seed, md.n_days)
            generate_sample_data(data_dir, md.seed, md.n_days, md.end_date)
        refresh = needs_refresh(repo)
        if refresh:
            logger.info("Sample dataset version changed; refreshing seeded data in place")
        seed_database(repo, settings, data_dir, refresh=refresh)
        repo.set_meta("market_data_version", repo.get_meta("seeded_at") or "")  # invalidates cached calculations
        status.seeded_now = True
    sync_scenario_library(repo, settings)
    return repo, status


def sync_scenario_library(repo: RiskRepository, settings: AppSettings) -> int:
    """Upsert the predefined scenarios from YAML so edits reach an already-seeded database.

    Idempotent (keyed on scenario id); user-saved custom scenarios are untouched.
    """
    try:
        scenarios = load_scenarios(settings.scenarios_path)
        return repo.upsert_scenarios([scenario_to_dict(s) for s in scenarios])
    except Exception:  # noqa: BLE001 - a bad scenario file must not stop the application
        logger.exception("Could not synchronise the scenario library from %s", settings.scenarios_path)
        return 0
