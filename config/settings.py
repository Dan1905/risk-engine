"""Application configuration.

Infrastructure settings (database URL, log level, file paths) come from
environment variables / ``.env``. Risk parameters, limits and thresholds come
from ``config/risk_config.yaml`` so that nothing important is hard-coded.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


class PortfolioSettings(BaseModel):
    portfolio_id: str = "GMF001"
    name: str = "Global Macro Fund"
    base_currency: str = "USD"


class RiskSettings(BaseModel):
    confidence_level: float = 0.99
    lookback_days: int = 250
    holding_period: int = 1
    ewma_lambda: float = 0.94
    return_type: Literal["log", "simple"] = "log"
    covariance_method: Literal["SAMPLE", "EWMA"] = "EWMA"
    horizon_method: Literal["OVERLAPPING", "SQRT_TIME"] = "OVERLAPPING"
    vol_factor_model: Literal["LEVEL", "LEVEL_SKEW_TERM"] = "LEVEL_SKEW_TERM"
    trading_days_per_year: int = 252
    confidence_options: list[float] = Field(default_factory=lambda: [0.95, 0.99])
    lookback_options: list[int] = Field(default_factory=lambda: [100, 250, 500])
    holding_period_options: list[int] = Field(default_factory=lambda: [1, 5, 10])
    decay_options: list[float] = Field(default_factory=lambda: [0.90, 0.94, 0.97, 0.99])
    min_observations: int = 50
    rolling_vol_windows: list[int] = Field(default_factory=lambda: [20, 60])

    @field_validator("confidence_level")
    @classmethod
    def _confidence(cls, v: float) -> float:
        if not 0.5 < v < 1.0:
            raise ValueError("confidence_level must be in (0.5, 1)")
        return v

    @field_validator("ewma_lambda")
    @classmethod
    def _lambda(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("ewma_lambda must be in (0, 1)")
        return v


class LimitSettings(BaseModel):
    var_limit: float = 0.05
    es_limit: float = 0.07
    equity_exposure_limit: float = 0.60
    single_position_limit: float = 0.15
    stress_loss_limit: float = 0.15


class ConcentrationSettings(BaseModel):
    var_contribution_warning: float = 0.25
    currency_concentration_warning: float = 0.70
    asset_class_concentration_warning: float = 0.60
    stress_loss_contribution_warning: float = 0.40


class FxShockSetting(BaseModel):
    currency: str
    shock: float


class StressSettings(BaseModel):
    default_equity_shock: float = -0.20
    default_rate_shock_bps: float = 200
    equity_shock_ladder: list[float] = Field(default_factory=lambda: [-0.05, -0.10, -0.20, -0.30, -0.40])
    rate_shock_ladder_bps: list[float] = Field(default_factory=lambda: [25, 50, 100, 200])
    vol_shock_ladder: list[float] = Field(default_factory=lambda: [0.05, 0.10, 0.15, -0.05])
    fx_shocks: list[FxShockSetting] = Field(default_factory=list)
    default_vol_method: Literal["STICKY_STRIKE", "STICKY_DELTA"] = "STICKY_STRIKE"
    default_bond_method: Literal["FULL_REPRICING", "SENSITIVITY"] = "FULL_REPRICING"
    min_volatility: float = 0.01
    beta_lookback_days: int = 250


class BacktestSettings(BaseModel):
    significance_level: float = 0.05
    rolling_window: int = 60
    default_method: Literal["HISTORICAL", "EW_HISTORICAL", "PARAMETRIC", "EW_PARAMETRIC", "FILTERED_HISTORICAL", "MONTE_CARLO"] = "HISTORICAL"


class PnLSettings(BaseModel):
    tolerance: float = 0.01
    attribution_days: int = 60


class SimulationSettings(BaseModel):
    """Filtered historical simulation and Monte Carlo parameters."""

    monte_carlo_paths: int = 10_000
    backtest_paths: int = 1_000
    distribution: Literal["NORMAL", "STUDENT_T"] = "STUDENT_T"
    degrees_of_freedom: float = 6.0
    seed: int = 2026
    antithetic: bool = True
    fhs_ratio_floor: float = 1.0 / 3.0
    fhs_ratio_cap: float = 3.0
    fhs_seed_observations: int = 20

    @field_validator("degrees_of_freedom")
    @classmethod
    def _dof(cls, v: float) -> float:
        if v <= 2:
            raise ValueError("degrees_of_freedom must exceed 2")
        return v


class MarketDataSettings(BaseModel):
    seed: int = 42
    n_days: int = 756
    end_date: str = "2026-09-30"
    max_forward_fill_days: int = 5


class AppSettings(BaseModel):
    """Top-level settings object combining environment and YAML configuration."""

    database_url: str
    log_level: str = "INFO"
    log_file: Path
    sample_data_dir: Path
    scenarios_path: Path
    risk_config_path: Path
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    limits: LimitSettings = Field(default_factory=LimitSettings)
    concentration: ConcentrationSettings = Field(default_factory=ConcentrationSettings)
    stress: StressSettings = Field(default_factory=StressSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    pnl: PnLSettings = Field(default_factory=PnLSettings)
    market_data: MarketDataSettings = Field(default_factory=MarketDataSettings)


def _resolve_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_database_url() -> str:
    db_path = PROJECT_ROOT / "data" / "processed" / "risk_engine.db"
    return f"sqlite:///{db_path}"


def _resolve_database_url(value: str | None) -> str:
    """Default to the bundled SQLite file; make relative SQLite paths project-relative."""
    if not value:
        return _default_database_url()
    prefix = "sqlite:///"
    if value.startswith(prefix) and not value.startswith(prefix + "/") and ":memory:" not in value:
        return prefix + str(PROJECT_ROOT / value[len(prefix):])
    return value


def load_yaml(path: Path) -> dict:
    """Load a YAML file, returning an empty dict when the file is empty."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the (cached) application settings.

    Environment variables:
        DATABASE_URL       SQLAlchemy URL. Defaults to a SQLite file under data/processed.
                           Swap for ``postgresql+psycopg://user:pwd@host/db`` to use PostgreSQL.
        LOG_LEVEL          Python logging level name (default INFO).
        LOG_FILE           Log file path (default logs/risk_engine.log).
        RISK_CONFIG_PATH   YAML risk configuration (default config/risk_config.yaml).
        SCENARIOS_PATH     YAML stress scenarios (default config/scenarios.yaml).
        SAMPLE_DATA_DIR    Directory for generated sample CSVs (default data/sample).
    """
    risk_config_path = _resolve_path(os.getenv("RISK_CONFIG_PATH"), PROJECT_ROOT / "config" / "risk_config.yaml")
    raw = load_yaml(risk_config_path) if risk_config_path.exists() else {}
    return AppSettings(
        database_url=_resolve_database_url(os.getenv("DATABASE_URL")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_file=_resolve_path(os.getenv("LOG_FILE"), PROJECT_ROOT / "logs" / "risk_engine.log"),
        sample_data_dir=_resolve_path(os.getenv("SAMPLE_DATA_DIR"), PROJECT_ROOT / "data" / "sample"),
        scenarios_path=_resolve_path(os.getenv("SCENARIOS_PATH"), PROJECT_ROOT / "config" / "scenarios.yaml"),
        risk_config_path=risk_config_path,
        **raw,
    )


_LOGGING_CONFIGURED = False


def configure_logging(settings: AppSettings | None = None) -> None:
    """Configure root logging once: console + rotating file handler.

    Technical detail goes to the log; the UI only shows user-friendly messages.
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    settings = settings or get_settings()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            settings.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except OSError:  # read-only file systems: console logging only
        root.warning("File logging unavailable; using console logging only")
    for noisy in ("sqlalchemy.engine", "urllib3", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _LOGGING_CONFIGURED = True
