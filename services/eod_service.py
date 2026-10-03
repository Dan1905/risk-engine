"""End-of-day batch run.

Runs the full risk process for one portfolio without the dashboard and
persists every result, so the database holds an auditable daily record:

    valuation -> VaR / ES (all methods) -> risk contributions -> stress library
    -> P&L attribution -> VaR backtests (all models) -> limit checks -> reports

Every write is an upsert on the table's natural key, so re-running the batch
for the same valuation date replaces that day's figures instead of
duplicating them. The run works unchanged on SQLite and PostgreSQL.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from config.settings import AppSettings
from database.repository import RiskRepository
from models.risk import RiskConfiguration
from services.backtest_service import BacktestService
from services.market_data_service import load_market_history
from services.pnl_service import PnLService
from services.portfolio_service import PortfolioService
from services.report_service import ReportService
from services.risk_service import METHODS, RiskService, build_context
from services.stress_service import StressService

logger = logging.getLogger(__name__)

EOD_META_KEY = "eod_last_run"


@dataclass
class EodResult:
    """Summary of one end-of-day run (also stored as JSON in ``app_metadata``)."""

    portfolio_id: str
    valuation_date: str
    started_at: str
    duration_seconds: float = 0.0
    nav: float = 0.0
    var_99: float = 0.0
    es_99: float = 0.0
    worst_stress: str = ""
    worst_stress_pnl: float = 0.0
    backtest_exceptions: dict[str, int] = field(default_factory=dict)
    basel_zones: dict[str, str] = field(default_factory=dict)
    limit_breaches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    rows_written: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.limit_breaches

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def default_configuration(settings: AppSettings) -> RiskConfiguration:
    """Risk configuration taken from ``config/risk_config.yaml``."""
    r = settings.risk
    return RiskConfiguration(
        confidence=r.confidence_level, lookback=r.lookback_days, holding_period=r.holding_period, decay_factor=r.ewma_lambda,
        covariance_method=r.covariance_method, return_type=r.return_type, horizon_method=r.horizon_method,
        vol_factors=r.vol_factor_model,
    )


def run_end_of_day(
    repo: RiskRepository,
    settings: AppSettings,
    portfolio_id: str | None = None,
    reports_dir: Path | None = None,
) -> EodResult:
    """Run and persist the full risk process for one portfolio.

    Args:
        repo: repository bound to the target database.
        settings: application settings (risk parameters, limits, scenarios).
        portfolio_id: portfolio to process (default: the configured portfolio).
        reports_dir: if given, the Excel workbook and PDF summary are written
            to ``<reports_dir>/<portfolio>_<yyyymmdd>/``.

    Raises:
        Exception: any failure is logged and re-raised so a scheduler can alert.
    """
    started = time.perf_counter()
    before = repo.table_counts()
    portfolio = PortfolioService(repo, settings).load_portfolio(portfolio_id)
    history = load_market_history(repo, portfolio.valuation_date, settings.market_data.max_forward_fill_days)
    ctx = build_context(portfolio, history, settings)
    config = default_configuration(settings)
    result = EodResult(
        portfolio_id=portfolio.portfolio_id, valuation_date=str(portfolio.valuation_date),
        started_at=datetime.now().isoformat(timespec="seconds"), nav=ctx.nav,
    )
    logger.info("EOD run started: %s valuation %s NAV %.0f", portfolio.portfolio_id, portfolio.valuation_date, ctx.nav)

    risk = RiskService(settings, repo)
    report = risk.var_report(ctx, config)  # persists risk_metrics
    headline = report.get("HISTORICAL", 0.99)
    result.var_99, result.es_99 = headline.var, headline.expected_shortfall
    contributions = risk.contributions(ctx, report, config.covariance_method)  # persists risk_contributions

    stress = StressService(settings, repo)
    stress_results = stress.run_predefined(ctx, settings.stress.default_vol_method, settings.stress.default_bond_method)
    worst = stress.worst(stress_results)
    if worst is not None:
        result.worst_stress, result.worst_stress_pnl = worst.scenario.name, worst.total_pnl

    attributions, attribution_frame = PnLService(settings, repo).attribution_history(ctx)  # persists pnl_attribution

    backtest = BacktestService(settings, repo)
    windows = backtest.forecast_windows(ctx, config.lookback, config.vol_factors)
    default_bt = None
    for method in METHODS:
        bt = backtest.run(ctx, windows, method, config.confidence, lam=config.decay_factor, cov_method=config.covariance_method)
        result.backtest_exceptions[method] = bt.kupiec.n_exceptions
        result.basel_zones[method] = backtest.basel_assessment(
            windows, method, config.decay_factor, ctx=ctx, cov_method=config.covariance_method
        ).zone
        if method == settings.backtest.default_method:
            default_bt = bt

    limits = risk.limit_checks(ctx, headline)
    result.limit_breaches = [f"{c.name}: {c.value:.2%} vs limit {c.limit:.2%}" for c in limits if c.status == "Breach"]
    concentration = risk.concentration(
        ctx, contributions, worst.position_pnl if worst else None, worst.scenario.name if worst else None
    )
    result.warnings = list(concentration.warnings)

    if reports_dir is not None and default_bt is not None:
        data = ReportService(settings).build(
            ctx=ctx, var_report=report, contributions=contributions,
            contributions_by_class=risk.contributions_by(contributions, "asset_class"),
            sensitivities=risk.sensitivities(ctx), stress_results=stress_results, attribution=attributions[-1],
            attribution_history=attribution_frame, backtest=default_bt, limit_checks=limits, concentration=concentration,
            portfolio_volatility=risk.portfolio_volatility_annual(report, config.covariance_method),
            basel=backtest.basel_assessment(
                windows, settings.backtest.default_method, config.decay_factor, ctx=ctx, cov_method=config.covariance_method
            ),
        )
        out = Path(reports_dir) / f"{portfolio.portfolio_id}_{portfolio.valuation_date:%Y%m%d}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "risk_report.xlsx").write_bytes(ReportService.excel(data))
        (out / "risk_summary.pdf").write_bytes(ReportService.pdf(data))
        result.reports = [str(out / "risk_report.xlsx"), str(out / "risk_summary.pdf")]

    after = repo.table_counts()
    result.rows_written = {k: after[k] - before.get(k, 0) for k in after if after[k] != before.get(k, 0)}
    result.duration_seconds = round(time.perf_counter() - started, 2)
    repo.set_meta(EOD_META_KEY, result.to_json())
    for breach in result.limit_breaches:
        logger.warning("Risk limit breach: %s", breach)
    logger.info(
        "EOD run completed in %.1fs: VaR99 %.0f, ES99 %.0f, worst stress %s %.0f, %d limit breach(es)",
        result.duration_seconds, result.var_99, result.es_99, result.worst_stress, result.worst_stress_pnl, len(result.limit_breaches),
    )
    return result


def last_run(repo: RiskRepository) -> dict | None:
    """Summary of the most recent EOD run, if any."""
    raw = repo.get_meta(EOD_META_KEY)
    return json.loads(raw) if raw else None
