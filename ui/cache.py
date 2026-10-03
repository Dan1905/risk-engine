"""Streamlit caching and per-session application state.

Caching rules
-------------
* ``st.cache_resource`` holds immutable, shareable objects: the initialised
  repository, market histories, risk contexts and calculation results. Every
  cache key includes the portfolio content hash and the market-data version,
  so an uploaded portfolio or a market-data ingest can never be served stale
  results for another portfolio.
* User-specific, mutable state (selected page, uploaded portfolio, parameter
  choices, limit overrides) lives in ``st.session_state`` and is never cached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import streamlit as st

from config.settings import AppSettings, LimitSettings, get_settings
from database.repository import RiskRepository
from models.portfolio import Portfolio
from models.risk import RiskConfiguration
from services.backtest_service import BacktestService, ForecastWindows
from services.bootstrap_service import StartupStatus, initialise_database
from services.market_data_service import MarketHistory, load_market_history
from services.pnl_service import PnLService
from services.portfolio_service import PortfolioService
from services.risk_service import RiskContext, RiskService, VarReport, build_context
from services.stress_service import StressService

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- shared resources


@st.cache_resource(show_spinner="Initialising risk engine: checking database and seed status...")
def initialise() -> tuple[RiskRepository, StartupStatus]:
    return initialise_database(get_settings())


def data_version(repo: RiskRepository) -> str:
    return repo.get_meta("market_data_version") or repo.get_meta("seeded_at") or "0"


@st.cache_resource(show_spinner="Loading market data...")
def market_history(_repo: RiskRepository, end: date, version: str) -> MarketHistory:
    return load_market_history(_repo, end, get_settings().market_data.max_forward_fill_days)


@st.cache_resource(show_spinner="Loading portfolio...")
def stored_portfolio(_repo: RiskRepository, portfolio_id: str, version: str) -> Portfolio:
    return PortfolioService(_repo, get_settings()).load_portfolio(portfolio_id)


@st.cache_resource(show_spinner="Valuing portfolio...")
def risk_context(key: str, _portfolio: Portfolio, _history: MarketHistory) -> RiskContext:
    return build_context(_portfolio, _history, get_settings())


@st.cache_resource(show_spinner="Running VaR / ES calculations...")
def var_report(key: str, config_json: str, _ctx: RiskContext, _repo: RiskRepository) -> VarReport:
    return RiskService(get_settings(), _repo).var_report(_ctx, RiskConfiguration.model_validate_json(config_json))


@st.cache_resource(show_spinner="Running stress scenarios...")
def stress_predefined(key: str, vol_method: str, bond_method: str, beta_mapping: bool, _ctx: RiskContext, _repo: RiskRepository):
    return StressService(get_settings(), _repo).run_predefined(_ctx, vol_method, bond_method, beta_mapping)


@st.cache_resource(show_spinner="Running stress ladders...")
def stress_ladders(key: str, vol_method: str, bond_method: str, _ctx: RiskContext, _repo: RiskRepository):
    return StressService(get_settings(), _repo).run_ladders(_ctx, vol_method, bond_method)


@st.cache_resource(show_spinner="Attributing P&L...")
def attribution_history(key: str, n_days: int, _ctx: RiskContext, _repo: RiskRepository):
    return PnLService(get_settings(), _repo).attribution_history(_ctx, n_days)


@st.cache_resource(show_spinner="Building out-of-sample VaR forecasts (no look-ahead)...")
def forecast_windows(key: str, lookback: int, vol_factors: str, _ctx: RiskContext) -> ForecastWindows:
    return BacktestService(get_settings()).forecast_windows(_ctx, lookback, vol_factors)


@st.cache_resource(show_spinner=False)
def sensitivities(key: str, _ctx: RiskContext):
    return RiskService.sensitivities(_ctx)


# --------------------------------------------------------------------- app state


@dataclass
class App:
    """Everything a view needs for the current session."""

    settings: AppSettings
    repo: RiskRepository
    status: StartupStatus
    portfolio: Portfolio
    history: MarketHistory
    ctx: RiskContext
    version: str

    @property
    def key(self) -> str:
        return f"{self.portfolio.key}:{self.version}"

    # services (cheap to construct; they hold no state beyond settings/repo)
    @property
    def risk(self) -> RiskService:
        return RiskService(self.settings, self.repo)

    @property
    def stress(self) -> StressService:
        return StressService(self.settings, self.repo)

    @property
    def backtest(self) -> BacktestService:
        return BacktestService(self.settings, self.repo)

    @property
    def portfolio_service(self) -> PortfolioService:
        return PortfolioService(self.repo, self.settings)

    # parameters chosen by the user in this session
    def config(self) -> RiskConfiguration:
        r = self.settings.risk
        p = st.session_state.get("risk_params", {})
        return RiskConfiguration(
            confidence=p.get("confidence", r.confidence_level),
            lookback=p.get("lookback", r.lookback_days),
            holding_period=p.get("holding_period", r.holding_period),
            decay_factor=p.get("decay_factor", r.ewma_lambda),
            covariance_method=p.get("covariance_method", r.covariance_method),
            return_type=p.get("return_type", r.return_type),
            horizon_method=p.get("horizon_method", r.horizon_method),
            vol_factors=p.get("vol_factors", r.vol_factor_model),
        )

    def set_params(self, **kwargs) -> None:
        st.session_state.setdefault("risk_params", {}).update(kwargs)

    def vol_method(self) -> str:
        return st.session_state.get("vol_method", self.settings.stress.default_vol_method)

    def bond_method(self) -> str:
        return st.session_state.get("bond_method", self.settings.stress.default_bond_method)

    def beta_mapping(self) -> bool:
        """Whether historical scenarios scale the index equity move by each name's beta."""
        return bool(st.session_state.get("beta_mapping", True))

    # cached calculations
    def var_report(self, config: RiskConfiguration | None = None) -> VarReport:
        cfg = config or self.config()
        return var_report(self.key, cfg.model_dump_json(), self.ctx, self.repo)

    def stress_results(self, vol_method: str | None = None, bond_method: str | None = None):
        return stress_predefined(
            self.key, vol_method or self.vol_method(), bond_method or self.bond_method(), self.beta_mapping(), self.ctx, self.repo
        )

    def stress_ladders(self, vol_method: str | None = None, bond_method: str | None = None):
        return stress_ladders(self.key, vol_method or self.vol_method(), bond_method or self.bond_method(), self.ctx, self.repo)

    def attribution(self, n_days: int | None = None):
        return attribution_history(self.key, n_days or self.settings.pnl.attribution_days, self.ctx, self.repo)

    def forecasts(self, lookback: int | None = None) -> ForecastWindows:
        cfg = self.config()
        return forecast_windows(self.key, lookback or cfg.lookback, cfg.vol_factors, self.ctx)

    def rolling(self, lookback: int | None = None, methods: tuple[str, ...] | None = None):
        """Rolling out-of-sample VaR / volatility; simulation methods are built on first request."""
        cfg = self.config()
        fw = self.forecasts(lookback)
        kwargs = {"methods": methods} if methods else {}
        return self.backtest.rolling_risk(fw, cfg.confidence, cfg.decay_factor, ctx=self.ctx, cov_method=cfg.covariance_method, **kwargs)

    def sensitivities(self):
        return sensitivities(self.key, self.ctx)

    def meta(self) -> dict[str, str]:
        return {
            "Portfolio": self.portfolio.name,
            "Valuation Date": f"{self.portfolio.valuation_date:%d %b %Y}",
            "Base Currency": self.portfolio.base_currency,
        }


def effective_settings() -> AppSettings:
    """Settings with any session-level limit overrides applied."""
    base = get_settings()
    overrides = st.session_state.get("limit_overrides")
    if not overrides:
        return base
    return base.model_copy(update={"limits": LimitSettings(**{**base.limits.model_dump(), **overrides})})


def clear_calculation_caches() -> None:
    for fn in (market_history, stored_portfolio, risk_context, var_report, stress_predefined, stress_ladders,
               attribution_history, forecast_windows, sensitivities):
        fn.clear()


def build_app() -> App:
    """Resolve the active portfolio and market data for this session."""
    repo, status = initialise()
    settings = effective_settings()
    version = data_version(repo)
    uploaded: Portfolio | None = None
    if st.session_state.get("portfolio_source") == "UPLOAD" and st.session_state.get("uploaded_portfolio"):
        uploaded = Portfolio.model_validate_json(st.session_state["uploaded_portfolio"])
    if uploaded is not None:
        portfolio = uploaded
    else:
        pid = st.session_state.get("portfolio_source") or settings.portfolio.portfolio_id
        portfolio = stored_portfolio(repo, pid, version)
    history = market_history(repo, portfolio.valuation_date, version)
    ctx = risk_context(f"{portfolio.key}:{version}", portfolio, history)
    return App(settings, repo, status, portfolio, history, ctx, version)
