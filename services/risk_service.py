"""Risk service: VaR / ES under four methodologies, risk decomposition,
sensitivities, limits and concentration.

Historical simulation pipeline (spec section 9)
    1. take the last ``lookback`` daily market-factor moves up to the valuation date
    2. apply each move to today's positions
    3. fully revalue bonds and options
    4. sum to a hypothetical portfolio P&L per scenario
    5-6. read VaR / ES off the (weighted) empirical loss distribution

Parametric VaR uses the same hypothetical P&L to build position return series
``r_i = PnL_i / MV_i`` and weights ``w_i = MV_i / NAV`` so that
``w' Sigma w`` is the variance of portfolio returns (sample or EWMA ``Sigma``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from analytics.correlation import cov_to_corr
from analytics.covariance import covariance_matrix, ensure_psd
from analytics.ewma import ewma_covariance
from analytics.expected_shortfall import historical_es, parametric_es, weighted_historical_es
from analytics.key_rate_duration import aggregate_krd
from analytics.filtered_simulation import filter_moves
from analytics.monte_carlo import simulate_factor_moves
from analytics.revaluation import PortfolioPricer, ScenarioSet
from analytics.risk_factors import FACTOR_GROUP_LABELS, aggregate_horizon, kind_of, select_vol_factors
from analytics.risk_limits import ConcentrationReport, concentration_report, evaluate_limits
from analytics.var import (
    component_var,
    historical_var,
    marginal_var,
    parametric_var,
    pnl_summary,
    portfolio_volatility,
    scale_to_horizon,
    weighted_historical_var,
)
from config.settings import AppSettings
from database.repository import RiskRepository
from models.portfolio import Portfolio
from models.risk import LimitCheck, RiskConfiguration, RiskResult
from services.market_data_service import MarketHistory
from services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)

BASE_METHODS: tuple[str, ...] = ("HISTORICAL", "EW_HISTORICAL", "PARAMETRIC", "EW_PARAMETRIC")
SIMULATION_METHODS: tuple[str, ...] = ("FILTERED_HISTORICAL", "MONTE_CARLO")
METHODS: tuple[str, ...] = BASE_METHODS + SIMULATION_METHODS
METHOD_LABELS: dict[str, str] = {
    "HISTORICAL": "Historical",
    "EW_HISTORICAL": "EW Historical",
    "PARAMETRIC": "Parametric",
    "EW_PARAMETRIC": "EW Parametric",
    "FILTERED_HISTORICAL": "Filtered Historical",
    "MONTE_CARLO": "Monte Carlo",
}


@dataclass
class RiskContext:
    """Everything a calculation needs for one portfolio on its valuation date."""

    portfolio: Portfolio
    history: MarketHistory
    pricer: PortfolioPricer
    valuation: pd.DataFrame

    @property
    def state(self):
        return self.pricer.base_state

    @property
    def valuation_date(self) -> date:
        return self.portfolio.valuation_date

    @property
    def market_values(self) -> pd.Series:
        return self.valuation.set_index("position_id")["market_value"]

    @property
    def nav(self) -> float:
        return float(self.market_values.sum())

    @property
    def weights(self) -> pd.Series:
        return self.market_values / self.nav


def build_context(portfolio: Portfolio, history: MarketHistory, settings: AppSettings) -> RiskContext:
    state = history.state_at(portfolio.valuation_date)
    pricer = PortfolioPricer(portfolio, state, settings.stress.min_volatility)
    valuation = PortfolioService.valuation(portfolio, pricer, history.instruments)
    return RiskContext(portfolio, history, pricer, valuation)


@dataclass
class VarReport:
    config: RiskConfiguration
    nav: float
    results: dict[str, dict[float, RiskResult]]
    pnl_matrix: pd.DataFrame
    returns: pd.DataFrame
    weights: pd.Series
    cov: dict[str, pd.DataFrame] = field(default_factory=dict)
    distributions: dict[str, pd.Series] = field(default_factory=dict)  # method -> scenario P&L at the scenario horizon
    factor_cov: pd.DataFrame | None = None  # daily factor covariance used by Monte Carlo

    @property
    def portfolio_pnl(self) -> pd.Series:
        return self.pnl_matrix.sum(axis=1).rename("pnl")

    @property
    def overlapping(self) -> bool:
        """True when scenarios are genuine multi-day moves (no square-root-of-time scaling)."""
        return self.config.horizon_method == "OVERLAPPING" and self.config.holding_period > 1

    @property
    def scale_days(self) -> int:
        """Days by which scenario-horizon figures are scaled with sqrt(time) (1 = none)."""
        return 1 if self.overlapping else self.config.holding_period

    @property
    def sqrt_scaled(self) -> tuple[str, ...]:
        """Methods whose multi-day figure is a 1-day estimate scaled by sqrt(h)."""
        if self.config.holding_period == 1:
            return ()
        return ("EW_HISTORICAL", "EW_PARAMETRIC") if self.overlapping else METHODS

    @property
    def scenario_days(self) -> int:
        """Horizon (days) of each scenario P&L in ``distributions``."""
        return self.config.holding_period if self.overlapping else 1

    def get(self, method: str, confidence: float | None = None) -> RiskResult:
        return self.results[method][confidence or self.config.confidence]

    def comparison_table(self, confidences: tuple[float, ...] = (0.95, 0.99), es_confidence: float = 0.99) -> pd.DataFrame:
        rows = {}
        for c in confidences:
            rows[f"VaR {c:.0%}"] = {METHOD_LABELS[m]: self.results[m][c].var for m in METHODS}
        rows[f"ES {es_confidence:.0%}"] = {METHOD_LABELS[m]: self.results[m][es_confidence].expected_shortfall for m in METHODS}
        return pd.DataFrame(rows).T

    def cov_method_key(self, method: str) -> str:
        return "EW_PARAMETRIC" if method == "EWMA" else "PARAMETRIC"


class RiskService:
    def __init__(self, settings: AppSettings, repo: RiskRepository | None = None):
        self.settings = settings
        self.repo = repo

    # ----------------------------------------------------------- scenarios
    @staticmethod
    def scenario_pnl(ctx: RiskContext, lookback: int, confidence: float | None = None) -> pd.DataFrame:
        """Hypothetical 1-day P&L matrix (lookback x positions) at the valuation date."""
        scen = ctx.history.historical_scenarios(ctx.valuation_date, lookback, confidence)
        return ctx.pricer.scenario_pnl(ctx.state, scen)

    @staticmethod
    def position_returns(pnl_matrix: pd.DataFrame, market_values: pd.Series, return_type: str = "log") -> pd.DataFrame:
        """Scenario returns of each position (``PnL / MV``; log-transformed if requested)."""
        mv = market_values.reindex(pnl_matrix.columns)
        safe = mv.where(mv.abs() > 1e-9)
        r = pnl_matrix.div(safe, axis=1).fillna(0.0)
        if return_type == "log":
            r = np.log1p(r.clip(lower=-0.999))
        return r

    # ------------------------------------------------------------------ VaR
    def scenario_frames(self, ctx: RiskContext, config: RiskConfiguration) -> dict[str, pd.DataFrame]:
        """Factor-move scenarios for each simulation methodology.

        Returns frames keyed ``HISTORICAL`` (plain replay), ``FILTERED_HISTORICAL``
        (volatility-rescaled replay), ``MONTE_CARLO`` (simulated) plus
        ``FACTOR_COV`` (the daily factor covariance behind the simulation).
        With ``horizon_method == OVERLAPPING`` every scenario is a genuine
        h-day move built from ``lookback + h - 1`` daily moves (returns
        compounded, curve / spread / vol changes summed), so positions are
        revalued at the multi-day move and non-linearity is captured. The
        volatility filter is applied to the daily moves *before* they are
        aggregated; Monte Carlo simulates the h-day move directly (daily
        covariance x h). Otherwise scenarios are 1-day moves and the caller
        scales the resulting risk figures by sqrt(h).

        Scenarios are instantaneous shocks: option time decay and bond carry
        over the horizon are not included, so the measure is pure market risk.
        """
        sim = self.settings.simulation
        h, lam, lookback = config.holding_period, config.decay_factor, config.lookback
        overlap = config.horizon_method == "OVERLAPPING" and h > 1
        extra = h - 1 if overlap else 0
        hist = ctx.history
        daily = select_vol_factors(hist.historical_moves(ctx.valuation_date, lookback, config.confidence, extra_days=extra), config.vol_factors)
        i = hist.index_of(ctx.valuation_date)
        # all moves up to the valuation date: warms up the volatility filter
        everything = select_vol_factors(hist.daily_moves(i, i), config.vol_factors)
        filtered = filter_moves(everything, lookback + extra, lam, (sim.fhs_ratio_floor, sim.fhs_ratio_cap), sim.fhs_seed_observations)
        window = daily.iloc[-lookback:]
        factor_cov = ensure_psd(ewma_covariance(window, lam) if config.covariance_method == "EWMA" else window.cov(ddof=1))
        simulated = simulate_factor_moves(
            factor_cov, sim.monte_carlo_paths, sim.seed, sim.distribution, sim.degrees_of_freedom,
            horizon=h if overlap else 1, antithetic=sim.antithetic,
        )
        return {
            "DAILY": window,
            "HISTORICAL": aggregate_horizon(daily, h) if overlap else daily,
            "FILTERED_HISTORICAL": aggregate_horizon(filtered, h) if overlap else filtered,
            "MONTE_CARLO": simulated,
            "FACTOR_COV": factor_cov,
        }

    def factor_group_var(self, ctx: RiskContext, config: RiskConfiguration, confidence: float | None = None) -> pd.DataFrame:
        """Stand-alone 1-day historical VaR by risk-factor group.

        Each row revalues the portfolio with only that group's historical moves
        (all other factors held unchanged), which shows what each type of
        market risk contributes on its own. Stand-alone figures do not add up
        to the total: the difference is the diversification between groups.
        """
        c = confidence or config.confidence
        daily = select_vol_factors(ctx.history.historical_moves(ctx.valuation_date, config.lookback, c), config.vol_factors)
        kinds = pd.Series([kind_of(col) for col in daily.columns], index=daily.columns)
        rows = []
        for kind in dict.fromkeys(kinds):  # preserves factor-frame order
            cols = kinds.index[kinds == kind]
            pnl = ctx.pricer.scenario_pnl(ctx.state, ScenarioSet.from_frame(daily[cols])).sum(axis=1)
            rows.append(
                {
                    "group": FACTOR_GROUP_LABELS.get(kind, kind), "factors": len(cols),
                    "var": historical_var(pnl, c), "expected_shortfall": historical_es(pnl, c),
                }
            )
        total = ctx.pricer.scenario_pnl(ctx.state, ScenarioSet.from_frame(daily)).sum(axis=1)
        out = pd.DataFrame(rows)
        standalone = float(out["var"].sum())
        out.loc[len(out)] = {"group": "All factors (diversified)", "factors": daily.shape[1], "var": historical_var(total, c),
                             "expected_shortfall": historical_es(total, c)}
        out["share_of_standalone"] = out["var"] / standalone if standalone else np.nan
        return out

    def var_report(self, ctx: RiskContext, config: RiskConfiguration) -> VarReport:
        """VaR and ES for every methodology at 95%, 99% and the configured confidence.

        Multi-day horizons (``holding_period`` > 1):

        * ``SQRT_TIME``: every method computes 1-day figures and scales by sqrt(h).
        * ``OVERLAPPING``: Historical, Filtered Historical, Monte Carlo and the
          sample-covariance Parametric method use genuine h-day scenarios. The
          two exponentially weighted estimators (EW Historical, EW Parametric)
          are defined on daily data - with lambda = 0.94 only ~17 days carry
          weight, i.e. fewer than two independent 10-day periods - so they keep
          their 1-day estimate scaled by sqrt(h). ``VarReport.sqrt_scaled``
          lists the methods that were scaled.
        """
        min_obs = self.settings.risk.min_observations
        if config.lookback < min_obs:
            raise ValueError(f"Lookback of {config.lookback} is below the configured minimum of {min_obs} observations.")
        frames = self.scenario_frames(ctx, config)
        price = lambda frame: ctx.pricer.scenario_pnl(ctx.state, ScenarioSet.from_frame(frame))  # noqa: E731
        pnl = price(frames["HISTORICAL"])
        port = pnl.sum(axis=1)
        dist = {"HISTORICAL": port.rename("pnl")}
        for m in SIMULATION_METHODS:
            dist[m] = price(frames[m]).sum(axis=1).rename("pnl")
        nav, lam, hp = ctx.nav, config.decay_factor, config.holding_period
        overlap = config.horizon_method == "OVERLAPPING" and hp > 1
        h = 1 if overlap else hp  # sqrt-of-time scaling for scenario methods only when scenarios are 1-day moves
        returns = self.position_returns(pnl, ctx.market_values, config.return_type)
        weights = ctx.weights.reindex(pnl.columns)
        # Exponentially weighted estimators always work on 1-day scenarios.
        pnl_1d = price(frames["DAILY"]) if overlap else pnl
        port_1d = pnl_1d.sum(axis=1)
        returns_1d = self.position_returns(pnl_1d, ctx.market_values, config.return_type) if overlap else returns
        # Both covariance matrices are expressed at the scenario horizon (h days when overlapping).
        ewma_scale = hp if overlap else 1
        cov = {"SAMPLE": covariance_matrix(returns, "SAMPLE"), "EWMA": covariance_matrix(returns_1d, "EWMA", lam) * ewma_scale}
        confidences = sorted({0.95, 0.99, config.confidence})
        results: dict[str, dict[float, RiskResult]] = {m: {} for m in METHODS}
        for c in confidences:
            common = dict(confidence=c, lookback=config.lookback, holding_period=hp, portfolio_value=nav)
            for method, series, decay in (
                ("HISTORICAL", port, None), ("FILTERED_HISTORICAL", dist["FILTERED_HISTORICAL"], lam),
                ("MONTE_CARLO", dist["MONTE_CARLO"], lam if config.covariance_method == "EWMA" else None),
            ):
                results[method][c] = RiskResult(
                    method=method, decay_factor=decay,
                    var=scale_to_horizon(historical_var(series, c), h), expected_shortfall=scale_to_horizon(historical_es(series, c), h),
                    volatility=float(series.std(ddof=1)), n_observations=len(series), **pnl_summary(series), **common,
                )
            results["EW_HISTORICAL"][c] = RiskResult(
                method="EW_HISTORICAL", decay_factor=lam,
                var=scale_to_horizon(weighted_historical_var(port_1d, c, lam), hp),
                expected_shortfall=scale_to_horizon(weighted_historical_es(port_1d, c, lam), hp),
                volatility=float(port_1d.std(ddof=1)), n_observations=len(port_1d), **pnl_summary(port_1d), **common,
            )
            for method, key in (("PARAMETRIC", "SAMPLE"), ("EW_PARAMETRIC", "EWMA")):
                sigma = portfolio_volatility(weights, cov[key])
                results[method][c] = RiskResult(
                    method=method, decay_factor=lam if key == "EWMA" else None,
                    var=parametric_var(weights, cov[key], c, nav, h), expected_shortfall=parametric_es(sigma, c, nav, h),
                    volatility=sigma * nav, n_observations=len(port), **common,
                )
        logger.info("Historical VaR calculation completed (lookback=%d, h=%d, %s)", config.lookback, hp, config.horizon_method)
        logger.info("EWMA covariance calculated (lambda=%.2f)", lam)
        logger.info("Monte Carlo VaR completed (%d paths, %s)", len(dist["MONTE_CARLO"]), self.settings.simulation.distribution)
        report = VarReport(config, nav, results, pnl, returns, weights, cov, dist, frames["FACTOR_COV"])
        self._persist_metrics(ctx, report)
        return report

    def _persist_metrics(self, ctx: RiskContext, report: VarReport) -> None:
        if self.repo is None or ctx.portfolio.portfolio_id.startswith("UPLOAD"):
            return
        rows = []
        for method, by_conf in report.results.items():
            for c, r in by_conf.items():
                rows.append(
                    {
                        "portfolio_id": ctx.portfolio.portfolio_id, "date": ctx.valuation_date, "method": method,
                        "confidence": c, "lookback": r.lookback, "holding_period": r.holding_period,
                        "decay_factor": report.config.decay_factor, "var": r.var, "expected_shortfall": r.expected_shortfall,
                        "volatility": r.volatility, "portfolio_value": r.portfolio_value,
                    }
                )
        try:
            self.repo.save_risk_metrics(rows)
        except Exception:  # noqa: BLE001 - persistence must not break the calculation
            logger.exception("Failed to persist risk metrics")

    # --------------------------------------------------------- contributions
    def contributions(self, ctx: RiskContext, report: VarReport, cov_method: str = "EWMA", confidence: float | None = None) -> pd.DataFrame:
        """Marginal / component / % VaR contributions (parametric Euler allocation)."""
        c = confidence or report.config.confidence
        cov = report.cov[cov_method]
        w = report.weights.to_numpy()
        h = report.scale_days
        mvar = marginal_var(w, cov, c) * np.sqrt(h)
        comp = component_var(w, cov, c, report.nav, h)
        total = parametric_var(w, cov, c, report.nav, h)
        val = ctx.valuation.set_index("position_id").reindex(report.weights.index)
        df = pd.DataFrame(
            {
                "position_id": report.weights.index,
                "ticker": val["ticker"].to_numpy(),
                "asset_class": val["asset_class"].to_numpy(),
                "currency": val["exposure_currency"].to_numpy(),
                "exposure": val["market_value"].to_numpy(),
                "weight": w,
                "marginal_var": mvar,
                "component_var": comp,
                "pct_contribution": comp / total if total else np.nan,
            }
        )
        if self.repo is not None and not ctx.portfolio.portfolio_id.startswith("UPLOAD"):
            try:
                self.repo.save_risk_contributions(
                    [
                        {
                            "portfolio_id": ctx.portfolio.portfolio_id, "date": ctx.valuation_date,
                            "method": f"PARAMETRIC_{cov_method}", "position_id": r.position_id, "exposure": r.exposure,
                            "marginal_var": r.marginal_var, "component_var": r.component_var, "pct_contribution": r.pct_contribution,
                        }
                        for r in df.itertuples()
                    ]
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist risk contributions")
        return df

    @staticmethod
    def contributions_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
        out = df.groupby(key).agg(exposure=("exposure", "sum"), component_var=("component_var", "sum"))
        out["pct_contribution"] = out["component_var"] / df["component_var"].sum()
        return out.sort_values("component_var", ascending=False)

    # -------------------------------------------------------- vol / corr
    @staticmethod
    def volatility_comparison(report: VarReport, periods: int = 252) -> pd.DataFrame:
        """Annualised position volatilities (the covariance horizon is ``report.scenario_days``)."""
        ann = np.sqrt(periods / report.scenario_days)
        hist = np.sqrt(np.diag(report.cov["SAMPLE"])) * ann
        ewma = np.sqrt(np.diag(report.cov["EWMA"])) * ann
        df = pd.DataFrame({"historical_vol": hist, "ewma_vol": ewma}, index=report.returns.columns)
        df["difference"] = df["ewma_vol"] - df["historical_vol"]
        return df

    @staticmethod
    def portfolio_volatility_annual(report: VarReport, cov_method: str, periods: int = 252) -> float:
        """Annualised portfolio volatility as a fraction of NAV."""
        return portfolio_volatility(report.weights, report.cov[cov_method]) * np.sqrt(periods / report.scenario_days)

    @staticmethod
    def correlation(report: VarReport, cov_method: str) -> pd.DataFrame:
        return cov_to_corr(report.cov[cov_method])

    # -------------------------------------------------------- sensitivities
    @staticmethod
    def sensitivities(ctx: RiskContext) -> dict[str, pd.DataFrame | pd.Series]:
        options = ctx.pricer.option_analytics()
        bonds = ctx.pricer.bond_analytics()
        krd = ctx.pricer.key_rate_durations()
        krd_agg = aggregate_krd(krd, ctx.market_values) if not krd.empty else pd.DataFrame()
        option_totals = (
            options[["delta", "delta_usd", "gamma", "gamma_usd", "vega", "theta", "rho"]].sum()
            if not options.empty else pd.Series(dtype=float)
        )
        bond_totals = pd.Series(dtype=float)
        if not bonds.empty:
            mv = bonds["market_value"]
            bond_totals = pd.Series(
                {
                    "market_value": mv.sum(),
                    "dv01": bonds["dv01"].sum(),
                    "modified_duration": float((bonds["modified_duration"] * mv).sum() / mv.sum()),
                    "macaulay_duration": float((bonds["macaulay_duration"] * mv).sum() / mv.sum()),
                    "convexity": float((bonds["convexity"] * mv).sum() / mv.sum()),
                    "ytm": float((bonds["ytm"] * mv).sum() / mv.sum()),
                }
            )
        return {"options": options, "bonds": bonds, "krd": krd, "krd_agg": krd_agg, "option_totals": option_totals, "bond_totals": bond_totals}

    # ------------------------------------------------------- limits & conc.
    def limit_checks(self, ctx: RiskContext, var: RiskResult) -> list[LimitCheck]:
        exp = PortfolioService.exposures(ctx.valuation)
        largest = float((ctx.market_values.abs() / ctx.nav).max())
        return evaluate_limits(var.var_pct, var.es_pct, exp["equity"] / ctx.nav, largest, self.settings.limits)

    def concentration(
        self, ctx: RiskContext, contributions: pd.DataFrame | None, worst_stress_pnl: pd.Series | None, worst_stress_name: str | None
    ) -> ConcentrationReport:
        comp = contributions.set_index("position_id")["component_var"] if contributions is not None else None
        return concentration_report(ctx.valuation, comp, worst_stress_pnl, worst_stress_name, self.settings.limits, self.settings.concentration)


def z_score(confidence: float) -> float:
    return float(norm.ppf(confidence))
