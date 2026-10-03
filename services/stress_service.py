"""Stress-testing service: predefined, ladder and custom scenarios."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from config.settings import AppSettings
from database.repository import RiskRepository
from models.risk import StressScenario
from services.risk_service import RiskContext
from stress.historical import replay_window_scenario
from stress.hypothetical import curve_scenarios, equity_ladder, fx_ladder, parallel_curve_scenarios, vol_ladder
from stress.scenario_engine import ScenarioEngine, StressResult, load_scenarios, results_frame, scenario_from_dict, scenario_to_dict

logger = logging.getLogger(__name__)

PREDEFINED_ORDER = (
    "GFC_2008",
    "COVID_2020",
    "EQUITY_CRASH",
    "RATES_UP_200",
    "GLOBAL_RISK_OFF",
    "CURVE_STEEPENER",
    "CURVE_FLATTENER",
    "CURVE_INVERSION",
)


class StressService:
    def __init__(self, settings: AppSettings, repo: RiskRepository | None = None):
        self.settings = settings
        self.repo = repo

    # ------------------------------------------------------------ library
    def predefined_scenarios(self) -> list[StressScenario]:
        """Predefined scenarios from the database (seeded from YAML); YAML fallback."""
        scenarios: list[StressScenario] = []
        if self.repo is not None:
            try:
                scenarios = [scenario_from_dict(d) for d in self.repo.load_scenarios() if d.get("is_predefined", True)]
            except Exception:  # noqa: BLE001
                logger.exception("Could not load scenarios from database; using YAML")
        if not scenarios:
            scenarios = load_scenarios(self.settings.scenarios_path)
        order = {sid: i for i, sid in enumerate(PREDEFINED_ORDER)}
        return sorted(scenarios, key=lambda s: order.get(s.scenario_id, len(order)))

    def custom_scenarios(self) -> list[StressScenario]:
        if self.repo is None:
            return []
        return [scenario_from_dict(d) for d in self.repo.load_scenarios() if not d.get("is_predefined", True)]

    def save_custom(self, scenario: StressScenario) -> None:
        if self.repo is not None:
            self.repo.upsert_scenarios([scenario_to_dict(scenario, predefined=False)])

    # ---------------------------------------------------------------- runs
    def betas(self, ctx: RiskContext) -> dict[str, float]:
        """Equity betas to the equal-weighted equity market, estimated up to the valuation date."""
        return ctx.history.equity_betas(ctx.valuation_date, self.settings.stress.beta_lookback_days)

    def engine(self, ctx: RiskContext) -> ScenarioEngine:
        return ScenarioEngine(ctx.portfolio, ctx.state, pricer=ctx.pricer, betas=self.betas(ctx))

    def run(self, ctx: RiskContext, scenario: StressScenario, vol_method: str, bond_method: str) -> StressResult:
        return self.engine(ctx).run(scenario, vol_method, bond_method)  # type: ignore[arg-type]

    def run_predefined(
        self, ctx: RiskContext, vol_method: str, bond_method: str, beta_mapping: bool | None = None
    ) -> list[StressResult]:
        """Run the scenario library.

        Args:
            beta_mapping: ``None`` uses each scenario's own setting; True / False
                forces beta-mapped / uniform equity shocks on the historical scenarios.
        """
        scenarios = self.predefined_scenarios()
        if beta_mapping is not None:
            scenarios = [
                s.model_copy(update={"equity_beta_mapping": beta_mapping}) if s.category == "HISTORICAL" else s for s in scenarios
            ]
        results = self.engine(ctx).run_many(scenarios, vol_method, bond_method)  # type: ignore[arg-type]
        self._persist(ctx, results)
        return results

    def replay(self, ctx: RiskContext, start: date, end: date, vol_method: str, bond_method: str) -> StressResult:
        """Stress today's portfolio with the stored market moves between two dates."""
        scenario = replay_window_scenario(ctx.history, start, end)
        return self.run(ctx, scenario, vol_method, bond_method)

    def run_ladders(self, ctx: RiskContext, vol_method: str, bond_method: str) -> dict[str, list[StressResult]]:
        eng = self.engine(ctx)
        s = self.settings.stress
        return {
            "equity": eng.run_many(equity_ladder(s), vol_method, bond_method),  # type: ignore[arg-type]
            "curve": eng.run_many(parallel_curve_scenarios(s.rate_shock_ladder_bps) + curve_scenarios(), vol_method, bond_method),  # type: ignore[arg-type]
            "fx": eng.run_many(fx_ladder(s), vol_method, bond_method),  # type: ignore[arg-type]
            "vol": eng.run_many(vol_ladder(s), vol_method, bond_method),  # type: ignore[arg-type]
        }

    def _persist(self, ctx: RiskContext, results: list[StressResult]) -> None:
        if self.repo is None or ctx.portfolio.portfolio_id.startswith("UPLOAD"):
            return
        rows = [
            {
                "portfolio_id": ctx.portfolio.portfolio_id, "scenario_id": r.scenario.scenario_id, "run_date": ctx.valuation_date,
                "vol_method": r.vol_method, "bond_method": r.bond_method, "total_pnl": r.total_pnl, "pnl_pct": r.portfolio_return,
                "stressed_nav": r.stressed_nav, "worst_position": r.worst_position, "worst_asset_class": r.worst_asset_class,
            }
            for r in results
        ]
        try:
            self.repo.save_stress_results(rows)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist stress results")

    # -------------------------------------------------------------- tables
    @staticmethod
    def summary_table(results: list[StressResult]) -> pd.DataFrame:
        return results_frame(results)

    @staticmethod
    def curve_table(results: list[StressResult]) -> pd.DataFrame:
        """Scenario | Portfolio P&L | Fixed-Income P&L | Worst Bond | KRD Impact."""
        return pd.DataFrame(
            [
                {
                    "Scenario": r.scenario.name,
                    "Portfolio P&L": r.total_pnl,
                    "Fixed-Income P&L": r.fixed_income_pnl,
                    "Worst Bond": r.worst_bond,
                    "KRD Impact": r.krd_pnl,
                    "Convexity / Non-linear": r.fixed_income_pnl - r.krd_pnl,
                }
                for r in results
            ]
        )

    @staticmethod
    def worst(results: list[StressResult]) -> StressResult | None:
        return min(results, key=lambda r: r.total_pnl) if results else None
