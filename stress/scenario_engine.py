"""Generic scenario engine.

Input:  Portfolio + StressScenario + MarketState.
Output: position P&L, asset-class P&L, total P&L, portfolio return and
        stressed NAV (plus fixed-income detail for curve scenarios).

Stress tests are deterministic "what-if" revaluations, not probability
forecasts: they say what the book would lose if the specified moves happened
instantaneously, with no assumption about how likely they are.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics.revaluation import BondMethod, MarketState, PortfolioPricer, VolDynamics
from config.settings import load_yaml
from models.portfolio import Portfolio
from models.positions import AssetType
from models.risk import ShockBasis, StressScenario
from stress.shock_application import apply_scenario, equity_moves, portfolio_factors
from stress.volatility_shocks import describe_method
from stress.yield_curve_shocks import BP, shock_vector, twist

logger = logging.getLogger(__name__)

Scenario = StressScenario  # the spec's name for the reusable scenario object

BOND_METHOD_LABELS = {
    "FULL_REPRICING": "Full repricing on the shocked curve",
    "SENSITIVITY": "Sensitivity approximation (KRD-weighted duration + convexity)",
}


@dataclass
class StressResult:
    scenario: StressScenario
    position_pnl: pd.Series
    asset_classes: pd.Series
    nav: float
    vol_method: str
    bond_method: str
    krd_pnl: float = 0.0
    notes: list[str] = field(default_factory=list)
    equity_moves: dict[str, float] = field(default_factory=dict)  # applied equity return per ticker
    betas: dict[str, float] = field(default_factory=dict)  # betas used (empty unless beta-mapped)

    @property
    def beta_mapped(self) -> bool:
        return bool(self.betas) and self.scenario.equity_beta_mapping

    @property
    def total_pnl(self) -> float:
        return float(self.position_pnl.sum())

    @property
    def portfolio_return(self) -> float:
        return self.total_pnl / self.nav if self.nav else float("nan")

    @property
    def stressed_nav(self) -> float:
        return self.nav + self.total_pnl

    @property
    def asset_class_pnl(self) -> pd.Series:
        return self.position_pnl.groupby(self.asset_classes.reindex(self.position_pnl.index)).sum()

    @property
    def worst_position(self) -> str | None:
        return None if self.position_pnl.empty else str(self.position_pnl.idxmin())

    @property
    def worst_asset_class(self) -> str | None:
        acp = self.asset_class_pnl
        return None if acp.empty else str(acp.idxmin())

    @property
    def fixed_income_pnl(self) -> float:
        return float(self.asset_class_pnl.get(AssetType.BOND.value, 0.0))

    @property
    def worst_bond(self) -> str | None:
        bonds = self.position_pnl[self.asset_classes.reindex(self.position_pnl.index) == AssetType.BOND.value]
        return None if bonds.empty else str(bonds.idxmin())

    def assumptions(self) -> list[str]:
        out = [
            f"Volatility: {describe_method(self.vol_method)}",
            f"Bonds: {BOND_METHOD_LABELS.get(self.bond_method, self.bond_method)}",
            "Shocks are instantaneous; positions are static (no hedging or rebalancing).",
        ]
        return out + self.notes

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.scenario_id,
            "scenario": self.scenario.name,
            "category": self.scenario.category,
            "total_pnl": self.total_pnl,
            "pnl_pct": self.portfolio_return,
            "stressed_nav": self.stressed_nav,
            "worst_position": self.worst_position,
            "worst_asset_class": self.worst_asset_class,
            "fixed_income_pnl": self.fixed_income_pnl,
            "worst_bond": self.worst_bond,
            "krd_pnl": self.krd_pnl,
        }


class ScenarioEngine:
    """Runs stress scenarios against one portfolio and market state."""

    def __init__(
        self,
        portfolio: Portfolio,
        market_state: MarketState,
        pricer: PortfolioPricer | None = None,
        min_vol: float = 0.01,
        betas: dict[str, float] | None = None,
    ):
        self.betas = dict(betas or {})
        self.portfolio = portfolio
        self.state = market_state
        self.pricer = pricer or PortfolioPricer(portfolio, market_state, min_vol=min_vol)
        self.nav = float(self.pricer.value(market_state).sum())
        self.asset_classes = pd.Series({p.position_id: p.asset_type.value for p in portfolio.positions})
        self._krd = self.pricer.key_rate_durations(market_state)
        self._bonds = self.pricer.bond_analytics(market_state)

    def krd_impact(self, scenario: StressScenario) -> float:
        """First-order fixed-income P&L from key-rate durations: ``-sum_i MV_i sum_k KRD_ik dz_k``."""
        if self._krd.empty:
            return 0.0
        curve = shock_vector(scenario.rates_shock, scenario.curve_shocks, self.state.curve.labels)
        credit = shock_vector(scenario.credit_spread_shock, scenario.credit_curve_shocks, self.state.curve.labels)
        total = 0.0
        bonds = self._bonds.set_index("position_id")
        for pid, krd in self._krd.iterrows():
            dz = curve + (credit if self.pricer.is_credit(pid) else 0.0)
            mv = float(bonds.at[pid, "market_value"])
            ticker = self.portfolio.get(pid).ticker
            override = scenario.yield_overrides.get(ticker, 0.0)
            total += -mv * float(np.dot(krd.to_numpy(), dz)) - mv * float(bonds.at[pid, "effective_duration"]) * override
        return total

    def run(
        self,
        scenario: StressScenario,
        vol_method: VolDynamics = "STICKY_STRIKE",
        bond_method: BondMethod = "FULL_REPRICING",
    ) -> StressResult:
        pnl = apply_scenario(self.pricer, self.state, scenario, vol_method, bond_method, self.betas)
        tickers = portfolio_factors(self.pricer)["tickers"]
        moves = equity_moves(scenario, tickers, self.betas)
        mapped = scenario.equity_beta_mapping and bool(self.betas) and scenario.equity_shock != 0
        notes = []
        if mapped:
            used = [self.betas.get(t, 1.0) for t in tickers if t not in scenario.equity_overrides]
            if used:
                notes.append(
                    f"Equity shock is beta-mapped: each name moves beta x {scenario.equity_shock:+.0%} "
                    f"(betas {min(used):.2f} to {max(used):.2f})."
                )
        elif scenario.equity_beta_mapping and scenario.equity_shock != 0:
            notes.append("Beta mapping requested but no betas were available; the equity shock was applied uniformly.")
        if vol_method == "STICKY_DELTA":
            missing = [
                p.option.underlying
                for p in self.portfolio.positions
                if p.option is not None and p.option.underlying not in self.state.vol_surfaces
            ]
            if missing:
                notes.append(f"Volatility surface data unavailable for {', '.join(sorted(set(missing)))}; sticky strike used.")
        logger.info("Stress scenario completed: %s (P&L %.0f)", scenario.name, pnl.sum())
        return StressResult(
            scenario=scenario,
            position_pnl=pnl,
            asset_classes=self.asset_classes,
            nav=self.nav,
            vol_method=vol_method,
            bond_method=bond_method,
            krd_pnl=self.krd_impact(scenario),
            notes=notes,
            equity_moves=moves,
            betas={t: self.betas.get(t, 1.0) for t in tickers} if mapped else {},
        )

    def run_many(
        self, scenarios: list[StressScenario], vol_method: VolDynamics = "STICKY_STRIKE", bond_method: BondMethod = "FULL_REPRICING"
    ) -> list[StressResult]:
        return [self.run(s, vol_method, bond_method) for s in scenarios]


def run_stress_test(
    portfolio: Portfolio,
    scenario: StressScenario,
    market_state: MarketState,
    vol_method: VolDynamics = "STICKY_STRIKE",
    bond_method: BondMethod = "FULL_REPRICING",
    betas: dict[str, float] | None = None,
) -> StressResult:
    """Convenience wrapper: Portfolio + Scenario + Market Data -> StressResult."""
    return ScenarioEngine(portfolio, market_state, betas=betas).run(scenario, vol_method, bond_method)


def results_frame(results: list[StressResult]) -> pd.DataFrame:
    return pd.DataFrame([r.summary() for r in results])


# ------------------------------------------------------------------ (de)serialisation


def scenario_from_dict(data: dict[str, Any]) -> StressScenario:
    """Build a scenario from YAML/JSON, converting ``*_bps`` fields and curve twists."""
    d = dict(data)
    if "rates_shock_bps" in d:
        d["rates_shock"] = float(d.pop("rates_shock_bps")) * BP
    if "credit_spread_shock_bps" in d:
        d["credit_spread_shock"] = float(d.pop("credit_spread_shock_bps")) * BP
    curve: dict[str, float] = {}
    if "curve_twist" in d:
        tw = d.pop("curve_twist")
        curve.update(twist(float(tw["short_bps"]), float(tw["long_bps"])))
    if "curve_shocks_bps" in d:
        for k, v in d.pop("curve_shocks_bps").items():
            curve[str(k)] = curve.get(str(k), 0.0) + float(v) * BP
    if curve:
        d["curve_shocks"] = {**d.get("curve_shocks", {}), **curve}
    if "credit_curve_shocks_bps" in d:
        d["credit_curve_shocks"] = {str(k): float(v) * BP for k, v in d.pop("credit_curve_shocks_bps").items()}
    if "yield_overrides_bps" in d:
        d["yield_overrides"] = {k: float(v) * BP for k, v in d.pop("yield_overrides_bps").items()}
    d.pop("is_predefined", None)
    return StressScenario.model_validate(d)


def scenario_to_dict(s: StressScenario, predefined: bool = True) -> dict[str, Any]:
    out = s.model_dump(mode="json")
    out["is_predefined"] = predefined
    return out


def load_scenarios(path: Path) -> list[StressScenario]:
    """Load predefined scenarios from YAML."""
    raw = load_yaml(path)
    scenarios = [scenario_from_dict(item) for item in raw.get("scenarios", [])]
    logger.info("Loaded %d predefined stress scenarios", len(scenarios))
    return scenarios


def basis_table(scenario: StressScenario) -> pd.DataFrame:
    """Which inputs are historical observations vs hypothetical assumptions."""
    rows = []
    comps = {
        "equity": scenario.equity_shock != 0 or bool(scenario.equity_overrides),
        "rates": scenario.rates_shock != 0 or bool(scenario.curve_shocks) or bool(scenario.yield_overrides),
        "fx": scenario.fx_shock != 0 or bool(scenario.fx_shocks),
        "volatility": scenario.volatility_shock != 0 or bool(scenario.vol_overrides or scenario.vol_skew_shocks or scenario.vol_term_shocks),
        "credit": scenario.credit_spread_shock != 0 or bool(scenario.credit_curve_shocks),
    }
    for comp, active in comps.items():
        if active:
            basis = scenario.shock_basis.get(comp, ShockBasis.ASSUMED)
            rows.append({"input": "FX" if comp == "fx" else comp.title(), "basis": "Historical observation" if basis is ShockBasis.OBSERVED else "Hypothetical assumption"})
    if scenario.equity_beta_mapping and scenario.equity_shock != 0:
        rows.append({"input": "Equity beta mapping", "basis": "Hypothetical assumption"})
    return pd.DataFrame(rows)
