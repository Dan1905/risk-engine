"""Translate a ``StressScenario`` into factor moves and apply them.

A stress scenario becomes a one-row ``ScenarioSet`` covering every factor the
portfolio is exposed to, which is then revalued by the same
``PortfolioPricer`` used for VaR.

FX convention: foreign currency c moves by ``(1 + s_c) / (1 + s_USD) - 1`` in
USD terms, where ``s_c`` is the currency-specific shock and ``s_USD`` the USD
shock (``fx_shock`` plus any ``fx_shocks['USD']``). "USD +10%" therefore makes
every foreign asset worth 1/1.1 of its USD value.
"""

from __future__ import annotations

import pandas as pd

from analytics import risk_factors as rf
from analytics.revaluation import BondMethod, MarketState, PortfolioPricer, ScenarioSet, VolDynamics
from models.positions import AssetType
from models.risk import StressScenario
from stress.yield_curve_shocks import shock_vector


def portfolio_factors(pricer: PortfolioPricer) -> dict[str, list[str]]:
    """Equity tickers, currencies, option underlyings and bond tickers in the book."""
    tickers, ccys, unds, bonds = set(), set(), set(), set()
    for p in pricer.positions:
        ccys.add(p.exposure_currency)
        ccys.add(p.currency)
        if p.asset_type == AssetType.EQUITY:
            tickers.add(p.ticker)
        elif p.asset_type == AssetType.OPTION and p.option is not None:
            tickers.add(p.option.underlying)
            unds.add(p.option.underlying)
        elif p.asset_type == AssetType.BOND:
            bonds.add(p.ticker)
    ccys.discard("USD")
    return {"tickers": sorted(tickers), "currencies": sorted(ccys), "underlyings": sorted(unds), "bonds": sorted(bonds)}


def fx_moves(scenario: StressScenario, currencies: list[str]) -> dict[str, float]:
    """USD-per-unit return for each foreign currency under the scenario."""
    usd = scenario.fx_shock + scenario.fx_shocks.get("USD", 0.0)
    return {c: (1.0 + scenario.fx_shocks.get(c, 0.0)) / (1.0 + usd) - 1.0 for c in currencies}


def equity_moves(scenario: StressScenario, tickers: list[str], betas: dict[str, float] | None = None) -> dict[str, float]:
    """Equity return per ticker under the scenario.

    Priority: explicit override > beta-mapped shock (``beta_i x equity_shock``
    when the scenario asks for it and betas are available) > uniform shock.
    A ticker without a beta estimate uses beta 1. Returns are floored at -95%.
    """
    out: dict[str, float] = {}
    for t in tickers:
        if t in scenario.equity_overrides:
            out[t] = scenario.equity_overrides[t]
        elif scenario.equity_beta_mapping and betas:
            out[t] = max(betas.get(t, 1.0) * scenario.equity_shock, rf.RETURN_FLOOR)
        else:
            out[t] = scenario.equity_shock
    return out


def build_scenario_set(
    scenario: StressScenario, pricer: PortfolioPricer, state: MarketState, betas: dict[str, float] | None = None
) -> ScenarioSet:
    """Build a single-scenario ``ScenarioSet`` for the portfolio's factors."""
    f = portfolio_factors(pricer)
    labels = state.curve.labels
    moves: dict[str, float] = {}
    for t, r in equity_moves(scenario, f["tickers"], betas).items():
        moves[rf.factor(rf.EQ, t)] = r
    for c, r in fx_moves(scenario, f["currencies"]).items():
        moves[rf.factor(rf.FX, c)] = r
    for u in f["underlyings"]:
        moves[rf.factor(rf.VOL, u)] = scenario.vol_overrides.get(u, scenario.volatility_shock)
        if u in scenario.vol_skew_shocks:
            moves[rf.factor(rf.VOLSKEW, u)] = scenario.vol_skew_shocks[u]
        if u in scenario.vol_term_shocks:
            moves[rf.factor(rf.VOLTERM, u)] = scenario.vol_term_shocks[u]
    for lbl, shift in zip(labels, shock_vector(scenario.rates_shock, scenario.curve_shocks, labels)):
        moves[rf.factor(rf.CURVE, rf.GOVT_CURVE, lbl)] = float(shift)
    for cid in state.spread_curves:
        for lbl, shift in zip(labels, shock_vector(scenario.credit_spread_shock, scenario.credit_curve_shocks, labels)):
            moves[rf.factor(rf.CURVE, cid, lbl)] = float(shift)
    for b in f["bonds"]:
        if b in scenario.yield_overrides:
            moves[rf.factor(rf.YIELD, b)] = scenario.yield_overrides[b]
    frame = pd.DataFrame([moves], index=pd.Index([scenario.scenario_id]))
    return ScenarioSet.from_frame(frame)


def apply_scenario(
    pricer: PortfolioPricer,
    state: MarketState,
    scenario: StressScenario,
    vol_method: VolDynamics = "STICKY_STRIKE",
    bond_method: BondMethod = "FULL_REPRICING",
    betas: dict[str, float] | None = None,
) -> pd.Series:
    """Position P&L (USD) under the scenario."""
    pnl = pricer.scenario_pnl(state, build_scenario_set(scenario, pricer, state, betas), vol_method, bond_method)
    return pnl.iloc[0].rename("pnl")
