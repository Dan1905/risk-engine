from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from analytics.pnl_attribution import COMPONENTS, attribute_pnl, attribution_series, summarise_components
from analytics.revaluation import PortfolioPricer
from analytics.volatility_surface import SmileVolSurface
from analytics.yield_curve import YieldCurve

NEXT_DAY = date(2026, 10, 1)


def moved_state(base_state, *, eq=0.01, fx=0.004, rates=0.0002, spread=0.0001, vol=0.005, d=NEXT_DAY):
    curve = YieldCurve(base_state.curve.tenors, base_state.curve.rates + rates, base_state.curve.labels)
    spreads = {k: YieldCurve(v.tenors, v.rates + spread, v.labels) for k, v in base_state.spread_curves.items()}
    return replace(
        base_state,
        date=d,
        equity_prices={k: v * (1 + eq) for k, v in base_state.equity_prices.items()},
        fx_rates={k: (v if k == "USD" else v * (1 + fx)) for k, v in base_state.fx_rates.items()},
        curve=curve,
        spread_curves=spreads,
        atm_vols={k: v + vol for k, v in base_state.atm_vols.items()},
        vol_surfaces={"AAPL": SmileVolSurface(atm=0.25 + vol, skew=-0.15, curvature=0.8)},
    )


def test_explained_plus_residual_equals_actual(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state))
    assert res.reconciles(tolerance=1e-6)
    by_pos = res.by_position
    np.testing.assert_allclose(by_pos[list(COMPONENTS)].sum(axis=1) + by_pos["residual"], by_pos["actual"], atol=1e-8)


def test_nav_waterfall_closes(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state))
    assert res.nav_start + res.actual == pytest.approx(res.nav_end)


def test_linear_components_are_exact(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state, eq=0.01, fx=0.004))
    bp = res.by_position
    assert bp.at["EQ1", "equity"] == pytest.approx(1000 * 200.0 * 0.01)
    assert bp.at["FX1", "fx"] == pytest.approx(200_000 * 1.10 * 0.004)
    assert bp.at["EQ1", "residual"] == pytest.approx(0.0, abs=1e-8)
    assert bp.at["FX1", "residual"] == pytest.approx(0.0, abs=1e-8)
    # Foreign equity: residual is only the FX x price cross term.
    cross = 500 * 150.0 * 0.01 * 1.10 * 0.004
    assert bp.at["EQ2", "residual"] == pytest.approx(cross, rel=1e-9)


def test_residual_small_for_small_moves(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state))
    gross = res.by_position[list(COMPONENTS)].abs().to_numpy().sum()
    assert abs(res.residual) < 0.02 * gross
    assert res.explained_pct > 0.95


def test_option_greeks_explain_option_pnl(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state, eq=0.005, vol=0.002))
    op = res.by_position.loc["OP1"]
    assert abs(op["residual"]) < 0.05 * abs(op["actual"])
    assert op["option_delta"] > 0 and op["option_vega"] > 0 and op["option_theta"] < 0


def test_time_decay_only(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state, eq=0, fx=0, rates=0, spread=0, vol=0))
    totals = res.totals
    assert totals["equity"] == 0 and totals["fx"] == 0
    assert totals["carry"] > 0 and totals["option_theta"] < 0
    assert abs(res.residual) < 0.1 * (abs(totals["carry"]) + abs(totals["option_theta"]))


def test_summary_tables(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    res = attribute_pnl(pricer, base_state, moved_state(base_state))
    table = summarise_components(res)
    assert table.iloc[-1]["pnl"] == pytest.approx(res.actual)
    series = attribution_series([res])
    assert series["actual"].iloc[0] == pytest.approx(res.actual)


def test_rejects_reversed_dates(portfolio, base_state):
    pricer = PortfolioPricer(portfolio, base_state)
    with pytest.raises(ValueError):
        attribute_pnl(pricer, moved_state(base_state), base_state)
