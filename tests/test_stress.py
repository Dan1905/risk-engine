import numpy as np
import pandas as pd
import pytest

from analytics.bonds import BondSchedule
from analytics.greeks import all_greeks
from analytics.revaluation import PortfolioPricer
from models.portfolio import Portfolio
from models.risk import StressScenario
from stress.hypothetical import custom_scenario, equity_shock_scenario, fx_shock_scenario, rate_shock_scenario
from stress.scenario_engine import ScenarioEngine, basis_table, run_stress_test, scenario_from_dict
from stress.shock_application import fx_moves
from stress.volatility_shocks import vol_shock_scenario
from stress.yield_curve_shocks import flattener, inversion, parallel_shift, shock_vector, steepener, twist
from tests.conftest import VAL_DATE, make_positions


def sub_portfolio(ids: list[str]) -> Portfolio:
    return Portfolio(portfolio_id="T", name="T", valuation_date=VAL_DATE, positions=[p for p in make_positions() if p.position_id in ids])


@pytest.fixture
def engine(portfolio, base_state) -> ScenarioEngine:
    return ScenarioEngine(portfolio, base_state)


def test_zero_scenario_zero_pnl(engine):
    res = engine.run(StressScenario(scenario_id="NONE", name="None"))
    assert res.total_pnl == pytest.approx(0.0, abs=1e-6)
    assert res.stressed_nav == pytest.approx(res.nav)


def test_equity_shock_linear(base_state):
    pf = sub_portfolio(["EQ1", "EQ2"])
    res = run_stress_test(pf, equity_shock_scenario(-0.20), base_state)
    assert res.position_pnl["EQ1"] == pytest.approx(1000 * 200 * -0.20)
    assert res.position_pnl["EQ2"] == pytest.approx(500 * 150 * 1.10 * -0.20)
    assert res.portfolio_return == pytest.approx(-0.20)
    assert res.worst_position == "EQ1"  # USD 200k loses more than the EUR 82.5k (USD) position
    assert res.asset_class_pnl["EQUITY"] == pytest.approx(res.total_pnl)


def test_equity_overrides(base_state):
    pf = sub_portfolio(["EQ1", "EQ2"])
    sc = custom_scenario("Tech", equity_shock=-0.10, equity_overrides={"AAPL": -0.25})
    res = run_stress_test(pf, sc, base_state)
    assert res.position_pnl["EQ1"] == pytest.approx(200_000 * -0.25)
    assert res.position_pnl["EQ2"] == pytest.approx(82_500 * -0.10)


def test_rate_shock_full_repricing(base_state):
    pf = sub_portfolio(["BD1"])
    pricer = PortfolioPricer(pf, base_state)
    p0, t, a = pricer.bond_price("BD1", base_state)
    spread = pricer.bond_spec("BD1").spread
    res = run_stress_test(pf, rate_shock_scenario(100), base_state)
    p1 = float(np.sum(a * (1 + 0.04 + 0.01 + spread) ** (-t)))
    assert res.total_pnl == pytest.approx(1_000_000 / 100 * (p1 - p0))
    assert res.total_pnl < 0


def test_sensitivity_approximation_close_to_full(base_state):
    pf = sub_portfolio(["BD1", "BD2"])
    eng = ScenarioEngine(pf, base_state)
    for bps in (25, 100):
        full = eng.run(rate_shock_scenario(bps), bond_method="FULL_REPRICING").total_pnl
        approx = eng.run(rate_shock_scenario(bps), bond_method="SENSITIVITY").total_pnl
        assert approx == pytest.approx(full, rel=0.01)
    # KRD (first-order) impact close for a small shock, overstates the loss for a large one (convexity)
    small = eng.run(rate_shock_scenario(10))
    assert small.krd_pnl == pytest.approx(small.fixed_income_pnl, rel=0.01)
    big = eng.run(rate_shock_scenario(200))
    assert big.krd_pnl < big.fixed_income_pnl


def test_credit_spread_shock_hits_only_credit_bonds(base_state):
    pf = sub_portfolio(["BD1", "BD2"])
    res = run_stress_test(pf, StressScenario(scenario_id="CR", name="Credit", credit_spread_shock=0.02), base_state)
    assert res.position_pnl["BD1"] == pytest.approx(0.0, abs=1e-6)
    assert res.position_pnl["BD2"] < 0


def test_fx_shock_conventions(base_state):
    pf = sub_portfolio(["FX1", "FX2", "EQ2"])
    eur_up = run_stress_test(pf, fx_shock_scenario("EUR", 0.10), base_state)
    assert eur_up.position_pnl["FX1"] == pytest.approx(200_000 * 1.10 * 0.10)
    assert eur_up.position_pnl["FX2"] == pytest.approx(0.0)
    assert eur_up.position_pnl["EQ2"] == pytest.approx(500 * 150 * 1.10 * 0.10)
    usd_up = run_stress_test(pf, fx_shock_scenario("USD", 0.10), base_state)
    assert usd_up.position_pnl["FX1"] == pytest.approx(200_000 * 1.10 * (1 / 1.1 - 1))
    inr_down = run_stress_test(pf, fx_shock_scenario("INR", -0.10), base_state)
    assert inr_down.position_pnl["FX2"] == pytest.approx(10_000_000 / 88.0 * -0.10)
    moves = fx_moves(StressScenario(scenario_id="X", name="X", fx_shock=0.05, fx_shocks={"EUR": 0.05}), ["EUR", "GBP"])
    assert moves["EUR"] == pytest.approx(0.0)
    assert moves["GBP"] == pytest.approx(1 / 1.05 - 1)


def test_curve_shock_shapes():
    st = steepener()
    assert st["3M"] == pytest.approx(0.0050) and st["2Y"] == pytest.approx(0.0050)
    assert st["10Y"] == pytest.approx(0.0150) and st["30Y"] == pytest.approx(0.0150)
    assert st["5Y"] == pytest.approx(0.0050 + 0.0100 * 3 / 8)
    assert flattener()["2Y"] > flattener()["10Y"]
    inv = inversion()
    assert inv["2Y"] == pytest.approx(0.0150) and inv["30Y"] == pytest.approx(-0.0050)
    assert all(v == pytest.approx(0.02) for v in parallel_shift(200).values())
    vec = shock_vector(0.001, twist(0, 100))
    assert vec[0] == pytest.approx(0.001) and vec[-1] == pytest.approx(0.011)
    with pytest.raises(ValueError):
        shock_vector(0.0, {"4Y": 0.01})


def test_curve_scenario_differs_from_parallel(base_state):
    pf = sub_portfolio(["BD1"])
    eng = ScenarioEngine(pf, base_state)
    steep = eng.run(StressScenario(scenario_id="S", name="S", curve_shocks=steepener()))
    flat = eng.run(StressScenario(scenario_id="F", name="F", curve_shocks=flattener()))
    # A 10Y bond is dominated by long-end nodes: steepener (+150 long) hurts more than flattener (+50 long)
    assert steep.total_pnl < flat.total_pnl < 0


def test_vol_shock_matches_vega(base_state):
    pf = sub_portfolio(["OP1"])
    pricer = PortfolioPricer(pf, base_state)
    opt = pricer.option_analytics().iloc[0]
    res = run_stress_test(pf, vol_shock_scenario(0.01), base_state)
    assert res.total_pnl == pytest.approx(opt["vega"], rel=0.02)  # position vega is per 1 vol point
    down = run_stress_test(pf, vol_shock_scenario(-0.05), base_state)
    assert down.total_pnl < 0


def test_sticky_strike_vs_sticky_delta(base_state):
    pf = sub_portfolio(["OP1"])
    eng = ScenarioEngine(pf, base_state)
    sc = equity_shock_scenario(-0.20)
    strike = eng.run(sc, vol_method="STICKY_STRIKE")
    delta = eng.run(sc, vol_method="STICKY_DELTA")
    surf = base_state.vol_surfaces["AAPL"]
    t = (pd.Timestamp("2027-03-19") - pd.Timestamp(VAL_DATE)).days / 365
    smile_move = float(surf.vol(210 / 160, t) - surf.vol(210 / 200, t))
    assert np.sign(delta.total_pnl - strike.total_pnl) == np.sign(smile_move)
    assert "Sticky delta" in delta.assumptions()[0]


def test_combined_scenario(engine):
    gro = scenario_from_dict(
        {
            "scenario_id": "GRO", "name": "Global Risk-Off", "equity_shock": -0.20, "rates_shock_bps": 150,
            "fx_shock": 0.10, "volatility_shock": 0.15, "credit_spread_shock_bps": 200,
        }
    )
    res = engine.run(gro)
    acp = res.asset_class_pnl
    assert acp["EQUITY"] < 0 and acp["BOND"] < 0 and acp["FX"] < 0
    assert res.total_pnl == pytest.approx(acp.sum())
    assert res.worst_asset_class in acp.index
    assert res.stressed_nav == pytest.approx(res.nav + res.total_pnl)


def test_scenario_from_dict_units():
    s = scenario_from_dict(
        {
            "scenario_id": "X", "name": "X", "rates_shock_bps": 25, "credit_spread_shock_bps": 100,
            "curve_twist": {"short_bps": 50, "long_bps": 150}, "shock_basis": {"equity": "HISTORICAL_OBSERVATION"},
            "equity_shock": -0.1,
        }
    )
    assert s.rates_shock == pytest.approx(0.0025)
    assert s.credit_spread_shock == pytest.approx(0.01)
    assert s.curve_shocks["30Y"] == pytest.approx(0.015)
    table = basis_table(s)
    assert set(table["input"]) == {"Equity", "Rates", "Credit"}
    assert table.set_index("input").at["Equity", "basis"] == "Historical observation"


def test_bond_schedule_coupons_for_stress_consistency():
    sched = BondSchedule.build(pd.Timestamp("2036-08-15").date(), 0.04, 2)
    assert sched.amounts[-1] == pytest.approx(102.0)


def test_volatility_surface_variants():
    from analytics.volatility_surface import FlatVolSurface, GridVolSurface, SmileVolSurface

    flat = FlatVolSurface(0.2)
    assert float(flat.vol(1.2, 0.5)) == pytest.approx(0.2)
    assert float(flat.shifted(0.05).vol(1.0, 1.0)) == pytest.approx(0.25)
    # Smile through ATM 20%, 25d call 23%, 25d put 25% (the spec example)
    smile = SmileVolSurface.from_delta_quotes(atm=0.20, call_25d=0.23, put_25d=0.25, maturity=0.25)
    assert smile.curvature > 0 and smile.skew < 0  # both wings up, put wing higher
    assert float(smile.vol(0.85, 0.25)) > float(smile.vol(1.0, 0.25))
    grid = GridVolSurface(np.array([0.25, 1.0]), np.array([0.9, 1.0, 1.1]), np.array([[0.30, 0.25, 0.24], [0.26, 0.22, 0.21]]))
    assert float(grid.vol(1.0, 0.625)) == pytest.approx(0.235)  # linear in maturity
    assert float(grid.vol(1.5, 0.25)) == pytest.approx(0.24)  # flat extrapolation in moneyness
    buckets = GridVolSurface(grid.maturities, grid.moneyness_grid, grid.vols, maturity_buckets=True)
    assert float(buckets.vol(1.0, 0.5)) == pytest.approx(0.22)  # next bucket up (1Y)
    with pytest.raises(ValueError):
        GridVolSurface(np.array([0.25]), np.array([1.0]), np.array([[np.nan]]))


# ------------------------------------------------------------- beta mapping & replay
def test_market_betas():
    from analytics.beta import market_betas

    rng = np.random.default_rng(60)
    m = rng.normal(0, 0.01, 2000)
    r = pd.DataFrame({"HIGH": 2.0 * m + rng.normal(0, 0.002, 2000), "LOW": 0.5 * m + rng.normal(0, 0.002, 2000),
                      "MID": 1.0 * m + rng.normal(0, 0.002, 2000)})
    b = market_betas(r)
    assert b.mean() == pytest.approx(1.0)  # equal-weighted betas average to exactly 1
    assert b["HIGH"] > b["MID"] > b["LOW"]
    assert b["HIGH"] / b["LOW"] == pytest.approx(4.0, rel=0.1)
    assert (market_betas(pd.DataFrame({"A": [0.0, 0.0, 0.0]})) == 1.0).all()  # no market variance -> uniform
    weighted = market_betas(r, pd.Series({"HIGH": 1.0, "LOW": 0.0, "MID": 0.0}))
    assert weighted["HIGH"] == pytest.approx(1.0)


def test_beta_mapped_equity_shock(base_state):
    pf = sub_portfolio(["EQ1", "EQ2", "OP1"])
    betas = {"AAPL": 1.5, "SAP": 0.5}
    sc = StressScenario(scenario_id="H", name="H", category="HISTORICAL", equity_shock=-0.20, equity_beta_mapping=True)
    mapped = ScenarioEngine(pf, base_state, betas=betas).run(sc)
    uniform = ScenarioEngine(pf, base_state, betas=betas).run(sc.model_copy(update={"equity_beta_mapping": False}))
    assert mapped.position_pnl["EQ1"] == pytest.approx(200_000 * 1.5 * -0.20)
    assert mapped.position_pnl["EQ2"] == pytest.approx(82_500 * 0.5 * -0.20)
    assert uniform.position_pnl["EQ1"] == pytest.approx(200_000 * -0.20)
    assert mapped.beta_mapped and not uniform.beta_mapped
    assert mapped.equity_moves == {"AAPL": pytest.approx(-0.30), "SAP": pytest.approx(-0.10)}
    assert any("beta-mapped" in n for n in mapped.assumptions())
    assert "Equity beta mapping" in set(basis_table(sc)["input"])
    # the option underlying is beta-mapped too: a larger fall hurts the long call more
    assert mapped.position_pnl["OP1"] < uniform.position_pnl["OP1"] < 0


def test_beta_mapping_overrides_floor_and_fallback(base_state):
    pf = sub_portfolio(["EQ1", "EQ2"])
    sc = StressScenario(scenario_id="H", name="H", equity_shock=-0.50, equity_beta_mapping=True, equity_overrides={"SAP": -0.05})
    res = ScenarioEngine(pf, base_state, betas={"AAPL": 3.0}).run(sc)
    assert res.equity_moves["AAPL"] == pytest.approx(-0.95)  # 3 x -50% floored at -95%
    assert res.equity_moves["SAP"] == pytest.approx(-0.05)  # explicit override wins
    no_betas = ScenarioEngine(pf, base_state).run(sc)  # no betas available -> uniform, with a note
    assert no_betas.equity_moves["AAPL"] == pytest.approx(-0.50)
    assert any("no betas were available" in n for n in no_betas.assumptions())


def test_credit_curve_node_shocks(base_state):
    pf = sub_portfolio(["BD1", "BD2"])
    eng = ScenarioEngine(pf, base_state)
    nodes = eng.run(scenario_from_dict({"scenario_id": "C", "name": "C", "credit_curve_shocks_bps": {"5Y": 100}}))
    parallel = eng.run(StressScenario(scenario_id="P", name="P", credit_spread_shock=0.01))
    assert nodes.position_pnl["BD1"] == pytest.approx(0.0, abs=1e-6)  # government bond unaffected
    assert parallel.position_pnl["BD2"] < nodes.position_pnl["BD2"] < 0  # one node moves less than the whole curve
    assert nodes.krd_pnl == pytest.approx(nodes.fixed_income_pnl, rel=0.05)


def test_worst_window():
    from stress.historical import worst_window

    idx = pd.bdate_range("2026-01-01", periods=10)
    pnl = pd.Series([1.0, -1.0, 2.0, -5.0, -6.0, -4.0, 3.0, 1.0, -2.0, 1.0], index=idx)
    start, end, total = worst_window(pnl, 3)
    assert (start, end, total) == (idx[2], idx[5], -15.0)  # days 3-5; start = close before the window
    with pytest.raises(ValueError):
        worst_window(pnl, 10)
    with pytest.raises(ValueError):
        worst_window(pnl, 0)
