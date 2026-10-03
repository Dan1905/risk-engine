"""Integration tests over a small generated dataset: risk aggregation,
look-ahead prevention in the backtest, and idempotent database seeding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.var import historical_var, parametric_var
from config.settings import get_settings
from database.connection import create_db_engine
from database.repository import RiskRepository
from models.portfolio import build_portfolio
from models.risk import RiskConfiguration
from scripts.generate_sample_data import generate_sample_data
from scripts.seed_database import is_seeded, seed_database
from services.backtest_service import BacktestService
from services.market_data_service import CsvMarketDataProvider, MarketHistory
from services.portfolio_service import PortfolioService, parse_portfolio_csv
from services.risk_service import RiskService, build_context

N_DAYS = 320
END = "2026-09-30"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("sample")
    generate_sample_data(out, seed=7, n_days=N_DAYS, end_date=END)
    return out


@pytest.fixture(scope="module")
def settings(data_dir, tmp_path_factory):
    base = get_settings()
    db = tmp_path_factory.mktemp("db") / "test.db"
    md = base.market_data.model_copy(update={"n_days": N_DAYS, "end_date": END, "seed": 7})
    return base.model_copy(update={"database_url": f"sqlite:///{db}", "sample_data_dir": data_dir, "market_data": md})


@pytest.fixture(scope="module")
def ctx(data_dir, settings):
    history = MarketHistory.build(CsvMarketDataProvider(data_dir))
    records = parse_portfolio_csv(data_dir / "sample_portfolio.csv")
    portfolio, _ = build_portfolio(records, portfolio_id="T", name="T", valuation_date=history.last_date)
    return build_context(portfolio, history, settings)


def test_historical_var_equals_analytics(ctx, settings):
    svc = RiskService(settings)
    rep = svc.var_report(ctx, RiskConfiguration(lookback=100, confidence=0.99))
    pnl = RiskService.scenario_pnl(ctx, 100).sum(axis=1)
    assert rep.get("HISTORICAL").var == pytest.approx(historical_var(pnl, 0.99))
    assert rep.get("HISTORICAL").n_observations == 100


def test_holding_period_scaling(ctx, settings):
    svc = RiskService(settings)
    one = svc.var_report(ctx, RiskConfiguration(lookback=100, holding_period=1))
    ten = svc.var_report(ctx, RiskConfiguration(lookback=100, holding_period=10, horizon_method="SQRT_TIME"))
    for m in ("HISTORICAL", "PARAMETRIC", "FILTERED_HISTORICAL"):
        assert ten.get(m).var == pytest.approx(one.get(m).var * np.sqrt(10))
    assert ten.scale_days == 10 and ten.scenario_days == 1 and not ten.overlapping


def test_overlapping_multi_day_scenarios(ctx, settings):
    from analytics.risk_factors import aggregate_horizon

    svc = RiskService(settings)
    h, lookback = 5, 100
    rep = svc.var_report(ctx, RiskConfiguration(lookback=lookback, holding_period=h, horizon_method="OVERLAPPING"))
    assert rep.overlapping and rep.scale_days == 1 and rep.scenario_days == h
    assert len(rep.portfolio_pnl) == lookback  # lookback scenarios, each a genuine h-day move
    # Scenarios are the compounded moves of lookback + h - 1 daily observations ending on the valuation date.
    daily = ctx.history.historical_moves(ctx.valuation_date, lookback, extra_days=h - 1)
    assert len(daily) == lookback + h - 1 and daily.index[-1] == pd.Timestamp(ctx.valuation_date)
    expected = ctx.pricer.scenario_pnl(ctx.state, ScenarioSet.from_frame(aggregate_horizon(daily, h))).sum(axis=1)
    np.testing.assert_allclose(rep.portfolio_pnl.to_numpy(), expected.to_numpy())
    assert rep.get("HISTORICAL").var == pytest.approx(historical_var(expected, 0.99))  # no sqrt-of-time scaling applied
    # Exponentially weighted estimators stay on daily data and are scaled by sqrt(h).
    one_day = svc.var_report(ctx, RiskConfiguration(lookback=lookback, holding_period=1))
    assert rep.sqrt_scaled == ("EW_HISTORICAL", "EW_PARAMETRIC")
    for m in rep.sqrt_scaled:
        assert rep.get(m).var == pytest.approx(one_day.get(m).var * np.sqrt(h))
    assert one_day.sqrt_scaled == ()
    # Euler components still add up to the (h-day) parametric VaR.
    contrib = svc.contributions(ctx, rep, "EWMA")
    assert contrib["component_var"].sum() == pytest.approx(rep.get("EW_PARAMETRIC").var)
    # With h = 1 the two horizon methods are identical.
    a = svc.var_report(ctx, RiskConfiguration(lookback=lookback, holding_period=1, horizon_method="OVERLAPPING"))
    b = svc.var_report(ctx, RiskConfiguration(lookback=lookback, holding_period=1, horizon_method="SQRT_TIME"))
    for m in METHODS:
        assert a.get(m).var == pytest.approx(b.get(m).var)
    # Annualised volatility is comparable across horizons (same order of magnitude, not sqrt(h) apart).
    vol_1d = svc.portfolio_volatility_annual(a, "SAMPLE")
    vol_hd = svc.portfolio_volatility_annual(rep, "SAMPLE")
    assert 0.5 < vol_hd / vol_1d < 2.0


def test_multi_day_requires_enough_history(ctx, settings):
    from analytics.var import InsufficientDataError

    n = len(ctx.history.dates)
    with pytest.raises(InsufficientDataError):
        RiskService(settings).var_report(ctx, RiskConfiguration(lookback=n - 2, holding_period=10, horizon_method="OVERLAPPING"))


def test_risk_aggregation(ctx, settings):
    svc = RiskService(settings)
    rep = svc.var_report(ctx, RiskConfiguration(lookback=100))
    for cov in ("SAMPLE", "EWMA"):
        contrib = svc.contributions(ctx, rep, cov)
        total = parametric_var(rep.weights, rep.cov[cov], 0.99, rep.nav)
        assert contrib["component_var"].sum() == pytest.approx(total)
        assert contrib["pct_contribution"].sum() == pytest.approx(1.0)
        by_class = svc.contributions_by(contrib, "asset_class")
        assert by_class["component_var"].sum() == pytest.approx(total)
        by_ccy = svc.contributions_by(contrib, "currency")
        assert by_ccy["pct_contribution"].sum() == pytest.approx(1.0)
    # weighted position returns reproduce portfolio P&L
    port = rep.pnl_matrix.sum(axis=1)
    simple = RiskService.position_returns(rep.pnl_matrix, ctx.market_values, "simple")
    np.testing.assert_allclose((simple * rep.weights).sum(axis=1) * rep.nav, port, rtol=1e-9)


def test_exposures_consistent(ctx):
    exp = PortfolioService.exposures(ctx.valuation)
    assert exp["nav"] == pytest.approx(ctx.nav)
    assert exp["gross"] >= abs(exp["net"])
    parts = exp["equity"] + exp["fixed_income"] + exp["fx"] + exp["options"] + exp["cash"]
    assert parts == pytest.approx(exp["nav"])


def test_scenarios_never_exceed_end_date(ctx):
    d = ctx.history.dates[200].date()
    scen = ctx.history.historical_scenarios(d, 60)
    assert scen.labels.max() == pd.Timestamp(d)
    assert len(scen.labels) == 60


def test_backtest_has_no_look_ahead(ctx, data_dir, settings):
    lookback = 60
    fw = BacktestService(settings).forecast_windows(ctx, lookback)
    for k in (0, len(fw.dates) // 2, len(fw.dates) - 1):
        t = fw.dates[k]
        i = ctx.history.dates.get_loc(t)
        prev = ctx.history.dates[i - 1].date()
        # Rebuild a history that physically cannot contain anything after t-1.
        truncated = MarketHistory.build(CsvMarketDataProvider(data_dir), end=prev)
        assert truncated.dates.max() == pd.Timestamp(prev)
        scen = truncated.historical_scenarios(prev, lookback)
        expected = ctx.pricer.scenario_pnl(truncated.state_at(prev), scen).sum(axis=1).to_numpy()
        np.testing.assert_allclose(fw.windows[k], expected, rtol=1e-10)


def test_backtest_methods_and_kupiec(ctx, settings):
    svc = BacktestService(settings)
    fw = svc.forecast_windows(ctx, 60)
    for m in ("HISTORICAL", "EW_HISTORICAL", "PARAMETRIC", "EW_PARAMETRIC"):
        res = svc.run(ctx, fw, m, 0.99)
        k = res.kupiec
        assert k.n_observations == len(fw.dates)
        assert k.n_exceptions == int(res.frame["exception"].sum())
        assert 0.0 <= k.p_value <= 1.0


def test_seeding_is_idempotent(settings):
    repo = RiskRepository(create_db_engine(settings.database_url))
    seed_database(repo, settings)
    first = repo.table_counts()
    assert is_seeded(repo)
    seed_database(repo, settings)
    assert repo.table_counts() == first
    assert first["positions"] == 20 and first["stress_scenarios"] >= 8
    loaded = PortfolioService(repo, settings).load_portfolio(settings.portfolio.portfolio_id)
    assert len(loaded.positions) == 20


# ------------------------------------------------------- filtered HS / Monte Carlo services
from analytics.filtered_simulation import filter_moves  # noqa: E402
from analytics.revaluation import ScenarioSet  # noqa: E402
from services.risk_service import METHODS  # noqa: E402


def test_var_report_covers_all_methods(ctx, settings):
    rep = RiskService(settings).var_report(ctx, RiskConfiguration(lookback=100))
    assert set(rep.results) == set(METHODS)
    for m in METHODS:
        r = rep.get(m, 0.99)
        assert r.var > 0 and r.expected_shortfall >= r.var - 1e-9
    assert rep.get("MONTE_CARLO").n_observations == settings.simulation.monte_carlo_paths
    assert set(rep.distributions) == {"HISTORICAL", "FILTERED_HISTORICAL", "MONTE_CARLO"}
    # deterministic: same inputs, same Monte Carlo number
    again = RiskService(settings).var_report(ctx, RiskConfiguration(lookback=100))
    assert again.get("MONTE_CARLO").var == rep.get("MONTE_CARLO").var


def test_filtered_equals_historical_when_scaling_is_disabled(ctx, settings):
    sim = settings.simulation.model_copy(update={"fhs_ratio_floor": 1.0, "fhs_ratio_cap": 1.0})
    rep = RiskService(settings.model_copy(update={"simulation": sim})).var_report(ctx, RiskConfiguration(lookback=100))
    assert rep.get("FILTERED_HISTORICAL").var == pytest.approx(rep.get("HISTORICAL").var)
    assert rep.get("FILTERED_HISTORICAL").expected_shortfall == pytest.approx(rep.get("HISTORICAL").expected_shortfall)


def test_simulation_backtests_have_no_look_ahead(ctx, data_dir, settings):
    lookback = 60
    svc = BacktestService(settings)
    fw = svc.forecast_windows(ctx, lookback)
    sim = settings.simulation
    filtered = svc.simulation_windows(ctx, fw, "FILTERED_HISTORICAL", 0.94)
    for k in (0, len(fw.dates) // 2, len(fw.dates) - 1):
        t = fw.dates[k]
        prev = ctx.history.dates[ctx.history.dates.get_loc(t) - 1].date()
        truncated = MarketHistory.build(CsvMarketDataProvider(data_dir), end=prev)  # cannot contain t or later
        i = truncated.index_of(prev)
        moves = filter_moves(truncated.daily_moves(i, i), lookback, 0.94, (sim.fhs_ratio_floor, sim.fhs_ratio_cap), sim.fhs_seed_observations)
        expected = ctx.pricer.scenario_pnl(truncated.state_at(prev), ScenarioSet.from_frame(moves)).sum(axis=1).to_numpy()
        np.testing.assert_allclose(filtered[k], expected, rtol=1e-9, atol=1e-6)
    mc = svc.simulation_windows(ctx, fw, "MONTE_CARLO", 0.94, "EWMA")
    assert mc.shape == (len(fw.dates), sim.backtest_paths)
    assert svc.simulation_windows(ctx, fw, "MONTE_CARLO", 0.94, "EWMA") is mc  # memoised
    for m in ("FILTERED_HISTORICAL", "MONTE_CARLO"):
        res = svc.run(ctx, fw, m, 0.99)
        assert res.kupiec.n_observations == len(fw.dates) and res.christoffersen is not None
        assert svc.basel_assessment(fw, m, ctx=ctx).zone in {"GREEN", "YELLOW", "RED"}
    with pytest.raises(ValueError):
        svc.var_series(svc.forecast_windows(ctx, lookback), "MONTE_CARLO", 0.99)  # fresh windows, no context


# ------------------------------------------------------------------ beta mapping & replay
from services.stress_service import StressService  # noqa: E402
from stress.historical import replay_window_scenario, snap_to_history, worst_window  # noqa: E402


def test_equity_betas_from_history(ctx, settings):
    betas = ctx.history.equity_betas(ctx.valuation_date, 250)
    assert set(betas) == set(ctx.history.equity_prices.columns)
    assert np.mean(list(betas.values())) == pytest.approx(1.0)
    assert ctx.history.equity_betas(ctx.history.dates[5].date(), 250) == {}  # too little history


def test_library_beta_mapping_toggle(ctx, settings):
    svc = StressService(settings)
    mapped = {r.scenario.scenario_id: r for r in svc.run_predefined(ctx, "STICKY_STRIKE", "FULL_REPRICING", True)}
    uniform = {r.scenario.scenario_id: r for r in svc.run_predefined(ctx, "STICKY_STRIKE", "FULL_REPRICING", False)}
    assert mapped["GFC_2008"].beta_mapped and not uniform["GFC_2008"].beta_mapped
    assert mapped["GFC_2008"].total_pnl != pytest.approx(uniform["GFC_2008"].total_pnl)
    # hypothetical scenarios are never beta-mapped by the toggle
    assert mapped["EQUITY_CRASH"].total_pnl == pytest.approx(uniform["EQUITY_CRASH"].total_pnl)
    default = {r.scenario.scenario_id: r for r in svc.run_predefined(ctx, "STICKY_STRIKE", "FULL_REPRICING")}
    assert default["GFC_2008"].beta_mapped  # YAML sets equity_beta_mapping: true for historical scenarios


def test_replay_window_reproduces_stored_moves(ctx, settings):
    hist = ctx.history
    s_, e_ = hist.dates[-11], hist.dates[-1]
    sc = replay_window_scenario(hist, s_.date(), e_.date())
    for ticker, move in sc.equity_overrides.items():
        px = hist.equity_returns_px[ticker]
        assert move == pytest.approx(px.loc[e_] / px.loc[s_] - 1)
    for tenor, move in sc.curve_shocks.items():
        curve = hist.curves["USD_TSY"][tenor]
        assert move == pytest.approx(curve.loc[e_] - curve.loc[s_])
    for ccy, move in sc.fx_shocks.items():
        assert move == pytest.approx(hist.fx_usd[ccy].loc[e_] / hist.fx_usd[ccy].loc[s_] - 1)
    assert set(sc.credit_curve_shocks) == set(hist.curves["USD_IG"].columns)
    assert sc.basis_for("equity") == "HISTORICAL_OBSERVATION"
    # A one-day replay ending on the valuation date equals that day's historical-simulation scenario.
    one = StressService(settings).replay(ctx, hist.dates[-2].date(), e_.date(), "STICKY_STRIKE", "FULL_REPRICING")
    scen_pnl = RiskService.scenario_pnl(ctx, 100).iloc[-1]
    np.testing.assert_allclose(one.position_pnl.to_numpy(), scen_pnl.to_numpy(), rtol=1e-9, atol=1e-6)


def test_replay_window_validation(ctx):
    hist = ctx.history
    weekend = hist.dates[-1] + pd.Timedelta(days=2)
    assert snap_to_history(hist, weekend.date()) == hist.dates[-1]  # snaps back to the last trading day
    with pytest.raises(ValueError):
        replay_window_scenario(hist, hist.dates[-1].date(), hist.dates[-5].date())
    with pytest.raises(ValueError):
        snap_to_history(hist, (hist.dates[0] - pd.Timedelta(days=30)).date())
    fw = BacktestService(get_settings()).forecast_windows(ctx, 60)
    start, end, total = worst_window(fw.actual_pnl, 5)
    assert total < 0 and start < end and start in hist.dates and end in hist.dates


# ------------------------------------------------- vol factor model & dataset refresh
def test_vol_factor_models(ctx, settings):
    frame = ctx.history.factor_frame()
    kinds = {c.split("|")[0] for c in frame.columns}
    assert {"vol", "volskew", "volterm"} <= kinds
    assert (frame[[c for c in frame.columns if c.startswith("volskew|")]].std() > 0).all()  # skew really moves in the data
    svc = RiskService(settings)
    level = svc.scenario_frames(ctx, RiskConfiguration(lookback=100, vol_factors="LEVEL"))
    full = svc.scenario_frames(ctx, RiskConfiguration(lookback=100, vol_factors="LEVEL_SKEW_TERM"))
    assert not any(c.startswith(("volskew|", "volterm|")) for c in level["HISTORICAL"].columns)
    assert len(full["HISTORICAL"].columns) == len(level["HISTORICAL"].columns) + 6
    assert full["MONTE_CARLO"].shape[1] == full["FACTOR_COV"].shape[0]
    groups = svc.factor_group_var(ctx, RiskConfiguration(lookback=100))
    assert groups.iloc[-1]["var"] < groups.iloc[:-1]["var"].sum()  # diversification across factor groups
    assert {"Implied vol skew", "Implied vol term structure"} <= set(groups["group"])


def test_skew_and_term_factors_explain_option_vol_moves(ctx):
    """The 3-factor surface model tracks each option's own vol far better than the ATM level alone."""
    from analytics.volatility_surface import skew_loading, term_loading

    hist, frame = ctx.history, ctx.history.factor_frame()
    for pid in [p.position_id for p in ctx.portfolio.positions if p.option is not None]:
        spec = ctx.pricer.option_spec(pid)
        u, err_level, err_full = spec.underlying, [], []
        for i in range(len(hist.dates) - 120, len(hist.dates)):
            s0, s1 = hist.state_at(hist.dates[i - 1].date()), hist.state_at(hist.dates[i].date())
            m, tau = spec.strike / s0.price(u), (spec.expiry - s0.date).days / 365
            actual = float(s1.vol_surfaces[u].vol(m, tau)) - float(s0.vol_surfaces[u].vol(m, tau))
            row = frame.iloc[i - 1]
            level = row[f"vol|{u}"]
            full = level + row[f"volskew|{u}"] * float(skew_loading(m)) + row[f"volterm|{u}"] * float(term_loading(tau))
            err_level.append(actual - level)
            err_full.append(actual - full)
        assert np.std(err_full) < 0.5 * np.std(err_level)


def test_backtest_respects_vol_factor_model(ctx, settings):
    svc = BacktestService(settings)
    a, b = svc.forecast_windows(ctx, 60, "LEVEL"), svc.forecast_windows(ctx, 60, "LEVEL_SKEW_TERM")
    assert a.vol_factors == "LEVEL" and b.vol_factors == "LEVEL_SKEW_TERM"
    np.testing.assert_allclose(a.actual_pnl, b.actual_pnl)  # realised P&L does not depend on the VaR factor model
    assert not np.allclose(a.windows, b.windows)


def test_dataset_refresh_updates_in_place(settings, tmp_path):
    from scripts.seed_database import SEED_VERSION, needs_refresh

    repo = RiskRepository(create_db_engine(f"sqlite:///{tmp_path / 'refresh.db'}"))
    seed_database(repo, settings)
    counts = repo.table_counts()
    assert not needs_refresh(repo)
    # Simulate a database seeded by an older dataset: stale marker and a corrupted surface point.
    repo.set_meta("seed_version", "0")
    stale = repo.load_vol_surface().head(1).assign(implied_vol=0.99)
    repo.upsert_vol_surface(stale, update=True)
    assert needs_refresh(repo) and not is_seeded(repo)
    seed_database(repo, settings, refresh=True)
    assert repo.get_meta("seed_version") == SEED_VERSION and is_seeded(repo)
    after = repo.table_counts()
    assert {k: after[k] for k in ("market_data", "volatility_surface", "yield_curve_data", "positions", "instruments")} == {
        k: counts[k] for k in ("market_data", "volatility_surface", "yield_curve_data", "positions", "instruments")
    }  # refreshed in place, nothing duplicated
    restored = repo.load_vol_surface().head(1)["implied_vol"].iloc[0]
    assert restored != pytest.approx(0.99)


# --------------------------------------------------------------------- end-of-day batch
def test_end_of_day_batch_persists_and_is_idempotent(settings, tmp_path):
    from services.eod_service import default_configuration, last_run, run_end_of_day

    repo = RiskRepository(create_db_engine(f"sqlite:///{tmp_path / 'eod.db'}"))
    seed_database(repo, settings)
    assert last_run(repo) is None
    risk = settings.risk.model_copy(update={"lookback_days": 100})
    cfg = settings.model_copy(update={"risk": risk})
    assert default_configuration(cfg).lookback == 100
    result = run_end_of_day(repo, cfg, reports_dir=tmp_path / "reports")
    counts = repo.table_counts()
    assert result.nav > 0 and result.es_99 >= result.var_99 > 0
    assert set(result.backtest_exceptions) == set(METHODS) and set(result.basel_zones) == set(METHODS)
    assert counts["risk_metrics"] == 12 and counts["risk_contributions"] == 20 and counts["stress_results"] == 8
    assert counts["backtest_results"] == 6 * (N_DAYS - 101) and counts["pnl_attribution"] > 0
    assert [p.rsplit("/", 1)[-1] for p in result.reports] == ["risk_report.xlsx", "risk_summary.pdf"]
    assert result.ok and result.rows_written["backtest_results"] == counts["backtest_results"]
    again = run_end_of_day(repo, cfg)  # same valuation date: results are replaced, not duplicated
    assert repo.table_counts() == counts and again.rows_written == {}
    assert again.var_99 == pytest.approx(result.var_99)
    assert last_run(repo)["valuation_date"] == result.valuation_date


def test_eod_scheduler_helpers():
    import argparse
    from datetime import datetime

    from scripts.run_eod import next_run_time, parse_time

    assert parse_time("18:00") == (18, 0)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_time("6pm")
    thursday_morning = datetime(2026, 10, 1, 9, 30)
    assert next_run_time(thursday_morning, 18, 0) == datetime(2026, 10, 1, 18, 0)  # later today
    friday_evening = datetime(2026, 10, 2, 19, 0)
    assert next_run_time(friday_evening, 18, 0) == datetime(2026, 10, 5, 18, 0)  # skips the weekend
    assert next_run_time(friday_evening, 18, 0, weekdays_only=False) == datetime(2026, 10, 3, 18, 0)
    assert next_run_time(datetime(2026, 10, 1, 18, 0), 18, 0) == datetime(2026, 10, 2, 18, 0)  # strictly after now
