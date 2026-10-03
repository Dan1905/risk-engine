"""07 Scenario Builder - interactive custom stress scenarios."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from analytics.risk_limits import stress_limit_checks
from models.positions import AssetType
from models.risk import StressScenario
from stress.hypothetical import custom_scenario
from ui import components as C
from ui.cache import App
from ui.stress_view import method_controls, scenario_detail

logger = logging.getLogger(__name__)


def _override_targets(app: App) -> dict[str, str]:
    """Instrument -> override type for the portfolio's risk factors."""
    out: dict[str, str] = {}
    for p in app.portfolio.positions:
        if p.asset_type == AssetType.EQUITY:
            out[p.ticker] = "Equity (%)"
        elif p.asset_type == AssetType.OPTION and p.option is not None:
            out.setdefault(p.option.underlying, "Equity (%)")
            out[f"{p.option.underlying} vol"] = "Implied vol (pts)"
        elif p.asset_type == AssetType.BOND:
            out[p.ticker] = "Bond yield (bp)"
        if p.exposure_currency != "USD":
            out[p.exposure_currency] = "Currency vs USD (%)"
    return dict(sorted(out.items()))


def render(app: App) -> None:
    C.page_header("Scenario Builder", app.meta(), app.status.ok)
    method_controls(app, "sb")
    targets = _override_targets(app)
    with st.form("scenario_builder"):
        name = st.text_input("Scenario name", "Custom Scenario")
        c1, c2, c3, c4, c5 = st.columns(5)
        eq = c1.number_input("Equity shock (%)", -99.0, 200.0, -10.0, 1.0)
        rates = c2.number_input("Rates shock (bp)", -500.0, 1000.0, 50.0, 5.0)
        fx = c3.number_input("FX shock: USD vs all (%)", -50.0, 100.0, 5.0, 1.0)
        vol = c4.number_input("Volatility shock (vol pts)", -50.0, 200.0, 5.0, 1.0)
        credit = c5.number_input("Credit spread shock (bp)", -300.0, 1000.0, 0.0, 5.0)
        st.markdown("**Asset-specific overrides** (replace the uniform shock for that instrument)")
        default = pd.DataFrame({"Instrument": pd.Series(dtype="str"), "Shock": pd.Series(dtype="float")})
        overrides = st.data_editor(
            default, num_rows="dynamic", width="stretch", key="sb_overrides",
            column_config={
                "Instrument": st.column_config.SelectboxColumn("Instrument", options=list(targets), required=True),
                "Shock": st.column_config.NumberColumn("Shock (units per instrument type)", format="%.2f", required=True),
            },
        )
        C.note("Units: equities and currencies in %, bond yields in bp, implied vol in vol points. "
               "Examples: AAPL = -25, MSFT = -20, EUR = +10, US10Y = +150. " +
               ", ".join(f"<b>{k}</b>: {v}" for k, v in targets.items()))
        save = st.checkbox("Save scenario to the scenario library", value=False)
        submitted = st.form_submit_button("RUN STRESS TEST", type="primary", width="stretch")

    if submitted:
        eq_o, fx_o, y_o, v_o = {}, {}, {}, {}
        for row in overrides.dropna().itertuples(index=False):
            inst, val = str(row.Instrument), float(row.Shock)
            kind = targets.get(inst)
            if kind == "Equity (%)":
                eq_o[inst] = val / 100
            elif kind == "Currency vs USD (%)":
                fx_o[inst] = val / 100
            elif kind == "Bond yield (bp)":
                y_o[inst] = val
            elif kind == "Implied vol (pts)":
                v_o[inst.replace(" vol", "")] = val / 100
        try:
            sc = custom_scenario(
                name, equity_shock=eq / 100, rates_shock_bps=rates, fx_shock=fx / 100, volatility_shock=vol / 100,
                credit_spread_shock_bps=credit, equity_overrides=eq_o, fx_overrides=fx_o, yield_overrides_bps=y_o, vol_overrides=v_o,
            )
        except ValidationError as exc:
            C.friendly_error("Scenario inputs are invalid: " + "; ".join(e["msg"].replace("Value error, ", "") for e in exc.errors()))
            return
        st.session_state["builder_scenario"] = sc.model_dump_json()
        if save:
            app.stress.save_custom(sc)
            st.toast(f"Scenario '{sc.name}' saved to the library")

    raw = st.session_state.get("builder_scenario")
    if not raw:
        C.note("Define shocks and press RUN STRESS TEST.")
        return
    sc = StressScenario.model_validate_json(raw)
    result = app.stress.run(app.ctx, sc, app.vol_method(), app.bond_method())
    C.section("Result")
    scenario_detail(app, result, "sb_detail")
    C.section("Risk limit status after stress")
    asset_classes = app.ctx.valuation.set_index("position_id")["asset_class"]
    C.limit_bars(stress_limit_checks(app.ctx.market_values, result.position_pnl, asset_classes, app.settings.limits))
    C.csv_download(result.position_pnl.to_frame(), f"{sc.scenario_id.lower()}_position_pnl.csv", key="dl_builder")

    saved = app.stress.custom_scenarios()
    if saved:
        C.section("Saved custom scenarios")
        C.html_table(pd.DataFrame([{"Scenario": s.name, "Equity": f"{s.equity_shock:+.0%}", "Rates": f"{s.rates_shock * 1e4:+.0f}bp",
                                    "USD": f"{s.fx_shock:+.0%}", "Vol": f"{s.volatility_shock * 100:+.0f}pts",
                                    "Overrides": len(s.equity_overrides) + len(s.fx_shocks) + len(s.yield_overrides) + len(s.vol_overrides)}
                                   for s in saved]))
