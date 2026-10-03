"""06 Stress Testing - predefined scenarios, curve/equity/FX/vol ladders."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from stress.historical import worst_window
from stress.scenario_engine import StressResult, basis_table
from stress.volatility_shocks import describe_method
from stress.yield_curve_shocks import BP
from ui import components as C
from ui.cache import App

VOL_LABELS = {"STICKY_STRIKE": "Sticky Strike", "STICKY_DELTA": "Sticky Delta"}
BOND_LABELS = {"FULL_REPRICING": "Full Repricing", "SENSITIVITY": "Sensitivity Approximation"}


def method_controls(app: App, key: str) -> None:
    c1, c2 = st.columns(2)
    vm = c1.radio("Volatility stress method", list(VOL_LABELS), index=list(VOL_LABELS).index(app.vol_method()),
                  format_func=VOL_LABELS.get, horizontal=True, key=f"{key}_vm")
    bm = c2.radio("Fixed-income repricing", list(BOND_LABELS), index=list(BOND_LABELS).index(app.bond_method()),
                  format_func=BOND_LABELS.get, horizontal=True, key=f"{key}_bm")
    st.session_state["vol_method"], st.session_state["bond_method"] = vm, bm


def _card(r: StressResult, names: pd.Series) -> str:
    worst = r.worst_position
    worst_txt = f"{names.get(worst, worst)} ({worst})" if worst else "n/a"
    cat = "Historical" if r.scenario.category == "HISTORICAL" else ("Yield curve" if r.scenario.category == "YIELD_CURVE" else "Hypothetical")
    if r.beta_mapped:
        cat += " · beta-mapped"
    color = "#ef6b6b" if r.total_pnl < 0 else "#0ca30c"
    return f"""<div class="re-card"><div class="c">{cat}</div><div class="n">{html.escape(r.scenario.name)}</div>
    <div class="p" style="color:{color}">{C.fmt_usd(r.total_pnl)} <span style="font-size:.8rem">({C.fmt_pct(r.portfolio_return, 2, True)})</span></div>
    <div class="d">Worst position: {html.escape(worst_txt)}<br>Worst asset class: {html.escape(C.asset_label(r.worst_asset_class))}<br>
    Stressed NAV: {C.fmt_usd(r.stressed_nav, decimals=2)}</div></div>"""


def scenario_detail(app: App, r: StressResult, key: str) -> None:
    names = app.ctx.valuation.set_index("position_id")["ticker"]
    C.kpi_row(
        [
            ("Portfolio Loss", C.fmt_usd(r.total_pnl, decimals=2), r.scenario.name, C.sign_class(r.total_pnl)),
            ("Loss %", C.fmt_pct(r.portfolio_return, 2, True), "of NAV", C.sign_class(r.total_pnl)),
            ("Worst Position", f"{names.get(r.worst_position, r.worst_position)}", f"{r.worst_position} · {C.fmt_usd(r.position_pnl.min())}", ""),
            ("Worst Asset Class", C.asset_label(r.worst_asset_class), C.fmt_usd(r.asset_class_pnl.min()), ""),
            ("Stressed NAV", C.fmt_usd(r.stressed_nav, decimals=2), f"from {C.fmt_usd(r.nav, decimals=2)}", ""),
        ]
    )
    left, right = st.columns([1.4, 1])
    with left:
        s = r.position_pnl.copy()
        s.index = [f"{names.get(i, i)} ({i})" for i in s.index]
        s = s[s.abs() > 0.5]
        C.show(C.signed_bar(s, "Position P&L"), 420, key=f"{key}_pos")
    with right:
        C.show(C.signed_bar(r.asset_class_pnl.rename(index=C.asset_label), "Asset-class P&L"), 220, key=f"{key}_cls")
        bt = basis_table(r.scenario)
        if not bt.empty:
            C.section("Scenario inputs: basis")
            C.html_table(bt.rename(columns={"input": "Input", "basis": "Basis"}))
    if r.beta_mapped:
        C.section("Beta-mapped equity shocks")
        betas = pd.DataFrame({"Beta": r.betas, "Applied shock": r.equity_moves}).sort_values("Beta", ascending=False)
        C.html_table(betas.reset_index(names="Ticker"), {"Beta": lambda v: f"{v:.2f}", "Applied shock": lambda v: C.fmt_pct(v, 1, True)},
                     signed=["Applied shock"])
        C.note(f"Shock = beta × {r.scenario.equity_shock:+.0%} index move. Betas are estimated against the equal-weighted equity "
               "market over the configured lookback and average 1, so the index-level move is preserved on average.")
    sc = r.scenario
    inputs = []
    if sc.equity_shock or sc.equity_overrides:
        label = "Equities (average)" if sc.scenario_id.startswith("REPLAY_") else "Equities"
        per_name = "per instrument" if sc.scenario_id.startswith("REPLAY_") else "overrides"
        inputs.append(f"{label} {sc.equity_shock:+.1%}" + (f" ({per_name}: {', '.join(f'{k} {v:+.1%}' for k, v in sc.equity_overrides.items())})" if sc.equity_overrides else ""))
    if sc.rates_shock:
        inputs.append(f"Rates parallel {sc.rates_shock / BP:+.0f} bp")
    if sc.curve_shocks:
        inputs.append("Curve nodes: " + ", ".join(f"{k} {v / BP:+.0f}bp" for k, v in sc.curve_shocks.items()))
    if sc.fx_shock:
        inputs.append(f"USD {sc.fx_shock:+.0%} vs all currencies")
    if sc.fx_shocks:
        inputs.append("FX: " + ", ".join(f"{k} {v:+.1%}" for k, v in sc.fx_shocks.items()))
    if sc.volatility_shock or sc.vol_overrides:
        inputs.append(f"Implied vol {sc.volatility_shock * 100:+.1f} pts" + (
            f" (per underlying: {', '.join(f'{k} {v * 100:+.1f}' for k, v in sc.vol_overrides.items())})" if sc.vol_overrides else ""))
    if sc.vol_skew_shocks or sc.vol_term_shocks:
        inputs.append("Implied-vol shape: " + ", ".join(
            [f"{k} skew {v * 100:+.2f}" for k, v in sc.vol_skew_shocks.items()] + [f"{k} term {v * 100:+.2f}" for k, v in sc.vol_term_shocks.items()]) + " pts")
    if sc.credit_spread_shock:
        inputs.append(f"Credit spreads {sc.credit_spread_shock / BP:+.0f} bp")
    if sc.credit_curve_shocks:
        inputs.append("Credit spread nodes: " + ", ".join(f"{k} {v / BP:+.0f}bp" for k, v in sc.credit_curve_shocks.items()))
    if sc.yield_overrides:
        inputs.append("Bond yield overrides: " + ", ".join(f"{k} {v / BP:+.0f}bp" for k, v in sc.yield_overrides.items()))
    text = "<br>".join(html.escape(x) for x in inputs) or "No shocks"
    extra = f"<br><i>{html.escape(sc.source_notes)}</i>" if sc.source_notes else ""
    window = f" · window {html.escape(sc.window)}" if sc.window else ""
    C.note(f"<b>{html.escape(sc.description or sc.name)}</b>{window}<br>{text}{extra}")
    C.note("<b>Assumptions:</b> " + " ".join(html.escape(a) for a in r.assumptions()))


def render(app: App) -> None:
    C.page_header("Stress Testing", app.meta(), app.status.ok)
    method_controls(app, "st")
    mapping = st.radio(
        "Equity shock mapping for historical scenarios", [True, False], index=0 if app.beta_mapping() else 1, horizontal=True,
        format_func=lambda b: "Beta-mapped (index move × each stock's beta)" if b else "Uniform (same index move for every stock)",
        key="st_beta",
    )
    st.session_state["beta_mapping"] = mapping
    results = app.stress_results()
    names = app.ctx.valuation.set_index("position_id")["ticker"]
    C.note(f"<b>Volatility:</b> {html.escape(describe_method(app.vol_method()))} &nbsp; <b>Bonds:</b> {BOND_LABELS[app.bond_method()]}. "
           "Stress tests are deterministic instantaneous revaluations, not probability forecasts.")

    for row in range(0, len(results), 4):
        cols = st.columns(4)
        for col, r in zip(cols, results[row : row + 4]):
            col.markdown(_card(r, names), unsafe_allow_html=True)
    st.write("")
    s = pd.Series({r.scenario.name: r.total_pnl for r in results})
    C.show(C.signed_bar(s, "Scenario comparison: portfolio P&L"), 320, key="st_cmp")
    C.csv_download(app.stress.summary_table(results).set_index("scenario_id"), "stress_results.csv", key="dl_stress")

    C.section("Scenario detail")
    pick = st.selectbox("Scenario", [r.scenario.name for r in results], key="st_pick")
    scenario_detail(app, next(r for r in results if r.scenario.name == pick), "st_detail")

    _replay(app)

    ladders = app.stress_ladders()
    tab_curve, tab_eq, tab_fx, tab_vol = st.tabs(["Yield curve", "Equity shocks", "FX shocks", "Volatility shocks"])
    with tab_curve:
        tbl = app.stress.curve_table(ladders["curve"])
        tbl["Worst Bond"] = [f"{names.get(b, b)} ({b})" if b else "n/a" for b in tbl["Worst Bond"]]
        C.html_table(tbl, {c: (lambda v: C.fmt_usd(v, decimals=2)) for c in ("Portfolio P&L", "Fixed-Income P&L", "KRD Impact", "Convexity / Non-linear")},
                     signed=["Portfolio P&L", "Fixed-Income P&L", "KRD Impact", "Convexity / Non-linear"])
        C.note("Shocks are applied node by node on the key-rate grid (3M-30Y); twists are flat to 2Y and beyond 10Y, linear in between. "
               "<b>KRD Impact</b> is the first-order key-rate approximation -Σ MV·KRD<sub>k</sub>·Δz<sub>k</sub>; the difference to the "
               "fixed-income P&L is convexity (or the approximation error when Sensitivity mode is selected).")
        C.show(C.signed_bar(tbl.set_index("Scenario")["Fixed-Income P&L"], "Fixed-income P&L by curve scenario"), 280, key="st_curve")
    for tab, k, title in ((tab_eq, "equity", "Equity shock ladder"), (tab_fx, "fx", "Currency shocks"), (tab_vol, "vol", "Implied-vol shocks")):
        with tab:
            rs = ladders[k]
            df = pd.DataFrame([{"Scenario": r.scenario.name, "Portfolio P&L": r.total_pnl, "P&L %": r.portfolio_return,
                                "Stressed NAV": r.stressed_nav, "Worst Position": f"{names.get(r.worst_position, '')} ({r.worst_position})"} for r in rs])
            C.html_table(df, {"Portfolio P&L": lambda v: C.fmt_usd(v, decimals=2), "P&L %": lambda v: C.fmt_pct(v, 2, True),
                              "Stressed NAV": lambda v: C.fmt_usd(v, decimals=2)}, signed=["Portfolio P&L", "P&L %"])
            C.show(C.signed_bar(df.set_index("Scenario")["Portfolio P&L"], title), 260, key=f"st_{k}")
            if k == "fx":
                C.note("USD ±x%: USD strengthens / weakens against every foreign currency. EUR/INR ±x%: that currency moves against USD.")
            if k == "vol":
                C.note(f"Method in use: {html.escape(describe_method(app.vol_method()))}")


def _replay(app: App) -> None:
    """Replay a window of the stored market history as a stress scenario."""
    C.section("Replay a historical window")
    C.note("Applies the market moves that actually occurred between two dates in the stored history - every equity, currency, "
           "curve node, credit-spread node and implied-vol level moves by its own observed amount - to today's portfolio as one "
           "instantaneous shock.")
    dates = app.history.dates
    lo, hi = dates[0].date(), dates[-1].date()
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    days = c3.number_input("Window length (trading days)", 1, 120, 10, 1, key="rp_days")
    if c4.button("Find worst window for this portfolio", key="rp_find"):
        try:
            start, end, _ = worst_window(app.forecasts().actual_pnl, int(days))
            # Widget state may be set before the widgets are created in this run.
            st.session_state["rp_start_input"], st.session_state["rp_end_input"] = start.date(), end.date()
        except ValueError as exc:
            C.friendly_error(str(exc))
    st.session_state.setdefault("rp_start_input", dates[max(len(dates) - 1 - int(days), 0)].date())
    st.session_state.setdefault("rp_end_input", hi)
    start = c1.date_input("Window start (close of)", min_value=lo, max_value=hi, key="rp_start_input")
    end = c2.date_input("Window end (close of)", min_value=lo, max_value=hi, key="rp_end_input")
    if start >= end:
        C.friendly_error("Replay window end date must be after the start date.")
        return
    try:
        result = app.stress.replay(app.ctx, start, end, app.vol_method(), app.bond_method())
    except ValueError as exc:
        C.friendly_error(str(exc))
        return
    scenario_detail(app, result, "rp_detail")
    C.note("A replay revalues today's positions at today's market levels, so it differs from the P&L the static portfolio "
           "accumulated day by day over the same window (option deltas, bond durations and price levels were different then).")
