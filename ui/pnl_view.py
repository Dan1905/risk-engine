"""04 P&L Attribution - waterfall, explained vs unexplained, history."""

from __future__ import annotations

import streamlit as st

from analytics.pnl_attribution import COMPONENT_LABELS, COMPONENTS, explained_ratio, summarise_components
from ui import components as C
from ui import styles as S
from ui.cache import App


def render(app: App) -> None:
    C.page_header("P&L Attribution", app.meta(), app.status.ok)
    n = app.settings.pnl.attribution_days
    results, frame = app.attribution(n)
    dates = [r.date_to for r in results]
    pick = st.select_slider("Attribution date", options=dates, value=dates[-1], format_func=lambda d: f"{d:%d %b %Y}")
    res = results[dates.index(pick)]
    tol = app.settings.pnl.tolerance
    ok = res.reconciles(tol)

    C.kpi_row(
        [
            ("Actual P&L", C.fmt_usd(res.actual, decimals=2), f"{res.date_from:%d %b} → {res.date_to:%d %b %Y}", C.sign_class(res.actual)),
            ("Explained P&L", C.fmt_usd(res.explained, decimals=2), "sum of risk-factor components", ""),
            ("Unexplained (Residual)", C.fmt_usd(res.residual, decimals=2), "higher-order & cross terms", ""),
            ("Explained %", C.fmt_pct(res.explained_pct, 1), "1 - |residual| / |actual|", ""),
            (f"{n}-day Explained %", C.fmt_pct(explained_ratio(frame), 1), "1 - Σ|residual| / Σ|actual|", ""),
            ("Reconciliation", "Pass" if ok else "Break", f"|explained + residual - actual| ≤ ${tol:g}", "gain" if ok else "loss"),
        ]
    )

    t = res.totals
    option_total = sum(t[c] for c in ("option_delta", "option_gamma", "option_vega", "option_theta"))
    labels = ["Previous NAV", "Equity P&L", "Rates P&L", "FX P&L", "Option Delta", "Option Gamma", "Option Vega", "Option Theta",
              "Carry", "Residual", "Current NAV"]
    values = [res.nav_start, t["equity"], t["bond_yield"], t["fx"], t["option_delta"], t["option_gamma"], t["option_vega"],
              t["option_theta"], t["carry"], t["residual"], res.nav_end]
    measures = ["absolute"] + ["relative"] * 9 + ["total"]
    left, right = st.columns([1.5, 1])
    with left:
        fig = C.waterfall(labels, values, measures, "NAV waterfall")
        path = [res.nav_start]
        for v in values[1:-1]:
            path.append(path[-1] + v)
        pad = max(max(path) - min(path), abs(res.nav_start) * 1e-4) * 0.35
        fig.update_yaxes(range=[min(path) - pad, max(path) + pad])
        C.show(fig, 360, key="pnl_wf")
    with right:
        tbl = summarise_components(res)
        C.section("Components (USD)")
        C.html_table(tbl.rename(columns={"component": "Component", "pnl": "P&L"}), {"P&L": lambda v: f"{v:,.2f}"}, signed=["P&L"])
        C.note(f"Option Greeks combined: {C.fmt_usd(option_total, decimals=2)}. Carry = bond accrual at yield. "
               "Rates P&L = -D<sub>mod</sub>·Δy + ½·C·Δy² per bond.")

    series = frame[list(COMPONENTS) + ["residual"]].rename(columns=COMPONENT_LABELS)
    colors = {COMPONENT_LABELS[c]: S.SERIES[i % 8] for i, c in enumerate(COMPONENTS)}
    colors["Residual"] = S.NEUTRAL
    C.show(C.stacked_bars(series, f"Daily attribution, last {len(frame)} trading days", colors, frame["actual"], "Actual P&L"), 360, key="pnl_hist")

    C.section("Position attribution")
    by_pos = res.by_position.copy()
    val = app.ctx.valuation.set_index("position_id")
    by_pos.insert(0, "ticker", val["ticker"].reindex(by_pos.index))
    by_pos.insert(1, "asset_class", val["asset_class"].reindex(by_pos.index))
    cfg = {k: C.money_col(COMPONENT_LABELS.get(k, k.title())) for k in list(COMPONENTS) + ["explained", "residual", "actual"]}
    cfg.update({"ticker": "Ticker", "asset_class": "Asset Class", "position_id": "Position"})
    C.dataframe(by_pos.reset_index(names="position_id"), cfg)
    by_class = by_pos.drop(columns=["ticker"]).groupby("asset_class").sum(numeric_only=True)
    C.dataframe(by_class.reset_index().rename(columns={"asset_class": "Asset Class"}), cfg)
    c1, c2 = st.columns(2)
    with c1:
        C.csv_download(by_pos, f"pnl_attribution_{res.date_to:%Y%m%d}.csv", "Download position attribution", key="dl_pnlpos")
    with c2:
        C.csv_download(frame, "pnl_attribution_history.csv", "Download attribution history", key="dl_pnlhist")
    C.note("Assumes static positions between close-to-close valuations (no intraday trading). Explained P&L uses start-of-day "
           "sensitivities; actual P&L is full revaluation plus coupon cash.")
