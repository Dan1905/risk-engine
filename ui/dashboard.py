"""01 Risk Overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.portfolio_service import PortfolioService
from ui import components as C
from ui import styles as S
from ui.cache import App


def render(app: App) -> None:
    C.page_header("Risk Overview", app.meta(), app.status.ok)
    cfg = app.config()
    rep = app.var_report()
    hist = rep.get("HISTORICAL", 0.99)
    vol = app.risk.portfolio_volatility_annual(rep, cfg.covariance_method)
    stress = app.stress_results()
    worst = app.stress.worst(stress)
    nav = app.ctx.nav
    h = f"{cfg.holding_period}D"

    C.kpi_row(
        [
            ("Portfolio NAV", C.fmt_usd(nav, decimals=2), f"{len(app.portfolio.positions)} positions", ""),
            (f"{h} VaR 99%", C.fmt_usd(hist.var), f"{C.fmt_pct(hist.var_pct)} of NAV · historical", ""),
            (f"{h} ES 99%", C.fmt_usd(hist.expected_shortfall), f"{C.fmt_pct(hist.es_pct)} of NAV · ES/VaR {hist.es_var_ratio:.2f}", ""),
            ("Portfolio Volatility", C.fmt_pct(vol), f"annualised · {'EWMA' if cfg.covariance_method == 'EWMA' else 'historical'} covariance", ""),
            ("Worst Stress Loss", C.fmt_usd(worst.total_pnl) if worst else "n/a",
             f"{C.fmt_pct(worst.portfolio_return, signed=True)} · {worst.scenario.name}" if worst else "", "loss" if worst else ""),
        ]
    )

    roll = app.rolling().iloc[-250:]
    left, right = st.columns(2)
    with left:
        C.show(C.time_bars(roll["pnl"], "Daily P&L (static portfolio, last 250 days)"), 300, key="ov_pnl")
    with right:
        df = roll[["var_HISTORICAL", "var_EW_HISTORICAL", "var_EW_PARAMETRIC"]].rename(
            columns={"var_HISTORICAL": "Historical", "var_EW_HISTORICAL": "EW Historical", "var_EW_PARAMETRIC": "EW Parametric"}
        )
        C.show(C.lines(df, f"Rolling 1D VaR {cfg.confidence:.0%} (out-of-sample, lookback {cfg.lookback})", money=True), 300, key="ov_var")

    left, right = st.columns(2)
    with left:
        vols = roll[[c for c in roll.columns if c.startswith("vol_")]].rename(
            columns=lambda c: "EWMA" if c == "vol_ewma" else c.replace("vol_", "").replace("d", "-day")
        )
        C.show(C.lines(vols, "Rolling volatility (annualised, % NAV)", pct=True), 300, key="ov_vol")
    with right:
        contrib = app.risk.contributions(app.ctx, rep, cfg.covariance_method)
        s = contrib.set_index("ticker")["pct_contribution"]
        s.index = [f"{t} ({pid})" for t, pid in zip(contrib["ticker"], contrib["position_id"])]
        s = s[s.abs() > 1e-4]
        fig = C.bar(s, f"Risk contribution (% of parametric VaR, {cfg.covariance_method})", money=False, pct=True)
        fig.update_traces(marker_color=[S.BLUE if v >= 0 else S.AQUA for v in s.sort_values().to_numpy()])
        C.show(fig, 300, key="ov_contrib")

    left, right = st.columns(2)
    with left:
        by_class = PortfolioService.exposure_by(app.ctx.valuation, "asset_class")["weight"]
        fig = C.bar(by_class, "Asset-class exposure (% NAV)", money=False, pct=True)
        fig.update_traces(marker_color=[S.ASSET_CLASS_COLORS.get(i, S.BLUE) for i in by_class.sort_values().index])
        C.show(fig, 280, key="ov_class")
    with right:
        s = pd.Series({r.scenario.name: r.total_pnl for r in stress})
        C.show(C.signed_bar(s, "Stress scenario P&L"), 280, key="ov_stress")

    left, right = st.columns([1.1, 1])
    with left:
        C.section("Risk limit status")
        C.limit_bars(app.risk.limit_checks(app.ctx, hist))
        C.note(f"VaR / ES: 1-day historical at 99%, lookback {cfg.lookback}. Limits from configuration (Settings page overrides apply to this session).")
    with right:
        C.section("Concentration warnings")
        conc = app.risk.concentration(app.ctx, contrib, worst.position_pnl if worst else None, worst.scenario.name if worst else None)
        C.warnings_block(conc.warnings)
