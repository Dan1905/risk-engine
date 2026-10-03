"""11 Reports - CSV extracts, Excel workbook, PDF executive summary."""

from __future__ import annotations

import logging

import streamlit as st

from services.report_service import SECTION_ORDER, ReportService, RiskReportData
from ui import components as C
from ui.cache import App

logger = logging.getLogger(__name__)


def build_report(app: App) -> RiskReportData:
    cfg = app.config()
    rep = app.var_report()
    risk = app.risk
    contrib = risk.contributions(app.ctx, rep, cfg.covariance_method)
    stress = app.stress_results()
    worst = app.stress.worst(stress)
    results, frame = app.attribution()
    fw = app.forecasts()
    bt = app.backtest.run(app.ctx, fw, app.settings.backtest.default_method, cfg.confidence, lam=cfg.decay_factor, persist=False,
                          cov_method=cfg.covariance_method)
    return ReportService(app.settings).build(
        ctx=app.ctx,
        var_report=rep,
        contributions=contrib,
        contributions_by_class=risk.contributions_by(contrib, "asset_class"),
        sensitivities=app.sensitivities(),
        stress_results=stress,
        attribution=results[-1],
        attribution_history=frame,
        backtest=bt,
        limit_checks=risk.limit_checks(app.ctx, rep.get("HISTORICAL", 0.99)),
        concentration=risk.concentration(app.ctx, contrib, worst.position_pnl if worst else None, worst.scenario.name if worst else None),
        portfolio_volatility=risk.portfolio_volatility_annual(rep, cfg.covariance_method),
        basel=app.backtest.basel_assessment(fw, app.settings.backtest.default_method, cfg.decay_factor, ctx=app.ctx,
                                            cov_method=cfg.covariance_method),
    )


def render(app: App) -> None:
    C.page_header("Reports", app.meta(), app.status.ok)
    cfg = app.config()
    C.note(f"Reports use the current session settings: confidence {cfg.confidence:.0%}, lookback {cfg.lookback}, holding period "
           f"{cfg.holding_period}D, λ {cfg.decay_factor:.2f}, {cfg.covariance_method} covariance, stress method "
           f"{app.vol_method().replace('_', ' ').title()} / {app.bond_method().replace('_', ' ').title()}.")
    data = build_report(app)
    stamp = f"{data.portfolio_id}_{data.valuation_date:%Y%m%d}"

    C.section("CSV extracts")
    exports = ReportService.csv_exports(data)
    cols = st.columns(5)
    for i, (label, (fname, payload)) in enumerate(exports.items()):
        cols[i % 5].download_button(label, payload, file_name=f"{stamp}_{fname}", mime="text/csv", key=f"csv_{fname}", width="stretch")

    C.section("Structured reports")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Excel risk report**")
        C.note(f"{len(data.tables())} sheets: " + ", ".join(data.tables()))
        if st.button("Generate Excel workbook", key="gen_xlsx"):
            try:
                st.session_state["report_xlsx"] = (stamp, ReportService.excel(data))
            except Exception:  # noqa: BLE001
                logger.exception("Excel report generation failed")
                C.friendly_error("The Excel report could not be generated. See the log for details.")
        if st.session_state.get("report_xlsx", (None,))[0] == stamp:
            st.download_button("Download risk_report.xlsx", st.session_state["report_xlsx"][1], file_name=f"{stamp}_risk_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", key="dl_xlsx")
    with c2:
        st.markdown("**PDF executive summary**")
        C.note("Sections: " + ", ".join(SECTION_ORDER))
        if st.button("Generate PDF summary", key="gen_pdf"):
            try:
                st.session_state["report_pdf"] = (stamp, ReportService.pdf(data))
            except Exception:  # noqa: BLE001
                logger.exception("PDF report generation failed")
                C.friendly_error("The PDF report could not be generated. See the log for details.")
        if st.session_state.get("report_pdf", (None,))[0] == stamp:
            st.download_button("Download risk_summary.pdf", st.session_state["report_pdf"][1], file_name=f"{stamp}_risk_summary.pdf",
                               mime="application/pdf", type="primary", key="dl_pdf")

    C.section("Report preview")
    tabs = st.tabs(["Summary", "Risk metrics", "Stress", "Backtest", "Limits"])
    with tabs[0]:
        C.html_table(data.summary.astype({"Value": str}))
    with tabs[1]:
        C.dataframe(data.risk_metrics, {"VaR % NAV": C.pct_col("VaR % NAV"), "ES % NAV": C.pct_col("ES % NAV"),
                                        "VaR (USD)": C.money_col("VaR (USD)"), "ES (USD)": C.money_col("ES (USD)"),
                                        "1D sigma (USD)": C.money_col("1D sigma (USD)"), "Confidence": C.pct_col("Confidence")})
    with tabs[2]:
        C.dataframe(data.stress, {"total_pnl": C.money_col("P&L"), "pnl_pct": C.pct_col("P&L %"), "stressed_nav": C.money_col("Stressed NAV"),
                                  "fixed_income_pnl": C.money_col("FI P&L"), "krd_pnl": C.money_col("KRD impact")})
    with tabs[3]:
        C.html_table(data.backtest_summary.astype({"Value": str}))
    with tabs[4]:
        C.dataframe(data.limits, {"Value": C.pct_col("Value"), "Limit level": C.pct_col("Limit"), "Utilisation": C.pct_col("Utilisation")})
        C.warnings_block(data.warnings)
