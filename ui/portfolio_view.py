"""02 Portfolio - valuation, exposures, position drilldown and CSV upload."""

from __future__ import annotations

import hashlib
import logging

import streamlit as st

from analytics.revaluation import MarketDataMissingError
from models.portfolio import PortfolioValidationError
from services.portfolio_service import PortfolioService, price_reconciliation
from services.risk_service import build_context
from ui import components as C
from ui import styles as S
from ui.cache import App, clear_calculation_caches, market_history

logger = logging.getLogger(__name__)


def render(app: App) -> None:
    C.page_header("Portfolio", app.meta(), app.status.ok)
    val = app.ctx.valuation
    exp = PortfolioService.exposures(val)
    nav = exp["nav"]

    def share(x: float) -> str:
        return f"{C.fmt_pct(x / nav, 1)} of NAV"

    C.kpi_row(
        [
            ("Portfolio NAV", C.fmt_usd(nav, decimals=2), app.portfolio.base_currency, ""),
            ("Equity Exposure", C.fmt_usd(exp["equity"]), share(exp["equity"]), ""),
            ("Fixed Income Exposure", C.fmt_usd(exp["fixed_income"]), share(exp["fixed_income"]), ""),
            ("FX Exposure", C.fmt_usd(exp["fx"]), share(exp["fx"]), ""),
        ]
    )
    C.kpi_row(
        [
            ("Options Exposure", C.fmt_usd(exp["options"]), share(exp["options"]), ""),
            ("Cash", C.fmt_usd(exp["cash"]), share(exp["cash"]), ""),
            ("Gross Exposure", C.fmt_usd(exp["gross"]), f"Long {C.fmt_usd(exp['long'])} · Short {C.fmt_usd(exp['short'])}", ""),
            ("Net Exposure", C.fmt_usd(exp["net"]), share(exp["net"]), ""),
        ]
    )

    tab_pos, tab_drill, tab_upload = st.tabs(["Positions", "Position drilldown", "Upload portfolio"])
    with tab_pos:
        left, right = st.columns(2)
        with left:
            by_class = PortfolioService.exposure_by(val, "asset_class")["weight"]
            fig = C.bar(by_class, "Asset-class weight (% NAV)", money=False, pct=True)
            fig.update_traces(marker_color=[S.ASSET_CLASS_COLORS.get(i, S.BLUE) for i in by_class.sort_values().index])
            C.show(fig, 250, key="pf_class")
        with right:
            by_ccy = PortfolioService.exposure_by(val, "exposure_currency")["gross_share"]
            C.show(C.bar(by_ccy, "Currency exposure (% of gross)", money=False, pct=True), 250, key="pf_ccy")
        table = val[["position_id", "ticker", "description", "asset_class", "quantity", "price", "currency", "exposure_currency", "market_value", "weight", "side"]]
        C.dataframe(
            table,
            {
                "position_id": "Position", "ticker": "Ticker", "description": "Instrument", "asset_class": "Asset Class",
                "quantity": st.column_config.NumberColumn("Quantity", format="localized"),
                "price": st.column_config.NumberColumn("Price (input)", format="%.4f"),
                "currency": "Ccy", "exposure_currency": "Exposure Ccy",
                "market_value": C.money_col("Market Value (USD)"), "weight": C.pct_col("Weight"), "side": "Side",
            },
        )
        C.note("Market value uses market / model prices on the valuation date. Bonds: dirty price x face / 100. "
               "FX: foreign-currency amount x USD per unit. Options: Black-Scholes x contracts x multiplier.")
        C.csv_download(table, "positions.csv", key="dl_positions")

    with tab_drill:
        _drilldown(app)
    with tab_upload:
        _upload(app)


def _drilldown(app: App) -> None:
    val = app.ctx.valuation.set_index("position_id")
    labels = {pid: f"{pid} · {r.ticker} · {r.asset_class}" for pid, r in val.iterrows()}
    pid = st.selectbox("Position", list(labels), format_func=labels.get, key="drill_pid")
    row = val.loc[pid]
    rep = app.var_report()
    cfg = app.config()
    contrib = app.risk.contributions(app.ctx, rep, cfg.covariance_method).set_index("position_id")
    worst = app.stress.worst(app.stress_results())
    stress_loss = float(worst.position_pnl[pid]) if worst else float("nan")
    C.kpi_row(
        [
            ("Ticker", row.ticker, row.description, ""),
            ("Asset Type", row.asset_type, row.asset_class, ""),
            ("Quantity", C.fmt_num(row.quantity, 0), f"price {C.fmt_num(row.price, 4)} {row.currency}", ""),
            ("Market Value", C.fmt_usd(row.market_value, decimals=2), f"{C.fmt_pct(row.weight)} of NAV", ""),
            ("VaR Contribution", C.fmt_usd(contrib.at[pid, "component_var"]), f"{C.fmt_pct(contrib.at[pid, 'pct_contribution'], 1)} of VaR ({cfg.covariance_method})", ""),
            ("Stress Loss", C.fmt_usd(stress_loss), worst.scenario.name if worst else "", C.sign_class(stress_loss)),
        ]
    )
    sens = app.sensitivities()
    if row.asset_type == "OPTION":
        o = sens["options"].set_index("position_id").loc[pid]
        C.section("Option sensitivities (position, USD)")
        C.kpi_row(
            [
                ("Delta", C.fmt_num(o["delta"], 0), f"shares · {C.fmt_usd(o['delta_usd'])} delta $", ""),
                ("Gamma", C.fmt_num(o["gamma"], 2), "delta per $1 spot", ""),
                ("Vega", C.fmt_usd(o["vega"], decimals=2), "per vol point", ""),
                ("Theta", C.fmt_usd(o["theta"], decimals=2), "per calendar day", ""),
                ("Rho", C.fmt_usd(o["rho"], decimals=2), "per 1% rates", ""),
                ("Implied Volatility", C.fmt_pct(o["implied_vol"], 1), f"K {o['strike']:g} · T {o['time_to_expiry']:.2f}y", ""),
            ]
        )
    elif row.asset_type == "BOND":
        b = sens["bonds"].set_index("position_id").loc[pid]
        C.section("Fixed-income sensitivities")
        C.kpi_row(
            [
                ("Yield", C.fmt_pct(b["ytm"], 3), f"coupon {C.fmt_pct(b['coupon'], 3)} · {b['maturity']:%d %b %Y}", ""),
                ("Duration", f"{b['macaulay_duration']:.2f}", "Macaulay (years)", ""),
                ("Modified Duration", f"{b['modified_duration']:.2f}", f"effective {b['effective_duration']:.2f}", ""),
                ("DV01", C.fmt_usd(b["dv01"], decimals=2), "USD per 1 bp", ""),
                ("Convexity", f"{b['convexity']:.1f}", f"spread {b['spread_bps']:.0f} bp · {b['credit_curve']}", ""),
            ]
        )
        krd = sens["krd"].loc[pid]
        C.show(C.bar(krd, "Key rate duration by tenor", money=False, horizontal=False, sort=False), 240, key="drill_krd")


def _upload(app: App) -> None:
    C.note(
        "Upload a CSV with columns <code>position_id, asset_type, ticker, quantity, price, market_value, currency, weight, "
        "maturity, face_value, coupon, strike, option_type, volatility</code> (optional: <code>coupon_frequency, credit_curve</code>). "
        "Market value and weight are recomputed. Invalid portfolios are rejected before reaching the risk engine."
    )
    sample = app.settings.sample_data_dir / "sample_portfolio.csv"
    if sample.exists():
        st.download_button("Download template (sample portfolio)", sample.read_bytes(), "sample_portfolio.csv", "text/csv", key="dl_template")
    file = st.file_uploader("Portfolio CSV", type=["csv"], key="pf_upload")
    c1, c2 = st.columns(2)
    name = c1.text_input("Portfolio name", "Uploaded Portfolio")
    vd = c2.date_input("Valuation date", value=app.history.last_date, max_value=app.history.last_date,
                       min_value=app.history.dates[0].date(), format="YYYY-MM-DD")
    if file is None:
        return
    data = file.getvalue()
    pid = "UPLOAD_" + hashlib.sha256(data).hexdigest()[:8]
    history = market_history(app.repo, vd, app.version)
    result = app.portfolio_service.validate_upload(data, history, vd, pid, name)
    if result.portfolio is None:
        st.error("The portfolio was rejected. Fix the issues below and upload again.")
        st.code(result.error_text or "Portfolio validation failed", language=None)
        return
    try:
        ctx = build_context(result.portfolio, history, app.settings)
    except (MarketDataMissingError, ValueError) as exc:
        logger.exception("Uploaded portfolio could not be valued")
        C.friendly_error(str(exc))
        return
    warnings = {**result.report.warnings}
    for k, v in price_reconciliation(result.portfolio, ctx.pricer).items():
        warnings.setdefault(k, []).extend(v)
    st.success(f"{len(result.portfolio.positions)} positions validated · NAV {C.fmt_usd(ctx.nav, decimals=2)}")
    if warnings:
        st.warning("\n".join(f"{k}: {m}" for k, msgs in warnings.items() for m in msgs))
    C.dataframe(ctx.valuation[["position_id", "ticker", "asset_class", "quantity", "market_value", "weight"]],
                {"market_value": C.money_col("Market Value (USD)"), "weight": C.pct_col("Weight")})
    b1, b2, b3 = st.columns([1, 1, 1])
    if b1.button("Use for this session", type="primary", key="use_upload"):
        st.session_state["uploaded_portfolio"] = result.portfolio.model_dump_json()
        st.session_state["portfolio_source"] = "UPLOAD"
        st.rerun()
    save_id = b2.text_input("Save as portfolio ID", value="", placeholder="e.g. MACRO2", key="save_id", label_visibility="collapsed")
    if b3.button("Save to database", key="save_upload", disabled=not save_id.strip()):
        pid_clean = "".join(ch for ch in save_id.strip().upper() if ch.isalnum() or ch in "_-")[:32]
        if not pid_clean or pid_clean.startswith("UPLOAD"):
            C.friendly_error("Choose a portfolio ID made of letters, digits, '_' or '-' (not starting with UPLOAD).")
            return
        try:
            to_save = result.portfolio.model_copy(update={"portfolio_id": pid_clean, "name": name})
            app.portfolio_service.save_portfolio(to_save, ctx.valuation, history.instruments)
        except (PortfolioValidationError, ValueError) as exc:
            logger.exception("Saving portfolio failed")
            C.friendly_error(f"Portfolio could not be saved: {exc}")
            return
        clear_calculation_caches()
        st.session_state["portfolio_source"] = pid_clean
        st.session_state.pop("uploaded_portfolio", None)
        st.rerun()
