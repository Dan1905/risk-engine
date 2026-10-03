"""Multi-Asset Portfolio Risk & Stress Testing Engine - Streamlit entry point.

    streamlit run app.py

On first launch the database is created, sample data is generated and seeded
(idempotently), the portfolio is validated, and the dashboard opens fully
populated. Subsequent launches find the seed marker and skip seeding.

This module only handles routing, session state and error presentation; all
financial logic lives in ``analytics``, ``stress`` and ``services``.
"""

from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(
    page_title="Multi-Asset Risk Engine",
    page_icon=":material/query_stats:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from analytics.covariance import SingularCovarianceError  # noqa: E402
from analytics.revaluation import MarketDataMissingError  # noqa: E402
from analytics.var import InsufficientDataError  # noqa: E402
from config.settings import configure_logging, get_settings  # noqa: E402
from models.portfolio import PortfolioValidationError  # noqa: E402
from ui import (  # noqa: E402
    backtest_view,
    contributions_view,
    dashboard,
    market_data_view,
    pnl_view,
    portfolio_view,
    reports_view,
    risk_view,
    scenario_builder_view,
    sensitivities_view,
    settings_view,
    stress_view,
)
from ui.cache import App, build_app, initialise  # noqa: E402
from ui.styles import CSS  # noqa: E402

configure_logging(get_settings())
logger = logging.getLogger("app")

PAGES = {
    "01  Risk Overview": dashboard.render,
    "02  Portfolio": portfolio_view.render,
    "03  VaR & Expected Shortfall": risk_view.render,
    "04  P&L Attribution": pnl_view.render,
    "05  Sensitivities": sensitivities_view.render,
    "06  Stress Testing": stress_view.render,
    "07  Scenario Builder": scenario_builder_view.render,
    "08  Risk Contributions": contributions_view.render,
    "09  VaR Backtesting": backtest_view.render,
    "10  Market Data": market_data_view.render,
    "11  Reports": reports_view.render,
    "12  Settings": settings_view.render,
}


def _friendly(exc: Exception) -> str:
    if isinstance(exc, PortfolioValidationError):
        return exc.format()
    if isinstance(exc, SingularCovarianceError):
        return "Covariance matrix is singular."
    if isinstance(exc, (MarketDataMissingError, InsufficientDataError)):
        return str(exc)
    if isinstance(exc, ValueError):
        return str(exc)
    return "An unexpected error occurred. Technical details were written to the log."


def sidebar(repo) -> str:
    with st.sidebar:
        st.markdown('<div class="re-brand">RISK <span>ENGINE</span></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), key="page", label_visibility="collapsed")
        st.markdown('<div class="re-section">Portfolio</div>', unsafe_allow_html=True)
        try:
            ports = repo.list_portfolios()
            options = {row.portfolio_id: f"{row.name} ({row.portfolio_id})" for row in ports.itertuples()}
        except Exception:  # noqa: BLE001
            logger.exception("Could not list portfolios")
            options = {}
        if st.session_state.get("uploaded_portfolio"):
            options["UPLOAD"] = "Uploaded portfolio (session)"
        current = st.session_state.get("portfolio_source") or get_settings().portfolio.portfolio_id
        keys = list(options)
        if keys:
            choice = st.selectbox("Active portfolio", keys, index=keys.index(current) if current in keys else 0,
                                  format_func=options.get, label_visibility="collapsed")
            if choice != current:
                st.session_state["portfolio_source"] = choice
                st.rerun()
    return page


def status_panel(app: App | None, error: bool = False) -> None:
    lines = []
    if app is not None:
        for step in app.status.steps:
            lines.append((step.ok, step.label))
        lines.append((True, f"Market data loaded ({len(app.history.raw):,} obs)"))
        lines.append((True, f"Portfolio validated ({len(app.portfolio.positions)} positions)"))
        lines.append((not error, "Risk engine ready" if not error else "Risk engine error"))
    html = "".join(
        f'<div><span class="{"ok" if ok else "bad"}">{"✓" if ok else "✗"}</span> {label}</div>' for ok, label in lines
    )
    with st.sidebar:
        st.markdown('<div class="re-section">System status</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="re-status-line">{html}</div>', unsafe_allow_html=True)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        repo, status = initialise()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Initialisation failed")
        st.error(f"Risk engine could not start: {_friendly(exc)}")
        st.stop()
    if status.seeded_now and not st.session_state.get("seed_toast_shown"):
        st.session_state["seed_toast_shown"] = True
        st.toast("First launch: database created and sample data seeded.")

    page = sidebar(repo)
    try:
        app = build_app()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load portfolio / market data")
        status_panel(None, error=True)
        st.error(_friendly(exc))
        if st.session_state.get("portfolio_source") not in (None, get_settings().portfolio.portfolio_id):
            if st.button("Return to the sample portfolio"):
                st.session_state.pop("portfolio_source", None)
                st.session_state.pop("uploaded_portfolio", None)
                st.rerun()
        st.stop()

    error = False
    try:
        PAGES[page](app)
    except Exception as exc:  # noqa: BLE001
        error = True
        logger.exception("Page %s failed", page)
        st.error(_friendly(exc))
    status_panel(app, error)


main()
