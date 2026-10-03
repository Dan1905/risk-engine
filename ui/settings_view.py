"""12 Settings - session risk parameters, limit overrides, data and engine status."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy.engine import make_url

from services.eod_service import last_run
from ui import components as C
from ui.cache import App, clear_calculation_caches
from ui.risk_view import controls


def render(app: App) -> None:
    C.page_header("Settings", app.meta(), app.status.ok)
    s = app.settings

    C.section("Risk parameters (this session)")
    controls(app, key="set")
    c1, c2 = st.columns(2)
    rt = c1.radio("Return type for volatility / covariance", ["log", "simple"], index=0 if app.config().return_type == "log" else 1,
                  horizontal=True, key="set_rt")
    app.set_params(return_type=rt)
    C.note("Defaults come from <code>config/risk_config.yaml</code>; changes here apply to your session only.")

    C.section("Risk limits (this session)")
    base = s.limits
    with st.form("limits_form"):
        l1, l2, l3, l4, l5 = st.columns(5)
        var_l = l1.number_input("VaR limit (% NAV)", 0.1, 50.0, base.var_limit * 100, 0.1)
        es_l = l2.number_input("ES limit (% NAV)", 0.1, 50.0, base.es_limit * 100, 0.1)
        eq_l = l3.number_input("Equity exposure limit (% NAV)", 1.0, 300.0, base.equity_exposure_limit * 100, 1.0)
        pos_l = l4.number_input("Single position limit (% NAV)", 1.0, 100.0, base.single_position_limit * 100, 1.0)
        st_l = l5.number_input("Stress loss limit (% NAV)", 1.0, 100.0, base.stress_loss_limit * 100, 1.0)
        a, b = st.columns([1, 5])
        if a.form_submit_button("Apply limits", type="primary"):
            st.session_state["limit_overrides"] = {
                "var_limit": var_l / 100, "es_limit": es_l / 100, "equity_exposure_limit": eq_l / 100,
                "single_position_limit": pos_l / 100, "stress_loss_limit": st_l / 100,
            }
            st.rerun()
        if b.form_submit_button("Reset to configuration"):
            st.session_state.pop("limit_overrides", None)
            st.rerun()

    C.section("Stress settings")
    st_s = s.stress
    C.html_table(
        pd.DataFrame(
            [
                ("Equity shock ladder", ", ".join(f"{x:+.0%}" for x in st_s.equity_shock_ladder)),
                ("Rate shock ladder", ", ".join(f"{x:+.0f}bp" for x in st_s.rate_shock_ladder_bps)),
                ("Vol shock ladder", ", ".join(f"{x * 100:+.0f} pts" for x in st_s.vol_shock_ladder)),
                ("FX shocks", ", ".join(f"{f.currency} {f.shock:+.0%}" for f in st_s.fx_shocks)),
                ("Default vol method", st_s.default_vol_method.replace("_", " ").title()),
                ("Default bond method", st_s.default_bond_method.replace("_", " ").title()),
                ("Minimum shocked vol", f"{st_s.min_volatility:.0%}"),
            ],
            columns=["Setting", "Value"],
        )
    )

    C.section("Data & engine")
    url = make_url(s.database_url)
    safe_url = url.render_as_string(hide_password=True)
    counts = app.repo.table_counts()
    left, right = st.columns(2)
    with left:
        C.html_table(
            pd.DataFrame(
                [
                    ("Database", safe_url),
                    ("Dialect", url.get_backend_name()),
                    ("Seed version", app.repo.get_meta("seed_version") or "n/a"),
                    ("Seeded at", app.repo.get_meta("seeded_at") or "n/a"),
                    ("Market data version", app.version),
                    ("History", f"{app.history.dates[0]:%d %b %Y} – {app.history.dates[-1]:%d %b %Y} ({len(app.history.dates)} days)"),
                    ("Risk config", str(s.risk_config_path.name)),
                    ("Scenarios", str(s.scenarios_path.name)),
                    ("Log file", str(s.log_file)),
                    ("Log level", s.log_level),
                ],
                columns=["Item", "Value"],
            )
        )
    with right:
        C.html_table(pd.DataFrame(sorted(counts.items()), columns=["Table", "Rows"]), {"Rows": lambda v: f"{v:,}"})
    eod = last_run(app.repo)
    if eod:
        C.note(f"<b>Last end-of-day batch:</b> {eod['started_at']} · {eod['portfolio_id']} valuation {eod['valuation_date']} · "
               f"{eod['duration_seconds']}s · {len(eod['limit_breaches'])} limit breach(es). "
               "Run it with <code>python scripts/run_eod.py</code> (add <code>--schedule 18:00</code> to run daily).")
    else:
        C.note("No end-of-day batch has run yet. Run <code>python scripts/run_eod.py</code> to compute and persist the full "
               "risk process and write the reports (add <code>--schedule 18:00</code> to run daily).")
    for step in app.status.steps:
        st.markdown(f'<div class="re-status-line"><span class="{"ok" if step.ok else "bad"}">{"✓" if step.ok else "✗"}</span> '
                    f"{step.label} <span style='color:var(--muted)'>{step.detail}</span></div>", unsafe_allow_html=True)
    if st.button("Clear calculation caches", key="clear_caches"):
        clear_calculation_caches()
        st.toast("Caches cleared; results will be recomputed.")
        st.rerun()
