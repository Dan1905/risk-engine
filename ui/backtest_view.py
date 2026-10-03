"""09 VaR Backtesting - exceptions, Kupiec POF test, diagnostic charts."""

from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.backtesting import exceptions_by_month, rolling_exception_rate
from services.risk_service import METHOD_LABELS, METHODS
from ui import components as C
from ui import styles as S
from ui.cache import App


def render(app: App) -> None:
    C.page_header("VaR Backtesting", app.meta(), app.status.ok)
    r = app.settings.risk
    cfg = app.config()
    c1, c2, c3, c4 = st.columns(4)
    look = c1.selectbox("Lookback window", r.lookback_options, index=r.lookback_options.index(cfg.lookback) if cfg.lookback in r.lookback_options else 0, key="bt_look")
    conf = c2.selectbox("Confidence level", r.confidence_options, index=r.confidence_options.index(cfg.confidence) if cfg.confidence in r.confidence_options else 0,
                        format_func=lambda x: f"{x:.0%}", key="bt_conf")
    method = c3.selectbox("VaR model", METHODS, index=METHODS.index(app.settings.backtest.default_method), format_func=METHOD_LABELS.get, key="bt_method")
    lam = c4.selectbox("Decay factor λ (EW models)", r.decay_options, index=r.decay_options.index(cfg.decay_factor) if cfg.decay_factor in r.decay_options else 0, key="bt_lam")
    cov = cfg.covariance_method  # Monte Carlo factor covariance follows the session's covariance method
    try:
        fw = app.forecasts(look)
    except ValueError as exc:
        C.friendly_error(str(exc))
        return
    lo, hi = fw.dates[0].date(), fw.dates[-1].date()
    d1, d2 = st.columns(2)
    start = d1.date_input("Backtest start date", lo, min_value=lo, max_value=hi, key="bt_start")
    end = d2.date_input("Backtest end date", hi, min_value=lo, max_value=hi, key="bt_end")
    if start >= end:
        C.friendly_error("Backtest start date must be before the end date.")
        return
    try:
        res = app.backtest.run(app.ctx, fw, method, conf, start, end, lam, cov_method=cov)
    except ValueError as exc:
        C.friendly_error(str(exc))
        return
    k = res.kupiec
    frame = res.frame

    C.kpi_row(
        [
            ("Backtest Period", f"{frame.index[0]:%d %b %y} – {frame.index[-1]:%d %b %y}", f"{METHOD_LABELS[method]} · lookback {look}", ""),
            ("Observations", f"{k.n_observations:,}", "trading days", ""),
            ("VaR Exceptions", f"{k.n_exceptions}", f"expected {k.expected_exceptions:.1f}", "loss" if k.n_exceptions > k.expected_exceptions else ""),
            ("Exception Rate", C.fmt_pct(k.exception_rate, 2), "observed", ""),
            ("Expected Exception Rate", C.fmt_pct(k.expected_rate, 2), f"1 - {conf:.0%}", ""),
        ]
    )
    ch = res.christoffersen
    C.section("Statistical tests")

    def _row(test: str, null: str, lr: float, df: int, pv: float, reject: bool) -> str:
        crit = {1: 3.841, 2: 5.991}[df] if abs(k.significance_level - 0.05) < 1e-9 else float("nan")
        verdict = C.pill("H0 rejected", "bad") if reject else C.pill("H0 not rejected", "ok")
        return (f"<tr><td>{test}</td><td style='text-align:left'>{null}</td><td>{lr:.3f}</td><td>{df}</td>"
                f"<td>{crit:.3f}</td><td>{pv:.4f}</td><td>{verdict}</td></tr>")

    st.markdown(
        '<table class="re-table"><thead><tr><th>Test</th><th style="text-align:left">Null hypothesis (H0)</th><th>LR statistic</th>'
        f"<th>d.f.</th><th>χ² critical ({k.significance_level:.0%})</th><th>p-value</th><th>Result</th></tr></thead><tbody>"
        + _row("Kupiec POF (unconditional coverage)", f"exception probability = {k.expected_rate:.0%}", k.lr_statistic, 1, k.p_value, k.reject_null)
        + _row("Christoffersen independence", "exceptions are independent over time", ch.lr_independence, 1,
               ch.p_value_independence, ch.reject_independence)
        + _row("Conditional coverage (joint)", "correct frequency and independence", ch.lr_conditional_coverage, 2,
               ch.p_value_conditional_coverage, ch.reject_conditional_coverage)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )
    C.note(
        f"<b>Frequency:</b> {html.escape(k.message)}<br>"
        f"<b>Independence:</b> {html.escape(ch.independence_message)} "
        f"P(exception | exception yesterday) = {ch.pi_11:.1%} vs P(exception | none yesterday) = {ch.pi_01:.1%} "
        f"(transitions n00={ch.n00}, n01={ch.n01}, n10={ch.n10}, n11={ch.n11}).<br>"
        f"<b>Conditional coverage:</b> {html.escape(ch.conditional_coverage_message)}"
    )
    C.note("Kupiec: LR = -2 ln[(1-p)<sup>N-x</sup>p<sup>x</sup> / ((1-x/N)<sup>N-x</sup>(x/N)<sup>x</sup>)] tests the exception "
           "count only. Christoffersen compares a first-order Markov model of exceptions with an independent one, so it detects "
           "clustering. LR<sub>cc</sub> = LR<sub>POF</sub> + LR<sub>ind</sub> tests both at once.")

    C.section("Basel traffic light")
    try:
        basel = app.backtest.basel_assessment(fw, method, lam, end, app.ctx, cov)
    except ValueError as exc:
        basel = None
        C.note(str(exc))
    if basel is not None:
        kind = {"GREEN": "ok", "YELLOW": "warn", "RED": "bad"}[basel.zone]
        plus = "n/a (window is not 250 days)" if basel.plus_factor is None else f"+{basel.plus_factor:.2f}  →  multiplier {basel.multiplier:.2f}"
        st.markdown(
            f"""<div class="re-kpis"><div class="re-kpi"><div class="l">Zone</div>
            <div class="v" style="font-size:1rem">{C.pill(basel.zone.title() + " zone", kind)}</div>
            <div class="s">{METHOD_LABELS[method]} · 99% VaR</div></div>
            <div class="re-kpi"><div class="l">Exceptions (last {basel.n_observations} days)</div><div class="v">{basel.n_exceptions}</div>
            <div class="s">green ≤ {basel.green_max} · yellow ≤ {basel.yellow_max} · red above</div></div>
            <div class="re-kpi"><div class="l">Cumulative probability</div><div class="v">{basel.cumulative_probability:.2%}</div>
            <div class="s">P(X ≤ x) under a correct 99% model</div></div>
            <div class="re-kpi"><div class="l">Capital plus factor</div><div class="v" style="font-size:1rem">{html.escape(plus)}</div>
            <div class="s">added to the base multiplier of 3</div></div></div>""",
            unsafe_allow_html=True,
        )
        C.note(f"{html.escape(basel.message)} The Basel test always uses the 99% VaR of the selected model over the most recent "
               f"250 trading days up to the backtest end date, independent of the confidence level selected above.")

    C.show(C.pnl_with_var(frame["actual_pnl"], frame["var"], frame["exception"], "Actual daily P&L vs VaR threshold"), 360, key="bt_main")
    left, right = st.columns(2)
    with left:
        cum = frame["actual_pnl"].cumsum().to_frame("Cumulative P&L")
        C.show(C.lines(cum, "Cumulative P&L (static portfolio)", money=True), 280, key="bt_cum")
    with right:
        win = app.settings.backtest.rolling_window
        roll = rolling_exception_rate(frame["exception"], win).to_frame(f"Rolling {win}-day exception rate")
        fig = C.lines(roll, f"Rolling exception rate ({win} days)", pct=True)
        fig.add_hline(y=1 - conf, line=dict(color=S.MUTED, width=1), annotation_text=f"expected {1 - conf:.0%}",
                      annotation_font=dict(size=11, color=S.TEXT_2), annotation_position="top left")
        C.show(fig, 280, key="bt_roll")
    monthly = exceptions_by_month(frame["exception"])
    fig = go.Figure(go.Bar(x=monthly.index, y=monthly["exceptions"], marker_color=S.LOSS,
                           customdata=monthly["observations"], hovertemplate="%{x|%b %Y}<br>%{y} exceptions in %{customdata} days<extra></extra>"))
    fig.update_layout(title="Exception distribution by month", bargap=0.35)
    fig.update_yaxes(dtick=1, showgrid=True)
    C.show(fig, 250, key="bt_month")

    C.section("Exceptions")
    ex = res.exceptions_table
    if ex.empty:
        C.note("No VaR exceptions in the selected period.")
    else:
        C.dataframe(ex.reset_index(names="Date"), {"Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                                                   "Actual P&L": C.money_col("Actual P&L"), "Predicted VaR": C.money_col("Predicted VaR"),
                                                   "Excess Loss": C.money_col("Excess Loss")})
    C.csv_download(frame, f"backtest_{method.lower()}_{int(conf * 100)}.csv", "Download backtest results", key="dl_bt")

    C.section("Model comparison (same period)")
    rows = []
    for m in METHODS:
        rr = app.backtest.var_series(fw, m, conf, lam, app.ctx, cov)
        sub = app.backtest.run(app.ctx, fw, m, conf, start, end, lam, persist=False, cov_method=cov) if m != method else res
        kk, cc = sub.kupiec, sub.christoffersen
        try:
            zone = app.backtest.basel_assessment(fw, m, lam, end, app.ctx, cov).zone.title()
        except ValueError:
            zone = "n/a"
        rows.append({"Model": METHOD_LABELS[m], "Exceptions": kk.n_exceptions, "Rate": kk.exception_rate, "Kupiec p": kk.p_value,
                     "Independence p": cc.p_value_independence, "Cond. coverage p": cc.p_value_conditional_coverage,
                     "Basel zone (99%)": zone, "Avg VaR": float(rr.mean())})
    C.html_table(pd.DataFrame(rows), {"Rate": lambda v: C.fmt_pct(v, 2), "Kupiec p": lambda v: f"{v:.4f}",
                                      "Independence p": lambda v: f"{v:.4f}", "Cond. coverage p": lambda v: f"{v:.4f}",
                                      "Avg VaR": lambda v: C.fmt_usd(v)})
    C.note(f"p-values below {k.significance_level:.0%} reject the corresponding null hypothesis. Filtered Historical rescales each "
           "day's window to the volatility forecast for that day; Monte Carlo simulates "
           f"{app.settings.simulation.backtest_paths:,} paths per day from the window's "
           f"{'EWMA' if cov == 'EWMA' else 'sample'} factor covariance using common random numbers.")
    C.note("Look-ahead prevention: VaR for day t uses the factor moves ending on t-1 and the t-1 market state only; the realised "
           "P&L on t is the full revaluation of today's positions from the t-1 to the t close (plus coupons). Positions are held "
           "static (hypothetical P&L backtest).")
