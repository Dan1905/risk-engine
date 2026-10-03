"""03 VaR & Expected Shortfall - four methodologies, EWMA vs historical, methodology notes."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.risk_service import METHOD_LABELS, METHODS
from ui import components as C
from ui import styles as S
from ui.cache import App

METHODOLOGY = {
    "Historical simulation VaR": (
        "Takes the last <i>lookback</i> daily market-factor moves (equity and FX returns, curve and spread node changes, "
        "implied-vol changes), applies each one to today's positions, fully revalues bonds and options, and sorts the "
        "resulting hypothetical P&L. VaR<sub>c</sub> = -Quantile(P&L, 1-c) with the inverted empirical CDF "
        "(k-th worst outcome, k = ceil(N(1-c)); with N = 250 at 99% that is the 3rd worst). "
        "Every scenario has equal weight, so a volatility spike moves VaR only when it becomes one of the worst k days, "
        "and it drops out abruptly after <i>lookback</i> days."
    ),
    "Exponentially weighted historical VaR": (
        "Same scenarios, but scenario t gets weight w<sub>t</sub> ∝ λ<sup>age</sup> (normalised to sum to 1). VaR is the "
        "loss where the cumulative weight of worse scenarios reaches 1-c. Because the last few weeks carry most of the "
        "weight, the estimate <b>reacts quickly when the volatility regime changes</b> - a calm year followed by a turbulent "
        "week lifts VaR immediately instead of waiting for the bad days to rank among the worst k. The trade-off is a "
        "small effective sample (roughly 1/(1-λ) ≈ 17 days at λ = 0.94), which makes 99% estimates noisy: a single "
        "recent observation can carry more than 1% of the weight, so VaR 95%, VaR 99% and ES can coincide."
    ),
    "Parametric (variance-covariance) VaR": (
        "Builds a return series for every position (r<sub>i</sub> = P&L<sub>i</sub>/MV<sub>i</sub>) and uses weights "
        "w<sub>i</sub> = MV<sub>i</sub>/NAV: σ<sub>p</sub> = sqrt(wᵀΣw), VaR = z<sub>c</sub> × σ<sub>p</sub> × NAV × sqrt(h) "
        "with z<sub>99</sub> ≈ 2.326. Assumes jointly normal, zero-mean returns, so it understates fat tails. "
        "<b>Parametric</b> uses the equally weighted sample covariance; <b>EW Parametric</b> uses the latest EWMA covariance."
    ),
    "Filtered historical simulation": (
        "Plain historical simulation replays each past move at the volatility of its own day. Filtered historical simulation "
        "first rescales every factor move to today's volatility: move*<sub>t</sub> = move<sub>t</sub> × σ<sub>T+1</sub>/σ<sub>t</sub>, "
        "where σ<sub>t</sub> is the EWMA forecast for day t (made with data up to t-1) and σ<sub>T+1</sub> the current forecast. "
        "The scenarios keep their empirical shape - fat tails and how factors move together - but their scale tracks the current "
        "regime. VaR and ES are then read off the equally weighted distribution, so the full window is used (unlike EW "
        "historical, whose effective sample is ~17 days). Scaling ratios are bounded to avoid one unusual day dominating."
    ),
    "Monte Carlo VaR": (
        "Simulates factor moves from a zero-mean multivariate distribution with the factor covariance of the lookback window "
        "(sample or EWMA, following the covariance method), then <b>fully revalues</b> every path - so option and bond convexity "
        "are captured, which the linear variance-covariance method ignores. Draws are multivariate Student-t (joint fat tails; "
        "all factors share one mixing variable) or normal, generated from an eigen-decomposition of the covariance so singular "
        "matrices are handled. Antithetic variates reduce sampling noise and a fixed seed makes results reproducible."
    ),
    "Implied-volatility risk factors": (
        "Each optionable underlying contributes three volatility factors taken from its stored surface history: "
        "<b>level</b> (3M at-the-money vol), <b>skew</b> (3M vol at 90% moneyness minus vol at 110%) and <b>term structure</b> "
        "(1Y minus 1M at-the-money vol). In a scenario the vol of an option with moneyness m and maturity τ moves by "
        "Δlevel + Δskew × (1-m)/0.2 + Δterm × (τ-0.25)/0.917. With the level factor alone, every strike and maturity is "
        "assumed to move in parallel; adding skew and term lets out-of-the-money and longer- or shorter-dated options move "
        "differently from the 3M at-the-money point, as they do in the data."
    ),
    "EWMA volatility and covariance": (
        "RiskMetrics recursion σ²<sub>t</sub> = λσ²<sub>t-1</sub> + (1-λ)r²<sub>t-1</sub> and "
        "Cov<sub>t</sub>(i,j) = λCov<sub>t-1</sub>(i,j) + (1-λ)r<sub>t-1,i</sub>r<sub>t-1,j</sub>, seeded with the window's "
        "sample covariance, zero mean. The latest EWMA covariance is the forecast for the next day."
    ),
    "Expected Shortfall": (
        "<b>VaR:</b> loss threshold exceeded in approximately (1-c) of observations. <b>Expected Shortfall:</b> average loss "
        "conditional on entering that worst (1-c) tail - ES<sub>c</sub> = mean of losses ≥ VaR<sub>c</sub> (weighted mean for "
        "the EW method; σφ(z)/(1-c) for parametric). ES looks inside the tail, so it distinguishes a thin tail from a fat one "
        "that VaR would report identically, and it is sub-additive."
    ),
    "Holding period": (
        "<b>Overlapping scenarios (default):</b> an h-day VaR uses genuine h-day factor moves - the last lookback + h - 1 daily "
        "moves are combined into overlapping h-day windows (returns compounded, curve / spread / vol changes summed) and every "
        "position is fully revalued at the multi-day move, so option and bond convexity and trending or mean-reverting markets "
        "are reflected. Adjacent windows share h-1 days, so the scenarios are serially dependent and the effective sample is "
        "smaller than the count suggests. The two exponentially weighted estimators are defined on daily data and keep their "
        "1-day estimate × sqrt(h); Filtered Historical and Monte Carlo are the methods that carry current-volatility "
        "information into a true multi-day scenario. <b>Square-root-of-time:</b> VaR<sub>h</sub> = VaR<sub>1</sub> × sqrt(h), which assumes "
        "i.i.d. daily P&L and a linear book. Both treat shocks as instantaneous (no time decay or carry over the horizon). "
        "Backtesting always uses 1-day VaR."
    ),
}


def controls(app: App, key: str = "var") -> None:
    cfg = app.config()
    r = app.settings.risk
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.4])
    conf = c1.selectbox("Confidence", r.confidence_options, index=r.confidence_options.index(cfg.confidence) if cfg.confidence in r.confidence_options else 0,
                        format_func=lambda x: f"{x:.0%}", key=f"{key}_conf")
    look = c2.selectbox("Lookback (days)", r.lookback_options, index=r.lookback_options.index(cfg.lookback) if cfg.lookback in r.lookback_options else 0, key=f"{key}_look")
    hp = c3.selectbox("Holding period (days)", r.holding_period_options,
                      index=r.holding_period_options.index(cfg.holding_period) if cfg.holding_period in r.holding_period_options else 0, key=f"{key}_hp")
    lam = c4.selectbox("Decay factor λ", r.decay_options, index=r.decay_options.index(cfg.decay_factor) if cfg.decay_factor in r.decay_options else 0,
                       format_func=lambda x: f"{x:.2f}", key=f"{key}_lam")
    cov = c5.radio("Covariance method", ["SAMPLE", "EWMA"], index=1 if cfg.covariance_method == "EWMA" else 0, horizontal=True,
                   format_func=lambda x: "Historical" if x == "SAMPLE" else "EWMA", key=f"{key}_cov")
    horizon = cfg.horizon_method
    if hp > 1:
        horizon = st.radio(
            f"{hp}-day VaR method", ["OVERLAPPING", "SQRT_TIME"], index=0 if cfg.horizon_method == "OVERLAPPING" else 1, horizontal=True,
            format_func=lambda x: (f"Overlapping {hp}-day scenarios (full revaluation at the multi-day move)" if x == "OVERLAPPING"
                                   else f"Square-root-of-time (1-day figures × √{hp})"),
            key=f"{key}_horizon",
        )
    vf = st.radio(
        "Implied-volatility risk factors", ["LEVEL_SKEW_TERM", "LEVEL"], index=0 if cfg.vol_factors == "LEVEL_SKEW_TERM" else 1, horizontal=True,
        format_func=lambda x: "Level + skew + term structure" if x == "LEVEL_SKEW_TERM" else "ATM level only", key=f"{key}_vf",
    )
    app.set_params(confidence=conf, lookback=look, holding_period=hp, decay_factor=lam, covariance_method=cov, horizon_method=horizon,
                   vol_factors=vf)


def render(app: App) -> None:
    C.page_header("VaR & Expected Shortfall", app.meta(), app.status.ok)
    controls(app)
    cfg = app.config()
    rep = app.var_report()
    nav = rep.nav
    c = cfg.confidence
    hist = rep.get("HISTORICAL", c)
    cov_key = "EW_PARAMETRIC" if cfg.covariance_method == "EWMA" else "PARAMETRIC"
    par = rep.get(cov_key, c)
    h = f"{cfg.holding_period}D"

    C.kpi_row(
        [
            (f"Historical VaR {c:.0%} ({h})", C.fmt_usd(hist.var), f"{C.fmt_pct(hist.var_pct)} of NAV", ""),
            (f"ES {c:.0%} ({h})", C.fmt_usd(hist.expected_shortfall), f"{C.fmt_pct(hist.es_pct)} of NAV", ""),
            ("ES / VaR", f"{hist.es_var_ratio:.2f}", "tail heaviness (normal ≈ 1.15 at 99%)", ""),
            ("Tail Loss (worst)", C.fmt_usd(-hist.worst_loss), f"of {hist.n_observations} scenarios, 1-day", "loss"),
            ("Best Gain", C.fmt_usd(hist.best_gain), "1-day", "gain"),
            ("Average Daily P&L", C.fmt_usd(hist.mean_pnl), f"{METHOD_LABELS[cov_key]} VaR {C.fmt_usd(par.var)}", ""),
        ]
    )

    basis = "" if cfg.holding_period == 1 else (" · overlapping scenarios" if rep.overlapping else " · √t scaling")
    C.section(f"Methodology comparison · {h}{basis} · lookback {cfg.lookback} · λ = {cfg.decay_factor:.2f}")
    table = rep.comparison_table((0.95, 0.99), 0.99)
    if c not in (0.95, 0.99):
        table = rep.comparison_table((0.95, 0.99, c), 0.99)
    display = table.copy()
    for col in display.columns:
        display[col] = [f"{C.fmt_usd(v, decimals=2)}  ({v / nav * 100:.2f}%)" for v in table[col]]
    display.insert(0, "Measure", display.index)
    C.html_table(display.reset_index(drop=True))
    C.csv_download(table, "var_comparison.csv", key="dl_varcmp")
    if rep.overlapping:
        C.note(f"Historical, Parametric, Filtered Historical and Monte Carlo use overlapping {cfg.holding_period}-day scenarios. "
               + " and ".join(METHOD_LABELS[m] for m in rep.sqrt_scaled)
               + f" are daily estimators (λ = {cfg.decay_factor:.2f} puts almost all weight on the last ~{1 / (1 - cfg.decay_factor):.0f} days, "
               f"too few independent {cfg.holding_period}-day periods), so they show their 1-day estimate × √{cfg.holding_period}.")

    left, right = st.columns([1.3, 1])
    with left:
        which = st.radio("Scenario distribution", list(rep.distributions), horizontal=True, format_func=METHOD_LABELS.get, key="var_dist_pick")
        pnl = rep.distributions[which]
        picked = rep.get(which, c)
        unscale = rep.scale_days ** 0.5  # chart shows scenario-horizon P&L; table values may be sqrt(time)-scaled
        markers = {
            f"VaR {c:.0%}": (-picked.var / unscale, S.ORANGE),
            f"ES {c:.0%}": (-picked.expected_shortfall / unscale, S.LOSS),
        }
        if which == "HISTORICAL":
            markers[f"EW VaR {c:.0%}"] = (-rep.get("EW_HISTORICAL", c).var / unscale, S.YELLOW)
        C.show(C.distribution(pnl, markers, f"{METHOD_LABELS[which]}: {rep.scenario_days}-day scenario P&L ({len(pnl):,} scenarios)"),
               340, key="var_dist")
    with right:
        rows = []
        for m in METHODS:
            r = rep.get(m, c)
            rows.append({"Method": METHOD_LABELS[m], "VaR": r.var, "ES": r.expected_shortfall, "ES/VaR": r.es_var_ratio,
                         "σ 1D (USD)": r.volatility})
        C.section(f"{c:.0%} detail")
        C.html_table(pd.DataFrame(rows), {"VaR": lambda v: C.fmt_usd(v), "ES": lambda v: C.fmt_usd(v),
                                          "ES/VaR": lambda v: f"{v:.2f}", "σ 1D (USD)": lambda v: C.fmt_usd(v)})
        C.note(f"Distribution shows {rep.scenario_days}-day scenario P&L; table values are for the selected holding period.")

    C.section(f"Stand-alone 1-day VaR {c:.0%} by risk-factor group")
    groups = app.risk.factor_group_var(app.ctx, cfg)
    left, right = st.columns([1.2, 1])
    with left:
        C.html_table(
            groups.rename(columns={"group": "Risk-factor group", "factors": "Factors", "var": "VaR", "expected_shortfall": "ES",
                                   "share_of_standalone": "% of stand-alone sum"}),
            {"VaR": lambda v: C.fmt_usd(v), "ES": lambda v: C.fmt_usd(v), "% of stand-alone sum": lambda v: C.fmt_pct(v, 1)},
        )
    with right:
        C.show(C.bar(groups.set_index("group")["var"].iloc[:-1], "Stand-alone historical VaR by factor group"), 260, key="var_groups")
    C.note("Each row replays only that group's historical moves with every other factor unchanged. The diversified total is "
           "below the sum of the rows because the groups do not move together. Implied-vol skew = change in the 3M 90%-110% "
           "vol spread; term structure = change in the 1Y-1M ATM slope; an option's vol moves by level + skew × (1-m)/0.2 + "
           "term × (τ-0.25)/0.917.")

    roll = app.rolling(methods=METHODS)
    left, right = st.columns(2)
    with left:
        df = roll[[f"var_{m}" for m in METHODS]].rename(columns=lambda x: METHOD_LABELS[x.replace("var_", "")])
        C.show(C.lines(df, f"Rolling 1D VaR {c:.0%} (out-of-sample)", money=True), 300, key="var_roll")
    with right:
        vols = roll[[x for x in roll.columns if x.startswith("vol_")]].rename(
            columns=lambda x: "EWMA" if x == "vol_ewma" else x.replace("vol_", "").replace("d", "-day historical"))
        C.show(C.lines(vols, "Rolling portfolio volatility (annualised)", pct=True), 300, key="var_vol")

    C.section("EWMA vs historical volatility & covariance")
    volcmp = app.risk.volatility_comparison(rep)
    names = app.ctx.valuation.set_index("position_id")["ticker"]
    option_ids = {p.position_id for p in app.portfolio.positions if p.asset_type.value == "OPTION"}
    volcmp = volcmp[volcmp["historical_vol"] > 0]
    linear = [f"{names[i]} ({i})" for i in volcmp.index if i not in option_ids]
    volcmp.index = [f"{names[i]} ({i})" for i in volcmp.index]
    left, right = st.columns([1, 1.2])
    with left:
        C.dataframe(volcmp.reset_index(names="Position"),
                    {"historical_vol": C.pct_col("Historical vol"), "ewma_vol": C.pct_col("EWMA vol"), "difference": C.pct_col("EWMA - Hist")}, height=360)
    with right:
        tbl = volcmp.loc[linear].sort_values("historical_vol")
        fig = go.Figure()
        fig.add_bar(y=tbl.index, x=tbl["historical_vol"], orientation="h", name="Historical", marker_color=S.BLUE)
        fig.add_bar(y=tbl.index, x=tbl["ewma_vol"], orientation="h", name="EWMA", marker_color=S.ORANGE)
        fig.update_layout(title="Annualised position volatility", barmode="group", bargap=0.3, bargroupgap=0.1)
        fig.update_xaxes(tickformat=".0%", showgrid=True)
        C.show(fig, 360, key="var_volbars")
        C.note("Chart excludes option positions, whose own returns are leveraged (see table for their volatility).")

    tabs = st.tabs(["EWMA covariance", "Historical covariance", "Difference (EWMA - Historical)"])
    labels = {pid: names[pid] + ("" if app.portfolio.get(pid).asset_type.value != "OPTION" else " opt") for pid in rep.returns.columns}
    ann = 252 / rep.scenario_days  # covariances are estimated on scenario-horizon returns
    mats = {
        "EWMA": rep.cov["EWMA"] * ann,
        "SAMPLE": rep.cov["SAMPLE"] * ann,
    }
    mats["DIFF"] = mats["EWMA"] - mats["SAMPLE"]
    keep = [p for p in rep.returns.columns if mats["SAMPLE"].loc[p, p] > 0 and app.portfolio.get(p).asset_type.value != "OPTION"]
    for tab, k in zip(tabs, ("EWMA", "SAMPLE", "DIFF")):
        with tab:
            m = mats[k].loc[keep, keep].rename(index=labels, columns=labels)
            C.show(C.heatmap(m * 1e4, "Annualised covariance (x 10⁻⁴, excl. options for scale)", diverging=True, fmt=".1f"), 520, key=f"cov_{k}")
            C.csv_download(mats[k].rename(index=labels, columns=labels), f"covariance_{k.lower()}.csv", key=f"dl_cov_{k}")

    C.section("Methodology")
    for title, text in METHODOLOGY.items():
        with st.expander(title):
            C.note(text)
