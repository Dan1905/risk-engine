"""08 Risk Contributions - Euler VaR decomposition, concentration, correlation."""

from __future__ import annotations

import html

import streamlit as st

from analytics.correlation import correlation_summary
from ui import components as C
from ui import styles as S
from ui.cache import App


def render(app: App) -> None:
    C.page_header("Risk Contributions", app.meta(), app.status.ok)
    cfg = app.config()
    c1, c2 = st.columns([1, 3])
    cov = c1.radio("Covariance", ["SAMPLE", "EWMA"], index=1 if cfg.covariance_method == "EWMA" else 0, horizontal=True,
                   format_func=lambda x: "Historical" if x == "SAMPLE" else "EWMA", key="rc_cov")
    app.set_params(covariance_method=cov)
    rep = app.var_report()
    df = app.risk.contributions(app.ctx, rep, cov)
    total = float(df["component_var"].sum())
    c2.markdown(
        f'<div class="re-note" style="padding-top:1.9rem">Parametric {cfg.confidence:.0%} {cfg.holding_period}D VaR '
        f'<b>{C.fmt_usd(total, decimals=2)}</b> = Σ Component VaR. Component VaR<sub>i</sub> = w<sub>i</sub> × Marginal VaR<sub>i</sub> × NAV; '
        "Marginal VaR = z·(Σw)<sub>i</sub>/σ<sub>p</sub> (VaR added per $1 of extra exposure). Negative contributions are hedges.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        C.dataframe(
            df[["position_id", "ticker", "asset_class", "currency", "exposure", "marginal_var", "component_var", "pct_contribution"]],
            {
                "position_id": "Position", "ticker": "Ticker", "asset_class": "Asset Class", "currency": "Ccy",
                "exposure": C.money_col("Exposure (USD)"), "marginal_var": C.num_col("Marginal VaR", "%.4f"),
                "component_var": C.money_col("Component VaR"), "pct_contribution": C.pct_col("% Total VaR"),
            },
            height=420,
        )
        C.csv_download(df.set_index("position_id"), "risk_contributions.csv", key="dl_rc")
    with right:
        s = df.set_index(df["ticker"] + " (" + df["position_id"] + ")")["component_var"]
        s = s[s.abs() > 1]
        fig = C.bar(s, "Component VaR by position")
        fig.update_traces(marker_color=[S.BLUE if v >= 0 else S.AQUA for v in s.sort_values().to_numpy()])
        C.show(fig, 440, key="rc_pos")

    left, right = st.columns(2)
    for col, key, title in ((left, "asset_class", "By asset class"), (right, "currency", "By currency")):
        with col:
            agg = app.risk.contributions_by(df, key)
            C.section(title)
            C.html_table(agg.reset_index().rename(columns={key: title.replace("By ", "").title(), "exposure": "Exposure",
                                                           "component_var": "Component VaR", "pct_contribution": "% Total VaR"}),
                         {"Exposure": lambda v: C.fmt_usd(v, decimals=2), "Component VaR": lambda v: C.fmt_usd(v, decimals=1),
                          "% Total VaR": lambda v: C.fmt_pct(v, 1)})

    C.section("Risk concentration")
    worst = app.stress.worst(app.stress_results())
    conc = app.risk.concentration(app.ctx, df, worst.position_pnl if worst else None, worst.scenario.name if worst else None)
    tick = app.ctx.valuation.set_index("position_id")["ticker"]
    items = [
        ("Largest Position", f"{tick[conc.largest_position[0]]}", f"{conc.largest_position[0]} · {C.fmt_pct(conc.largest_position[1], 1)} of NAV", ""),
        ("Largest Asset Class", conc.largest_asset_class[0], f"{C.fmt_pct(conc.largest_asset_class[1], 1)} of NAV", ""),
        ("Largest Currency", conc.largest_currency[0], f"{C.fmt_pct(conc.largest_currency[1], 1)} of gross", ""),
    ]
    if conc.largest_var_contributor:
        pid, sh = conc.largest_var_contributor
        items.append(("Largest VaR Contributor", tick[pid], f"{pid} · {C.fmt_pct(sh, 1)} of VaR", ""))
    if conc.largest_stress_contributor:
        pid, sh, sc = conc.largest_stress_contributor
        items.append(("Largest Stress Loss Contributor", tick[pid], f"{C.fmt_pct(sh, 1)} of loss · {sc}", ""))
    C.kpi_row(items)
    C.warnings_block(conc.warnings)
    th = app.settings.concentration
    C.note(f"Thresholds (configuration): VaR contribution {th.var_contribution_warning:.0%}, currency {th.currency_concentration_warning:.0%}, "
           f"asset class {th.asset_class_concentration_warning:.0%}, stress-loss contribution {th.stress_loss_contribution_warning:.0%}, "
           f"single position {app.settings.limits.single_position_limit:.0%}, equity {app.settings.limits.equity_exposure_limit:.0%}.")

    _correlation(app, rep, cov)


def _correlation(app: App, rep, cov: str) -> None:
    C.section("Correlation & covariance")
    val = app.ctx.valuation.set_index("position_id")
    corr = app.risk.correlation(rep, cov)
    groups = {
        "Equities": val.index[val["asset_type"] == "EQUITY"],
        "Bonds": val.index[val["asset_type"] == "BOND"],
        "FX": val.index[val["asset_type"] == "FX"],
        "Options & underlyings": val.index[(val["asset_type"] == "OPTION") | val["ticker"].isin(val.loc[val["asset_type"] == "OPTION", "ticker"]) & (val["asset_type"] == "EQUITY")],
        "All risky positions": val.index[val["asset_type"] != "CASH"],
    }
    pick = st.radio("Group", list(groups), horizontal=True, index=4, key="rc_group")
    ids = [i for i in groups[pick] if i in corr.index]
    if len(ids) < 2:
        C.note("Not enough positions in this group.")
        return
    labels = {i: f"{val.at[i, 'ticker']}{' opt' if val.at[i, 'asset_type'] == 'OPTION' else ''}" for i in ids}
    sub = corr.loc[ids, ids].rename(index=labels, columns=labels)
    summ = correlation_summary(sub)
    hp = f"{summ.highest_positive[0]} / {summ.highest_positive[1]}" if summ.highest_positive else "none"
    hn = f"{summ.highest_negative[0]} / {summ.highest_negative[1]}" if summ.highest_negative else "none"
    C.kpi_row(
        [
            ("Highest Positive Correlation", f"{summ.highest_positive[2]:.2f}" if summ.highest_positive else "n/a", hp, ""),
            ("Highest Negative Correlation", f"{summ.highest_negative[2]:.2f}" if summ.highest_negative else "n/a", hn, ""),
            ("Average Correlation", f"{summ.average:.2f}", f"{len(ids)} positions · {'EWMA' if cov == 'EWMA' else 'historical'}", ""),
        ]
    )
    C.show(C.heatmap(sub, f"Correlation of position returns ({'EWMA' if cov == 'EWMA' else 'historical'})", diverging=True, zmax=1.0),
           max(320, 34 * len(ids) + 120), key="rc_corr")
    c1, c2 = st.columns(2)
    with c1:
        C.csv_download(corr.rename(index=val["ticker"], columns=val["ticker"]), "correlation_matrix.csv", "Download correlation matrix", key="dl_corr")
    with c2:
        C.csv_download((rep.cov[cov] * 252).rename(index=val["ticker"], columns=val["ticker"]), f"covariance_{cov.lower()}_annualised.csv",
                       "Download covariance matrix", key="dl_cov_rc")
    C.note("Correlations are between the hypothetical daily returns of each current position under the same historical "
           f"factor moves used for VaR ({rep.config.lookback} days). A short option shows negative correlation to its long underlying. "
           + html.escape("Cash has zero variance and is excluded."))
