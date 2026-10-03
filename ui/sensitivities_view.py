"""05 Sensitivities - option Greeks, bond risk measures, key rate duration, vol surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import components as C
from ui import styles as S
from ui.cache import App


def render(app: App) -> None:
    C.page_header("Sensitivities", app.meta(), app.status.ok)
    sens = app.sensitivities()
    tab_opt, tab_fi, tab_krd, tab_vol = st.tabs(["Options", "Fixed income", "Yield curve (KRD)", "Volatility surface"])
    with tab_opt:
        _options(sens)
    with tab_fi:
        _fixed_income(sens)
    with tab_krd:
        _krd(app, sens)
    with tab_vol:
        _surface(app)


def _options(sens) -> None:
    opt, tot = sens["options"], sens["option_totals"]
    if opt.empty:
        C.note("No option positions.")
        return
    C.kpi_row(
        [
            ("Portfolio Delta", C.fmt_usd(tot["delta_usd"]), "delta-equivalent USD", ""),
            ("Portfolio Gamma", C.fmt_usd(tot["gamma_usd"], decimals=2), "P&L for a 1% move (½ΓΔS²)", ""),
            ("Portfolio Vega", C.fmt_usd(tot["vega"], decimals=2), "per vol point", ""),
            ("Portfolio Theta", C.fmt_usd(tot["theta"], decimals=2), "per calendar day", "loss" if tot["theta"] < 0 else "gain"),
            ("Portfolio Rho", C.fmt_usd(tot["rho"], decimals=2), "per 1% rates", ""),
        ]
    )
    table = opt[["position_id", "underlying", "option_type", "quantity", "strike", "spot", "expiry", "implied_vol", "price", "market_value",
                 "delta", "delta_usd", "gamma", "vega", "theta", "rho"]]
    C.dataframe(
        table,
        {
            "position_id": "Position", "underlying": "Underlying", "option_type": "Type", "quantity": "Contracts",
            "strike": C.num_col("Strike", "%.2f"), "spot": C.num_col("Spot", "%.2f"), "expiry": "Expiry",
            "implied_vol": C.pct_col("Implied Vol"), "price": C.num_col("Price", "%.4f"), "market_value": C.money_col("Market Value"),
            "delta": C.num_col("Delta (sh)", "%.0f"), "delta_usd": C.money_col("Delta $"), "gamma": C.num_col("Gamma", "%.2f"),
            "vega": C.money_col("Vega /vol pt"), "theta": C.money_col("Theta /day"), "rho": C.money_col("Rho /1%"),
        },
    )
    greek = st.radio("Greek", ["delta_usd", "gamma_usd", "vega", "theta", "rho"], horizontal=True,
                     format_func={"delta_usd": "Delta $", "gamma_usd": "Gamma (1% move)", "vega": "Vega", "theta": "Theta", "rho": "Rho"}.get)
    s = opt.set_index(opt["underlying"] + " " + opt["option_type"].str[0] + opt["strike"].map("{:g}".format))[greek]
    C.show(C.signed_bar(s, "Position sensitivity (USD) · blue = long, green = short exposure", exposure=True), 220, key="opt_bar")
    grid = opt.set_index(opt["underlying"] + " " + opt["option_type"].str[0] + opt["strike"].map("{:g}".format))[["delta_usd", "gamma_usd", "vega", "theta", "rho"]]
    norm = grid / grid.abs().max().replace(0, 1)
    fig = C.heatmap(norm.rename(columns={"delta_usd": "Delta $", "gamma_usd": "Gamma", "vega": "Vega", "theta": "Theta", "rho": "Rho"}),
                    "Greek heatmap (scaled to largest |value| per Greek)", diverging=True, zmax=1.0)
    C.show(fig, 240, key="opt_heat")
    C.note("Black-Scholes assumptions: European exercise, constant volatility and risk-free rate (curve zero rate at expiry, "
           "continuously compounded), lognormal underlying, no arbitrage, continuous trading, no dividends.")


def _fixed_income(sens) -> None:
    b, tot = sens["bonds"], sens["bond_totals"]
    if b.empty:
        C.note("No bond positions.")
        return
    C.kpi_row(
        [
            ("Fixed-Income MV", C.fmt_usd(tot["market_value"], decimals=2), f"{len(b)} bonds", ""),
            ("Portfolio DV01", C.fmt_usd(tot["dv01"], compact=False), "USD per 1 bp", ""),
            ("Modified Duration", f"{tot['modified_duration']:.2f}", "MV-weighted", ""),
            ("Macaulay Duration", f"{tot['macaulay_duration']:.2f}", "years, MV-weighted", ""),
            ("Convexity", f"{tot['convexity']:.1f}", "MV-weighted", ""),
            ("Yield", C.fmt_pct(tot["ytm"], 3), "MV-weighted YTM", ""),
        ]
    )
    C.dataframe(
        b[["position_id", "ticker", "face", "coupon", "maturity", "price", "market_value", "ytm", "macaulay_duration", "modified_duration",
           "convexity", "dv01", "effective_duration", "spread_bps", "credit_curve"]],
        {
            "position_id": "Position", "ticker": "Bond", "face": st.column_config.NumberColumn("Face", format="localized"),
            "coupon": C.pct_col("Coupon"), "maturity": "Maturity", "price": C.num_col("Dirty Price", "%.3f"),
            "market_value": C.money_col("Market Value"), "ytm": C.pct_col("Yield"), "macaulay_duration": C.num_col("Duration", "%.2f"),
            "modified_duration": C.num_col("Mod. Duration", "%.2f"), "convexity": C.num_col("Convexity", "%.1f"),
            "dv01": C.money_col("DV01"), "effective_duration": C.num_col("Eff. Duration", "%.2f"),
            "spread_bps": C.num_col("Spread (bp)", "%.0f"), "credit_curve": "Curve",
        },
    )
    left, right = st.columns(2)
    with left:
        C.show(C.bar(b.set_index("ticker")["dv01"], "DV01 by bond (USD per bp)"), 260, key="fi_dv01")
    with right:
        C.show(C.bar(b.set_index("ticker")["modified_duration"], "Modified duration by bond", money=False), 260, key="fi_dur")
    C.note("Yield-based measures use the bond's YTM with its coupon frequency: D<sub>mod</sub> = D<sub>mac</sub>/(1+y/f), "
           "DV01 = D<sub>mod</sub> × P × 0.0001. Effective duration comes from ±1 bp parallel shifts of the zero curve "
           "(equal to the sum of key rate durations).")


def _krd(app: App, sens) -> None:
    krd, agg = sens["krd"], sens["krd_agg"]
    if krd.empty:
        C.note("No bond positions.")
        return
    names = app.ctx.valuation.set_index("position_id")["ticker"]
    left, right = st.columns([1, 1.2])
    with left:
        C.section("Portfolio key-rate risk")
        tbl = agg.reset_index(names="Tenor")
        C.html_table(tbl, {"KR01": lambda v: f"{v:,.0f}", "KRD": lambda v: f"{v:.3f}"})
        C.note("KR01 = USD gained for a 1 bp fall at that node (equivalently lost for a 1 bp rise). KRD = MV-weighted key rate "
               "duration of the bond book. KRD<sub>i</sub> = -(ΔP/P)/Δy<sub>i</sub> with only node i shocked (central difference).")
    with right:
        s = agg["KR01"]
        fig = go.Figure(go.Bar(x=list(s.index), y=s.to_numpy(), marker_color=S.BLUE, hovertemplate="%{x}<br>$%{y:,.0f} per bp<extra></extra>"))
        fig.update_layout(title="Key rate DV01 by tenor (USD gain per 1 bp fall)")
        fig.update_yaxes(tickprefix="$", tickformat="~s", showgrid=True)
        C.show(fig, 300, key="krd_bar")
    m = krd.rename(index=names)
    C.show(C.heatmap(m, "Key rate duration by bond and tenor", diverging=False, fmt=".2f"), 280, key="krd_heat")
    C.csv_download(krd.rename(index=names), "key_rate_durations.csv", key="dl_krd")
    curve = app.ctx.state.curve.to_series()
    hist = app.history.curves["USD_TSY"]
    ago = {"1M ago": 21, "1Y ago": 252}
    df = pd.DataFrame({"Today": curve})
    for lbl, lag in ago.items():
        if len(hist) > lag:
            df[lbl] = hist.iloc[-1 - lag].reindex(curve.index)
    fig = C.lines(df, "USD Treasury zero curve", pct=True)
    fig.update_traces(mode="lines+markers", marker=dict(size=8, line=dict(width=2, color=S.SURFACE)))
    C.show(fig, 280, key="krd_curve")


def _surface(app: App) -> None:
    surfaces = app.ctx.state.vol_surfaces
    if not surfaces:
        C.note("Volatility surface data unavailable.")
        return
    und = st.selectbox("Underlying", sorted(surfaces), key="surf_und")
    surf = surfaces[und]
    mats = getattr(surf, "maturities", np.array([1 / 12, 0.25, 0.5, 1.0]))
    mny = getattr(surf, "moneyness_grid", np.array([0.8, 0.9, 1.0, 1.1, 1.2]))
    labels = {round(1 / 12, 6): "1M", 0.25: "3M", 0.5: "6M", 1.0: "1Y"}
    grid = pd.DataFrame(surf.vol(mny[None, :], mats[:, None]), index=[labels.get(round(m, 6), f"{m:.2f}y") for m in mats],
                        columns=[f"{m:.0%}" for m in mny])
    left, right = st.columns(2)
    with left:
        C.show(C.heatmap(grid * 100, f"{und} implied vol surface (%)", diverging=False, fmt=".1f"), 280, key="surf_heat")
    with right:
        smile = pd.DataFrame({lbl: grid.loc[lbl].to_numpy() for lbl in grid.index}, index=[float(m) for m in mny])
        fig = C.lines(smile, "Volatility smile by maturity (x: moneyness K/S)", pct=True)
        fig.update_traces(mode="lines+markers", marker=dict(size=8, line=dict(width=2, color=S.SURFACE)))
        fig.update_xaxes(tickformat=".0%")
        C.show(fig, 280, key="surf_smile")
    atm = pd.DataFrame({"ATM vol": surf.vol(np.ones_like(mats), mats)}, index=[labels.get(round(m, 6), f"{m:.2f}y") for m in mats])
    atm_hist = app.history.atm_vols[[und]].rename(columns={und: f"{und} 3M ATM"}) if und in app.history.atm_vols else None
    left, right = st.columns(2)
    with left:
        fig = C.lines(atm, "ATM term structure", pct=True)
        fig.update_traces(mode="lines+markers", marker=dict(size=8, line=dict(width=2, color=S.SURFACE)))
        C.show(fig, 260, key="surf_ts")
    with right:
        if atm_hist is not None:
            C.show(C.lines(atm_hist, "3M ATM implied vol history", pct=True), 260, key="surf_hist")
    shape = pd.DataFrame({"Skew (90%-110%, 3M)": app.history.vol_skew.get(und), "Term slope (1Y-1M ATM)": app.history.vol_term.get(und)}).dropna()
    if not shape.empty:
        C.show(C.lines(shape, f"{und} surface shape factors (VaR risk factors alongside the 3M ATM level)", pct=True), 260, key="surf_shape")
    C.note("Surface σ = f(K/S, T): bilinear interpolation on the maturity × moneyness grid with flat extrapolation. "
           "Option positions are anchored to their input implied vol and move with the surface.")
