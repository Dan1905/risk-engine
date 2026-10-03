"""Reusable UI pieces: formatting, KPI cards, tables and Plotly chart builders.

Only presentation lives here - every number shown is computed by the services.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import styles as S

# --------------------------------------------------------------------- formatting


def fmt_usd(x: float | None, compact: bool = True, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if compact:
        for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if a >= div:
                return f"{sign}${a / div:,.{decimals}f}{unit}"
    return f"{sign}${a:,.0f}"


def fmt_pct(x: float | None, decimals: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x * 100:+.{decimals}f}%" if signed else f"{x * 100:.{decimals}f}%"


def fmt_num(x: float | None, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.{decimals}f}"


def sign_class(x: float) -> str:
    return "gain" if x > 0 else "loss" if x < 0 else ""


ASSET_LABELS = {"EQUITY": "Equity", "BOND": "Fixed Income", "FX": "FX", "OPTION": "Options", "CASH": "Cash"}


def asset_label(code: str | None) -> str:
    return ASSET_LABELS.get(str(code), str(code))


# ------------------------------------------------------------------------ layout


def page_header(title: str, ctx_meta: dict[str, str] | None = None, healthy: bool = True) -> None:
    meta = ctx_meta or {}
    items = "".join(f"<div>{html.escape(k)} <b>{html.escape(v)}</b></div>" for k, v in meta.items())
    health = (
        '<div class="re-health">● Risk Engine Healthy</div>' if healthy else '<div class="re-health bad">● Risk Engine Degraded</div>'
    )
    st.markdown(
        f"""<div class="re-header"><div><div class="re-title">Multi-Asset Risk Engine</div>
        <div class="re-page">{html.escape(title)}</div></div>
        <div class="re-meta">{items}{health}</div></div>""",
        unsafe_allow_html=True,
    )


def section(label: str) -> None:
    st.markdown(f'<div class="re-section">{html.escape(label)}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="re-note">{text}</div>', unsafe_allow_html=True)


def kpi_row(items: Sequence[tuple[str, str, str | None, str]]) -> None:
    """items: (label, value, sub-text, sub-class) - sub-class in {'', 'gain', 'loss'}."""
    cards = []
    for label, value, sub, cls in items:
        sub_html = f'<div class="s {cls}">{html.escape(sub)}</div>' if sub else ""
        cards.append(f'<div class="re-kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>{sub_html}</div>')
    st.markdown(f'<div class="re-kpis">{"".join(cards)}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="re-pill {kind}">{html.escape(text)}</span>'


def warnings_block(messages: Iterable[str]) -> None:
    msgs = list(messages)
    if not msgs:
        st.markdown(f'<div class="re-note">{pill("No concentration warnings", "ok")}</div>', unsafe_allow_html=True)
        return
    st.markdown("".join(f'<div class="re-warning">▲ {html.escape(m)}</div>' for m in msgs), unsafe_allow_html=True)


def limit_bars(checks) -> None:
    """Render limit utilisation rows: name | bar | value / limit | status."""
    rows = []
    for c in checks:
        util = min(c.utilisation, 1.5)
        color = S.LOSS if c.status == "Breach" else (S.WARN if util > 0.8 else S.GAIN)
        width = min(util / 1.5 * 100, 100)
        marker = 100 / 1.5
        status = pill(c.status, "bad" if c.status == "Breach" else "ok")
        rows.append(
            f"""<div class="re-limit"><div>{html.escape(c.name)}</div>
            <div class="track" style="position:relative"><div class="fill" style="width:{width:.1f}%;background:{color}"></div>
            <div style="position:absolute;left:{marker:.1f}%;top:-3px;width:1px;height:12px;background:{S.TEXT_2}"></div></div>
            <div>{c.value * 100:.1f}% / {c.limit * 100:.1f}%</div><div>{status}</div></div>"""
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def html_table(df: pd.DataFrame, formats: dict[str, Any] | None = None, signed: Sequence[str] = ()) -> None:
    """Compact read-only HTML table with per-column formatters and gain/loss colouring."""
    formats = formats or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            f = formats.get(c)
            text = f(v) if callable(f) else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            cls = sign_class(float(v)) if c in signed and isinstance(v, (int, float, np.floating)) else ""
            cells.append(f'<td class="{cls}">{html.escape(text)}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(f'<table class="re-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>', unsafe_allow_html=True)


def money_col(label: str) -> Any:
    return st.column_config.NumberColumn(label, format="accounting")


def pct_col(label: str) -> Any:
    return st.column_config.NumberColumn(label, format="percent")


def num_col(label: str, fmt: str = "%.4f") -> Any:
    return st.column_config.NumberColumn(label, format=fmt)


def dataframe(df: pd.DataFrame, column_config: dict[str, Any] | None = None, height: int | None = None) -> None:
    kwargs: dict[str, Any] = {"hide_index": True, "width": "stretch", "column_config": column_config or {}}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def csv_download(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=True).encode(), file_name=filename, mime="text/csv", key=key or f"dl_{filename}")


# ------------------------------------------------------------------------ charts


def _cap_bar_thickness(fig: go.Figure, height: int) -> None:
    """Keep bars <= ~22px thick and show every category label on bar charts."""
    for tr in fig.data:
        if tr.type != "bar" or fig.layout.barmode in ("relative", "stack"):
            continue
        horizontal = tr.orientation == "h"
        cats = tr.y if horizontal else tr.x
        if cats is None or len(cats) == 0 or not isinstance(cats[0], str):
            continue
        n = len(cats)
        plot = max(height - 90, 60) if horizontal else 700
        fig.update_layout(bargap=float(np.clip(1 - 22 * n / plot, 0.25, 0.85)))
        if horizontal:
            fig.update_yaxes(dtick=1)
        break


def show(fig: go.Figure, height: int = 320, key: str | None = None) -> None:
    fig.update_layout(height=height)
    _cap_bar_thickness(fig, height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "displaylogo": False}, key=key)


def _money_axis(fig: go.Figure, axis: str = "y") -> None:
    fig.update_layout(**{f"{axis}axis": dict(tickprefix="$", tickformat="~s", showgrid=True)})


def signed_bar(series: pd.Series, title: str = "", horizontal: bool = True, hover_fmt: str = "$%{x:,.0f}",
               exposure: bool = False) -> go.Figure:
    """Bars sorted by value. P&L bars use gain/loss colours; ``exposure=True`` uses neutral long/short hues."""
    s = series.sort_values(ascending=horizontal)
    pos, neg = (S.BLUE, S.AQUA) if exposure else (S.GAIN, S.LOSS)
    colors = [pos if v >= 0 else neg for v in s.to_numpy()]
    if horizontal:
        fig = go.Figure(go.Bar(x=s.to_numpy(), y=[str(i) for i in s.index], orientation="h", marker_color=colors,
                               hovertemplate=f"%{{y}}<br>{hover_fmt}<extra></extra>"))
        _money_axis(fig, "x")
        fig.update_xaxes(showgrid=True)
    else:
        fig = go.Figure(go.Bar(x=[str(i) for i in s.index], y=s.to_numpy(), marker_color=colors,
                               hovertemplate=f"%{{x}}<br>{hover_fmt.replace('x:', 'y:')}<extra></extra>"))
        _money_axis(fig, "y")
    fig.update_layout(title=title, showlegend=False, bargap=0.45)
    return fig


def bar(series: pd.Series, title: str = "", color: str = S.BLUE, horizontal: bool = True, money: bool = True,
        pct: bool = False, sort: bool = True) -> go.Figure:
    s = series.sort_values(ascending=horizontal) if sort else series
    fmt = "$%{v:,.0f}" if money else ("%{v:.2%}" if pct else "%{v:,.3f}")
    if horizontal:
        fig = go.Figure(go.Bar(x=s.to_numpy(), y=[str(i) for i in s.index], orientation="h", marker_color=color,
                               hovertemplate="%{y}<br>" + fmt.replace("v:", "x:") + "<extra></extra>"))
        fig.update_xaxes(showgrid=True)
        if money:
            _money_axis(fig, "x")
        if pct:
            fig.update_xaxes(tickformat=".0%")
    else:
        fig = go.Figure(go.Bar(x=[str(i) for i in s.index], y=s.to_numpy(), marker_color=color,
                               hovertemplate="%{x}<br>" + fmt.replace("v:", "y:") + "<extra></extra>"))
        fig.update_yaxes(showgrid=True)
        if money:
            _money_axis(fig, "y")
        if pct:
            fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title=title, showlegend=False, bargap=0.45)
    return fig


def lines(df: pd.DataFrame, title: str = "", colors: Sequence[str] | None = None, money: bool = False, pct: bool = False,
          y_title: str = "", dash: dict[str, str] | None = None) -> go.Figure:
    fig = go.Figure()
    colors = list(colors or S.SERIES)
    fmt = "$%{y:,.0f}" if money else ("%{y:.2%}" if pct else "%{y:,.4f}")
    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines",
                                 line=dict(color=colors[i % len(colors)], width=2, dash=(dash or {}).get(col, "solid")),
                                 hovertemplate=f"{col}: {fmt}<extra></extra>"))
    fig.update_layout(title=title, hovermode="x unified", showlegend=len(df.columns) > 1)
    fig.update_yaxes(showgrid=True, title=y_title)
    if money:
        _money_axis(fig, "y")
    if pct:
        fig.update_yaxes(tickformat=".1%")
    return fig


def pnl_with_var(pnl: pd.Series, var: pd.Series, exceptions: pd.Series | None = None, title: str = "") -> go.Figure:
    """Daily P&L bars against the negative VaR threshold; exceptions highlighted."""
    exc = exceptions.reindex(pnl.index).fillna(False).astype(bool) if exceptions is not None else pd.Series(False, index=pnl.index)
    colors = np.where(exc, S.LOSS, np.where(pnl >= 0, "#3a5a86", "#56657d"))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pnl.index, y=pnl, name="Actual P&L", marker=dict(color=colors, cornerradius=0),
                         hovertemplate="%{x|%d %b %Y}<br>P&L $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=var.index, y=-var, name="-VaR threshold", mode="lines", line=dict(color=S.ORANGE, width=2),
                             hovertemplate="-VaR $%{y:,.0f}<extra></extra>"))
    if exc.any():
        fig.add_trace(go.Scatter(x=pnl.index[exc], y=pnl[exc], mode="markers", name="VaR exception",
                                 marker=dict(color=S.LOSS, size=9, symbol="x-thin", line=dict(width=2, color=S.LOSS)),
                                 hovertemplate="%{x|%d %b %Y}<br>Exception: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=title, bargap=0.1, hovermode="x unified")
    _money_axis(fig, "y")
    return fig


def distribution(pnl: pd.Series, markers: dict[str, tuple[float, str]], title: str = "") -> go.Figure:
    """Histogram of scenario P&L with vertical threshold lines (label -> (x, color))."""
    fig = go.Figure(go.Histogram(x=pnl, nbinsx=60, marker=dict(color="#2c4a73", line=dict(width=1, color=S.SURFACE)),
                                 hovertemplate="P&L %{x}<br>%{y} scenarios<extra></extra>", name="Scenario P&L", showlegend=False))
    ymax = float(np.histogram(pnl, bins=60)[0].max())
    for label, (x, color) in markers.items():
        # Threshold as a vertical line trace so it gets a legend entry and a hover value.
        fig.add_trace(go.Scatter(x=[x, x], y=[0, ymax * 1.05], mode="lines", line=dict(color=color, width=2),
                                 name=f"{label}: {fmt_usd(x)}", hovertemplate=f"{label}<br>{fmt_usd(x)}<extra></extra>"))
    fig.update_layout(title=title, showlegend=True, bargap=0.02)
    fig.update_xaxes(tickprefix="$", tickformat="~s")  # the chart title names the quantity; an axis title would collide with the legend
    fig.update_yaxes(title="Scenarios", showgrid=True)
    return fig


def heatmap(matrix: pd.DataFrame, title: str = "", diverging: bool = True, zmin: float | None = None, zmax: float | None = None,
            fmt: str = ".2f", show_values: bool = True) -> go.Figure:
    z = matrix.to_numpy(dtype=float)
    if diverging:
        bound = zmax if zmax is not None else float(np.nanmax(np.abs(z))) or 1.0
        zmin, zmax, scale = -bound, bound, S.DIVERGING
    else:
        # Dark surface: near-zero recedes toward the background (dark), magnitude gets lighter.
        scale = [[i / (len(S.SEQUENTIAL) - 1), c] for i, c in enumerate(S.SEQUENTIAL)]
    text = np.vectorize(lambda v: format(v, fmt))(z) if show_values else None
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in matrix.columns], y=[str(i) for i in matrix.index], colorscale=scale,
                               zmin=zmin, zmax=zmax, xgap=2, ygap=2, text=text, texttemplate="%{text}" if show_values else None,
                               textfont=dict(size=10), hovertemplate="%{y} / %{x}: %{z:" + fmt + "}<extra></extra>",
                               colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10))))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")
    return fig


def waterfall(labels: list[str], values: list[float], measures: list[str], title: str = "") -> go.Figure:
    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures, connector=dict(line=dict(color=S.AXIS, width=1)),
        increasing=dict(marker=dict(color=S.GAIN)), decreasing=dict(marker=dict(color=S.LOSS)),
        totals=dict(marker=dict(color=S.BLUE)), hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(title=title, showlegend=False)
    _money_axis(fig, "y")
    return fig


def stacked_bars(df: pd.DataFrame, title: str = "", colors: dict[str, str] | None = None, line: pd.Series | None = None,
                 line_name: str = "Actual") -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        c = (colors or {}).get(col, S.SERIES[i % len(S.SERIES)])
        fig.add_trace(go.Bar(x=df.index, y=df[col], name=str(col), marker=dict(color=c, cornerradius=0),
                             hovertemplate=f"{col}: $%{{y:,.0f}}<extra></extra>"))
    if line is not None:
        fig.add_trace(go.Scatter(x=line.index, y=line, name=line_name, mode="lines+markers",
                                 line=dict(color=S.TEXT, width=2), marker=dict(size=5, color=S.TEXT),
                                 hovertemplate=f"{line_name}: $%{{y:,.0f}}<extra></extra>"))
    fig.update_layout(title=title, barmode="relative", hovermode="x unified", bargap=0.25)
    _money_axis(fig, "y")
    return fig


def friendly_error(message: str, exc: Exception | None = None) -> None:
    """User-facing error; technical detail is logged by the caller."""
    st.error(message, icon=":material/error:")


def time_bars(series: pd.Series, title: str = "") -> go.Figure:
    """Chronological bars coloured by sign (daily P&L)."""
    fig = go.Figure(go.Bar(x=series.index, y=series, marker=dict(color=[S.GAIN if v >= 0 else S.LOSS for v in series], cornerradius=0),
                           hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=title, bargap=0.15, showlegend=False)
    _money_axis(fig, "y")
    return fig
