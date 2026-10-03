"""Visual system: colour tokens, custom CSS and the Plotly template.

Categorical series use the validated dark palette (fixed order, never cycled;
checked against the #111a2b chart surface for CVD separation and contrast).
Status colours (gain/loss, within/breach) are reserved and never reused as a
series colour.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Surfaces & ink
BG = "#0b1220"
SURFACE = "#111a2b"
SURFACE_2 = "#16213a"
BORDER = "#1f2a3d"
GRID = "#1c2638"
AXIS = "#2b3850"
TEXT = "#e2e8f0"
TEXT_2 = "#a3aec2"
MUTED = "#6b7a93"

# Categorical order (validated) - identity only
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[0], SERIES[1], SERIES[2], SERIES[3]

# Status (reserved meaning)
GAIN = "#0ca30c"
LOSS = "#d03b3b"
WARN = "#fab219"
NEUTRAL = "#8a96ab"

# Sequential blue ramp (magnitude) and diverging blue <-> red with a gray midpoint
SEQUENTIAL = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
DIVERGING = [[0.0, "#d03b3b"], [0.25, "#a0514f"], [0.5, "#383f4d"], [0.75, "#2f6fbf"], [1.0, "#3987e5"]]

ASSET_CLASS_COLORS = {
    "Equity": SERIES[0],
    "Fixed Income": SERIES[1],
    "FX": SERIES[2],
    "Options": SERIES[3],
    "Cash": SERIES[6],
}

FONT = 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=12, color=TEXT_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES,
        margin=dict(l=8, r=8, t=36, b=8),
        title=dict(font=dict(size=13, color=TEXT), x=0.0, xanchor="left", y=0.98),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, showgrid=False, ticks="", title=dict(font=dict(size=11, color=MUTED))),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, zerolinewidth=1, showline=False, ticks="", title=dict(font=dict(size=11, color=MUTED))),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0.0, font=dict(size=11, color=TEXT_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(family=FONT, size=12, color=TEXT)),
        hovermode="closest",
        bargap=0.35,
    )
    t.data.bar = [go.Bar(marker=dict(cornerradius=4, line=dict(width=0)))]
    t.data.scatter = [go.Scatter(line=dict(width=2))]
    return t


pio.templates["risk_dark"] = _template()
pio.templates.default = "risk_dark"


CSS = f"""
<style>
:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface-2: {SURFACE_2}; --border: {BORDER};
  --text: {TEXT}; --text-2: {TEXT_2}; --muted: {MUTED};
  --gain: {GAIN}; --loss: {LOSS}; --warn: {WARN}; --accent: {BLUE};
}}
html, body, [class*="css"] {{ font-family: {FONT}; }}
.stApp {{ background: var(--bg); }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }}
h1, h2, h3, h4 {{ letter-spacing: 0.01em; color: var(--text); }}
h1 {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
h2 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 0.95rem !important; font-weight: 600 !important; }}
[data-testid="stSidebar"] {{ background: #0a1020; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] .block-container {{ padding-top: 1rem; }}
[data-testid="stHeader"] {{ background: transparent; }}

/* sidebar navigation */
.re-brand {{ font-size: 0.8rem; font-weight: 700; letter-spacing: 0.22em; color: var(--text); margin: 0.2rem 0 0.9rem 0; }}
.re-brand span {{ color: var(--accent); }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 5px 8px; border-radius: 4px; margin: 0; width: 100%;
  font-size: 0.83rem; color: var(--text-2); font-variant-numeric: tabular-nums;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--surface-2); color: var(--text); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {{ background: var(--surface-2); box-shadow: inset 2px 0 0 var(--accent); }}
[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {{ color: var(--text); font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child:not([data-testid]) {{ display: none; }}
[data-testid="stSidebar"] [data-testid="stRadioOption"] p {{ font-size: 0.83rem; color: var(--text-2); white-space: pre; }}

/* page header */
.re-header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; margin-bottom: 1rem; gap: 1rem; flex-wrap: wrap; }}
.re-title {{ font-size: 0.78rem; font-weight: 700; letter-spacing: 0.2em; color: var(--muted); text-transform: uppercase; }}
.re-page {{ font-size: 1.3rem; font-weight: 600; color: var(--text); margin-top: 0.15rem; }}
.re-meta {{ display: flex; gap: 1.4rem; font-size: 0.78rem; color: var(--text-2); flex-wrap: wrap; }}
.re-meta b {{ color: var(--text); font-weight: 600; }}
.re-health {{ color: var(--gain); font-weight: 600; }}
.re-health.bad {{ color: var(--loss); }}

/* KPI cards */
.re-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-bottom: 0.9rem; }}
.re-kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px; }}
.re-kpi .l {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }}
.re-kpi .v {{ font-size: 1.28rem; font-weight: 600; color: var(--text); margin-top: 3px; white-space: nowrap; }}
.re-kpi .s {{ font-size: 0.74rem; color: var(--text-2); margin-top: 2px; }}
.re-kpi .s.gain {{ color: var(--gain); }} .re-kpi .s.loss {{ color: #ef6b6b; }}

/* section labels */
.re-section {{ font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em; color: var(--muted); text-transform: uppercase; margin: 1.1rem 0 0.5rem 0; }}
.re-note {{ font-size: 0.78rem; color: var(--text-2); line-height: 1.5; }}

/* status pills */
.re-pill {{ display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: 0.72rem; font-weight: 600; border: 1px solid; }}
.re-pill.ok {{ color: var(--gain); border-color: rgba(12,163,12,.45); background: rgba(12,163,12,.08); }}
.re-pill.bad {{ color: #ef6b6b; border-color: rgba(208,59,59,.5); background: rgba(208,59,59,.1); }}
.re-pill.warn {{ color: var(--warn); border-color: rgba(250,178,25,.45); background: rgba(250,178,25,.08); }}
.re-pill.info {{ color: var(--text-2); border-color: var(--border); background: var(--surface-2); }}

/* tables rendered as HTML */
table.re-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; font-variant-numeric: tabular-nums; }}
table.re-table th {{ text-align: right; color: var(--muted); font-weight: 600; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
table.re-table td {{ text-align: right; padding: 6px 8px; border-bottom: 1px solid #172033; color: var(--text); }}
table.re-table th:first-child, table.re-table td:first-child {{ text-align: left; }}
table.re-table td.gain {{ color: var(--gain); }} table.re-table td.loss {{ color: #ef6b6b; }}

/* limit bars */
.re-limit {{ display: grid; grid-template-columns: 120px 1fr 110px 110px; gap: 10px; align-items: center; font-size: 0.8rem; padding: 6px 0; border-bottom: 1px solid #172033; font-variant-numeric: tabular-nums; }}
.re-limit .track {{ height: 6px; background: #1d2940; border-radius: 3px; overflow: hidden; }}
.re-limit .fill {{ height: 100%; border-radius: 3px; }}

/* scenario cards */
.re-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px; height: 100%; }}
.re-card .n {{ font-size: 0.78rem; font-weight: 600; color: var(--text); }}
.re-card .c {{ font-size: 0.66rem; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }}
.re-card .p {{ font-size: 1.15rem; font-weight: 600; margin-top: 4px; }}
.re-card .d {{ font-size: 0.72rem; color: var(--text-2); margin-top: 4px; line-height: 1.45; }}

.re-warning {{ border-left: 2px solid var(--warn); background: rgba(250,178,25,.06); padding: 6px 10px; margin: 4px 0; font-size: 0.8rem; color: var(--text); border-radius: 2px; }}
.re-status-line {{ font-size: 0.74rem; color: var(--text-2); line-height: 1.7; }}
.re-status-line .ok {{ color: var(--gain); }} .re-status-line .bad {{ color: var(--loss); }}

[data-testid="stMetric"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 8px 12px; }}
[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: 4px; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--border) !important; border-radius: 4px !important; background: var(--surface); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ font-size: 0.82rem; padding: 6px 12px; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 4px; font-size: 0.82rem; }}
</style>
"""
