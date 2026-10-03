"""10 Market Data - instrument status, data quality, curves, CSV upload."""

from __future__ import annotations

import logging
from datetime import datetime

import streamlit as st

from services.market_data_service import instrument_summary, validate_market_data_upload
from ui import components as C
from ui.cache import App, clear_calculation_caches

logger = logging.getLogger(__name__)


def render(app: App) -> None:
    C.page_header("Market Data", app.meta(), app.status.ok)
    hist = app.history
    q = hist.quality_report()
    lo, hi = q["date_range"]
    C.kpi_row(
        [
            ("Observations", f"{q['observations']:,}", "instrument-days", ""),
            ("Missing Values", f"{q['missing_values']:,}", "price fields", "loss" if q["missing_values"] else ""),
            ("Duplicate Rows", f"{q['duplicate_rows']:,}", "instrument / date", "loss" if q["duplicate_rows"] else ""),
            ("Date Range", f"{lo:%d %b %y} – {hi:%d %b %y}", f"{q['trading_days']} trading days", ""),
            ("Number of Instruments", f"{q['instruments']}", "plus 2 curves, 3 vol surfaces", ""),
        ]
    )
    summary = instrument_summary(hist)
    display = summary.copy()
    for col in ("yield", "implied_vol"):  # sparse by asset class: show a dash rather than "None"
        display[col] = [f"{v * 100:.2f}%" if v == v else "–" for v in summary[col]]
    C.dataframe(
        display[["instrument", "name", "asset_class", "last_price", "daily_return", "yield", "implied_vol", "realised_vol_60d", "currency", "last_date", "data_status"]],
        {
            "instrument": "Instrument", "name": "Description", "asset_class": "Asset Class", "last_price": C.num_col("Last Price", "%.4f"),
            "daily_return": C.pct_col("Daily Return"), "yield": "Yield", "implied_vol": "Implied Vol",
            "realised_vol_60d": C.pct_col("Volatility (60d realised)"), "currency": "Ccy", "last_date": "Last Obs",
            "data_status": "Data Status",
        },
    )
    C.csv_download(summary.set_index("instrument"), "market_data_summary.csv", key="dl_md_summary")

    left, right = st.columns(2)
    with left:
        inst = st.selectbox("Instrument history", summary["instrument"].tolist(), key="md_inst")
        raw = hist.raw[hist.raw["instrument_id"] == inst].set_index("date")["price"].to_frame(inst)
        C.show(C.lines(raw, f"{inst} price", money=False), 300, key="md_price")
    with right:
        curves = hist.curves["USD_TSY"]
        tenors = st.multiselect("Treasury tenors", list(curves.columns), default=["3M", "2Y", "10Y", "30Y"], key="md_tenors")
        if tenors:
            C.show(C.lines(curves[tenors], "USD Treasury zero rates", pct=True), 300, key="md_curves")
    spread = hist.curves.get("USD_IG")
    if spread is not None:
        C.show(C.lines(spread[["2Y", "5Y", "10Y"]], "USD IG credit spread curve", pct=True), 240, key="md_spread")

    C.section("Upload market data (CSV)")
    C.note("Normalised schema: <code>date, instrument_id, asset_type, price, yield, volatility, currency</code> "
           "(optional <code>open, high, low, adj_close, volume, duration, convexity</code>). Rows are validated with the "
           "MarketDataPoint schema; only instruments already in the instrument master are accepted. Ingestion upserts on "
           "(instrument, date), so re-uploading the same file is idempotent.")
    file = st.file_uploader("Market data CSV", type=["csv"], key="md_upload")
    if file is None:
        return
    known = set(hist.instruments["instrument_id"])
    up = validate_market_data_upload(file, known)
    if up.stats:
        s = up.stats
        C.kpi_row(
            [
                ("Observations", f"{s['observations']:,}", "rows in file", ""),
                ("Missing Values", f"{s['missing_values']:,}", "date / id / price", ""),
                ("Duplicate Rows", f"{s['duplicate_rows']:,}", "instrument / date", ""),
                ("Date Range", f"{s['date_range'][0]} – {s['date_range'][1]}", "", ""),
                ("Number of Instruments", f"{s['instruments']}", "", ""),
            ]
        )
    if up.errors:
        st.warning("\n".join(up.errors))
    if up.frame.empty:
        C.friendly_error("No valid rows to ingest.")
        return
    st.success(f"{len(up.frame):,} valid rows ready to ingest.")
    C.dataframe(up.frame.head(50))
    if st.button("Ingest into database", type="primary", key="md_ingest"):
        try:
            app.repo.upsert_market_data(up.frame.rename(columns={"yield_": "yield"}), update=True)
            app.repo.set_meta("market_data_version", datetime.now().isoformat(timespec="seconds"))
        except Exception:  # noqa: BLE001
            logger.exception("Market data ingest failed")
            C.friendly_error("Market data could not be written to the database. See the log for details.")
            return
        clear_calculation_caches()
        st.toast("Market data ingested; calculations will refresh.")
        st.rerun()
