"""Market-data layer: provider abstraction and point-in-time market history.

``MarketDataProvider`` is the plug-in point for data sources. The application
ships a database provider (reads the seeded SQL tables) and a CSV provider
(used for seeding and uploads). An external API (Bloomberg, Refinitiv,
Polygon, ...) only needs to implement the four ``load_*`` methods.

``MarketHistory`` turns long-format observations into wide, cleaned factor
frames and exposes:

* ``state_at(d)``            - the ``MarketState`` on date d
* ``historical_scenarios``   - daily factor moves in a lookback window

Look-ahead bias prevention: a history is always built with an ``end`` date and
never contains observations after it; ``historical_scenarios(end, n)`` only
returns moves dated on or before ``end``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analytics import risk_factors as rf
from analytics.beta import market_betas
from analytics.returns import clean_price_frame, simple_returns
from analytics.revaluation import MarketDataMissingError, MarketState, ScenarioSet
from analytics.var import InsufficientDataError
from analytics.volatility_surface import GridVolSurface, surface_factors
from analytics.yield_curve import TENOR_LABELS, YieldCurve, tenor_to_years
from database.repository import RiskRepository

logger = logging.getLogger(__name__)

GOVT_CURVE = rf.GOVT_CURVE


# ------------------------------------------------------------------------ providers


class MarketDataProvider(ABC):
    """Source of instruments, prices, curves and vol surfaces."""

    name: str = "abstract"

    @abstractmethod
    def load_instruments(self) -> pd.DataFrame: ...

    @abstractmethod
    def load_market_data(self, end: date | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def load_yield_curves(self, end: date | None = None) -> pd.DataFrame: ...

    @abstractmethod
    def load_vol_surfaces(self, end: date | None = None) -> pd.DataFrame: ...


class DatabaseMarketDataProvider(MarketDataProvider):
    name = "database"

    def __init__(self, repo: RiskRepository):
        self.repo = repo

    def load_instruments(self) -> pd.DataFrame:
        return self.repo.load_instruments()

    def load_market_data(self, end: date | None = None) -> pd.DataFrame:
        return self.repo.load_market_data(end=end)

    def load_yield_curves(self, end: date | None = None) -> pd.DataFrame:
        return self.repo.load_yield_curves(end=end)

    def load_vol_surfaces(self, end: date | None = None) -> pd.DataFrame:
        return self.repo.load_vol_surface(end=end)


class CsvMarketDataProvider(MarketDataProvider):
    """Reads the files written by ``scripts/generate_sample_data.py``."""

    name = "csv"

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def _read(self, name: str, end: date | None, date_col: str = "date") -> pd.DataFrame:
        df = pd.read_csv(self.dir / name)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            if end is not None:
                df = df[df[date_col] <= pd.Timestamp(end)]
        return df

    def load_instruments(self) -> pd.DataFrame:
        return pd.read_csv(self.dir / "instruments.csv")

    def load_market_data(self, end: date | None = None) -> pd.DataFrame:
        return self._read("market_data.csv", end)

    def load_yield_curves(self, end: date | None = None) -> pd.DataFrame:
        return self._read("yield_curves.csv", end)

    def load_vol_surfaces(self, end: date | None = None) -> pd.DataFrame:
        return self._read("vol_surface.csv", end)


# --------------------------------------------------------------------------- history


@dataclass
class _Surface:
    maturities: np.ndarray
    moneyness: np.ndarray
    grid: pd.DataFrame  # index date, columns (maturity, moneyness) flattened

    def at(self, d: pd.Timestamp) -> GridVolSurface | None:
        if d not in self.grid.index:
            return None
        vals = self.grid.loc[d].to_numpy(dtype=float).reshape(self.maturities.size, self.moneyness.size)
        if not np.all(np.isfinite(vals)):
            return None
        return GridVolSurface(self.maturities, self.moneyness, vals)


@dataclass
class MarketHistory:
    """Cleaned, wide market-factor history up to (and including) ``end``."""

    end: pd.Timestamp
    dates: pd.DatetimeIndex
    equity_prices: pd.DataFrame  # valuation prices (close), local currency
    equity_returns_px: pd.DataFrame  # prices used for returns (adjusted close)
    fx_quotes: pd.DataFrame  # quoted pair rates
    fx_usd: pd.DataFrame  # USD per unit of foreign currency
    curves: dict[str, pd.DataFrame]  # curve id -> dates x tenor labels
    surfaces: dict[str, _Surface]
    atm_vols: pd.DataFrame  # dates x underlyings (3M ATM level)
    vol_skew: pd.DataFrame  # dates x underlyings (3M 90%-110% skew)
    vol_term: pd.DataFrame  # dates x underlyings (1Y-1M ATM slope)
    raw: pd.DataFrame  # long market data (for display / quality stats)
    instruments: pd.DataFrame
    _state_cache: dict[pd.Timestamp, MarketState] = field(default_factory=dict, repr=False)
    _factor_frame: pd.DataFrame | None = field(default=None, repr=False)

    # --------------------------------------------------------------- building
    @classmethod
    def build(cls, provider: MarketDataProvider, end: date | None = None, max_ffill: int = 5) -> "MarketHistory":
        raw = provider.load_market_data(end=end)
        if raw.empty:
            raise MarketDataMissingError("No market data available for selected valuation date.")
        raw = raw.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        end_ts = pd.Timestamp(end) if end is not None else raw["date"].max()
        raw = raw[raw["date"] <= end_ts]
        instruments = provider.load_instruments()

        def wide(asset_type: str, col: str = "price") -> pd.DataFrame:
            sub = raw[raw["asset_type"] == asset_type]
            if sub.empty:
                return pd.DataFrame()
            frame = sub.pivot_table(index="date", columns="instrument_id", values=col, aggfunc="last")
            return clean_price_frame(frame, max_ffill=max_ffill)

        eq = wide("EQUITY")
        eq_adj = wide("EQUITY", "adj_close") if "adj_close" in raw.columns else eq
        if eq_adj.empty or eq_adj.isna().all().all():
            eq_adj = eq
        fxq = wide("FX")
        fx_usd = pd.DataFrame(index=fxq.index)
        for pair in fxq.columns:
            base, quote = pair[:3], pair[3:]
            if base == "USD":
                fx_usd[quote] = 1.0 / fxq[pair]
            elif quote == "USD":
                fx_usd[base] = fxq[pair]
        curves_long = provider.load_yield_curves(end=end_ts.date())
        curves_long["date"] = pd.to_datetime(curves_long["date"])
        curves: dict[str, pd.DataFrame] = {}
        for cid, grp in curves_long.groupby("curve_id"):
            w = grp.pivot_table(index="date", columns="tenor", values="rate", aggfunc="last")
            cols = sorted(w.columns, key=tenor_to_years)
            curves[str(cid)] = w[cols].sort_index().ffill(limit=max_ffill)
        if GOVT_CURVE not in curves:
            raise MarketDataMissingError("Yield curve data missing for selected valuation date.")

        surf_long = provider.load_vol_surfaces(end=end_ts.date())
        surfaces: dict[str, _Surface] = {}
        atm, skew, term = {}, {}, {}
        if not surf_long.empty:
            surf_long["date"] = pd.to_datetime(surf_long["date"])
            for und, grp in surf_long.groupby("underlying"):
                mats = np.sort(grp["maturity_years"].unique())
                mny = np.sort(grp["moneyness"].unique())
                g = grp.pivot_table(index="date", columns=["maturity_years", "moneyness"], values="implied_vol", aggfunc="last")
                g = g.reindex(columns=pd.MultiIndex.from_product([mats, mny])).sort_index()
                surf_hist = _Surface(mats, mny, g)
                surfaces[str(und)] = surf_hist
                # Level / skew / term factors of each day's surface (NaN where the surface is incomplete).
                rows = [surface_factors(sf) if (sf := surf_hist.at(d)) is not None else (np.nan,) * 3 for d in g.index]
                fac = pd.DataFrame(rows, index=g.index, columns=["level", "skew", "term"])
                atm[str(und)], skew[str(und)], term[str(und)] = fac["level"], fac["skew"], fac["term"]
        atm_vols, vol_skew, vol_term = pd.DataFrame(atm), pd.DataFrame(skew), pd.DataFrame(term)

        # Common calendar: dates on which equities, FX and the govt curve all exist.
        idx = eq.index
        for frame in (fxq, curves[GOVT_CURVE]):
            if not frame.empty:
                idx = idx.intersection(frame.index)
        idx = idx[idx <= end_ts]
        if len(idx) == 0:
            raise MarketDataMissingError("No market data available for selected valuation date.")
        hist = cls(
            end=end_ts,
            dates=pd.DatetimeIndex(idx),
            equity_prices=eq.reindex(idx),
            equity_returns_px=eq_adj.reindex(idx),
            fx_quotes=fxq.reindex(idx),
            fx_usd=fx_usd.reindex(idx),
            curves={k: v.reindex(idx).ffill(limit=max_ffill) for k, v in curves.items()},
            surfaces=surfaces,
            atm_vols=atm_vols.reindex(idx).ffill(limit=max_ffill),
            vol_skew=vol_skew.reindex(idx).ffill(limit=max_ffill),
            vol_term=vol_term.reindex(idx).ffill(limit=max_ffill),
            raw=raw,
            instruments=instruments,
        )
        logger.info("%d market observations loaded (%d dates, %s -> %s)", len(raw), len(idx), idx[0].date(), idx[-1].date())
        return hist

    # ----------------------------------------------------------------- access
    @property
    def last_date(self) -> date:
        return self.dates[-1].date()

    def has_date(self, d: date) -> bool:
        return pd.Timestamp(d) in self.dates

    def index_of(self, d: date) -> int:
        ts = pd.Timestamp(d)
        if ts not in self.dates:
            raise MarketDataMissingError(f"No market data available for selected valuation date ({ts:%d %b %Y}).")
        return int(self.dates.get_loc(ts))

    def known_tickers(self) -> set[str]:
        return set(self.equity_prices.columns)

    def known_fx_pairs(self) -> set[str]:
        return set(self.fx_quotes.columns)

    def state_at(self, d: date) -> MarketState:
        """Market state on ``d`` (must be a history date)."""
        ts = pd.Timestamp(d)
        if ts in self._state_cache:
            return self._state_cache[ts]
        self.index_of(d)
        eq_row = self.equity_prices.loc[ts].dropna()
        fx_row = self.fx_usd.loc[ts].dropna()
        curve_row = self.curves[GOVT_CURVE].loc[ts]
        if curve_row.isna().any():
            raise MarketDataMissingError(f"Yield curve data missing on {ts:%d %b %Y}.")
        curve = YieldCurve.from_series(curve_row)
        spreads = {}
        for cid, frame in self.curves.items():
            if cid == GOVT_CURVE:
                continue
            row = frame.loc[ts]
            if not row.isna().any():
                spreads[cid] = YieldCurve.from_series(row)
        surfaces = {}
        for und, s in self.surfaces.items():
            surf = s.at(ts)
            if surf is not None:
                surfaces[und] = surf
        atm_row = self.atm_vols.loc[ts].dropna() if not self.atm_vols.empty else pd.Series(dtype=float)
        state = MarketState(
            date=ts.date(),
            equity_prices={k: float(v) for k, v in eq_row.items()},
            fx_rates={"USD": 1.0, **{k: float(v) for k, v in fx_row.items()}},
            curve=curve,
            spread_curves=spreads,
            atm_vols={k: float(v) for k, v in atm_row.items()},
            vol_surfaces=surfaces,
        )
        self._state_cache[ts] = state
        return state

    # ------------------------------------------------------------ factor moves
    def factor_frame(self) -> pd.DataFrame:
        """Daily moves of every market factor over the full history.

        Row t holds the move from t-1 to t (the first history date has no
        move), with columns named as in :mod:`analytics.risk_factors`.
        """
        if self._factor_frame is None:
            parts = [
                simple_returns(self.equity_returns_px).rename(columns=lambda c: rf.factor(rf.EQ, c)),
                simple_returns(self.fx_usd).rename(columns=lambda c: rf.factor(rf.FX, c)),
            ]
            for cid, frame in self.curves.items():
                parts.append(frame.diff().iloc[1:].rename(columns=lambda c, cid=cid: rf.factor(rf.CURVE, cid, c)))
            for kind, levels in ((rf.VOL, self.atm_vols), (rf.VOLSKEW, self.vol_skew), (rf.VOLTERM, self.vol_term)):
                if not levels.empty:
                    parts.append(levels.diff().iloc[1:].rename(columns=lambda c, kind=kind: rf.factor(kind, c)))
            self._factor_frame = pd.concat(parts, axis=1).reindex(self.dates[1:]).fillna(0.0)
        return self._factor_frame

    def daily_moves(self, end_idx: int, n_rows: int) -> pd.DataFrame:
        """The ``n_rows`` daily factor moves ending at history index ``end_idx`` (inclusive)."""
        start = end_idx - n_rows + 1
        if start < 1:
            raise InsufficientDataError("Insufficient historical observations for the selected window.")
        frame = self.factor_frame().iloc[start - 1 : end_idx]  # frame row k-1 = move into history row k
        assert frame.index.max() <= self.end, "look-ahead: scenario after history end"
        return frame

    def historical_moves(self, end: date, lookback: int, confidence: float | None = None, extra_days: int = 0) -> pd.DataFrame:
        """Daily factor moves for a ``lookback`` window ending on ``end`` (plus ``extra_days`` of earlier history).

        Raises:
            InsufficientDataError: if fewer than ``lookback + extra_days`` moves precede ``end``.
        """
        i = self.index_of(end)
        needed = lookback + extra_days
        if i < needed:
            label = f" for {confidence:.0%} VaR" if confidence else ""
            raise InsufficientDataError(
                f"Insufficient historical observations{label}: {i} available before "
                f"{pd.Timestamp(end):%d %b %Y}, {needed} required."
            )
        return self.daily_moves(i, needed)

    def historical_scenarios(self, end: date, lookback: int, confidence: float | None = None) -> ScenarioSet:
        """The ``lookback`` daily factor moves ending on ``end`` (inclusive) as a ``ScenarioSet``."""
        return ScenarioSet.from_frame(self.historical_moves(end, lookback, confidence))

    def scenarios_between(self, start_idx: int, stop_idx: int) -> ScenarioSet:
        """Moves for history rows ``[start_idx, stop_idx)`` (row k = move from k-1 to k)."""
        return ScenarioSet.from_frame(self.daily_moves(stop_idx - 1, stop_idx - start_idx))

    def equity_betas(self, end: date, lookback: int = 250) -> dict[str, float]:
        """Betas of every equity to the equal-weighted equity market over ``lookback`` days ending on ``end``."""
        i = self.index_of(end)
        n = min(lookback, i)
        if n < 20:
            return {}
        cols = [c for c in self.factor_frame().columns if rf.kind_of(c) == rf.EQ]
        window = self.daily_moves(i, n)[cols].rename(columns=lambda c: c.split(rf.SEP, 1)[1])
        return {str(k): float(v) for k, v in market_betas(window).items()}

    # ---------------------------------------------------------- data quality
    def quality_report(self) -> dict[str, object]:
        raw = self.raw
        dupes = int(raw.duplicated(subset=["instrument_id", "date"]).sum())
        return {
            "observations": int(len(raw)),
            "missing_values": int(raw["price"].isna().sum()),
            "duplicate_rows": dupes,
            "date_range": (raw["date"].min().date(), raw["date"].max().date()),
            "instruments": int(raw["instrument_id"].nunique()),
            "trading_days": int(len(self.dates)),
        }

    def curve_labels(self) -> tuple[str, ...]:
        cols = tuple(self.curves[GOVT_CURVE].columns)
        return cols if cols else TENOR_LABELS


def load_market_history(repo: RiskRepository, end: date | None = None, max_ffill: int = 5) -> MarketHistory:
    """Convenience: build a history from the database provider."""
    return MarketHistory.build(DatabaseMarketDataProvider(repo), end=end, max_ffill=max_ffill)


# ------------------------------------------------------------------ presentation helpers

ASSET_LABELS = {"EQUITY": "Equity", "BOND": "Fixed Income", "FX": "FX", "OPTION": "Options", "CASH": "Cash"}


def instrument_summary(history: MarketHistory, stale_days: int = 3) -> pd.DataFrame:
    """Instrument | Asset Class | Last Price | Daily Return | Volatility | Currency | Data Status."""
    raw = history.raw.sort_values("date")
    rows = []
    names = dict(zip(history.instruments["instrument_id"], history.instruments["name"])) if not history.instruments.empty else {}
    for iid, grp in raw.groupby("instrument_id"):
        g = grp.dropna(subset=["price"])
        if g.empty:
            continue
        last, prev = g.iloc[-1], (g.iloc[-2] if len(g) > 1 else None)
        atype = str(last["asset_type"])
        rets = np.log(g["price"] / g["price"].shift(1)).dropna()
        realised = float(rets.tail(60).std(ddof=1) * np.sqrt(252)) if len(rets) > 5 else np.nan
        lag = int(np.busday_count(pd.Timestamp(last["date"]).date(), history.end.date()))
        missing = int(grp["price"].isna().sum())
        status = "Stale" if lag > stale_days else ("Gaps" if missing else "OK")
        rows.append(
            {
                "instrument": iid,
                "name": names.get(iid, iid),
                "asset_class": ASSET_LABELS.get(atype, atype),
                "last_price": float(last["price"]),
                "daily_return": float(last["price"] / prev["price"] - 1) if prev is not None else np.nan,
                "yield": float(last["yield"]) if atype == "BOND" and pd.notna(last.get("yield")) else np.nan,
                "implied_vol": float(last["volatility"]) if atype == "OPTION" and pd.notna(last.get("volatility")) else np.nan,
                "realised_vol_60d": realised,
                "currency": last["currency"],
                "last_date": pd.Timestamp(last["date"]).date(),
                "observations": int(len(g)),
                "data_status": status,
            }
        )
    order = {"Equity": 0, "Fixed Income": 1, "FX": 2, "Options": 3, "Cash": 4}
    df = pd.DataFrame(rows)
    return df.sort_values(["asset_class", "instrument"], key=lambda s: s.map(order) if s.name == "asset_class" else s).reset_index(drop=True)


@dataclass
class MarketDataUpload:
    frame: pd.DataFrame
    errors: list[str]
    stats: dict[str, object]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.frame.empty


def validate_market_data_upload(source, known_instruments: set[str], max_errors: int = 20) -> MarketDataUpload:
    """Parse and validate a normalised market-data CSV upload (Pydantic per row)."""
    from pydantic import ValidationError

    from models.market_data import MarketDataPoint

    try:
        df = pd.read_csv(source, index_col=False)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        logger.warning("Market data upload could not be parsed: %s", exc)
        return MarketDataUpload(pd.DataFrame(), ["File could not be parsed as CSV."], {})
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"date", "instrument_id", "asset_type", "price"}
    if missing := required - set(df.columns):
        return MarketDataUpload(pd.DataFrame(), [f"Missing required columns: {', '.join(sorted(missing))}"], {})
    stats = {
        "observations": int(len(df)),
        "missing_values": int(df[["date", "instrument_id", "price"]].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=["instrument_id", "date"]).sum()),
        "date_range": (str(pd.to_datetime(df["date"], errors="coerce").min().date()) if len(df) else "", str(pd.to_datetime(df["date"], errors="coerce").max().date()) if len(df) else ""),
        "instruments": int(df["instrument_id"].nunique()),
    }
    errors: list[str] = []
    valid_rows = []
    for i, rec in enumerate(df.to_dict(orient="records"), start=2):
        try:
            pt = MarketDataPoint.model_validate(rec)
        except ValidationError as exc:
            if len(errors) < max_errors:
                errors.append(f"Row {i}: {exc.errors()[0]['msg'].replace('Value error, ', '')}")
            continue
        if pt.instrument_id not in known_instruments:
            if len(errors) < max_errors:
                errors.append(f"Row {i}: unknown instrument {pt.instrument_id} (add it to the instrument master first).")
            continue
        valid_rows.append(pt.model_dump(mode="json", by_alias=True))
    if stats["duplicate_rows"]:
        errors.append(f"{stats['duplicate_rows']} duplicate instrument/date rows; the last occurrence will be kept if ingested.")
    frame = pd.DataFrame(valid_rows)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["instrument_id", "date"], keep="last")
    return MarketDataUpload(frame, errors, stats)
