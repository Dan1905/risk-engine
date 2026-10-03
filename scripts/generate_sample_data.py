"""Generate deterministic synthetic market data and the sample portfolio.

Outputs (CSV, written to ``data/sample`` by default):

    instruments.csv       instrument master with bond / option / FX terms
    market_data.csv       normalised daily observations (date, instrument_id, asset_type,
                          price, yield, volatility, currency + OHLCV / duration / convexity)
    yield_curves.csv      USD_TSY zero curve and USD_IG spread curve, 10 tenors
    vol_surface.csv       daily implied-vol surfaces (1M-1Y x 80%-120% moneyness)
    sample_portfolio.csv  the "Global Macro Fund" in the portfolio upload format

Data-generating process (all shocks are Student-t(5) scaled to unit variance):

* A GARCH(1,1)-type volatility regime ``g_t`` driven by the market factor
  produces volatility clustering; a handful of injected synthetic stress days
  create sharp regime changes for the backtest to react to.
* Equity returns load on market and technology factors plus idiosyncratic
  noise, which gives realistic sector correlations.
* FX (USD per unit of foreign currency) loads on the market and a USD factor.
* The Treasury zero curve follows a mean-reverting Nelson-Siegel
  level / slope / curvature model around a path from an inverted 2023 curve to
  a normal 2026 curve; the IG spread curve is lognormal and widens in risk-off.
* 3M ATM implied vols rise with the regime and fall with the underlying
  (leverage effect); surfaces add a term structure and an equity skew/smile
  whose slope and term tilt vary over time (steeper skew and an inverted term
  structure in stressed regimes), so skew and term structure are genuine risk
  factors.

Everything is seeded (``--seed``), so identical inputs produce identical files.

Usage::

    python scripts/generate_sample_data.py [--seed 42] [--days 756] [--end 2026-09-30] [--out data/sample]
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.bonds import BondSchedule, convexity, modified_duration, yield_to_maturity  # noqa: E402
from analytics.options import black_scholes_price  # noqa: E402
from analytics.volatility_surface import GridVolSurface  # noqa: E402
from analytics.yield_curve import TENOR_LABELS, TENOR_YEARS, interpolation_weights  # noqa: E402

logger = logging.getLogger(__name__)

# Bump when the generated data changes; a seeded database with an older version is refreshed in place.
DATASET_VERSION = "2"
VERSION_FILE = ".dataset_version"

# --------------------------------------------------------------------------- universe


@dataclass(frozen=True)
class EquitySpec:
    ticker: str
    name: str
    sector: str
    currency: str
    end_price: float
    vol: float
    beta_mkt: float
    beta_tech: float
    drift: float
    base_iv: float | None = None  # optionable underlyings get a vol surface


EQUITIES: tuple[EquitySpec, ...] = (
    EquitySpec("AAPL", "Apple Inc.", "Technology", "USD", 245.0, 0.26, 0.65, 0.35, 0.12, 0.25),
    EquitySpec("MSFT", "Microsoft Corp.", "Technology", "USD", 520.0, 0.24, 0.65, 0.35, 0.14, 0.23),
    EquitySpec("NVDA", "NVIDIA Corp.", "Technology", "USD", 185.0, 0.48, 0.60, 0.45, 0.35, 0.45),
    EquitySpec("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "USD", 228.0, 0.30, 0.60, 0.35, 0.12),
    EquitySpec("JPM", "JPMorgan Chase & Co.", "Financials", "USD", 312.0, 0.24, 0.65, 0.00, 0.15),
    EquitySpec("XOM", "Exxon Mobil Corp.", "Energy", "USD", 114.0, 0.25, 0.45, -0.10, 0.04),
    EquitySpec("JNJ", "Johnson & Johnson", "Health Care", "USD", 178.0, 0.16, 0.40, -0.10, 0.05),
    EquitySpec("SAP", "SAP SE", "Technology", "EUR", 238.0, 0.25, 0.55, 0.25, 0.10),
)


@dataclass(frozen=True)
class FxSpec:
    pair: str
    currency: str  # foreign currency
    end_quote: float
    vol: float
    beta_mkt: float
    beta_usd: float


FX: tuple[FxSpec, ...] = (
    FxSpec("EURUSD", "EUR", 1.17, 0.075, 0.15, -0.70),
    FxSpec("GBPUSD", "GBP", 1.34, 0.080, 0.20, -0.62),
    FxSpec("USDINR", "INR", 88.5, 0.045, 0.15, -0.35),
)


@dataclass(frozen=True)
class BondSpec:
    ticker: str
    name: str
    coupon: float
    maturity: date
    credit_curve: str | None = None
    frequency: int = 2


BONDS: tuple[BondSpec, ...] = (
    BondSpec("US2Y", "US Treasury 3.625% 30-Sep-2028", 0.03625, date(2028, 9, 30)),
    BondSpec("US5Y", "US Treasury 3.750% 30-Sep-2031", 0.0375, date(2031, 9, 30)),
    BondSpec("US10Y", "US Treasury 4.000% 15-Aug-2036", 0.04, date(2036, 8, 15)),
    BondSpec("US30Y", "US Treasury 4.625% 15-Aug-2056", 0.04625, date(2056, 8, 15)),
    BondSpec("CORP7Y", "IG Corporate 4.850% 15-Sep-2033", 0.0485, date(2033, 9, 15), "USD_IG"),
)


@dataclass(frozen=True)
class OptionSpec:
    underlying: str
    strike: float
    expiry: date
    option_type: str

    @property
    def instrument_id(self) -> str:
        return f"{self.underlying}_{self.option_type[0]}{self.strike:g}_{self.expiry:%Y%m%d}"


OPTIONS: tuple[OptionSpec, ...] = (
    OptionSpec("AAPL", 260.0, date(2027, 3, 19), "CALL"),
    OptionSpec("MSFT", 490.0, date(2026, 12, 18), "PUT"),
    OptionSpec("NVDA", 210.0, date(2027, 1, 15), "CALL"),
)

# (position_id, asset_type, instrument reference, quantity)
PORTFOLIO: tuple[tuple[str, str, str, float], ...] = (
    ("POS001", "EQUITY", "AAPL", 8_200),
    ("POS002", "EQUITY", "MSFT", 3_400),
    ("POS003", "EQUITY", "NVDA", 8_100),
    ("POS004", "EQUITY", "AMZN", 6_600),
    ("POS005", "EQUITY", "JPM", 4_000),
    ("POS006", "EQUITY", "XOM", 8_800),
    ("POS007", "EQUITY", "JNJ", 7_000),
    ("POS008", "EQUITY", "SAP", 4_500),
    ("POS009", "BOND", "US2Y", 2_000_000),
    ("POS010", "BOND", "US5Y", 2_500_000),
    ("POS011", "BOND", "US10Y", 2_250_000),
    ("POS012", "BOND", "US30Y", 1_000_000),
    ("POS013", "BOND", "CORP7Y", 1_500_000),
    ("POS014", "FX", "EURUSD", 1_000_000),
    ("POS015", "FX", "GBPUSD", 600_000),
    ("POS016", "FX", "USDINR", 90_000_000),
    ("POS017", "OPTION", "AAPL_C260_20270319", 150),
    ("POS018", "OPTION", "MSFT_P490_20261218", 80),
    ("POS019", "OPTION", "NVDA_C210_20270115", -120),
    ("POS020", "CASH", "USD", 1_200_000),
)

START_CURVE = np.array([5.50, 5.55, 5.40, 4.95, 4.65, 4.35, 4.30, 4.20, 4.45, 4.30]) / 100
END_CURVE = np.array([3.95, 3.85, 3.70, 3.55, 3.55, 3.70, 3.90, 4.10, 4.65, 4.70]) / 100
IG_SPREAD = np.array([0.45, 0.50, 0.55, 0.65, 0.72, 0.85, 0.95, 1.05, 1.25, 1.30]) / 100

SURFACE_MATURITIES = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0}
MONEYNESS = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

# Synthetic stress days: market-factor shock (in standard deviations).
# Each episode is a sell-off followed by a partial rebound.
STRESS_DAYS: dict[str, float] = {
    "2024-08-02": -2.0,
    "2024-08-05": -3.5,
    "2024-08-06": 1.5,
    "2024-08-07": 1.0,
    "2025-04-03": -4.0,
    "2025-04-04": -4.0,
    "2025-04-07": 0.5,
    "2025-04-08": -1.5,
    "2025-04-09": 5.5,
    "2025-04-10": -2.0,
    "2025-04-11": 1.5,
    "2026-06-15": -3.0,
    "2026-06-16": -1.5,
    "2026-06-17": 1.5,
}


@dataclass
class SampleDataset:
    instruments: pd.DataFrame
    market_data: pd.DataFrame
    yield_curves: pd.DataFrame
    vol_surface: pd.DataFrame
    portfolio: pd.DataFrame


# ----------------------------------------------------------------------------- helpers


def _t(rng: np.random.Generator, size, df: int = 5) -> np.ndarray:
    """Student-t shocks scaled to unit variance."""
    return rng.standard_t(df, size) / np.sqrt(df / (df - 2))


def _ar1(shocks: np.ndarray, phi: float) -> np.ndarray:
    """Mean-zero AR(1) path ``x_t = phi x_{t-1} + shock_t`` starting at 0."""
    x = np.zeros(len(shocks))
    for t in range(1, len(shocks)):
        x[t] = phi * x[t - 1] + shocks[t]
    return x


def _ramp(n: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, n)
    return x * x * (3 - 2 * x)  # smoothstep


def _ns_loadings(tau: np.ndarray, lam: float = 2.0) -> np.ndarray:
    """Nelson-Siegel level / slope / curvature loadings (n_tenors x 3)."""
    x = tau / lam
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return np.column_stack([np.ones_like(tau), slope, curv])


def _vol_grid(iv3m: np.ndarray, base_iv: float, skew_mult: np.ndarray | None = None, term_tilt: np.ndarray | None = None) -> np.ndarray:
    """Surface (dates x maturities x moneyness) from the 3M ATM vol series.

    Args:
        iv3m: 3M at-the-money implied vol per date (the surface's level factor).
        base_iv: long-run vol the term structure mean-reverts to.
        skew_mult: per-date multiplier on the skew slope (1 = base skew); makes the
            smile steepen and flatten over time.
        term_tilt: per-date tilt of the ATM term structure, in vol per year of
            maturity relative to the 3M point.

    The 3M ATM point always equals ``iv3m`` exactly: the tilt is zero at 3M and
    the skew has no effect at 100% moneyness.
    """
    n = len(iv3m)
    skew_mult = np.ones(n) if skew_mult is None else skew_mult
    term_tilt = np.zeros(n) if term_tilt is None else term_tilt
    mats = np.array(list(SURFACE_MATURITIES.values()))
    atm = base_iv + (iv3m[:, None] - base_iv) * np.exp(-1.2 * (mats[None, :] - 0.25))  # mean-reverting term structure
    atm = atm + term_tilt[:, None] * (mats[None, :] - 0.25)
    scale = np.sqrt(0.25 / mats)
    skew, curv = -0.15 * scale, 0.8 * scale
    m = MONEYNESS - 1.0
    grid = (
        atm[:, :, None]
        + skew_mult[:, None, None] * skew[None, :, None] * m[None, None, :]
        + curv[None, :, None] * m[None, None, :] ** 2
    )
    return np.maximum(grid, 0.05)


# ----------------------------------------------------------------------------- generator


def generate(seed: int = 42, n_days: int = 756, end_date: str = "2026-09-30") -> SampleDataset:
    """Generate the complete synthetic dataset (deterministic for a given seed)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp(end_date), periods=n_days)
    n = len(dates)

    # ---- common factors & volatility regime ---------------------------------------
    z_mkt = _t(rng, n)
    for d, shock in STRESS_DAYS.items():
        ts = pd.Timestamp(d)
        if ts in dates:
            z_mkt[dates.get_loc(ts)] = shock
    z_tech, z_usd = _t(rng, n), _t(rng, n)
    alpha, beta = 0.07, 0.90
    g2 = np.ones(n)
    for t in range(1, n):
        g2[t] = (1 - alpha - beta) + alpha * g2[t - 1] * z_mkt[t - 1] ** 2 + beta * g2[t - 1]
    # Normalise the regime to unit mean-square so long-run realised vol matches each
    # instrument's target vol; clustering and spikes are preserved.
    g = np.minimum(np.sqrt(g2 / g2.mean()), 3.5)

    # ---- equities ------------------------------------------------------------------
    eps_eq = _t(rng, (n, len(EQUITIES)))
    eq_prices, eq_rets = {}, {}
    for i, e in enumerate(EQUITIES):
        resid = np.sqrt(max(1 - e.beta_mkt**2 - e.beta_tech**2, 0.05))
        shock = e.beta_mkt * z_mkt + e.beta_tech * z_tech + resid * eps_eq[:, i]
        r = e.drift / 252 + e.vol / np.sqrt(252) * g * shock
        r[0] = 0.0
        r = np.clip(r, -0.45, 0.45)
        path = np.cumprod(1 + r)
        eq_prices[e.ticker] = path / path[-1] * e.end_price
        eq_rets[e.ticker] = r

    # ---- FX (USD per unit of foreign currency) ---------------------------------------
    fx_usd = {}
    for f in FX:
        resid = np.sqrt(1 - f.beta_mkt**2 - f.beta_usd**2)
        r = f.vol / np.sqrt(252) * np.sqrt(g) * (f.beta_mkt * z_mkt + f.beta_usd * z_usd + resid * _t(rng, n))
        r[0] = 0.0
        path = np.cumprod(1 + r)
        end_usd = 1 / f.end_quote if f.pair.startswith("USD") else f.end_quote
        fx_usd[f.currency] = path / path[-1] * end_usd

    # ---- Treasury curve (Nelson-Siegel factors around a start->end path) --------------
    load = _ns_loadings(TENOR_YEARS)
    phi = np.array([0.995, 0.993, 0.99])
    sig = np.array([0.00045, 0.00028, 0.00018])
    dx = np.column_stack(
        [
            sig[0] * g**0.35 * (0.25 * z_mkt + np.sqrt(1 - 0.0625) * _t(rng, n)),
            sig[1] * g**0.35 * (-0.20 * z_mkt + np.sqrt(1 - 0.04) * _t(rng, n)),
            sig[2] * g**0.30 * _t(rng, n),
        ]
    )
    x = np.zeros((n, 3))
    for t in range(1, n):
        x[t] = phi * x[t - 1] + dx[t]
    ramp = _ramp(n)
    base = START_CURVE[None, :] + (END_CURVE - START_CURVE)[None, :] * ramp[:, None]
    noise = x @ load.T
    curve = base + noise - ramp[:, None] * noise[-1][None, :]
    curve = np.maximum(curve, 0.0025)

    # ---- IG spread curve (lognormal level factor) ---------------------------------------
    u = np.zeros(n)
    su = 0.03 * g * (-0.55 * z_mkt + np.sqrt(1 - 0.3025) * _t(rng, n))
    for t in range(1, n):
        u[t] = 0.99 * u[t - 1] + su[t]
    spread = IG_SPREAD[None, :] * np.exp(u - ramp * u[-1])[:, None]

    # ---- implied vols & surfaces -----------------------------------------------------
    rng_surface = np.random.default_rng(seed + 1000)
    skew_common = _ar1(rng_surface.normal(0, 0.03, n), 0.97)
    surfaces: dict[str, np.ndarray] = {}
    for i, e in enumerate(EQUITIES):
        if e.base_iv is None:
            continue
        v = np.zeros(n)
        shocks = 0.035 * (-0.55 * z_mkt - 0.35 * eps_eq[:, i] + 0.76 * _t(rng, n))
        for t in range(1, n):
            v[t] = 0.97 * v[t - 1] + shocks[t]
        iv3m = np.clip(e.base_iv * g**0.5 * np.exp(v), 0.08, 1.5)
        # Smile dynamics come from their own random stream, so adding them leaves every
        # other series (prices, curves, ATM vols) exactly as before.
        own_skew = _ar1(rng_surface.normal(0, 0.03, n), 0.97)
        skew_mult = np.clip(g**0.5 * np.exp(skew_common + own_skew), 0.4, 2.5)  # skew steepens in stressed regimes
        term_tilt = _ar1(rng_surface.normal(0, 0.003, n), 0.97) - 0.02 * (g - 1.0)  # term structure inverts in stress
        surfaces[e.ticker] = _vol_grid(iv3m, e.base_iv, skew_mult, term_tilt)

    # ---- assemble frames ---------------------------------------------------------------
    md_frames = []
    for e in EQUITIES:
        close = eq_prices[e.ticker]
        prev = np.concatenate([[close[0]], close[:-1]])
        gap = 1 + 0.25 * (close / prev - 1) + rng.normal(0, 0.002, n)
        open_ = prev * gap
        spread_hl = np.abs(rng.normal(0, 0.006, n)) * g
        high = np.maximum(open_, close) * (1 + spread_hl)
        low = np.minimum(open_, close) * (1 - spread_hl)
        volume = np.round(rng.lognormal(np.log(2e7 / close.mean() * 100), 0.35, n) * g)
        rv = pd.Series(np.log(close)).diff().rolling(21).std().to_numpy() * np.sqrt(252)
        md_frames.append(
            pd.DataFrame(
                {
                    "date": dates, "instrument_id": e.ticker, "asset_type": "EQUITY", "price": close, "yield": np.nan,
                    "volatility": rv, "currency": e.currency, "open": open_, "high": high, "low": low,
                    "adj_close": close, "volume": volume, "duration": np.nan, "convexity": np.nan,
                }
            )
        )
    for f in FX:
        usd = fx_usd[f.currency]
        quote = 1 / usd if f.pair.startswith("USD") else usd
        rv = pd.Series(np.log(usd)).diff().rolling(21).std().to_numpy() * np.sqrt(252)
        md_frames.append(
            pd.DataFrame(
                {
                    "date": dates, "instrument_id": f.pair, "asset_type": "FX", "price": quote, "yield": np.nan,
                    "volatility": rv, "currency": f.pair[3:],
                }
            )
        )

    tenor_idx = {lbl: i for i, lbl in enumerate(TENOR_LABELS)}
    for b in BONDS:
        sched = BondSchedule.build(b.maturity, b.coupon, b.frequency)
        rows = []
        for t, d in enumerate(dates):
            times, amounts = sched.flows_after(d.date())
            w = interpolation_weights(TENOR_YEARS, times)
            z = w @ curve[t]
            if b.credit_curve:
                z = z + w @ spread[t]
            px = float(np.sum(amounts * (1 + z) ** (-times)))
            y = yield_to_maturity(px, times, amounts, b.frequency)
            rows.append((px, y, modified_duration(y, times, amounts, b.frequency), convexity(y, times, amounts, b.frequency)))
        arr = np.array(rows)
        md_frames.append(
            pd.DataFrame(
                {
                    "date": dates, "instrument_id": b.ticker, "asset_type": "BOND", "price": arr[:, 0], "yield": arr[:, 1],
                    "volatility": np.nan, "currency": "USD", "duration": arr[:, 2], "convexity": arr[:, 3],
                }
            )
        )

    mats = np.array(list(SURFACE_MATURITIES.values()))
    for o in OPTIONS:
        spot = eq_prices[o.underlying]
        tau = np.array([(o.expiry - d.date()).days / 365.0 for d in dates])
        grid = surfaces[o.underlying]
        ivs = np.array(
            [float(GridVolSurface(mats, MONEYNESS, grid[t]).vol(o.strike / spot[t], tau[t])) for t in range(n)]
        )
        w_tau = np.stack([interpolation_weights(TENOR_YEARS, max(x, 1e-6))[0] for x in tau])
        r = np.log1p(np.sum(w_tau * curve, axis=1))
        px = black_scholes_price(spot, o.strike, tau, r, ivs, o.option_type)
        md_frames.append(
            pd.DataFrame(
                {
                    "date": dates, "instrument_id": o.instrument_id, "asset_type": "OPTION", "price": np.maximum(px, 0.01),
                    "yield": np.nan, "volatility": ivs, "currency": "USD",
                }
            )
        )
    market_data = pd.concat(md_frames, ignore_index=True)
    market_data["date"] = market_data["date"].dt.date

    curve_rows = []
    for cid, levels in (("USD_TSY", curve), ("USD_IG", spread)):
        frame = pd.DataFrame(levels, index=dates.date, columns=list(TENOR_LABELS)).stack().reset_index()
        frame.columns = ["date", "tenor", "rate"]
        frame["curve_id"] = cid
        frame["tenor_years"] = frame["tenor"].map(lambda lbl: float(TENOR_YEARS[tenor_idx[lbl]]))
        curve_rows.append(frame)
    yield_curves = pd.concat(curve_rows, ignore_index=True)[["date", "curve_id", "tenor", "tenor_years", "rate"]]

    surf_rows = []
    labels = list(SURFACE_MATURITIES)
    for und, grid in surfaces.items():
        d_idx, m_idx, k_idx = np.meshgrid(np.arange(n), np.arange(len(labels)), np.arange(MONEYNESS.size), indexing="ij")
        surf_rows.append(
            pd.DataFrame(
                {
                    "date": dates.date[d_idx.ravel()],
                    "underlying": und,
                    "maturity": np.array(labels)[m_idx.ravel()],
                    "maturity_years": mats[m_idx.ravel()],
                    "moneyness": MONEYNESS[k_idx.ravel()],
                    "implied_vol": grid.ravel(),
                }
            )
        )
    vol_surface = pd.concat(surf_rows, ignore_index=True)

    instruments = _instrument_master()
    portfolio = _portfolio_frame(market_data, dates[-1].date(), instruments)
    logger.info(
        "Generated %d market observations, %d curve points, %d surface points",
        len(market_data), len(yield_curves), len(vol_surface),
    )
    return SampleDataset(instruments, market_data, yield_curves, vol_surface, portfolio)


def _instrument_master() -> pd.DataFrame:
    rows = []
    for e in EQUITIES:
        rows.append({"instrument_id": e.ticker, "ticker": e.ticker, "name": e.name, "asset_type": "EQUITY", "currency": e.currency, "sector": e.sector})
    for b in BONDS:
        rows.append(
            {
                "instrument_id": b.ticker, "ticker": b.ticker, "name": b.name, "asset_type": "BOND", "currency": "USD",
                "sector": "Corporate" if b.credit_curve else "Government", "coupon_rate": b.coupon,
                "maturity_date": b.maturity, "coupon_frequency": b.frequency, "face_value": 100.0, "credit_curve": b.credit_curve,
            }
        )
    for f in FX:
        rows.append(
            {
                "instrument_id": f.pair, "ticker": f.pair, "name": f"{f.pair[:3]}/{f.pair[3:]} spot", "asset_type": "FX",
                "currency": f.pair[3:], "sector": "FX", "base_currency": f.pair[:3], "quote_currency": f.pair[3:],
            }
        )
    for o in OPTIONS:
        rows.append(
            {
                "instrument_id": o.instrument_id, "ticker": o.underlying,
                "name": f"{o.underlying} {o.strike:g} {o.option_type.title()} {o.expiry:%b-%y}", "asset_type": "OPTION",
                "currency": "USD", "sector": "Equity Derivatives", "underlying_id": o.underlying, "strike": o.strike,
                "expiry_date": o.expiry, "option_type": o.option_type, "contract_multiplier": 100.0,
            }
        )
    rows.append({"instrument_id": "CASH_USD", "ticker": "USD", "name": "US Dollar cash", "asset_type": "CASH", "currency": "USD", "sector": "Cash"})
    cols = [
        "instrument_id", "ticker", "name", "asset_type", "currency", "sector", "coupon_rate", "maturity_date",
        "coupon_frequency", "face_value", "credit_curve", "underlying_id", "strike", "expiry_date", "option_type",
        "contract_multiplier", "base_currency", "quote_currency",
    ]
    return pd.DataFrame(rows).reindex(columns=cols)


def _portfolio_frame(market_data: pd.DataFrame, valuation_date: date, instruments: pd.DataFrame) -> pd.DataFrame:
    """Sample portfolio in the upload format, priced on the valuation date."""
    snap = market_data[market_data["date"] == valuation_date].set_index("instrument_id")
    eurusd = float(snap.at["EURUSD", "price"])
    inst = instruments.set_index("instrument_id")
    rows = []
    for pid, atype, ref, qty in PORTFOLIO:
        row: dict = {"position_id": pid, "asset_type": atype, "quantity": qty}
        if atype == "EQUITY":
            px = float(snap.at[ref, "price"])
            ccy = inst.at[ref, "currency"]
            fx = eurusd if ccy == "EUR" else 1.0
            row.update(ticker=ref, price=round(px, 4), currency=ccy, market_value=qty * px * fx)
        elif atype == "BOND":
            px = float(snap.at[ref, "price"])
            row.update(
                ticker=ref, price=round(px, 6), currency="USD", market_value=qty * px / 100,
                maturity=inst.at[ref, "maturity_date"], face_value=qty, coupon=inst.at[ref, "coupon_rate"],
                coupon_frequency=int(inst.at[ref, "coupon_frequency"]), credit_curve=inst.at[ref, "credit_curve"],
            )
        elif atype == "FX":
            px = float(snap.at[ref, "price"])
            usd = 1 / px if ref.startswith("USD") else px
            row.update(ticker=ref, price=round(px, 6), currency="USD", market_value=qty * usd)
        elif atype == "OPTION":
            px = float(snap.at[ref, "price"])
            row.update(
                ticker=inst.at[ref, "underlying_id"], price=round(px, 4), currency="USD", market_value=qty * 100 * px,
                maturity=inst.at[ref, "expiry_date"], strike=inst.at[ref, "strike"], option_type=inst.at[ref, "option_type"],
                volatility=round(float(snap.at[ref, "volatility"]), 6),
            )
        else:
            row.update(ticker=ref, price=1.0, currency=ref, market_value=qty)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["weight"] = df["market_value"] / df["market_value"].sum()
    cols = [
        "position_id", "asset_type", "ticker", "quantity", "price", "market_value", "currency", "weight", "maturity",
        "face_value", "coupon", "strike", "option_type", "volatility", "coupon_frequency", "credit_curve",
    ]
    df = df.reindex(columns=cols)
    df["market_value"] = df["market_value"].round(2)
    df["weight"] = df["weight"].round(6)
    return df


def write_dataset(dataset: SampleDataset, out_dir: Path) -> dict[str, Path]:
    """Write the dataset CSVs; returns the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "instruments": out_dir / "instruments.csv",
        "market_data": out_dir / "market_data.csv",
        "yield_curves": out_dir / "yield_curves.csv",
        "vol_surface": out_dir / "vol_surface.csv",
        "portfolio": out_dir / "sample_portfolio.csv",
    }
    dataset.instruments.to_csv(files["instruments"], index=False)
    dataset.market_data.to_csv(files["market_data"], index=False, float_format="%.8g")
    dataset.yield_curves.to_csv(files["yield_curves"], index=False, float_format="%.8g")
    dataset.vol_surface.to_csv(files["vol_surface"], index=False, float_format="%.6g")
    dataset.portfolio.to_csv(files["portfolio"], index=False)
    (out_dir / VERSION_FILE).write_text(DATASET_VERSION, encoding="utf-8")
    return files


def generate_sample_data(out_dir: Path | None = None, seed: int = 42, n_days: int = 756, end_date: str = "2026-09-30") -> dict[str, Path]:
    """Generate the dataset and write it to ``out_dir`` (default ``data/sample``)."""
    out = out_dir or ROOT / "data" / "sample"
    files = write_dataset(generate(seed, n_days, end_date), out)
    logger.info("Sample data written to %s", out)
    return files


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=756)
    parser.add_argument("--end", default="2026-09-30")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sample")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    files = generate_sample_data(args.out, args.seed, args.days, args.end)
    for name, path in files.items():
        print(f"{name:14s} -> {path}")


if __name__ == "__main__":
    main()
