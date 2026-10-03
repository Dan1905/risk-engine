# Multi-Asset Portfolio Risk & Stress Testing Engine

An internal-style market-risk platform in Python. It ingests a multi-asset portfolio (equities, government and corporate bonds, FX, equity options, cash) and its market history, and produces:

- historical, exponentially weighted historical, filtered historical, parametric and Monte Carlo VaR, plus Expected Shortfall
- EWMA volatility and covariance
- option Greeks, bond duration, convexity, DV01 and key rate durations
- P&L attribution and Euler risk contributions
- historical and hypothetical stress tests, including beta-mapped crisis scenarios, replays of stored market history, yield-curve twists and sticky-strike / sticky-delta volatility shocks
- VaR backtesting with the Kupiec, Christoffersen and Basel traffic-light tests
- risk-limit monitoring and concentration warnings
- CSV, Excel and PDF reports
- a scheduled end-of-day batch, Docker packaging and CI, on SQLite or PostgreSQL

Everything is served through a Streamlit dashboard. Every number on screen is computed from the stored portfolio and market data. There are no hard-coded KPIs.

---

## 1. Project Overview

| Layer | Package | Responsibility |
|---|---|---|
| UI | `ui/`, `app.py` | Streamlit pages, Plotly charts, session state and caching. No financial logic. |
| Application services | `services/` | Orchestration: portfolio, market data, risk, stress, P&L, backtest, reports, bootstrap |
| Stress engine | `stress/` | Scenario objects, shock application, curve and volatility shocks, scenario library |
| Analytics | `analytics/` | Pure functions: returns, EWMA, VaR/ES, Black-Scholes, Greeks, bonds, KRD, attribution, backtesting, full revaluation |
| Validation models | `models/` | Pydantic schemas: Position, Bond, Option, Portfolio, MarketDataPoint, RiskConfiguration, StressScenario, RiskResult |
| Persistence | `database/` | SQLAlchemy ORM, connection management, repository with idempotent upserts |
| Data | `scripts/`, `data/` | Deterministic synthetic data generator and seeder |

The sample book is the **Global Macro Fund** (about $25.2M NAV, valued 30 Sep 2026). It holds 8 equities (one EUR-listed), 5 bonds (4 Treasuries and 1 IG corporate), 3 FX holdings (EUR, GBP, INR), 3 options (long AAPL call, long MSFT put, short NVDA call) and USD cash, with 756 trading days of history.

## 2. Why Market Risk?

A trading or investment book changes value every day as prices, rates, spreads, currencies and volatilities move. Market risk asks four questions:

1. How much could we lose on a normal bad day? (VaR)
2. How bad is the tail beyond that? (ES)
3. What drives the risk? (sensitivities, contributions)
4. What happens in a crisis we can name? (stress tests)

Backtesting then checks that the VaR model is honest. This engine answers all of them from one consistent revaluation path.

## 3. Architecture

```text
Streamlit UI (ui/, app.py)                 presentation, session state, caching
        │
Application services (services/)           orchestration, persistence of results
        │
 ┌──────┼──────────────┬──────────────┐
Risk (VaR/ES)   Stress engine (stress/)   P&L attribution / backtesting
        │
Pricing & sensitivities (analytics/revaluation.py, options, greeks, bonds, KRD)
        │
Market-data layer (services/market_data_service.py: provider abstraction, MarketHistory)
        │
SQL database (database/: SQLAlchemy ORM + repository)
```

**One pricing path.** `analytics/revaluation.PortfolioPricer` values the static portfolio under any `MarketState`, or under a batch of factor scenarios (`ScenarioSet`). VaR, stress, attribution and backtesting all use it.

**Market factors**

- equity prices, per ticker
- FX as USD per unit of foreign currency
- Treasury zero-curve nodes (3M to 30Y)
- IG credit-spread curve nodes
- implied volatility (3M ATM level plus surface shape)

**Market-data provider abstraction.** `MarketDataProvider` has database and CSV implementations. An external API only needs to implement `load_instruments`, `load_market_data`, `load_yield_curves` and `load_vol_surfaces`.

## 4. Installation

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
cp .env.example .env                                   # optional
```

## 5. Startup

```bash
streamlit run app.py
```

**First launch:**

1. Check the database.
2. Create the schema.
3. Detect that there is no seed marker.
4. Generate the sample data (`scripts/generate_sample_data.py`).
5. Seed and validate (`scripts/seed_database.py`).
6. Validate the portfolio.
7. Open the dashboard, fully populated.

This takes about 5 seconds. The sidebar shows `✓ Database connected · ✓ Market data loaded · ✓ Portfolio validated · ✓ Risk engine ready`.

**Later launches** find the `seed_version` marker in `app_metadata` and skip seeding. Seeding is idempotent: every insert is `INSERT … ON CONFLICT DO NOTHING` on the table's natural key, so instruments, positions, market data and scenarios are never duplicated.

The scripts can still be run by hand for development:

```bash
python scripts/generate_sample_data.py --seed 42 --days 756 --end 2026-09-30
python scripts/seed_database.py [--force-generate]
pytest
```

**PostgreSQL** only needs a connection-string change. Set `DATABASE_URL=postgresql+psycopg://user:pwd@host/db` and `pip install -r requirements-postgres.txt`. The schema and upsert statements are dialect-aware, and the engine produces the same figures on both databases.

**Dataset upgrades.** The generated sample data carries a version. When it changes, an already-seeded database is refreshed in place on the next launch (`ON CONFLICT DO UPDATE` on the seeded market data, curves, surfaces and sample positions). Row counts do not change, and saved portfolios and custom scenarios are kept. The predefined scenario library is re-synchronised from `config/scenarios.yaml` on every launch.

### End-of-day batch

`scripts/run_eod.py` runs the whole risk process without the dashboard and persists every result:

```bash
python scripts/run_eod.py                      # run now for the configured portfolio
python scripts/run_eod.py --all                # every portfolio in the database
python scripts/run_eod.py --schedule 18:00     # stay running; execute every weekday at 18:00 local time
```

One run does the following, in about 4 seconds for the sample book:

1. initialises the database if needed (schema, sample data, scenario library)
2. computes VaR and ES for all six methods, and risk contributions
3. runs the stress library and the P&L attribution
4. backtests all six VaR models
5. checks limits
6. writes `risk_report.xlsx` and `risk_summary.pdf` to `reports/<portfolio>_<date>/`

Every write is an upsert, so re-running for the same valuation date replaces that day's figures. A summary of the last run is stored in `app_metadata` and shown on the Settings page.

Exit codes are 0 for success, 1 for failure, and 2 for a limit breach when `--fail-on-breach` is set. A cron entry such as `0 18 * * 1-5 python scripts/run_eod.py` is equivalent to the built-in scheduler.

### Docker

```bash
docker compose up --build                                   # dashboard on http://localhost:8501, backed by PostgreSQL
docker compose run --rm eod python scripts/run_eod.py       # one-off end-of-day run
```

`docker-compose.yml` starts three services:

| Service | Role |
|---|---|
| `db` | PostgreSQL 16, with data in a named volume |
| `app` | the dashboard, with a health check on `/_stcore/health` |
| `eod` | the scheduled batch, weekdays at `EOD_TIME` (default 18:00) |

The image runs as an unprivileged user. Set `POSTGRES_PASSWORD` in the environment or a `.env` file; the default is for local use only. `APP_PORT`, `EOD_TIME` and `TZ` are optional.

### Continuous integration

`.github/workflows/ci.yml` runs three jobs:

- **Tests:** the test suite and a first-launch batch run on Python 3.11, 3.12 and 3.13.
- **PostgreSQL integration:** seeding, repository and end-of-day batch against a PostgreSQL service container.
- **Docker image:** builds the image, waits for the dashboard to report healthy, and runs the batch inside it.

## 6. Portfolio Input Format

```csv
position_id,asset_type,ticker,quantity,price,market_value,currency,weight,maturity,face_value,coupon,strike,option_type,volatility[,coupon_frequency,credit_curve]
POS001,EQUITY,AAPL,8200,245,,USD,,,,,,,
POS011,BOND,US10Y,2250000,100.425,,USD,,2036-08-15,2250000,0.04,,,,2,
POS014,FX,EURUSD,1000000,1.17,,USD,,,,,,,
POS017,OPTION,AAPL,150,11.86,,USD,,2027-03-19,,,260,CALL,0.2434
POS020,CASH,USD,1200000,1,,USD,,,,,,,
```

| Asset type | `quantity` | `price` |
|---|---|---|
| EQUITY | shares | local-currency price |
| BOND | face amount held | dirty price per 100 |
| FX | foreign-currency amount (+ = long) | quoted rate (EURUSD = USD per EUR, USDINR = INR per USD); only USD-quoted pairs |
| OPTION | contracts (×100 multiplier) | premium per share; `maturity` = expiry |
| CASH | amount | 1 |

`market_value` and `weight` are recomputed; inputs are ignored and reconciled with a warning. Bonds are priced off the curve plus a residual spread that is calibrated to reproduce the input price.

**Validation (Pydantic)** checks:

- asset type, quantity, price, currency, maturity, strike, volatility, option type, position ID and dates
- duplicate IDs
- market-data coverage

All problems are reported together, and invalid portfolios never reach the engine:

```text
Portfolio validation failed

POS008:
Missing volatility for OPTION position.

POS014:
Maturity date is earlier than valuation date.

POS022:
Unsupported asset type: CRYPTO.
```

Uploads are made on the **Portfolio** page. A validated upload can be used for the session or saved to the database.

## 7. Market Data

Normalised table `market_data`:

`date | instrument_id | asset_type | price | yield | volatility | currency`

It also has OHLC, adjusted close and volume for equities, and duration and convexity for bonds. Curves are stored in `yield_curve_data` (`USD_TSY` zero curve and `USD_IG` spread curve, 10 tenors). Daily implied-vol surfaces are stored in `volatility_surface` (1M–1Y × 80–120% moneyness).

**Synthetic data** (deterministic seed) has:

- **Volatility regime.** A GARCH(1,1) regime on the market factor, normalised to unit mean-square, plus injected stress episodes. This produces volatility clustering.
- **Equities.** Student-t(5) shocks with market and sector factor loadings, giving realistic correlations of 0.2–0.6.
- **FX.** Loadings on a USD factor.
- **Treasury curve.** A Nelson-Siegel level/slope/curvature model moving from the inverted 2023 curve to a normal 2026 curve.
- **IG spreads.** Lognormal; they widen in risk-off.
- **Implied vol.** ATM vols rise with the regime and fall with spot. Surfaces have a term structure and a skew/smile whose slope and tilt vary over time (steeper skew and an inverted term structure in stressed regimes).

**Cleaning** (`analytics/returns.clean_price_frame`):

- sorts dates and keeps the last of any duplicate dates
- drops weekends
- treats non-positive or non-finite prices as missing
- forward-fills gaps of at most `max_forward_fill_days`, and never back-fills

The **Market Data** page shows data-quality statistics and accepts validated CSV uploads. These are upserted on (instrument, date), and caches are invalidated through a data version.

## 8. VaR Methodologies

The defaults are 99% confidence, a 250-day lookback and a 1-day horizon. Confidence (95 or 99%), lookback (100, 250 or 500), holding period (1, 5 or 10 days) and decay factor (0.90, 0.94, 0.97 or 0.99) are all configurable.

Six methods are computed side by side. All of them work on one **factor frame**: a matrix with one row per scenario and one column per market factor (`eq|AAPL`, `fx|EUR`, `curve|USD_TSY|10Y`, `vol|AAPL`, …; see `analytics/risk_factors.py`). The sample book has 40 factors.

**Historical simulation.** For each of the last *N* days:

1. take that day's factor moves (equity and FX returns, curve and spread node changes, ATM-vol change)
2. apply them to today's positions
3. fully revalue bonds and options
4. sum to portfolio P&L

The quantile uses the inverted empirical CDF:

```text
VaR_c = −Q_{1−c}(P&L),   Q = k-th worst P&L,  k = ceil(N(1−c))      (N=250, 99% → 3rd worst)
```

**Exponentially weighted historical.** Each scenario is weighted by its age:

```text
w_t ∝ λ^age,   Σ w_t = 1;   VaR = −(first P&L, sorted ascending, whose cumulative weight ≥ 1−c)
```

With equal weights this reduces exactly to the standard estimator, and a unit test verifies it. It reacts faster to regime changes because the most recent weeks carry most of the weight. The cost is a small effective sample of about 1/(1−λ), which is roughly 17 days at λ = 0.94. At 99% one recent observation can therefore set VaR, VaR 95%, and ES together.

**Parametric (variance-covariance).** Position returns are `r_i = PnL_i / MV_i` and weights are `w_i = MV_i / NAV`:

```text
σ_p = sqrt(wᵀ Σ w),     VaR = z_c · σ_p · NAV · sqrt(h)      (z_99 ≈ 2.326)
```

Σ is either the sample covariance or the latest EWMA covariance (the **EW Parametric** method). Means are assumed zero and returns jointly normal.

**Filtered historical simulation.** Plain historical simulation replays each past move at the volatility of its own day. The filtered version first rescales every factor move to today's volatility:

```text
move*_t = move_t · σ_{T+1} / σ_t        σ_t = EWMA forecast for day t, made with data up to t−1
```

The scenarios keep their empirical shape (fat tails and how factors move together) but their scale follows the current regime. VaR is then read off the equally weighted distribution, so the whole window is used. The scaling ratio is bounded (default ⅓ to 3) so one unusual day cannot dominate.

**Monte Carlo.** Factor moves are simulated from a zero-mean multivariate distribution with the lookback window's factor covariance (sample or EWMA), and every path is **fully revalued**. This captures option and bond convexity, which the linear variance-covariance method ignores.

- **Distribution:** multivariate Student-t (ν = 6 by default; all factors share one mixing variable, giving joint fat tails) or normal.
- **Covariance root:** from an eigen-decomposition, so singular matrices work.
- **Paths:** 10,000 by default, with antithetic variates.
- **Reproducibility:** a fixed seed, so identical inputs give identical results.

```text
X = Z·Aᵀ · sqrt((ν−2)/W),   Z ~ N(0, I),  A·Aᵀ = Σ,  W ~ χ²(ν)        (covariance of X is still Σ)
```

**Stand-alone VaR by factor group.** The VaR page also replays one factor group at a time: equity, FX, rates and credit curves, implied-vol level, skew and term structure. The diversified total is below the sum of the groups.

**Holding period.** Two methods are available for h > 1 day:

- **Overlapping scenarios (default).** The last `lookback + h − 1` daily moves are combined into overlapping h-day moves, with returns compounded and curve, spread and vol changes summed. Every position is revalued at the multi-day move, so compounding and non-linearity are captured. Adjacent windows share h − 1 days, so the effective sample is smaller than the scenario count. The volatility filter is applied to daily moves before they are aggregated. Monte Carlo simulates the h-day move directly (daily covariance × h).
- **Square-root-of-time.** `VaR_h = VaR_1 · sqrt(h)`, which assumes i.i.d. daily P&L and a linear book.

The two exponentially weighted estimators (EW Historical and EW Parametric) are defined on daily data. With λ = 0.94 only about 17 days carry weight, which is fewer than two independent 10-day periods, so they always report their 1-day estimate × sqrt(h). Applying them to overlapping windows produced figures less than half the other methods' in testing.

Scenarios are instantaneous shocks. Option time decay and bond carry over the horizon are not included.

## 9. EWMA Methodology

RiskMetrics recursion with λ = 0.94 by default:

```text
σ²_t        = λ σ²_{t−1}        + (1−λ) r²_{t−1}
Cov_t(i,j)  = λ Cov_{t−1}(i,j)  + (1−λ) r_{t−1,i} r_{t−1,j}
```

The recursion is seeded with the window's sample covariance and assumes a zero mean. The latest forecast is computed in closed form:

```text
Σ_{T+1} = (1−λ) Σ_k λ^k r_{T−k} r_{T−k}ᵀ + λ^n Σ_0
```

A unit test checks this closed form against the explicit loop. Variance series are vectorised with `scipy.signal.lfilter`. The VaR page compares EWMA and historical volatility per position, and shows both covariance matrices and their difference, with CSV download.

**Numerical safeguards.** VaR only needs `wᵀΣw`, so Σ is never inverted. `ensure_psd` clips materially negative eigenvalues, which can appear with EWMA or short windows, and logs a warning. Zero-variance assets such as cash get zero correlation instead of NaN. If a future feature needs `Σ⁻¹` (for example a minimum-variance hedge), `np.linalg.pinv` should be used. `condition_number` / `is_singular` report when that applies.

## 10. Expected Shortfall

```text
Historical:  ES_c = mean{ L_i : L_i ≥ VaR_c }            (L = −P&L)
Weighted:    ES_c = Σ_tail w_i L_i / Σ_tail w_i
Parametric:  ES_c = σ φ(z_c) / (1−c) · NAV · sqrt(h)
```

- **VaR** is the loss threshold exceeded in about (1−c) of observations.
- **ES** is the average loss conditional on entering that tail.

ES/VaR measures tail heaviness: about 1.15 for a normal distribution at 99%, and 1.28 for the sample book.

## 11. Black-Scholes

European options, with `d1 = [ln(S/K) + (r − q + σ²/2)T] / (σ√T)` and `d2 = d1 − σ√T`:

```text
Call = S e^{−qT} N(d1) − K e^{−rT} N(d2)
Put  = K e^{−rT} N(−d2) − S e^{−qT} N(−d1)
```

The rate *r* is the continuously compounded curve zero rate at expiry.

**Assumptions:**

- European exercise
- constant volatility and constant risk-free rate
- lognormal underlying
- no arbitrage
- continuous trading
- no dividends by default

**Safeguards:**

- T below 1 hour → intrinsic value
- σ ≈ 0 → discounted forward intrinsic
- non-positive S or K, or negative σ → error
- implied vol above 500% is rejected at validation, and above 200% triggers a warning

## 12. Greeks

Analytic Delta, Gamma, Vega, Theta and Rho, verified against finite differences and textbook values (S=K=100, T=1, r=5%, σ=20%: call 10.4506, Δ 0.6368, Γ 0.01876, ν 37.52).

Position reporting units:

| Greek | Unit |
|---|---|
| Delta | shares and USD |
| Gamma | Δ per $1 move, and P&L for a 1% move |
| Vega | per volatility point |
| Theta | per calendar day |
| Rho | per 1% move in rates |

The Sensitivities page shows position-level and portfolio-level aggregates.

## 13. Bond Analytics

Per 100 face, with coupon frequency *f*; with f = 1 this reduces to the specified formula `Σ C/(1+y)^t + F/(1+y)^T`:

```text
P      = Σ a_j (1 + y/f)^{−f t_j}
D_mac  = Σ t_j PV_j / P,            D_mod = D_mac / (1 + y/f)
C      = Σ a_j t_j (t_j + 1/f)(1 + y/f)^{−f t_j − 2} / P
DV01   = D_mod · P · 0.0001
```

Curve-based pricing is `P = Σ a_j (1 + z(t_j) + s)^{−t_j}`. The z-spread *s* is calibrated to the market price. Effective duration and convexity come from ±1 bp parallel curve shifts. Dates use ACT/365, prices are dirty, and coupon dates roll back from maturity.

## 14. Key Rate Duration

Each node of the 10-node curve (3M to 30Y) is bumped ±1 bp. Linear interpolation spreads each bump as a "tent" over the neighbouring maturities:

```text
KRD_k = −(P(z_k + h) − P(z_k − h)) / (2 h P)
```

**Properties (unit-tested):**

- Σ_k KRD_k equals effective duration.
- A 2Y bond has zero KRD beyond the 3Y node.
- KRD P&L matches full repricing for small node shocks.
- Dollar KR01 (`KRD·MV·1bp`) adds up across positions.

The page shows a portfolio KR01/KRD table, a bar chart and a bond × tenor heatmap.

## 15. Yield Curve Stress

Shocks are applied node by node, and bonds are fully repriced on the shocked curve. They are not modelled as a single yield shock.

**Parallel shifts:** +25, +50, +100 and +200 bp.

**Twists:** the short shock applies up to 2Y, the long shock from 10Y, linearly interpolated in between.

| Scenario | Short end | Long end |
|---|---|---|
| Steepener | +50 bp | +150 bp |
| Flattener | +150 bp | +50 bp |
| Inversion | +150 bp | −50 bp |

The output table has: Scenario, Portfolio P&L, Fixed-Income P&L, Worst Bond, KRD Impact (first-order `−Σ MV·KRD_k·Δz_k`), and the convexity / non-linear remainder.

Fixed-income P&L can be computed two ways, and the page labels which one is used:

- **Full Repricing:** prices are recomputed on the shocked curve.
- **Sensitivity Approximation:** `ΔP/P ≈ −D·Δy + ½C·Δy²`, with Δy built as a KRD-weighted combination of the node shocks.

## 16. Volatility Surface

The surface is `σ = f(K/S, T)`. Three implementations:

- `FlatVolSurface`
- `SmileVolSurface`: a quadratic smile, which can be built from ATM / 25-delta call / 25-delta put quotes
- `GridVolSurface`: a maturity × moneyness grid with bilinear interpolation, or piecewise-constant **maturity buckets**

Option positions are anchored to their input implied vol and move with their surface.

**Stress conventions:**

- **Sticky strike:** `σ_new = σ_old + shift`.
- **Sticky delta (moneyness):** `σ_new = σ_old + [f(K/S_new) − f(K/S_old)] + shift`.

You select the convention on the stress pages (shifts of +5, +10, +15 or −5 vol points), and every result states which one was used.

**Volatility risk factors in VaR.** Each optionable underlying contributes three factors taken from its stored surface history:

| Factor | Definition |
|---|---|
| Level | 3M at-the-money vol |
| Skew | 3M vol at 90% moneyness minus vol at 110% |
| Term structure | 1Y minus 1M at-the-money vol |

In a scenario, an option with moneyness m = K/S and maturity τ has its vol moved by:

```text
Δσ(m, τ) = Δlevel + Δskew · (1 − m)/0.2 + Δterm · (τ − 0.25)/0.917        (m clipped to 80–120%, τ to 1M–1Y)
```

The loadings reproduce the three reference quantities exactly. With the level factor alone, every strike and maturity is assumed to move in parallel.

On the sample data the three-factor model cuts the unexplained daily vol change at each option's own strike and maturity by about 85–90% (for the AAPL call, from 0.62 to 0.08 vol points). VaR exceptions in the backtest fall from 12 to 10 for historical simulation. The factor model can be switched between "level only" and "level + skew + term" on the VaR page.

## 17. Historical Stress

The **2008 Financial Crisis** (29 Aug to 20 Nov 2008) and the **COVID crash** (19 Feb to 23 Mar 2020) are defined in `config/scenarios.yaml`. Each uses approximate, rounded index-level moves:

- S&P 500
- the UST curve, node by node
- EUR, GBP and INR against USD
- IG credit spreads

Every input is tagged either `HISTORICAL_OBSERVATION` or `HYPOTHETICAL_ASSUMPTION`. For example, mapping the VIX change onto single-name ATM vols is tagged as an assumption. The UI shows this basis table for each scenario.

**Beta mapping.** The index-level equity move can be applied two ways, selectable on the Stress Testing page:

- **Beta-mapped (default for the historical scenarios):** each stock moves by `β_i × index move`, floored at −95%. β_i = Cov(r_i, r_m)/Var(r_m) is estimated over the last 250 days against the equal-weighted equity market. The betas average exactly 1, so the index move is preserved on average while high-beta names fall further. In the sample book betas run from 0.44 (JNJ) to 2.08 (NVDA), and the 2008 scenario loss rises from −$3.52M (uniform) to −$3.73M. Option underlyings are mapped the same way.
- **Uniform:** every stock gets the index move.

Explicit per-instrument overrides always win. The mapping itself is tagged as a modelling assumption.

**Replay a historical window.** `stress.historical.replay_window_scenario(history, start, end)` builds a scenario from the engine's own stored market data. It takes the cumulative move of every equity, currency, curve node, credit-spread node and implied-vol factor between two dates, and applies them to today's portfolio as one shock. Each instrument gets its own observed move, so dispersion and co-movement are exactly as they occurred.

The Stress Testing page has date pickers and a "find worst N-day window" button, which locates the window with the largest cumulative loss for the current portfolio. A one-day replay ending on the valuation date equals that day's historical-simulation scenario exactly; a unit test checks this.

## 18. Hypothetical Stress

| Scenario | Shocks |
|---|---|
| Equity Crash | equities −20% |
| +200bps Rates | parallel +200 bp |
| Global Risk-Off | equities −20%, rates +150 bp, USD +10%, vol +15 pts, credit +200 bp |

**Ladders:**

- equities: −5, −10, −20, −30 and −40%
- currencies: USD ±5%, EUR ±10%, INR ±10%
- volatility: +5, +10, +15 and −5 points

**Scenario Builder.** Set equity, rates, FX, volatility and credit shocks, with per-instrument overrides (for example AAPL −25%, EUR +10%, US10Y +150 bp). **RUN STRESS TEST** returns stressed NAV, P&L, P&L %, worst position, worst asset class and post-stress limit status. Scenarios can be saved to the library.

**Engine API:**

```python
from stress.scenario_engine import Scenario, run_stress_test
Scenario(scenario_id="GRO", name="Global Risk-Off", equity_shock=-0.20, rates_shock=0.015, fx_shock=0.10, volatility_shock=0.15)
result = run_stress_test(portfolio, scenario, market_state)   # position/asset-class P&L, total, return, stressed NAV
```

**FX conventions.**

- "USD +x%" means every foreign currency's USD value is multiplied by 1/(1+x).
- "EUR +x%" means EURUSD is multiplied by (1+x).

Stress tests are deterministic what-if revaluations, not probability forecasts.

## 19. P&L Attribution

Actual P&L is the full revaluation of the static book between two closes, plus coupon cash. Explained P&L is built from start-of-day sensitivities:

```text
Equity price  q·ΔP·FX₀                  FX         q·P₀·ΔFX
Bond yield    MV·(−D_mod·Δy + ½C·Δy²)   Carry      MV·((1+y/f)^{fΔt} − 1)
Option        Δ·ΔS + ½Γ·ΔS² + ν·Δσ + Θ·Δt
Residual      = Actual − Explained      (so Explained + Residual = Actual, checked within the configured tolerance)
```

The page shows:

- a NAV waterfall (Previous NAV → components → Current NAV)
- Explained / Unexplained / Explained %; the sample book explains 97.1% on the valuation date and 99.4% over 60 days
- a daily stacked history
- position and asset-class tables

## 20. Risk Contribution

Euler allocation of parametric VaR (sample or EWMA covariance):

```text
Marginal VaR_i  = z (Σw)_i / σ_p          (VaR per $1 of extra exposure)
Component VaR_i = w_i · Marginal VaR_i · NAV        Σ_i Component VaR_i = VaR
```

Results are aggregated by position, asset class and currency. Negative contributions are hedges, such as the long MSFT put.

**Concentration.** The engine reports the largest position, asset class, currency exposure, VaR contributor and stress-loss contributor. It raises warnings when configured thresholds are crossed, for example "NVDA contributes 26% of total portfolio VaR" or "USD exposure represents 83% of gross currency exposure".

The page also has a correlation heatmap (equities, bonds, FX, options with underlyings), highest positive, highest negative and average correlation, and covariance download.

## 21. VaR Backtesting

For every day *t* after the first `lookback + 1` days:

- **Predicted VaR_t** is computed from the `lookback` factor moves **ending on t−1**, applied to the **t−1 market state**.
- **Actual P&L_t** is `V(state_t) − V(state_{t−1}) + coupons`.

An exception occurs when `−P&L_t > VaR_t`. You can configure the start and end dates, confidence level, lookback and model.

**Outputs:**

- **Charts:** actual P&L vs the VaR threshold (with breaches marked), cumulative P&L, rolling exception rate and exceptions by month.
- **Exceptions table:** Date, Actual P&L, Predicted VaR, Excess Loss.
- **Model comparison:** all four VaR models over the same period.

All six VaR models can be backtested. Filtered historical simulation rebuilds each day's window with volatility forecasts available on t−1. Monte Carlo estimates the factor covariance from the window and simulates 1,000 paths per day with common random numbers (the same standard draws every day), so the VaR path reflects covariance and market changes, not sampling noise.

Results on the sample data (99%, 250-day lookback, 505 days, expected 5.1 exceptions):

| Model | Exceptions | Kupiec p | Independence p | Cond. coverage p | Basel zone |
|---|---|---|---|---|---|
| Historical | 10 | 0.051 | 0.012 | 0.006 | Green |
| EW Historical | 15 | 0.000 | 0.072 | 0.000 | Yellow |
| Parametric | 13 | 0.003 | 0.039 | 0.001 | Green |
| EW Parametric | 10 | 0.051 | 0.184 | 0.061 | Green |
| Filtered Historical | 6 | 0.680 | 0.053 | 0.142 | Green |
| Monte Carlo | 7 | 0.410 | 0.078 | 0.151 | Green |

Exceptions cluster after the April-2025 synthetic stress episode. Equally weighted historical VaR adapts slowly and fails the independence test. The volatility-aware methods (filtered historical, Monte Carlo on EWMA covariance) pass all three tests at the 5% level. These results come from the data, not from tuning.

## 22. Kupiec POF Test

With N observations, x exceptions and p = 1 − c:

```text
LR_POF = −2 ln[ (1−p)^{N−x} p^x / ((1 − x/N)^{N−x} (x/N)^x) ]  ~  χ²(1)
```

The p-value is `1 − F_χ²₁(LR)`, with `0·ln0 = 0` so that x = 0 and x = N work. The UI reports the result neutrally:

- "No statistically significant evidence of incorrect exception frequency at the selected significance level."
- "Observed exception frequency differs significantly from the model's expected exception frequency."

Test values: N=250, x=5, p=1% gives LR = 1.957 and p = 0.162.

### Christoffersen independence and conditional coverage

Kupiec checks only how many exceptions occurred. A model can have the right count and still be wrong if its exceptions arrive in clusters. Let n_ij be the number of days in state j (1 = exception) that follow a day in state i:

```text
π01 = n01/(n00+n01)     π11 = n11/(n10+n11)     π = (n01+n11)/(n00+n01+n10+n11)

LR_ind = −2 ln[ (1−π)^{n00+n10} π^{n01+n11} / ((1−π01)^{n00} π01^{n01} (1−π11)^{n10} π11^{n11}) ]  ~  χ²(1)
LR_cc  = LR_POF + LR_ind                                                                          ~  χ²(2)
```

`LR_ind` tests whether an exception is more (or less) likely after an exception. `LR_cc` tests frequency and independence jointly. The page reports the transition counts and both conditional probabilities.

### Basel traffic light

The Basel framework classifies a 99% VaR model by its exception count over the most recent 250 trading days. Zones follow the cumulative binomial probability P(X ≤ x) under a correct model:

| Zone | Exceptions (250 days) | P(X ≤ x) | Capital plus factor |
|---|---|---|---|
| Green | 0–4 | below 95% | 0.00 |
| Yellow | 5–9 | 95% to 99.99% | 0.40, 0.50, 0.65, 0.75, 0.85 |
| Red | 10 or more | 99.99% or above | 1.00 |

The plus factor is added to the base multiplier of 3. The test is always run at 99% on the last 250 days of the backtest period, whatever confidence is being viewed. With fewer than 250 days the binomial rule still gives a zone, but no plus factor.

## 23. Database Schema

**Reference data**

- `portfolios`
- `instruments`, with 1:1 extension tables `bond_terms`, `option_terms` and `fx_terms`
- `positions`, unique on (portfolio, position, valuation_date)

**Market data**

- `market_data`, unique on (instrument, date)
- `yield_curve_data`, unique on (curve, date, tenor)
- `volatility_surface`, unique on (underlying, date, maturity, moneyness)

**Results** (each written with an upsert on its natural key)

- `risk_metrics`
- `risk_contributions`
- `stress_scenarios`, with JSON definitions
- `stress_results`
- `pnl_attribution`, in long format
- `backtest_results`

**Seed status**

- `app_metadata`

## 24. Testing

```bash
pytest          # 204 tests, ~12 s (3 PostgreSQL tests are skipped unless TEST_DATABASE_URL is set)
```

| File | Covers |
|---|---|
| `test_var.py` | known distributions, 95/99%, weighted VaR, parametric scaling, Euler sums, row-wise estimators, edge cases |
| `test_es.py` | tail selection, weighted ES, parametric ES, ES ≥ VaR |
| `test_ewma.py` | variance and covariance recursion vs closed form, no look-ahead in the forecast, λ validation, PSD repair, correlation helpers |
| `test_options.py` | BS call/put, put-call parity, all Greeks (reference values and finite differences), degenerate limits, implied vol |
| `test_bonds.py` | price formula, par bond, durations, convexity, DV01 vs finite difference, YTM round trip, schedules, spread calibration |
| `test_krd.py` | localisation, Σ KRD = effective duration, repricing consistency, aggregation |
| `test_stress.py` | equity, rate, credit, FX, curve and vol shocks; sticky strike vs delta; combined scenario; vol surfaces; betas and beta-mapped shocks; credit-curve node shocks; worst-window search |
| `test_pnl.py` | Explained + Residual = Actual, NAV waterfall, exact linear components, small residuals, time decay |
| `test_backtesting.py` | exception counting, rates, Kupiec statistic and p-value, Christoffersen statistics by hand, clustered vs spread exceptions, Basel zones and plus factors, edge cases |
| `test_simulation.py` | factor frame and multi-day aggregation, volatility filter (recursion, no look-ahead, bounds), Monte Carlo (covariance, fat tails, determinism, convexity), skew / term loadings |
| `test_services.py` | risk aggregation, all six methods, overlapping multi-day scenarios, backtest look-ahead for every model (rebuilds a history truncated at t−1 and compares), beta mapping, replay scenarios, vol factor models, idempotent seeding, in-place dataset refresh, end-of-day batch |
| `test_postgres.py` | seeding, portfolio round trip and end-of-day batch on PostgreSQL |
| `test_validation.py` | spec error messages, field validation, aggregation, CSV and price cleaning |

## 25. Assumptions

- **Static portfolio.** Today's quantities are used for historical P&L, backtests and attribution, with no intraday trading or rebalancing.
- **Factor model.** Equity and FX moves use simple returns. Curves, spreads and implied vols use absolute changes. Implied vol has level, skew and term-structure factors per underlying, applied sticky-strike in VaR. Returns are converted to log form only for covariance estimation, if configured.
- **Zero-coupon curve.** Annually compounded with linear interpolation and flat extrapolation. Credit bonds discount on curve + IG spread curve + a constant calibrated residual. Government bonds discount on curve + residual.
- **Black-Scholes.** European options; the option currency equals the underlying's currency.
- **Base currency.** USD only. Bonds must be USD. FX pairs must be quoted against USD.
- **Horizons.** Overlapping multi-day scenarios by default (square-root-of-time optional, and always used for the two exponentially weighted estimators). VaR scenarios and stress shocks are instantaneous (no time decay or carry).
- **Monte Carlo.** Zero-mean elliptical factor distribution (Student-t or normal) with the window's covariance; multi-day moves use daily covariance × h.
- **Betas.** Estimated against the equal-weighted average of the equities in the market data, which stands in for a market index.
- **Synthetic data.** All market data is synthetic and deterministic (seed 42). The historical stress moves are rounded approximations used as a scenario framework.

## 26. Limitations

- There is no dividend or corporate-action modelling. Adjusted close equals close in the sample data.
- **Beta mapping** uses a single market factor. There is no sector or style factor model.
- **Volatility factors** are three per underlying (level, skew, term). Smile curvature and maturity-dependent skew are not separate factors.
- **Overlapping multi-day scenarios** are serially dependent, which reduces the effective sample size; no correction is applied.
- **Monte Carlo** does not model volatility clustering within the horizon or factor-specific tail behaviour.
- **Stored results** are keyed by portfolio, date, method, confidence, lookback, holding period and decay factor. Runs that differ only in horizon method or vol factor model overwrite each other.
- **No schema migrations.** New tables are created automatically, but changing an existing table needs a manual migration.
- **Backtest exposures:** P&L is hypothetical for the static book. Clean versus dirty P&L and risk-not-in-VaR are not separated.
- **Instrument scope:** there are no American options, callable or floating-rate bonds, inflation linkers, futures or swaps, and there is no multi-currency curve.
- **No user management:** there is no authentication, audit trail of users, or scheduling.

## 27. Future Improvements

Each of these is a candidate enhancement, not yet implemented:

- Sector or multi-factor equity model for stress mapping and risk decomposition.
- GARCH-filtered (rather than EWMA-filtered) historical simulation, and Monte Carlo with stochastic volatility.
- Expected Shortfall backtesting, and a correction for the dependence of overlapping windows.
- Curvature and maturity-dependent skew as volatility factors.
- More instruments: multi-currency curves, swaps and futures.
- Schema migrations (Alembic), authentication and an audit trail of users.

---

### Look-Ahead Bias Prevention

When historical VaR is calculated for valuation date T, only market information available on or before T is used. Three mechanisms enforce this:

1. `MarketHistory.build(end=T)` never loads observations after T.
2. `historical_scenarios(end, n)` returns only moves dated on or before `end`, and asserts it.
3. In the backtest, VaR_t is built from moves ending on t−1 and the t−1 market state. `tests/test_services.py::test_backtest_has_no_look_ahead` rebuilds a history that physically ends at t−1 and checks that the forecast window is identical.
4. Filtered historical simulation uses volatility forecasts σ_t built only from moves before t; `test_simulation_backtests_have_no_look_ahead` applies the same truncated-history check to it.
5. Equity betas for stress mapping are estimated from data up to the valuation date.

### Configuration

All risk parameters, limits, concentration thresholds, stress ladders and data-generation settings are in `config/risk_config.yaml`. Scenarios are in `config/scenarios.yaml`. Infrastructure settings are in `.env` (see `.env.example`). The Settings page applies session-level overrides.

### Performance (sample book, Apple Silicon)

| Operation | Time |
|---|---|
| Portfolio and market-data load | ≈ 0.4 s |
| VaR, all six methods (including 10,000 Monte Carlo paths, fully revalued) | ≈ 0.03 s |
| All predefined stresses | ≈ 0.02 s |
| Backtest forecasts, four base models (505 days × 250 scenarios, full revaluation) | ≈ 1.0 s, cached |
| Backtest, filtered historical / Monte Carlo (built on first use) | ≈ 0.7 s / 1.1 s, cached |
| End-of-day batch (everything, with reports) | ≈ 4 s |
| Database seeding (SQLite / PostgreSQL) | ≈ 2 s / 4 s |
