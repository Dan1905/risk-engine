"""VaR backtesting service with strict look-ahead prevention.

For every backtest day t (index i in the market history):

    base state      = market state on t-1
    scenarios       = the ``lookback`` daily factor moves ending on t-1
    predicted VaR_t = VaR of the static portfolio's hypothetical P&L under
                      those scenarios, revalued from the t-1 state
    actual P&L_t    = V(state_t) - V(state_{t-1}) + coupons in (t-1, t]

Nothing dated t or later enters VaR_t. The forecasts for all methods share the
same scenario P&L windows, so switching method or confidence only re-reads the
quantiles.

Positions are held constant at today's quantities (a "hypothetical P&L"
backtest, which isolates model performance from trading activity).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from analytics.backtesting import BASEL_CONFIDENCE, BASEL_WINDOW, BacktestResult, BaselResult, basel_traffic_light, count_exceptions, run_backtest
from analytics.ewma import ewma_variance_series, ewma_weights
from analytics.volatility import rolling_volatility
from analytics.filtered_simulation import ewma_vol_forecasts, scaling_ratios
from analytics.monte_carlo import covariance_root, standard_draws
from analytics.revaluation import ScenarioSet
from analytics.risk_factors import RETURN_FLOOR, return_mask, select_vol_factors
from analytics.var import historical_var_rows, parametric_var_rows, weighted_historical_var_rows
from config.settings import AppSettings
from database.repository import RiskRepository
from services.risk_service import RiskContext

logger = logging.getLogger(__name__)


@dataclass
class ForecastWindows:
    """Per-day scenario P&L windows (rows = forecast dates, cols = oldest->newest scenario)."""

    dates: pd.DatetimeIndex
    windows: np.ndarray
    actual_pnl: pd.Series
    nav_prev: pd.Series
    lookback: int
    # Scenario P&L windows of the simulation methods, built on demand and memoised
    # (key includes the parameters they depend on).
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    vol_factors: str = "LEVEL_SKEW_TERM"  # implied-vol factor model behind every window


class BacktestService:
    def __init__(self, settings: AppSettings, repo: RiskRepository | None = None):
        self.settings = settings
        self.repo = repo

    def forecast_windows(self, ctx: RiskContext, lookback: int, vol_factors: str = "LEVEL_SKEW_TERM") -> ForecastWindows:
        """Scenario P&L windows and realised P&L for every day with enough history.

        Args:
            vol_factors: implied-vol factor model used in the scenarios
                (``LEVEL`` or ``LEVEL_SKEW_TERM``).
        """
        hist = ctx.history
        frame = select_vol_factors(hist.factor_frame(), vol_factors)
        values, columns = frame.to_numpy(dtype=float), [str(c) for c in frame.columns]
        end = hist.index_of(ctx.valuation_date)
        first = lookback + 1  # need `lookback` moves ending on t-1, i.e. rows 1..t-1
        if first > end:
            raise ValueError(f"Insufficient historical observations for a {lookback}-day backtest window.")
        dates, windows, actual, nav_prev = [], [], [], []
        pricer = ctx.pricer
        prev_value = pricer.value(hist.state_at(hist.dates[first - 1].date()))
        for i in range(first, end + 1):
            d_prev, d = hist.dates[i - 1].date(), hist.dates[i].date()
            base = hist.state_at(d_prev)
            a, b = i - lookback - 1, i - 1  # factor-frame rows = moves into history days i-lookback .. i-1
            labels = frame.index[a:b]
            if labels[-1] >= hist.dates[i]:  # pragma: no cover - defensive
                raise AssertionError("Look-ahead detected in backtest scenarios")
            scen = ScenarioSet(labels, {c: values[a:b, j] for j, c in enumerate(columns)})
            windows.append(pricer.scenario_pnl(base, scen).sum(axis=1).to_numpy())
            value = pricer.value(hist.state_at(d))
            cash = pricer.coupon_cash(d_prev, d)
            actual.append(float(value.sum() - prev_value.sum() + cash.sum()))
            nav_prev.append(float(prev_value.sum()))
            dates.append(hist.dates[i])
            prev_value = value
        idx = pd.DatetimeIndex(dates)
        logger.info("VaR backtest forecasts built: %d days, lookback %d", len(idx), lookback)
        return ForecastWindows(
            idx, np.vstack(windows), pd.Series(actual, index=idx), pd.Series(nav_prev, index=idx), lookback, vol_factors=vol_factors
        )

    # ----------------------------------------------------- simulation windows
    def _simulation_key(self, method: str, lam: float, cov_method: str) -> str:
        sim = self.settings.simulation
        if method == "FILTERED_HISTORICAL":
            return f"FILTERED_HISTORICAL|{lam}|{sim.fhs_ratio_floor}|{sim.fhs_ratio_cap}"
        return f"MONTE_CARLO|{cov_method}|{lam}|{sim.backtest_paths}|{sim.distribution}|{sim.degrees_of_freedom}|{sim.seed}"

    def simulation_windows(self, ctx: RiskContext, fw: ForecastWindows, method: str, lam: float, cov_method: str = "EWMA") -> np.ndarray:
        """Scenario P&L windows for filtered historical simulation or Monte Carlo.

        For forecast day t (history index i) both methods only use factor moves
        dated on or before t-1:

        * filtered HS rescales the window's moves by ``sigma_t / sigma_s``, where
          every sigma is an EWMA forecast built from earlier moves;
        * Monte Carlo estimates the factor covariance from the window and
          simulates with common random numbers (the same standard draws every
          day), so the VaR path reflects changes in covariance and market
          state rather than sampling noise.
        """
        key = self._simulation_key(method, lam, cov_method)
        if key in fw.extra:
            return fw.extra[key]
        sim = self.settings.simulation
        hist, pricer, lookback = ctx.history, ctx.pricer, fw.lookback
        frame = select_vol_factors(hist.factor_frame(), fw.vol_factors)
        values = frame.to_numpy(dtype=float)
        columns = [str(c) for c in frame.columns]
        is_ret = return_mask(frame.columns)
        if method == "FILTERED_HISTORICAL":
            vols = ewma_vol_forecasts(frame, lam, sim.fhs_seed_observations)
            bounds = (sim.fhs_ratio_floor, sim.fhs_ratio_cap)
        else:
            draws = standard_draws(sim.backtest_paths, len(columns), sim.seed, sim.distribution, sim.degrees_of_freedom, sim.antithetic)
            ew = (1.0 - lam) * lam ** np.arange(lookback - 1, -1, -1, dtype=float)
        out = []
        for ts in fw.dates:
            i = int(hist.dates.get_loc(ts))
            a, b = i - lookback - 1, i - 1  # factor-frame rows: moves into history days i-lookback .. i-1
            window = values[a:b]
            if method == "FILTERED_HISTORICAL":
                moves = window * scaling_ratios(vols, a, b, bounds)
                labels: pd.Index = frame.index[a:b]
            else:
                cov = np.cov(window, rowvar=False, ddof=1)
                if cov_method == "EWMA":
                    cov = (window * ew[:, None]).T @ window + lam**lookback * cov
                moves = draws @ covariance_root(cov).T
                labels = pd.RangeIndex(len(draws))
            moves[:, is_ret] = np.maximum(moves[:, is_ret], RETURN_FLOOR)
            scen = ScenarioSet(labels, {c: moves[:, j] for j, c in enumerate(columns)})
            out.append(pricer.scenario_pnl(hist.state_at(hist.dates[i - 1].date()), scen).sum(axis=1).to_numpy())
        fw.extra[key] = np.vstack(out)
        logger.info("%s backtest windows built: %d days x %d scenarios", method, len(out), fw.extra[key].shape[1])
        return fw.extra[key]

    def var_series(
        self, fw: ForecastWindows, method: str, confidence: float, lam: float = 0.94,
        ctx: RiskContext | None = None, cov_method: str = "EWMA",
    ) -> pd.Series:
        """Predicted 1-day VaR for each forecast date under ``method``.

        The simulation methods need ``ctx`` the first time (to build their
        scenario windows); afterwards the windows are memoised on ``fw``.
        """
        w = fw.windows
        if method in ("FILTERED_HISTORICAL", "MONTE_CARLO"):
            key = self._simulation_key(method, lam, cov_method)
            if key not in fw.extra:
                if ctx is None:
                    raise ValueError(f"{method} backtest windows have not been built; pass the risk context.")
                self.simulation_windows(ctx, fw, method, lam, cov_method)
            return pd.Series(historical_var_rows(fw.extra[key], confidence), index=fw.dates, name="var")
        if method == "HISTORICAL":
            v = historical_var_rows(w, confidence)
        elif method == "EW_HISTORICAL":
            v = weighted_historical_var_rows(w, confidence, lam)
        elif method == "PARAMETRIC":
            v = parametric_var_rows(w.std(axis=1, ddof=1), confidence)
        elif method == "EW_PARAMETRIC":
            # EWMA variance of portfolio P&L, seeded with the window's sample variance
            n = w.shape[1]
            wt = (1 - lam) * ewma_weights(n, lam, normalise=False)
            var = (w**2) @ wt + lam**n * w.var(axis=1, ddof=1)
            v = parametric_var_rows(np.sqrt(var), confidence)
        else:
            raise ValueError(f"Unknown backtest method {method}")
        return pd.Series(v, index=fw.dates, name="var")

    def run(
        self,
        ctx: RiskContext,
        fw: ForecastWindows,
        method: str,
        confidence: float,
        start: date | None = None,
        end: date | None = None,
        lam: float = 0.94,
        persist: bool = True,
        cov_method: str = "EWMA",
    ) -> BacktestResult:
        var = self.var_series(fw, method, confidence, lam, ctx, cov_method)
        actual = fw.actual_pnl
        mask = pd.Series(True, index=fw.dates)
        if start:
            mask &= fw.dates >= pd.Timestamp(start)
        if end:
            mask &= fw.dates <= pd.Timestamp(end)
        result = run_backtest(actual[mask], var[mask], confidence, fw.lookback, method, self.settings.backtest.significance_level)
        logger.info(
            "VaR backtest completed: %s %.0f%% - %d exceptions in %d days (Kupiec p=%.3f)",
            method, confidence * 100, result.kupiec.n_exceptions, result.kupiec.n_observations, result.kupiec.p_value,
        )
        if persist:
            self._persist(ctx, result)
        return result

    def basel_assessment(
        self, fw: ForecastWindows, method: str, lam: float = 0.94, end: date | None = None,
        ctx: RiskContext | None = None, cov_method: str = "EWMA",
    ) -> BaselResult:
        """Basel traffic light for ``method``: 99% VaR exceptions over the trailing 250 days.

        The regulatory test is always run at 99% on the most recent 250
        observations (up to ``end``), whatever confidence the user is viewing.
        Fewer than 250 available days still yields a zone (binomial rule) but
        no plus factor.
        """
        var = self.var_series(fw, method, BASEL_CONFIDENCE, lam, ctx, cov_method)
        exc = count_exceptions(fw.actual_pnl, var)
        if end is not None:
            exc = exc[exc.index <= pd.Timestamp(end)]
        exc = exc.iloc[-BASEL_WINDOW:]
        if exc.empty:
            raise ValueError("Insufficient historical observations for the Basel traffic-light test.")
        return basel_traffic_light(len(exc), int(exc.sum()))

    def rolling_risk(
        self,
        fw: ForecastWindows,
        confidence: float,
        lam: float,
        methods: tuple[str, ...] = ("HISTORICAL", "EW_HISTORICAL", "PARAMETRIC", "EW_PARAMETRIC"),
        ctx: RiskContext | None = None,
        cov_method: str = "EWMA",
        periods: int = 252,
        vol_seed_obs: int = 20,
    ) -> pd.DataFrame:
        """Out-of-sample rolling VaR and realised volatility of the static portfolio.

        Columns: ``pnl``, ``return``, ``var_<METHOD>`` and ``vol_<n>d`` / ``vol_ewma``
        (annualised, as a fraction of NAV). Every value at date t uses data up to t
        (VaR and the EWMA volatility forecast use data up to t-1; the EWMA recursion
        is seeded with the first ``vol_seed_obs`` returns only).
        """
        out = pd.DataFrame({"pnl": fw.actual_pnl, "return": fw.actual_pnl / fw.nav_prev})
        for m in methods:
            out[f"var_{m}"] = self.var_series(fw, m, confidence, lam, ctx, cov_method)
        for n in self.settings.risk.rolling_vol_windows:
            out[f"vol_{n}d"] = rolling_volatility(out["return"], n, periods=periods)
        seed_var = float(np.mean(out["return"].iloc[:vol_seed_obs] ** 2))
        out["vol_ewma"] = np.sqrt(ewma_variance_series(out["return"], lam, seed_var)) * np.sqrt(periods)
        return out

    def _persist(self, ctx: RiskContext, result: BacktestResult) -> None:
        if self.repo is None or ctx.portfolio.portfolio_id.startswith("UPLOAD"):
            return
        rows = [
            {
                "portfolio_id": ctx.portfolio.portfolio_id, "method": result.method, "confidence": result.confidence,
                "lookback": result.lookback, "date": ts.date(), "predicted_var": float(var),
                "actual_pnl": float(pnl), "exception": bool(exc),
            }
            for ts, var, pnl, exc in zip(
                result.frame.index, result.frame["var"], result.frame["actual_pnl"], result.frame["exception"]
            )
        ]
        try:
            self.repo.save_backtest_results(rows)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist backtest results")
