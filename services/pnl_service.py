"""P&L attribution service (daily and over a window)."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from analytics.pnl_attribution import AttributionResult, attribute_pnl, attribution_series
from config.settings import AppSettings
from database.repository import RiskRepository
from services.risk_service import RiskContext

logger = logging.getLogger(__name__)


class PnLService:
    def __init__(self, settings: AppSettings, repo: RiskRepository | None = None):
        self.settings = settings
        self.repo = repo

    def attribution(self, ctx: RiskContext, as_of: date | None = None) -> AttributionResult:
        """Attribute the P&L from the previous trading day to ``as_of`` (default valuation date)."""
        d = as_of or ctx.valuation_date
        i = ctx.history.index_of(d)
        if i == 0:
            raise ValueError("No prior trading day available for P&L attribution.")
        prev = ctx.history.dates[i - 1].date()
        return attribute_pnl(ctx.pricer, ctx.history.state_at(prev), ctx.history.state_at(d))

    def attribution_history(self, ctx: RiskContext, n_days: int | None = None) -> tuple[list[AttributionResult], pd.DataFrame]:
        """Daily attributions for the last ``n_days`` trading days up to the valuation date."""
        n = n_days or self.settings.pnl.attribution_days
        end = ctx.history.index_of(ctx.valuation_date)
        start = max(1, end - n + 1)
        results = [
            attribute_pnl(ctx.pricer, ctx.history.state_at(ctx.history.dates[i - 1].date()), ctx.history.state_at(ctx.history.dates[i].date()))
            for i in range(start, end + 1)
        ]
        frame = attribution_series(results)
        tol = self.settings.pnl.tolerance
        breaks = [r.date_to.date() for r in results if not r.reconciles(tol)]
        if breaks:
            logger.error("P&L attribution does not reconcile within tolerance on %s", breaks)
        self._persist(ctx, frame)
        return results, frame

    def _persist(self, ctx: RiskContext, frame: pd.DataFrame) -> None:
        if self.repo is None or ctx.portfolio.portfolio_id.startswith("UPLOAD") or frame.empty:
            return
        long = frame.reset_index().melt(id_vars="date", var_name="component", value_name="amount")
        rows = [
            {"portfolio_id": ctx.portfolio.portfolio_id, "date": pd.Timestamp(r.date).date(), "component": r.component, "amount": float(r.amount)}
            for r in long.itertuples()
        ]
        try:
            self.repo.save_pnl_attribution(rows)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to persist P&L attribution")
