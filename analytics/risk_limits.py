"""Risk-limit checks and data-driven concentration warnings.

All thresholds come from configuration (``limits`` and ``concentration``
sections of ``config/risk_config.yaml``); nothing here is hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config.settings import ConcentrationSettings, LimitSettings
from models.risk import LimitCheck


def _check(name: str, value: float, limit: float) -> LimitCheck:
    return LimitCheck(name=name, value=value, limit=limit, status="Breach" if abs(value) > limit else "Within Limit")


def evaluate_limits(
    var_pct: float, es_pct: float, equity_pct: float, largest_position_pct: float, limits: LimitSettings
) -> list[LimitCheck]:
    """Compare headline metrics (fractions of NAV) with configured limits."""
    return [
        _check("VaR", var_pct, limits.var_limit),
        _check("ES", es_pct, limits.es_limit),
        _check("Equity", equity_pct, limits.equity_exposure_limit),
        _check("Single Position", largest_position_pct, limits.single_position_limit),
    ]


@dataclass
class ConcentrationReport:
    largest_position: tuple[str, float]
    largest_asset_class: tuple[str, float]
    largest_currency: tuple[str, float]
    largest_var_contributor: tuple[str, float] | None
    largest_stress_contributor: tuple[str, float, str] | None
    warnings: list[str] = field(default_factory=list)


def concentration_report(
    valuation: pd.DataFrame,
    component_var: pd.Series | None,
    worst_stress_pnl: pd.Series | None,
    worst_stress_name: str | None,
    limits: LimitSettings,
    thresholds: ConcentrationSettings,
) -> ConcentrationReport:
    """Identify the largest exposures / contributors and emit threshold-based warnings.

    Args:
        valuation: position valuation table (``position_id``, ``ticker``,
            ``asset_class``, ``exposure_currency``, ``market_value``).
        component_var: component VaR by position id (sums to total VaR).
        worst_stress_pnl: position P&L in the worst stress scenario.
    """
    v = valuation.set_index("position_id")
    nav = float(v["market_value"].sum())
    gross = float(v["market_value"].abs().sum())
    tickers = v["ticker"].to_dict()
    warnings: list[str] = []

    pos_share = v["market_value"].abs() / nav
    top_pos = str(pos_share.idxmax())
    largest_position = (top_pos, float(pos_share.max()))
    if largest_position[1] > limits.single_position_limit:
        warnings.append(
            f"{tickers[top_pos]} ({top_pos}) is {largest_position[1]:.1%} of NAV, above the "
            f"{limits.single_position_limit:.0%} single-position limit."
        )

    by_class = v.groupby("asset_class")["market_value"].sum() / nav
    top_class = str(by_class.abs().idxmax())
    largest_asset_class = (top_class, float(by_class[top_class]))
    equity_share = float(by_class.get("Equity", 0.0))
    if equity_share > limits.equity_exposure_limit:
        warnings.append(f"Equity exposure ({equity_share:.1%} of NAV) exceeds the configured {limits.equity_exposure_limit:.0%} limit.")
    if abs(largest_asset_class[1]) > thresholds.asset_class_concentration_warning:
        warnings.append(f"{top_class} represents {largest_asset_class[1]:.1%} of NAV.")

    by_ccy = v.groupby("exposure_currency")["market_value"].apply(lambda s: s.abs().sum()) / gross
    top_ccy = str(by_ccy.idxmax())
    largest_currency = (top_ccy, float(by_ccy.max()))
    if largest_currency[1] > thresholds.currency_concentration_warning:
        warnings.append(f"{top_ccy} exposure represents {largest_currency[1]:.0%} of gross currency exposure.")

    largest_var = None
    if component_var is not None and component_var.sum() > 0:
        share = component_var / component_var.sum()
        top = str(share.idxmax())
        largest_var = (top, float(share.max()))
        if largest_var[1] > thresholds.var_contribution_warning:
            warnings.append(f"{tickers.get(top, top)} contributes {largest_var[1]:.0%} of total portfolio VaR.")

    largest_stress = None
    if worst_stress_pnl is not None and worst_stress_pnl.sum() < 0:
        losses = worst_stress_pnl.clip(upper=0.0)
        share = losses / losses.sum()
        top = str(share.idxmax())
        largest_stress = (top, float(share.max()), worst_stress_name or "")
        if largest_stress[1] > thresholds.stress_loss_contribution_warning:
            warnings.append(
                f"{tickers.get(top, top)} accounts for {largest_stress[1]:.0%} of losses in the {worst_stress_name} scenario."
            )

    return ConcentrationReport(largest_position, largest_asset_class, largest_currency, largest_var, largest_stress, warnings)


def stress_limit_checks(
    market_values: pd.Series, position_pnl: pd.Series, asset_classes: pd.Series, limits: LimitSettings
) -> list[LimitCheck]:
    """Limit status after a stress: scenario loss vs the stress-loss limit, plus
    equity and single-position concentration on the stressed book."""
    nav = float(market_values.sum())
    stressed = market_values.add(position_pnl.reindex(market_values.index).fillna(0.0))
    nav_s = float(stressed.sum())
    loss = max(-float(position_pnl.sum()) / nav, 0.0) if nav else float("nan")
    equity = float(stressed[asset_classes.reindex(stressed.index) == "Equity"].sum()) / nav_s if nav_s else float("nan")
    largest = float((stressed.abs() / nav_s).max()) if nav_s else float("nan")
    return [
        _check("Stress Loss", loss, limits.stress_loss_limit),
        _check("Equity (stressed)", equity, limits.equity_exposure_limit),
        _check("Single Position (stressed)", largest, limits.single_position_limit),
    ]
