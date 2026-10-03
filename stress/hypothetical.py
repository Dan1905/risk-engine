"""Hypothetical scenario builders (single-factor ladders and custom scenarios)."""

from __future__ import annotations

from config.settings import StressSettings
from models.risk import ShockBasis, StressScenario
from stress.volatility_shocks import vol_shock_scenario
from stress.yield_curve_shocks import BP, flattener, inversion, parallel_shift, steepener

ASSUMED = ShockBasis.ASSUMED


def equity_shock_scenario(shock: float) -> StressScenario:
    return StressScenario(
        scenario_id=f"EQ_{shock * 100:+.0f}",
        name=f"Equities {shock * 100:+.0f}%",
        category="EQUITY",
        description=f"All equities and option underlyings move {shock * 100:+.0f}%.",
        equity_shock=shock,
        shock_basis={"equity": ASSUMED},
    )


def rate_shock_scenario(bps: float) -> StressScenario:
    return StressScenario(
        scenario_id=f"RATES_{bps:+.0f}",
        name=f"Rates {bps:+.0f}bp",
        category="RATES",
        description=f"Parallel {bps:+.0f} bp shift of the Treasury curve.",
        rates_shock=bps * BP,
        shock_basis={"rates": ASSUMED},
    )


def fx_shock_scenario(currency: str, shock: float) -> StressScenario:
    """``USD +5%`` strengthens USD vs all currencies; ``EUR +10%`` lifts EUR vs USD."""
    ccy = currency.upper()
    kwargs = {"fx_shock": shock} if ccy == "USD" else {"fx_shocks": {ccy: shock}}
    return StressScenario(
        scenario_id=f"FX_{ccy}_{shock * 100:+.0f}",
        name=f"{ccy} {shock * 100:+.0f}%",
        category="FX",
        description=f"{ccy} moves {shock * 100:+.0f}% against {'all currencies' if ccy == 'USD' else 'USD'}.",
        shock_basis={"fx": ASSUMED},
        **kwargs,
    )


def curve_scenarios() -> list[StressScenario]:
    """Parallel ladder is built separately; these are the twist shapes."""
    shapes = (("STEEPENER", "Steepener", steepener()), ("FLATTENER", "Flattener", flattener()), ("INVERSION", "Inversion", inversion()))
    return [
        StressScenario(scenario_id=f"CURVE_{sid}", name=name, category="YIELD_CURVE", curve_shocks=shifts, shock_basis={"rates": ASSUMED})
        for sid, name, shifts in shapes
    ]


def parallel_curve_scenarios(ladder_bps: list[float]) -> list[StressScenario]:
    return [
        StressScenario(
            scenario_id=f"CURVE_PAR_{b:+.0f}",
            name=f"Parallel {b:+.0f}bp",
            category="YIELD_CURVE",
            curve_shocks=parallel_shift(b),
            shock_basis={"rates": ASSUMED},
        )
        for b in ladder_bps
    ]


def equity_ladder(settings: StressSettings) -> list[StressScenario]:
    return [equity_shock_scenario(s) for s in settings.equity_shock_ladder]


def fx_ladder(settings: StressSettings) -> list[StressScenario]:
    return [fx_shock_scenario(f.currency, f.shock) for f in settings.fx_shocks]


def vol_ladder(settings: StressSettings) -> list[StressScenario]:
    return [vol_shock_scenario(v) for v in settings.vol_shock_ladder]


def custom_scenario(
    name: str,
    *,
    equity_shock: float = 0.0,
    rates_shock_bps: float = 0.0,
    fx_shock: float = 0.0,
    volatility_shock: float = 0.0,
    credit_spread_shock_bps: float = 0.0,
    equity_overrides: dict[str, float] | None = None,
    fx_overrides: dict[str, float] | None = None,
    yield_overrides_bps: dict[str, float] | None = None,
    vol_overrides: dict[str, float] | None = None,
) -> StressScenario:
    """Scenario from the interactive builder (all inputs are hypothetical assumptions)."""
    sid = "CUSTOM_" + "".join(ch if ch.isalnum() else "_" for ch in name.upper())[:40]
    basis = {k: ASSUMED for k in ("equity", "rates", "fx", "volatility", "credit")}
    return StressScenario(
        scenario_id=sid,
        name=name or "Custom scenario",
        category="CUSTOM",
        description="User-defined scenario from the Scenario Builder.",
        equity_shock=equity_shock,
        rates_shock=rates_shock_bps * BP,
        fx_shock=fx_shock,
        fx_shocks=dict(fx_overrides or {}),
        volatility_shock=volatility_shock,
        credit_spread_shock=credit_spread_shock_bps * BP,
        equity_overrides=dict(equity_overrides or {}),
        yield_overrides={k: v * BP for k, v in (yield_overrides_bps or {}).items()},
        vol_overrides=dict(vol_overrides or {}),
        shock_basis=basis,
    )
