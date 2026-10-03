"""Pydantic schemas for risk configuration, stress scenarios and risk results."""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VarMethod = Literal["HISTORICAL", "EW_HISTORICAL", "PARAMETRIC", "EW_PARAMETRIC", "FILTERED_HISTORICAL", "MONTE_CARLO"]
CovarianceMethod = Literal["SAMPLE", "EWMA"]
HorizonMethod = Literal["SQRT_TIME", "OVERLAPPING"]
VolFactorModel = Literal["LEVEL", "LEVEL_SKEW_TERM"]
VolDynamics = Literal["STICKY_STRIKE", "STICKY_DELTA"]
BondRepricing = Literal["FULL_REPRICING", "SENSITIVITY"]


class RiskConfiguration(BaseModel):
    """User-selectable VaR / ES parameters."""

    model_config = ConfigDict(frozen=True)

    confidence: float = 0.99
    lookback: int = 250
    holding_period: int = 1
    decay_factor: float = 0.94
    covariance_method: CovarianceMethod = "EWMA"
    return_type: Literal["log", "simple"] = "log"
    horizon_method: HorizonMethod = "OVERLAPPING"
    vol_factors: VolFactorModel = "LEVEL_SKEW_TERM"

    @field_validator("confidence")
    @classmethod
    def _conf(cls, v: float) -> float:
        if not 0.5 < v < 1.0:
            raise ValueError("Confidence must be between 50% and 100% (exclusive).")
        return v

    @field_validator("lookback")
    @classmethod
    def _lookback(cls, v: int) -> int:
        if v < 20:
            raise ValueError("Lookback must be at least 20 observations.")
        return v

    @field_validator("holding_period")
    @classmethod
    def _hp(cls, v: int) -> int:
        if not 1 <= v <= 250:
            raise ValueError("Holding period must be between 1 and 250 days.")
        return v

    @field_validator("decay_factor")
    @classmethod
    def _decay(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("Decay factor must be strictly between 0 and 1.")
        return v


class ShockBasis(str, Enum):
    """Whether a scenario input is an observed historical move or an assumption."""

    OBSERVED = "HISTORICAL_OBSERVATION"
    ASSUMED = "HYPOTHETICAL_ASSUMPTION"


class StressScenario(BaseModel):
    """A deterministic market scenario.

    Sign conventions
    ----------------
    equity_shock          simple return applied to every equity / option underlying
    rates_shock           parallel shift of every curve node (decimal; 0.015 = +150 bp)
    curve_shocks          per-tenor shifts (decimal), added on top of ``rates_shock``
    fx_shock              USD move vs every foreign currency: +0.10 = USD strengthens 10%
    fx_shocks             currency-specific moves vs USD: {"EUR": 0.10} = EUR +10%;
                          the key "USD" behaves like ``fx_shock``
    volatility_shock      absolute implied-vol shift (0.15 = +15 vol points)
    credit_spread_shock   parallel shift of credit spread curves (decimal)
    credit_curve_shocks   per-tenor spread shifts (decimal), added on top of ``credit_spread_shock``
    equity_beta_mapping   scale ``equity_shock`` by each name's beta to the equity market
                          (shock_i = beta_i x equity_shock) instead of applying it uniformly
    vol_skew_shocks       per-underlying change of the 3M 90%-110% implied-vol skew
    vol_term_shocks       per-underlying change of the 1Y-1M ATM implied-vol slope
    *_overrides           instrument-specific values that replace the uniform shock
    """

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    name: str
    category: Literal["HISTORICAL", "HYPOTHETICAL", "YIELD_CURVE", "EQUITY", "RATES", "FX", "VOLATILITY", "CUSTOM"] = (
        "HYPOTHETICAL"
    )
    description: str = ""
    window: str | None = None
    equity_shock: float = 0.0
    rates_shock: float = 0.0
    curve_shocks: dict[str, float] = Field(default_factory=dict)
    fx_shock: float = 0.0
    fx_shocks: dict[str, float] = Field(default_factory=dict)
    volatility_shock: float = 0.0
    credit_spread_shock: float = 0.0
    credit_curve_shocks: dict[str, float] = Field(default_factory=dict)
    equity_beta_mapping: bool = False
    equity_overrides: dict[str, float] = Field(default_factory=dict)
    yield_overrides: dict[str, float] = Field(default_factory=dict)
    vol_overrides: dict[str, float] = Field(default_factory=dict)
    vol_skew_shocks: dict[str, float] = Field(default_factory=dict)
    vol_term_shocks: dict[str, float] = Field(default_factory=dict)
    shock_basis: dict[str, ShockBasis] = Field(default_factory=dict)
    source_notes: str = ""

    @field_validator("equity_shock", "fx_shock", "volatility_shock")
    @classmethod
    def _gt_minus_one(cls, v: float) -> float:
        if not math.isfinite(v) or v <= -1.0:
            raise ValueError("Shock must be finite and greater than -100%.")
        return v

    @field_validator("rates_shock", "credit_spread_shock")
    @classmethod
    def _rate_bounds(cls, v: float) -> float:
        if not math.isfinite(v) or abs(v) > 0.20:
            raise ValueError("Rate / spread shocks must be within +/- 2000 bp.")
        return v

    @model_validator(mode="after")
    def _overrides(self) -> "StressScenario":
        for name, mapping in (("equity_overrides", self.equity_overrides), ("fx_shocks", self.fx_shocks)):
            for key, val in mapping.items():
                if not math.isfinite(val) or val <= -1.0:
                    raise ValueError(f"{name}[{key}] must be greater than -100%.")
        for name, mapping in (("curve_shocks", self.curve_shocks), ("credit_curve_shocks", self.credit_curve_shocks)):
            for key, val in mapping.items():
                if not math.isfinite(val) or abs(val) > 0.20:
                    raise ValueError(f"{name}[{key}] must be within +/- 2000 bp.")
        return self

    def basis_for(self, component: str) -> str:
        basis = self.shock_basis.get(component)
        return basis.value if basis else ShockBasis.ASSUMED.value


class RiskResult(BaseModel):
    """Headline output of one VaR / ES calculation."""

    model_config = ConfigDict(frozen=True)

    method: VarMethod
    confidence: float
    lookback: int
    holding_period: int
    decay_factor: float | None = None
    portfolio_value: float
    var: float
    expected_shortfall: float
    volatility: float = Field(description="1-day portfolio P&L standard deviation (USD)")
    n_observations: int
    worst_loss: float | None = None
    best_gain: float | None = None
    mean_pnl: float | None = None

    @property
    def var_pct(self) -> float:
        return self.var / self.portfolio_value if self.portfolio_value else float("nan")

    @property
    def es_pct(self) -> float:
        return self.expected_shortfall / self.portfolio_value if self.portfolio_value else float("nan")

    @property
    def es_var_ratio(self) -> float:
        return self.expected_shortfall / self.var if self.var else float("nan")


class LimitCheck(BaseModel):
    """Result of checking one metric against its configured limit."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float
    limit: float
    status: Literal["Within Limit", "Breach"]

    @property
    def utilisation(self) -> float:
        return abs(self.value) / self.limit if self.limit else float("nan")
