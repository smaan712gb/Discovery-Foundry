"""Critic thresholds. Fixed by config; the meta layer can never change them (ADR 0008)."""

from __future__ import annotations

from pydantic import Field, model_validator

from foundry.core.config import StrictModel


class MinTrades(StrictModel):
    search: int = Field(ge=1)
    validation: int = Field(ge=1)


class CPCVConfig(StrictModel):
    groups: int = Field(ge=3)
    test_groups: int = Field(ge=1)
    embargo_days: int = Field(ge=0)
    min_positive_fraction: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _order(self) -> CPCVConfig:
        if self.test_groups >= self.groups:
            raise ValueError("test_groups must be smaller than groups")
        return self


class ConcentrationConfig(StrictModel):
    top_day_fraction: float = Field(gt=0, lt=1)
    max_profit_share: float = Field(gt=0, le=1)


class CriticConfig(StrictModel):
    min_trades: MinTrades
    cost_stress_multiplier: float = Field(gt=1)
    concentration: ConcentrationConfig
    min_profitable_regimes: int = Field(ge=1, le=3)
    dsr_threshold: float = Field(gt=0, lt=1)
    perturbations: list[float] = Field(min_length=1)
    pbo_threshold: float = Field(gt=0, lt=1)
    pbo_min_trials: int = Field(ge=2)
    cscv_blocks: int = Field(ge=2)
    cpcv: CPCVConfig

    @model_validator(mode="after")
    def _checks(self) -> CriticConfig:
        if self.cscv_blocks % 2:
            raise ValueError("cscv_blocks must be even")
        if any(p == 0 or abs(p) >= 1 for p in self.perturbations):
            raise ValueError("perturbations must be non-zero fractions in (-1, 1)")
        return self
