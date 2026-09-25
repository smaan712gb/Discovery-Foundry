"""Validated config for the NQ domain. Every tunable number lives in config/*.yaml."""

from __future__ import annotations

import re
from datetime import date, time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator

from foundry.core.budgets import BudgetConfig
from foundry.core.config import StrictModel
from foundry.critics.config import CriticConfig
from foundry.search.llm_client import LLMConfig

EXCHANGE_TZ = "America/New_York"
EMBEDDED_TZ = "embedded"


def _check_tz(value: str) -> str:
    if value == EMBEDDED_TZ:
        return value
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone: {value!r}") from exc
    return value


class ColumnMap(StrictModel):
    timestamp: str
    open: str
    high: str
    low: str
    close: str
    volume: str
    contract: str | None = None


class SourceConfig(StrictModel):
    """One family of raw files. Timezone and bar-label convention have no defaults."""

    name: str
    format: Literal["ninjatrader_txt", "csv", "parquet"]
    directory: Path
    filename_regex: str
    timezone: str
    timestamp_label: Literal["start", "end"]
    # How to tell which contract a file holds: named groups `month` and `year` in
    # filename_regex, a contract column, or a series that is already continuous.
    contract_from: Literal["filename", "column", "continuous"] = "filename"
    columns: ColumnMap | None = None
    # strftime format, or "unix_s" / "unix_ms"; None lets Polars infer ISO-8601.
    timestamp_format: str | None = None
    csv_separator: str = ","

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        return _check_tz(v)

    @field_validator("filename_regex")
    @classmethod
    def _regex(cls, v: str) -> str:
        re.compile(v)
        return v

    @model_validator(mode="after")
    def _consistent(self) -> SourceConfig:
        groups = re.compile(self.filename_regex).groupindex
        if self.contract_from == "filename" and not {"month", "year"} <= set(groups):
            raise ValueError("contract_from=filename needs named groups 'month' and 'year'")
        if self.format != "ninjatrader_txt" and self.columns is None:
            raise ValueError(f"format {self.format} needs a 'columns' map")
        if self.contract_from == "column" and (self.columns is None or not self.columns.contract):
            raise ValueError("contract_from=column needs columns.contract")
        if self.timezone == EMBEDDED_TZ and self.format == "ninjatrader_txt":
            raise ValueError("ninjatrader_txt timestamps carry no offset; give an IANA timezone")
        return self


class InstrumentConfig(StrictModel):
    root: str = Field(min_length=1)
    tick_size: float = Field(gt=0)
    bar_minutes: int = Field(ge=1)


class SessionConfig(StrictModel):
    rth_open: time
    rth_close: time
    globex_close: time
    globex_open: time

    @model_validator(mode="after")
    def _order(self) -> SessionConfig:
        if not (self.rth_open < self.rth_close <= self.globex_close < self.globex_open):
            raise ValueError("need rth_open < rth_close <= globex_close < globex_open")
        return self


class RollConfig(StrictModel):
    rule: Literal["volume_crossover"]


class QualityConfig(StrictModel):
    bad_tick_atr_period: int = Field(ge=2)
    bad_tick_atr_multiple: float = Field(gt=0)
    max_break_fraction: float = Field(ge=0, le=1)
    top_gaps_reported: int = Field(ge=1)


class DateRange(StrictModel):
    start: date
    end: date

    @model_validator(mode="after")
    def _order(self) -> DateRange:
        if self.start > self.end:
            raise ValueError(f"range start {self.start} is after end {self.end}")
        return self


class SplitConfig(StrictModel):
    search: DateRange
    validation: DateRange
    holdout_start: date
    min_trading_days_search: int = Field(ge=1)
    min_trading_days_validation: int = Field(ge=1)

    @model_validator(mode="after")
    def _order(self) -> SplitConfig:
        if not self.search.end < self.validation.start:
            raise ValueError("validation must start after search ends")
        if not self.validation.end < self.holdout_start:
            raise ValueError("holdout must start after validation ends")
        return self


class DataConfig(StrictModel):
    instrument: InstrumentConfig
    sources: list[SourceConfig] = Field(min_length=1)
    calendar_file: Path
    sessions: SessionConfig
    roll: RollConfig
    quality: QualityConfig
    splits: SplitConfig
    processed_dir: Path


class HoldoutConfig(StrictModel):
    directory: Path


class ContractSpec(StrictModel):
    tick_size: float = Field(gt=0)
    point_value: float = Field(gt=0)
    commission_round_turn: float = Field(ge=0)


class SlippageConfig(StrictModel):
    market: int = Field(ge=0)
    stop: int = Field(ge=0)
    limit: int = Field(ge=0)


class RegimeConfig(StrictModel):
    vol_lookback_days: int = Field(ge=10)
    trend_days: int = Field(ge=2)


class EvaluatorConfig(StrictModel):
    """Backtest semantics are fixed in ADR 0007; these are its numbers."""

    contracts: dict[str, ContractSpec]
    default_contract: str
    contracts_per_trade: int = Field(ge=1)
    slippage_ticks: SlippageConfig
    limit_trade_through_ticks: int = Field(ge=0)
    annualization_days: int = Field(ge=1)
    block_signals_on_flags: list[
        Literal["zero_volume", "bad_tick", "ohlc_invalid", "off_grid", "outside_session"]
    ]
    regimes: RegimeConfig
    events_file: Path
    results_db: Path

    @model_validator(mode="after")
    def _default_known(self) -> EvaluatorConfig:
        if self.default_contract not in self.contracts:
            raise ValueError(f"default_contract {self.default_contract!r} not in contracts")
        return self


class SearchConfig(StrictModel):
    engine_file: Path
    budgets: BudgetConfig
    llm: LLMConfig


class FoundryConfig(StrictModel):
    """Top-level config. The meta layer (phase 5) adds its own section."""

    seed: int
    data: DataConfig
    holdout: HoldoutConfig
    evaluator: EvaluatorConfig
    critics: CriticConfig
    search: SearchConfig


class EventEntry(StrictModel):
    date: date
    kind: str
    note: str = ""


class EventsFile(StrictModel):
    name: str
    entries: list[EventEntry]


# ---- calendar file ---------------------------------------------------------------------


class CalendarEntry(StrictModel):
    date: date
    kind: Literal["closed", "holiday", "early_close"]
    note: str
    rth_close: time | None = None
    globex_close: time | None = None


class CalendarDefaults(StrictModel):
    holiday_globex_close: time
    early_close_rth_close: time
    early_close_globex_close: time


class CalendarFile(StrictModel):
    name: str
    covers_from: date
    covers_to: date
    defaults: CalendarDefaults
    entries: list[CalendarEntry]

    @model_validator(mode="after")
    def _unique(self) -> CalendarFile:
        seen: set[date] = set()
        for e in self.entries:
            if e.date in seen:
                raise ValueError(f"duplicate calendar entry for {e.date}")
            if not (self.covers_from <= e.date <= self.covers_to):
                raise ValueError(f"calendar entry {e.date} outside covered range")
            seen.add(e.date)
        return self
