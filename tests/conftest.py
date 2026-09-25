from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest
import yaml

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EXCHANGE_TZ, QualityConfig, SessionConfig
from foundry.domains.nq.synthetic import SyntheticData, SyntheticSpec, generate, write_ninjatrader

REPO = Path(__file__).resolve().parents[1]
CALENDAR_FILE = REPO / "config" / "calendar_cme_equity.yaml"
ET = ZoneInfo(EXCHANGE_TZ)
SESSIONS = SessionConfig(
    rth_open=time(9, 30), rth_close=time(16, 0), globex_close=time(17, 0), globex_open=time(18, 0)
)
QUALITY = QualityConfig(
    bad_tick_atr_period=60,
    bad_tick_atr_multiple=20.0,
    max_break_fraction=0.001,
    top_gaps_reported=10,
)

# Synthetic e2e window: two rolls (Z23->H24, H24->M24), the March 2024 DST change,
# an early close (Nov 24 not included) and MLK / Presidents Day holidays.
SYN_SPEC = SyntheticSpec(start=date(2023, 12, 1), end=date(2024, 4, 30), seed=7)
SYN_SPLITS = {
    "search": {"start": "2023-12-01", "end": "2024-02-15"},
    "validation": {"start": "2024-02-16", "end": "2024-03-31"},
    "holdout_start": "2024-04-01",
    "min_trading_days_search": 20,
    "min_trading_days_validation": 10,
}


EVALUATOR_CFG: dict[str, object] = {
    "contracts": {
        "NQ": {"tick_size": 0.25, "point_value": 20.0, "commission_round_turn": 4.50},
        "MNQ": {"tick_size": 0.25, "point_value": 2.0, "commission_round_turn": 1.50},
    },
    "default_contract": "NQ",
    "contracts_per_trade": 1,
    "slippage_ticks": {"market": 1, "stop": 1, "limit": 0},
    "limit_trade_through_ticks": 1,
    "annualization_days": 252,
    "block_signals_on_flags": ["bad_tick", "ohlc_invalid"],
    "regimes": {"vol_lookback_days": 20, "trend_days": 10},
    "events_file": str(REPO / "config" / "events.yaml"),
    "results_db": "results/unused.duckdb",
}


# Synthetic splits are short (about 50 search days), so trade minimums are scaled down.
CRITICS_CFG: dict[str, object] = {
    "min_trades": {"search": 30, "validation": 15},
    "cost_stress_multiplier": 2.0,
    "concentration": {"top_day_fraction": 0.05, "max_profit_share": 0.5},
    "min_profitable_regimes": 2,
    "dsr_threshold": 0.95,
    "perturbations": [-0.2, -0.1, 0.1, 0.2],
    "pbo_threshold": 0.2,
    "pbo_min_trials": 4,
    "cscv_blocks": 8,
    "cpcv": {"groups": 5, "test_groups": 2, "embargo_days": 1, "min_positive_fraction": 0.6},
}


SEARCH_CFG: dict[str, object] = {
    "engine_file": str(REPO / "config" / "engines" / "engine_v0.yaml"),
    "budgets": {
        "wall_clock_seconds": 600,
        "max_evaluations": 200,
        "llm_dollars": 0.0,
        "llm_tokens": 0,
    },
    "llm": {
        "base_url": "https://llm.invalid",
        "model": "test-model",
        "api_key_env": "FOUNDRY_TEST_LLM_KEY",
        "price_input_per_mtok": 0.30,
        "price_output_per_mtok": 1.20,
        "max_output_tokens": 2000,
        "temperature": 1.0,
        "timeout_seconds": 5,
        "max_retries": 0,
    },
}


META_CFG: dict[str, object] = {
    "seeds": 3,
    "run_budget": {
        "wall_clock_seconds": 300,
        "max_evaluations": 40,
        "llm_dollars": 0.0,
        "llm_tokens": 0,
    },
    "wilcoxon_alpha": 0.2,
    "max_runtime_ratio": 3.0,
    "validation_folds": 2,
    "workspace": "results/unused",
    "mix_concentration": 40,
    "jitter": 0.25,
}


def et(y: int, mo: int, d: int, h: int, mi: int) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=ET)


@pytest.fixture(scope="session")
def calendar() -> TradingCalendar:
    return TradingCalendar.from_file(CALENDAR_FILE, SESSIONS)


@pytest.fixture(scope="session")
def synthetic(calendar: TradingCalendar) -> SyntheticData:
    return generate(SYN_SPEC, calendar)


def ts_frame(stamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame({"ts": pl.Series(stamps, dtype=pl.Datetime("us", EXCHANGE_TZ))})


def write_config(
    tmp: Path,
    raw_dir: Path,
    *,
    timezone: str = "UTC",
    label: str = "end",
    splits: dict[str, object] | None = None,
    critics: dict[str, object] | None = None,
    search: dict[str, object] | None = None,
    meta: dict[str, object] | None = None,
) -> Path:
    cfg = {
        "seed": 1,
        "data": {
            "instrument": {"root": "NQ", "tick_size": 0.25, "bar_minutes": 1},
            "sources": [
                {
                    "name": "synthetic",
                    "format": "ninjatrader_txt",
                    "directory": str(raw_dir),
                    "filename_regex": r"^NQ (?P<month>\d{2})-(?P<year>\d{2})\.Last-[Mm]inute\.txt$",
                    "timezone": timezone,
                    "timestamp_label": label,
                }
            ],
            "calendar_file": str(CALENDAR_FILE),
            "sessions": {
                "rth_open": "09:30",
                "rth_close": "16:00",
                "globex_close": "17:00",
                "globex_open": "18:00",
            },
            "roll": {"rule": "volume_crossover"},
            "quality": {
                "bad_tick_atr_period": 60,
                "bad_tick_atr_multiple": 20.0,
                "max_break_fraction": 0.001,
                "top_gaps_reported": 10,
            },
            "splits": splits or SYN_SPLITS,
            "processed_dir": str(tmp / "processed"),
        },
        "holdout": {"directory": str(tmp / "sealed")},
        "evaluator": EVALUATOR_CFG | {"results_db": str(tmp / "results.duckdb")},
        "critics": critics or CRITICS_CFG,
        "search": search or SEARCH_CFG,
        "meta": meta or META_CFG | {"workspace": str(tmp / "tournaments")},
    }
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_raw(tmp_path: Path, synthetic: SyntheticData) -> Path:
    raw = tmp_path / "raw"
    write_ninjatrader(synthetic.frames, raw)
    return raw
