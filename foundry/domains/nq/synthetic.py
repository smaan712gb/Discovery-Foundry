"""Synthetic per-contract NQ minute data, written in the NinjaTrader export format.

Used by tests and, per ADR 0005, by evaluator/critic/search development until the owner's
2018-2024 export is available. Construction: one common random walk on the tick grid, plus a
constant basis per contract. A correct back-adjustment must therefore recover
`underlying + basis[anchor]` exactly, which is what the roll tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EXCHANGE_TZ
from foundry.domains.nq.loaders import MONTH_CODES
from foundry.domains.nq.sessions import expected_minutes

_ET = ZoneInfo(EXCHANGE_TZ)
_QUARTER_MONTHS = (3, 6, 9, 12)
_TRADED = ("ETH", "RTH", "POST", "HOLIDAY")


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


@dataclass(frozen=True)
class SyntheticSpec:
    start: date
    end: date
    seed: int
    start_price: float = 15000.0
    tick_size: float = 0.25
    overlap_days: int = 12  # a contract's data starts this many days before the prior expiry
    crossover_days: int = 8  # the next contract takes over volume this many days before expiry
    front_volume: int = 400
    back_volume: int = 40
    # Planted edge for evaluator checks: after the first `edge_minutes` of RTH, every minute drifts
    # this many ticks in the direction of that opening range's net move. 0 = pure random walk.
    orb_drift_ticks: int = 0
    edge_minutes: int = 30
    edge_day_share: float = 1.0  # share of RTH days (random) on which the planted drift appears


@dataclass(frozen=True)
class SyntheticContract:
    code: str
    first_day: date
    expiry: date
    basis_ticks: int
    front_from: date | None  # None: front from the start of the data
    front_until: date  # exclusive


@dataclass(frozen=True)
class SyntheticData:
    frames: dict[str, pl.DataFrame]
    contracts: list[SyntheticContract]
    underlying: pl.DataFrame  # ts, trading_date, session, close (no basis)


def contract_schedule(spec: SyntheticSpec) -> list[SyntheticContract]:
    rng = np.random.default_rng([spec.seed, 1])
    out: list[SyntheticContract] = []
    prev_expiry: date | None = None
    for yy in range(spec.start.year - 1, spec.end.year + 2):
        for m in _QUARTER_MONTHS:
            expiry = third_friday(yy, m)
            if expiry < spec.start:
                prev_expiry = expiry
                continue
            first = (
                spec.start
                if prev_expiry is None
                else max(spec.start, prev_expiry - timedelta(days=spec.overlap_days))
            )
            if first > spec.end:
                return out
            front_from = (
                None
                if prev_expiry is None or prev_expiry < spec.start
                else (prev_expiry - timedelta(days=spec.crossover_days))
            )
            out.append(
                SyntheticContract(
                    code=f"NQ{MONTH_CODES[m]}{yy % 100:02d}",
                    first_day=first,
                    expiry=expiry,
                    basis_ticks=int(rng.integers(-400, 400)),
                    front_from=front_from,
                    front_until=expiry - timedelta(days=spec.crossover_days),
                )
            )
            prev_expiry = expiry
    return out


def generate(spec: SyntheticSpec, calendar: TradingCalendar) -> SyntheticData:
    """Per-contract bar frames with tz-aware ET bar-start `ts`."""
    rng_path = np.random.default_rng([spec.seed, 0])
    rng_vol = np.random.default_rng([spec.seed, 2])
    grid = expected_minutes(calendar, spec.start, spec.end, 1).filter(
        pl.col("session").is_in(_TRADED)
    )
    n = grid.height
    steps = rng_path.integers(-4, 5, size=n)
    if spec.orb_drift_ticks:
        steps = steps + _orb_drift(grid, steps, spec)
    close_t = round(spec.start_price / spec.tick_size) + np.cumsum(steps)
    open_t = np.concatenate([close_t[:1], close_t[:-1]])
    high_t = np.maximum(open_t, close_t) + rng_path.integers(0, 3, size=n)
    low_t = np.minimum(open_t, close_t) - rng_path.integers(0, 3, size=n)
    under = grid.with_columns(
        pl.Series("_o", open_t),
        pl.Series("_h", high_t),
        pl.Series("_l", low_t),
        pl.Series("_c", close_t),
    )

    contracts = contract_schedule(spec)
    frames: dict[str, pl.DataFrame] = {}
    for c in contracts:
        cutoff = datetime.combine(c.expiry, time(9, 30), tzinfo=_ET)
        seg = under.filter(
            (pl.col("trading_date") >= c.first_day)
            & (pl.col("trading_date") <= c.expiry)
            & (pl.col("ts") < pl.lit(cutoff).dt.convert_time_zone(EXCHANGE_TZ))
        )
        if seg.is_empty():
            continue
        td = pl.col("trading_date")
        is_front = td < pl.lit(c.front_until)
        if c.front_from is not None:
            is_front = is_front & (td >= pl.lit(c.front_from))
        noise = pl.Series("_noise", rng_vol.integers(0, 20, size=seg.height))
        seg = seg.with_columns(noise).with_columns(
            *[
                ((pl.col(k) + c.basis_ticks) * spec.tick_size).alias(name)
                for k, name in (("_o", "open"), ("_h", "high"), ("_l", "low"), ("_c", "close"))
            ],
            (
                pl.when(is_front)
                .then(pl.lit(spec.front_volume))
                .otherwise(pl.lit(spec.back_volume))
                + pl.col("_noise")
            )
            .cast(pl.Int64)
            .alias("volume"),
        )
        frames[c.code] = seg.select("ts", "open", "high", "low", "close", "volume")
    underlying = under.select(
        "ts", "trading_date", "session", (pl.col("_c") * spec.tick_size).alias("close")
    )
    return SyntheticData(frames=frames, contracts=contracts, underlying=underlying)


def _orb_drift(grid: pl.DataFrame, steps: np.ndarray, spec: SyntheticSpec) -> np.ndarray:
    """Extra ticks per bar: after the RTH opening range, drift with the range's net move."""
    rth_open = time(9, 30)
    f = grid.with_columns(pl.Series("_s", steps)).with_columns(
        (
            pl.col("ts")
            - pl.col("trading_date").dt.combine(rth_open).dt.replace_time_zone(EXCHANGE_TZ)
        )
        .dt.total_minutes()
        .alias("_m")
    )
    rth = pl.col("session") == "RTH"
    in_range = rth & (pl.col("_m") < spec.edge_minutes)
    direction = pl.when(in_range).then(pl.col("_s")).sum().over("trading_date").sign()
    days = f["trading_date"].unique().sort()
    keep = np.random.default_rng([spec.seed, 3]).random(days.len()) < spec.edge_day_share
    f = f.with_columns(
        pl.col("trading_date").is_in(days.filter(pl.Series(keep)).implode()).alias("_edge_day")
    )
    drift = (
        pl.when(rth & pl.col("_edge_day") & (pl.col("_m") >= spec.edge_minutes))
        .then(pl.when(direction >= 0).then(1).otherwise(-1) * spec.orb_drift_ticks)
        .otherwise(0)
    )
    return f.select(drift.cast(pl.Int64).alias("d"))["d"].to_numpy()


def write_ninjatrader(frames: dict[str, pl.DataFrame], directory: Path) -> list[Path]:
    """Write frames as `NQ MM-YY.Last-Minute.txt`: UTC, bar-END stamps, like the owner's files."""
    directory.mkdir(parents=True, exist_ok=True)
    rev = {v: k for k, v in MONTH_CODES.items()}
    paths = []
    for code, f in frames.items():
        path = directory / f"NQ {rev[code[2]]:02d}-{code[3:]}.Last-Minute.txt"
        f.select(
            (pl.col("ts").dt.convert_time_zone("UTC") + pl.duration(minutes=1))
            .dt.strftime("%Y%m%d %H%M%S")
            .alias("stamp"),
            "open",
            "high",
            "low",
            "close",
            "volume",
        ).write_csv(path, separator=";", include_header=False)
        paths.append(path)
    return paths
