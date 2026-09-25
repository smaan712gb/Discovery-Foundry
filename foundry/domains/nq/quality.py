"""Data-quality detectors. They flag and report; they never fill or modify prices (ADR 0004)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EXCHANGE_TZ, QualityConfig, SessionConfig
from foundry.domains.nq.loaders import ISSUE_SCHEMA
from foundry.domains.nq.sessions import OUTSIDE, expected_minutes

# Bitmask stored per bar in `quality_flags`.
ZERO_VOLUME = 1
BAD_TICK = 2
OHLC_INVALID = 4
OFF_GRID = 8
OUTSIDE_SESSION = 16
FLAG_NAMES: dict[int, str] = {
    ZERO_VOLUME: "zero_volume",
    BAD_TICK: "bad_tick",
    OHLC_INVALID: "ohlc_invalid",
    OFF_GRID: "off_grid",
    OUTSIDE_SESSION: "outside_session",
}
# Price-grid tolerance: prices are decimal multiples of the tick, stored as float64.
_GRID_EPS = 1e-6


def flag_bars(
    bars: pl.DataFrame, tick_size: float, bar_minutes: int, q: QualityConfig
) -> pl.DataFrame:
    """Add `quality_flags` (UInt32 bitmask). `bars`: per-contract data carrying `session`."""
    frame = bars.sort(["contract", "ts"])
    prev_close = pl.col("close").shift(1).over("contract")
    prev_ts = pl.col("ts").shift(1).over("contract")
    contiguous = (pl.col("ts") - prev_ts) == pl.duration(minutes=bar_minutes)
    rng = pl.col("high") - pl.col("low")
    true_range = (
        pl.when(contiguous)
        .then(
            pl.max_horizontal(
                rng, (pl.col("high") - prev_close).abs(), (pl.col("low") - prev_close).abs()
            )
        )
        .otherwise(rng)
    )
    frame = frame.with_columns(
        true_range.alias("_tr"),
        contiguous.fill_null(False).alias("_contig"),
        prev_close.alias("_pc"),
    )
    # ATR of the *preceding* bars only: the check itself must not look ahead. Floored at one
    # tick so perfectly flat stretches don't turn every later move into a "bad tick".
    atr_prev = (
        pl.col("_tr").rolling_mean(window_size=q.bad_tick_atr_period).shift(1).over("contract")
    )
    frame = frame.with_columns(pl.max_horizontal(atr_prev, pl.lit(tick_size)).alias("_atr"))
    limit = pl.col("_atr") * q.bad_tick_atr_multiple
    atr_known = atr_prev.is_not_null()
    bad_tick = atr_known & (
        (rng > limit) | (pl.col("_contig") & ((pl.col("close") - pl.col("_pc")).abs() > limit))
    )

    prices = [pl.col(c) for c in ("open", "high", "low", "close")]
    ohlc_bad = (
        (pl.col("high") < pl.max_horizontal(pl.col("open"), pl.col("close")))
        | (pl.col("low") > pl.min_horizontal(pl.col("open"), pl.col("close")))
        | (pl.col("high") < pl.col("low"))
        | pl.any_horizontal([p <= 0 for p in prices])
        | pl.any_horizontal([p.is_null() | p.is_nan() for p in prices])
    )
    off_grid = pl.any_horizontal(
        [((p / tick_size) - (p / tick_size).round(0)).abs() > _GRID_EPS for p in prices]
    )

    def bit(cond: pl.Expr, value: int) -> pl.Expr:
        return (
            pl.when(cond.fill_null(False))
            .then(pl.lit(value, pl.UInt32))
            .otherwise(pl.lit(0, pl.UInt32))
        )

    flags = (
        bit(pl.col("volume") == 0, ZERO_VOLUME)
        + bit(bad_tick, BAD_TICK)
        + bit(ohlc_bad, OHLC_INVALID)
        + bit(off_grid, OFF_GRID)
        + bit(pl.col("session") == OUTSIDE, OUTSIDE_SESSION)
    )
    return frame.with_columns(flags.alias("quality_flags"), pl.col("_atr").alias("_atr_prev")).drop(
        "_tr", "_contig", "_pc", "_atr"
    )


def issues_from_flags(frame: pl.DataFrame) -> pl.DataFrame:
    """One issue row per (flagged bar, flag). Expects `_atr_prev` from flag_bars."""
    parts = []
    for value, name in FLAG_NAMES.items():
        hit = frame.filter((pl.col("quality_flags") & value) != 0)
        if hit.is_empty():
            continue
        if value == BAD_TICK:
            detail = pl.format(
                "range={} close={} prior_atr={}",
                (pl.col("high") - pl.col("low")).round(2),
                pl.col("close"),
                pl.col("_atr_prev").round(3),
            )
        elif value == OUTSIDE_SESSION:
            detail = pl.format(
                "bar start {} labelled OUTSIDE", pl.col("ts").dt.strftime("%Y-%m-%d %H:%M %Z")
            )
        else:
            detail = pl.format(
                "o={} h={} l={} c={} v={}",
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                pl.col("volume"),
            )
        parts.append(
            hit.select(
                "ts",
                "contract",
                pl.lit(name).alias("check"),
                pl.lit("warning").alias("severity"),
                detail.alias("detail"),
            )
        )
    if not parts:
        return pl.DataFrame(schema=ISSUE_SCHEMA)
    return pl.concat(parts).cast(ISSUE_SCHEMA).sort(["ts", "check"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class AlignmentResult:
    bars_in_break: int
    total_bars: int
    fraction: float
    passed: bool


def session_alignment(bars: pl.DataFrame, s: SessionConfig, max_fraction: float) -> AlignmentResult:
    """Fraction of bars inside the daily maintenance break; a wrong source timezone shows here."""
    t = pl.col("ts").dt.convert_time_zone(EXCHANGE_TZ).dt.time()
    in_break = bars.filter((t >= pl.lit(s.globex_close)) & (t < pl.lit(s.globex_open))).height
    total = bars.height
    frac = in_break / total if total else 0.0
    return AlignmentResult(in_break, total, frac, frac <= max_fraction)


@dataclass(frozen=True)
class GapReport:
    runs: pl.DataFrame  # every gap run: trading_date, session, start, end, minutes
    missing_by_session: pl.DataFrame
    missing_dates: list[date]  # session dates with no bars at all
    expected_minutes: int
    present_minutes: int


def find_gaps(series: pl.DataFrame, calendar: TradingCalendar, bar_minutes: int) -> GapReport:
    """Expected session minutes with no bar in the continuous series. Report, never fill."""
    empty_runs = pl.DataFrame(
        schema={
            "trading_date": pl.Date,
            "session": pl.String,
            "start": pl.Datetime("us", EXCHANGE_TZ),
            "end": pl.Datetime("us", EXCHANGE_TZ),
            "minutes": pl.Int64,
        }
    )
    if series.is_empty():
        return GapReport(
            empty_runs, pl.DataFrame(schema={"session": pl.String, "missing": pl.Int64}), [], 0, 0
        )
    first: date = series["trading_date"].min()  # type: ignore[assignment]
    last: date = series["trading_date"].max()  # type: ignore[assignment]
    expected = expected_minutes(calendar, first, last, bar_minutes)
    present_dates = set(series["trading_date"].unique().to_list())
    session_dates = expected["trading_date"].unique().sort().to_list()
    missing_dates = [d for d in session_dates if d not in present_dates]

    expected_live = expected.filter(~pl.col("trading_date").is_in(missing_dates))
    missing = expected_live.join(series.select("ts"), on="ts", how="anti").sort("ts")
    present = expected_live.height - missing.height

    if missing.is_empty():
        runs = empty_runs
    else:
        step = pl.duration(minutes=bar_minutes)
        new_run = (
            ((pl.col("ts") - pl.col("ts").shift(1)) != step)
            | (pl.col("session") != pl.col("session").shift(1))
            | (pl.col("trading_date") != pl.col("trading_date").shift(1))
        ).fill_null(True)
        runs = (
            missing.with_columns(new_run.cum_sum().alias("_run"))
            .group_by("_run", maintain_order=True)
            .agg(
                pl.col("trading_date").first(),
                pl.col("session").first(),
                pl.col("ts").min().alias("start"),
                pl.col("ts").max().alias("end"),
                pl.len().cast(pl.Int64).alias("minutes"),
            )
            .drop("_run")
            .with_columns(pl.col("minutes") * bar_minutes)
        )
    by_session = (
        missing.group_by("session").agg(pl.len().cast(pl.Int64).alias("missing")).sort("session")
    )
    return GapReport(runs, by_session, missing_dates, expected_live.height, present)
