"""Session labels and trading dates for bars keyed by their start time in America/New_York.

See ADR 0004 for the rules. Evening bars (at or after the Globex open) belong to the next
calendar day's trading date; every other bar belongs to its own calendar date.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EXCHANGE_TZ

RTH = "RTH"
ETH = "ETH"
POST = "POST"
HOLIDAY = "HOLIDAY"
OUTSIDE = "OUTSIDE"
SESSION_LABELS: tuple[str, ...] = (RTH, ETH, POST, HOLIDAY, OUTSIDE)
TRADEABLE: tuple[str, ...] = (RTH, ETH)


def annotate_sessions(bars: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    """Add `trading_date` and `session` columns. `bars.ts` must be tz-aware ET bar starts."""
    if bars.is_empty():
        return bars.with_columns(
            pl.lit(None, dtype=pl.Date).alias("trading_date"),
            pl.lit(None, dtype=pl.String).alias("session"),
        )
    s = calendar.sessions
    local = pl.col("ts").dt.convert_time_zone(EXCHANGE_TZ)
    frame = bars.with_columns(
        local.dt.date().alias("_d"),
        local.dt.time().alias("_t"),
    )
    lo: date = frame["_d"].min()  # type: ignore[assignment]
    hi: date = frame["_d"].max()  # type: ignore[assignment]
    cal = calendar.frame(lo - timedelta(days=1), hi + timedelta(days=1))
    cal_today = cal.rename(
        {
            "date": "_d",
            "is_session": "_sess",
            "kind": "_kind",
            "rth_close": "_rth_close",
            "globex_close": "_gx_close",
        }
    )
    cal_next = cal.select(
        (pl.col("date") - pl.duration(days=1)).alias("_d"),
        pl.col("is_session").alias("_sess_next"),
    )
    frame = frame.join(cal_today, on="_d", how="left").join(cal_next, on="_d", how="left")

    t = pl.col("_t")
    evening = t >= pl.lit(s.globex_open)
    session = (
        pl.when(evening & pl.col("_sess_next"))
        .then(pl.lit(ETH))
        .when(evening)
        .then(pl.lit(OUTSIDE))
        .when(~pl.col("_sess"))
        .then(pl.lit(OUTSIDE))
        .when(t >= pl.col("_gx_close"))
        .then(pl.lit(OUTSIDE))
        .when(t < pl.lit(s.rth_open))
        .then(pl.lit(ETH))
        .when(pl.col("_kind") == "holiday")
        .then(pl.lit(HOLIDAY))
        .when(t < pl.col("_rth_close"))
        .then(pl.lit(RTH))
        .otherwise(pl.lit(POST))
    )
    trading_date = pl.when(evening).then(pl.col("_d") + pl.duration(days=1)).otherwise(pl.col("_d"))
    return frame.with_columns(trading_date.alias("trading_date"), session.alias("session")).drop(
        "_d", "_t", "_sess", "_kind", "_rth_close", "_gx_close", "_sess_next"
    )


def expected_minutes(
    calendar: TradingCalendar, first: date, last: date, bar_minutes: int
) -> pl.DataFrame:
    """Every bar start the exchange should produce for trading dates in [first, last]."""
    starts, ends, labels, tdates = [], [], [], []
    d = first
    while d <= last:
        for iv in calendar.session_intervals(d):
            starts.append(iv.start)
            ends.append(iv.end)
            labels.append(iv.label)
            tdates.append(d)
        d += timedelta(days=1)
    dtype = pl.Datetime("us", EXCHANGE_TZ)
    if not starts:
        return pl.DataFrame(schema={"ts": dtype, "trading_date": pl.Date, "session": pl.String})
    intervals = pl.DataFrame(
        {
            "_start": pl.Series(starts, dtype=dtype),
            "_end": pl.Series(ends, dtype=dtype),
            "session": labels,
            "trading_date": pl.Series(tdates, dtype=pl.Date),
        }
    )
    return (
        intervals.with_columns(
            pl.datetime_ranges(
                "_start", "_end", interval=f"{bar_minutes}m", closed="left", time_unit="us"
            ).alias("ts")
        )
        .explode("ts", empty_as_null=False)
        .select("ts", "trading_date", "session")
        .sort("ts")
    )
