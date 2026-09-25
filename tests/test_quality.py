"""Quality detectors flag and report; they never modify prices (ADR 0004)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.quality import (
    BAD_TICK,
    OFF_GRID,
    OHLC_INVALID,
    OUTSIDE_SESSION,
    ZERO_VOLUME,
    find_gaps,
    flag_bars,
    issues_from_flags,
    session_alignment,
)
from foundry.domains.nq.sessions import annotate_sessions
from tests.conftest import QUALITY, SESSIONS, et


def steady_bars(n: int, start=None) -> pl.DataFrame:
    start = start or et(2024, 3, 5, 10, 0)
    ts = [start + timedelta(minutes=i) for i in range(n)]
    close = [18000.0 + (i % 4) * 0.25 for i in range(n)]
    return pl.DataFrame(
        {
            "ts": pl.Series(ts, dtype=pl.Datetime("us", "America/New_York")),
            "contract": ["NQH24"] * n,
            "open": close,
            "high": [c + 0.5 for c in close],
            "low": [c - 0.5 for c in close],
            "close": close,
            "volume": [10] * n,
            "source_file": ["t"] * n,
        }
    )


def flagged(frame: pl.DataFrame, calendar: TradingCalendar) -> pl.DataFrame:
    return flag_bars(annotate_sessions(frame, calendar), 0.25, 1, QUALITY)


def set_row(frame: pl.DataFrame, i: int, **values: object) -> pl.DataFrame:
    idx = pl.int_range(pl.len())
    return frame.with_columns(
        [pl.when(idx == i).then(pl.lit(v)).otherwise(pl.col(k)).alias(k) for k, v in values.items()]
    )


def test_clean_bars_have_no_flags(calendar) -> None:
    out = flagged(steady_bars(120), calendar)
    assert (out["quality_flags"] == 0).all()


def test_each_detector(calendar) -> None:
    f = steady_bars(120)
    f = set_row(f, 70, volume=0)
    f = set_row(f, 80, high=18100.0, close=18090.0)  # spike: bad tick
    f = set_row(f, 90, high=17990.0)  # high below open/close
    f = set_row(f, 100, close=18000.1, high=18001.0)  # off the 0.25 grid
    out = flagged(f, calendar)
    flags = out["quality_flags"].to_list()
    assert flags[70] & ZERO_VOLUME
    assert flags[80] & BAD_TICK
    assert flags[90] & OHLC_INVALID
    assert flags[100] & OFF_GRID
    assert sum(1 for x in flags if x) == 4 + 1  # the bar after the spike moves back by >20 ATR
    kinds = set(issues_from_flags(out)["check"].to_list())
    assert kinds == {"zero_volume", "bad_tick", "ohlc_invalid", "off_grid"}


def test_prices_are_never_modified(calendar) -> None:
    f = set_row(steady_bars(120), 80, high=18100.0, close=18090.0)
    out = flagged(f, calendar).sort("ts")
    for col in ("open", "high", "low", "close", "volume"):
        assert out[col].equals(f.sort("ts")[col])


def test_bad_tick_check_has_no_lookahead(calendar) -> None:
    """Changing bars after t must not change the flag at t."""
    base = flagged(steady_bars(150), calendar)
    changed = flagged(set_row(steady_bars(150), 120, high=19000.0, close=18900.0), calendar)
    assert base["quality_flags"][:120].equals(changed["quality_flags"][:120])


def test_warmup_bars_are_not_bad_ticks(calendar) -> None:
    f = set_row(steady_bars(120), 5, high=18100.0, close=18090.0)
    out = flagged(f, calendar)
    assert not out["quality_flags"][5] & BAD_TICK


def test_outside_session_flag(calendar) -> None:
    f = steady_bars(3, start=et(2024, 3, 5, 17, 10))
    out = flagged(f, calendar)
    assert (out["quality_flags"] & OUTSIDE_SESSION != 0).all()


def test_session_alignment_detects_wrong_timezone(synthetic) -> None:
    bars = pl.concat([f for f in synthetic.frames.values()])
    assert session_alignment(bars, SESSIONS, 0.001).passed
    # Misreading ET wall times as UTC shifts every bar by 4-5 hours.
    shifted = bars.with_columns(
        pl.col("ts")
        .dt.replace_time_zone(None)
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("America/New_York")
    )
    res = session_alignment(shifted, SESSIONS, 0.001)
    assert not res.passed
    assert res.fraction > 0.02


def test_gaps_reported_not_filled(calendar) -> None:
    tdate = date(2024, 3, 5)
    grid = annotate_sessions(
        pl.DataFrame(
            {
                "ts": pl.datetime_range(
                    et(2024, 3, 4, 18, 0),
                    et(2024, 3, 5, 16, 59),
                    "1m",
                    eager=True,
                    time_zone="America/New_York",
                )
            }
        ),
        calendar,
    )
    # Drop 10:00-10:04 RTH and 02:00-02:29 ETH; also drop the whole of 2024-03-06.
    t = pl.col("ts").dt.time()
    holes = ((t >= pl.time(10, 0)) & (t < pl.time(10, 5))) | (
        (t >= pl.time(2, 0)) & (t < pl.time(2, 30))
    )
    series = grid.filter(~holes)
    later = annotate_sessions(
        pl.DataFrame(
            {"ts": pl.Series([et(2024, 3, 7, 10, 0)], dtype=pl.Datetime("us", "America/New_York"))}
        ),
        calendar,
    )
    series = pl.concat([series, later])
    g = find_gaps(series, calendar, 1)
    runs = g.runs.sort("start")
    first_day = runs.filter(pl.col("trading_date") == tdate)
    assert first_day.select("session", "minutes").rows() == [("ETH", 30), ("RTH", 5)]
    assert g.missing_dates == [date(2024, 3, 6)]
    # The input is not modified or filled.
    assert series.height == grid.height - 35 + 1
