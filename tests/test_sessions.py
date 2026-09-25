"""Session labels, trading dates, holidays, early closes and DST (ADR 0004)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.sessions import annotate_sessions, expected_minutes
from tests.conftest import et, ts_frame

CASES = [
    # (bar start ET, expected trading date, expected session)
    (et(2024, 3, 4, 9, 29), date(2024, 3, 4), "ETH"),
    (et(2024, 3, 4, 9, 30), date(2024, 3, 4), "RTH"),
    (et(2024, 3, 4, 15, 59), date(2024, 3, 4), "RTH"),
    (et(2024, 3, 4, 16, 0), date(2024, 3, 4), "POST"),
    (et(2024, 3, 4, 16, 59), date(2024, 3, 4), "POST"),
    (et(2024, 3, 4, 17, 0), date(2024, 3, 4), "OUTSIDE"),  # maintenance break
    (et(2024, 3, 4, 17, 59), date(2024, 3, 4), "OUTSIDE"),
    (et(2024, 3, 4, 18, 0), date(2024, 3, 5), "ETH"),  # evening -> next trading date
    (et(2024, 3, 4, 23, 59), date(2024, 3, 5), "ETH"),
    (et(2024, 3, 5, 0, 0), date(2024, 3, 5), "ETH"),
    # weekend and the DST Sunday (2024-03-10, clocks jump at 02:00)
    (et(2024, 3, 8, 17, 0), date(2024, 3, 8), "OUTSIDE"),
    (et(2024, 3, 8, 18, 0), date(2024, 3, 9), "OUTSIDE"),  # Friday evening: no session
    (et(2024, 3, 9, 12, 0), date(2024, 3, 9), "OUTSIDE"),
    (et(2024, 3, 10, 17, 59), date(2024, 3, 10), "OUTSIDE"),
    (et(2024, 3, 10, 18, 0), date(2024, 3, 11), "ETH"),  # Sunday open, now EDT
    (et(2024, 3, 11, 9, 30), date(2024, 3, 11), "RTH"),
    # November fall-back weekend
    (et(2024, 11, 3, 18, 0), date(2024, 11, 4), "ETH"),
    (et(2024, 11, 4, 9, 30), date(2024, 11, 4), "RTH"),
    # MLK Day: abbreviated Globex, no RTH
    (et(2024, 1, 14, 18, 0), date(2024, 1, 15), "ETH"),
    (et(2024, 1, 15, 8, 0), date(2024, 1, 15), "ETH"),
    (et(2024, 1, 15, 10, 0), date(2024, 1, 15), "HOLIDAY"),
    (et(2024, 1, 15, 12, 59), date(2024, 1, 15), "HOLIDAY"),
    (et(2024, 1, 15, 13, 0), date(2024, 1, 15), "OUTSIDE"),
    (et(2024, 1, 15, 18, 0), date(2024, 1, 16), "ETH"),
    # Day after Thanksgiving: RTH to 13:00, Globex halt 13:15
    (et(2024, 11, 29, 12, 59), date(2024, 11, 29), "RTH"),
    (et(2024, 11, 29, 13, 0), date(2024, 11, 29), "POST"),
    (et(2024, 11, 29, 13, 14), date(2024, 11, 29), "POST"),
    (et(2024, 11, 29, 13, 15), date(2024, 11, 29), "OUTSIDE"),
    # Christmas: early close on the 24th, fully closed on the 25th
    (et(2024, 12, 24, 12, 59), date(2024, 12, 24), "RTH"),
    (et(2024, 12, 24, 13, 15), date(2024, 12, 24), "OUTSIDE"),
    (et(2024, 12, 24, 18, 0), date(2024, 12, 25), "OUTSIDE"),
    (et(2024, 12, 25, 10, 0), date(2024, 12, 25), "OUTSIDE"),
    (et(2024, 12, 25, 18, 0), date(2024, 12, 26), "ETH"),
    # 2025-01-01 closed; the first session opens that evening for 2025-01-02
    (et(2025, 1, 1, 18, 0), date(2025, 1, 2), "ETH"),
]


@pytest.mark.parametrize(("ts", "tdate", "session"), CASES, ids=[str(c[0]) for c in CASES])
def test_session_labels(calendar: TradingCalendar, ts, tdate, session) -> None:
    out = annotate_sessions(ts_frame([ts]), calendar)
    assert out["trading_date"][0] == tdate
    assert out["session"][0] == session


@pytest.mark.parametrize(
    ("tdate", "expected"),
    [
        (date(2024, 3, 5), {"ETH": 930, "RTH": 390, "POST": 60}),
        (date(2024, 3, 11), {"ETH": 930, "RTH": 390, "POST": 60}),  # DST Sunday open
        (date(2024, 11, 4), {"ETH": 930, "RTH": 390, "POST": 60}),  # fall-back Sunday open
        (date(2024, 11, 29), {"ETH": 930, "RTH": 210, "POST": 15}),  # early close
        (date(2024, 1, 15), {"ETH": 930, "HOLIDAY": 210}),  # MLK
        (date(2024, 12, 25), {}),  # closed
        (date(2024, 3, 9), {}),  # Saturday
    ],
)
def test_expected_minutes_per_trading_date(calendar: TradingCalendar, tdate, expected) -> None:
    grid = expected_minutes(calendar, tdate, tdate, 1)
    counts = dict(grid.group_by("session").len().iter_rows())
    assert counts == expected
    if grid.height:
        assert grid["ts"].is_unique().all()
        assert grid["ts"].is_sorted()


def test_expected_minutes_agree_with_labels(calendar: TradingCalendar) -> None:
    """The gap grid and the bar labeller must implement the same rules."""
    grid = expected_minutes(calendar, date(2024, 1, 10), date(2024, 3, 20), 1)
    labelled = annotate_sessions(grid.select("ts"), calendar)
    assert (labelled["session"] == grid["session"]).all()
    assert (labelled["trading_date"] == grid["trading_date"]).all()


def test_dst_session_is_constant_length_in_absolute_time(calendar: TradingCalendar) -> None:
    """Sunday 18:00 -> Monday 17:00 is 23 real hours on both DST weekends."""
    for tdate in (date(2024, 3, 11), date(2024, 11, 4)):
        grid = expected_minutes(calendar, tdate, tdate, 1)
        span = grid["ts"].max() - grid["ts"].min()  # type: ignore[operator]
        assert span.total_seconds() == (23 * 60 - 1) * 60


def test_calendar_rules(calendar: TradingCalendar) -> None:
    assert calendar.rule(date(2024, 12, 25)).kind == "closed"
    assert calendar.rule(date(2024, 3, 9)).kind == "weekend"
    assert calendar.rule(date(2024, 1, 15)).kind == "holiday"
    assert calendar.rule(date(2024, 3, 5)).kind == "normal"
    assert not calendar.covers(date(2017, 12, 31))


def test_empty_frame_annotates(calendar: TradingCalendar) -> None:
    out = annotate_sessions(ts_frame([]), calendar)
    assert out.height == 0
    assert {"trading_date", "session"} <= set(out.columns)
    assert out.schema["trading_date"] == pl.Date
