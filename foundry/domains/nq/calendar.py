"""Exchange calendar: which dates trade, and when RTH and Globex close on each."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import polars as pl

from foundry.core.config import load_model
from foundry.domains.nq.config import EXCHANGE_TZ, CalendarFile, SessionConfig

DayKind = Literal["normal", "holiday", "early_close", "closed", "weekend"]
_TZ = ZoneInfo(EXCHANGE_TZ)


@dataclass(frozen=True)
class DayRule:
    kind: DayKind
    rth_close: time | None
    globex_close: time | None
    note: str = ""

    @property
    def is_session(self) -> bool:
        return self.kind not in ("closed", "weekend")


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime
    label: str


class TradingCalendar:
    def __init__(self, cal: CalendarFile, sessions: SessionConfig) -> None:
        self.name = cal.name
        self.covers_from = cal.covers_from
        self.covers_to = cal.covers_to
        self.sessions = sessions
        self._rules: dict[date, DayRule] = {}
        d = cal.defaults
        for e in cal.entries:
            if e.kind == "closed":
                rule = DayRule("closed", None, None, e.note)
            elif e.kind == "holiday":
                rule = DayRule("holiday", None, e.globex_close or d.holiday_globex_close, e.note)
            else:
                rule = DayRule(
                    "early_close",
                    e.rth_close or d.early_close_rth_close,
                    e.globex_close or d.early_close_globex_close,
                    e.note,
                )
            self._rules[e.date] = rule

    @classmethod
    def from_file(cls, path: Path, sessions: SessionConfig) -> TradingCalendar:
        return cls(load_model(path, CalendarFile), sessions)

    def covers(self, d: date) -> bool:
        return self.covers_from <= d <= self.covers_to

    def rule(self, d: date) -> DayRule:
        if d.weekday() >= 5:
            return DayRule("weekend", None, None)
        return self._rules.get(
            d, DayRule("normal", self.sessions.rth_close, self.sessions.globex_close)
        )

    def is_session_date(self, d: date) -> bool:
        return self.rule(d).is_session

    def frame(self, start: date, end: date) -> pl.DataFrame:
        """One row per calendar date in [start, end]."""
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        rules = [self.rule(d) for d in days]
        return pl.DataFrame(
            {
                "date": days,
                "is_session": [r.is_session for r in rules],
                "kind": [r.kind for r in rules],
                "rth_close": [r.rth_close for r in rules],
                "globex_close": [r.globex_close for r in rules],
            },
            schema={
                "date": pl.Date,
                "is_session": pl.Boolean,
                "kind": pl.String,
                "rth_close": pl.Time,
                "globex_close": pl.Time,
            },
        )

    def session_intervals(self, trading_date: date) -> list[Interval]:
        """Tradeable-hours intervals (ETH, RTH, POST, HOLIDAY) belonging to one trading date."""
        rule = self.rule(trading_date)
        if not rule.is_session or rule.globex_close is None:
            return []
        s = self.sessions

        def at(d: date, t: time) -> datetime:
            return datetime.combine(d, t, tzinfo=_TZ)

        out: list[Interval] = []
        eth_end = min(s.rth_open, rule.globex_close)
        out.append(
            Interval(
                at(trading_date - timedelta(days=1), s.globex_open),
                at(trading_date, eth_end),
                "ETH",
            )
        )
        if rule.globex_close <= s.rth_open:
            return out
        if rule.kind == "holiday":
            out.append(
                Interval(
                    at(trading_date, s.rth_open), at(trading_date, rule.globex_close), "HOLIDAY"
                )
            )
            return out
        assert rule.rth_close is not None
        out.append(Interval(at(trading_date, s.rth_open), at(trading_date, rule.rth_close), "RTH"))
        if rule.globex_close > rule.rth_close:
            out.append(
                Interval(
                    at(trading_date, rule.rth_close), at(trading_date, rule.globex_close), "POST"
                )
            )
        return out
