"""Continuous contract: volume-crossover rolls, anchored additive adjustment (ADR 0003)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from typing import cast

import polars as pl

from foundry.domains.nq.loaders import ISSUE_SCHEMA, TS_DTYPE, parse_contract_code

ROLL_SCHEMA: dict[str, pl.DataType] = {
    "decision_date": pl.Date(),
    "effective_date": pl.Date(),
    "from_contract": pl.String(),
    "to_contract": pl.String(),
    "reason": pl.String(),
    "from_volume": pl.Int64(),
    "to_volume": pl.Int64(),
    "gap_ts": TS_DTYPE,
    "gap_points": pl.Float64(),
    "from_adjustment": pl.Float64(),
    "to_adjustment": pl.Float64(),
    "continuity_ok": pl.Boolean(),
}
# Prices sit on a 0.25 grid and adjustments are sums of grid differences, so continuity
# should hold exactly; this only absorbs float64 representation error.
_CONTINUITY_EPS = 1e-9


@dataclass(frozen=True)
class Segment:
    contract: str
    start: date
    end: date


@dataclass(frozen=True)
class RollResult:
    series: pl.DataFrame
    rolls: pl.DataFrame
    segments: list[Segment]
    anchor: str | None
    issues: pl.DataFrame


def _contract_order(codes: list[str]) -> list[str]:
    def key(c: str) -> tuple[int, str]:
        cid = parse_contract_code(c)
        return (cid.sort_key if cid else 0, c)

    return sorted(codes, key=key)


def _missing_contract_issues(order: list[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for a, b in pairwise(order):
        ca, cb = parse_contract_code(a), parse_contract_code(b)
        if ca is None or cb is None:
            continue
        months = (cb.year - ca.year) * 12 + (cb.month - ca.month)
        if months > 3:
            out.append(
                {
                    "ts": None,
                    "contract": b,
                    "check": "missing_contract",
                    "severity": "warning",
                    "detail": f"{months} months between {a} and {b}: quarterly contract(s) missing",
                }
            )
    return out


def build_continuous(bars: pl.DataFrame, holdout_start: date) -> RollResult:
    """`bars`: all contracts, annotated with trading_date/session and quality_flags."""
    issues: list[dict[str, object]] = []
    if bars.is_empty():
        return RollResult(
            bars, pl.DataFrame(schema=ROLL_SCHEMA), [], None, pl.DataFrame(schema=ISSUE_SCHEMA)
        )
    order = _contract_order(bars["contract"].unique().to_list())
    issues += _missing_contract_issues(order)
    successor = {a: b for a, b in pairwise(order)}

    daily = bars.group_by(["contract", "trading_date"]).agg(pl.col("volume").sum())
    vol: dict[tuple[str, date], int] = {
        (r["contract"], r["trading_date"]): r["volume"] for r in daily.iter_rows(named=True)
    }
    dates: list[date] = sorted(daily["trading_date"].unique().to_list())

    current = next(c for c in order if (c, dates[0]) in vol)
    seg_start = dates[0]
    segments: list[Segment] = []
    raw_rolls: list[dict[str, object]] = []
    for i, d in enumerate(dates[:-1]):
        nxt = successor.get(current)
        if nxt is None:
            continue
        d1 = dates[i + 1]
        v_cur, v_nxt = vol.get((current, d), 0), vol.get((nxt, d), 0)
        crossover = v_nxt > v_cur
        forced = (current, d1) not in vol and (nxt, d1) in vol
        if not (crossover or forced):
            continue
        segments.append(Segment(current, seg_start, d))
        raw_rolls.append(
            {
                "decision_date": d,
                "effective_date": d1,
                "from_contract": current,
                "to_contract": nxt,
                "reason": "volume_crossover" if crossover else "forced",
                "from_volume": v_cur,
                "to_volume": v_nxt,
            }
        )
        current, seg_start = nxt, d1
    segments.append(Segment(current, seg_start, dates[-1]))

    # Gap at the last minute of the decision day where both contracts traded.
    for r in raw_rolls:
        day = bars.filter(pl.col("trading_date") == r["decision_date"])
        a = day.filter(pl.col("contract") == r["from_contract"]).select(
            "ts", pl.col("close").alias("ca")
        )
        b = day.filter(pl.col("contract") == r["to_contract"]).select(
            "ts", pl.col("close").alias("cb")
        )
        common = a.join(b, on="ts").sort("ts")
        if common.is_empty():
            r["gap_ts"], r["gap_points"] = None, None
            issues.append(
                {
                    "ts": None,
                    "contract": r["to_contract"],
                    "check": "roll_no_overlap",
                    "severity": "fatal",
                    "detail": f"roll {r['from_contract']}->{r['to_contract']} decided "
                    f"{r['decision_date']}: no common bar, gap unmeasurable",
                }
            )
        else:
            last = common.row(-1, named=True)
            r["gap_ts"], r["gap_points"] = last["ts"], last["cb"] - last["ca"]

    # Anchor: the contract that is front on the last pre-holdout trading date.
    pre = [s for s in segments if s.start < holdout_start]
    anchor = pre[-1].contract if pre else segments[0].contract
    adj: dict[str, float | None] = {anchor: 0.0}
    k = next(i for i, s in enumerate(segments) if s.contract == anchor)
    for r in raw_rolls[k:]:  # forward from the anchor
        prev = adj.get(str(r["from_contract"]))
        gap = r["gap_points"]
        adj[str(r["to_contract"])] = None if prev is None or gap is None else prev - float(gap)  # type: ignore[arg-type]
    for r in reversed(raw_rolls[:k]):  # backward from the anchor
        nxt_adj = adj.get(str(r["to_contract"]))
        gap = r["gap_points"]
        adj[str(r["from_contract"])] = (
            None if nxt_adj is None or gap is None else nxt_adj + float(gap)  # type: ignore[arg-type]
        )

    parts = []
    for s in segments:
        seg_adj = adj.get(s.contract)
        seg = bars.filter(
            (pl.col("contract") == s.contract)
            & (pl.col("trading_date") >= s.start)
            & (pl.col("trading_date") <= s.end)
        )
        seg = seg.with_columns(pl.lit(seg_adj, dtype=pl.Float64).alias("adjustment"))
        parts.append(seg)
    series = pl.concat(parts).sort("ts")
    series = series.rename({c: f"raw_{c}" for c in ("open", "high", "low", "close")}).with_columns(
        [
            (pl.col(f"raw_{c}") + pl.col("adjustment")).alias(c)
            for c in ("open", "high", "low", "close")
        ]
    )

    for r in raw_rolls:
        r["from_adjustment"] = adj.get(str(r["from_contract"]))
        r["to_adjustment"] = adj.get(str(r["to_contract"]))
        r["continuity_ok"] = _continuity_ok(series, bars, r)
        # Rolls downstream of an unmeasurable one have unknown adjustments; that is reported
        # once as roll_no_overlap, not as a jump at every later roll.
        known = r["from_adjustment"] is not None and r["to_adjustment"] is not None
        if not r["continuity_ok"] and known:
            issues.append(
                {
                    "ts": None,
                    "contract": r["to_contract"],
                    "check": "roll_discontinuity",
                    "severity": "fatal",
                    "detail": f"adjusted series jumps at roll {r['from_contract']}->"
                    f"{r['to_contract']} on {r['effective_date']}",
                }
            )
    rolls = (
        pl.DataFrame(raw_rolls, schema=ROLL_SCHEMA)
        if raw_rolls
        else pl.DataFrame(schema=ROLL_SCHEMA)
    )
    return RollResult(
        series,
        rolls,
        segments,
        anchor,
        pl.DataFrame(issues, schema=ISSUE_SCHEMA) if issues else pl.DataFrame(schema=ISSUE_SCHEMA),
    )


def _continuity_ok(series: pl.DataFrame, bars: pl.DataFrame, r: dict[str, object]) -> bool:
    """Adjusted move across the roll == new contract's own move - old contract's (both from t*)."""
    if r["gap_ts"] is None or r["from_adjustment"] is None or r["to_adjustment"] is None:
        return False
    decided, effective = cast(date, r["decision_date"]), cast(date, r["effective_date"])
    old = series.filter(pl.col("trading_date") <= decided).tail(1)
    new = series.filter(pl.col("trading_date") >= effective).head(1)
    if old.is_empty() or new.is_empty():
        return False
    if old["contract"][0] != r["from_contract"] or new["contract"][0] != r["to_contract"]:
        return False

    def raw_close(contract: object, ts: object) -> float | None:
        hit = bars.filter((pl.col("contract") == contract) & (pl.col("ts") == ts))
        return None if hit.is_empty() else float(hit["close"][0])

    t_star = r["gap_ts"]
    old_star, new_star = raw_close(r["from_contract"], t_star), raw_close(r["to_contract"], t_star)
    if old_star is None or new_star is None:
        return False
    jump = float(new["close"][0]) - float(old["close"][0])
    expected = (float(new["raw_close"][0]) - new_star) - (float(old["raw_close"][0]) - old_star)
    return abs(jump - expected) <= _CONTINUITY_EPS
