"""Continuous contract: roll timing, no lookahead, exact continuity, anchoring (ADR 0003)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.loaders import TS_DTYPE
from foundry.domains.nq.quality import flag_bars
from foundry.domains.nq.rolls import build_continuous
from foundry.domains.nq.sessions import annotate_sessions
from foundry.domains.nq.synthetic import SyntheticData, SyntheticSpec, generate
from tests.conftest import QUALITY

HOLDOUT = date(2024, 4, 1)


def prepared(data: SyntheticData, calendar: TradingCalendar) -> pl.DataFrame:
    frames = [
        f.with_columns(pl.lit(code).alias("contract"), pl.lit("syn").alias("source_file"))
        for code, f in data.frames.items()
    ]
    bars = pl.concat(frames).with_columns(pl.col("ts").cast(TS_DTYPE))
    return flag_bars(annotate_sessions(bars, calendar), 0.25, 1, QUALITY)


def basis(data: SyntheticData, code: str) -> float:
    return next(c.basis_ticks for c in data.contracts if c.code == code) * 0.25


def test_adjusted_series_recovers_underlying_exactly(synthetic, calendar) -> None:
    rr = build_continuous(prepared(synthetic, calendar), HOLDOUT)
    assert rr.issues.filter(pl.col("severity") == "fatal").is_empty()
    joined = rr.series.join(synthetic.underlying.select("ts", pl.col("close").alias("u")), on="ts")
    assert joined.height == rr.series.height
    offset = basis(synthetic, rr.anchor)  # type: ignore[arg-type]
    assert ((joined["close"] - joined["u"] - offset).abs() == 0).all()


def test_roll_dates_follow_volume_crossover(synthetic, calendar) -> None:
    rr = build_continuous(prepared(synthetic, calendar), HOLDOUT)
    rolls = rr.rolls
    assert rolls["from_contract"].to_list() == ["NQZ23", "NQH24"]
    assert rolls["to_contract"].to_list() == ["NQH24", "NQM24"]
    assert set(rolls["reason"].to_list()) == {"volume_crossover"}
    session_dates = sorted(rr.series["trading_date"].unique().to_list())
    for r in rolls.iter_rows(named=True):
        c = next(x for x in synthetic.contracts if x.code == r["from_contract"])
        # First trading date on/after the synthetic crossover day decides; next date takes effect.
        decided = min(d for d in session_dates if d >= c.front_until)
        assert r["decision_date"] == decided
        assert r["effective_date"] == session_dates[session_dates.index(decided) + 1]
        assert r["continuity_ok"]


def test_roll_decision_uses_no_future_data(synthetic, calendar) -> None:
    """Truncating the data at any date must not change front-contract choices before it."""
    full = build_continuous(prepared(synthetic, calendar), HOLDOUT).series
    bars = prepared(synthetic, calendar)
    for cut in (date(2023, 12, 8), date(2023, 12, 11), date(2024, 3, 6), date(2024, 3, 8)):
        part = build_continuous(bars.filter(pl.col("trading_date") <= cut), HOLDOUT).series
        a = full.filter(pl.col("trading_date") <= cut).select("ts", "contract")
        b = part.select("ts", "contract")
        assert a.equals(b), f"front contract before {cut} depends on later data"


def test_processed_prices_do_not_depend_on_holdout_data(synthetic, calendar) -> None:
    bars = prepared(synthetic, calendar)
    full = build_continuous(bars, HOLDOUT).series.filter(pl.col("trading_date") < HOLDOUT)
    pre = build_continuous(bars.filter(pl.col("trading_date") < HOLDOUT), HOLDOUT).series
    cols = ["ts", "contract", "open", "high", "low", "close", "adjustment"]
    assert full.select(cols).equals(pre.select(cols))


def test_forward_adjustment_after_anchor(synthetic, calendar) -> None:
    """Holdout boundary before the March roll: H24 anchors and M24 is forward-adjusted."""
    boundary = date(2024, 2, 1)
    rr = build_continuous(prepared(synthetic, calendar), boundary)
    assert rr.anchor == "NQH24"
    joined = rr.series.join(synthetic.underlying.select("ts", pl.col("close").alias("u")), on="ts")
    assert ((joined["close"] - joined["u"] - basis(synthetic, "NQH24")).abs() == 0).all()
    m24 = rr.rolls.filter(pl.col("to_contract") == "NQM24").row(0, named=True)
    assert m24["to_adjustment"] == -m24["gap_points"]


def test_anchor_is_front_at_last_pre_holdout_date(synthetic, calendar) -> None:
    rr = build_continuous(prepared(synthetic, calendar), HOLDOUT)
    assert rr.anchor == "NQM24"
    last_pre = rr.series.filter(pl.col("trading_date") < HOLDOUT).tail(1)
    assert last_pre["adjustment"][0] == 0.0
    assert last_pre["close"][0] == last_pre["raw_close"][0]


def test_no_overlap_roll_is_fatal_not_zero(synthetic, calendar) -> None:
    bars = prepared(synthetic, calendar)
    # Remove the H24 bars that overlap Z23 so the roll gap cannot be measured.
    z_last = bars.filter(pl.col("contract") == "NQZ23")["trading_date"].max()
    bars = bars.filter(~((pl.col("contract") == "NQH24") & (pl.col("trading_date") <= z_last)))
    rr = build_continuous(bars, HOLDOUT)
    fatal = rr.issues.filter(pl.col("severity") == "fatal")
    assert "roll_no_overlap" in fatal["check"].to_list()
    assert rr.rolls["reason"][0] == "forced"
    assert rr.rolls["gap_points"][0] is None
    assert rr.series.filter(pl.col("contract") == "NQZ23")["adjustment"].null_count() > 0


def test_missing_quarter_is_reported(synthetic, calendar) -> None:
    bars = prepared(synthetic, calendar).filter(pl.col("contract") != "NQH24")
    rr = build_continuous(bars, HOLDOUT)
    assert "missing_contract" in rr.issues["check"].to_list()


@settings(
    max_examples=12, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    seed=st.integers(0, 10_000),
    crossover=st.integers(2, 10),
    overlap=st.integers(11, 20),
    start_offset=st.integers(0, 20),
    holdout_offset=st.integers(5, 70),
)
def test_continuity_property(
    calendar, seed, crossover, overlap, start_offset, holdout_offset
) -> None:
    # The holdout boundary moves across the March roll, so both the backward (pre-anchor) and
    # forward (post-anchor) adjustment paths are exercised.
    start = date(2024, 2, 1) + timedelta(days=start_offset)
    spec = SyntheticSpec(
        start=start,
        end=start + timedelta(days=75),
        seed=seed,
        crossover_days=crossover,
        overlap_days=overlap,
    )
    data = generate(spec, calendar)
    rr = build_continuous(prepared(data, calendar), start + timedelta(days=holdout_offset))
    assert rr.issues.filter(pl.col("severity") == "fatal").is_empty()
    assert rr.rolls["continuity_ok"].all()
    joined = rr.series.join(data.underlying.select("ts", pl.col("close").alias("u")), on="ts")
    offset = basis(data, rr.anchor)  # type: ignore[arg-type]
    assert ((joined["close"] - joined["u"] - offset).abs() == 0).all()
    assert rr.series["ts"].is_sorted() and rr.series["ts"].is_unique().all()
