"""Features: hand-computed values, prefix invariance, and the future-shuffle lookahead test."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import EvaluatorConfig, RegimeConfig
from foundry.domains.nq.dsl import FEATURE_KINDS, validate_spec
from foundry.domains.nq.features import MarketContext, build_context, compute_feature
from foundry.domains.nq.loaders import TS_DTYPE
from foundry.domains.nq.quality import flag_bars
from foundry.domains.nq.rolls import build_continuous
from foundry.domains.nq.sessions import annotate_sessions, expected_minutes
from foundry.domains.nq.signals import compile_inputs
from foundry.domains.nq.synthetic import generate
from tests.conftest import EVALUATOR_CFG, QUALITY, SYN_SPEC

REGIMES = RegimeConfig(vol_lookback_days=10, trend_days=3)


def small_bars(calendar: TradingCalendar, seed: int = 1) -> pl.DataFrame:
    grid = expected_minutes(calendar, date(2024, 3, 5), date(2024, 3, 8), 1)
    grid = grid.filter(pl.col("session").is_in(["ETH", "RTH", "POST"]))
    rng = np.random.default_rng(seed)
    n = grid.height
    close = 18000 + np.cumsum(rng.integers(-6, 7, n)) * 0.25
    open_ = np.concatenate([[close[0]], close[:-1]])
    return grid.with_columns(
        pl.Series("open", open_),
        pl.Series("close", close),
        pl.Series("high", np.maximum(open_, close) + rng.integers(0, 4, n) * 0.25),
        pl.Series("low", np.minimum(open_, close) - rng.integers(0, 4, n) * 0.25),
        pl.Series("volume", rng.integers(1, 60, n)),
        pl.lit(0, pl.UInt32).alias("quality_flags"),
        pl.lit("search").alias("split"),
    )


@pytest.fixture(scope="module")
def small(calendar: TradingCalendar) -> MarketContext:
    return build_context(small_bars(calendar), calendar, 1, REGIMES, set())


def rows(ctx: MarketContext) -> list[dict[str, Any]]:
    return ctx.frame.select(
        "ts", "trading_date", "session", "open", "high", "low", "close", "volume", "seg_minute"
    ).to_dicts()


def same(a: np.ndarray, b: list[float] | np.ndarray) -> bool:
    return bool(np.allclose(a, np.asarray(b, float), equal_nan=True, rtol=0, atol=1e-9))


def test_minutes_into_session(small: MarketContext) -> None:
    r = rows(small)
    got = small.feature("minutes_into_session", ())
    for i, x in enumerate(r):
        if x["session"] == "RTH":
            assert got[i] == (x["ts"].hour * 60 + x["ts"].minute) - (9 * 60 + 30)
        elif x["session"] == "ETH":
            m = x["ts"].hour * 60 + x["ts"].minute
            assert got[i] == (m - 18 * 60 if m >= 18 * 60 else m + 6 * 60)
        else:
            assert math.isnan(got[i])


def test_opening_range(small: MarketContext) -> None:
    r = rows(small)
    minutes = 15
    hi, lo = defaultdict(lambda: -math.inf), defaultdict(lambda: math.inf)
    for x in r:
        if x["session"] == "RTH" and x["seg_minute"] < minutes:
            hi[x["trading_date"]] = max(hi[x["trading_date"]], x["high"])
            lo[x["trading_date"]] = min(lo[x["trading_date"]], x["low"])
    exp_hi = [
        hi[x["trading_date"]]
        if x["session"] == "RTH" and x["seg_minute"] >= minutes - 1
        else math.nan
        for x in r
    ]
    exp_lo = [
        lo[x["trading_date"]]
        if x["session"] == "RTH" and x["seg_minute"] >= minutes - 1
        else math.nan
        for x in r
    ]
    assert same(small.feature("opening_range_high", (minutes,)), exp_hi)
    assert same(small.feature("opening_range_low", (minutes,)), exp_lo)


def test_overnight_levels_and_gap(small: MarketContext) -> None:
    r = rows(small)
    on_hi, on_lo = defaultdict(lambda: -math.inf), defaultdict(lambda: math.inf)
    first_open: dict[date, float] = {}
    last_close: dict[date, float] = {}
    for x in r:
        d = x["trading_date"]
        if x["session"] == "ETH":
            on_hi[d] = max(on_hi[d], x["high"])
            on_lo[d] = min(on_lo[d], x["low"])
        if x["session"] == "RTH":
            first_open.setdefault(d, x["open"])
            last_close[d] = x["close"]
    days = sorted(last_close)
    prior = {d: last_close[days[k - 1]] for k, d in enumerate(days) if k}
    rth = [x["session"] == "RTH" for x in r]
    assert same(
        small.feature("overnight_high", ()),
        [on_hi[x["trading_date"]] if m else math.nan for x, m in zip(r, rth, strict=True)],
    )
    assert same(
        small.feature("overnight_low", ()),
        [on_lo[x["trading_date"]] if m else math.nan for x, m in zip(r, rth, strict=True)],
    )
    exp_gap = [
        first_open[x["trading_date"]] - prior[x["trading_date"]]
        if m and x["trading_date"] in prior
        else math.nan
        for x, m in zip(r, rth, strict=True)
    ]
    assert same(small.feature("overnight_gap", ()), exp_gap)
    assert same(
        small.feature("prior_day_close", ()), [prior.get(x["trading_date"], math.nan) for x in r]
    )


def test_vwap_distance(small: MarketContext) -> None:
    r = rows(small)
    exp, pv, v, key = [], 0.0, 0.0, None
    for x in r:
        seg = (x["trading_date"], x["session"]) if x["session"] in ("RTH", "ETH") else None
        if seg is None:
            exp.append(math.nan)
            continue
        if seg != key:
            pv, v, key = 0.0, 0.0, seg
        tp = (x["high"] + x["low"] + x["close"]) / 3
        pv += tp * x["volume"]
        v += x["volume"]
        exp.append(x["close"] - pv / v)
    assert same(small.feature("vwap_distance", ()), exp)


def test_bar_based_features(small: MarketContext) -> None:
    c = small.close
    h, lo = small.high, small.low
    n = 5
    # returns
    exp_ret = np.full(c.size, np.nan)
    exp_ret[n:] = np.log(c[n:] / c[:-n])
    assert same(small.feature("returns", (n,)), exp_ret)
    # Wilder ATR (alpha = 1/n, recursive, first n-1 undefined)
    tr = np.empty(c.size)
    tr[0] = h[0] - lo[0]
    tr[1:] = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])])
    atr = np.empty(c.size)
    atr[0] = tr[0]
    for i in range(1, c.size):
        atr[i] = (1 - 1 / n) * atr[i - 1] + tr[i] / n
    atr[: n - 1] = np.nan
    assert same(small.feature("atr", (n,)), atr)
    # realized vol and z-score over the last n bars (sample std)
    lr = np.concatenate([[np.nan], np.log(c[1:] / c[:-1])])
    rv = np.full(c.size, np.nan)
    z = np.full(c.size, np.nan)
    for i in range(c.size):
        w = lr[i - n + 1 : i + 1] if i >= n - 1 else np.array([])
        if w.size == n and not np.isnan(w).any():
            rv[i] = np.std(w, ddof=1)
        cw = c[i - n + 1 : i + 1] if i >= n - 1 else np.array([])
        if cw.size == n and np.std(cw, ddof=1) > 0:
            z[i] = (c[i] - cw.mean()) / np.std(cw, ddof=1)
    assert same(small.feature("realized_vol", (n,)), rv)
    assert same(small.feature("close_zscore", (n,)), z)


def test_session_volume_ratio(small: MarketContext) -> None:
    r = rows(small)
    cum: dict[tuple[date, str], float] = defaultdict(float)
    by_minute: dict[tuple[str, int], list[float]] = defaultdict(list)
    exp = []
    for x in r:
        if x["session"] not in ("RTH", "ETH"):
            exp.append(math.nan)
            continue
        seg = (x["trading_date"], x["session"])
        cum[seg] += x["volume"]
        hist = by_minute[(x["session"], x["seg_minute"])]
        exp.append(cum[seg] / hist[-1] if hist else math.nan)
        hist.append(cum[seg])
    assert same(small.feature("session_volume_ratio", (1,)), exp)


# ---- no lookahead ----------------------------------------------------------------------


def syn_series(calendar: TradingCalendar) -> pl.DataFrame:
    data = generate(SYN_SPEC, calendar)
    bars = pl.concat(
        [
            f.with_columns(pl.lit(c).alias("contract"), pl.lit("s").alias("source_file"))
            for c, f in data.frames.items()
        ]
    ).with_columns(pl.col("ts").cast(TS_DTYPE))
    series = build_continuous(
        flag_bars(annotate_sessions(bars, calendar), 0.25, 1, QUALITY), SYN_SPEC.end
    ).series
    return series.with_columns(pl.lit("search").alias("split"))


@pytest.fixture(scope="module")
def series(calendar: TradingCalendar) -> pl.DataFrame:
    return syn_series(calendar).filter(pl.col("trading_date") <= date(2024, 1, 31))


ALL_FEATURES = [
    (k, tuple(3 if a != "minutes" else 20 for a in args)) for k, (args, _u) in FEATURE_KINDS.items()
]


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(frac=st.floats(0.05, 0.95))
def test_features_are_prefix_invariant(
    calendar: TradingCalendar, series: pl.DataFrame, frac: float
) -> None:
    cut = int(series.height * frac)
    full = build_context(series, calendar, 1, REGIMES, set())
    part = build_context(series.head(cut), calendar, 1, REGIMES, set())
    for kind, args in ALL_FEATURES:
        a = compute_feature(full.frame, kind, args)[:cut]
        b = compute_feature(part.frame, kind, args)
        assert same(a, b), f"{kind}{args} changes when future bars are removed (cut {cut})"
    assert np.array_equal(full.vol_bucket[:cut], part.vol_bucket)
    assert np.array_equal(full.trend_state[:cut], part.trend_state)
    for name in ("RTH", "ETH", "both"):
        assert np.array_equal(full.windows[name].window_id[:cut], part.windows[name].window_id)


KITCHEN_SINK: dict[str, Any] = {
    "dsl_version": 1,
    "name": "kitchen_sink",
    "session": "both",
    "params": {
        "n": {"type": "int", "low": 2, "high": 50, "value": 10},
        "m": {"type": "int", "low": 5, "high": 60, "value": 20},
        "k": {"type": "float", "low": 0.1, "high": 3.0, "value": 0.5},
        "z": {"type": "float", "low": -3.0, "high": 3.0, "value": 0.0},
    },
    "features": [
        {"id": "atr", "kind": "atr", "args": {"bars": {"param": "n"}}},
        {"id": "orh", "kind": "opening_range_high", "args": {"minutes": {"param": "m"}}},
        {"id": "zs", "kind": "close_zscore", "args": {"bars": {"param": "n"}}},
        {"id": "vw", "kind": "vwap_distance"},
        {"id": "pdh", "kind": "prior_day_high"},
        {"id": "svr", "kind": "session_volume_ratio", "args": {"days": {"param": "n"}}},
    ],
    "entry": {
        "long": {
            "any": [
                {
                    "op": "cross_above",
                    "left": {"price": "close"},
                    "right": {
                        "add": [{"feature": "orh"}, {"mul": [{"param": "k"}, {"feature": "atr"}]}]
                    },
                },
                {
                    "all": [
                        {"op": ">", "left": {"feature": "zs"}, "right": {"param": "z"}},
                        {"op": ">", "left": {"feature": "vw"}, "right": {"feature": "atr"}},
                        {"op": ">", "left": {"feature": "svr"}, "right": {"param": "k"}},
                    ]
                },
            ]
        },
        "short": {"op": "cross_below", "left": {"price": "close"}, "right": {"feature": "pdh"}},
    },
    "exit": {"stop": {"atr_mult": {"param": "k"}, "atr_feature": "atr"}, "session_flat": True},
    "filters": {"days_of_week": ["mon", "tue", "wed", "thu"]},
}


@settings(
    max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(frac=st.floats(0.05, 0.9), seed=st.integers(0, 1000))
def test_shuffling_future_bars_changes_no_signal_up_to_t(
    calendar: TradingCalendar, series: pl.DataFrame, frac: float, seed: int
) -> None:
    """design.md section 4: shuffle bars after t; signals and order inputs up to t don't change."""
    spec = validate_spec(KITCHEN_SINK)
    ev = EvaluatorConfig.model_validate(EVALUATOR_CFG)
    cs = ev.contracts["NQ"]
    t = int(series.height * frac)
    rng = np.random.default_rng(seed)
    perm = np.concatenate([np.arange(t + 1), t + 1 + rng.permutation(series.height - t - 1)])
    price_cols = ["open", "high", "low", "close", "volume"]
    shuffled = series.with_columns(series.select(price_cols)[perm])
    a = compile_inputs(spec, build_context(series, calendar, 1, REGIMES, set()), ev, cs)
    b = compile_inputs(spec, build_context(shuffled, calendar, 1, REGIMES, set()), ev, cs)
    for f in ("long_sig", "short_sig", "stop_dist", "long_px", "short_px"):
        assert same(getattr(a, f)[: t + 1], getattr(b, f)[: t + 1]), f"{f} used future bars"
    # enter_ok looks one bar ahead only through the calendar (is bar t+1 in the same window),
    # never through prices, so it may differ only at t itself.
    assert np.array_equal(a.enter_ok[:t], b.enter_ok[:t])
