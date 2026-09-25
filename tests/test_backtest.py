"""Backtest kernel: golden hand-computed trades (ADR 0007), properties, costs, no lookahead."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from foundry.domains.nq.backtest import (
    EXIT_FLAT,
    EXIT_RULE,
    EXIT_STOP,
    EXIT_TARGET,
    EXIT_TIME,
    EXIT_TRAIL,
    EXIT_WINDOW_GAP,
    LIMIT,
    STOP,
    KernelInputs,
    KernelTrades,
    run,
)

TICK = 0.25


def inputs(
    o: list[float], h: list[float], lo: list[float], c: list[float], **kw: Any
) -> KernelInputs:
    n = len(o)
    z = np.zeros(n, bool)
    nan = np.full(n, np.nan)
    base = KernelInputs(
        open=np.array(o, float),
        high=np.array(h, float),
        low=np.array(lo, float),
        close=np.array(c, float),
        window=np.zeros(n, np.int64),
        flat=z.copy(),
        enter_ok=np.ones(n, bool),
        long_sig=z.copy(),
        short_sig=z.copy(),
        exit_long=z.copy(),
        exit_short=z.copy(),
        stop_dist=nan.copy(),
        target_dist=nan.copy(),
        trail_dist=nan.copy(),
        long_px=nan.copy(),
        short_px=nan.copy(),
        order_type=0,
        valid_bars=1,
        time_stop=0,
        session_flat=True,
        slip_market=TICK,
        slip_stop=TICK,
        slip_limit=0.0,
        through=TICK,
    )
    return replace(base, **kw)


def at(n: int, idx: list[int], value: Any = True, dtype: Any = bool) -> np.ndarray:
    arr = np.zeros(n, dtype) if dtype is bool else np.full(n, np.nan)
    for i in idx:
        arr[i] = value
    return arr


def only(t: KernelTrades) -> dict[str, Any]:
    assert t.count == 1, f"expected one trade, got {t.count}"
    return {
        "signal": int(t.signal_i[0]),
        "entry": int(t.entry_i[0]),
        "exit": int(t.exit_i[0]),
        "dir": int(t.direction[0]),
        "entry_px": float(t.entry_px[0]),
        "exit_px": float(t.exit_px[0]),
        "entry_ref": float(t.entry_ref[0]),
        "exit_ref": float(t.exit_ref[0]),
        "reason": int(t.reason[0]),
    }


FLAT = [100.0] * 6


def test_market_entry_next_open_and_rule_exit_next_open() -> None:
    o = [100, 101, 102, 103, 104, 105]
    k = inputs(
        o,
        [x + 1 for x in o],
        [x - 1 for x in o],
        [x + 0.5 for x in o],
        long_sig=at(6, [1]),
        exit_long=at(6, [3]),
    )
    tr = only(run(k))
    # decided at close of bar 1 -> filled at open of bar 2 (+1 tick); exit decided at bar 3 -> bar 4
    assert tr == {
        "signal": 1,
        "entry": 2,
        "exit": 4,
        "dir": 1,
        "entry_px": 102.25,
        "exit_px": 103.75,
        "entry_ref": 102.0,
        "exit_ref": 104.0,
        "reason": EXIT_RULE,
    }


def test_short_entry_mirrors_slippage() -> None:
    o = [100, 100, 100, 99, 98, 97]
    k = inputs(
        o, [x + 1 for x in o], [x - 1 for x in o], o, short_sig=at(6, [0]), exit_short=at(6, [3])
    )
    tr = only(run(k))
    assert (tr["entry_px"], tr["exit_px"], tr["dir"]) == (99.75, 98.25, -1)


def test_stop_gap_through_fills_at_open() -> None:
    o, h, lo, c = [100, 100, 93, 93], [101, 101, 94, 94], [99, 99, 92, 92], [100, 100, 93, 93]
    k = inputs(o, h, lo, c, long_sig=at(4, [0]), stop_dist=at(4, [0], 5.0, float))
    tr = only(run(k))
    # entry ref 100 -> stop level 95; bar 2 opens at 93, below the stop -> fill 93 - 1 tick
    assert (tr["entry"], tr["exit"], tr["exit_ref"], tr["exit_px"], tr["reason"]) == (
        1,
        2,
        93.0,
        92.75,
        EXIT_STOP,
    )


def test_target_needs_trade_through() -> None:
    o = [100, 100, 105, 108]
    lo = [99, 99, 104, 107]
    c = [100, 100, 106, 109]
    touch = inputs(
        o,
        [101, 101, 110, 110],
        lo,
        c,
        long_sig=at(4, [0]),
        target_dist=at(4, [0], 10.0, float),
        session_flat=False,
    )
    assert run(touch).reason.tolist() == [8]  # only touched 110: no fill, closed at end
    through = inputs(
        o, [101, 101, 110.25, 110], lo, c, long_sig=at(4, [0]), target_dist=at(4, [0], 10.0, float)
    )
    tr = only(run(through))
    assert (tr["exit"], tr["exit_px"], tr["reason"]) == (2, 110.0, EXIT_TARGET)  # no slippage


def test_stop_wins_when_stop_and_target_share_a_bar() -> None:
    k = inputs(
        [100, 100, 100],
        [101, 101, 120],
        [99, 99, 80],
        [100, 100, 100],
        long_sig=at(3, [0]),
        stop_dist=at(3, [0], 5.0, float),
        target_dist=at(3, [0], 5.0, float),
    )
    tr = only(run(k))
    assert (tr["reason"], tr["exit_ref"]) == (EXIT_STOP, 95.0)


def test_session_flat_at_close_of_flat_bar() -> None:
    k = inputs(
        FLAT,
        [101] * 6,
        [99] * 6,
        [100, 100, 100, 100, 100.5, 100],
        long_sig=at(6, [0]),
        flat=at(6, [4]),
    )
    tr = only(run(k))
    assert (tr["exit"], tr["exit_ref"], tr["exit_px"], tr["reason"]) == (
        4,
        100.5,
        100.25,
        EXIT_FLAT,
    )


def test_no_entry_that_would_open_on_the_flat_bar_when_caller_blocks_it() -> None:
    # compile_inputs blocks entries whose next bar is the flat bar; the kernel also refuses to
    # fill a pending entry on a flat bar.
    k = inputs(FLAT, [101] * 6, [99] * 6, FLAT, long_sig=at(6, [3]), flat=at(6, [4]))
    assert run(k).count == 0


def test_time_stop_counts_closes_from_entry_bar() -> None:
    k = inputs(FLAT, [101] * 6, [99] * 6, FLAT, long_sig=at(6, [0]), time_stop=2)
    tr = only(run(k))
    # entry at bar 1; closes counted at bars 1 and 2 -> market exit at open of bar 3
    assert (tr["entry"], tr["exit"], tr["reason"]) == (1, 3, EXIT_TIME)


def test_trailing_stop_follows_completed_highs() -> None:
    o = [100, 100, 101, 103, 102]
    h = [100.5, 101, 104, 103.5, 102.5]
    lo = [99.5, 99.5, 100.5, 102, 100.5]
    c = [100, 100.5, 103.5, 103, 101]
    k = inputs(o, h, lo, c, long_sig=at(5, [0]), trail_dist=at(5, [0], 3.0, float))
    tr = only(run(k))
    # highs through bar 2 reach 104 -> trail 101; bar 3 low 102 holds; bar 4 low 100.5 hits
    assert (tr["exit"], tr["exit_ref"], tr["exit_px"], tr["reason"]) == (
        4,
        101.0,
        100.75,
        EXIT_TRAIL,
    )


def test_stop_entry_order_valid_for_n_bars() -> None:
    o, h = [100, 100, 101, 104], [100.5, 104, 106, 107]
    lo, c = [99.5, 99.5, 100.5, 103], [100, 101, 105, 106]
    k = inputs(
        o,
        h,
        lo,
        c,
        long_sig=at(4, [0]),
        long_px=at(4, [0], 105.0, float),
        order_type=STOP,
        valid_bars=2,
        session_flat=False,
    )
    t = run(k)
    # bar 1 high 104 < 105; bar 2 high 106 triggers: fill max(101, 105) + 1 tick
    assert (int(t.entry_i[0]), float(t.entry_px[0]), float(t.entry_ref[0])) == (2, 105.25, 105.0)
    expired = inputs(
        o,
        h,
        lo,
        c,
        long_sig=at(4, [0]),
        long_px=at(4, [0], 105.0, float),
        order_type=STOP,
        valid_bars=1,
    )
    assert run(expired).count == 0


def test_limit_entry_needs_trade_through() -> None:
    o, h, c = [100, 100, 99], [100.5, 100.5, 99.5], [100, 99.5, 99]
    k = inputs(
        o,
        h,
        [99.5, 98.0, 98.75],
        c,
        long_sig=at(3, [0]),
        long_px=at(3, [0], 98.0, float),
        order_type=LIMIT,
        valid_bars=2,
        session_flat=False,
    )
    assert run(k).count == 0  # low 98.0 only touches the 98 limit
    k2 = replace(k, low=np.array([99.5, 97.75, 98.75]))
    t = run(k2)
    assert (int(t.entry_i[0]), float(t.entry_px[0])) == (1, 98.0)


def test_window_change_without_flat_bar_exits_at_next_open() -> None:
    k = inputs(
        FLAT,
        [101] * 6,
        [99] * 6,
        FLAT,
        long_sig=at(6, [0]),
        window=np.array([0, 0, 0, 1, 1, 1], np.int64),
    )
    tr = only(run(k))
    assert (tr["exit"], tr["reason"], tr["exit_px"]) == (3, EXIT_WINDOW_GAP, 99.75)


def test_entries_respect_enter_ok_and_no_pyramiding() -> None:
    # Signals at 1 and 2 arrive while long (ignored); the exit fills at bar 3's open, so the
    # signal at the close of bar 3 re-enters at bar 4.
    sig = at(6, [0, 1, 2, 3])
    k = inputs(FLAT, [101] * 6, [99] * 6, FLAT, long_sig=sig, exit_long=at(6, [2]))
    t = run(k)
    assert t.entry_i.tolist() == [1, 4]
    blocked = replace(k, enter_ok=np.zeros(6, bool))
    assert run(blocked).count == 0


# ---- properties -------------------------------------------------------------------------


@st.composite
def random_market(draw: Any) -> KernelInputs:
    n = draw(st.integers(20, 120))
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    close = 1000 + np.cumsum(rng.integers(-8, 9, n)) * TICK
    open_ = np.concatenate([[close[0]], close[:-1]]) + rng.integers(-2, 3, n) * TICK
    high = np.maximum(open_, close) + rng.integers(0, 6, n) * TICK
    low = np.minimum(open_, close) - rng.integers(0, 6, n) * TICK
    window = np.cumsum(rng.random(n) < 0.05).astype(np.int64)
    flat = np.zeros(n, bool)
    flat[:-1] = window[1:] != window[:-1]

    def dist() -> np.ndarray:
        return np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 40, n) * TICK)

    return inputs(
        list(open_),
        list(high),
        list(low),
        list(close),
        window=window,
        flat=flat,
        long_sig=rng.random(n) < 0.15,
        short_sig=rng.random(n) < 0.15,
        exit_long=rng.random(n) < 0.05,
        exit_short=rng.random(n) < 0.05,
        stop_dist=dist(),
        target_dist=dist(),
        trail_dist=dist(),
        long_px=close + rng.integers(-8, 9, n) * TICK,
        short_px=close + rng.integers(-8, 9, n) * TICK,
        order_type=int(draw(st.sampled_from([0, 1, 2]))),
        valid_bars=int(draw(st.integers(1, 4))),
        time_stop=int(draw(st.integers(0, 10))),
        session_flat=draw(st.booleans()),
    )


@settings(max_examples=200, deadline=None)
@given(k=random_market())
def test_trade_invariants(k: KernelInputs) -> None:
    t = run(k)
    for j in range(t.count):
        e, x, s = int(t.entry_i[j]), int(t.exit_i[j]), int(t.signal_i[j])
        assert s < e, "filled on the signal bar"
        assert e <= x
        assert k.low[e] <= t.entry_ref[j] <= k.high[e], "entry outside its bar"
        assert k.low[x] <= t.exit_ref[j] <= k.high[x], "exit outside its bar"
        if j:
            assert e > int(t.exit_i[j - 1]), "overlapping positions"
        # costs never help: slippage makes every fill worse than its reference
        d = int(t.direction[j])
        assert d * (t.exit_px[j] - t.entry_px[j]) <= d * (t.exit_ref[j] - t.entry_ref[j])
    assert 0 <= t.bars_in_market <= k.open.shape[0]


@settings(max_examples=100, deadline=None)
@given(k=random_market(), cut=st.floats(0.3, 0.9))
def test_kernel_has_no_lookahead(k: KernelInputs, cut: float) -> None:
    """Trades that are complete before the cut are identical when later bars don't exist."""
    n = k.open.shape[0]
    t_cut = int(n * cut)
    trimmed = replace(
        k,
        **{
            f: getattr(k, f)[:t_cut]
            for f in (
                "open",
                "high",
                "low",
                "close",
                "window",
                "flat",
                "enter_ok",
                "long_sig",
                "short_sig",
                "exit_long",
                "exit_short",
                "stop_dist",
                "target_dist",
                "trail_dist",
                "long_px",
                "short_px",
            )
        },
    )
    full, part = run(k), run(trimmed)
    done_full = full.exit_i < t_cut - 1
    done_part = part.exit_i < t_cut - 1
    for field in ("signal_i", "entry_i", "exit_i", "direction", "entry_px", "exit_px", "reason"):
        assert np.array_equal(getattr(full, field)[done_full], getattr(part, field)[done_part])


@settings(max_examples=60, deadline=None)
@given(k=random_market())
def test_zero_costs_make_fills_equal_references(k: KernelInputs) -> None:
    free = replace(k, slip_market=0.0, slip_stop=0.0, slip_limit=0.0)
    t = run(free)
    assert np.array_equal(t.entry_px, t.entry_ref)
    assert np.array_equal(t.exit_px, t.exit_ref)


def test_more_slippage_never_improves_pnl() -> None:
    rng = np.random.default_rng(3)
    n = 400
    close = 1000 + np.cumsum(rng.integers(-8, 9, n)) * TICK
    k = inputs(
        list(close),
        list(close + 1),
        list(close - 1),
        list(close),
        long_sig=rng.random(n) < 0.1,
        exit_long=rng.random(n) < 0.1,
    )
    base = run(k)
    doubled = run(replace(k, slip_market=2 * TICK, slip_stop=2 * TICK))
    pnl = lambda t: float(np.sum(t.direction * (t.exit_px - t.entry_px)))  # noqa: E731
    assert base.count == doubled.count
    assert pnl(doubled) < pnl(base)
