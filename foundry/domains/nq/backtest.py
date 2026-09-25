"""Event-driven bar backtester (Numba). The semantics are fixed in ADR 0007.

The kernel knows nothing about specs: it takes precomputed boolean signal arrays and per-bar
distances, and returns trades. Every fill happens on a bar after the one whose close produced
the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

# Order types
MARKET = 0
STOP = 1
LIMIT = 2

# Exit reasons
EXIT_RULE = 1
EXIT_TIME = 2
EXIT_STOP = 3
EXIT_TRAIL = 4
EXIT_TARGET = 5
EXIT_FLAT = 6
EXIT_WINDOW_GAP = 7
EXIT_END_OF_DATA = 8
EXIT_REASONS = {
    EXIT_RULE: "rule",
    EXIT_TIME: "time_stop",
    EXIT_STOP: "stop",
    EXIT_TRAIL: "trailing_stop",
    EXIT_TARGET: "target",
    EXIT_FLAT: "session_flat",
    EXIT_WINDOW_GAP: "window_gap",
    EXIT_END_OF_DATA: "end_of_data",
}


@dataclass(frozen=True)
class KernelInputs:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    window: np.ndarray  # int64: window id, -1 = not tradeable
    flat: np.ndarray  # bool: flatten at this bar's close
    enter_ok: np.ndarray  # bool: an entry decided at this bar's close may be placed
    long_sig: np.ndarray
    short_sig: np.ndarray
    exit_long: np.ndarray
    exit_short: np.ndarray
    stop_dist: np.ndarray  # float64 points, NaN = none (fixed at the signal bar)
    target_dist: np.ndarray
    trail_dist: np.ndarray
    long_px: np.ndarray  # entry level for stop/limit orders, NaN = no order
    short_px: np.ndarray
    order_type: int
    valid_bars: int
    time_stop: int  # bars held before a time exit; 0 = none
    session_flat: bool
    slip_market: float  # in price units
    slip_stop: float
    slip_limit: float
    through: float  # limit trade-through, price units


@dataclass(frozen=True)
class KernelTrades:
    signal_i: np.ndarray
    entry_i: np.ndarray
    exit_i: np.ndarray
    direction: np.ndarray
    entry_px: np.ndarray  # with slippage
    exit_px: np.ndarray
    entry_ref: np.ndarray  # the same fills before slippage (for the net < gross cost test)
    exit_ref: np.ndarray
    reason: np.ndarray
    bars_in_market: int

    @property
    def count(self) -> int:
        return int(self.entry_i.shape[0])


def run(k: KernelInputs) -> KernelTrades:
    n = k.open.shape[0]
    if n == 0:
        z = np.zeros(0, np.int64)
        f = np.zeros(0, np.float64)
        return KernelTrades(z, z, z, z, f, f, f, f, z, 0)
    out_i = np.zeros((n, 5), np.int64)
    out_f = np.zeros((n, 4), np.float64)
    m, in_mkt = _kernel(
        k.open,
        k.high,
        k.low,
        k.close,
        k.window,
        k.flat,
        k.enter_ok,
        k.long_sig,
        k.short_sig,
        k.exit_long,
        k.exit_short,
        k.stop_dist,
        k.target_dist,
        k.trail_dist,
        k.long_px,
        k.short_px,
        k.order_type,
        k.valid_bars,
        k.time_stop,
        k.session_flat,
        k.slip_market,
        k.slip_stop,
        k.slip_limit,
        k.through,
        out_i,
        out_f,
    )
    return KernelTrades(
        signal_i=out_i[:m, 0].copy(),
        entry_i=out_i[:m, 1].copy(),
        exit_i=out_i[:m, 2].copy(),
        direction=out_i[:m, 3].copy(),
        reason=out_i[:m, 4].copy(),
        entry_px=out_f[:m, 0].copy(),
        exit_px=out_f[:m, 1].copy(),
        entry_ref=out_f[:m, 2].copy(),
        exit_ref=out_f[:m, 3].copy(),
        bars_in_market=int(in_mkt),
    )


@nb.njit(cache=True)
def _kernel(  # type: ignore[no-untyped-def]
    o,
    h,
    lo,
    c,
    window,
    flat,
    enter_ok,
    long_sig,
    short_sig,
    exit_long,
    exit_short,
    stop_dist,
    target_dist,
    trail_dist,
    long_px,
    short_px,
    order_type,
    valid_bars,
    time_stop,
    session_flat,
    slip_mkt,
    slip_stop,
    slip_lim,
    through,
    out_i,
    out_f,
):
    n = o.shape[0]
    m = 0
    in_mkt = 0
    pos = 0
    pos_window = -1
    e_i = -1
    sig_i = -1
    e_px = 0.0
    e_ref = 0.0
    stop_lvl = np.nan
    tgt_lvl = np.nan
    trail_d = np.nan
    ext = 0.0
    held = 0
    pending_exit = 0  # reason code of a market exit for this bar's open, 0 = none
    # pending entry
    pe_dir = 0
    pe_px = np.nan
    pe_expire = -1
    pe_window = -1
    pe_sig = -1

    for i in range(n):
        exit_px = np.nan
        exit_ref = np.nan
        reason = 0

        # A. position still open when its window has ended without a flat bar (data gap)
        if pos != 0 and session_flat and window[i] != pos_window:
            exit_ref = o[i]
            exit_px = o[i] - pos * slip_mkt
            reason = 7
        # B. market exit decided at the previous close
        elif pos != 0 and pending_exit != 0:
            exit_ref = o[i]
            exit_px = o[i] - pos * slip_mkt
            reason = pending_exit
        if reason != 0:
            out_i[m, 0] = sig_i
            out_i[m, 1] = e_i
            out_i[m, 2] = i
            out_i[m, 3] = pos
            out_i[m, 4] = reason
            out_f[m, 0] = e_px
            out_f[m, 1] = exit_px
            out_f[m, 2] = e_ref
            out_f[m, 3] = exit_ref
            m += 1
            pos = 0
            pending_exit = 0

        # C. pending entry order
        if pos == 0 and pe_dir != 0:
            filled = False
            fill = 0.0
            ref = 0.0
            if window[i] != pe_window or flat[i]:
                pe_dir = 0
            elif order_type == 0:
                ref = o[i]
                fill = ref + pe_dir * slip_mkt
                filled = True
            elif order_type == 1:
                if pe_dir == 1 and h[i] >= pe_px:
                    ref = max(o[i], pe_px)
                    fill = ref + slip_stop
                    filled = True
                elif pe_dir == -1 and lo[i] <= pe_px:
                    ref = min(o[i], pe_px)
                    fill = ref - slip_stop
                    filled = True
            elif pe_dir == 1 and lo[i] <= pe_px - through:
                ref = min(o[i], pe_px)
                fill = ref + slip_lim
                filled = True
            elif pe_dir == -1 and h[i] >= pe_px + through:
                ref = max(o[i], pe_px)
                fill = ref - slip_lim
                filled = True
            if filled:
                pos = pe_dir
                e_i = i
                sig_i = pe_sig
                e_px = fill
                e_ref = ref
                sd = stop_dist[pe_sig]
                td = target_dist[pe_sig]
                trail_d = trail_dist[pe_sig]
                stop_lvl = e_ref - pos * sd if not np.isnan(sd) else np.nan
                tgt_lvl = e_ref + pos * td if not np.isnan(td) else np.nan
                ext = e_ref
                held = 0
                pos_window = window[i]
                pe_dir = 0
            elif pe_dir != 0 and i >= pe_expire:
                pe_dir = 0

        # D. intrabar protective stop (checked before the target) and target
        if pos != 0:
            eff = stop_lvl
            is_trail = False
            if not np.isnan(trail_d):
                tl = ext - pos * trail_d
                if np.isnan(eff) or (pos == 1 and tl > eff) or (pos == -1 and tl < eff):
                    eff = tl
                    is_trail = True
            hit = False
            if not np.isnan(eff):
                if pos == 1 and lo[i] <= eff:
                    exit_ref = min(o[i], eff)
                    exit_px = exit_ref - slip_stop
                    hit = True
                elif pos == -1 and h[i] >= eff:
                    exit_ref = max(o[i], eff)
                    exit_px = exit_ref + slip_stop
                    hit = True
                if hit:
                    reason = 4 if is_trail else 3
            if not hit and not np.isnan(tgt_lvl):
                if pos == 1 and h[i] >= tgt_lvl + through:
                    exit_ref = max(o[i], tgt_lvl)
                    exit_px = exit_ref - slip_lim
                    hit = True
                elif pos == -1 and lo[i] <= tgt_lvl - through:
                    exit_ref = min(o[i], tgt_lvl)
                    exit_px = exit_ref + slip_lim
                    hit = True
                if hit:
                    reason = 5
            if hit:
                in_mkt += 1
                out_i[m, 0] = sig_i
                out_i[m, 1] = e_i
                out_i[m, 2] = i
                out_i[m, 3] = pos
                out_i[m, 4] = reason
                out_f[m, 0] = e_px
                out_f[m, 1] = exit_px
                out_f[m, 2] = e_ref
                out_f[m, 3] = exit_ref
                m += 1
                pos = 0

        # E. close of bar with an open position
        if pos != 0:
            in_mkt += 1
            held += 1
            close_out = 0
            if session_flat and flat[i] and window[i] == pos_window:
                close_out = 6
            elif i == n - 1:
                close_out = 8
            if close_out != 0:
                out_i[m, 0] = sig_i
                out_i[m, 1] = e_i
                out_i[m, 2] = i
                out_i[m, 3] = pos
                out_i[m, 4] = close_out
                out_f[m, 0] = e_px
                out_f[m, 1] = c[i] - pos * slip_mkt
                out_f[m, 2] = e_ref
                out_f[m, 3] = c[i]
                m += 1
                pos = 0
            else:
                if pos == 1 and h[i] > ext:
                    ext = h[i]
                elif pos == -1 and lo[i] < ext:
                    ext = lo[i]
                if (pos == 1 and exit_long[i]) or (pos == -1 and exit_short[i]):
                    pending_exit = 1
                elif time_stop > 0 and held >= time_stop:
                    pending_exit = 2

        # F. new entry decision at this bar's close
        if pos == 0 and pe_dir == 0 and enter_ok[i] and i + 1 < n:
            d = 0
            if long_sig[i] and not short_sig[i]:
                d = 1
            elif short_sig[i] and not long_sig[i]:
                d = -1
            if d != 0:
                px = long_px[i] if d == 1 else short_px[i]
                if order_type == 0 or not np.isnan(px):
                    pe_dir = d
                    pe_px = px
                    pe_expire = i + valid_bars
                    pe_window = window[i]
                    pe_sig = i
    return m, in_mkt
