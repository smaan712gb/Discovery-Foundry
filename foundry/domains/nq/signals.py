"""Compile a validated spec into backtest-kernel inputs over a MarketContext."""

from __future__ import annotations

from typing import Any

import numpy as np

from foundry.domains.nq.backtest import LIMIT, MARKET, STOP, KernelInputs
from foundry.domains.nq.config import ContractSpec, EvaluatorConfig
from foundry.domains.nq.dsl import FEATURE_KINDS, Spec
from foundry.domains.nq.features import MarketContext

_ORDER_TYPES = {"market": MARKET, "stop": STOP, "limit": LIMIT}


class Compiler:
    def __init__(self, spec: Spec, ctx: MarketContext) -> None:
        self.spec = spec
        self.ctx = ctx
        self.raw = spec.raw
        self.features = {f["id"]: f for f in self.raw["features"]}

    # ---- operands and rules ------------------------------------------------------------

    def feature(self, fid: str) -> np.ndarray:
        f = self.features[fid]
        names, _unit = FEATURE_KINDS[f["kind"]]
        args = tuple(self.spec.int_param(f["args"][a]["param"]) for a in names)
        return self.ctx.feature(f["kind"], args)

    def operand(self, op: dict[str, Any]) -> np.ndarray:
        n = self.ctx.n
        if "feature" in op:
            return self.feature(op["feature"])
        if "param" in op:
            return np.full(n, self.spec.param(op["param"]), np.float64)
        if "price" in op:
            return {
                "open": self.ctx.open,
                "high": self.ctx.high,
                "low": self.ctx.low,
                "close": self.ctx.close,
            }[op["price"]]
        key = next(k for k in ("add", "sub", "mul") if k in op)
        a, b = self.operand(op[key][0]), self.operand(op[key][1])
        out: np.ndarray = {"add": a + b, "sub": a - b, "mul": a * b}[key]
        return out

    def rule(self, r: dict[str, Any]) -> np.ndarray:
        if "all" in r:
            out = np.ones(self.ctx.n, bool)
            for sub in r["all"]:
                out &= self.rule(sub)
            return out
        if "any" in r:
            out = np.zeros(self.ctx.n, bool)
            for sub in r["any"]:
                out |= self.rule(sub)
            return out
        if "not" in r:
            # NaN-driven False stays False under `not`: an undefined comparison never trades.
            inner = r["not"]
            return np.asarray(~self.rule(inner) & self._defined(inner), dtype=bool)
        left, right = self.operand(r["left"]), self.operand(r["right"])
        op = r["op"]
        with np.errstate(invalid="ignore"):
            if op in (">", "<", ">=", "<="):
                res = {
                    ">": left > right,
                    "<": left < right,
                    ">=": left >= right,
                    "<=": left <= right,
                }[op]
                return np.asarray(res & ~np.isnan(left) & ~np.isnan(right), bool)
            pl_, pr = np.roll(left, 1), np.roll(right, 1)
            pl_[0] = np.nan
            pr[0] = np.nan
            if op == "cross_above":
                res = (pl_ <= pr) & (left > right)
            else:
                res = (pl_ >= pr) & (left < right)
            ok = ~np.isnan(left) & ~np.isnan(right) & ~np.isnan(pl_) & ~np.isnan(pr)
            return np.asarray(res & ok, bool)

    def _defined(self, r: dict[str, Any]) -> np.ndarray:
        """Bars where every operand inside a rule is defined (not NaN)."""
        if "all" in r or "any" in r:
            out = np.ones(self.ctx.n, bool)
            for sub in r.get("all", r.get("any", [])):
                out &= self._defined(sub)
            return out
        if "not" in r:
            return self._defined(r["not"])
        return ~np.isnan(self.operand(r["left"])) & ~np.isnan(self.operand(r["right"]))

    def distance(self, d: dict[str, Any] | None) -> np.ndarray:
        if d is None:
            return np.full(self.ctx.n, np.nan)
        if "points" in d:
            return np.full(self.ctx.n, self.spec.param(d["points"]["param"]))
        return self.spec.param(d["atr_mult"]["param"]) * self.feature(d["atr_feature"])

    # ---- filters -----------------------------------------------------------------------

    def filters(self) -> np.ndarray:
        f = self.raw.get("filters", {})
        ctx = self.ctx
        ok = np.ones(ctx.n, bool)
        if "volatility_buckets" in f:
            ok &= np.isin(ctx.vol_bucket, f["volatility_buckets"])
        if "trend_states" in f:
            ok &= np.isin(ctx.trend_state, f["trend_states"])
        if "days_of_week" in f:
            ok &= np.isin(ctx.dow, f["days_of_week"])
        if f.get("exclude_event_days"):
            ok &= ~ctx.event_day
        return ok


def compile_inputs(
    spec: Spec,
    ctx: MarketContext,
    ev: EvaluatorConfig,
    contract: ContractSpec,
    trade_mask: np.ndarray | None = None,
    slippage_multiplier: float = 1.0,
) -> KernelInputs:
    """`trade_mask`: bars on which entries may be decided (e.g. validation bars only, with the
    search bars before them present as feature warm-up)."""
    comp = Compiler(spec, ctx)
    raw = spec.raw
    win = ctx.windows[spec.session]
    n = ctx.n
    nxt_same = np.zeros(n, bool)
    if n > 1:
        nxt_same[:-1] = (win.window_id[1:] == win.window_id[:-1]) & ~win.flat[1:]
    enter_ok = (win.window_id >= 0) & nxt_same & ~win.flat & comp.filters()
    enter_ok &= ~ctx.blocked(list(ev.block_signals_on_flags))
    if trade_mask is not None:
        enter_ok &= trade_mask

    entry, ex = raw["entry"], raw["exit"]
    none = np.zeros(n, bool)
    order = entry.get("order", {"type": "market"})
    otype = _ORDER_TYPES[order["type"]]
    nan = np.full(n, np.nan)
    tick = contract.tick_size
    s = ev.slippage_ticks
    return KernelInputs(
        open=ctx.open,
        high=ctx.high,
        low=ctx.low,
        close=ctx.close,
        window=win.window_id,
        flat=win.flat,
        enter_ok=enter_ok,
        long_sig=comp.rule(entry["long"]) if "long" in entry else none,
        short_sig=comp.rule(entry["short"]) if "short" in entry else none,
        exit_long=comp.rule(ex["rule_long"]) if "rule_long" in ex else none,
        exit_short=comp.rule(ex["rule_short"]) if "rule_short" in ex else none,
        stop_dist=comp.distance(ex.get("stop")),
        target_dist=comp.distance(ex.get("target")),
        trail_dist=comp.distance(ex.get("trailing")),
        long_px=comp.operand(order["long_price"]) if "long_price" in order else nan,
        short_px=comp.operand(order["short_price"]) if "short_price" in order else nan,
        order_type=otype,
        valid_bars=spec.int_param(order["valid_bars"]["param"]) if otype != MARKET else 1,
        time_stop=spec.int_param(ex["time_stop_bars"]["param"]) if "time_stop_bars" in ex else 0,
        session_flat=bool(ex.get("session_flat", True)),
        slip_market=s.market * tick * slippage_multiplier,
        slip_stop=s.stop * tick * slippage_multiplier,
        slip_limit=s.limit * tick * slippage_multiplier,
        through=ev.limit_trade_through_ticks * tick,
    )
