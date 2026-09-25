"""The NQ spec space: random specs, mutation, crossover, surrogate features, CMA view, LLM text.

The generic search (foundry/search) drives these operators; every spec they produce passes
`validate_spec` before it can be evaluated. Operators work on the raw JSON dict and finish with
`prune` (drop unreferenced params and features, which validation would reject).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from foundry.domains.nq.dsl import FEATURE_KINDS, SCHEMA_PATH, Spec, SpecError, validate_spec

Raw = dict[str, Any]

# Scalar features compared with a bounded threshold: kind -> (int arg, arg range, centre, scale).
THRESHOLD_FEATURES: dict[str, tuple[str | None, tuple[int, int], float, float]] = {
    "close_zscore": ("bars", (5, 240), 0.0, 2.0),
    "returns": ("bars", (5, 240), 0.0, 0.003),
    "realized_vol": ("bars", (10, 240), 0.0006, 0.0004),
    "vwap_distance": (None, (0, 0), 0.0, 20.0),
    "session_volume_ratio": ("days", (2, 10), 1.0, 0.5),
    "overnight_gap": (None, (0, 0), 0.0, 40.0),
}
# Price levels the close can cross.
LEVEL_FEATURES: dict[str, tuple[str | None, tuple[int, int]]] = {
    "opening_range_high": ("minutes", (5, 90)),
    "opening_range_low": ("minutes", (5, 90)),
    "overnight_high": (None, (0, 0)),
    "overnight_low": (None, (0, 0)),
    "prior_day_high": (None, (0, 0)),
    "prior_day_low": (None, (0, 0)),
    "prior_day_close": (None, (0, 0)),
}
SESSIONS = ("RTH", "ETH", "both")
OP_FLIP = {
    ">": "<",
    "<": ">",
    ">=": "<=",
    "<=": ">=",
    "cross_above": "cross_below",
    "cross_below": "cross_above",
}
FEATURE_MEANINGS = {
    "returns": "log return over the last N bars (unitless)",
    "atr": "Wilder average true range over N bars (points)",
    "vwap_distance": "close minus session VWAP (points)",
    "opening_range_high": "high of the first N minutes of RTH, once complete (price)",
    "opening_range_low": "low of the first N minutes of RTH (price)",
    "overnight_high": "high of the overnight ETH session, on RTH bars (price)",
    "overnight_low": "low of the overnight ETH session, on RTH bars (price)",
    "overnight_gap": "first RTH open minus prior RTH close (points)",
    "session_volume_ratio": "session volume so far vs its N-session average at that minute",
    "realized_vol": "std of 1-minute log returns over N bars (unitless)",
    "prior_day_high": "previous RTH day's high (price)",
    "prior_day_low": "previous RTH day's low (price)",
    "prior_day_close": "previous RTH day's last close (price)",
    "close_zscore": "(close - N-bar mean) / N-bar std (unitless)",
    "minutes_into_session": "minutes since the session opened (RTH 09:30, ETH 18:00)",
}


# ---- reference bookkeeping ----------------------------------------------------------------


def _walk(node: Any, fn: Callable[[dict[str, Any]], None]) -> None:
    if isinstance(node, dict):
        fn(node)
        for v in node.values():
            _walk(v, fn)
    elif isinstance(node, list):
        for v in node:
            _walk(v, fn)


def references(raw: Raw) -> tuple[set[str], set[str]]:
    """(params referenced, features referenced) anywhere outside the params block."""
    params: set[str] = set()
    feats: set[str] = set()

    def visit(d: dict[str, Any]) -> None:
        if isinstance(d.get("param"), str):
            params.add(d["param"])
        if isinstance(d.get("feature"), str):
            feats.add(d["feature"])
        if isinstance(d.get("atr_feature"), str):
            feats.add(d["atr_feature"])

    body = {k: v for k, v in raw.items() if k not in ("params", "features")}
    _walk(body, visit)
    for f in raw["features"]:
        if f["id"] in feats:
            _walk(f.get("args", {}), visit)
    return params, feats


def prune(raw: Raw) -> Raw:
    params, feats = references(raw)
    raw["features"] = [f for f in raw["features"] if f["id"] in feats]
    params, _ = references(raw)
    raw["params"] = {k: v for k, v in raw["params"].items() if k in params}
    return raw


def _fresh(raw: Raw, prefix: str) -> str:
    taken = set(raw["params"]) | {f["id"] for f in raw["features"]}
    i = 0
    while f"{prefix}{i}" in taken:
        i += 1
    return f"{prefix}{i}"


def _rename(raw: Raw, pmap: dict[str, str], fmap: dict[str, str]) -> Raw:
    def visit(d: dict[str, Any]) -> None:
        if isinstance(d.get("param"), str) and d["param"] in pmap:
            d["param"] = pmap[d["param"]]
        if isinstance(d.get("feature"), str) and d["feature"] in fmap:
            d["feature"] = fmap[d["feature"]]
        if isinstance(d.get("atr_feature"), str) and d["atr_feature"] in fmap:
            d["atr_feature"] = fmap[d["atr_feature"]]

    _walk({k: v for k, v in raw.items() if k != "params"}, visit)
    for f in raw["features"]:
        f["id"] = fmap.get(f["id"], f["id"])
    raw["params"] = {pmap.get(k, k): v for k, v in raw["params"].items()}
    return raw


# ---- building blocks ------------------------------------------------------------------------


def _new_condition(raw: Raw, rng: np.random.Generator) -> dict[str, Any]:
    """A random condition, adding the features and params it needs to `raw`."""
    if rng.random() < 0.6:
        kind = list(THRESHOLD_FEATURES)[int(rng.integers(len(THRESHOLD_FEATURES)))]
        arg, (lo, hi), centre, scale = THRESHOLD_FEATURES[kind]
        fid = _fresh(raw, "f")
        feat: dict[str, Any] = {"id": fid, "kind": kind}
        raw["features"].append(feat)
        if arg:
            pn = _fresh(raw, "n")
            raw["params"][pn] = {
                "type": "int",
                "low": lo,
                "high": hi,
                "value": int(rng.integers(lo, hi + 1)),
            }
            feat["args"] = {arg: {"param": pn}}
        tn = _fresh(raw, "t")
        raw["params"][tn] = {
            "type": "float",
            "low": centre - 3 * scale,
            "high": centre + 3 * scale,
            "value": float(centre + rng.uniform(-scale, scale)),
        }
        return {
            "op": ">" if rng.random() < 0.5 else "<",
            "left": {"feature": fid},
            "right": {"param": tn},
        }
    kind = list(LEVEL_FEATURES)[int(rng.integers(len(LEVEL_FEATURES)))]
    arg, (lo, hi) = LEVEL_FEATURES[kind]
    fid = _fresh(raw, "f")
    feat = {"id": fid, "kind": kind}
    raw["features"].append(feat)
    if arg:
        pn = _fresh(raw, "n")
        raw["params"][pn] = {
            "type": "int",
            "low": lo,
            "high": hi,
            "value": int(rng.integers(lo, hi + 1)),
        }
        feat["args"] = {arg: {"param": pn}}
    op = ["cross_above", "cross_below", ">", "<"][int(rng.integers(4))]
    return {"op": op, "left": {"price": "close"}, "right": {"feature": fid}}


def _time_window(raw: Raw, rng: np.random.Generator) -> list[dict[str, Any]]:
    mid = next((f["id"] for f in raw["features"] if f["kind"] == "minutes_into_session"), None)
    if mid is None:
        mid = _fresh(raw, "f")
        raw["features"].append({"id": mid, "kind": "minutes_into_session"})
    start = int(rng.integers(0, 300))
    a, b = _fresh(raw, "m"), None
    raw["params"][a] = {"type": "int", "low": 0, "high": 900, "value": start}
    b = _fresh(raw, "m")
    raw["params"][b] = {
        "type": "int",
        "low": 1,
        "high": 930,
        "value": min(930, start + int(rng.integers(15, 180))),
    }
    return [
        {"op": ">=", "left": {"feature": mid}, "right": {"param": a}},
        {"op": "<", "left": {"feature": mid}, "right": {"param": b}},
    ]


def random_raw(rng: np.random.Generator, name: str) -> Raw:
    raw: Raw = {
        "dsl_version": 1,
        "name": name,
        "session": SESSIONS[int(rng.integers(3))],
        "params": {},
        "features": [],
        "entry": {},
        "exit": {"session_flat": True},
    }
    conds = [_new_condition(raw, rng) for _ in range(int(rng.integers(1, 4)))]
    if rng.random() < 0.7:
        conds += _time_window(raw, rng)
    side = "long" if rng.random() < 0.5 else "short"
    raw["entry"][side] = {"all": conds} if len(conds) > 1 else conds[0]
    raw["params"]["sp"] = {
        "type": "float",
        "low": 4,
        "high": 100,
        "value": float(rng.uniform(5, 60)),
    }
    raw["params"]["tp"] = {
        "type": "float",
        "low": 4,
        "high": 200,
        "value": float(rng.uniform(5, 120)),
    }
    raw["exit"]["stop"] = {"points": {"param": "sp"}}
    raw["exit"]["target"] = {"points": {"param": "tp"}}
    return raw


# ---- the space ------------------------------------------------------------------------------


class NQSpecSpace:
    """Implements foundry.search.protocols.SpecSpace for the NQ DSL."""

    max_attempts = 20

    def validate(self, raw: Raw) -> Spec:
        return validate_spec(raw)

    def random(self, rng: np.random.Generator, name: str) -> Spec:
        for _ in range(self.max_attempts):
            try:
                return validate_spec(prune(random_raw(rng, name)))
            except SpecError:
                continue
        raise SpecError(["could not build a valid random spec"])

    # -- mutation --

    def mutate(self, spec: Spec, rng: np.random.Generator, sigma: float, name: str) -> Spec | None:
        ops: list[Callable[[Raw, np.random.Generator, float], None]] = [
            _m_params,
            _m_params,
            _m_flip_op,
            _m_add_condition,
            _m_remove_condition,
            _m_session,
            _m_exit,
            _m_filters,
            _m_flip_side,
        ]
        for _ in range(self.max_attempts):
            raw = copy.deepcopy(spec.raw)
            raw["name"] = name
            ops[int(rng.integers(len(ops)))](raw, rng, sigma)
            try:
                child = validate_spec(prune(raw))
            except SpecError:
                continue
            if child.hash != spec.hash:
                return child
        return None

    def crossover(self, a: Spec, b: Spec, rng: np.random.Generator, name: str) -> Spec | None:
        """Entry logic from one parent, exits and session from the other."""
        for _ in range(self.max_attempts):
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            child = copy.deepcopy(x.raw)
            donor = copy.deepcopy(y.raw)
            # Rename the donor's params and features so none collides with the child's.
            taken = set(child["params"]) | {f["id"] for f in child["features"]}
            pmap: dict[str, str] = {}
            fmap: dict[str, str] = {}
            for names, mapping, prefix in (
                (list(donor["params"]), pmap, "x"),
                ([f["id"] for f in donor["features"]], fmap, "y"),
            ):
                i = 0
                for old in names:
                    while f"{prefix}{i}" in taken:
                        i += 1
                    mapping[old] = f"{prefix}{i}"
                    taken.add(mapping[old])
            donor = _rename(donor, pmap, fmap)
            child["name"] = name
            child["exit"] = donor["exit"]
            child["session"] = donor["session"] if rng.random() < 0.5 else child["session"]
            for side in ("long", "short"):
                if f"rule_{side}" in child["exit"] and side not in child["entry"]:
                    del child["exit"][f"rule_{side}"]
            child["params"].update(donor["params"])
            child["features"] += donor["features"]
            try:
                out = validate_spec(prune(child))
            except SpecError:
                continue
            if out.hash not in (a.hash, b.hash):
                return out
        return None

    # -- CMA view --

    def numeric_params(self, spec: Spec) -> list[tuple[str, str, float, float, float]]:
        return [
            (k, p["type"], float(p["low"]), float(p["high"]), float(p["value"]))
            for k, p in sorted(spec.params().items())
            if p["high"] > p["low"]
        ]

    def with_unit_vector(self, spec: Spec, x: np.ndarray, name: str) -> Spec:
        vals = {}
        for (k, t, lo, hi, _v), xi in zip(self.numeric_params(spec), x, strict=True):
            v = lo + float(np.clip(xi, 0.0, 1.0)) * (hi - lo)
            vals[k] = float(round(v)) if t == "int" else v
        raw = copy.deepcopy(spec.with_params(vals).raw)
        raw["name"] = name
        return validate_spec(raw)

    # -- surrogate features --

    def featurize(self, spec: Spec) -> np.ndarray:
        raw = spec.raw
        v: list[float] = [float(raw["session"] == s) for s in SESSIONS]
        kinds = [f["kind"] for f in raw["features"]]
        v += [float(kinds.count(k)) for k in FEATURE_KINDS]
        ops: list[str] = []
        depth = [0]

        def visit_rule(r: dict[str, Any], d: int) -> None:
            depth[0] = max(depth[0], d)
            for key in ("all", "any"):
                for sub in r.get(key, []):
                    visit_rule(sub, d + 1)
            if "not" in r:
                visit_rule(r["not"], d + 1)
            if "op" in r:
                ops.append(r["op"])

        for side in ("long", "short"):
            if side in raw["entry"]:
                visit_rule(raw["entry"][side], 1)
        v += [float(ops.count(o)) for o in OP_FLIP]
        v += [float(depth[0]), float("long" in raw["entry"]), float("short" in raw["entry"])]
        ex = raw["exit"]
        v += [
            float(k in ex)
            for k in ("stop", "target", "trailing", "time_stop_bars", "rule_long", "rule_short")
        ]
        v.append(float(ex.get("session_flat", True)))
        order = raw["entry"].get("order", {"type": "market"})["type"]
        v += [float(order == o) for o in ("market", "stop", "limit")]

        def dist(key: str) -> float:
            d = ex.get(key)
            return spec.param(d["points"]["param"]) if d and "points" in d else -1.0

        stop, tgt = dist("stop"), dist("target")
        v += [stop, tgt, tgt / stop if stop > 0 and tgt > 0 else -1.0]
        f = raw.get("filters", {})
        v += [
            float(len(f.get("volatility_buckets", []))),
            float(len(f.get("trend_states", []))),
            float(len(f.get("days_of_week", []))),
            float(bool(f.get("exclude_event_days"))),
        ]
        norm = [
            (p["value"] - p["low"]) / (p["high"] - p["low"])
            for p in raw["params"].values()
            if p["high"] > p["low"]
        ]
        v += [
            float(len(raw["params"])),
            float(np.mean(norm)) if norm else 0.5,
            float(min(norm)) if norm else 0.5,
            float(max(norm)) if norm else 0.5,
        ]
        mins = [
            spec.param(c["right"]["param"])
            for c in _comparisons(raw)
            if c["left"].get("feature")
            in {x["id"] for x in raw["features"] if x["kind"] == "minutes_into_session"}
            and "param" in c["right"]
        ]
        v += [min(mins) if mins else -1.0, max(mins) if mins else -1.0]
        return np.array(v, dtype=np.float64)

    def summarize(self, spec: Spec) -> str:
        return spec_summary(spec)

    # -- LLM description --

    def describe_for_llm(self) -> str:
        rows = "\n".join(
            f"- {k}: args={list(a) or 'none'}, unit={u}. {FEATURE_MEANINGS[k]}"
            for k, (a, u) in FEATURE_KINDS.items()
        )
        return (
            "## DSL JSON schema\n"
            + SCHEMA_PATH.read_text(encoding="utf-8")
            + "\n\n## Features\n"
            + rows
            + "\n\n## Rules\n"
            '- Every number is a bounded param: {"param": name} with type/low/high/value. '
            "Feature args must be int params with low >= 1. Every param must be used.\n"
            "- Units: price may be compared only with price (e.g. close vs opening_range_high); "
            "never compare a price with a param. price - price = points; price + points = price; "
            "points * unitless param = points.\n"
            "- Semantics: signal on bar close, fill at next bar open; stop checked before target; "
            "costs (commission + 1 tick slippage) always apply; "
            "session_flat closes at session end.\n"
        )


def _comparisons(raw: Raw) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def visit(d: dict[str, Any]) -> None:
        if "op" in d and "left" in d:
            out.append(d)

    _walk(raw["entry"], visit)
    _walk({k: v for k, v in raw["exit"].items() if k.startswith("rule_")}, visit)
    return out


# ---- mutation operators (in place on a raw copy) --------------------------------------------


def _m_params(raw: Raw, rng: np.random.Generator, sigma: float) -> None:
    names = list(raw["params"])
    for k in rng.choice(names, size=min(len(names), int(rng.integers(1, 3))), replace=False):
        p = raw["params"][str(k)]
        span = p["high"] - p["low"]
        if span <= 0:
            continue
        x = (p["value"] - p["low"]) / span + rng.normal(0, sigma)
        v = p["low"] + float(np.clip(x, 0, 1)) * span
        p["value"] = round(v) if p["type"] == "int" else v


def _m_flip_op(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    comps = _comparisons(raw)
    if comps:
        c = comps[int(rng.integers(len(comps)))]
        c["op"] = OP_FLIP[c["op"]]


def _entry_side(raw: Raw, rng: np.random.Generator) -> str:
    sides = [s for s in ("long", "short") if s in raw["entry"]]
    return sides[int(rng.integers(len(sides)))]


def _m_add_condition(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    side = _entry_side(raw, rng)
    rule = raw["entry"][side]
    new = _new_condition(raw, rng)
    if "all" in rule:
        rule["all"].append(new)
    else:
        raw["entry"][side] = {"all": [rule, new]}


def _m_remove_condition(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    side = _entry_side(raw, rng)
    rule = raw["entry"][side]
    for key in ("all", "any"):
        if key in rule and len(rule[key]) >= 2:
            rule[key].pop(int(rng.integers(len(rule[key]))))
            if len(rule[key]) == 1:
                raw["entry"][side] = rule[key][0]
            return


def _m_session(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    raw["session"] = [s for s in SESSIONS if s != raw["session"]][int(rng.integers(2))]


def _m_exit(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    ex = raw["exit"]
    choice = int(rng.integers(3))
    if choice == 0:
        if "trailing" in ex:
            del ex["trailing"]
        else:
            n = _fresh(raw, "tr")
            raw["params"][n] = {
                "type": "float",
                "low": 4,
                "high": 100,
                "value": float(rng.uniform(5, 60)),
            }
            ex["trailing"] = {"points": {"param": n}}
    elif choice == 1:
        if "time_stop_bars" in ex:
            del ex["time_stop_bars"]
        else:
            n = _fresh(raw, "ts")
            raw["params"][n] = {
                "type": "int",
                "low": 1,
                "high": 390,
                "value": int(rng.integers(5, 240)),
            }
            ex["time_stop_bars"] = {"param": n}
    elif "target" in ex:
        del ex["target"]
    else:
        n = _fresh(raw, "tp")
        raw["params"][n] = {
            "type": "float",
            "low": 4,
            "high": 200,
            "value": float(rng.uniform(5, 120)),
        }
        ex["target"] = {"points": {"param": n}}


def _m_filters(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    f = raw.setdefault("filters", {})
    choice = int(rng.integers(3))
    if choice == 0:
        if "volatility_buckets" in f:
            del f["volatility_buckets"]
        else:
            k = int(rng.integers(1, 3))
            f["volatility_buckets"] = sorted(
                str(x) for x in rng.choice(["low", "mid", "high"], size=k, replace=False)
            )
    elif choice == 1:
        if "days_of_week" in f:
            del f["days_of_week"]
        else:
            days = ["mon", "tue", "wed", "thu", "fri"]
            drop = days[int(rng.integers(5))]
            f["days_of_week"] = [d for d in days if d != drop]
    else:
        f["exclude_event_days"] = not f.get("exclude_event_days", False)
    if not f:
        del raw["filters"]


def _m_flip_side(raw: Raw, rng: np.random.Generator, _s: float) -> None:
    if len(raw["entry"]) != 1 or "order" in raw["entry"]:
        return
    side = next(iter(raw["entry"]))
    other = "short" if side == "long" else "long"
    raw["entry"] = {other: raw["entry"][side]}
    if f"rule_{side}" in raw["exit"]:
        raw["exit"][f"rule_{other}"] = raw["exit"].pop(f"rule_{side}")


def spec_summary(spec: Spec) -> str:
    """Compact structural description for LLM feedback (no numbers from any split)."""
    raw = spec.raw
    sides = ",".join(s for s in ("long", "short") if s in raw["entry"])
    kinds = sorted({f["kind"] for f in raw["features"]})
    exits = sorted(k for k in raw["exit"] if k != "session_flat")
    return json.dumps(
        {
            "session": raw["session"],
            "sides": sides,
            "features": kinds,
            "exits": exits,
            "filters": sorted(raw.get("filters", {})),
        }
    )
