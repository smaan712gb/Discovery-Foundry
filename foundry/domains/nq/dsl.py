"""Strategy DSL v1: JSON-schema validation, semantic checks, canonical hashing.

A spec is data, never code. Validation has two layers:

1. `dsl_schema.json` (structure, enums, sizes), which is what an LLM proposer is held to.
2. Semantic checks here: parameters exist and sit inside their bounds, feature arguments are
   integer parameters, operand units are consistent, and no absolute price threshold appears.

Units stop specs that would only work at one price level (an overfit trap made worse by
back-adjustment, ADR 0003): a `price` can only be compared with another `price`, and a parameter
can never stand in for a price.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import jsonschema

SCHEMA_PATH = Path(__file__).with_name("dsl_schema.json")
Unit = Literal["price", "points", "unitless"]
PARAM = "param"  # a bare parameter adopts the unit of what it's combined with, except price

# kind -> (integer argument names, output unit)
FEATURE_KINDS: dict[str, tuple[tuple[str, ...], Unit]] = {
    "returns": (("bars",), "unitless"),
    "atr": (("bars",), "points"),
    "vwap_distance": ((), "points"),
    "opening_range_high": (("minutes",), "price"),
    "opening_range_low": (("minutes",), "price"),
    "overnight_high": ((), "price"),
    "overnight_low": ((), "price"),
    "overnight_gap": ((), "points"),
    "session_volume_ratio": (("days",), "unitless"),
    "realized_vol": (("bars",), "unitless"),
    "prior_day_high": ((), "price"),
    "prior_day_low": ((), "price"),
    "prior_day_close": ((), "price"),
    "close_zscore": (("bars",), "unitless"),
    "minutes_into_session": ((), "unitless"),
}
MAX_RULE_DEPTH = 5
MAX_RULE_NODES = 40
NON_SEMANTIC_KEYS = ("name", "rationale")


class SpecError(ValueError):
    """A spec that fails schema or semantic validation. `.problems` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("invalid strategy spec:\n" + "\n".join(f"  - {p}" for p in problems))
        self.problems = problems


@cache
def schema() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return data


@dataclass(frozen=True)
class Spec:
    """A validated spec. `raw` is the canonical dict; treat it as read-only."""

    raw: dict[str, Any]
    hash: str

    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def session(self) -> str:
        return str(self.raw["session"])

    def param(self, name: str) -> float:
        return float(self.raw["params"][name]["value"])

    def int_param(self, name: str) -> int:
        return int(self.raw["params"][name]["value"])

    def params(self) -> dict[str, dict[str, Any]]:
        """name -> {type, low, high, value}."""
        return {k: dict(v) for k, v in self.raw["params"].items()}

    def traded_regimes(self) -> list[str] | None:
        """Volatility buckets the spec restricts itself to, or None if it trades all of them."""
        vb = self.raw.get("filters", {}).get("volatility_buckets")
        return list(vb) if vb else None

    def with_params(self, values: dict[str, float]) -> Spec:
        """A copy with new parameter values, re-validated (so out-of-bounds values are rejected)."""
        raw = copy.deepcopy(self.raw)
        for k, v in values.items():
            if k not in raw["params"]:
                raise SpecError([f"unknown parameter {k!r}"])
            p = raw["params"][k]
            p["value"] = round(v) if p["type"] == "int" else float(v)
        return validate_spec(raw)


def spec_hash(raw: dict[str, Any]) -> str:
    """Hash of the semantic content: name and rationale don't count, so renaming a spec doesn't
    let it be re-evaluated and escape the trial count."""
    body = {k: v for k, v in raw.items() if k not in NON_SEMANTIC_KEYS}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_spec(path: Path) -> Spec:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError([f"{path.name}: not valid JSON: {exc}"]) from exc
    return validate_spec(raw)


def validate_spec(raw: Any) -> Spec:
    validator = jsonschema.Draft202012Validator(schema())
    errors = sorted(validator.iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        raise SpecError(
            [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors]
        )
    problems: list[str] = []
    _Checker(raw, problems).run()
    if problems:
        raise SpecError(problems)
    canonical: dict[str, Any] = json.loads(json.dumps(raw, sort_keys=True))
    # Numbers are canonicalized by declared type, so 25 and 25.0 hash the same: a spec can't
    # escape de-duplication (and the trial count) through JSON formatting.
    for p in canonical["params"].values():
        cast = int if p["type"] == "int" else float
        for k in ("low", "high", "value"):
            p[k] = cast(p[k])
    return Spec(raw=canonical, hash=spec_hash(canonical))


class _Checker:
    def __init__(self, raw: dict[str, Any], problems: list[str]) -> None:
        self.raw = raw
        self.p = problems
        self.params: dict[str, dict[str, Any]] = raw["params"]
        self.used: set[str] = set()
        self.features: dict[str, str] = {}

    def run(self) -> None:
        for name, prm in self.params.items():
            lo, hi, val = prm["low"], prm["high"], prm["value"]
            if lo > hi:
                self.p.append(f"param {name}: low {lo} > high {hi}")
            if not lo <= val <= hi:
                self.p.append(f"param {name}: value {val} outside [{lo}, {hi}]")
            if prm["type"] == "int" and any(float(x) != int(x) for x in (lo, hi, val)):
                self.p.append(f"param {name}: int param with non-integer bounds or value")

        for f in self.raw["features"]:
            if f["id"] in self.features:
                self.p.append(f"feature id {f['id']!r} defined twice")
            self.features[f["id"]] = f["kind"]
            want, _unit = FEATURE_KINDS[f["kind"]]
            args = f.get("args", {})
            if set(args) != set(want):
                self.p.append(
                    f"feature {f['id']} ({f['kind']}): args must be exactly {list(want)}, "
                    f"got {sorted(args)}"
                )
            for a, ref in args.items():
                fprm = self._ref(ref, f"feature {f['id']}.{a}")
                if fprm is not None:
                    if fprm["type"] != "int":
                        self.p.append(f"feature {f['id']}.{a}: must reference an int param")
                    elif fprm["low"] < 1:
                        self.p.append(f"feature {f['id']}.{a}: param lower bound must be >= 1")

        entry = self.raw["entry"]
        for side in ("long", "short"):
            if side in entry:
                self._rule(entry[side], f"entry.{side}", 1, [0])
        order = entry.get("order", {"type": "market"})
        if order["type"] != "market":
            self._int_ref(order["valid_bars"], "entry.order.valid_bars", minimum=1)
            for side in ("long", "short"):
                key = f"{side}_price"
                if side in entry and key not in order:
                    self.p.append(
                        f"entry.order: {order['type']} order needs {key} for the {side} side"
                    )
                if key in order:
                    if side not in entry:
                        self.p.append(f"entry.order.{key} given but there is no {side} entry")
                    u = self._operand(order[key], f"entry.order.{key}")
                    if u != "price":
                        self.p.append(f"entry.order.{key}: must be a price, got {u}")

        ex = self.raw["exit"]
        for side in ("long", "short"):
            key = f"rule_{side}"
            if key in ex:
                if side not in entry:
                    self.p.append(f"exit.{key} given but there is no {side} entry")
                self._rule(ex[key], f"exit.{key}", 1, [0])
        for key in ("stop", "target", "trailing"):
            if key in ex:
                self._distance(ex[key], f"exit.{key}")
        if "time_stop_bars" in ex:
            self._int_ref(ex["time_stop_bars"], "exit.time_stop_bars", minimum=1)

        unused = set(self.params) - self.used
        if unused:
            self.p.append(f"unused params: {sorted(unused)}")

    # ---- helpers -----------------------------------------------------------------------

    def _ref(self, ref: dict[str, str], where: str) -> dict[str, Any] | None:
        name = ref["param"]
        if name not in self.params:
            self.p.append(f"{where}: unknown param {name!r}")
            return None
        self.used.add(name)
        return self.params[name]

    def _int_ref(self, ref: dict[str, str], where: str, minimum: int) -> None:
        prm = self._ref(ref, where)
        if prm is None:
            return
        if prm["type"] != "int":
            self.p.append(f"{where}: must reference an int param")
        elif prm["low"] < minimum:
            self.p.append(f"{where}: param lower bound must be >= {minimum}")

    def _distance(self, d: dict[str, Any], where: str) -> None:
        if "points" in d:
            prm = self._ref(d["points"], f"{where}.points")
            if prm is not None and prm["low"] <= 0:
                self.p.append(f"{where}.points: param lower bound must be > 0")
        else:
            prm = self._ref(d["atr_mult"], f"{where}.atr_mult")
            if prm is not None and prm["low"] <= 0:
                self.p.append(f"{where}.atr_mult: param lower bound must be > 0")
            kind = self.features.get(d["atr_feature"])
            if kind != "atr":
                self.p.append(f"{where}.atr_feature: {d['atr_feature']!r} is not an atr feature")

    def _rule(self, rule: dict[str, Any], where: str, depth: int, nodes: list[int]) -> None:
        nodes[0] += 1
        if depth > MAX_RULE_DEPTH:
            self.p.append(f"{where}: rule tree deeper than {MAX_RULE_DEPTH}")
            return
        if nodes[0] > MAX_RULE_NODES:
            self.p.append(f"{where}: rule tree has more than {MAX_RULE_NODES} nodes")
            return
        for key in ("all", "any"):
            if key in rule:
                for i, sub in enumerate(rule[key]):
                    self._rule(sub, f"{where}.{key}[{i}]", depth + 1, nodes)
                return
        if "not" in rule:
            self._rule(rule["not"], f"{where}.not", depth + 1, nodes)
            return
        lu = self._operand(rule["left"], f"{where}.left")
        ru = self._operand(rule["right"], f"{where}.right")
        if lu is None or ru is None:
            return
        if PARAM in (lu, ru):
            other = ru if lu == PARAM else lu
            if other == PARAM:
                self.p.append(f"{where}: compares two bare params")
            elif other == "price":
                self.p.append(f"{where}: compares a price with a param (absolute price level)")
        elif lu != ru:
            self.p.append(f"{where}: compares {lu} with {ru}")

    def _operand(self, op: dict[str, Any], where: str) -> str | None:
        if "feature" in op:
            kind = self.features.get(op["feature"])
            if kind is None:
                self.p.append(f"{where}: unknown feature {op['feature']!r}")
                return None
            return FEATURE_KINDS[kind][1]
        if "param" in op:
            return PARAM if self._ref(op, where) is not None else None
        if "price" in op:
            return "price"
        key = next(k for k in ("add", "sub", "mul") if k in op)
        a = self._operand(op[key][0], f"{where}.{key}[0]")
        b = self._operand(op[key][1], f"{where}.{key}[1]")
        if a is None or b is None:
            return None
        unit = _combine(key, a, b)
        if unit is None:
            self.p.append(f"{where}: cannot {key} {a} and {b}")
        return unit


def _combine(op: str, a: str, b: str) -> str | None:
    if op == "mul":
        # Scaling by a unitless quantity or bare param keeps the unit; prices can't be scaled.
        pair = {a, b}
        if pair <= {"unitless", PARAM}:
            return "unitless" if "unitless" in pair else None
        if "price" in pair:
            return None
        rest = pair - {"unitless", PARAM}
        return "points" if rest == {"points"} and len(pair) == 2 else None
    if PARAM in (a, b):
        other = b if a == PARAM else a
        return None if other in ("price", PARAM) else other
    if op == "add":
        if {a, b} == {"price", "points"}:
            return "price"
        return a if a == b and a != "price" else None
    # sub
    if a == b == "price":
        return "points"
    if a == "price" and b == "points":
        return "price"
    return a if a == b else None
