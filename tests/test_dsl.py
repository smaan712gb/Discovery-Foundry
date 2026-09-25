"""Strategy DSL: schema, semantic checks (units, bounds, references), hashing."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from foundry.domains.nq.dsl import SpecError, load_spec, validate_spec
from tests.conftest import REPO

BASELINE = REPO / "strategies" / "baseline_rth_orb.json"


def base() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))
    return raw


def problems(raw: dict[str, Any]) -> list[str]:
    with pytest.raises(SpecError) as exc:
        validate_spec(raw)
    return exc.value.problems


def test_baseline_is_valid() -> None:
    spec = load_spec(BASELINE)
    assert spec.session == "RTH"
    assert spec.int_param("or_minutes") == 30


def test_schema_rejects_unknown_keys_and_kinds() -> None:
    raw = base()
    raw["leverage"] = 10
    assert any("leverage" in p for p in problems(raw))
    raw = base()
    raw["features"][0]["kind"] = "tomorrow_close"
    assert problems(raw)


def test_every_number_is_a_bounded_param() -> None:
    raw = base()
    raw["exit"]["stop"] = {"points": 25}  # literal number, not a param ref
    assert problems(raw)
    raw = base()
    raw["params"]["stop_pts"]["value"] = 100  # outside [10, 60]
    assert any("outside" in p for p in problems(raw))
    raw = base()
    raw["params"]["or_minutes"]["value"] = 30.5  # int param, non-integer
    assert any("non-integer" in p for p in problems(raw))


def test_references_must_exist_and_params_must_be_used() -> None:
    raw = base()
    raw["exit"]["target"] = {"points": {"param": "nope"}}
    assert any("unknown param 'nope'" in p for p in problems(raw))
    raw = base()
    raw["params"]["spare"] = {"type": "float", "low": 0, "high": 1, "value": 0.5}
    assert any("unused params" in p for p in problems(raw))
    raw = base()
    raw["entry"]["long"]["all"][0]["right"] = {"feature": "ghost"}
    assert any("unknown feature 'ghost'" in p for p in problems(raw))


def test_price_cannot_be_compared_with_a_param() -> None:
    raw = base()
    raw["params"]["level"] = {"type": "float", "low": 10000, "high": 30000, "value": 20000}
    raw["entry"]["long"]["all"].append(
        {"op": ">", "left": {"price": "close"}, "right": {"param": "level"}}
    )
    assert any("absolute price level" in p for p in problems(raw))


def test_unit_arithmetic() -> None:
    raw = base()
    raw["features"].append({"id": "atr", "kind": "atr", "args": {"bars": {"param": "atr_n"}}})
    raw["params"]["atr_n"] = {"type": "int", "low": 5, "high": 50, "value": 14}
    raw["params"]["k"] = {"type": "float", "low": 0.1, "high": 3, "value": 0.5}
    # close > or_high + k * atr   (price > price + points): valid
    raw["entry"]["long"]["all"][0] = {
        "op": ">",
        "left": {"price": "close"},
        "right": {"add": [{"feature": "or_high"}, {"mul": [{"param": "k"}, {"feature": "atr"}]}]},
    }
    validate_spec(raw)
    bad = copy.deepcopy(raw)
    # price + price is meaningless
    bad["entry"]["long"]["all"][0]["right"] = {"add": [{"feature": "or_high"}, {"price": "close"}]}
    assert any("cannot add" in p for p in problems(bad))
    bad = copy.deepcopy(raw)
    # comparing points with price
    bad["entry"]["long"]["all"][0]["left"] = {"feature": "atr"}
    assert any("compares points with price" in p for p in problems(bad))


def test_feature_args_must_be_int_params() -> None:
    raw = base()
    raw["params"]["or_minutes"]["type"] = "float"
    assert any("must reference an int param" in p for p in problems(raw))


def test_stop_order_needs_prices_for_its_sides() -> None:
    raw = base()
    raw["params"]["valid"] = {"type": "int", "low": 1, "high": 10, "value": 3}
    raw["entry"]["order"] = {
        "type": "stop",
        "valid_bars": {"param": "valid"},
        "long_price": {"feature": "or_high"},
    }
    assert any("short_price" in p for p in problems(raw))
    raw["entry"]["order"]["short_price"] = {"feature": "or_low"}
    validate_spec(raw)
    raw["entry"]["order"]["short_price"] = {"feature": "minute"}
    assert any("must be a price" in p for p in problems(raw))


def test_atr_distance_requires_an_atr_feature() -> None:
    raw = base()
    raw["params"]["m"] = {"type": "float", "low": 0.5, "high": 5, "value": 1}
    raw["exit"]["trailing"] = {"atr_mult": {"param": "m"}, "atr_feature": "or_high"}
    assert any("not an atr feature" in p for p in problems(raw))


def test_rule_depth_limit() -> None:
    raw = base()
    rule: dict[str, Any] = raw["entry"]["long"]
    for _ in range(6):
        rule = {"not": rule}
    raw["entry"]["long"] = rule
    assert any("deeper than" in p for p in problems(raw))


def test_hash_ignores_name_and_rationale_but_not_params() -> None:
    a = validate_spec(base())
    renamed = base()
    renamed["name"] = "something_else"
    renamed["rationale"] = "different words"
    assert validate_spec(renamed).hash == a.hash
    changed = base()
    changed["params"]["stop_pts"]["value"] = 26
    assert validate_spec(changed).hash != a.hash


def test_with_params_revalidates() -> None:
    spec = load_spec(BASELINE)
    moved = spec.with_params({"stop_pts": 30})
    assert moved.param("stop_pts") == 30
    assert moved.hash != spec.hash
    with pytest.raises(SpecError):
        spec.with_params({"stop_pts": 1000})


def test_hash_ignores_number_formatting() -> None:
    a = base()
    b = base()
    b["params"]["stop_pts"]["value"] = 25.0  # float param written as 25 vs 25.0
    b["params"]["or_minutes"]["low"] = 15.0  # int param written as 15.0
    assert validate_spec(a).hash == validate_spec(b).hash
