"""NQ spec space: every operator yields valid, distinct specs; features are fixed length."""

from __future__ import annotations

import copy

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from foundry.domains.nq.dsl import load_spec
from foundry.domains.nq.space import NQSpecSpace, prune, references
from tests.conftest import REPO

BASELINE = REPO / "strategies" / "baseline_rth_orb.json"
SPACE = NQSpecSpace()


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_random_mutate_crossover_are_valid(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = SPACE.random(rng, "a")
    b = SPACE.random(rng, "b")
    for spec in (a, b):
        pr, fr = references(spec.raw)
        assert set(spec.raw["params"]) == pr  # nothing unused, nothing missing
        assert {f["id"] for f in spec.raw["features"]} == fr
    m = SPACE.mutate(a, rng, 0.2, "m")
    if m is not None:
        assert m.hash != a.hash
    x = SPACE.crossover(a, b, rng, "x")
    if x is not None:
        assert x.hash not in (a.hash, b.hash)
        assert len(x.raw["params"]) == len(set(x.raw["params"]))


def test_crossover_never_overwrites_params() -> None:
    rng = np.random.default_rng(3)
    base = load_spec(BASELINE)
    for _ in range(30):
        other = SPACE.random(rng, "o")
        child = SPACE.crossover(base, other, rng, "c")
        if child is None:
            continue
        # every param value in the child comes from one of the parents unchanged
        parent_values = {
            (p["type"], p["low"], p["high"], p["value"])
            for s in (base, other)
            for p in s.raw["params"].values()
        }
        for p in child.raw["params"].values():
            assert (p["type"], p["low"], p["high"], p["value"]) in parent_values


def test_prune_removes_unreferenced() -> None:
    raw = copy.deepcopy(load_spec(BASELINE).raw)
    raw["features"].append({"id": "orphan", "kind": "atr", "args": {"bars": {"param": "orphan_n"}}})
    raw["params"]["orphan_n"] = {"type": "int", "low": 2, "high": 9, "value": 3}
    out = prune(raw)
    assert "orphan_n" not in out["params"]
    assert "orphan" not in {f["id"] for f in out["features"]}


def test_featurize_fixed_length_and_cma_roundtrip() -> None:
    rng = np.random.default_rng(1)
    lengths = {SPACE.featurize(SPACE.random(rng, f"s{i}")).shape for i in range(20)}
    assert len(lengths) == 1
    base = load_spec(BASELINE)
    params = SPACE.numeric_params(base)
    x0 = np.array([(v - lo) / (hi - lo) for _, _, lo, hi, v in params])
    again = SPACE.with_unit_vector(base, x0, base.name)
    assert again.hash == base.hash
    corner = SPACE.with_unit_vector(base, np.ones(len(params)), "corner")
    assert corner.param("stop_pts") == 60 and corner.int_param("or_minutes") == 60


def test_llm_description_mentions_units_and_schema() -> None:
    text = SPACE.describe_for_llm()
    assert '"dsl_version"' in text and "never compare a price with a param" in text
    assert "opening_range_high" in text
