"""Random DSL specs for multiple-testing tests: many unrelated strategies, no economic idea."""

from __future__ import annotations

from typing import Any

import numpy as np

from foundry.domains.nq.dsl import Spec, SpecError, validate_spec

# kind, integer arg name, arg range, threshold centre, threshold scale
_FEATURES: list[tuple[str, str | None, tuple[int, int], float, float]] = [
    ("close_zscore", "bars", (5, 240), 0.0, 2.0),
    ("returns", "bars", (5, 240), 0.0, 0.003),
    ("realized_vol", "bars", (10, 240), 0.0006, 0.0004),
    ("vwap_distance", None, (0, 0), 0.0, 20.0),
    ("session_volume_ratio", "days", (2, 10), 1.0, 0.5),
]


def random_spec(rng: np.random.Generator, i: int) -> Spec | None:
    params: dict[str, Any] = {}
    feats: list[dict[str, Any]] = []
    conds: list[dict[str, Any]] = []
    for j in range(int(rng.integers(1, 4))):
        kind, arg, (lo, hi), centre, scale = _FEATURES[int(rng.integers(len(_FEATURES)))]
        f: dict[str, Any] = {"id": f"f{j}", "kind": kind}
        if arg:
            params[f"n{j}"] = {
                "type": "int",
                "low": lo,
                "high": hi,
                "value": int(rng.integers(lo, hi)),
            }
            f["args"] = {arg: {"param": f"n{j}"}}
        feats.append(f)
        params[f"t{j}"] = {
            "type": "float",
            "low": centre - 3 * scale,
            "high": centre + 3 * scale,
            "value": float(centre + rng.uniform(-scale, scale)),
        }
        conds.append(
            {
                "op": ">" if rng.random() < 0.5 else "<",
                "left": {"feature": f"f{j}"},
                "right": {"param": f"t{j}"},
            }
        )
    feats.append({"id": "mins", "kind": "minutes_into_session"})
    start = int(rng.integers(0, 300))
    params["m0"] = {"type": "int", "low": 0, "high": 900, "value": start}
    params["m1"] = {
        "type": "int",
        "low": 1,
        "high": 930,
        "value": start + int(rng.integers(15, 120)),
    }
    conds += [
        {"op": ">=", "left": {"feature": "mins"}, "right": {"param": "m0"}},
        {"op": "<", "left": {"feature": "mins"}, "right": {"param": "m1"}},
    ]
    params["sp"] = {"type": "float", "low": 5, "high": 80, "value": float(rng.uniform(5, 80))}
    params["tp"] = {"type": "float", "low": 5, "high": 160, "value": float(rng.uniform(5, 160))}
    side = "long" if rng.random() < 0.5 else "short"
    try:
        return validate_spec(
            {
                "dsl_version": 1,
                "name": f"random_{i}",
                "session": ["RTH", "ETH", "both"][int(rng.integers(3))],
                "params": params,
                "features": feats,
                "entry": {side: {"all": conds}},
                "exit": {
                    "stop": {"points": {"param": "sp"}},
                    "target": {"points": {"param": "tp"}},
                    "session_flat": True,
                },
            }
        )
    except SpecError:
        return None
