"""The sealed holdout: manifest, tamper detection, once-per-candidate, access log, isolation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest

from foundry.core.holdout import ACCESS_LOG_NAME, HoldoutError, HoldoutVault
from tests.conftest import REPO


def table(n: int = 5, offset: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame({"i": list(range(n)), "x": [float(i) + offset for i in range(n)]})


def sealed(tmp_path: Path) -> HoldoutVault:
    v = HoldoutVault(tmp_path / "sealed")
    v.seal({"bars": table()}, {"rows": 5})
    return v


def test_seal_writes_manifest_that_verifies(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    m = v.verify()
    assert set(m["files"]) == {"bars.parquet"}
    assert m["summary"] == {"rows": 5}
    assert v.access_log() == []


def test_open_logs_before_returning_and_is_once_per_candidate(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    data = v.open("engine-1", ["c1", "c2"], "final check")
    assert data.tables["bars"].equals(table())
    log = v.access_log()
    assert len(log) == 1
    assert log[0]["candidate_ids"] == ["c1", "c2"]
    assert log[0]["engine_id"] == "engine-1"

    with pytest.raises(HoldoutError, match="already evaluated"):
        v.open("engine-2", ["c2", "c3"], "retry")
    assert len(v.access_log()) == 1  # a refused request is not an opening
    v.open("engine-2", ["c3"], "new candidate")
    assert v.evaluated_candidates() == {"c1", "c2", "c3"}


def test_open_requires_engine_candidates_and_reason(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    for args in (("", ["c"], "r"), ("e", [], "r"), ("e", ["c", "c"], "r"), ("e", ["c"], " ")):
        with pytest.raises(HoldoutError):
            v.open(*args)
    assert v.access_log() == []


def test_tampering_is_detected(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    table(offset=1.0).write_parquet(tmp_path / "sealed" / "bars.parquet")
    with pytest.raises(HoldoutError, match="modified"):
        v.verify()
    with pytest.raises(HoldoutError):
        v.open("e", ["c"], "r")
    assert v.access_log() == []


def test_missing_file_is_detected(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    (tmp_path / "sealed" / "bars.parquet").unlink()
    with pytest.raises(HoldoutError, match="missing"):
        v.verify()


def test_reseal_rules(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    assert not v.seal({"bars": table()}, {"rows": 5}).changed  # identical: no-op
    assert v.seal({"bars": table(6)}, {"rows": 6}).changed  # not opened yet: allowed
    v.open("e", ["c"], "r")
    assert not v.seal({"bars": table(6)}, {"rows": 6}).changed
    with pytest.raises(HoldoutError, match="already been opened"):
        v.seal({"bars": table(7)}, {"rows": 7})
    assert v.verify()["summary"] == {"rows": 6}


def test_access_log_is_append_only_jsonl(tmp_path: Path) -> None:
    v = sealed(tmp_path)
    v.open("e", ["a"], "r1")
    v.open("e", ["b"], "r2")
    lines = (tmp_path / "sealed" / ACCESS_LOG_NAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["candidate_ids"] for x in lines] == [["a"], ["b"]]


def test_no_other_module_touches_the_holdout() -> None:
    """Only foundry/core/holdout.py may know the sealed directory's layout or read it."""
    forbidden = re.compile(r"holdout_sealed|ACCESS_LOG|SEAL_LOG|MANIFEST\.json|_read_manifest")
    allowed = REPO / "foundry" / "core" / "holdout.py"
    offenders = []
    for path in (REPO / "foundry").rglob("*.py"):
        if path == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden.search(text):
            offenders.append(str(path.relative_to(REPO)))
        # Code that opens the vault must go through HoldoutVault.open, never private members.
        if re.search(r"HoldoutVault\([^)]*\)\._|vault\._", text):
            offenders.append(f"{path.relative_to(REPO)} (private vault access)")
    assert offenders == []
