"""End-to-end `foundry data build` on synthetic files: outputs, sealing, checks, reproducibility."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from foundry.cli import app
from foundry.core.holdout import HoldoutVault
from foundry.domains.nq.build import (
    MANIFEST_FILE,
    REPORT_FILE,
    SERIES_FILE,
    SERIES_SCHEMA,
    run_build,
)
from tests.conftest import REPO, write_config

REAL_DATA = Path("C:/Projects/favio_NQ_lucid/data/historical")


def test_synthetic_build_passes_and_verifies(tmp_path: Path, synthetic_raw: Path) -> None:
    cfg = write_config(tmp_path, synthetic_raw)
    out = run_build(cfg, REPO)
    assert out.ok, [c for c in out.checks if not c.passed]
    processed = tmp_path / "processed"
    bars = pl.read_parquet(processed / SERIES_FILE)
    assert dict(bars.schema) == SERIES_SCHEMA
    assert set(bars["split"].unique().to_list()) == {"search", "validation"}
    assert bars["trading_date"].max() < date(2024, 4, 1)
    manifest = json.loads((processed / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert manifest["holdout"]["sealed"]
    assert manifest["anchor_contract"] == "NQM24"
    assert {"seed", "config_hash", "git", "python", "packages"} <= set(manifest["run"])
    report = (processed / REPORT_FILE).read_text(encoding="utf-8")
    assert "Data build report: PASS" in report
    # The report shows holdout counts only, never holdout prices.
    vault = HoldoutVault(tmp_path / "sealed")
    assert vault.verify()["summary"]["first"] == "2024-04-01"
    assert vault.access_log() == []


def test_build_is_reproducible(tmp_path: Path, synthetic_raw: Path) -> None:
    cfg = write_config(tmp_path, synthetic_raw)
    assert run_build(cfg, REPO).ok
    m1 = json.loads((tmp_path / "processed" / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert run_build(cfg, REPO).ok
    m2 = json.loads((tmp_path / "processed" / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert m1["outputs"] == m2["outputs"]
    assert m1["data_hash"] == m2["data_hash"]
    assert m1["holdout"]["content_hash"] == m2["holdout"]["content_hash"]
    assert m2["holdout"]["changed"] is False


def test_wrong_timezone_fails_loudly_and_removes_stale_outputs(
    tmp_path: Path, synthetic_raw: Path
) -> None:
    assert run_build(write_config(tmp_path, synthetic_raw), REPO).ok
    assert (tmp_path / "processed" / SERIES_FILE).exists()
    out = run_build(write_config(tmp_path, synthetic_raw, timezone="America/New_York"), REPO)
    assert not out.ok
    assert "session_alignment" in {c.name for c in out.checks if not c.passed}
    assert not (tmp_path / "processed" / SERIES_FILE).exists()
    manifest = json.loads((tmp_path / "processed" / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["status"] == "FAIL"


def test_empty_search_split_fails(tmp_path: Path, synthetic_raw: Path) -> None:
    splits = {
        "search": {"start": "2020-01-01", "end": "2020-12-31"},
        "validation": {"start": "2021-01-01", "end": "2021-12-31"},
        "holdout_start": "2022-01-01",
        "min_trading_days_search": 1,
        "min_trading_days_validation": 1,
    }
    out = run_build(write_config(tmp_path, synthetic_raw, splits=splits), REPO)
    failed = {c.name for c in out.checks if not c.passed}
    assert {"search_split_size", "validation_split_size"} <= failed
    # Nothing is sealed by a failing build.
    assert not (tmp_path / "sealed").exists()


def test_resealing_changed_data_after_opening_fails(tmp_path: Path, synthetic_raw: Path) -> None:
    cfg = write_config(tmp_path, synthetic_raw)
    assert run_build(cfg, REPO).ok
    HoldoutVault(tmp_path / "sealed").open("engine-0", ["cand-1"], "test")
    victim = next(synthetic_raw.glob("NQ 06-24*"))
    lines = victim.read_text(encoding="utf-8").splitlines()
    victim.write_text("\n".join(lines[:-100]) + "\n", encoding="utf-8")
    out = run_build(cfg, REPO)
    assert not out.ok
    assert "holdout_seal" in {c.name for c in out.checks if not c.passed}


def test_cli_exit_codes(
    tmp_path: Path, synthetic_raw: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO)
    runner = CliRunner()
    ok = runner.invoke(
        app, ["data", "build", "--config", str(write_config(tmp_path, synthetic_raw))]
    )
    assert ok.exit_code == 0, ok.output
    status = runner.invoke(app, ["holdout", "status", "--config", str(tmp_path / "config.yaml")])
    assert status.exit_code == 0
    assert '"openings": 0' in status.output
    bad = runner.invoke(
        app,
        [
            "data",
            "build",
            "--config",
            str(write_config(tmp_path, synthetic_raw, timezone="Asia/Tokyo")),
        ],
    )
    assert bad.exit_code == 1


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_DATA.is_dir(), reason="owner's raw files not present")
def test_real_2025_files_fail_honestly(tmp_path: Path) -> None:
    """The owner's 2025-only files: timezone checks pass, but the build must fail because the
    search/validation splits are empty and the June->September roll has no overlap."""
    cfg = write_config(
        tmp_path,
        REAL_DATA,
        splits={
            "search": {"start": "2018-01-01", "end": "2022-12-31"},
            "validation": {"start": "2023-01-01", "end": "2024-12-31"},
            "holdout_start": "2025-01-01",
            "min_trading_days_search": 250,
            "min_trading_days_validation": 120,
        },
    )
    out = run_build(cfg, REPO)
    results = {c.name: c.passed for c in out.checks}
    assert results["session_alignment"]
    assert results["calendar_covers_data"]
    assert results["no_conflicting_duplicates"]
    assert not results["search_split_size"]
    assert not results["rolls_measurable_and_continuous"]
    assert not out.ok
    assert not (tmp_path / "sealed").exists()
