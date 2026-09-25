"""Raw loaders: formats, timezones, DST ambiguity, bar labels, duplicates (ADR 0002)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from foundry.domains.nq.config import SourceConfig
from foundry.domains.nq.loaders import (
    ContractId,
    LoadError,
    load_all,
    load_source,
    parse_contract_code,
    resolve_duplicates,
)
from tests.conftest import et

NT_REGEX = r"^NQ (?P<month>\d{2})-(?P<year>\d{2})\.Last-[Mm]inute\.txt$"


def nt_source(directory: Path, tz: str = "UTC", label: str = "end") -> SourceConfig:
    return SourceConfig(
        name="nt",
        format="ninjatrader_txt",
        directory=directory,
        filename_regex=NT_REGEX,
        timezone=tz,
        timestamp_label=label,
    )


def csv_source(directory: Path, tz: str, label: str = "start", **kw: object) -> SourceConfig:
    return SourceConfig.model_validate(
        {
            "name": "csv",
            "format": "csv",
            "directory": directory,
            "filename_regex": r"^NQ (?P<month>\d{2})-(?P<year>\d{2})\.csv$",
            "timezone": tz,
            "timestamp_label": label,
            "columns": {
                "timestamp": "time",
                "open": "o",
                "high": "h",
                "low": "l",
                "close": "c",
                "volume": "v",
            },
            **kw,
        }
    )


def write_csv(path: Path, stamps: list[str]) -> None:
    n = len(stamps)
    pl.DataFrame(
        {
            "time": stamps,
            "o": [100.0] * n,
            "h": [101.0] * n,
            "l": [99.0] * n,
            "c": [100.5] * n,
            "v": [5] * n,
        }
    ).write_csv(path)


def test_ninjatrader_utc_bar_end_to_et_bar_start(tmp_path: Path) -> None:
    # 23:01 UTC bar-end on 2025-01-01 is the 18:00 EST bar (Globex open); 13:31 UTC on a July
    # date is the 09:30 EDT RTH open bar.
    (tmp_path / "NQ 03-25.Last-Minute.txt").write_text(
        "20250101 230100;21269;21282.75;21253.5;21261.25;393\n"
        "20250708 133100;22000;22001;21999;22000.5;50\n",
        encoding="utf-8",
    )
    bars = load_source(nt_source(tmp_path), "NQ", 1).bars.sort("ts")
    assert bars["ts"].to_list() == [et(2025, 1, 1, 18, 0), et(2025, 7, 8, 9, 30)]
    assert bars["contract"].to_list() == ["NQH25", "NQH25"]
    assert bars["volume"].dtype == pl.Int64


def test_start_label_is_not_shifted(tmp_path: Path) -> None:
    (tmp_path / "NQ 03-25.Last-Minute.txt").write_text(
        "20250102 143000;1;1;1;1;1\n", encoding="utf-8"
    )
    bars = load_source(nt_source(tmp_path, label="start"), "NQ", 1).bars
    assert bars["ts"][0] == et(2025, 1, 2, 9, 30)


def test_et_local_fall_back_hour_resolves_by_order(tmp_path: Path) -> None:
    # 2024-11-03: 01:00-01:59 happens twice. The export lists EDT first, then EST.
    stamps = [
        "2024-11-03 00:59:00",
        "2024-11-03 01:00:00",
        "2024-11-03 01:59:00",
        "2024-11-03 01:00:00",
        "2024-11-03 01:59:00",
        "2024-11-03 02:00:00",
    ]
    write_csv(tmp_path / "NQ 12-24.csv", stamps)
    bars = load_source(csv_source(tmp_path, "America/New_York"), "NQ", 1).bars
    utc = bars["ts"].dt.convert_time_zone("UTC").dt.replace_time_zone(None).to_list()
    assert utc == [
        datetime(2024, 11, 3, h, m) for h, m in [(4, 59), (5, 0), (5, 59), (6, 0), (6, 59), (7, 0)]
    ]
    assert bars["ts"].is_unique().all()


def test_et_local_spring_forward_gap_is_dropped_and_reported(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "NQ 03-24.csv",
        ["2024-03-10 01:59:00", "2024-03-10 02:30:00", "2024-03-10 03:00:00"],
    )
    res = load_source(csv_source(tmp_path, "America/New_York"), "NQ", 1)
    assert res.bars.height == 2
    assert res.issues["check"].to_list() == ["nonexistent_local_time"]


def test_embedded_offsets(tmp_path: Path) -> None:
    write_csv(tmp_path / "NQ 03-25.csv", ["2025-03-11 07:23:00-05:00"])
    bars = load_source(csv_source(tmp_path, "embedded"), "NQ", 1).bars
    assert bars["ts"][0] == et(2025, 3, 11, 8, 23)


def test_embedded_offsets_require_embedded_timezone(tmp_path: Path) -> None:
    write_csv(tmp_path / "NQ 03-25.csv", ["2025-03-11 07:23:00-05:00"])
    with pytest.raises(LoadError, match="embedded"):
        load_source(csv_source(tmp_path, "UTC"), "NQ", 1)


def test_unix_seconds(tmp_path: Path) -> None:
    stamp = int(datetime(2025, 1, 2, 14, 30, tzinfo=UTC).timestamp())
    pl.DataFrame(
        {"time": [stamp], "o": [1.0], "h": [1.0], "l": [1.0], "c": [1.0], "v": [1]}
    ).write_csv(tmp_path / "NQ 03-25.csv")
    bars = load_source(csv_source(tmp_path, "UTC", timestamp_format="unix_s"), "NQ", 1).bars
    assert bars["ts"][0] == et(2025, 1, 2, 9, 30)


def test_timezone_and_label_are_required() -> None:
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "name": "x",
                "format": "ninjatrader_txt",
                "directory": ".",
                "filename_regex": NT_REGEX,
                "timestamp_label": "end",
            }
        )
    with pytest.raises(ValidationError):
        SourceConfig.model_validate(
            {
                "name": "x",
                "format": "ninjatrader_txt",
                "directory": ".",
                "filename_regex": NT_REGEX,
                "timezone": "UTC",
            }
        )
    with pytest.raises(ValidationError, match="timezone"):
        nt_source(Path("."), tz="Mars/Olympus")


def test_duplicates_exact_collapsed_conflicting_fatal() -> None:
    ts = et(2024, 3, 5, 10, 0)
    base = {
        "ts": [ts, ts],
        "contract": ["NQH24"] * 2,
        "open": [1.0, 1.0],
        "high": [1.0, 1.0],
        "low": [1.0, 1.0],
        "close": [1.0, 1.0],
        "volume": [3, 3],
        "source_file": ["a", "b"],
    }
    frame = pl.DataFrame(base).with_columns(pl.col("ts").dt.convert_time_zone("America/New_York"))
    deduped, issues = resolve_duplicates(frame)
    assert deduped.height == 1
    assert issues["check"].to_list() == ["duplicate_exact"]

    conflicting = frame.with_columns(pl.Series("volume", [3, 4]))
    _, issues = resolve_duplicates(conflicting)
    assert issues.filter(pl.col("severity") == "fatal")["check"].to_list() == ["duplicate_conflict"]


def test_load_all_merges_sources_and_dedupes(tmp_path: Path) -> None:
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "NQ 03-25.Last-Minute.txt").write_text(
            "20250102 143100;1;1;1;1;1\n", encoding="utf-8"
        )
    srcs = [nt_source(tmp_path / "a"), nt_source(tmp_path / "b")]
    res = load_all(srcs, "NQ", 1)
    assert res.bars.height == 1
    assert len(res.files) == 2


def test_missing_directory_and_no_matches(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="not found"):
        load_source(nt_source(tmp_path / "nope"), "NQ", 1)
    with pytest.raises(LoadError, match="no files match"):
        load_source(nt_source(tmp_path), "NQ", 1)


def test_contract_codes_roundtrip() -> None:
    for y in (2018, 2025, 2026):
        for m in (3, 6, 9, 12):
            cid = ContractId("NQ", y, m)
            assert parse_contract_code(cid.code) == cid
    assert parse_contract_code("CONT") is None
    assert ContractId("NQ", 2025, 12).sort_key < ContractId("NQ", 2026, 3).sort_key
