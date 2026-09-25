"""`foundry data build`: raw files -> processed Parquet + sealed holdout + data report.

The build ends by re-reading and verifying its own outputs, and returns a non-zero exit code
when any check fails. On a fatal data problem it writes the report and manifest (for
diagnosis) but no series, and removes stale series from earlier builds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from foundry.core.config import config_hash, load_model
from foundry.core.hashing import combined_hash, sha256_file
from foundry.core.holdout import HoldoutError, HoldoutVault
from foundry.core.repro import run_metadata
from foundry.core.verify import Check, Verifier
from foundry.domains.nq.calendar import TradingCalendar
from foundry.domains.nq.config import FoundryConfig
from foundry.domains.nq.loaders import ISSUE_SCHEMA, TS_DTYPE, LoadError, load_all
from foundry.domains.nq.quality import (
    FLAG_NAMES,
    GapReport,
    find_gaps,
    flag_bars,
    issues_from_flags,
    session_alignment,
)
from foundry.domains.nq.report import write_data_report
from foundry.domains.nq.rolls import build_continuous
from foundry.domains.nq.sessions import SESSION_LABELS, annotate_sessions
from foundry.domains.nq.splits import (
    HOLDOUT,
    PROCESSED_SPLITS,
    SEARCH,
    UNASSIGNED,
    VALIDATION,
    assign_splits,
)

SERIES_FILE = "bars.parquet"
ROLLS_FILE = "rolls.parquet"
ISSUES_FILE = "quality_issues.parquet"
GAPS_FILE = "gaps.parquet"
MANIFEST_FILE = "build_manifest.json"
REPORT_FILE = "data_report.md"
DATA_OUTPUTS = (SERIES_FILE, ROLLS_FILE, ISSUES_FILE, GAPS_FILE)

SERIES_SCHEMA: dict[str, pl.DataType] = {
    "ts": TS_DTYPE,
    "trading_date": pl.Date(),
    "session": pl.String(),
    "split": pl.String(),
    "contract": pl.String(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Int64(),
    "raw_open": pl.Float64(),
    "raw_high": pl.Float64(),
    "raw_low": pl.Float64(),
    "raw_close": pl.Float64(),
    "adjustment": pl.Float64(),
    "quality_flags": pl.UInt32(),
    "source_file": pl.String(),
}


def date_bounds(frame: pl.DataFrame) -> tuple[date, date]:
    """First and last trading date of a non-empty frame."""
    col = frame["trading_date"]
    return cast(date, col.min()), cast(date, col.max())


@dataclass
class SplitStats:
    rows: int
    trading_days: int
    first: date | None
    last: date | None

    @classmethod
    def of(cls, frame: pl.DataFrame) -> SplitStats:
        if frame.is_empty():
            return cls(0, 0, None, None)
        first, last = date_bounds(frame)
        return cls(frame.height, frame["trading_date"].n_unique(), first, last)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "trading_days": self.trading_days,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
        }


@dataclass
class BuildOutcome:
    ok: bool
    processed_dir: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1


def _resolve(p: Path, root: Path) -> Path:
    return p if p.is_absolute() else root / p


def _gap_frame(g: GapReport) -> pl.DataFrame:
    return g.runs.sort(["minutes", "start"], descending=[True, False])


def run_build(config_path: Path, repo_root: Path) -> BuildOutcome:
    cfg = load_model(config_path, FoundryConfig)
    d = cfg.data
    out_dir = _resolve(d.processed_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    vault = HoldoutVault(_resolve(cfg.holdout.directory, repo_root))
    calendar_path = _resolve(d.calendar_file, repo_root)
    calendar = TradingCalendar.from_file(calendar_path, d.sessions)
    pre = Verifier()  # fatal data checks, before anything is written

    # 1. Load raw files.
    try:
        loaded = load_all(d.sources, d.instrument.root, d.instrument.bar_minutes)
    except LoadError as exc:
        pre.check("load", False, str(exc))
        return _fail(out_dir, cfg, config_path, calendar_path, repo_root, pre, {})
    raw = loaded.bars
    if not pre.check(
        "load_nonempty", raw.height > 0, f"{raw.height} bars from {len(loaded.files)} files"
    ):
        return _fail(
            out_dir, cfg, config_path, calendar_path, repo_root, pre, {"files": loaded.files}
        )
    conflicts = loaded.issues.filter(pl.col("severity") == "fatal")
    pre.check(
        "no_conflicting_duplicates",
        conflicts.is_empty(),
        f"{conflicts.height} conflicting duplicate timestamps",
    )

    # 2. Sessions, calendar coverage, timezone sanity.
    bars = annotate_sessions(raw, calendar)
    tmin, tmax = date_bounds(bars)
    covered = calendar.covers(tmin) and calendar.covers(tmax)
    pre.check(
        "calendar_covers_data",
        covered,
        f"data {tmin}..{tmax}, calendar {calendar.covers_from}..{calendar.covers_to}",
    )
    align = session_alignment(bars, d.sessions, d.quality.max_break_fraction)
    pre.check(
        "session_alignment",
        align.passed,
        f"{align.bars_in_break}/{align.total_bars} bars ({align.fraction:.4%}) in the "
        f"{d.sessions.globex_close}-{d.sessions.globex_open} ET break; limit "
        f"{d.quality.max_break_fraction:.4%}. If this fails, check each source's timezone "
        "and timestamp_label.",
    )

    # 3. Per-contract quality flags, then the continuous series.
    flagged = flag_bars(bars, d.instrument.tick_size, d.instrument.bar_minutes, d.quality)
    rr = build_continuous(flagged, d.splits.holdout_start)
    roll_fatal = rr.issues.filter(pl.col("severity") == "fatal")
    pre.check(
        "rolls_measurable_and_continuous",
        roll_fatal.is_empty(),
        "; ".join(roll_fatal["detail"].to_list()) or f"{rr.rolls.height} rolls ok",
    )
    series = assign_splits(rr.series, d.splits)
    pre.check(
        "adjustments_known",
        series["adjustment"].null_count() == 0,
        f"{series['adjustment'].null_count()} bars with unknown adjustment",
    )

    stats = {
        name: SplitStats.of(series.filter(pl.col("split") == name))
        for name in (SEARCH, VALIDATION, HOLDOUT, UNASSIGNED)
    }
    pre.check(
        "search_split_size",
        stats[SEARCH].trading_days >= d.splits.min_trading_days_search,
        f"{stats[SEARCH].trading_days} trading days in search "
        f"{d.splits.search.start}..{d.splits.search.end}; need "
        f"{d.splits.min_trading_days_search}",
    )
    pre.check(
        "validation_split_size",
        stats[VALIDATION].trading_days >= d.splits.min_trading_days_validation,
        f"{stats[VALIDATION].trading_days} trading days in validation "
        f"{d.splits.validation.start}..{d.splits.validation.end}; need "
        f"{d.splits.min_trading_days_validation}",
    )

    # 4. Issues and gaps, kept apart for processed vs holdout (ADR 0005).
    in_processed = pl.col("split").is_in(PROCESSED_SPLITS)
    processed = series.filter(in_processed)
    holdout = series.filter(pl.col("split") == HOLDOUT)
    hstart = d.splits.holdout_start
    # Bar-level issues follow their bar's split (by trading date, so an evening bar that
    # opens the first holdout session stays sealed). File-level issues carry no bar.
    file_level = pl.concat([loaded.issues, rr.issues]).cast(ISSUE_SCHEMA)  # type: ignore[arg-type]
    file_in_holdout = (
        pl.col("ts").is_not_null() & (pl.col("ts").dt.date() >= pl.lit(hstart))
    ).fill_null(False)
    proc_issues = pl.concat([file_level.filter(~file_in_holdout), issues_from_flags(processed)])
    hold_issues = pl.concat([file_level.filter(file_in_holdout), issues_from_flags(holdout)])
    proc_gaps = find_gaps(processed, calendar, d.instrument.bar_minutes)
    hold_gaps = find_gaps(holdout, calendar, d.instrument.bar_minutes)
    proc_rolls = rr.rolls.filter(pl.col("effective_date") < pl.lit(hstart))
    hold_rolls = rr.rolls.filter(pl.col("effective_date") >= pl.lit(hstart))

    facts: dict[str, Any] = {
        "files": loaded.files,
        "stats": stats,
        "alignment": align,
        "anchor": rr.anchor,
        "rolls": proc_rolls,
        "holdout_roll_count": hold_rolls.height,
        "issues": proc_issues,
        "holdout_issue_count": hold_issues.height,
        "gaps": proc_gaps,
        "holdout_gap_minutes": int(hold_gaps.runs["minutes"].sum() or 0),
        "session_counts": processed.group_by("session").len().sort("session"),
        "flag_counts": {
            name: processed.filter((pl.col("quality_flags") & bit) != 0).height
            for bit, name in FLAG_NAMES.items()
        },
        "config": cfg,
    }
    if not pre.ok:
        return _fail(out_dir, cfg, config_path, calendar_path, repo_root, pre, facts)

    # 5. Write processed outputs and seal the holdout.
    processed_out = processed.select(list(SERIES_SCHEMA)).cast(SERIES_SCHEMA)  # type: ignore[arg-type]
    processed_out.write_parquet(out_dir / SERIES_FILE, statistics=False)
    proc_rolls.write_parquet(out_dir / ROLLS_FILE, statistics=False)
    proc_issues.sort(["check", "ts"], nulls_last=True).write_parquet(
        out_dir / ISSUES_FILE, statistics=False
    )
    _gap_frame(proc_gaps).write_parquet(out_dir / GAPS_FILE, statistics=False)
    outputs = {name: sha256_file(out_dir / name) for name in DATA_OUTPUTS}

    post = Verifier(checks=list(pre.checks))
    holdout_info: dict[str, Any] = {"sealed": False}
    if holdout.height:
        summary = {
            **stats[HOLDOUT].as_dict(),
            "contracts": sorted(holdout["contract"].unique().to_list()),
            "rolls": hold_rolls.height,
            "quality_issues": hold_issues.height,
            "gap_minutes": facts["holdout_gap_minutes"],
        }
        try:
            seal = vault.seal(
                {
                    "bars": holdout.select(list(SERIES_SCHEMA)).cast(SERIES_SCHEMA),  # type: ignore[arg-type]
                    "rolls": hold_rolls,
                    "quality_issues": hold_issues.sort(["check", "ts"], nulls_last=True),
                    "gaps": _gap_frame(hold_gaps),
                },
                summary,
            )
            holdout_info = {
                "sealed": True,
                "content_hash": seal.content_hash,
                "changed": seal.changed,
                **summary,
            }
            post.check(
                "holdout_seal", True, f"content {seal.content_hash[:12]}, changed={seal.changed}"
            )
        except HoldoutError as exc:
            post.check("holdout_seal", False, str(exc))
    facts["holdout"] = holdout_info

    # 6. Self-verification of what was written.
    _verify_outputs(post, out_dir, processed_out.height, outputs, d.splits, vault, holdout.height)

    manifest = _manifest(
        cfg,
        config_path,
        calendar_path,
        repo_root,
        loaded.files,
        outputs,
        stats,
        holdout_info,
        rr.anchor,
        post,
    )
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_data_report(out_dir / REPORT_FILE, facts, post, manifest)
    post.check("report_written", (out_dir / REPORT_FILE).stat().st_size > 0, REPORT_FILE)
    if not post.ok:
        manifest["status"] = "FAIL"
        manifest["checks"] = [c.__dict__ for c in post.checks]
        (out_dir / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return BuildOutcome(ok=post.ok, processed_dir=out_dir, checks=post.checks)


def _verify_outputs(
    v: Verifier,
    out_dir: Path,
    expected_rows: int,
    outputs: dict[str, str],
    splits: Any,
    vault: HoldoutVault,
    holdout_rows: int,
) -> None:
    back = pl.read_parquet(out_dir / SERIES_FILE)
    v.check("series_schema", dict(back.schema) == SERIES_SCHEMA, str(dict(back.schema)))
    v.check("series_rows", back.height == expected_rows, f"{back.height} == {expected_rows}")
    nulls = {c: n for c, n in zip(back.columns, back.null_count().row(0), strict=True) if n}
    v.check("series_no_nulls", not nulls, f"nulls: {nulls}" if nulls else "none")
    nans = {
        c: back[c].is_nan().sum()
        for c in ("open", "high", "low", "close", "adjustment")
        if back[c].is_nan().sum()
    }
    v.check("series_no_nans", not nans, f"NaNs: {nans}" if nans else "none")
    strictly = back["ts"].is_sorted() and bool(back["ts"].is_unique().all())
    v.check("series_ts_strictly_increasing", strictly, "")
    v.check(
        "series_splits",
        set(back["split"].unique().to_list()) <= set(PROCESSED_SPLITS),
        str(sorted(back["split"].unique().to_list())),
    )
    v.check("series_sessions", set(back["session"].unique().to_list()) <= set(SESSION_LABELS), "")
    no_holdout_dates = back.is_empty() or back["trading_date"].max() < splits.holdout_start
    v.check("processed_contains_no_holdout_dates", bool(no_holdout_dates), "")
    for name, h in outputs.items():
        v.check(f"hash_{name}", sha256_file(out_dir / name) == h, h[:12])
    if holdout_rows:
        try:
            m = vault.verify()
            v.check(
                "holdout_manifest_verifies",
                m["summary"]["rows"] == holdout_rows,
                f"{m['summary']['rows']} rows sealed",
            )
        except HoldoutError as exc:
            v.check("holdout_manifest_verifies", False, str(exc))


def _manifest(
    cfg: FoundryConfig,
    config_path: Path,
    calendar_path: Path,
    repo_root: Path,
    files: list[Any],
    outputs: dict[str, str],
    stats: dict[str, SplitStats],
    holdout: dict[str, Any],
    anchor: str | None,
    v: Verifier,
) -> dict[str, Any]:
    return {
        "status": "PASS" if v.ok else "FAIL",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run_metadata(cfg.seed, config_hash(cfg), repo_root),
        "config_file": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "calendar_file": {"path": str(calendar_path), "sha256": sha256_file(calendar_path)},
        "inputs": [
            {
                "file": str(f.path),
                "source": f.source,
                "sha256": f.sha256,
                "rows": f.rows,
                "contract": f.contract,
            }
            for f in files
        ],
        "outputs": outputs,
        "data_hash": combined_hash(outputs.items()) if outputs else None,
        "splits": {k: s.as_dict() for k, s in stats.items() if k != HOLDOUT},
        "holdout": holdout,
        "anchor_contract": anchor,
        "checks": [c.__dict__ for c in v.checks],
    }


def _fail(
    out_dir: Path,
    cfg: FoundryConfig,
    config_path: Path,
    calendar_path: Path,
    repo_root: Path,
    v: Verifier,
    facts: dict[str, Any],
) -> BuildOutcome:
    for name in DATA_OUTPUTS:
        (out_dir / name).unlink(missing_ok=True)
    files = facts.get("files", [])
    stats = facts.get("stats", {})
    manifest = _manifest(
        cfg,
        config_path,
        calendar_path,
        repo_root,
        files,
        {},
        stats,
        {"sealed": False},
        facts.get("anchor"),
        v,
    )
    (out_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    facts.setdefault("config", cfg)
    facts["holdout"] = {"sealed": False}
    write_data_report(out_dir / REPORT_FILE, facts, v, manifest)
    return BuildOutcome(ok=False, processed_dir=out_dir, checks=v.checks)
