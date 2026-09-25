"""Markdown data report. Holdout data appears only as counts, date ranges and hashes (ADR 0005)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from foundry.core.verify import Verifier


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_none_\n"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def write_data_report(
    path: Path, facts: dict[str, Any], v: Verifier, manifest: dict[str, Any]
) -> None:
    cfg = facts["config"]
    d = cfg.data
    lines: list[str] = []
    status = "PASS" if v.ok else "FAIL"
    lines.append(f"# Data build report: {status}\n")
    run = manifest["run"]
    lines.append(f"- Built: {manifest['created_at']}")
    lines.append(f"- Config hash: `{run['config_hash'][:16]}`, seed {run['seed']}")
    git = run["git"]
    lines.append(f"- Git: `{git['commit'] or 'no commit'}`{' (dirty)' if git['dirty'] else ''}")
    lines.append(f"- Data hash: `{(manifest.get('data_hash') or 'n/a')[:16]}`")
    lines.append(f"- Continuous anchor contract (adjustment 0): {facts.get('anchor') or 'n/a'}\n")

    failed = v.failed
    if failed:
        lines.append("## Failed checks\n")
        lines += [f"- **{c.name}**: {c.detail}" for c in failed]
        lines.append("")

    lines.append("## Inputs\n")
    lines.append(
        _table(
            ["file", "source", "contract", "rows", "sha256"],
            [
                [f.path.name, f.source, f.contract, f.rows, f.sha256[:12]]
                for f in facts.get("files", [])
            ],
        )
    )

    stats = facts.get("stats")
    if stats:
        lines.append("## Splits\n")
        sp = d.splits
        rng = {
            "search": f"{sp.search.start} .. {sp.search.end}",
            "validation": f"{sp.validation.start} .. {sp.validation.end}",
            "holdout": f"{sp.holdout_start} .. (sealed)",
            "unassigned": "outside all splits",
        }
        lines.append(
            _table(
                ["split", "configured range", "rows", "trading days", "first", "last"],
                [[k, rng[k], s.rows, s.trading_days, s.first, s.last] for k, s in stats.items()],
            )
        )
        h = facts.get("holdout", {})
        if h.get("sealed"):
            lines.append(
                f"Holdout sealed: content hash `{h['content_hash'][:16]}`, {h['rows']} rows, "
                f"{h['rolls']} rolls, {h['quality_issues']} quality issues and "
                f"{h['gap_minutes']} gap minutes. Only counts are shown here; the data can be "
                "read only through `foundry holdout open`.\n"
            )
        else:
            lines.append("Holdout: not sealed by this build.\n")

    align = facts.get("alignment")
    if align:
        lines.append("## Timezone and session alignment\n")
        lines.append(
            f"{align.bars_in_break} of {align.total_bars} loaded bars "
            f"({align.fraction:.4%}) start inside the {d.sessions.globex_close}-"
            f"{d.sessions.globex_open} ET maintenance break "
            f"(limit {d.quality.max_break_fraction:.4%}): **{'ok' if align.passed else 'FAIL'}**.\n"
        )

    sc = facts.get("session_counts")
    if sc is not None and sc.height:
        lines.append("## Session labels (search + validation)\n")
        lines.append(_table(["session", "bars"], [list(r) for r in sc.iter_rows()]))

    rolls: pl.DataFrame | None = facts.get("rolls")
    if rolls is not None:
        lines.append("## Rolls (pre-holdout)\n")
        lines.append(
            _table(
                [
                    "decided",
                    "effective",
                    "from",
                    "to",
                    "reason",
                    "vol from",
                    "vol to",
                    "gap pts",
                    "continuity",
                ],
                [
                    [
                        r["decision_date"],
                        r["effective_date"],
                        r["from_contract"],
                        r["to_contract"],
                        r["reason"],
                        r["from_volume"],
                        r["to_volume"],
                        r["gap_points"],
                        "ok" if r["continuity_ok"] else "FAIL",
                    ]
                    for r in rolls.iter_rows(named=True)
                ],
            )
        )
        lines.append(
            f"Rolls inside the holdout period: {facts.get('holdout_roll_count', 0)} "
            "(details sealed).\n"
        )

    fc = facts.get("flag_counts")
    if fc is not None:
        lines.append("## Quality flags on the continuous series (search + validation)\n")
        lines.append("Flagged bars stay in the data; nothing is filled (ADR 0004).\n")
        lines.append(_table(["flag", "bars"], [[k, n] for k, n in fc.items()]))

    issues: pl.DataFrame | None = facts.get("issues")
    if issues is not None and issues.height:
        lines.append("## Issues by check\n")
        by = issues.group_by(["check", "severity"]).len().sort(["severity", "check"])
        lines.append(_table(["check", "severity", "count"], [list(r) for r in by.iter_rows()]))
        file_level = issues.filter(pl.col("ts").is_null())
        if file_level.height:
            lines.append("File- and contract-level issues:\n")
            lines += [
                f"- `{r['check']}` ({r['severity']}, {r['contract']}): {r['detail']}"
                for r in file_level.iter_rows(named=True)
            ]
            lines.append("")

    gaps = facts.get("gaps")
    if gaps is not None:
        lines.append("## Gaps (search + validation)\n")
        cover = gaps.present_minutes / gaps.expected_minutes if gaps.expected_minutes else 0.0
        lines.append(
            f"Expected session minutes: {gaps.expected_minutes}; present: {gaps.present_minutes} "
            f"({cover:.2%}). Missing minutes are reported, never filled.\n"
        )
        lines.append(
            _table(
                ["session", "missing minutes"],
                [list(r) for r in gaps.missing_by_session.iter_rows()],
            )
        )
        if gaps.missing_dates:
            lines.append(
                f"Session dates with no bars at all ({len(gaps.missing_dates)}): "
                + ", ".join(str(x) for x in gaps.missing_dates[:60])
                + (" ..." if len(gaps.missing_dates) > 60 else "")
                + "\n"
            )
        top = gaps.runs.sort("minutes", descending=True).head(d.quality.top_gaps_reported)
        lines.append(f"Largest {top.height} gap runs:\n")
        lines.append(
            _table(
                ["trading date", "session", "start", "end", "minutes"],
                [
                    [r["trading_date"], r["session"], r["start"], r["end"], r["minutes"]]
                    for r in top.iter_rows(named=True)
                ],
            )
        )

    lines.append("## All checks\n")
    lines.append(
        _table(
            ["check", "result", "detail"],
            [
                [c.name, "pass" if c.passed else "**FAIL**", c.detail.replace("|", "/")]
                for c in v.checks
            ],
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")
