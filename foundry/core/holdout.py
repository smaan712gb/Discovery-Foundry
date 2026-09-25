"""The sealed holdout vault.

This is the only module allowed to read or write the sealed holdout directory (enforced by
tests/test_holdout.py::test_no_other_module_touches_the_holdout). Opening the holdout:

* verifies every file against the SHA-256 manifest,
* refuses any candidate that has already been evaluated on the holdout,
* appends the opening to ACCESS_LOG.jsonl *before* any data is returned, so a crash after
  reading still counts as an opening.
"""

from __future__ import annotations

import getpass
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from foundry.core.hashing import combined_hash, sha256_file

MANIFEST_NAME = "MANIFEST.json"
ACCESS_LOG_NAME = "ACCESS_LOG.jsonl"
SEAL_LOG_NAME = "SEAL_LOG.jsonl"
MANIFEST_VERSION = 1


class HoldoutError(Exception):
    """Any violation of the holdout rules."""


@dataclass(frozen=True)
class SealResult:
    content_hash: str
    changed: bool


@dataclass(frozen=True)
class HoldoutData:
    tables: dict[str, pl.DataFrame]
    manifest: dict[str, Any]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class HoldoutVault:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    # ---- sealing -----------------------------------------------------------------------

    def seal(self, tables: dict[str, pl.DataFrame], summary: dict[str, Any]) -> SealResult:
        """Write the holdout tables and manifest.

        Resealing identical content is a no-op. Resealing *different* content is refused
        once the holdout has been opened, because that would silently change the data a
        past evaluation was scored on.
        """
        if not tables:
            raise HoldoutError("nothing to seal")
        self._dir.mkdir(parents=True, exist_ok=True)
        staging = self._dir / ".staging"
        staging.mkdir(exist_ok=True)
        for old in staging.iterdir():
            old.unlink()

        file_hashes: dict[str, str] = {}
        for name, frame in sorted(tables.items()):
            fname = f"{name}.parquet"
            frame.write_parquet(staging / fname, statistics=False)
            file_hashes[fname] = sha256_file(staging / fname)
        content_hash = combined_hash(file_hashes.items())

        existing = self._read_manifest_or_none()
        if existing is not None and existing.get("content_hash") == content_hash:
            for f in staging.iterdir():
                f.unlink()
            staging.rmdir()
            return SealResult(content_hash=content_hash, changed=False)
        if existing is not None and self.access_log():
            for f in staging.iterdir():
                f.unlink()
            staging.rmdir()
            raise HoldoutError(
                "the holdout has already been opened; refusing to reseal it with different "
                f"content (sealed {existing.get('content_hash')}, new {content_hash})"
            )

        if existing is not None:
            for fname in existing.get("files", {}):
                target = self._dir / fname
                if target.exists():
                    target.unlink()
        for fname in file_hashes:
            os.replace(staging / fname, self._dir / fname)
        staging.rmdir()

        manifest = {
            "version": MANIFEST_VERSION,
            "sealed_at": _now(),
            "files": file_hashes,
            "content_hash": content_hash,
            "summary": summary,
        }
        tmp = self._dir / (MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._dir / MANIFEST_NAME)
        _append_jsonl(
            self._dir / SEAL_LOG_NAME,
            {"event": "seal", "at": manifest["sealed_at"], "content_hash": content_hash},
        )
        return SealResult(content_hash=content_hash, changed=True)

    # ---- inspection (no data content is read) ------------------------------------------

    def _read_manifest_or_none(self) -> dict[str, Any] | None:
        path = self._dir / MANIFEST_NAME
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def manifest(self) -> dict[str, Any]:
        m = self._read_manifest_or_none()
        if m is None:
            raise HoldoutError(f"no sealed holdout at {self._dir}")
        return m

    def verify(self) -> dict[str, Any]:
        """Check every sealed file against the manifest hashes. Reads bytes, not data."""
        m = self.manifest()
        files: dict[str, str] = m["files"]
        for fname, expected in files.items():
            path = self._dir / fname
            if not path.is_file():
                raise HoldoutError(f"sealed file missing: {fname}")
            actual = sha256_file(path)
            if actual != expected:
                raise HoldoutError(
                    f"sealed file {fname} was modified (hash {actual} != {expected})"
                )
        if combined_hash(files.items()) != m["content_hash"]:
            raise HoldoutError("manifest content hash does not match its file list")
        return m

    def access_log(self) -> list[dict[str, Any]]:
        path = self._dir / ACCESS_LOG_NAME
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def evaluated_candidates(self) -> set[str]:
        return {cid for entry in self.access_log() for cid in entry["candidate_ids"]}

    # ---- the one way in ----------------------------------------------------------------

    def open(self, engine_id: str, candidate_ids: list[str], reason: str) -> HoldoutData:
        if not engine_id:
            raise HoldoutError("engine_id is required to open the holdout")
        if not candidate_ids:
            raise HoldoutError("at least one candidate id is required to open the holdout")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise HoldoutError("duplicate candidate ids in holdout request")
        if not reason.strip():
            raise HoldoutError("a reason is required to open the holdout")

        manifest = self.verify()
        already = self.evaluated_candidates() & set(candidate_ids)
        if already:
            raise HoldoutError(
                f"candidate(s) already evaluated on the holdout: {sorted(already)}; "
                "a candidate may be evaluated on the holdout once"
            )

        _append_jsonl(
            self._dir / ACCESS_LOG_NAME,
            {
                "event": "open",
                "at": _now(),
                "engine_id": engine_id,
                "candidate_ids": list(candidate_ids),
                "reason": reason,
                "content_hash": manifest["content_hash"],
                "user": getpass.getuser(),
            },
        )
        tables = {
            fname.removesuffix(".parquet"): pl.read_parquet(self._dir / fname)
            for fname in sorted(manifest["files"])
        }
        return HoldoutData(tables=tables, manifest=manifest)
