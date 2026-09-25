"""Reproducibility metadata stored with every artefact: seed, git state, versions."""

from __future__ import annotations

import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

# Packages whose versions can change numerical results.
TRACKED_PACKAGES: tuple[str, ...] = (
    "polars",
    "pyarrow",
    "numpy",
    "duckdb",
    "pydantic",
    "pyyaml",
    "tzdata",
)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state(repo_root: Path) -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    return {
        "commit": commit,
        # No commit or unreadable status counts as dirty: the code state is not pinned.
        "dirty": commit is None or status is None or status != "",
    }


def package_versions(names: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_metadata(seed: int, config_hash: str, repo_root: Path) -> dict[str, Any]:
    return {
        "seed": seed,
        "config_hash": config_hash,
        "git": git_state(repo_root),
        "python": platform.python_version(),
        "packages": package_versions(),
    }
