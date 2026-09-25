"""The protocol a discovery domain implements. Core code depends only on this."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Domain(Protocol):
    name: str

    def build_data(self, config_path: Path, repo_root: Path) -> int:
        """Build processed data and sealed holdout; return a process exit code."""
        ...
