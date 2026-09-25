"""SHA-256 helpers for files and byte strings."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def combined_hash(parts: Iterable[tuple[str, str]]) -> str:
    """Order-independent hash over (name, hash) pairs, e.g. all output files of a build."""
    digest = hashlib.sha256()
    for name, value in sorted(parts):
        digest.update(f"{name}={value}\n".encode())
    return digest.hexdigest()
