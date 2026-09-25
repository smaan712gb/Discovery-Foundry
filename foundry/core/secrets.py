"""Secrets come from environment variables only, optionally seeded from a git-ignored `.env` file.

Values are never logged, stored or returned in error messages. Use `redact` on any text that
might contain a secret before it leaves the process.
"""

from __future__ import annotations

import os
from pathlib import Path


class SecretError(Exception):
    pass


def load_dotenv(path: Path) -> list[str]:
    """Set variables from a KEY=VALUE file without overriding the environment.

    Returns the names that were set (never the values).
    """
    if not path.is_file():
        return []
    set_names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            set_names.append(key)
    return set_names


def require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise SecretError(f"environment variable {name} is not set (see .env.example)")
    return value


def redact(text: str, *secrets: str) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[REDACTED]")
    return text
