"""Self-verification: pipelines record named checks and fail loudly if any does not pass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


class VerificationError(Exception):
    def __init__(self, failed: list[Check]) -> None:
        lines = "\n".join(f"  - {c.name}: {c.detail}" for c in failed)
        super().__init__(f"{len(failed)} verification check(s) failed:\n{lines}")
        self.failed = failed


@dataclass
class Verifier:
    checks: list[Check] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name=name, passed=bool(passed), detail=detail))
        return bool(passed)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return not self.failed

    def raise_if_failed(self) -> None:
        if self.failed:
            raise VerificationError(self.failed)
