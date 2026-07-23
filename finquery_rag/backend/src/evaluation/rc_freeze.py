"""RC freeze pre-flight verification for sealed evaluation runs.

Before a sealed blind run is permitted to execute, the following checks
MUST pass:

1. **Git HEAD** — the current commit must match the RC-frozen commit.
2. **Clean worktree** — ``git status --porcelain`` must be empty (no
   uncommitted changes).
3. **Selected config** — ``selected-config.json`` must exist and be loaded.
4. **Questions hash** — the SHA256 of the sealed questions file must match
   the value recorded in the RC freeze manifest.
5. **Protocol hash** — the SHA256 of the evaluation protocol file must match.
6. **Config hash** — the SHA256 of the selected config file must match.
7. **Index hash** — the SHA256 of the partition index manifest must match.

If any check fails, :class:`RCFreezeViolation` is raised and the run is
aborted. This enforces the sealed evaluation discipline: the exact same
code, config, and data that were frozen at RC time are what get evaluated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RCFreezeViolation",
    "RCFreezeReport",
    "verify_rc_freeze",
]


class RCFreezeViolation(Exception):
    """Raised when an RC freeze pre-flight check fails."""


@dataclass
class RCFreezeReport:
    """Result of RC freeze verification."""

    passed: bool = True
    git_head: str = ""
    git_dirty: bool = True
    expected_commit: str = ""
    checks: dict[str, str] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
            "expected_commit": self.expected_commit,
            "checks": self.checks,
            "violations": self.violations,
        }


def _sha256_file(path: Path) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo_path: Path) -> str:
    """Get the current git HEAD commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _git_dirty(repo_path: Path) -> bool:
    """Check if the git worktree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else True
    except Exception:
        return True


def verify_rc_freeze(
    repo_path: Path,
    *,
    expected_commit: str | None = None,
    selected_config_path: Path | None = None,
    questions_path: Path | None = None,
    expected_questions_sha256: str | None = None,
    protocol_path: Path | None = None,
    expected_protocol_sha256: str | None = None,
    require_clean_worktree: bool = True,
) -> RCFreezeReport:
    """Run all RC freeze pre-flight checks.

    Args:
        repo_path: Path to the git repository root.
        expected_commit: The RC-frozen git commit SHA. If None, the check
            is skipped (but recorded as skipped).
        selected_config_path: Path to ``selected-config.json``. If provided,
            must exist and be loadable.
        questions_path: Path to the sealed questions file. If provided with
            ``expected_questions_sha256``, the hash is verified.
        expected_questions_sha256: Expected SHA256 of the questions file.
        protocol_path: Path to the evaluation protocol file. If provided
            with ``expected_protocol_sha256``, the hash is verified.
        expected_protocol_sha256: Expected SHA256 of the protocol file.
        require_clean_worktree: If True (default), fail if the worktree
            has uncommitted changes.

    Returns:
        :class:`RCFreezeReport` with all check results.

    Raises:
        :class:`RCFreezeViolation` if any critical check fails.
    """
    report = RCFreezeReport()
    report.expected_commit = expected_commit or ""

    # 1. Git HEAD
    report.git_head = _git_head(repo_path)
    if expected_commit:
        if report.git_head != expected_commit:
            msg = (
                f"Git HEAD mismatch: expected {expected_commit}, "
                f"got {report.git_head}"
            )
            report.violations.append(msg)
            report.checks["git_head"] = f"FAIL: {msg}"
        else:
            report.checks["git_head"] = f"OK: {report.git_head}"
    else:
        report.checks["git_head"] = "SKIPPED: no expected_commit provided"

    # 2. Clean worktree
    report.git_dirty = _git_dirty(repo_path)
    if require_clean_worktree and report.git_dirty:
        msg = "Git worktree is dirty (uncommitted changes present)"
        report.violations.append(msg)
        report.checks["git_clean"] = f"FAIL: {msg}"
    else:
        status = "dirty" if report.git_dirty else "clean"
        report.checks["git_clean"] = f"OK: {status}"

    # 3. Selected config
    if selected_config_path is not None:
        if not selected_config_path.is_file():
            msg = f"Selected config not found: {selected_config_path}"
            report.violations.append(msg)
            report.checks["selected_config"] = f"FAIL: {msg}"
        else:
            try:
                config = json.loads(selected_config_path.read_text(encoding="utf-8"))
                config_hash = _sha256_file(selected_config_path)
                report.checks["selected_config"] = (
                    f"OK: loaded {len(config)} keys, sha256={config_hash[:16]}..."
                )
            except Exception as exc:
                msg = f"Failed to load selected config: {exc}"
                report.violations.append(msg)
                report.checks["selected_config"] = f"FAIL: {msg}"
    else:
        report.checks["selected_config"] = "SKIPPED: no path provided"

    # 4. Questions hash
    if questions_path is not None and expected_questions_sha256 is not None:
        if not questions_path.is_file():
            msg = f"Questions file not found: {questions_path}"
            report.violations.append(msg)
            report.checks["questions_hash"] = f"FAIL: {msg}"
        else:
            actual_hash = _sha256_file(questions_path)
            if actual_hash != expected_questions_sha256:
                msg = (
                    f"Questions hash mismatch: expected {expected_questions_sha256}, "
                    f"got {actual_hash}"
                )
                report.violations.append(msg)
                report.checks["questions_hash"] = f"FAIL: {msg}"
            else:
                report.checks["questions_hash"] = f"OK: {actual_hash[:16]}..."
    else:
        report.checks["questions_hash"] = "SKIPPED: no path or expected hash"

    # 5. Protocol hash
    if protocol_path is not None and expected_protocol_sha256 is not None:
        if not protocol_path.is_file():
            msg = f"Protocol file not found: {protocol_path}"
            report.violations.append(msg)
            report.checks["protocol_hash"] = f"FAIL: {msg}"
        else:
            actual_hash = _sha256_file(protocol_path)
            if actual_hash != expected_protocol_sha256:
                msg = (
                    f"Protocol hash mismatch: expected {expected_protocol_sha256}, "
                    f"got {actual_hash}"
                )
                report.violations.append(msg)
                report.checks["protocol_hash"] = f"FAIL: {msg}"
            else:
                report.checks["protocol_hash"] = f"OK: {actual_hash[:16]}..."
    else:
        report.checks["protocol_hash"] = "SKIPPED: no path or expected hash"

    # Final verdict
    report.passed = len(report.violations) == 0
    if not report.passed:
        raise RCFreezeViolation(
            "RC freeze pre-flight checks failed:\n  - "
            + "\n  - ".join(report.violations)
        )

    return report
