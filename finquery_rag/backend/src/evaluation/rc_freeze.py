"""RC freeze pre-flight verification for sealed evaluation runs.

Before a sealed blind run is permitted to execute, the following checks
MUST pass:

1. **Git HEAD** — the current commit must match the RC-frozen commit.
2. **Clean worktree** — ``git status --porcelain`` must be empty (no
   uncommitted changes).
3. **Selected config hash** — the SHA256 of ``selected-config.json`` must
   match the frozen hash (not just existence — full hash equality).
4. **Questions hash** — the SHA256 of the sealed questions file must match.
5. **Protocol hash** — the SHA256 of the evaluation protocol file must match.
6. **Corpus manifest hash** — the SHA256 of the corpus manifest must match.
7. **Sealed chroma hash** — the recursive directory hash of the sealed
   Chroma index must match.
8. **Sealed BM25 hash** — the SHA256 of the sealed BM25 database must match.
9. **Model checkpoint hash** — the SHA256 of the model checkpoint file must
   match.
10. **Tokenizer hash** — the SHA256 of the tokenizer file must match.
11. **Dependency lock hash** — the SHA256 of ``uv.lock`` must match.
12. **Model server name** — the actual runtime model name (from
    ``LLM_MODEL_NAME``) must match the frozen model server name.

If any check fails, :class:`RCFreezeViolation` is raised and the run is
aborted (fail-closed). This enforces the sealed evaluation discipline: the
exact same code, config, data, model, and tokenizer that were frozen at RC
time are what get evaluated.
"""

from __future__ import annotations

import hashlib
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
    preflight_git_clean: bool = False
    expected_commit: str = ""
    checks: dict[str, str] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "git_head": self.git_head,
            "git_dirty": self.git_dirty,
            "preflight_git_clean": self.preflight_git_clean,
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


def _verify_hash_equality(
    report: 'RCFreezeReport',
    check_name: str,
    file_path: Path | None,
    expected_hash: str | None,
) -> None:
    if expected_hash is None:
        report.checks[check_name] = 'SKIPPED: no expected hash'
        return
    if file_path is None or not file_path.exists():
        msg = f'{check_name}: file not found: {file_path}'
        report.violations.append(msg)
        report.checks[check_name] = f'FAIL: {msg}'
        return
    actual = _sha256_file(file_path)
    if actual != expected_hash:
        msg = f'{check_name} mismatch: exp={expected_hash[:16]}... got={actual[:16]}...'
        report.violations.append(msg)
        report.checks[check_name] = f'FAIL: {msg}'
    else:
        report.checks[check_name] = f'OK: {actual[:16]}...'


def _verify_dir_hash_equality(
    report: 'RCFreezeReport',
    check_name: str,
    dir_path: Path | None,
    expected_hash: str | None,
) -> None:
    if expected_hash is None:
        report.checks[check_name] = 'SKIPPED: no expected hash'
        return
    if dir_path is None or not dir_path.is_dir():
        msg = f'{check_name}: dir not found: {dir_path}'
        report.violations.append(msg)
        report.checks[check_name] = f'FAIL: {msg}'
        return
    h = hashlib.sha256()
    for fp in sorted(dir_path.rglob('*')):
        if fp.is_file():
            h.update(str(fp.relative_to(dir_path)).encode())
            h.update(b'\0')
            with open(fp, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            h.update(b'\0')
    actual = h.hexdigest()
    if actual != expected_hash:
        msg = f'{check_name} mismatch: exp={expected_hash[:16]}... got={actual[:16]}...'
        report.violations.append(msg)
        report.checks[check_name] = f'FAIL: {msg}'
    else:
        report.checks[check_name] = f'OK: {actual[:16]}...'


def verify_rc_freeze(
    repo_path: Path,
    *,
    expected_commit: str | None = None,
    selected_config_path: Path | None = None,
    questions_path: Path | None = None,
    expected_questions_sha256: str | None = None,
    protocol_path: Path | None = None,
    expected_protocol_sha256: str | None = None,
    expected_selected_config_sha256: str | None = None,
    corpus_manifest_path: Path | None = None,
    expected_corpus_manifest_sha256: str | None = None,
    sealed_chroma_path: Path | None = None,
    expected_chroma_sha256: str | None = None,
    sealed_bm25_path: Path | None = None,
    expected_bm25_sha256: str | None = None,
    model_checkpoint_path: Path | None = None,
    expected_model_checkpoint_sha256: str | None = None,
    tokenizer_path: Path | None = None,
    expected_tokenizer_sha256: str | None = None,
    dependency_lock_path: Path | None = None,
    expected_dependency_lock_sha256: str | None = None,
    expected_model_server_name: str | None = None,
    actual_model_server_name: str | None = None,
    require_clean_worktree: bool = True,
) -> RCFreezeReport:
    """Run all RC freeze pre-flight checks.

    Args:
        repo_path: Path to the git repository root.
        expected_commit: The RC-frozen git commit SHA. If None, the check
            is skipped (but recorded as skipped).
        selected_config_path: Path to ``selected-config.json``.
        questions_path: Path to the sealed questions file.
        expected_questions_sha256: Expected SHA256 of the questions file.
        protocol_path: Path to the evaluation protocol file.
        expected_protocol_sha256: Expected SHA256 of the protocol file.
        expected_selected_config_sha256: Expected SHA256 of selected config.
        corpus_manifest_path: Path to the corpus manifest JSON.
        expected_corpus_manifest_sha256: Expected SHA256 of corpus manifest.
        sealed_chroma_path: Path to the sealed Chroma index directory.
        expected_chroma_sha256: Expected recursive directory hash of Chroma.
        sealed_bm25_path: Path to the sealed BM25 database file.
        expected_bm25_sha256: Expected SHA256 of the BM25 database.
        model_checkpoint_path: Path to the model checkpoint file.
        expected_model_checkpoint_sha256: Expected SHA256 of the checkpoint.
        tokenizer_path: Path to the tokenizer file.
        expected_tokenizer_sha256: Expected SHA256 of the tokenizer.
        dependency_lock_path: Path to the dependency lock file (``uv.lock``).
        expected_dependency_lock_sha256: Expected SHA256 of the lock file.
        expected_model_server_name: Frozen model server name.
        actual_model_server_name: Runtime model name (from ``LLM_MODEL_NAME``).
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

    # 2. Clean worktree (preflight)
    report.git_dirty = _git_dirty(repo_path)
    report.preflight_git_clean = not report.git_dirty
    if require_clean_worktree and report.git_dirty:
        msg = "Git worktree is dirty (uncommitted changes present)"
        report.violations.append(msg)
        report.checks["git_clean"] = f"FAIL: {msg}"
    else:
        status = "dirty" if report.git_dirty else "clean"
        report.checks["git_clean"] = f"OK: {status}"

    # 3. Selected config — hash equality (not just existence)
    _verify_hash_equality(
        report,
        "selected_config_hash",
        selected_config_path,
        expected_selected_config_sha256,
    )

    # 4. Questions hash
    _verify_hash_equality(
        report,
        "questions_hash",
        questions_path,
        expected_questions_sha256,
    )

    # 5. Protocol hash
    _verify_hash_equality(
        report,
        "protocol_hash",
        protocol_path,
        expected_protocol_sha256,
    )

    # 6. Corpus manifest hash
    _verify_hash_equality(
        report,
        "corpus_manifest_hash",
        corpus_manifest_path,
        expected_corpus_manifest_sha256,
    )

    # 7. Sealed chroma directory hash
    _verify_dir_hash_equality(
        report,
        "sealed_chroma_hash",
        sealed_chroma_path,
        expected_chroma_sha256,
    )

    # 8. Sealed BM25 database hash
    _verify_hash_equality(
        report,
        "sealed_bm25_hash",
        sealed_bm25_path,
        expected_bm25_sha256,
    )

    # 9. Model checkpoint hash
    _verify_hash_equality(
        report,
        "model_checkpoint_hash",
        model_checkpoint_path,
        expected_model_checkpoint_sha256,
    )

    # 10. Tokenizer hash
    _verify_hash_equality(
        report,
        "tokenizer_hash",
        tokenizer_path,
        expected_tokenizer_sha256,
    )

    # 11. Dependency lock hash
    _verify_hash_equality(
        report,
        "dependency_lock_hash",
        dependency_lock_path,
        expected_dependency_lock_sha256,
    )

    # 12. Model server name binding
    if expected_model_server_name is not None:
        if actual_model_server_name is None:
            msg = (
                f"model_server_name: actual not provided "
                f"(expected={expected_model_server_name})"
            )
            report.violations.append(msg)
            report.checks["model_server_name"] = f"FAIL: {msg}"
        elif actual_model_server_name != expected_model_server_name:
            msg = (
                f"model_server_name mismatch: expected={expected_model_server_name}, "
                f"got={actual_model_server_name}"
            )
            report.violations.append(msg)
            report.checks["model_server_name"] = f"FAIL: {msg}"
        else:
            report.checks["model_server_name"] = f"OK: {actual_model_server_name}"
    else:
        report.checks["model_server_name"] = "SKIPPED: no expected_model_server_name"

    # Final verdict
    report.passed = len(report.violations) == 0
    if not report.passed:
        raise RCFreezeViolation(
            "RC freeze pre-flight checks failed:\n  - "
            + "\n  - ".join(report.violations)
        )

    return report
