"""Phase 5 run manifests for sealed evaluation reproducibility.

A ``RunManifest`` captures the full environment and input state of an
evaluation run: git commit, model checkpoint hashes, corpus/index hashes,
Python/dependency info, and the SHA256 of the predictions/questions/labels
files. The manifest is written next to the predictions so a sealed scorer
can independently verify integrity before scoring.

All helpers are offline and deterministic: ``compute_file_sha256`` and
``compute_jsonl_sha256`` only read bytes, and ``compute_git_state`` shells
out to git but never raises.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    """Immutable record of an evaluation run's environment and inputs."""

    run_id: str
    run_type: str  # "baseline", "calibration", "sealed", "ablation"
    git_commit: str
    git_dirty: bool  # post-run dirty state (may include new artifacts)
    preflight_git_clean: bool  # pre-run clean state from RC freeze
    model_checkpoint_path: str | None
    model_checkpoint_sha256: str | None
    effective_model_name: str | None
    effective_checkpoint_sha256: str | None
    tokenizer_sha256: str | None
    embedding_model: str | None
    reranker_model: str | None
    corpus_manifest_hash: str | None
    vector_index_hash: str | None
    bm25_index_manifest: str | None
    config_hash: str | None
    python_version: str
    dependency_lock_hash: str | None
    cpu_info: str | None
    gpu_info: str | None
    random_seed: int
    n_results: int
    temperature: float | None
    run_started_at: str
    run_ended_at: str | None
    predictions_sha256: str | None
    questions_sha256: str | None
    labels_sha256: str | None
    case_count: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        """Reconstruct a ``RunManifest`` from a JSON-compatible dict."""
        return cls(
            run_id=str(data["run_id"]),
            run_type=str(data["run_type"]),
            git_commit=str(data.get("git_commit", "")),
            git_dirty=bool(data.get("git_dirty", False)),
            preflight_git_clean=bool(data.get("preflight_git_clean", not data.get("git_dirty", False))),
            model_checkpoint_path=data.get("model_checkpoint_path"),
            model_checkpoint_sha256=data.get("model_checkpoint_sha256"),
            effective_model_name=data.get("effective_model_name"),
            effective_checkpoint_sha256=data.get("effective_checkpoint_sha256"),
            tokenizer_sha256=data.get("tokenizer_sha256"),
            embedding_model=data.get("embedding_model"),
            reranker_model=data.get("reranker_model"),
            corpus_manifest_hash=data.get("corpus_manifest_hash"),
            vector_index_hash=data.get("vector_index_hash"),
            bm25_index_manifest=data.get("bm25_index_manifest"),
            config_hash=data.get("config_hash"),
            python_version=str(data.get("python_version", "")),
            dependency_lock_hash=data.get("dependency_lock_hash"),
            cpu_info=data.get("cpu_info"),
            gpu_info=data.get("gpu_info"),
            random_seed=int(data.get("random_seed", 0)),
            n_results=int(data.get("n_results", 0)),
            temperature=data.get("temperature"),
            run_started_at=str(data.get("run_started_at", "")),
            run_ended_at=data.get("run_ended_at"),
            predictions_sha256=data.get("predictions_sha256"),
            questions_sha256=data.get("questions_sha256"),
            labels_sha256=data.get("labels_sha256"),
            case_count=int(data.get("case_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict with stable key ordering."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "post_run_git_dirty": self.git_dirty,
            "preflight_git_clean": self.preflight_git_clean,
            "model_checkpoint_path": self.model_checkpoint_path,
            "model_checkpoint_sha256": self.model_checkpoint_sha256,
            "effective_model_name": self.effective_model_name,
            "effective_checkpoint_sha256": self.effective_checkpoint_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "vector_index_hash": self.vector_index_hash,
            "bm25_index_manifest": self.bm25_index_manifest,
            "config_hash": self.config_hash,
            "python_version": self.python_version,
            "dependency_lock_hash": self.dependency_lock_hash,
            "cpu_info": self.cpu_info,
            "gpu_info": self.gpu_info,
            "random_seed": self.random_seed,
            "n_results": self.n_results,
            "temperature": self.temperature,
            "run_started_at": self.run_started_at,
            "run_ended_at": self.run_ended_at,
            "predictions_sha256": self.predictions_sha256,
            "questions_sha256": self.questions_sha256,
            "labels_sha256": self.labels_sha256,
            "case_count": self.case_count,
        }


def compute_git_state(repo_path: str | Path = ".") -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)`` for the git repo at ``repo_path``.

    Runs ``git rev-parse HEAD`` and ``git status --porcelain``. Returns
    ``("", False)`` when git is unavailable or the path is not a repo.
    Never raises.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        status = ""
    return commit, bool(status.strip())


def compute_file_sha256(path: str | Path) -> str | None:
    """Return the SHA256 hex digest of a file's raw bytes."""
    hasher = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def compute_jsonl_sha256(path: str | Path) -> str:
    """Return a deterministic SHA256 over the canonical JSONL content.

    Each non-empty, non-comment line is parsed as JSON and re-serialized
    with sorted keys so the hash is stable regardless of key ordering or
    insignificant whitespace. This makes the manifest hash reproducible
    across re-writes of the same logical content. Lines that fail to parse
    are hashed as-is so a corrupt file still produces a stable digest.
    """
    hasher = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj: Any = json.loads(stripped)
                canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True)
            except json.JSONDecodeError:
                canonical = stripped
            hasher.update(canonical.encode("utf-8"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def compute_raw_file_sha256(path: str | Path) -> str:
    """Return SHA256 of raw file bytes (alias for compute_file_sha256)."""
    return compute_file_sha256(path)


def compute_dual_hashes(path: str | Path) -> dict[str, str]:
    """Compute both raw and canonical SHA256 hashes.

    Returns ``{"raw_sha256": ..., "canonical_sha256": ...}``. The raw hash
    is over the exact file bytes and changes with any byte-level change
    (including key reordering). The canonical hash is order-independent
    and stable across re-serialization of the same logical JSONL content.
    """
    return {
        "raw_sha256": compute_file_sha256(path),
        "canonical_sha256": compute_jsonl_sha256(path),
    }


def verify_dual_hashes(
    path: str | Path, expected_raw: str, expected_canonical: str
) -> bool:
    """Verify both raw and canonical hashes match the expected values."""
    actual = compute_dual_hashes(path)
    return (
        actual["raw_sha256"] == expected_raw
        and actual["canonical_sha256"] == expected_canonical
    )


def _safe_cpu_info() -> str | None:
    """Best-effort offline CPU description; never raises."""
    try:
        return platform.processor() or None
    except Exception:  # noqa: BLE001 - diagnostic only.
        return None


def _safe_gpu_info() -> str | None:
    """Offline GPU description.

    Intentionally returns ``None``: this module never imports GPU drivers
    or shells out to ``nvidia-smi`` so it stays deterministic and offline.
    Callers that need GPU info should populate the field explicitly.
    """
    return None


def create_manifest(
    *,
    run_id: str,
    run_type: str,
    n_results: int,
    case_count: int,
    run_started_at: str,
    questions_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    model_checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    corpus_manifest_path: str | Path | None = None,
    vector_index_path: str | Path | None = None,
    bm25_index_manifest: str | None = None,
    config_path: str | Path | None = None,
    dependency_lock_path: str | Path | None = None,
    random_seed: int = 0,
    temperature: float | None = None,
    run_ended_at: str | None = None,
    repo_path: str | Path = ".",
    preflight_git_clean: bool | None = None,
    effective_model_name: str | None = None,
    effective_checkpoint_sha256: str | None = None,
) -> RunManifest:
    """Build a ``RunManifest``, filling in git/env info and file hashes.

    File hash fields are computed only when the corresponding path is
    provided. Git state is always attempted but degrades gracefully to
    ``("", False)`` outside a repo.

    Args:
        preflight_git_clean: Clean state captured BEFORE the blind run
            (from RC freeze report). If None, defaults to the post-run
            inverse of ``git_dirty``.
        effective_model_name: The actual model name used at runtime
            (from ``LLM_MODEL_NAME``).
        effective_checkpoint_sha256: SHA256 of the actual checkpoint used.
    """
    commit, dirty = compute_git_state(repo_path)
    return RunManifest(
        run_id=run_id,
        run_type=run_type,
        git_commit=commit,
        git_dirty=dirty,
        preflight_git_clean=(not dirty if preflight_git_clean is None else preflight_git_clean),
        model_checkpoint_path=model_checkpoint_path,
        model_checkpoint_sha256=(
            compute_file_sha256(model_checkpoint_path)
            if model_checkpoint_path
            else None
        ),
        effective_model_name=effective_model_name,
        effective_checkpoint_sha256=effective_checkpoint_sha256,
        tokenizer_sha256=(
            compute_file_sha256(tokenizer_path) if tokenizer_path else None
        ),
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        corpus_manifest_hash=(
            compute_file_sha256(corpus_manifest_path) if corpus_manifest_path else None
        ),
        vector_index_hash=(
            compute_file_sha256(vector_index_path) if vector_index_path else None
        ),
        bm25_index_manifest=bm25_index_manifest,
        config_hash=compute_file_sha256(config_path) if config_path else None,
        python_version=sys.version,
        dependency_lock_hash=(
            compute_file_sha256(dependency_lock_path) if dependency_lock_path else None
        ),
        cpu_info=_safe_cpu_info(),
        gpu_info=_safe_gpu_info(),
        random_seed=random_seed,
        n_results=n_results,
        temperature=temperature,
        run_started_at=run_started_at,
        run_ended_at=run_ended_at,
        predictions_sha256=(
            compute_jsonl_sha256(predictions_path) if predictions_path else None
        ),
        questions_sha256=(
            compute_jsonl_sha256(questions_path) if questions_path else None
        ),
        labels_sha256=compute_jsonl_sha256(labels_path) if labels_path else None,
        case_count=case_count,
    )
