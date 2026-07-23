#!/usr/bin/env python3
"""Freeze the Phase 5 release candidate.

This script records the frozen state of the release candidate:
- Git commit SHA
- Model checkpoint hash
- Tokenizer hash
- Corpus/index hash
- Config hash
- Selected configuration
- Protocol hash

The RC freeze must happen BEFORE the sealed blind run. Once frozen,
the worktree must be clean (no uncommitted changes) before running
the sealed evaluation.

Usage:
    python3 scripts/freeze_phase5_rc.py
"""
from __future__ import annotations

import hashlib
import os
import json
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.manifests import compute_jsonl_sha256  # noqa: E402

OUTPUT_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "rc-freeze-manifest.json"
)

PROTOCOL_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "protocol"
    / "phase5-evaluation-protocol.json"
)

SELECTED_CONFIG_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "calibration-v2"
    / "selected-config.json"
)

SEALED_QUESTIONS_PATH = (
    BACKEND_DIR / "eval_data" / "phase5" / "sealed" / "questions.jsonl"
)

SEALED_INDEX_DIR = BACKEND_DIR / "indexes" / "phase5" / "sealed"
CORPUS_MANIFEST_PATH = (
    BACKEND_DIR / "eval_corpus" / "phase5" / "corpus-manifest.json"
)


def compute_sha256(filepath: Path) -> str | None:
    if not filepath.is_file():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_sha256(dirpath: Path) -> str | None:
    if not dirpath.is_dir():
        return None
    h = hashlib.sha256()
    files = sorted(dirpath.rglob("*"))
    for fp in files:
        if fp.is_file():
            rel = str(fp.relative_to(dirpath))
            h.update(rel.encode())
            h.update(b"\0")
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


def git_is_clean() -> bool:
    """Check if the git worktree is clean (no modified tracked files).

    Untracked files are ignored — only modifications to tracked files
    prevent the sealed run.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, cwd=str(BACKEND_DIR)
    )
    return result.stdout.strip() == ""


def main():
    print("=" * 60)
    print("Phase 5 Release Candidate Freeze")
    print("=" * 60)

    # Check worktree is clean
    clean = git_is_clean()
    print(f"\nGit worktree clean: {clean}")
    if not clean:
        print("WARNING: Worktree is not clean. Sealed run should not proceed.")
        print("Commit all changes before running sealed evaluation.")

    # Get current commit
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(BACKEND_DIR)
    )
    rc_commit = result.stdout.strip()
    print(f"RC commit: {rc_commit}")

    # Compute questions hash for sealed partition
    questions_sha256 = None
    if SEALED_QUESTIONS_PATH.is_file():
        questions_sha256 = compute_jsonl_sha256(SEALED_QUESTIONS_PATH)

    # Compute sealed partition index hashes (not default chroma_db)
    sealed_chroma_dir = SEALED_INDEX_DIR / "chroma"
    sealed_bm25_db = SEALED_INDEX_DIR / "rag_bm25.db"
    sealed_chunk_manifest = SEALED_INDEX_DIR / "chunk-manifest.json"

    # Compute corpus manifest hash
    corpus_manifest_sha256 = compute_sha256(CORPUS_MANIFEST_PATH)

    # Compute hashes
    freeze_record = {
        "rc_commit": rc_commit,
        "expected_commit": rc_commit,  # for blind runner verification
        "worktree_clean": clean,
        "frozen_at": "2026-07-23T00:00:00Z",  # deterministic
        "protocol_sha256": compute_sha256(PROTOCOL_PATH),
        "selected_config_sha256": compute_sha256(SELECTED_CONFIG_PATH),
        "config_sha256": compute_sha256(BACKEND_DIR / "pyproject.toml"),
        "questions_sha256": questions_sha256,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "sealed_chroma_sha256": compute_dir_sha256(sealed_chroma_dir),
        "sealed_bm25_sha256": compute_sha256(sealed_bm25_db),
        "sealed_chunk_manifest_sha256": compute_sha256(sealed_chunk_manifest),
        "model_checkpoint_path": "chatsft_checkpoints/d24_finance_v2_lr010/model_000275.pt",
        "model_checkpoint_sha256": compute_sha256(
            Path(os.path.expanduser("~/.cache/nanochat/chatsft_checkpoints/d24_finance_v2_lr010/model_000275.pt"))
        ),
        "tokenizer_path": "tokenizer/tokenizer.pkl",
        "tokenizer_sha256": compute_sha256(
            Path("/mnt/disk/mxf/.cache/nanochat/tokenizer/tokenizer.pkl")
        ),
        "model_server_endpoint": "http://localhost:8500",
        "model_server_name": "finquery-finance-sft1147",
    }

    # Load selected config — use "params" key (not "selected_params")
    with open(SELECTED_CONFIG_PATH, "r", encoding="utf-8") as f:
        selected_config = json.load(f)
    freeze_record["selected_params"] = selected_config.get("params", {})

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(freeze_record, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nRC freeze record saved to: {OUTPUT_PATH}")

    print("\nFreeze Summary:")
    print(f"  RC commit: {rc_commit}")
    print(f"  Worktree clean: {clean}")
    print(f"  Protocol SHA256: {freeze_record['protocol_sha256']}")
    print(f"  Selected config SHA256: {freeze_record['selected_config_sha256']}")
    print(f"  Questions SHA256: {freeze_record['questions_sha256']}")
    print(f"  Corpus manifest SHA256: {freeze_record['corpus_manifest_sha256']}")
    print(f"  Sealed chroma SHA256: {freeze_record['sealed_chroma_sha256']}")
    print(f"  Sealed BM25 SHA256: {freeze_record['sealed_bm25_sha256']}")
    print(f"  Model checkpoint SHA256: {freeze_record['model_checkpoint_sha256']}")
    print(f"  Tokenizer SHA256: {freeze_record['tokenizer_sha256']}")
    print(f"  Selected params: {freeze_record['selected_params']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
