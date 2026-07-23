"""Tests for the dual hash utilities in manifests.py."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.manifests import (
    compute_dual_hashes,
    compute_file_sha256,
    compute_jsonl_sha256,
    verify_dual_hashes,
)


def _write_jsonl(path: Path, rows: list[dict], *, sort_keys: bool) -> None:
    path.write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, sort_keys=sort_keys) + "\n" for r in rows
        ),
        encoding="utf-8",
    )


def test_compute_dual_hashes(tmp_path: Path) -> None:
    """Both raw and canonical hashes are computed and match the primitives."""
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1, "b": 2}], sort_keys=True)

    hashes = compute_dual_hashes(p)

    assert set(hashes.keys()) == {"raw_sha256", "canonical_sha256"}
    assert hashes["raw_sha256"] == compute_file_sha256(p)
    assert hashes["canonical_sha256"] == compute_jsonl_sha256(p)


def test_verify_dual_hashes_match(tmp_path: Path) -> None:
    """Verification passes when both hashes match the file."""
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}], sort_keys=True)
    hashes = compute_dual_hashes(p)

    assert (
        verify_dual_hashes(p, hashes["raw_sha256"], hashes["canonical_sha256"]) is True
    )


def test_verify_dual_hashes_mismatch(tmp_path: Path) -> None:
    """Verification fails when the raw hash differs."""
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}], sort_keys=True)
    canonical = compute_jsonl_sha256(p)

    assert verify_dual_hashes(p, "0" * 64, canonical) is False


def test_raw_vs_canonical_differ(tmp_path: Path) -> None:
    """Raw and canonical hashes differ when key order is non-canonical.

    Two files with the same logical content but different key order have
    different raw hashes (byte-level) but identical canonical hashes.
    """
    sorted_p = tmp_path / "sorted.jsonl"
    reordered_p = tmp_path / "reordered.jsonl"
    rows = [{"a": 1, "b": 2, "c": 3}]
    _write_jsonl(sorted_p, rows, sort_keys=True)
    # Preserved (non-canonical) key order via sort_keys=False.
    _write_jsonl(reordered_p, [{"c": 3, "b": 2, "a": 1}], sort_keys=False)

    sorted_hashes = compute_dual_hashes(sorted_p)
    reordered_hashes = compute_dual_hashes(reordered_p)

    assert sorted_hashes["raw_sha256"] != reordered_hashes["raw_sha256"]
    assert sorted_hashes["canonical_sha256"] == reordered_hashes["canonical_sha256"]

    # A non-canonical file's raw and canonical hashes differ from each other.
    assert reordered_hashes["raw_sha256"] != reordered_hashes["canonical_sha256"]
