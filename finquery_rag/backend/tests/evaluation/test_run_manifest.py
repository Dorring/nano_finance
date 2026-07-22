"""Tests for run manifest building and integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from src.evaluation.manifests import (
    compute_file_sha256,
    compute_git_state,
    compute_jsonl_sha256,
)
from src.evaluation.report_builder import build_run_manifest_output


class TestManifestRoundTrip:
    def test_manifest_round_trip(self) -> None:
        """Manifest survives JSON serialize → deserialize without loss."""
        manifest = build_run_manifest_output(
            git_commit="abc123",
            git_dirty=False,
            predictions_sha256="pred_hash",
            questions_sha256="q_hash",
            labels_sha256="l_hash",
            case_count=10,
            run_type="sealed",
            n_results=5,
            random_seed=42,
        )
        serialized = json.dumps(manifest, sort_keys=True)
        restored = json.loads(serialized)
        assert restored == manifest
        assert restored["git_commit"] == "abc123"
        assert restored["case_count"] == 10
        assert restored["run_type"] == "sealed"


class TestGitStateDetection:
    def test_git_state_detection(self) -> None:
        """compute_git_state returns (str, bool) without raising.

        We mock the subprocess to avoid depending on the test environment's
        git state.
        """
        with patch("src.evaluation.manifests.subprocess.run") as mock_run:
            commit_call = type(
                "R",
                (),
                {"stdout": "abc123def456\n", "returncode": 0},
            )()
            status_call = type(
                "R",
                (),
                {"stdout": "", "returncode": 0},
            )()
            mock_run.side_effect = [commit_call, status_call]
            commit, dirty = compute_git_state(".")
        assert commit == "abc123def456"
        assert dirty is False

    def test_git_state_dirty(self) -> None:
        """A non-empty porcelain output means dirty."""
        with patch("src.evaluation.manifests.subprocess.run") as mock_run:
            commit_call = type(
                "R", (), {"stdout": "abc123\n", "returncode": 0}
            )()
            status_call = type(
                "R", (), {"stdout": " M file.py\n", "returncode": 0}
            )()
            mock_run.side_effect = [commit_call, status_call]
            commit, dirty = compute_git_state(".")
        assert commit == "abc123"
        assert dirty is True


class TestSha256Deterministic:
    def test_sha256_deterministic(self, tmp_path: Path) -> None:
        """Same file content → same SHA256."""
        p = tmp_path / "data.txt"
        p.write_text("hello world", encoding="utf-8")
        h1 = compute_file_sha256(p)
        h2 = compute_file_sha256(p)
        assert h1 == h2
        assert h1 == hashlib.sha256(b"hello world").hexdigest()

    def test_sha256_changes_with_content(self, tmp_path: Path) -> None:
        """Different content → different SHA256."""
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_text("hello", encoding="utf-8")
        p2.write_text("world", encoding="utf-8")
        assert compute_file_sha256(p1) != compute_file_sha256(p2)


class TestJsonlSha256Stable:
    def test_jsonl_sha256_stable(self, tmp_path: Path) -> None:
        """Re-writing JSONL with different key order → same hash."""
        rows = [
            {"b": 2, "a": 1},
            {"c": 3, "d": 4},
        ]
        p1 = tmp_path / "ordered.jsonl"
        p2 = tmp_path / "reordered.jsonl"
        p1.write_text(
            "".join(
                json.dumps(r, sort_keys=True) + "\n" for r in rows
            ),
            encoding="utf-8",
        )
        p2.write_text(
            "".join(
                json.dumps(r, sort_keys=False) + "\n" for r in rows
            ),
            encoding="utf-8",
        )
        assert compute_jsonl_sha256(p1) == compute_jsonl_sha256(p2)

    def test_jsonl_sha256_ignores_blank_lines(self, tmp_path: Path) -> None:
        """Blank lines and comments do not affect the hash."""
        p1 = tmp_path / "clean.jsonl"
        p2 = tmp_path / "with_blanks.jsonl"
        p1.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
        p2.write_text('{"a": 1}\n\n# comment\n{"b": 2}\n', encoding="utf-8")
        assert compute_jsonl_sha256(p1) == compute_jsonl_sha256(p2)
