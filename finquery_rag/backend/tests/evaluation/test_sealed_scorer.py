"""Tests for the Phase 5 sealed scorer.

Verifies that perfect predictions pass, case_id mismatches fail, the
predictions file is never modified, scoring is deterministic, and the
scorer never calls the RAG engine.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation.manifests import compute_jsonl_sha256
from src.evaluation.sealed_scorer import score_sealed_predictions


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL file with sorted keys for stable hashing."""
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_protocol(
    path: Path,
    *,
    predictions_sha: str,
    labels_sha: str,
    run_id: str = "run-1",
) -> None:
    """Write a minimal run manifest (protocol) JSON."""
    manifest = {
        "run_id": run_id,
        "run_type": "sealed",
        "git_commit": "abc123",
        "git_dirty": False,
        "predictions_sha256": predictions_sha,
        "labels_sha256": labels_sha,
        "case_count": 1,
        "python_version": "3.12",
        "random_seed": 0,
        "n_results": 3,
        "run_started_at": "2026-07-23T00:00:00Z",
    }
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def _make_perfect_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create matching predictions + labels + protocol files."""
    preds_path = tmp_path / "preds.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    protocol_path = tmp_path / "protocol.json"

    _write_jsonl(
        preds_path,
        [
            {
                "case_id": "c1",
                "answer": "Revenue was 1000.",
                "sources": [{"filename": "r.pdf", "page": 1}],
                "calculations": [],
                "answerability": {"status": "answerable"},
                "validation": {"status": "passed"},
            }
        ],
    )
    _write_jsonl(
        labels_path,
        [
            {
                "case_id": "c1",
                "required_answer_terms": ["Revenue"],
                "expected_numbers": ["1000"],
                "expected_answerability": "answerable",
                "expected_validation_status": "passed",
            }
        ],
    )
    _write_protocol(
        protocol_path,
        predictions_sha=compute_jsonl_sha256(preds_path),
        labels_sha=compute_jsonl_sha256(labels_path),
    )
    return {
        "predictions": preds_path,
        "labels": labels_path,
        "protocol": protocol_path,
    }


def test_perfect_predictions(tmp_path: Path) -> None:
    """All checks pass when predictions match the labels."""
    fixture = _make_perfect_fixture(tmp_path)
    out_path = tmp_path / "report.json"

    report = score_sealed_predictions(
        predictions_path=fixture["predictions"],
        labels_path=fixture["labels"],
        protocol_path=fixture["protocol"],
        output_path=out_path,
    )

    assert report["summary"]["total"] == 1
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 0
    assert report["summary"]["pass_rate"] == 1.0
    assert out_path.exists()
    assert report["cases"][0]["passed"] is True


def test_missing_prediction_fails(tmp_path: Path) -> None:
    """A case_id present in labels but absent from predictions must fail."""
    preds_path = tmp_path / "preds.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    protocol_path = tmp_path / "protocol.json"

    _write_jsonl(preds_path, [{"case_id": "c1", "answer": "ok"}])
    _write_jsonl(labels_path, [{"case_id": "c1"}, {"case_id": "c2"}])
    _write_protocol(
        protocol_path,
        predictions_sha=compute_jsonl_sha256(preds_path),
        labels_sha=compute_jsonl_sha256(labels_path),
    )

    with pytest.raises(ValueError, match="missing"):
        score_sealed_predictions(
            predictions_path=preds_path,
            labels_path=labels_path,
            protocol_path=protocol_path,
            output_path=tmp_path / "out.json",
        )


def test_extra_prediction_fails(tmp_path: Path) -> None:
    """A case_id present in predictions but absent from labels must fail."""
    preds_path = tmp_path / "preds.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    protocol_path = tmp_path / "protocol.json"

    _write_jsonl(
        preds_path,
        [{"case_id": "c1", "answer": "ok"}, {"case_id": "c2", "answer": "ok"}],
    )
    _write_jsonl(labels_path, [{"case_id": "c1"}])
    _write_protocol(
        protocol_path,
        predictions_sha=compute_jsonl_sha256(preds_path),
        labels_sha=compute_jsonl_sha256(labels_path),
    )

    with pytest.raises(ValueError, match="extra"):
        score_sealed_predictions(
            predictions_path=preds_path,
            labels_path=labels_path,
            protocol_path=protocol_path,
            output_path=tmp_path / "out.json",
        )


def test_scorer_does_not_modify_predictions(tmp_path: Path) -> None:
    """The predictions file SHA256 must be unchanged after scoring."""
    fixture = _make_perfect_fixture(tmp_path)
    preds_path = fixture["predictions"]
    before = hashlib.sha256(preds_path.read_bytes()).hexdigest()

    score_sealed_predictions(
        predictions_path=preds_path,
        labels_path=fixture["labels"],
        protocol_path=fixture["protocol"],
        output_path=tmp_path / "report.json",
    )

    after = hashlib.sha256(preds_path.read_bytes()).hexdigest()
    assert before == after


def test_scorer_is_deterministic(tmp_path: Path) -> None:
    """Same inputs must produce byte-identical output across runs."""
    fixture = _make_perfect_fixture(tmp_path)
    out1 = tmp_path / "report1.json"
    out2 = tmp_path / "report2.json"

    score_sealed_predictions(
        predictions_path=fixture["predictions"],
        labels_path=fixture["labels"],
        protocol_path=fixture["protocol"],
        output_path=out1,
    )
    score_sealed_predictions(
        predictions_path=fixture["predictions"],
        labels_path=fixture["labels"],
        protocol_path=fixture["protocol"],
        output_path=out2,
    )

    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_scorer_does_not_call_rag() -> None:
    """sealed_scorer.py source must not import or reference the RAG engine."""
    source = Path(inspect.getfile(score_sealed_predictions)).read_text(
        encoding="utf-8"
    )
    assert "rag_engine" not in source
    assert "RAGEngine" not in source
