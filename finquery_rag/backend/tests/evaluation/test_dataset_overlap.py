"""Tests for the Phase 5 dataset overlap checker.

Verifies that :mod:`check_phase5_dataset_overlap` detects data leakage
between dev / calibration / sealed partitions, while high-similarity
questions only produce warnings (not errors).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_phase5_dataset_overlap as overlap  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    """Write a JSONL file with the given records (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_duplicate_case_id_detected() -> None:
    """Two partitions sharing a case_id must produce an error."""
    cases_a = [{"id": "case-001", "question": "What is revenue?"}]
    cases_b = [{"id": "case-001", "question": "What is profit?"}]
    violations = overlap.check_case_id_overlap(cases_a, cases_b, "dev", "sealed")
    assert len(violations) == 1
    assert "case-001" in violations[0]


def test_identical_question_detected() -> None:
    """Same question text in different partitions must produce an error."""
    cases_a = [{"id": "a-1", "question": "What is revenue?"}]
    cases_b = [{"id": "b-1", "question": "What is revenue?"}]
    violations = overlap.check_question_overlap(cases_a, cases_b, "dev", "sealed")
    assert any("identical" in v for v in violations), violations


def test_normalized_question_detected() -> None:
    """Same question with different whitespace/case must produce an error."""
    cases_a = [{"id": "a-1", "question": "What is Revenue?"}]
    cases_b = [{"id": "b-1", "question": "what   is   revenue?"}]
    violations = overlap.check_question_overlap(cases_a, cases_b, "dev", "sealed")
    assert any("normalized" in v for v in violations), violations


def test_high_similarity_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """High-similarity questions produce warnings, not errors.

    "apple banana cherry date" vs "apple banana cherry date elder" yields
    Jaccard = 4/5 = 0.8, which meets the threshold.
    """
    cases_a = [{"id": "a-1", "question": "apple banana cherry date"}]
    cases_b = [{"id": "b-1", "question": "apple banana cherry date elder"}]
    warnings = overlap.check_high_similarity(cases_a, cases_b, "dev", "sealed")
    assert len(warnings) > 0, warnings
    assert any("high similarity" in w for w in warnings)

    phase5 = tmp_path / "eval_data" / "phase5"
    _write_jsonl(phase5 / "dev" / "questions.jsonl", cases_a)
    _write_jsonl(phase5 / "sealed" / "questions.jsonl", cases_b)
    monkeypatch.setattr(overlap, "PHASE5_DIR", phase5)
    monkeypatch.setattr(overlap, "BACKEND_DIR", tmp_path)
    assert overlap.main() == 0


def test_no_overlap_passes() -> None:
    """Distinct questions with distinct ids must produce no violations."""
    cases_a = [{"id": "a-1", "question": "What is revenue?"}]
    cases_b = [
        {"id": "b-1", "question": "How many employees does the company have?"}
    ]
    assert overlap.check_case_id_overlap(cases_a, cases_b, "dev", "sealed") == []
    assert overlap.check_question_overlap(cases_a, cases_b, "dev", "sealed") == []


def test_empty_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When eval_data/phase5 doesn't exist, exit 0 with a message."""
    monkeypatch.setattr(overlap, "PHASE5_DIR", tmp_path / "nonexistent")
    assert overlap.main() == 0
