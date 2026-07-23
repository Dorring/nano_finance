"""Phase 5 acceptance tests.

These tests verify the structural and data-level invariants required for the
Phase 5 RAG evaluation system. They check:

 1.  ``EvaluationQuery`` declares no ``expected_*`` fields (label isolation).
 2.  ``EvaluationLabel`` declares all ``expected_*`` fields.
 3.  ``EvaluationPrediction`` carries Phase 3/4 envelope fields.
 4.  The blind runner source never imports labels.
 5.  The sealed scorer source never imports the RAG engine.
 6.  The frozen protocol file exists and is valid JSON.
 7.  The protocol's ``primary_metric`` is ``macro_strict_pass_rate``.
 8.  The protocol defines exactly 10 ablation variants (A0–A9).
 9.  The protocol carries a ``held_out_run_policy`` block.
 10. The dev partition has ``questions.jsonl`` and ``labels.jsonl``.
 11. Dev questions and labels share the same ``case_id`` set.
 12. Dev questions contain no ``expected_*`` fields on disk.
 13. ``.sealed/`` is listed in the backend ``.gitignore``.
 14. Phase 5 documentation files exist under ``docs/evaluation/``.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest

from src.evaluation.blind_runner import run_blind_query
from src.evaluation.dataset_loader import load_queries_and_labels
from src.evaluation.schemas import (
    EvaluationLabel,
    EvaluationPrediction,
    EvaluationQuery,
)
from src.evaluation.sealed_scorer import score_sealed_predictions

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROTOCOL_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "protocol"
    / "phase5-evaluation-protocol.json"
)
DEV_DIR = BACKEND_DIR / "eval_data" / "phase5" / "dev"
DOCS_DIR = BACKEND_DIR / "docs" / "evaluation"


# ---------------------------------------------------------------------------
# 1-3: Schema invariants
# ---------------------------------------------------------------------------
class TestSchemaInvariants:
    def test_evaluation_query_has_no_expected_fields(self) -> None:
        """EvaluationQuery dataclass must not declare any expected_* fields."""
        field_names = {f.name for f in fields(EvaluationQuery)}
        expected_fields = {
            name for name in field_names if name.startswith("expected_")
        }
        assert expected_fields == set(), (
            f"EvaluationQuery must not declare expected_* fields, found: {expected_fields}"
        )

    def test_evaluation_label_has_all_expected_fields(self) -> None:
        """EvaluationLabel must declare every expected_* field."""
        field_names = {f.name for f in fields(EvaluationLabel)}
        required_expected = {
            "expected_sources",
            "expected_numbers",
            "expected_calculations",
            "expected_intent",
            "expected_answerability",
            "expected_validation_status",
            "expected_no_answer",
        }
        missing = required_expected - field_names
        assert missing == set(), (
            f"EvaluationLabel missing expected_* fields: {missing}"
        )

    def test_evaluation_prediction_has_phase3_4_fields(self) -> None:
        """EvaluationPrediction must carry calculations, answerability, validation, warnings."""
        field_names = {f.name for f in fields(EvaluationPrediction)}
        required_phase34 = {"calculations", "answerability", "validation", "warnings"}
        missing = required_phase34 - field_names
        assert missing == set(), (
            f"EvaluationPrediction missing Phase 3/4 fields: {missing}"
        )


# ---------------------------------------------------------------------------
# 4-5: Source-level isolation
# ---------------------------------------------------------------------------
class TestSourceIsolation:
    def test_blind_runner_does_not_import_labels(self) -> None:
        """blind_runner.py source must not import labels or EvaluationLabel."""
        source = Path(inspect.getfile(run_blind_query)).read_text(encoding="utf-8")
        assert "EvaluationLabel" not in source, (
            "blind_runner.py must not reference EvaluationLabel"
        )
        assert "load_labels" not in source, (
            "blind_runner.py must not import load_labels"
        )

    def test_sealed_scorer_does_not_import_rag_engine(self) -> None:
        """sealed_scorer.py source must not import or reference rag_engine."""
        source = Path(inspect.getfile(score_sealed_predictions)).read_text(
            encoding="utf-8"
        )
        assert "rag_engine" not in source, (
            "sealed_scorer.py must not import or reference rag_engine"
        )


# ---------------------------------------------------------------------------
# 6-9: Protocol invariants
# ---------------------------------------------------------------------------
class TestProtocol:
    @pytest.fixture(scope="class")
    def protocol(self) -> dict:
        assert PROTOCOL_PATH.is_file(), (
            f"protocol file not found: {PROTOCOL_PATH}"
        )
        data = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "protocol must be a JSON object"
        return data

    def test_protocol_file_exists_and_is_valid_json(self, protocol: dict) -> None:
        """Protocol file exists and parses as a JSON object."""
        # The fixture above asserts existence and validity.
        assert isinstance(protocol, dict)

    def test_protocol_primary_metric(self, protocol: dict) -> None:
        """Protocol primary_metric must be macro_strict_pass_rate."""
        assert protocol.get("primary_metric") == "macro_strict_pass_rate"

    def test_protocol_has_ten_ablation_variants(self, protocol: dict) -> None:
        """Protocol must define exactly 10 ablation variants A0-A9."""
        variants = protocol.get("ablation_variants", [])
        assert isinstance(variants, list)
        assert len(variants) == 10, f"expected 10 variants, got {len(variants)}"
        ids = [v.get("id") for v in variants]
        assert ids == [f"A{i}" for i in range(10)], f"variant ids: {ids}"

    def test_protocol_has_held_out_run_policy(self, protocol: dict) -> None:
        """Protocol must carry a held_out_run_policy block."""
        policy = protocol.get("held_out_run_policy")
        assert isinstance(policy, dict), "held_out_run_policy must be an object"
        assert len(policy) > 0, "held_out_run_policy must not be empty"


# ---------------------------------------------------------------------------
# 10-12: Dev dataset invariants
# ---------------------------------------------------------------------------
class TestDevDataset:
    @pytest.fixture(scope="class")
    def dev_paths(self) -> tuple[Path, Path]:
        questions = DEV_DIR / "questions.jsonl"
        labels = DEV_DIR / "labels.jsonl"
        assert questions.is_file(), f"missing dev questions: {questions}"
        assert labels.is_file(), f"missing dev labels: {labels}"
        return questions, labels

    def test_dev_dataset_files_exist(self, dev_paths: tuple[Path, Path]) -> None:
        """Dev partition must have questions.jsonl and labels.jsonl."""
        questions, labels = dev_paths
        assert questions.is_file()
        assert labels.is_file()

    def test_dev_questions_and_labels_have_matching_case_ids(
        self, dev_paths: tuple[Path, Path]
    ) -> None:
        """Dev questions and labels must share the same case_id set."""
        questions, labels = dev_paths
        queries, label_objs = load_queries_and_labels(questions, labels)
        q_ids = {q.case_id for q in queries}
        l_ids = {label.case_id for label in label_objs}
        assert q_ids == l_ids, (
            f"case_id mismatch: only-in-questions={q_ids - l_ids}, "
            f"only-in-labels={l_ids - q_ids}"
        )

    def test_dev_questions_contain_no_expected_fields(
        self, dev_paths: tuple[Path, Path]
    ) -> None:
        """On-disk dev questions must not contain any expected_* keys."""
        questions, _ = dev_paths
        with questions.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                row = json.loads(stripped)
                leaked = [k for k in row if k.startswith("expected_")]
                assert leaked == [], (
                    f"question at line {line_no} (case_id={row.get('case_id')}) "
                    f"contains expected_* fields: {leaked}"
                )


# ---------------------------------------------------------------------------
# 13: .sealed/ gitignore
# ---------------------------------------------------------------------------
class TestSealedGitignore:
    def test_sealed_directory_is_gitignored(self) -> None:
        """The backend .gitignore must list .sealed/ so labels are never committed."""
        gitignore = (BACKEND_DIR / ".gitignore").read_text(encoding="utf-8")
        assert ".sealed/" in gitignore, (
            ".sealed/ must be listed in backend/.gitignore"
        )


# ---------------------------------------------------------------------------
# 14: Documentation
# ---------------------------------------------------------------------------
class TestDocumentation:
    def test_phase5_docs_exist(self) -> None:
        """Phase 5 documentation files must exist under docs/evaluation/."""
        assert DOCS_DIR.is_dir(), f"docs/evaluation/ not found: {DOCS_DIR}"
        expected_docs = [
            "phase5-evaluation-protocol.md",
            "phase5-ablation-plan.md",
            "phase5-calibration.md",
            "phase5-sealed-runbook.md",
            "phase5-metrics.md",
        ]
        missing = [d for d in expected_docs if not (DOCS_DIR / d).is_file()]
        assert missing == [], f"missing Phase 5 docs: {missing}"
