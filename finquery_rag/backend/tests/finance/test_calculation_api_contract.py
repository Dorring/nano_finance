"""Phase 3 API contract tests.

Verifies that the API response structure remains compatible when
calculation results are present or absent.
"""
import os, sys
from decimal import Decimal
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.answer import AnswerResult, AnswerPath
from src.domain.calculation import (
    CalculationOperand, CalculationOperation, CalculationResult, CalculationStatus,
)


def _calc_result(status=CalculationStatus.EXECUTED, operation=CalculationOperation.SUM):
    return CalculationResult(
        status=status, operation=operation,
        value=Decimal("42") if status is CalculationStatus.EXECUTED else None,
        unit="base" if status is CalculationStatus.EXECUTED else None,
        formula="a + b" if status is CalculationStatus.EXECUTED else None,
        formula_version="sum.v1",
        target_metric=operation.value,
        operands=(CalculationOperand(
            name="a", value=Decimal("20"), source_text="20", evidence_chunk_id="c1",
        ),) if status is CalculationStatus.EXECUTED else (),
        error_code=None if status is CalculationStatus.EXECUTED else "SOME_ERROR",
        error_message=None if status is CalculationStatus.EXECUTED else "error",
    )


class TestAPIContractCalculations:
    def test_ordinary_question_no_calculations(self):
        r = AnswerResult(answer="hello", path=AnswerPath.FULL)
        d = r.to_legacy_dict()
        assert "calculations" not in d

    def test_calculation_success_has_calculations(self):
        calc = _calc_result().to_dict()
        r = AnswerResult(answer="42", path=AnswerPath.FULL, calculations=(calc,))
        d = r.to_legacy_dict()
        assert "calculations" in d
        assert len(d["calculations"]) == 1
        assert d["calculations"][0]["status"] == "executed"

    def test_blocked_returns_structured_status(self):
        calc = _calc_result(CalculationStatus.BLOCKED).to_dict()
        r = AnswerResult(answer="blocked", path=AnswerPath.FULL, calculations=(calc,))
        d = r.to_legacy_dict()
        assert d["calculations"][0]["status"] == "blocked"
        assert d["calculations"][0]["error_code"] == "SOME_ERROR"

    def test_failed_returns_safe_error(self):
        calc = _calc_result(CalculationStatus.FAILED).to_dict()
        r = AnswerResult(answer="safe msg", path=AnswerPath.FULL, calculations=(calc,))
        d = r.to_legacy_dict()
        assert d["calculations"][0]["status"] == "failed"
        assert d["calculations"][0]["error_code"] == "SOME_ERROR"

    # --- Points 5-9, 11 + extras (8 additional tests) ---

    def test_all_legacy_fields_present(self):
        """Point 5: original Response fields all present (FULL path)."""
        r = AnswerResult(
            answer="hello",
            path=AnswerPath.FULL,
            sources=({"doc": "a"},),
            context="ctx",
            searched_docs=("doc1",),
            confidence=0.9,
            context_sufficient=True,
            intent="numeric",
            intent_confidence=0.8,
            rewritten_question=None,
            retrieved_chunks=({"doc_id": "c1"},),
            retrieval_debug={"k": 1},
            trace_id="t-1",
        )
        d = r.to_legacy_dict()
        expected = {
            "answer", "sources", "context", "searched_docs",
            "confidence", "context_sufficient", "intent",
            "intent_confidence", "rewritten_question", "retrieved_chunks",
            "retrieval_debug", "trace_id",
        }
        assert expected.issubset(set(d.keys()))

    def test_field_types_unchanged(self):
        """Point 6: field types remain compatible with legacy consumers."""
        r = AnswerResult(
            answer="hello",
            path=AnswerPath.FULL,
            sources=({"doc": "a"},),
            confidence=0.9,
            context_sufficient=True,
            intent="numeric",
            intent_confidence=0.8,
            retrieved_chunks=({"doc_id": "c1"},),
            retrieval_debug={"k": 1},
            trace_id="t-1",
        )
        d = r.to_legacy_dict()
        assert isinstance(d["answer"], str)
        assert isinstance(d["sources"], list)
        assert isinstance(d["confidence"], (int, float))
        assert isinstance(d["context_sufficient"], bool)
        assert isinstance(d["intent"], str)
        assert isinstance(d["intent_confidence"], (int, float))
        assert isinstance(d["retrieved_chunks"], list)
        assert isinstance(d["retrieval_debug"], dict)
        assert isinstance(d["trace_id"], str)

    def test_trace_id_present(self):
        """Point 7: Trace ID still present in legacy dict."""
        r = AnswerResult(answer="hello", path=AnswerPath.FULL, trace_id="trace-abc")
        d = r.to_legacy_dict()
        assert d["trace_id"] == "trace-abc"

    def test_sources_compatible(self):
        """Point 8: Sources structure remains compatible (list of dicts)."""
        src = {"document_name": "doc1", "page": 1, "content": "text"}
        r = AnswerResult(
            answer="hello", path=AnswerPath.FULL, sources=(src,)
        )
        d = r.to_legacy_dict()
        assert isinstance(d["sources"], list)
        assert len(d["sources"]) == 1
        assert d["sources"][0]["document_name"] == "doc1"

    def test_calculations_optional_schema(self):
        """Point 9: calculations is Optional (omitted when empty).

        AnswerResult is a dataclass; to_legacy_dict omits 'calculations'
        when empty so any downstream Pydantic schema with Optional
        calculations remains valid. The dict must also be JSON-serializable.
        """
        import json
        r = AnswerResult(answer="hello", path=AnswerPath.FULL)
        d = r.to_legacy_dict()
        assert "calculations" not in d
        json.dumps(d)  # must not raise

    def test_old_frontend_ignores_calculations(self):
        """Point 11: old frontend ignoring calculations does not error.

        Both a dict without 'calculations' (old frontend) and with
        'calculations' (new frontend) must be JSON-serializable.
        """
        import json
        r1 = AnswerResult(answer="hello", path=AnswerPath.FULL)
        d1 = r1.to_legacy_dict()
        json.dumps(d1)
        assert "calculations" not in d1
        calc = _calc_result().to_dict()
        r2 = AnswerResult(
            answer="42", path=AnswerPath.FULL, calculations=(calc,)
        )
        d2 = r2.to_legacy_dict()
        json.dumps(d2)
        assert "calculations" in d2

    def test_multiple_calculations_carried(self):
        """Multiple calculation results can be carried in one response."""
        calc1 = _calc_result(CalculationStatus.EXECUTED).to_dict()
        calc2 = _calc_result(
            CalculationStatus.BLOCKED, CalculationOperation.DIFFERENCE
        ).to_dict()
        r = AnswerResult(
            answer="multi", path=AnswerPath.FULL,
            calculations=(calc1, calc2),
        )
        d = r.to_legacy_dict()
        assert len(d["calculations"]) == 2
        assert d["calculations"][0]["status"] == "executed"
        assert d["calculations"][1]["status"] == "blocked"

    def test_formula_version_and_operands_in_output(self):
        """formula_version and operands are present in the calculation dict."""
        calc = _calc_result().to_dict()
        r = AnswerResult(
            answer="42", path=AnswerPath.FULL, calculations=(calc,)
        )
        d = r.to_legacy_dict()
        c = d["calculations"][0]
        assert c["formula_version"] == "sum.v1"
        assert isinstance(c["operands"], list)
        assert len(c["operands"]) == 1
        assert c["operands"][0]["name"] == "a"
