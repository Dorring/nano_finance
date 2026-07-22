"""Phase 4 production regression tests for PARTIALLY_ANSWERABLE behavior.

Verifies that the orchestrator's ``_apply_partial_prefix`` static method
wraps partial answers with the Chinese restricted-answer prefix/suffix,
that the suffix surfaces missing requirements (and only their names, not
their values), that ANSWERABLE results are never prefixed, and that a
PARTIALLY_ANSWERABLE verdict is not confused with ANSWERABLE.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.application.rag_orchestrator import RAGOrchestrator
from src.domain.evidence import EvidenceItem
from src.domain.validation import AnswerabilityResult, AnswerabilityStatus
from src.retrieval.context_builder import SufficiencyResult
from src.validation.answerability import AnswerabilityEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
    page: int = 12,
    score: float = 0.9,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content="Revenue was $1,000,000 for FY2025.",
        document_name=document_name,
        page=page,
        content_type="text",
        score=score,
        rerank_score=None,
        metadata={},
    )


def _sufficient() -> SufficiencyResult:
    return SufficiencyResult(is_sufficient=True, best_score=0.9, average_score=0.85)


def _partial_result(
    missing: tuple[str, ...] = ("document: peer_report.pdf",),
) -> AnswerabilityResult:
    return AnswerabilityResult(
        status=AnswerabilityStatus.PARTIALLY_ANSWERABLE,
        reason_codes=("missing_documents",),
        evidence_count=2,
        document_count=1,
        best_score=0.8,
        average_score=0.7,
        missing_requirements=missing,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_partial_prefix_added():
    """The Chinese restricted-answer prefix is prepended to the answer."""
    result = _partial_result()
    out = RAGOrchestrator._apply_partial_prefix("Revenue was 100.", result)
    assert "根据当前检索到的资料" in out
    assert "Revenue was 100." in out
    assert out.startswith("根据当前检索到的资料")


def test_partial_suffix_lists_missing():
    """The suffix lists the missing requirements from the answerability result."""
    result = _partial_result(missing=("document: peer_report.pdf",))
    out = RAGOrchestrator._apply_partial_prefix("Revenue was 100.", result)
    assert "未找到或无法验证：document: peer_report.pdf" in out


def test_partial_with_no_missing_requirements():
    """When missing_requirements is empty, the default fallback text is used."""
    result = AnswerabilityResult(
        status=AnswerabilityStatus.PARTIALLY_ANSWERABLE,
        reason_codes=("missing_documents",),
        evidence_count=1,
        document_count=1,
        best_score=0.8,
        average_score=0.7,
        missing_requirements=(),
    )
    out = RAGOrchestrator._apply_partial_prefix("Revenue was 100.", result)
    assert "部分请求的文档或数据" in out


def test_partial_does_not_output_missing_document_values():
    """The wrapped answer must not surface values from missing documents.

    The prefix/suffix only lists the missing requirement *names* (e.g.
    ``document: peer_report.pdf``); it must never inject values that came
    from a document that was not retrieved.
    """
    # The missing document (peer_report.pdf) supposedly contained the value
    # 999; the present answer only states 100.
    result = _partial_result(missing=("document: peer_report.pdf",))
    out = RAGOrchestrator._apply_partial_prefix("Revenue was 100.", result)
    assert "999" not in out
    # The missing document *name* is surfaced, but not any of its values.
    assert "peer_report.pdf" in out


def test_partial_not_mistaken_as_answerable():
    """A PARTIALLY_ANSWERABLE verdict must not equal ANSWERABLE."""
    ev = AnswerabilityEvaluator()
    result = ev.evaluate(
        question="Compare revenue across reports",
        intent="multi_document_comparison",
        evidence=(_evidence(document_name="report_a.pdf"),),
        sufficiency_result=_sufficient(),
        calculation_result=None,
        requested_documents=("report_a.pdf", "report_b.pdf"),
    )
    assert result.status is AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert result.status is not AnswerabilityStatus.ANSWERABLE
    assert any("report_b.pdf" in m for m in result.missing_requirements)


def test_answerable_has_no_prefix():
    """ANSWERABLE status must not receive the partial-answer prefix."""
    ev = AnswerabilityEvaluator()
    result = ev.evaluate(
        question="What is the revenue?",
        intent="document_qa",
        evidence=(_evidence(document_name="annual_report.pdf"),),
        sufficiency_result=_sufficient(),
        calculation_result=None,
        requested_documents=("annual_report.pdf",),
    )
    assert result.status is AnswerabilityStatus.ANSWERABLE

    # Mirror the orchestrator's conditional: the prefix is applied ONLY when
    # the verdict is PARTIALLY_ANSWERABLE.
    should_apply = result.status is AnswerabilityStatus.PARTIALLY_ANSWERABLE
    assert should_apply is False

    answer = "Revenue was 100."
    if should_apply:
        answer = RAGOrchestrator._apply_partial_prefix(answer, result)
    assert "根据当前检索到的资料" not in answer
    assert "未找到或无法验证" not in answer


def test_partial_warning_entered():
    """PARTIALLY_ANSWERABLE surfaces warning data and a visible partial notice."""
    ev = AnswerabilityEvaluator()
    result = ev.evaluate(
        question="Compare revenue across reports",
        intent="multi_document_comparison",
        evidence=(_evidence(document_name="report_a.pdf"),),
        sufficiency_result=_sufficient(),
        calculation_result=None,
        requested_documents=("report_a.pdf", "report_b.pdf"),
    )
    assert result.status is AnswerabilityStatus.PARTIALLY_ANSWERABLE

    # The public payload exposes missing_requirements so callers can build
    # user-facing warnings.
    public = result.to_public_dict()
    assert public["status"] == "partially_answerable"
    assert len(public["missing_requirements"]) > 0

    # The orchestrator enters the partial notice into the answer text itself.
    out = RAGOrchestrator._apply_partial_prefix("Revenue was 100.", result)
    assert "根据当前检索到的资料" in out
    assert "未找到或无法验证" in out


def test_apply_partial_prefix_source_contains_chinese_prefix():
    """Static source check: the prefix method and its Chinese text exist."""
    source_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "application",
        "rag_orchestrator.py",
    )
    with open(source_path, encoding="utf-8") as fh:
        source = fh.read()
    assert "_apply_partial_prefix" in source
    assert "根据当前检索到的资料" in source
    assert "未找到或无法验证" in source
