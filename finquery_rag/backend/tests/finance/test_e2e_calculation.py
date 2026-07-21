"""End-to-end regression tests for the orchestrated financial calculation pipeline.

Phase 3 Commit 9: verifies the full RAG orchestrator integration from query
input to AnswerResult output, including LLM bypass, trace diagnostics, and
legacy API compatibility.

Test groups:
- TestE2EExecutedCalculation: each metric (gross_margin, growth_rate,
  percentage_share, debt_ratio, net_margin) exercised end-to-end.
- TestE2EBlockedCalculation: insufficient evidence -> BLOCKED, LLM bypass.
- TestE2ENotApplicableRegression: non-calculation questions still call LLM.
- TestE2ENoPipelineRegression: orchestrator without calculation_pipeline
  behaves exactly as before Phase 3.
- TestE2ETraceDiagnostics: trace_data carries calculation diagnostics.
- TestE2ELegacyAPICompatibility: to_legacy_dict emits calculations only when
  non-empty.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.application.rag_orchestrator import RAGOrchestrator
from src.domain.calculation import CalculationOperation, CalculationStatus
from src.domain.query import QueryRequest
from src.finance.calculation_pipeline import CalculationPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    content: str,
    chunk_id: str = "chunk_001",
    document_name: str = "annual_report.pdf",
    page: int = 12,
) -> dict:
    """Build a chunk dict matching the retrieval pipeline output shape."""
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id,
        "content": content,
        "document_name": document_name,
        "page": page,
        "metadata": {"document_name": document_name, "page": page},
        "score": 0.95,
    }


def _intent_calc() -> dict:
    return {
        "intent": "financial_calculation",
        "requires_retrieval": True,
        "confidence": 0.9,
    }


def _intent_doc_qa() -> dict:
    return {"intent": "document_qa", "requires_retrieval": True, "confidence": 0.9}


def _make_orchestrator(
    *,
    with_pipeline: bool = True,
    intent: dict | None = None,
    chunks: list[dict] | None = None,
):
    """Create a RAGOrchestrator with mocked dependencies.

    The orchestrator is configured so that:
    - Front-matter lookup returns nothing (forces retrieval path).
    - Retrieval returns ``chunks`` (default: revenue + COGS chunk).
    - Context sufficiency is True.
    - Deterministic extractors return None (forces calculation/LLM path).
    - LLM generate is an AsyncMock (assertable for bypass verification).
    - Trace logger captures kwargs (assertable for diagnostics).
    """
    if chunks is None:
        chunks = [
            _chunk(
                "Total revenue for FY2025 was $1,000,000. "
                "Cost of goods sold was $600,000."
            )
        ]

    retrieval_pipeline = MagicMock()
    retrieval_pipeline.retrieve_single = MagicMock(return_value=chunks)
    retrieval_pipeline.last_retrieval_debug = {}

    context_builder = MagicMock()
    context_builder.build = MagicMock(
        return_value=("context text", [{"doc_id": "chunk_001", "page": 12}])
    )

    sufficiency_evaluator = MagicMock()
    sufficiency_evaluator.evaluate = MagicMock(
        return_value=MagicMock(is_sufficient=True)
    )
    sufficiency_evaluator.confidence = MagicMock(return_value=0.9)

    llm_gateway = MagicMock()
    llm_gateway.generate = AsyncMock(return_value="LLM generated answer")

    deterministic_extractor = MagicMock()
    deterministic_extractor.answer_front_matter_query = MagicMock(return_value=None)
    deterministic_extractor.answer_deterministic_query_from_context = MagicMock(
        return_value=None
    )

    trace_logger = MagicMock()
    trace_logger.log = MagicMock(return_value="trace_e2e_001")

    intent_classifier = MagicMock(
        return_value=intent if intent is not None else _intent_calc()
    )

    list_all_documents = MagicMock(return_value=[{"name": "annual_report.pdf"}])
    get_front_matter_chunks = MagicMock(return_value=[])

    query_processor = MagicMock()
    query_processor.is_title_query = MagicMock(return_value=False)
    query_processor.should_generate_with_low_confidence = MagicMock(return_value=False)

    calculation_pipeline = CalculationPipeline() if with_pipeline else None

    orchestrator = RAGOrchestrator(
        query_processor=query_processor,
        retrieval_pipeline=retrieval_pipeline,
        context_builder=context_builder,
        sufficiency_evaluator=sufficiency_evaluator,
        llm_gateway=llm_gateway,
        deterministic_extractor=deterministic_extractor,
        trace_logger=trace_logger,
        intent_classifier=intent_classifier,
        list_all_documents_fn=list_all_documents,
        get_front_matter_chunks_fn=get_front_matter_chunks,
        calculation_pipeline=calculation_pipeline,
    )
    return orchestrator, llm_gateway, trace_logger


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _request(question: str) -> QueryRequest:
    return QueryRequest(
        question=question,
        document_names=("annual_report.pdf",),
    )


# ---------------------------------------------------------------------------
# EXECUTED calculations (end-to-end)
# ---------------------------------------------------------------------------


class TestE2EExecutedCalculation:
    """Full pipeline: question -> retrieval -> route -> extract -> execute ->
    render -> AnswerResult. LLM must be bypassed."""

    def test_gross_margin_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[
                _chunk(
                    "Total revenue for FY2025 was $1,000,000. "
                    "Cost of goods sold was $600,000."
                )
            ]
        )
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )

        assert result.calculations, "calculations must be non-empty"
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.EXECUTED.value
        assert calc["operation"] == CalculationOperation.GROSS_MARGIN.value
        assert calc["formula_version"] == "gross_margin.v1"
        assert "40.00%" in result.answer
        llm.generate.assert_not_called()

    def test_growth_rate_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[
                _chunk(
                    "Revenue for the current period FY2025 was $120 million. "
                    "Previous period FY2024 revenue was $100 million."
                )
            ]
        )
        result = _run(orch.answer(_request("Compute the YoY growth rate of revenue")))

        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.EXECUTED.value
        assert calc["operation"] == CalculationOperation.GROWTH_RATE.value
        assert "20.00%" in result.answer
        llm.generate.assert_not_called()

    def test_percentage_share_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[
                _chunk(
                    "The segment revenue was $30 million. "
                    "Total revenue was $120 million."
                )
            ]
        )
        result = _run(
            orch.answer(
                _request(
                    "Calculate the percentage share of segment revenue in total revenue"
                )
            )
        )

        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.EXECUTED.value
        assert calc["operation"] == CalculationOperation.PERCENTAGE_SHARE.value
        assert "25.00%" in result.answer
        llm.generate.assert_not_called()

    def test_debt_ratio_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[
                _chunk(
                    "Total liabilities were $400 million. "
                    "Total assets were $1,000 million."
                )
            ]
        )
        result = _run(
            orch.answer(
                _request(
                    "Calculate the debt ratio from total liabilities and total assets"
                )
            )
        )

        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.EXECUTED.value
        assert calc["operation"] == CalculationOperation.DEBT_RATIO.value
        assert "40.00%" in result.answer
        llm.generate.assert_not_called()

    def test_net_margin_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[_chunk("Revenue was $1,000 million. Net income was $150 million.")]
        )
        result = _run(
            orch.answer(
                _request("Calculate the net margin from revenue and net income")
            )
        )

        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.EXECUTED.value
        assert calc["operation"] == CalculationOperation.NET_MARGIN.value
        assert "15.00%" in result.answer
        llm.generate.assert_not_called()


# ---------------------------------------------------------------------------
# BLOCKED calculation (end-to-end)
# ---------------------------------------------------------------------------


class TestE2EBlockedCalculation:
    """Calculation requested but evidence insufficient -> BLOCKED, LLM bypass."""

    def test_blocked_missing_cogs_e2e(self):
        orch, llm, _ = _make_orchestrator(
            chunks=[_chunk("Total revenue for FY2025 was $1,000,000.")]
        )
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )

        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.BLOCKED.value
        assert calc["error_code"] == "PLAN_BLOCKED"
        # BLOCKED bypasses LLM too.
        llm.generate.assert_not_called()

    def test_blocked_no_evidence_e2e(self):
        orch, llm, _ = _make_orchestrator(chunks=[])
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )

        # No chunks -> empty evidence -> BLOCKED.
        assert result.calculations
        calc = result.calculations[0]
        assert calc["status"] == CalculationStatus.BLOCKED.value


# ---------------------------------------------------------------------------
# NOT_APPLICABLE regression (LLM must be called)
# ---------------------------------------------------------------------------


class TestE2ENotApplicableRegression:
    """Non-calculation questions must still go through the LLM."""

    def test_document_qa_question_calls_llm(self):
        orch, llm, _ = _make_orchestrator(
            intent=_intent_doc_qa(),
            chunks=[_chunk("The revenue for FY2025 was $1,000,000.")],
        )
        result = _run(orch.answer(_request("What was the revenue in FY2025?")))

        assert result.calculations == ()
        assert result.answer == "LLM generated answer"
        llm.generate.assert_called_once()

    def test_calculation_verb_but_no_metric_calls_llm(self):
        """Calculation verb but no recognized metric -> NOT_APPLICABLE -> LLM."""
        orch, llm, _ = _make_orchestrator(
            intent=_intent_calc(),
            chunks=[_chunk("Some general financial information without metrics.")],
        )
        result = _run(orch.answer(_request("计算所有内容的详细信息")))

        assert result.calculations == ()
        assert result.answer == "LLM generated answer"
        llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# No-pipeline regression (Phase 2 behavior unchanged)
# ---------------------------------------------------------------------------


class TestE2ENoPipelineRegression:
    """Orchestrator without calculation_pipeline behaves exactly as before."""

    def test_no_pipeline_llm_always_called(self):
        orch, llm, _ = _make_orchestrator(
            with_pipeline=False,
            chunks=[
                _chunk(
                    "Total revenue for FY2025 was $1,000,000. "
                    "Cost of goods sold was $600,000."
                )
            ],
        )
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )

        # Without pipeline, LLM is always called even for calculation questions.
        assert result.answer == "LLM generated answer"
        llm.generate.assert_called_once()

    def test_no_pipeline_calculations_empty(self):
        orch, llm, _ = _make_orchestrator(with_pipeline=False)
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )

        assert result.calculations == ()


# ---------------------------------------------------------------------------
# Trace diagnostics
# ---------------------------------------------------------------------------


class TestE2ETraceDiagnostics:
    """trace_data must carry calculation diagnostics when pipeline runs."""

    def test_trace_contains_calculation_diagnostics(self):
        orch, _, trace_logger = _make_orchestrator()
        _run(orch.answer(_request("Calculate the gross margin from revenue and COGS")))

        kwargs = trace_logger.log.call_args.kwargs
        diag = kwargs["diagnostics"]
        assert diag["calculation"] is not None
        assert diag["calculation"]["status"] == CalculationStatus.EXECUTED.value
        assert (
            diag["calculation"]["operation"] == CalculationOperation.GROSS_MARGIN.value
        )
        assert diag["calculation"]["formula_version"] == "gross_margin.v1"
        assert diag["calculation"]["operand_count"] == 2
        assert diag["calculation"]["error_code"] is None

    def test_trace_calculation_null_when_not_applicable(self):
        orch, _, trace_logger = _make_orchestrator(intent=_intent_doc_qa())
        _run(orch.answer(_request("What was the revenue in FY2025?")))

        kwargs = trace_logger.log.call_args.kwargs
        diag = kwargs["diagnostics"]
        # NOT_APPLICABLE -> calculation is None in trace.
        assert diag["calculation"] is None


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------


class TestE2ELegacyAPICompatibility:
    """to_legacy_dict must emit calculations only when non-empty (additive)."""

    def test_legacy_dict_has_calculations_when_executed(self):
        orch, _, _ = _make_orchestrator()
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )
        legacy = result.to_legacy_dict()

        assert "calculations" in legacy
        assert len(legacy["calculations"]) == 1
        assert legacy["calculations"][0]["status"] == CalculationStatus.EXECUTED.value

    def test_legacy_dict_no_calculations_when_not_applicable(self):
        orch, _, _ = _make_orchestrator(intent=_intent_doc_qa())
        result = _run(orch.answer(_request("What was the revenue in FY2025?")))
        legacy = result.to_legacy_dict()

        assert "calculations" not in legacy

    def test_legacy_dict_no_calculations_when_no_pipeline(self):
        orch, _, _ = _make_orchestrator(with_pipeline=False)
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )
        legacy = result.to_legacy_dict()

        assert "calculations" not in legacy

    def test_legacy_full_path_field_set_unchanged_with_calculations(self):
        """FULL path field set must be unchanged; calculations is additive."""
        orch, _, _ = _make_orchestrator()
        result = _run(
            orch.answer(_request("Calculate the gross margin from revenue and COGS"))
        )
        legacy = result.to_legacy_dict()

        # Core FULL-path fields must all be present.
        for field in (
            "answer",
            "sources",
            "context",
            "searched_docs",
            "confidence",
            "context_sufficient",
            "intent",
            "intent_confidence",
            "rewritten_question",
            "retrieved_chunks",
            "retrieval_debug",
            "trace_id",
        ):
            assert field in legacy, f"missing core field: {field}"
        # calculations is additive (not part of the original field set).
        assert "calculations" in legacy
