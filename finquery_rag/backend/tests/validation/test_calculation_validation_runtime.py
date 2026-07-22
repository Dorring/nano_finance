"""Phase 4 hotfix: Calculation validation runtime wiring tests.

Uses a REAL RAGOrchestrator with Fake dependencies to verify that
Calculation EXECUTED/BLOCKED/FAILED paths actually invoke answerability
and validation at runtime — not just that the source contains the right
strings.
"""

from __future__ import annotations

import os
import sys
import asyncio
import inspect
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

# ---------------------------------------------------------------------------
# Optional imports — skip entire module if core deps are missing
# ---------------------------------------------------------------------------

try:
    from src.domain.calculation import (
        CalculationOperand,
        CalculationResult,
        CalculationStatus,
    )

    _HAS_CALC = True
except ImportError:
    _HAS_CALC = False

try:
    from src.application.rag_orchestrator import RAGOrchestrator

    _HAS_ORCH = True
except ImportError:
    _HAS_ORCH = False

try:
    from src.validation.validation_pipeline import GroundedValidationPipeline

    _HAS_VP = True
except ImportError:
    _HAS_VP = False

try:
    from src.validation.response_repair import ResponseRepair

    _HAS_RR = True
except ImportError:
    _HAS_RR = False


if not (_HAS_CALC and _HAS_ORCH):
    pytest.skip(
        "Required modules (src.domain.calculation / "
        "src.application.rag_orchestrator) not available",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Domain-object construction helpers
# ---------------------------------------------------------------------------


def _make_operand(
    value: Decimal | str | None = Decimal("100"),
    evidence_chunk_id: str | None = "chunk_001",
    raw_text: str = "Revenue was 100 million in FY2024.",
    metric: str = "revenue",
):
    """Build a real CalculationOperand; fall back to SimpleNamespace."""
    try:
        return CalculationOperand(
            name=metric,
            value=value if isinstance(value, Decimal) else Decimal(str(value)),
            evidence_chunk_id=evidence_chunk_id or "",
            source_text=raw_text,
            document_name="annual_report.pdf",
            page=12,
        )
    except TypeError:
        return SimpleNamespace(
            name=metric,
            value=value if isinstance(value, Decimal) else Decimal(str(value)),
            evidence_chunk_id=evidence_chunk_id or "",
            raw_text=raw_text,
            metric=metric,
            document_name="annual_report.pdf",
            page=12,
            source_text=raw_text,
        )


def _to_public_dict_proxy(status, value, operands, formula_version, error_code=None):
    """Return a ``to_public_dict`` callable for SimpleNamespace fallbacks."""

    def _dict():
        return {
            "status": status.value if hasattr(status, "value") else str(status),
            "value": value,
            "operands": [
                {
                    "value": getattr(op, "value", None),
                    "evidence_chunk_id": getattr(op, "evidence_chunk_id", None),
                }
                for op in operands
            ],
            "formula_version": formula_version,
            "error_code": error_code,
        }

    return _dict


def _make_calc_result_executed():
    """Build a CalculationResult with status EXECUTED."""
    operand = _make_operand(value=Decimal("100"))
    kwargs = dict(
        status=CalculationStatus.EXECUTED,
        value=Decimal("100"),
        target_metric="revenue",
        operands=(operand,),
        formula_version="v1",
        error_code=None,
    )
    try:
        return CalculationResult(**kwargs)
    except TypeError:
        ns = SimpleNamespace(**kwargs)
        ns.to_public_dict = _to_public_dict_proxy(
            CalculationStatus.EXECUTED, Decimal("100"), (operand,), "v1"
        )
        ns.to_trace_dict = ns.to_public_dict
        return ns


def _make_calc_result_blocked():
    """Build a CalculationResult with status BLOCKED."""
    kwargs = dict(
        status=CalculationStatus.BLOCKED,
        value=None,
        target_metric="revenue",
        operands=(),
        formula_version="v1",
        error_code="INSUFFICIENT_DATA",
    )
    try:
        return CalculationResult(**kwargs)
    except TypeError:
        ns = SimpleNamespace(**kwargs)
        ns.to_public_dict = _to_public_dict_proxy(
            CalculationStatus.BLOCKED, None, (), "v1", "INSUFFICIENT_DATA"
        )
        ns.to_trace_dict = ns.to_public_dict
        return ns


def _make_calc_result_failed():
    """Build a CalculationResult with status FAILED."""
    kwargs = dict(
        status=CalculationStatus.FAILED,
        value=None,
        target_metric="revenue",
        operands=(),
        formula_version="v1",
        error_code="CALCULATION_ERROR",
    )
    try:
        return CalculationResult(**kwargs)
    except TypeError:
        ns = SimpleNamespace(**kwargs)
        ns.to_public_dict = _to_public_dict_proxy(
            CalculationStatus.FAILED, None, (), "v1", "CALCULATION_ERROR"
        )
        ns.to_trace_dict = ns.to_public_dict
        return ns


# ---------------------------------------------------------------------------
# Orchestrator builder
# ---------------------------------------------------------------------------


def _make_orchestrator(calc_result, llm_should_fail=True):
    """Build a REAL RAGOrchestrator with fake dependencies.

    Sets up mocks to support both the attribute-injection API (described
    in the task spec) and the constructor-injection API (used by the
    existing codebase).  All known method names are mocked so the
    orchestrator can invoke whichever ones it needs.
    """
    # --- Query processor ---
    query_processor = MagicMock()
    query_context = SimpleNamespace(
        query_text="What was the revenue in FY2024?",
        tenant_id=1,
        user_id=1,
        rewritten_query="revenue FY2024",
        is_title_query=False,
    )
    query_processor.process.return_value = query_context
    query_processor.is_title_query.return_value = False
    query_processor.should_generate_with_low_confidence.return_value = False

    # --- Retrieval pipeline ---
    retrieval_pipeline = MagicMock()
    retrieval_pipeline.retrieve.return_value = (
        SimpleNamespace(
            id="chunk_001",
            chunk_id="chunk_001",
            doc_id="chunk_001",
            content="Revenue was 100 million in FY2024.",
            text="Revenue was 100 million in FY2024.",
            metadata={"document_name": "report.pdf", "page": 12},
            score=0.95,
        ),
    )
    retrieval_pipeline.retrieve_single.return_value = [
        {
            "doc_id": "chunk_001",
            "content": "Revenue was 100 million in FY2024.",
            "metadata": {"document_name": "report.pdf", "page": 12},
            "score": 0.95,
        }
    ]
    retrieval_pipeline.retrieve_multiple.return_value = (
        retrieval_pipeline.retrieve_single.return_value
    )
    retrieval_pipeline.last_retrieval_debug = {}

    # --- Context builder ---
    context_builder = MagicMock()
    context_builder.build.return_value = (
        "Revenue was 100 million in FY2024.",
        [{"filename": "report.pdf", "page": 12, "chunk_id": "chunk_001"}],
    )

    # --- Sufficiency evaluator ---
    sufficiency_evaluator = MagicMock()
    try:
        from src.retrieval.context_builder import SufficiencyResult

        sufficiency_evaluator.evaluate.return_value = SufficiencyResult(
            is_sufficient=True, best_score=0.95, average_score=0.90
        )
    except ImportError:
        sufficiency_evaluator.evaluate.return_value = SimpleNamespace(
            is_sufficient=True, best_score=0.95, average_score=0.90
        )
    sufficiency_evaluator.confidence.return_value = 0.95

    # --- LLM gateway ---
    llm_gateway = MagicMock()
    llm_gateway.rewrite_query = AsyncMock(return_value="revenue question")
    if llm_should_fail:
        llm_gateway.generate = AsyncMock(
            side_effect=AssertionError("LLM should not be called")
        )
    else:
        llm_gateway.generate = AsyncMock(
            return_value="Revenue was 100 million in FY2024."
        )

    # --- Deterministic extractor ---
    deterministic_extractor = MagicMock()
    deterministic_extractor.answer_front_matter_query.return_value = None
    deterministic_extractor.answer_deterministic_query_from_context.return_value = None

    # --- Trace logger ---
    trace_logger = MagicMock()
    trace_logger.log.return_value = "trace-001"

    # --- Intent classifier ---
    # Support both .classify() -> str and callable -> dict APIs
    intent_classifier = MagicMock()
    intent_classifier.classify.return_value = "financial_calculation"
    intent_classifier.return_value = {
        "intent": "financial_calculation",
        "requires_retrieval": True,
        "confidence": 0.9,
    }

    # --- Document helpers ---
    list_all_documents = MagicMock(return_value=[{"name": "report.pdf"}])
    get_front_matter = MagicMock(return_value=[])

    # --- Calculation pipeline ---
    calc_pipeline = MagicMock()
    calc_pipeline.execute.return_value = calc_result
    calc_pipeline.try_calculate.return_value = calc_result

    # --- Validation pipeline ---
    validation_pipeline = None
    if _HAS_VP:
        try:
            validation_pipeline = GroundedValidationPipeline()
        except TypeError:
            validation_pipeline = MagicMock(spec=GroundedValidationPipeline)
    else:
        validation_pipeline = MagicMock()

    # --- Construct orchestrator ---
    constructor_kwargs = dict(
        query_processor=query_processor,
        retrieval_pipeline=retrieval_pipeline,
        context_builder=context_builder,
        sufficiency_evaluator=sufficiency_evaluator,
        llm_gateway=llm_gateway,
        deterministic_extractor=deterministic_extractor,
        trace_logger=trace_logger,
        intent_classifier=intent_classifier,
        list_all_documents_fn=list_all_documents,
        get_front_matter_chunks_fn=get_front_matter,
        calculation_pipeline=calc_pipeline,
        validation_pipeline=validation_pipeline,
    )

    orch = None
    # Try full constructor
    try:
        orch = RAGOrchestrator(**constructor_kwargs)
    except TypeError:
        # Progressively remove kwargs that might not exist
        for key_to_remove in (
            "deterministic_extractor",
            "list_all_documents_fn",
            "get_front_matter_chunks_fn",
            "sufficiency_evaluator",
            "validation_pipeline",
            "calculation_pipeline",
        ):
            kw = {k: v for k, v in constructor_kwargs.items() if k != key_to_remove}
            try:
                orch = RAGOrchestrator(**kw)
                break
            except TypeError:
                continue
        if orch is None:
            try:
                orch = RAGOrchestrator()
            except TypeError:
                orch = RAGOrchestrator.__new__(RAGOrchestrator)

    # Always set attributes (for attribute-injection API)
    for attr, mock in [
        ("_query_processor", query_processor),
        ("_retrieval_pipeline", retrieval_pipeline),
        ("_context_builder", context_builder),
        ("_calculation_pipeline", calc_pipeline),
        ("_validation_pipeline", validation_pipeline),
        ("_llm_gateway", llm_gateway),
        ("_trace_logger", trace_logger),
        ("_intent_classifier", intent_classifier),
        ("_sufficiency_evaluator", sufficiency_evaluator),
    ]:
        try:
            setattr(orch, attr, mock)
        except (AttributeError, TypeError):
            pass

    # Set _response_repair if missing
    if _HAS_RR:
        try:
            current = getattr(orch, "_response_repair", None)
            if current is None:
                orch._response_repair = ResponseRepair()
        except (AttributeError, TypeError):
            pass

    return orch, llm_gateway, validation_pipeline


# ---------------------------------------------------------------------------
# Answer invocation helper
# ---------------------------------------------------------------------------


def _call_answer(orch, query="What was the revenue in FY2024?"):
    """Call orchestrator.answer() handling both API styles.

    Tries keyword-arg API (query=, tenant_id=, user_id=) first, then
    falls back to request-object API.
    """
    # Try keyword-arg API (task description)
    try:
        sig = inspect.signature(orch.answer)
        params = sig.parameters
        if "query" in params and "tenant_id" in params:
            return asyncio.run(orch.answer(query=query, tenant_id=1, user_id=1))
    except (ValueError, TypeError):
        pass

    # Fall back to request-object API (existing codebase pattern)
    request = SimpleNamespace(
        question=query,
        query=query,
        text=query,
        tenant_id=1,
        user_id=1,
        conversation_history=[],
        document_ids=None,
        document_names=None,
        filters=None,
        memory_profile=None,
    )
    return asyncio.run(orch.answer(request))


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_status(obj):
    """Extract a string status from a dict, object, or enum value."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        status = obj.get("status")
    else:
        status = getattr(obj, "status", None)
    if status is None:
        return None
    if hasattr(status, "value"):
        return status.value
    return str(status)


def _to_dict(obj):
    """Best-effort conversion of obj to a dict for key-presence checks."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_public_dict"):
        try:
            d = obj.to_public_dict()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_executed_runs_answerability_and_validation():
    """EXECUTED calculation must run answerability + validation."""
    calc_result = _make_calc_result_executed()
    orch, llm_gateway, validation_pipeline = _make_orchestrator(calc_result)

    # Track evaluate_answerability calls on real pipeline
    ea_called = [False]
    original_ea = getattr(validation_pipeline, "evaluate_answerability", None)
    if original_ea is not None and not isinstance(validation_pipeline, MagicMock):

        def _tracking_ea(*args, **kwargs):
            ea_called[0] = True
            return original_ea(*args, **kwargs)

        validation_pipeline.evaluate_answerability = _tracking_ea

    result = _call_answer(orch)

    # answerability was called
    if isinstance(validation_pipeline, MagicMock):
        validation_pipeline.evaluate_answerability.assert_called()
    else:
        assert ea_called[0], "evaluate_answerability was not called"

    # validation result is not None
    assert result.validation is not None

    # LLM was not called
    llm_gateway.generate.assert_not_called()


def test_blocked_has_answerability_and_validation():
    """BLOCKED calculation must produce answerability + validation dicts."""
    calc_result = _make_calc_result_blocked()
    orch, llm_gateway, _ = _make_orchestrator(calc_result)

    result = _call_answer(orch)

    assert result.answerability is not None
    assert _get_status(result.answerability) == "calculation_blocked"
    assert result.validation is not None
    assert _get_status(result.validation) == "blocked"
    llm_gateway.generate.assert_not_called()


def test_failed_has_answerability_and_validation():
    """FAILED calculation must produce answerability + validation dicts."""
    calc_result = _make_calc_result_failed()
    orch, llm_gateway, _ = _make_orchestrator(calc_result)

    result = _call_answer(orch)

    assert result.answerability is not None
    assert _get_status(result.answerability) == "calculation_blocked"
    assert result.validation is not None
    assert _get_status(result.validation) == "failed"
    llm_gateway.generate.assert_not_called()

    # No internal error leaked in public response
    validation_dict = _to_dict(result.validation)
    assert "error_message" not in validation_dict
    answer_text = result.answer or ""
    assert "CALCULATION_ERROR" not in answer_text


def test_executed_validation_result_not_none():
    """EXECUTED: AnswerResult.validation is not None."""
    calc_result = _make_calc_result_executed()
    orch, _, _ = _make_orchestrator(calc_result)

    result = _call_answer(orch)

    assert result.validation is not None
    validation_dict = _to_dict(result.validation)
    assert "status" in validation_dict


def test_blocked_validation_result_not_none():
    """BLOCKED: AnswerResult.validation is not None."""
    calc_result = _make_calc_result_blocked()
    orch, _, _ = _make_orchestrator(calc_result)

    result = _call_answer(orch)

    assert result.validation is not None
    validation_dict = _to_dict(result.validation)
    assert "status" in validation_dict


def test_failed_validation_result_not_none():
    """FAILED: AnswerResult.validation is not None."""
    calc_result = _make_calc_result_failed()
    orch, _, _ = _make_orchestrator(calc_result)

    result = _call_answer(orch)

    assert result.validation is not None
    validation_dict = _to_dict(result.validation)
    assert "status" in validation_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
