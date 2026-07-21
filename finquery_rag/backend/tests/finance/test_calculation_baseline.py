"""Phase 3 baseline: characterize current financial calculation behavior.

These tests pin down the *current* (pre-orchestrator-integration) behavior of
the RAG pipeline when handling financial calculation questions. Phase 3 will
introduce a deterministic calculation pipeline that bypasses the LLM for
solvable calculations; these baseline tests document the state so that
behavior changes are intentional and visible.

Locked behaviors:
- Intent classification routes calculation keywords to ``financial_calculation``
  and reported-metric lookups to ``document_qa``.
- ``AnswerResult`` has no ``calculations`` field; ``to_legacy_dict`` does not
  emit a ``calculations`` key.
- Deterministic financial primitives live at ``src.finance.primitive_tools``
  (migrated from ``src.services.financial_tools`` in Commit 3); the legacy
  module re-exports them for backward compatibility.
- ``RAGOrchestrator`` has no ``calculation_pipeline`` dependency yet.

When Phase 3 commits change these behaviors, update the corresponding tests
in lockstep — do NOT silently delete them.
"""

import importlib
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.answer import AnswerPath, AnswerResult
from src.services.intent import classify_query_intent


# ---------------------------------------------------------------------------
# Intent classification baseline
# ---------------------------------------------------------------------------


class TestIntentClassificationBaseline:
    """Lock current intent routing for financial calculation queries.

    Phase 3 Commit 4 (operation router) will *reuse* this intent signal
    without changing it. These tests guard the contract.
    """

    @pytest.mark.parametrize(
        "question",
        [
            # English explicit calculation verbs
            "Calculate the gross margin from revenue and COGS",
            "Compute the YoY growth rate of revenue",
            # Chinese explicit calculation verbs
            "根据收入和营业成本计算毛利率",
            "计算收入的同比增长率",
            # Chinese metric lookup WITHOUT an English document-lookup phrase.
            # The current intent module only recognizes English
            # DOCUMENT_LOOKUP_PATTERNS ("what is", "how much", ...), so
            # STRONG_CALCULATION_KEYWORDS (毛利率) wins for Chinese "是多少"
            # questions. This is a known quirk we lock here; Phase 3 router
            # must not regress it.
            "报表中显示的毛利率是多少",
        ],
    )
    def test_explicit_calculation_routed_to_financial_calculation(self, question):
        result = classify_query_intent(question)
        assert result["intent"] == "financial_calculation"
        assert result["requires_retrieval"] is True

    @pytest.mark.parametrize(
        "question",
        [
            # English "what is/was/how much" triggers DOCUMENT_LOOKUP_PATTERNS
            # → reported_metric_lookup → document_qa, even when calculation
            # keywords (margin, growth rate, percentage) are present.
            "What was the gross margin last quarter?",
            "What is the revenue growth rate shown in the report?",
            "How much was the net margin?",
            "What is the percentage share of net income in revenue?",
        ],
    )
    def test_reported_metric_lookup_routed_to_document_qa(self, question):
        """Reported-metric lookups stay document_qa (conservative routing)."""
        result = classify_query_intent(question)
        assert result["intent"] == "document_qa"
        assert result["requires_retrieval"] is True

    def test_non_calculation_finance_question_is_document_qa(self):
        result = classify_query_intent("Tell me about the revenue trend")
        assert result["intent"] == "document_qa"

    def test_conversational_question_bypasses_retrieval(self):
        result = classify_query_intent("你好")
        assert result["intent"] == "conversation"
        assert result["requires_retrieval"] is False


# ---------------------------------------------------------------------------
# AnswerResult / legacy API baseline
# ---------------------------------------------------------------------------


class TestAnswerResultBaseline:
    """Lock current AnswerResult shape: no ``calculations`` field yet.

    Phase 3 Commit 8 will add ``calculations: tuple[dict, ...] = ()`` and
    ``to_legacy_dict`` will emit ``calculations`` when non-empty. Update
    these tests in that commit.
    """

    def test_answer_result_has_no_calculations_field(self):
        result = AnswerResult(answer="test")
        assert not hasattr(result, "calculations")

    def test_legacy_dict_omits_calculations_key_full_path(self):
        result = AnswerResult(answer="test", path=AnswerPath.FULL)
        legacy = result.to_legacy_dict()
        assert "calculations" not in legacy

    def test_legacy_dict_omits_calculations_key_conversational_path(self):
        result = AnswerResult(answer="test", path=AnswerPath.CONVERSATIONAL)
        legacy = result.to_legacy_dict()
        assert "calculations" not in legacy

    def test_legacy_full_path_field_set(self):
        """Lock the exact field set for FULL path (no calculations yet)."""
        result = AnswerResult(answer="test", path=AnswerPath.FULL)
        legacy = result.to_legacy_dict()
        expected = {
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
        }
        assert set(legacy.keys()) == expected


# ---------------------------------------------------------------------------
# Financial primitives location baseline
# ---------------------------------------------------------------------------


class TestFinancialPrimitivesLocationBaseline:
    """Lock the post-migration location of deterministic financial primitives.

    Commit 3 migrated primitives from ``src.services.financial_tools`` to
    ``src.finance.primitive_tools`` and added 5 new functions. The legacy
    module re-exports everything for backward compatibility.
    """

    def test_financial_tools_shim_still_accessible_in_services(self):
        """``src.services.financial_tools`` must remain importable (shim)."""
        mod = importlib.import_module("src.services.financial_tools")
        for fn_name in (
            "growth_rate",
            "percentage_share",
            "sum_values",
            "convert_scale",
            "format_ratio_percent",
            "parse_financial_number",
            "verify_sum",
        ):
            assert hasattr(mod, fn_name), f"missing {fn_name}"

    def test_finance_package_exists_with_new_primitives(self):
        """``src.finance.primitive_tools`` is the canonical home post-migration."""
        mod = importlib.import_module("src.finance.primitive_tools")
        for fn_name in (
            "difference",
            "average_values",
            "gross_margin",
            "net_margin",
            "debt_ratio",
        ):
            assert hasattr(mod, fn_name), f"missing new primitive {fn_name}"


# ---------------------------------------------------------------------------
# Orchestrator / RAGEngine baseline
# ---------------------------------------------------------------------------


class TestOrchestratorCalculationBaseline:
    """Lock that the orchestrator has no calculation-pipeline hook yet.

    Phase 3 Commit 8 will add a ``calculation_pipeline`` constructor param
    and call ``calculation_pipeline.try_calculate(...)`` after context build.
    Update these tests in that commit.
    """

    def test_rag_orchestrator_has_no_calculation_pipeline_param(self):
        from src.application.rag_orchestrator import RAGOrchestrator

        sig = inspect.signature(RAGOrchestrator.__init__)
        assert "calculation_pipeline" not in sig.parameters

    def test_rag_orchestrator_has_no_calculation_pipeline_attribute(self):
        """Constructor must not set ``_calculation_pipeline`` before Phase 3."""
        from src.application.rag_orchestrator import RAGOrchestrator

        source = inspect.getsource(RAGOrchestrator.__init__)
        assert "_calculation_pipeline" not in source

    def test_answer_result_factory_does_not_build_calculations(self):
        """``AnswerResult`` construction must not reference calculations."""
        source = inspect.getsource(AnswerResult)
        assert "calculations" not in source
