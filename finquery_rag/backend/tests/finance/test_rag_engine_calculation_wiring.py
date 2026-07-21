"""Phase 3 production wiring tests.

Verifies that RAGEngine injects CalculationPipeline into the orchestrator
by default and that the calculation path is exercised through the real
``engine.query()`` facade — not just through directly constructed
orchestrators.

Test groups:
- Wiring: default engine has pipeline, same instance passed, can disable.
- Behavior: calculation success/blocked bypass LLM; non-calculation calls LLM.
"""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Mock heavy imports before importing RAGEngine (mirrors test_phase0.py).
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain", "langchain_core", "langchain_core.documents",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine  # noqa: E402
from src.finance.calculation_pipeline import CalculationPipeline  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _chunk(content, chunk_id="chunk_001", document_name="annual_report.pdf", page=12):
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id,
        "content": content,
        "document_name": document_name,
        "page": page,
        "metadata": {"document_name": document_name, "page": page},
        "score": 0.95,
    }


def _create_engine(tmp_path, enable_calc=True):
    """Create a real RAGEngine with mocked LLM client and temp DBs."""
    llm_client = MagicMock()
    engine = RAGEngine(
        llm_client,
        model_name="test-model",
        use_hybrid=False,
        bm25_db_path=str(tmp_path / "bm25.db"),
        trace_db_path=str(tmp_path / "trace.db"),
        reranker_name=None,
        enable_calculation_pipeline=enable_calc,
    )
    return engine


def _wire_orchestrator_mocks(engine, *, chunks, intent, llm_answer="LLM answer"):
    """Mock orchestrator boundary dependencies for controlled testing.

    The CalculationPipeline and all finance modules run for real; only
    retrieval/LLM/trace boundaries are mocked so the test is deterministic.
    """
    orch = engine._orchestrator

    # Mock retrieval boundary.
    orch._retrieval_pipeline.retrieve_single = MagicMock(return_value=chunks)
    orch._retrieval_pipeline.retrieve_multiple = AsyncMock(return_value=chunks)
    # last_retrieval_debug is a read-only property; patch the private attr.
    orch._retrieval_pipeline._last_retrieval_debug = {}

    # Mock context builder.
    sources = [{"doc_id": c.get("doc_id", "chunk_001"), "page": c.get("page", 12)} for c in chunks]
    orch._context_builder.build = MagicMock(return_value=("context text", sources))

    # Mock sufficiency.
    orch._sufficiency_evaluator.evaluate = MagicMock(
        return_value=MagicMock(is_sufficient=True)
    )
    orch._sufficiency_evaluator.confidence = MagicMock(return_value=0.9)

    # Mock deterministic extractors (return None to force calculation/LLM path).
    orch._deterministic_extractor.answer_front_matter_query = MagicMock(return_value=None)
    orch._deterministic_extractor.answer_deterministic_query_from_context = MagicMock(
        return_value=None
    )

    # Mock LLM boundary.
    orch._llm_gateway.generate = AsyncMock(return_value=llm_answer)
    orch._llm_gateway.rewrite_query = AsyncMock(return_value="rewritten question")

    # Mock trace logger.
    orch._trace_logger.log = MagicMock(return_value="trace_wiring_001")

    # Mock intent classifier.
    orch._classify_intent = MagicMock(return_value=intent)

    # Mock document listing.
    orch._list_all_documents = MagicMock(return_value=[{"name": "annual_report.pdf"}])
    orch._get_front_matter_chunks = MagicMock(return_value=[])

    # Mock query processor.
    orch._query_processor.is_title_query = MagicMock(return_value=False)
    orch._query_processor.should_generate_with_low_confidence = MagicMock(return_value=False)

    return orch._llm_gateway


class TestRAGEngineCalculationWiring:
    """Verify RAGEngine wires CalculationPipeline into the orchestrator."""

    def test_default_engine_has_calculation_pipeline(self, tmp_path):
        """Point 1: Default RAGEngine has CalculationPipeline."""
        engine = _create_engine(tmp_path)
        assert engine._calculation_pipeline is not None
        assert isinstance(engine._calculation_pipeline, CalculationPipeline)

    def test_orchestrator_receives_same_pipeline(self, tmp_path):
        """Point 2: Orchestrator receives the same Pipeline instance."""
        engine = _create_engine(tmp_path)
        assert engine._orchestrator._calculation_pipeline is engine._calculation_pipeline

    def test_disable_calculation_pipeline(self, tmp_path):
        """Point 3: enable_calculation_pipeline=False disables it."""
        engine = _create_engine(tmp_path, enable_calc=False)
        assert engine._calculation_pipeline is None
        assert engine._orchestrator._calculation_pipeline is None

    def test_calculation_success_bypasses_llm(self, tmp_path):
        """Point 4: Default engine.query() calculation success doesn't call LLM."""
        engine = _create_engine(tmp_path)
        chunks = [
            _chunk("Total revenue for FY2025 was $1,000,000."),
            _chunk("Total cost of goods sold for FY2025 was $600,000."),
        ]
        llm_gw = _wire_orchestrator_mocks(
            engine,
            chunks=chunks,
            intent={"intent": "financial_calculation", "requires_retrieval": True, "confidence": 0.9},
        )
        result = _run(engine.query(
            question="Calculate the gross margin.",
            doc_names=["annual_report.pdf"],
            user_id=1,
        ))
        assert result["answer"]
        assert "calculations" in result
        assert result["calculations"][0]["status"] == "executed"
        # LLM must NOT be called for successful calculations.
        llm_gw.generate.assert_not_called()

    def test_calculation_blocked_bypasses_llm(self, tmp_path):
        """Point 5: Default engine.query() calculation blocked doesn't call LLM."""
        engine = _create_engine(tmp_path)
        # Chunks without relevant financial numbers -> extraction fails -> BLOCKED.
        chunks = [_chunk("Some general text about the company history.")]
        llm_gw = _wire_orchestrator_mocks(
            engine,
            chunks=chunks,
            intent={"intent": "financial_calculation", "requires_retrieval": True, "confidence": 0.9},
        )
        result = _run(engine.query(
            question="Calculate the gross margin.",
            doc_names=["annual_report.pdf"],
            user_id=1,
        ))
        assert "calculations" in result
        assert result["calculations"][0]["status"] == "blocked"
        # LLM must NOT be called for blocked calculations.
        llm_gw.generate.assert_not_called()

    def test_non_calculation_calls_llm(self, tmp_path):
        """Point 6: Default engine.query() non-calculation question calls LLM."""
        engine = _create_engine(tmp_path)
        chunks = [_chunk("The company manufactures electronic components.")]
        llm_gw = _wire_orchestrator_mocks(
            engine,
            chunks=chunks,
            intent={"intent": "document_qa", "requires_retrieval": True, "confidence": 0.9},
        )
        result = _run(engine.query(
            question="What is the main business of the company?",
            doc_names=["annual_report.pdf"],
            user_id=1,
        ))
        assert result["answer"] == "LLM answer"
        # Non-calculation queries must NOT carry calculations.
        assert "calculations" not in result
        # LLM must be called for non-calculation queries.
        llm_gw.generate.assert_called_once()
