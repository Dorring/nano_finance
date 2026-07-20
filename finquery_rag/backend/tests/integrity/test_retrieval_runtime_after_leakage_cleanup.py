"""Runtime regression tests: verify retrieval paths work after Phase 1 leakage cleanup.

These tests confirm that:
1. _apply_reranker() works with and without a reranker
2. retrieve_single_document() executes end-to-end with mock retrieval
3. Hybrid retrieval (dense + BM25 + RRF) executes end-to-end
4. No AttributeError from removed page_fallback methods
"""
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# Mock heavy optional dependencies before importing RAGEngine.
mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain_core", "langchain_core.documents",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["langchain_core.documents"].Document = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_text_splitters"].MarkdownHeaderTextSplitter = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]

from services.rag_engine import RAGEngine
from services.retrieval import rrf


class _DummyLLM:
    pass


def _make_engine(use_hybrid=False, **kwargs):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = RAGEngine(
        _DummyLLM(),
        use_hybrid=use_hybrid,
        bm25_db_path=tmp.name,
        **kwargs,
    )
    return engine, tmp.name


def _cleanup(path):
    import gc
    gc.collect()
    for _ in range(3):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.05)


def _sample_chunks(n=5):
    return [
        {
            "doc_id": f"user_1_report.pdf::page_{i}::chunk_{i}",
            "content": f"Sample content for chunk {i} with revenue $100 million.",
            "metadata": {"type": "text", "page": i, "doc_name": "report.pdf"},
            "score": 0.9 - i * 0.1,
        }
        for i in range(1, n + 1)
    ]


# --- Test 1: _apply_reranker without reranker returns top-K ---

def test_apply_reranker_no_reranker_returns_top_k():
    """When no reranker is configured, _apply_reranker returns first top_k chunks."""
    engine, path = _make_engine()
    try:
        chunks = _sample_chunks(5)
        result = engine._apply_reranker("revenue", chunks, top_k=3)
        assert len(result) == 3
        assert result[0]["doc_id"] == chunks[0]["doc_id"]
        assert result[1]["doc_id"] == chunks[1]["doc_id"]
        assert result[2]["doc_id"] == chunks[2]["doc_id"]
    finally:
        _cleanup(path)


# --- Test 2: _apply_reranker with fake reranker returns reranked results ---

def test_apply_reranker_with_reranker_returns_reranked():
    """When a reranker is configured, _apply_reranker delegates to it."""

    class FakeReranker:
        name = "fake"

        def rerank(self, query, chunks, top_k=3):
            # Reverse the order
            return list(reversed(chunks[:top_k]))

    engine, path = _make_engine()
    try:
        engine.reranker = FakeReranker()
        chunks = _sample_chunks(5)
        result = engine._apply_reranker("revenue", chunks, top_k=3)
        assert len(result) == 3
        # FakeReranker reverses, so first should be the original third
        assert result[0]["doc_id"] == chunks[2]["doc_id"]
        assert result[2]["doc_id"] == chunks[0]["doc_id"]
    finally:
        _cleanup(path)


# --- Test 3: retrieve_single_document with mock query_collection ---

def test_retrieve_single_document_dense_only_executes():
    """Dense-only retrieval path executes without AttributeError."""
    engine, path = _make_engine(use_hybrid=False)
    try:
        mock_results = _sample_chunks(3)

        with patch("services.rag_engine.query_collection", return_value=mock_results):
            result = engine.retrieve_single_document("report.pdf", "revenue", user_id=1, n_results=3)

        assert len(result) <= 3
        assert all("doc_id" in chunk for chunk in result)
        # Verify no AttributeError from removed methods
        for chunk in result:
            assert "page_fallback" not in chunk.get("metadata", {})
            assert "supporting_source_page" not in chunk.get("metadata", {})
    finally:
        _cleanup(path)


# --- Test 4: Hybrid retrieval with mock dense, BM25, and RRF ---

def test_retrieve_single_document_hybrid_executes():
    """Hybrid retrieval (dense + BM25 + RRF) path executes without AttributeError."""
    engine, path = _make_engine(use_hybrid=True)
    try:
        mock_dense = _sample_chunks(3)
        mock_sparse = [
            {
                "doc_id": "user_1_report.pdf::page_7::chunk_7",
                "content": "Cash flow from operating activities was $50 million.",
                "metadata": {"type": "text", "page": 7, "doc_name": "report.pdf"},
                "score": 0.85,
            }
        ]

        with patch("services.rag_engine.query_collection", return_value=mock_dense):
            with patch.object(engine.bm25_retriever, "search", return_value=mock_sparse):
                result = engine.retrieve_single_document("report.pdf", "revenue", user_id=1, n_results=3)

        assert len(result) <= 3
        assert all("doc_id" in chunk for chunk in result)
        # Verify no page_fallback or supporting_source_page in results
        for chunk in result:
            assert "page_fallback" not in chunk.get("metadata", {})
            assert "supporting_source_page" not in chunk.get("metadata", {})
    finally:
        _cleanup(path)


# --- Test 5: No AttributeError from removed methods ---

def test_no_attribute_error_from_removed_methods():
    """Calling _apply_reranker must not trigger AttributeError from removed page_fallback methods."""
    engine, path = _make_engine()
    try:
        chunks = _sample_chunks(3)
        # This should execute cleanly without any AttributeError
        try:
            result = engine._apply_reranker("revenue", chunks, top_k=3)
            assert len(result) == 3
        except AttributeError as e:
            pytest.fail(f"_apply_reranker raised AttributeError: {e}")
    finally:
        _cleanup(path)


# --- Test 6: RRF fusion works with mock inputs ---

def test_rrf_fusion_produces_results():
    """RRF fusion of dense and sparse results produces valid output."""
    dense = _sample_chunks(3)
    sparse = [
        {
            "doc_id": "user_1_report.pdf::page_7::chunk_7",
            "content": "Cash flow from operating activities.",
            "metadata": {"type": "text", "page": 7, "doc_name": "report.pdf"},
            "score": 0.85,
        }
    ]
    fused = rrf([dense, sparse])
    assert len(fused) > 0
    assert all("fused_score" in chunk for chunk in fused)


import pytest
