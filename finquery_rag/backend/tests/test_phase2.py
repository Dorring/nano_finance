"""Phase 2 tests: Retrieval quality - context builder and tracing."""
import os
import sys
import time
import tempfile
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.rag_engine import RAGEngine
from services.trace import TraceLogger


class MockLLMClient:
    """Mock LLM client for testing RAGEngine without real model."""
    def __init__(self):
        self.chat = self

    def completions_create(self, **kwargs):
        class MockResponse:
            choices = [type("Choice", (), {"message": type("Msg", (), {"content": "mock answer"})()})()]
        return MockResponse()

    def create(self, **kwargs):
        return self.completions_create(**kwargs)


def make_engine(**kwargs):
    """Create RAGEngine with mock client and temp BM25 DB."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = RAGEngine(
        llm_client=MockLLMClient(),
        bm25_db_path=tmp.name,
        **kwargs,
    )
    return engine, tmp.name


def cleanup(path):
    """Clean up temp file with Windows retry."""
    import gc as _gc
    _gc.collect()
    for _ in range(3):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.05)
    try:
        os.unlink(path)
    except PermissionError:
        pass


def make_chunk(doc_id, content, score=0.9, chunk_type="text", page=1, table_num=None):
    """Helper to create a chunk dict matching vector store output format."""
    meta = {"type": chunk_type, "page": page}
    if table_num is not None:
        meta["table_num"] = table_num
    return {"doc_id": doc_id, "content": content, "metadata": meta, "score": score}


class TestContextBuilderDedup:
    def test_duplicate_chunks_removed(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("doc1::0", "This is the first chunk with enough content for dedup testing. " * 3)
            c2 = make_chunk("doc1::1", "This is the first chunk with enough content for dedup testing. " * 3)
            c3 = make_chunk("doc1::2", "This is a different chunk entirely. " * 3)
            context, sources = engine.build_context([c1, c2, c3])
            assert len(sources) == 2, "Should dedup to 2 unique chunks"
        finally:
            cleanup(path)

    def test_no_dedup_when_unique(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("doc1::0", "Alpha content " * 10)
            c2 = make_chunk("doc1::1", "Beta content " * 10)
            c3 = make_chunk("doc1::2", "Gamma content " * 10)
            context, sources = engine.build_context([c1, c2, c3])
            assert len(sources) == 3
        finally:
            cleanup(path)

    def test_empty_chunks_list(self):
        engine, path = make_engine()
        try:
            context, sources = engine.build_context([])
            assert context == ""
            assert sources == []
        finally:
            cleanup(path)


class TestContextBuilderScoreThreshold:
    def test_below_threshold_filtered(self):
        engine, path = make_engine()
        try:
            engine.min_score_threshold = 0.5
            c1 = make_chunk("doc1::0", "High score chunk " * 10, score=0.9)
            c2 = make_chunk("doc1::1", "Low score chunk " * 10, score=0.2)
            c3 = make_chunk("doc1::2", "Medium score chunk " * 10, score=0.6)
            context, sources = engine.build_context([c1, c2, c3])
            assert len(sources) == 2, "Low score chunk should be filtered"
            scores = [s["score"] for s in sources]
            assert all(s >= 0.5 for s in scores)
        finally:
            cleanup(path)

    def test_threshold_zero_keeps_all(self):
        engine, path = make_engine()
        try:
            engine.min_score_threshold = 0.0
            c1 = make_chunk("doc1::0", "Any content " * 10, score=0.01)
            c2 = make_chunk("doc1::1", "More content " * 10, score=0.0)
            context, sources = engine.build_context([c1, c2])
            assert len(sources) == 2
        finally:
            cleanup(path)

    def test_all_filtered_returns_empty(self):
        engine, path = make_engine()
        try:
            engine.min_score_threshold = 0.99
            c1 = make_chunk("doc1::0", "Low content " * 10, score=0.1)
            context, sources = engine.build_context([c1])
            assert context == ""
            assert sources == []
        finally:
            cleanup(path)


class TestContextBuilderFilenameParsing:
    def test_scoped_chunk_id_with_user_prefix(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("user_5_report.pdf::chunk_0", "Financial report content " * 10, page=3)
            context, sources = engine.build_context([c1])
            assert sources[0]["filename"] == "report.pdf"
            assert sources[0]["page"] == 3
        finally:
            cleanup(path)

    def test_scoped_chunk_id_without_user_prefix(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("report.pdf::chunk_0", "Financial report content " * 10, page=5)
            context, sources = engine.build_context([c1])
            assert sources[0]["filename"] == "report.pdf"
        finally:
            cleanup(path)

    def test_plain_doc_id(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("simple_doc", "Simple content " * 10, page=1)
            context, sources = engine.build_context([c1])
            assert sources[0]["filename"] == "simple_doc"
        finally:
            cleanup(path)

    def test_table_source_format(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("user_3_data.xlsx::t0", "Table data " * 10,
                           chunk_type="table", page=2, table_num="3.1")
            context, sources = engine.build_context([c1])
            assert sources[0]["filename"] == "data.xlsx"
            assert sources[0]["type"] == "table"
            assert "T3.1" in context
        finally:
            cleanup(path)


class TestContextBuilderSourceProvenance:
    def test_sources_include_exact_chunk_id(self):
        engine, path = make_engine()
        try:
            chunk_id = "user_5_report.pdf::page_3::chunk_7"
            c1 = make_chunk(chunk_id, "Financial report content " * 10, page=3)
            context, sources = engine.build_context([c1])

            assert sources[0]["chunk_id"] == chunk_id
            assert sources[0]["filename"] == "report.pdf"
            assert "report.pdf, p3" in context
        finally:
            cleanup(path)

    def test_truncated_sources_keep_exact_chunk_id(self):
        engine, path = make_engine(max_context_tokens=320)
        try:
            chunk_id = "user_8_large.pdf::page_1::chunk_0"
            c1 = make_chunk(chunk_id, "x" * 1000, page=1)
            context, sources = engine.build_context([c1])

            assert len(sources) == 1
            assert sources[0]["chunk_id"] == chunk_id
            assert "[...]" in context
        finally:
            cleanup(path)


class TestContextBuilderTokenBudget:
    def test_truncation_when_exceeds_budget(self):
        engine, path = make_engine()
        try:
            huge_content = "x" * 5000
            c1 = make_chunk("doc1::0", huge_content, score=0.9)
            c2 = make_chunk("doc1::1", huge_content, score=0.9)
            c3 = make_chunk("doc1::2", huge_content, score=0.9)
            context, sources = engine.build_context([c1, c2, c3])
            assert len(sources) < 3 or len(context) < len(huge_content) * 3
        finally:
            cleanup(path)

    def test_small_chunks_all_fit(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("doc1::0", "Short. ", score=0.9)
            c2 = make_chunk("doc1::1", "Also short. ", score=0.9)
            context, sources = engine.build_context([c1, c2])
            assert len(sources) == 2
        finally:
            cleanup(path)


class TestTraceIntegration:
    def test_trace_logger_created(self):
        engine, path = make_engine()
        try:
            assert hasattr(engine, "trace_logger")
            assert isinstance(engine.trace_logger, TraceLogger)
        finally:
            cleanup(path)

    def test_trace_sample_rate(self):
        engine, path = make_engine()
        try:
            assert engine.trace_logger.sample_rate == 1.0
        finally:
            cleanup(path)

    def test_trace_redact_content(self):
        engine, path = make_engine()
        try:
            assert engine.trace_logger.redact_content is True
        finally:
            cleanup(path)


class TestConversationalQueryExtended:
    def test_greeting_returns_answer(self):
        engine, path = make_engine()
        try:
            result = engine._handle_conversational_query("hello")
            assert result is not None
        finally:
            cleanup(path)

    def test_real_question_returns_none(self):
        engine, path = make_engine()
        try:
            result = engine._handle_conversational_query("2024年第三季度营收是多少?")
            assert result is None
        finally:
            cleanup(path)

    def test_empty_question_returns_none(self):
        engine, path = make_engine()
        try:
            result = engine._handle_conversational_query("")
            assert result is None
        finally:
            cleanup(path)
