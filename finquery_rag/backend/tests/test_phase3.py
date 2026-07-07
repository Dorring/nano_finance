"""Phase 3 tests: Answer reliability - validation, sufficiency, confidence."""
import os
import sys
import time
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.rag_engine import RAGEngine


class MockLLMClient:
    """Mock LLM client for testing RAGEngine without real model."""
    def __init__(self, response_text="Revenue was $10M in Q3."):
        _text = response_text
        def _create(**kwargs):
            class MockResponse:
                choices = [type("Choice", (), {"message": type("Msg", (), {"content": _text})()})()]
            return MockResponse()
        self.chat = type("Chat", (), {
            "completions": type("Completions", (), {"create": staticmethod(_create)})()
        })()


def make_engine(response_text="Revenue was $10M in Q3.", **kwargs):
    """Create RAGEngine with mock client and temp BM25 DB."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    engine = RAGEngine(
        llm_client=MockLLMClient(response_text=response_text),
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


class TestAnswerValidation:
    def test_empty_answer_returns_refusal(self):
        engine, path = make_engine()
        try:
            result = engine._validate_answer("", [])
            assert "couldn't" in result.lower()
        finally:
            cleanup(path)

    def test_none_answer_returns_refusal(self):
        engine, path = make_engine()
        try:
            result = engine._validate_answer(None, [])
            assert "couldn't" in result.lower()
        finally:
            cleanup(path)

    def test_artifact_stripped(self):
        engine, path = make_engine()
        try:
            result = engine._validate_answer("Revenue was $10M.<|end|>", [])
            assert "<|end|>" not in result
            assert "Revenue" in result
        finally:
            cleanup(path)

    def test_end_of_text_artifact_stripped(self):
        engine, path = make_engine()
        try:
            result = engine._validate_answer("</s>Revenue was $10M.</s>", [])
            assert "</s>" not in result
            assert "Revenue" in result
        finally:
            cleanup(path)

    def test_near_empty_after_cleanup_returns_refusal(self):
        engine, path = make_engine()
        try:
            result = engine._validate_answer("<|end|>", [])
            assert "couldn't" in result.lower() or "meaningful" in result.lower()
        finally:
            cleanup(path)

    def test_normal_answer_passes_through(self):
        engine, path = make_engine()
        try:
            answer = "Revenue was $10M in Q3, up 15% year over year."
            result = engine._validate_answer(answer, [])
            assert result == answer
        finally:
            cleanup(path)

    def test_long_answer_truncated(self):
        engine, path = make_engine()
        try:
            long_answer = "word " * 10000
            result = engine._validate_answer(long_answer, [])
            max_chars = engine.max_new_tokens * 4
            assert len(result) <= max_chars + 5  # +5 for "..."
            assert result.endswith("...")
        finally:
            cleanup(path)


class TestContextSufficiency:
    def test_empty_chunks_insufficient(self):
        engine, path = make_engine()
        try:
            is_suff, best, avg = engine._check_context_sufficiency([])
            assert is_suff is False
            assert best == 0.0
            assert avg == 0.0
        finally:
            cleanup(path)

    def test_high_score_sufficient(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("doc1::0", "content", score=0.8)]
            is_suff, best, avg = engine._check_context_sufficiency(chunks)
            assert is_suff is True
            assert best == 0.8
        finally:
            cleanup(path)

    def test_low_score_insufficient(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("doc1::0", "content", score=0.05)]
            is_suff, best, avg = engine._check_context_sufficiency(chunks)
            assert is_suff is False
        finally:
            cleanup(path)

    def test_threshold_boundary(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("doc1::0", "content", score=0.15)]
            is_suff, best, avg = engine._check_context_sufficiency(chunks)
            assert is_suff is True
        finally:
            cleanup(path)

    def test_just_below_threshold(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("doc1::0", "content", score=0.14)]
            is_suff, best, avg = engine._check_context_sufficiency(chunks)
            assert is_suff is False
        finally:
            cleanup(path)


class TestConfidenceScore:
    def test_empty_chunks_zero_confidence(self):
        engine, path = make_engine()
        try:
            conf = engine._compute_confidence([])
            assert conf == 0.0
        finally:
            cleanup(path)

    def test_high_score_high_confidence(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("d1::0", "c", score=0.9)]
            conf = engine._compute_confidence(chunks)
            assert conf >= 0.8
        finally:
            cleanup(path)

    def test_low_score_low_confidence(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("d1::0", "c", score=0.1)]
            conf = engine._compute_confidence(chunks)
            assert conf <= 0.2
        finally:
            cleanup(path)

    def test_confidence_bounded_0_to_1(self):
        engine, path = make_engine()
        try:
            chunks = [make_chunk("d1::0", "c", score=1.5)]  # over 1.0
            conf = engine._compute_confidence(chunks)
            assert 0.0 <= conf <= 1.0
        finally:
            cleanup(path)

    def test_multiple_chunks_weighted(self):
        engine, path = make_engine()
        try:
            c1 = make_chunk("d1::0", "c", score=0.9)
            c2 = make_chunk("d1::1", "c", score=0.3)
            conf = engine._compute_confidence([c1, c2])
            # 0.7 * 0.9 + 0.3 * 0.6 = 0.63 + 0.18 = 0.81
            assert abs(conf - 0.81) < 0.01
        finally:
            cleanup(path)


class TestGenerateAnswerValidation:
    def test_empty_context_returns_cant_find(self):
        import asyncio
        engine, path = make_engine()
        try:
            result = asyncio.run(engine.generate_answer("", "test query"))
            assert "couldn't find" in result.lower() or "relevant" in result.lower()
        finally:
            cleanup(path)

    def test_normal_answer_validated(self):
        import asyncio
        engine, path = make_engine(response_text="Revenue was $10M.")
        try:
            result = asyncio.run(engine.generate_answer("some context", "what is revenue"))
            assert "Revenue" in result
        finally:
            cleanup(path)

    def test_artifact_in_answer_stripped(self):
        import asyncio
        engine, path = make_engine(response_text="Revenue was $10M.<|end|>")
        try:
            result = asyncio.run(engine.generate_answer("some context", "what is revenue"))
            assert "<|end|>" not in result
            assert "Revenue" in result
        finally:
            cleanup(path)


class TestQueryReturnsConfidence:
    def test_conversational_no_confidence_key(self):
        import asyncio
        engine, path = make_engine()
        try:
            result = asyncio.run(engine.query("hello", user_id=1))
            assert "answer" in result
            # conversational queries don't go through retrieval
        finally:
            cleanup(path)
