import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

from services.ingest import _extract_title_from_first_page
from services.rag_engine import RAGEngine


class _MockLLMClient:
    def __init__(self, response_text="Revenue was $219 million, up 22% year over year."):
        self.call_count = 0
        self.prompts = []
        text = response_text

        def _create(**kwargs):
            self.call_count += 1
            self.prompts.append(kwargs)

            class MockResponse:
                choices = [type("Choice", (), {"message": type("Msg", (), {"content": text})()})()]

            return MockResponse()

        self.chat = type("Chat", (), {
            "completions": type("Completions", (), {"create": staticmethod(_create)})()
        })()


class _FakePage:
    def __init__(self, lines, height=1000):
        self._lines = lines
        self.rect = type("Rect", (), {"height": height})()

    def get_text(self, mode, *args, **kwargs):
        assert mode == "dict"
        blocks = []
        for text, size, y0 in self._lines:
            blocks.append({
                "type": 0,
                "lines": [{
                    "spans": [{
                        "text": text,
                        "size": size,
                        "bbox": [0, y0, 100, y0 + 10],
                    }]
                }]
            })
        return {"blocks": blocks}


def _engine(client=None):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return RAGEngine(client or _MockLLMClient(), use_hybrid=False, bm25_db_path=tmp.name), tmp.name


def _cleanup(path):
    import gc

    gc.collect()
    for _ in range(3):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.05)


def _chunk(score=0.01, content="Record revenue was $219 million, up 22% year-over-year."):
    return {
        "doc_id": "user_1_FINAL Annual Report.pdf::page_3::chunk_2_1",
        "content": content,
        "metadata": {"type": "text", "page": 3, "doc_name": "FINAL Annual Report.pdf"},
        "score": score,
    }


def test_annual_report_title_keeps_cover_subtitle_lines():
    page = _FakePage([
        ("2025 Driving Smart Solutions", 18, 80),
        ("ANNUAL REPORT", 32, 140),
        ("PDF Solutions, Inc.", 9, 260),
        ("Table of contents", 8, 520),
    ])

    title = _extract_title_from_first_page(page)

    assert title == "2025 Driving Smart Solutions ANNUAL REPORT"


def test_numeric_finance_query_can_generate_from_low_rrf_score():
    engine, path = _engine()
    try:
        assert engine._should_generate_with_low_confidence(
            "What record revenue did PDF Solutions report for 2025?",
            [_chunk(score=0.01)],
        )
    finally:
        _cleanup(path)


def test_non_numeric_low_confidence_query_still_refuses_without_llm(monkeypatch):
    client = _MockLLMClient()
    engine, path = _engine(client)
    try:
        monkeypatch.setattr(engine, "retrieve_single_document", lambda *args, **kwargs: [_chunk(score=0.003)])

        result = asyncio.run(engine.query("Summarize the document strategy.", doc_names=["FINAL Annual Report.pdf"], user_id=1))

        assert result["context_sufficient"] is False
        assert "sufficiently relevant" in result["answer"]
        assert client.call_count == 0
    finally:
        _cleanup(path)


def test_numeric_low_confidence_query_calls_llm(monkeypatch):
    client = _MockLLMClient()
    engine, path = _engine(client)
    try:
        monkeypatch.setattr(engine, "retrieve_single_document", lambda *args, **kwargs: [_chunk(score=0.01)])

        result = asyncio.run(engine.query(
            "What record revenue did PDF Solutions report for 2025?",
            doc_names=["FINAL Annual Report.pdf"],
            user_id=1,
        ))

        assert result["context_sufficient"] is True
        assert "$219 million" in result["answer"]
        assert client.call_count == 1
    finally:
        _cleanup(path)
