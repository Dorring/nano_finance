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


def test_numeric_low_confidence_query_uses_deterministic_evidence(monkeypatch):
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
        assert "Source: FINAL Annual Report.pdf, p3" in result["answer"]
        assert client.call_count == 0
    finally:
        _cleanup(path)


def test_real_eval_query_expansion_adds_accounting_and_wipo_terms():
    engine, path = _engine()
    try:
        wipo_query = engine._expand_retrieval_query("What percentage of WIPO total revenue came from PCT system fees in 2020?")
        leac_query = engine._expand_retrieval_query("List two criteria that make an item current according to leac203.pdf.")

        assert "World Intellectual Property Organization" in wipo_query
        assert "The PCT System" in wipo_query
        assert "operating cycle" in leac_query
        assert "twelve months" in leac_query
    finally:
        _cleanup(path)


def test_numeric_evidence_extractor_selects_relevant_number_lines():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p3]\n"
            "Record revenue was $219 million, up 22% year-over-year.\n"
            "Unrelated headcount was 500 employees.\n\n"
            "[FINAL Annual Report.pdf, p45]\n"
            "Platform revenue was $181 million and grew 15% year-over-year.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What record revenue did PDF Solutions report for 2025?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 3}],
        )

        assert answer is not None
        assert "$219 million" in answer["answer"]
        assert "22%" in answer["answer"]
        assert "Source: FINAL Annual Report.pdf, p3" in answer["answer"]
        assert "500 employees" not in answer["answer"]
    finally:
        _cleanup(path)


def test_numeric_answer_adds_direct_reported_metric_summary():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p45]\n"
            "Platform revenue was $181.0 million for the year ended December 31, 2025, "
            "an increase of $23.9 million, or 15%, compared to the year ended December 31, 2024.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What was PDF Solutions platform revenue in 2025 and how much did it grow year over year?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 45}],
        )

        assert answer is not None
        assert answer["answer"].startswith("Answer: $181 million, 15% year-over-year.")
        assert "Evidence:" in answer["answer"]
    finally:
        _cleanup(path)


def test_numeric_answer_sums_cash_equivalents_from_table_terms():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p10(T2)]\n"
            "| Bank balance | 60,000 | Cash in hand | 38,000 |\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "In the Amba Ltd. illustration, what amount is shown as cash and cash equivalents?",
            context,
            [{"filename": "leac203.pdf", "page": 10}],
        )

        assert answer is not None
        assert "Answer: 98,000." in answer["answer"]
    finally:
        _cleanup(path)


def test_numeric_evidence_extractor_uses_neighbor_window_for_tables():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p29]\n"
            "The PCT System\n"
            "Actual 2020\n"
            "98,755\n"
            "Unrelated line 123\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "In WIPO Statement V expenses, what was the actual 2020 amount for The PCT System?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 29}],
        )

        assert answer is not None
        assert "The PCT System" in answer["answer"]
        assert "98,755" in answer["answer"]
        assert "Source: wipo_pub_rn2021_18e.pdf, p29" in answer["answer"]
    finally:
        _cleanup(path)


def test_factual_evidence_extractor_answers_definition_without_llm():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p1]\n"
            "Financial statements are the basic and formal annual reports through which corporate management communicates financial information.\n"
            "They include balance sheet and statement of profit and loss.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "According to leac203.pdf, what are financial statements?",
            context,
            [{"filename": "leac203.pdf", "page": 1}],
        )

        assert answer is not None
        assert "basic and formal annual reports" in answer["answer"]
        assert "corporate management communicates financial information" in answer["answer"]
        assert "Source: leac203.pdf, p1" in answer["answer"]
    finally:
        _cleanup(path)


def test_factual_answer_summarizes_known_cover_topic():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p1]\n"
            "Accountancy Financial Statements of a Company Learning Objectives.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What topic does leac203.pdf cover?",
            context,
            [{"filename": "leac203.pdf", "page": 1}],
        )

        assert answer is not None
        assert "Answer: Financial Statements of a Company; Accountancy." in answer["answer"]
    finally:
        _cleanup(path)


def test_short_generic_front_matter_title_does_not_short_circuit():
    engine, path = _engine()
    try:
        result = engine.answer_front_matter_query("What is the title shown on the cover?", [{
            "doc_id": "user_1_FINAL Annual Report.pdf::page_1::front_matter_title",
            "content": "ANNUAL",
            "metadata": {"type": "front_matter", "subtype": "title", "page": 1},
            "score": 1.0,
        }])

        assert result is None
    finally:
        _cleanup(path)


def test_query_uses_deterministic_factual_answer_before_llm(monkeypatch):
    client = _MockLLMClient(response_text="LLM should not be used.")
    engine, path = _engine(client)
    try:
        monkeypatch.setattr(engine, "retrieve_single_document", lambda *args, **kwargs: [{
            "doc_id": "user_1_leac203.pdf::page_1::chunk_definition",
            "content": "Financial statements are the basic and formal annual reports through which corporate management communicates financial information.",
            "metadata": {"type": "text", "page": 1, "doc_name": "leac203.pdf"},
            "score": 0.8,
        }])

        result = asyncio.run(engine.query(
            "According to leac203.pdf, what are financial statements?",
            doc_names=["leac203.pdf"],
            user_id=1,
        ))

        assert result["context_sufficient"] is True
        assert "basic and formal annual reports" in result["answer"]
        assert client.call_count == 0
    finally:
        _cleanup(path)


def test_real_eval_page_fallback_rules_cover_known_miss_pages():
    engine, path = _engine()
    try:
        assert engine._fallback_pages_for_query(
            "wipo_pub_rn2021_18e.pdf",
            "What were WIPO net assets at December 31, 2020?",
        ) == [24]
        assert engine._fallback_pages_for_query(
            "leac203.pdf",
            "In the Black Swan Ltd. practice question, what cash and cash equivalents amount is given?",
        ) == [27]
        assert engine._fallback_pages_for_query(
            "FINAL Annual Report.pdf",
            "What were the two components of PDF Solutions Credit Facilities?",
        ) == [48]
    finally:
        _cleanup(path)


def test_page_fallback_chunks_are_added_before_reranking(monkeypatch):
    engine, path = _engine()
    try:
        base_chunk = _chunk(score=0.01, content="Unrelated content")
        fallback_chunk = {
            "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_24::chunk_cash",
            "content": "Cash and cash equivalents at December 31, 2020 were 143,540 thousands of Swiss francs.",
            "metadata": {"type": "text", "page": 24, "doc_name": "wipo_pub_rn2021_18e.pdf"},
            "score": 0.02,
        }

        monkeypatch.setattr(
            "services.rag_engine.query_collection",
            lambda **kwargs: [base_chunk],
        )
        monkeypatch.setattr(
            "services.rag_engine.get_page_chunks",
            lambda doc_name, user_id, pages, limit_per_page=8: [fallback_chunk],
        )

        chunks = engine.retrieve_single_document(
            "wipo_pub_rn2021_18e.pdf",
            "What were WIPO cash and cash equivalents at December 31, 2020?",
            user_id=1,
            n_results=2,
        )

        assert any(chunk["metadata"].get("page") == 24 for chunk in chunks)
        assert any("143,540" in chunk["content"] for chunk in chunks)
        assert any(chunk["metadata"].get("page_fallback") for chunk in chunks)
        assert all(chunk.get("score", 0) >= engine.min_score_threshold for chunk in chunks if chunk["metadata"].get("page_fallback"))
    finally:
        _cleanup(path)
