import pytest
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
        "doc_id": "user_1_neutral_report.pdf::page_3::chunk_2_1",
        "content": content,
        "metadata": {"type": "text", "page": 3, "doc_name": "neutral_report.pdf"},
        "score": score,
    }


def test_annual_report_title_keeps_cover_subtitle_lines():
    page = _FakePage([
        ("2025 Driving Smart Solutions", 18, 80),
        ("ANNUAL REPORT", 32, 140),
        ("Acme Corp, Inc.", 9, 260),
        ("Table of contents", 8, 520),
    ])

    title = _extract_title_from_first_page(page)

    assert title == "2025 Driving Smart Solutions ANNUAL REPORT"


def test_numeric_finance_query_can_generate_from_low_rrf_score():
    engine, path = _engine()
    try:
        assert engine._should_generate_with_low_confidence(
            "What record revenue did Acme Corp report for 2025?",
            [_chunk(score=0.01)],
        )
    finally:
        _cleanup(path)


def test_supporting_source_pages_are_retained_in_final_sources():
    """Phase 1: _ensure_supporting_sources and supporting_source_page removed. Verify gone."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, "_ensure_supporting_sources"), (
            "_ensure_supporting_sources must be removed (Phase 1 retrieval integrity)"
        )
        chunk = {
            "doc_id": "test::page_1::c1",
            "metadata": {"page": 1, "supporting_source_page": True},
            "score": 0.5,
        }
        from src.services.rag_engine import RAGEngine
        summary = RAGEngine._summarize_retrieved_chunks([chunk])
        assert "supporting_source_page" not in summary[0]
    finally:
        _cleanup(path)



def test_numeric_evidence_extractor_selects_relevant_number_lines():
    engine, path = _engine()
    try:
        context = (
            "[neutral_report.pdf, p3]\n"
            "Record revenue was $219 million, up 22% year-over-year.\n"
            "Unrelated headcount was 500 employees.\n\n"
            "[neutral_report.pdf, p45]\n"
            "Platform revenue was $181 million and grew 15% year-over-year.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What record revenue did Acme Corp report for 2025?",
            context,
            [{"filename": "neutral_report.pdf", "page": 3}],
        )

        assert answer is not None
        assert "$219 million" in answer["answer"]
        assert "22%" in answer["answer"]
        assert "Source: neutral_report.pdf, p3" in answer["answer"]
        assert "500 employees" not in answer["answer"]
    finally:
        _cleanup(path)




def test_numeric_answer_extracts_credit_facility_components():
    engine, path = _engine()
    try:
        context = (
            "[neutral_report.pdf, p48]\n"
            "Credit Facilities consisted of a Revolving Credit Facility of $45.0 million "
            "and a Term Loan of $25.0 million.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What were the two components of Acme Corp Credit Facilities?",
            context,
            [{"filename": "neutral_report.pdf", "page": 48}],
        )

        assert answer is not None
        assert "Revolving Credit Facility, $45 million" in answer["answer"]
        assert "Term Loan, $25 million" in answer["answer"]
    finally:
        _cleanup(path)



def test_factual_answer_uses_full_context_for_wipo_title_and_reporting_period():
    engine, path = _engine()
    try:
        context = (
            "[sample_financial_report.pdf, p1]\n"
            "Annual financial report and financial statements\n"
            "Year to December 31, 2020\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What is the title and reporting period of the Global Org financial report?",
            context,
            [{"filename": "sample_financial_report.pdf", "page": 1}],
        )

        assert answer is not None
        assert "Annual financial report and financial statements" in answer["answer"]
        assert "Year to December 31, 2020" in answer["answer"]
    finally:
        _cleanup(path)





def test_factual_answer_uses_expected_current_item_wording():
    engine, path = _engine()
    try:
        context = (
            "[sample_textbook.pdf, p10]\n"
            "An item is current when it is expected to be realized in the operating cycle, "
            "realized within twelve months, held primarily for trading, or is cash and cash equivalent.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "List two criteria that make an item current according to sample_textbook.pdf.",
            context,
            [{"filename": "sample_textbook.pdf", "page": 10}],
        )

        assert answer is not None
        assert "operating cycle" in answer["answer"]
        assert "within twelve months" in answer["answer"]
        assert "held primarily for trading" in answer["answer"]
        assert "cash and cash equivalent" in answer["answer"]
    finally:
        _cleanup(path)



def test_supporting_source_pages_for_real_eval_metric_queries():
    """Phase 1: _supporting_pages_for_query was removed. Verify it no longer exists."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, '_supporting_pages_for_query'), (
            "_supporting_pages_for_query must be removed (Phase 1 retrieval integrity)"
        )
    finally:
        _cleanup(path)



def test_multi_doc_coverage_keeps_one_candidate_per_requested_document():
    engine, path = _engine()
    try:
        selected = [
            {
                "doc_id": "user_1_neutral_report.pdf::page_45::chunk_revenue",
                "content": "Platform revenue was $181.0 million.",
                "metadata": {"type": "text", "page": 45, "doc_name": "neutral_report.pdf"},
                "score": 0.9,
            },
            {
                "doc_id": "user_1_neutral_report.pdf::page_3::chunk_record",
                "content": "Record revenue was $219 million.",
                "metadata": {
                    "type": "text",
                    "page": 3,
                    "doc_name": "neutral_report.pdf",
                    "supporting_source_page": True,
                },
                "score": 0.8,
            },
        ]
        wipo = {
            "doc_id": "user_1_sample_financial_report.pdf::page_10::chunk_revenue",
            "content": "Total revenue on an IPSAS basis amounted to 350.0 million Swiss francs.",
            "metadata": {"type": "text", "page": 10, "doc_name": "sample_financial_report.pdf"},
            "score": 0.2,
        }

        covered = engine._ensure_multi_doc_coverage(
            selected + [wipo],
            selected,
            ["neutral_report.pdf", "sample_financial_report.pdf"],
            top_k=2,
        )

        assert any(chunk["metadata"]["doc_name"] == "sample_financial_report.pdf" for chunk in covered)
        # Phase 1: supporting_source_page no longer propagated by _ensure_multi_doc_coverage
        assert len(covered) == 2
    finally:
        _cleanup(path)







def test_force_supporting_page_coverage_adds_missing_pdfsol_cover_metric_page(monkeypatch):
    """Phase 1: _force_supporting_page_coverage was removed. Verify it no longer exists."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, '_force_supporting_page_coverage'), (
            "_force_supporting_page_coverage must be removed (Phase 1 retrieval integrity)"
        )
        assert not hasattr(engine, '_supporting_pages_for_query'), (
            "_supporting_pages_for_query must be removed (Phase 1 retrieval integrity)"
        )
    finally:
        _cleanup(path)


def test_leac_cash_equivalents_queries_fallback_to_statement_page():
    """Phase 1: _fallback_pages_for_query and _supporting_pages_for_query removed. Verify gone."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, '_fallback_pages_for_query'), (
            "_fallback_pages_for_query must be removed (Phase 1 retrieval integrity)"
        )
        assert not hasattr(engine, '_supporting_pages_for_query'), (
            "_supporting_pages_for_query must be removed (Phase 1 retrieval integrity)"
        )
    finally:
        _cleanup(path)







