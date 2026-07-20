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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_extracts_operating_cash_flow_table_row():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p50]\n"
            "Net cash flows provided by (used in): Operating activities $ 24,053 $ 9,703 $ 14,600.\n"
            "Financing activities 64,563 (11,233) (5,890).\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What net cash was provided by operating activities in 2025?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 50}],
        )

        assert answer is not None
        assert "Answer: $24.1 million, 24,053." in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_extracts_wipo_statement_position_values():
    engine, path = _engine()
    try:
        cash_context = (
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "STATEMENT I: Statement of Financial Position (in thousands of Swiss francs) "
            "Cash and cash equivalents 143,540.\n"
        )
        assets_context = (
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "STATEMENT I: Statement of Financial Position (in thousands of Swiss francs) "
            "Net assets 387,063.\n"
        )

        cash_answer = engine.answer_numeric_query_from_context(
            "What were WIPO cash and cash equivalents at December 31, 2020?",
            cash_context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 24}],
        )
        assets_answer = engine.answer_numeric_query_from_context(
            "What were WIPO net assets at December 31, 2020?",
            assets_context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 24}],
        )

        assert cash_answer is not None
        assert "143,540, thousands of Swiss francs" in cash_answer["answer"]
        assert assets_answer is not None
        assert "387,063, thousands of Swiss francs" in assets_answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_uses_wipo_statement_position_page_fallback_when_table_value_is_missing():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "STATEMENT I: Statement of Financial Position (in thousands of Swiss francs).\n"
        )

        cash_answer = engine.answer_numeric_query_from_context(
            "What were WIPO cash and cash equivalents at December 31, 2020?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 24}],
        )
        assets_answer = engine.answer_numeric_query_from_context(
            "What were WIPO net assets at December 31, 2020?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 24}],
        )

        assert cash_answer is not None
        assert "143,540, thousands of Swiss francs" in cash_answer["answer"]
        assert assets_answer is not None
        assert "387,063, thousands of Swiss francs" in assets_answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_uses_wipo_total_revenue_table_page_fallback():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p25]\n"
            "STATEMENT II: Statement of Financial Performance (in thousands of Swiss francs).\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What was WIPO total revenue in 2020 on an IPSAS basis?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 25}],
        )

        assert answer is not None
        assert "468.3 million Swiss francs" in answer["answer"]
        assert "468,272" in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_extracts_credit_facility_components():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p48]\n"
            "Credit Facilities consisted of a Revolving Credit Facility of $45.0 million "
            "and a Term Loan of $25.0 million.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What were the two components of PDF Solutions Credit Facilities?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 48}],
        )

        assert answer is not None
        assert "Revolving Credit Facility, $45 million" in answer["answer"]
        assert "Term Loan, $25 million" in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_prefers_pdfsol_cash_equivalents_over_nearby_tax_values():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p48]\n"
            "The Company had cash and cash equivalents of $42.2 million as of December 31, 2025.\n"
            "[FINAL Annual Report.pdf, p40]\n"
            "Unrelated income tax expense was $1.3 million and the effective tax rate was 40%.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What cash and cash equivalents did PDF Solutions report at year-end 2025?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 48}],
        )

        assert answer is not None
        assert "Answer: $42.2 million." in answer["answer"]
        assert "$1.3 million" not in answer["answer"]
        assert "40%" not in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_prefers_wipo_total_revenue_share_over_growth_rate():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "PCT system fees, accounting for 76.6 per cent of total revenue, "
            "increased by 6.1 per cent compared to 2019.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What percentage of WIPO total revenue came from PCT system fees in 2020?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 10}],
        )

        assert answer is not None
        assert "Answer: 76.6 per cent." in answer["answer"]
        assert "6.1 per cent" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_extracts_wipo_madrid_total_revenue_share():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Madrid system fees amounted to 76.2 million Swiss francs, "
            "representing 16.3 per cent of total revenue.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "What percentage of WIPO total revenue came from Madrid system fees in 2020?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 10}],
        )

        assert answer is not None
        assert "Answer: 16.3 per cent." in answer["answer"]
        assert "76.2 million" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_answer_prefers_wipo_statement_v_pct_system_actual_over_pct_share():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p29]\n"
            "STATEMENT V: Comparison of budget and actual amounts. The PCT System actual 2020 amount is 98,755.\n"
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "PCT system fees accounted for 76.6 per cent of total revenue.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "In WIPO Statement V expenses, what was the actual 2020 amount for The PCT System?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 29}],
        )

        assert answer is not None
        assert "The PCT System" in answer["answer"]
        assert "98,755" in answer["answer"]
        assert "76.6 per cent" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_extracts_pdfsol_cover_title_from_context():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p53]\n"
            "CONSOLIDATED STATEMENTS OF STOCKHOLDERS' EQUITY.\n"
            "[FINAL Annual Report.pdf, p1]\n"
            "2025 Driving Smart Solutions ANNUAL REPORT.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What is the title shown on the cover of the PDF Solutions report?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 1}],
        )

        assert answer is not None
        assert "2025 Driving Smart Solutions Annual Report" in answer["answer"]
        assert "STOCKHOLDERS" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_falls_back_to_known_pdfsol_cover_title_when_context_is_polluted():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p53]\n"
            "CONSOLIDATED STATEMENTS OF STOCKHOLDERS' EQUITY.\n"
            "[FINAL Annual Report.pdf, p55]\n"
            "Basis for Opinion These consolidated financial statements are management's responsibility.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What is the title shown on the cover of the PDF Solutions report?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 1}],
        )

        assert answer is not None
        assert "2025 Driving Smart Solutions Annual Report" in answer["answer"]
        assert "STOCKHOLDERS" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_uses_full_context_for_wipo_title_and_reporting_period():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p1]\n"
            "Annual financial report and financial statements\n"
            "Year to December 31, 2020\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What is the title and reporting period of the WIPO financial report?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 1}],
        )

        assert answer is not None
        assert "Annual financial report and financial statements" in answer["answer"]
        assert "Year to December 31, 2020" in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_falls_back_to_wipo_title_and_reporting_period():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p1]\n"
            "WIPO ANNUAL FINANCIAL REPORT AND FINANCIAL STATEMENTS 2020.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "What is the title and reporting period of the WIPO document?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 1}],
        )

        assert answer is not None
        assert "Annual financial report and financial statements" in answer["answer"]
        assert "Year to December 31, 2020" in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_resolves_wipo_organization_from_query_when_context_is_sparse():
    engine, path = _engine()
    try:
        context = (
            "[wipo_pub_rn2021_18e.pdf, p1]\n"
            "Annual financial report and financial statements.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "Which organization prepared the WIPO annual financial report?",
            context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 1}],
        )

        assert answer is not None
        assert "World Intellectual Property Organization (WIPO)." in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_uses_stable_leac_financial_statement_definition():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p1]\n"
            "Financial Statements of a Company.\n"
            "[leac203.pdf, p22]\n"
            "Thus, financial statements form the basis for granting of credit.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "According to leac203.pdf, what are financial statements?",
            context,
            [{"filename": "leac203.pdf", "page": 1}],
        )

        assert answer is not None
        assert "basic and formal annual reports" in answer["answer"]
        assert "corporate management communicates financial information" in answer["answer"]
        assert "basis for granting of credit" not in answer["answer"].split("The document states:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_answer_uses_expected_current_item_wording():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p10]\n"
            "An item is current when it is expected to be realized in the operating cycle, "
            "realized within twelve months, held primarily for trading, or is cash and cash equivalent.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "List two criteria that make an item current according to leac203.pdf.",
            context,
            [{"filename": "leac203.pdf", "page": 10}],
        )

        assert answer is not None
        assert "operating cycle" in answer["answer"]
        assert "within twelve months" in answer["answer"]
        assert "held primarily for trading" in answer["answer"]
        assert "cash and cash equivalent" in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
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
    """Phase 1: _fallback_pages_for_query was removed. Verify it no longer exists."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, '_fallback_pages_for_query'), (
            "_fallback_pages_for_query must be removed (Phase 1 retrieval integrity)"
        )
    finally:
        _cleanup(path)


def test_page_fallback_chunks_are_added_before_reranking(monkeypatch):
    """Phase 1: page fallback augmentation removed. Verify no hardcoded page injection."""
    engine, path = _engine()
    try:
        assert not hasattr(engine, '_augment_with_page_fallbacks'), (
            "_augment_with_page_fallbacks must be removed (Phase 1 retrieval integrity)"
        )
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_page_fallback_pages_are_preserved_after_reranking():
    engine, path = _engine()
    try:
        selected = [
            _chunk(score=0.9, content="High scoring but unrelated page 1"),
            {
                "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_25::chunk_stmt",
                "content": "Statement page.",
                "metadata": {"type": "text", "page": 25, "doc_name": "wipo_pub_rn2021_18e.pdf"},
                "score": 0.8,
            },
        ]
        fallback_page_10 = {
            "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_10::chunk_revenue",
            "content": "PCT system fees accounted for 76.6 per cent of total revenue.",
            "metadata": {"type": "text", "page": 10, "doc_name": "wipo_pub_rn2021_18e.pdf", "page_fallback": True},
            "score": 0.05,
        }

        covered = engine._ensure_page_fallback_coverage(
            selected + [fallback_page_10],
            selected,
            top_k=2,
        )

        assert any(chunk["metadata"].get("page") == 10 for chunk in covered)
        assert len(covered) == 2
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_supporting_source_pages_are_prioritized_over_plain_fallbacks():
    engine, path = _engine()
    try:
        selected = [
            _chunk(score=0.9, content="High scoring but unrelated page 1"),
            {
                "doc_id": "user_1_FINAL Annual Report.pdf::page_40::chunk_tax",
                "content": "Unrelated tax disclosure.",
                "metadata": {"type": "text", "page": 40, "doc_name": "FINAL Annual Report.pdf"},
                "score": 0.8,
            },
        ]
        supporting_page_3 = {
            "doc_id": "user_1_FINAL Annual Report.pdf::page_3::chunk_revenue",
            "content": "Record revenue was $219 million.",
            "metadata": {
                "type": "text",
                "page": 3,
                "doc_name": "FINAL Annual Report.pdf",
                "page_fallback": True,
                "supporting_source_page": True,
            },
            "score": 0.08,
        }
        plain_fallback_page_48 = {
            "doc_id": "user_1_FINAL Annual Report.pdf::page_48::chunk_cash",
            "content": "Cash and cash equivalents.",
            "metadata": {
                "type": "text",
                "page": 48,
                "doc_name": "FINAL Annual Report.pdf",
                "page_fallback": True,
            },
            "score": 0.05,
        }

        covered = engine._ensure_page_fallback_coverage(
            selected + [plain_fallback_page_48, supporting_page_3],
            selected,
            top_k=1,
        )

        assert any(chunk["metadata"].get("page") == 3 for chunk in covered)
        assert all(chunk["metadata"].get("page") != 48 for chunk in covered)
        assert len(covered) == 1
    finally:
        _cleanup(path)


def test_multi_doc_coverage_keeps_one_candidate_per_requested_document():
    engine, path = _engine()
    try:
        selected = [
            {
                "doc_id": "user_1_FINAL Annual Report.pdf::page_45::chunk_revenue",
                "content": "Platform revenue was $181.0 million.",
                "metadata": {"type": "text", "page": 45, "doc_name": "FINAL Annual Report.pdf"},
                "score": 0.9,
            },
            {
                "doc_id": "user_1_FINAL Annual Report.pdf::page_3::chunk_record",
                "content": "Record revenue was $219 million.",
                "metadata": {
                    "type": "text",
                    "page": 3,
                    "doc_name": "FINAL Annual Report.pdf",
                    "supporting_source_page": True,
                },
                "score": 0.8,
            },
        ]
        wipo = {
            "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_10::chunk_revenue",
            "content": "Total revenue on an IPSAS basis amounted to 468.3 million Swiss francs.",
            "metadata": {"type": "text", "page": 10, "doc_name": "wipo_pub_rn2021_18e.pdf"},
            "score": 0.2,
        }

        covered = engine._ensure_multi_doc_coverage(
            selected + [wipo],
            selected,
            ["FINAL Annual Report.pdf", "wipo_pub_rn2021_18e.pdf"],
            top_k=2,
        )

        assert any(chunk["metadata"]["doc_name"] == "wipo_pub_rn2021_18e.pdf" for chunk in covered)
        # Phase 1: supporting_source_page no longer propagated by _ensure_multi_doc_coverage
        assert len(covered) == 2
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_multi_doc_compare_revenue_uses_grouped_deterministic_answer():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p3]\n"
            "PDF Solutions achieved record revenue of $219 million in 2025.\n"
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Total revenue on an IPSAS basis amounted to 468.3 million Swiss francs in 2020.\n"
        )

        answer = engine.answer_multi_doc_query_from_context(
            "Compare revenue between PDF Solutions and WIPO.",
            context,
            [
                {"filename": "FINAL Annual Report.pdf", "page": 3},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 10},
            ],
        )

        assert answer is not None
        assert "$219 million" in answer["answer"]
        assert "468.3 million Swiss francs" in answer["answer"]
        assert answer["diagnostic"] == "deterministic_multi_doc_compare"
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_multi_doc_cash_terms_query_is_not_treated_as_numeric_extraction():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p48]\n"
            "Cash and cash equivalents were included in the balance sheet.\n"
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "Cash and cash equivalents were reported in the statement of financial position.\n"
            "[leac203.pdf, p10]\n"
            "Cash and cash equivalents include bank balance and cash in hand.\n"
        )

        answer = engine.answer_deterministic_query_from_context(
            "Which documents mention cash and cash equivalents?",
            context,
            [
                {"filename": "FINAL Annual Report.pdf", "page": 48},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 24},
                {"filename": "leac203.pdf", "page": 10},
            ],
        )

        assert not engine._is_numeric_financial_query("Which documents mention cash and cash equivalents?")
        assert answer is not None
        assert "FINAL Annual Report.pdf mentions cash and cash equivalents" in answer["answer"]
        assert "wipo_pub_rn2021_18e.pdf mentions cash and cash equivalents" in answer["answer"]
        assert "leac203.pdf mentions cash and cash equivalents" in answer["answer"]
        assert "$1.3 million" not in answer["answer"]
    finally:
        _cleanup(path)
@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_pdfsol_cash_equivalents_uses_statement_page_when_text_is_polluted():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p48]\n"
            "A nearby tax disclosure mentions a cash tax benefit of $1.3 million and a 40% effective tax rate.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "How much cash and cash equivalents did PDF Solutions have as of December 31, 2025?",
            context,
            [{"filename": "FINAL Annual Report.pdf", "page": 48}],
        )

        assert answer is not None
        assert "Answer: $42.2 million." in answer["answer"]
        assert "$1.3 million" not in answer["answer"].split("Evidence:", 1)[0]
        assert "40%" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_wipo_revenue_share_uses_expected_page_metric_over_growth_rate():
    engine, path = _engine()
    try:
        pct_context = (
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Revenue from PCT system fees increased by 6.1 per cent compared to 2019.\n"
        )
        madrid_context = (
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Madrid system fees were 76.2 million Swiss francs in 2020.\n"
        )

        pct_answer = engine.answer_numeric_query_from_context(
            "What percentage of WIPO total revenue came from PCT system fees in 2020?",
            pct_context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 10}],
        )
        madrid_answer = engine.answer_numeric_query_from_context(
            "What percentage of WIPO total revenue came from Madrid system fees in 2020?",
            madrid_context,
            [{"filename": "wipo_pub_rn2021_18e.pdf", "page": 10}],
        )

        assert pct_answer is not None
        assert "Answer: 76.6 per cent." in pct_answer["answer"]
        assert "6.1 per cent" not in pct_answer["answer"].split("Evidence:", 1)[0]
        assert madrid_answer is not None
        assert "Answer: 16.3 per cent." in madrid_answer["answer"]
        assert "76.2 million" not in madrid_answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_sunfill_reserve_surplus_uses_known_balance_sheet_page():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p13]\n"
            "The table contains page numbers and unrelated dates, but it is the Sunfill Ltd. balance sheet page.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "In the Sunfill Ltd. illustration, what reserve and surplus amount is shown for March 31, 2017?",
            context,
            [{"filename": "leac203.pdf", "page": 13}],
        )

        assert answer is not None
        assert "Answer: 2,00,000." in answer["answer"]
        assert "156" not in answer["answer"].split("Evidence:", 1)[0]
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


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_multi_doc_coverage_prefers_supporting_candidate_for_missing_doc():
    engine, path = _engine()
    try:
        selected = [{
            "doc_id": "user_1_FINAL Annual Report.pdf::page_3::chunk_revenue",
            "content": "Record revenue was $219 million.",
            "metadata": {"type": "text", "page": 3, "doc_name": "FINAL Annual Report.pdf"},
            "score": 0.9,
        }]
        noisy_wipo = {
            "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_1::chunk_intro",
            "content": "Introductory WIPO text.",
            "metadata": {"type": "text", "page": 1, "doc_name": "wipo_pub_rn2021_18e.pdf"},
            "score": 0.8,
        }
        supporting_wipo = {
            "doc_id": "user_1_wipo_pub_rn2021_18e.pdf::page_10::chunk_revenue",
            "content": "Total revenue was 468.3 million Swiss francs.",
            "metadata": {
                "type": "text",
                "page": 10,
                "doc_name": "wipo_pub_rn2021_18e.pdf",
                "page_fallback": True,
                "supporting_source_page": True,
            },
            "score": 0.12,
        }

        covered = engine._ensure_multi_doc_coverage(
            [selected[0], noisy_wipo, supporting_wipo],
            selected,
            ["FINAL Annual Report.pdf", "wipo_pub_rn2021_18e.pdf"],
            top_k=2,
        )

        assert any(chunk["metadata"].get("page") == 10 for chunk in covered)
        assert all(chunk["metadata"].get("page") != 1 for chunk in covered)
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_multi_doc_revenue_compare_uses_document_level_figures_from_context():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p45]\n"
            "Platform revenue was $181.0 million.\n"
            "[FINAL Annual Report.pdf, p3]\n"
            "Record revenue reached $219 million.\n"
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Total revenue was 468.3 million Swiss francs.\n"
        )

        answer = engine.answer_multi_doc_query_from_context(
            "Compare the main 2025/2020 revenue figures in the PDF Solutions and WIPO reports.",
            context,
            [
                {"filename": "FINAL Annual Report.pdf", "page": 45},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 10},
            ],
        )

        assert answer is not None
        assert "$219 million" in answer["answer"]
        assert "468.3 million Swiss francs" in answer["answer"]
        assert "$181" not in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_multi_doc_cash_terms_lists_all_documents_from_context_sources():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p10]\n"
            "Cash and cash equivalents is shown as a line item.\n"
            "[FINAL Annual Report.pdf, p48]\n"
            "Cash and cash equivalents are reported.\n"
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "Cash and cash equivalents are reported.\n"
        )

        answer = engine.answer_multi_doc_query_from_context(
            "Which documents mention cash and cash equivalents as a reported line item?",
            context,
            [{"filename": "leac203.pdf", "page": 10}],
        )

        assert answer is not None
        for expected in ("FINAL Annual Report.pdf", "wipo_pub_rn2021_18e.pdf", "leac203.pdf"):
            assert expected in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_numeric_summary_prefers_main_revenue_for_multi_doc_compare():
    engine, path = _engine()
    try:
        context = (
            "[FINAL Annual Report.pdf, p45]\n"
            "Platform revenue was $181.0 million.\n"
            "[FINAL Annual Report.pdf, p3]\n"
            "Record revenue reached $219 million.\n"
            "[wipo_pub_rn2021_18e.pdf, p10]\n"
            "Total revenue was 468.3 million Swiss francs.\n"
        )

        answer = engine.answer_numeric_query_from_context(
            "Compare the main 2025/2020 revenue figures in the PDF Solutions and WIPO reports.",
            context,
            [
                {"filename": "FINAL Annual Report.pdf", "page": 3},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 10},
            ],
        )

        assert answer is not None
        assert "$219 million" in answer["answer"]
        assert "468.3 million Swiss francs" in answer["answer"]
        assert "$181" not in answer["answer"].split("Evidence:", 1)[0]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_summary_lists_all_cash_equivalent_documents():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p10]\n"
            "Cash and cash equivalents is shown as a line item.\n"
            "[FINAL Annual Report.pdf, p48]\n"
            "Cash and cash equivalents are reported.\n"
            "[wipo_pub_rn2021_18e.pdf, p24]\n"
            "Cash and cash equivalents are reported.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "Which documents mention cash and cash equivalents as a reported line item?",
            context,
            [
                {"filename": "leac203.pdf", "page": 10},
                {"filename": "FINAL Annual Report.pdf", "page": 48},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 24},
            ],
        )

        assert answer is not None
        for expected in ("FINAL Annual Report.pdf", "wipo_pub_rn2021_18e.pdf", "leac203.pdf"):
            assert expected in answer["answer"]
    finally:
        _cleanup(path)


@pytest.mark.skip(reason="Phase 1: tests removed benchmark-specific behavior")
def test_factual_cash_terms_uses_sources_when_context_is_truncated():
    engine, path = _engine()
    try:
        context = (
            "[leac203.pdf, p10]\n"
            "Cash and cash equivalents is shown as a line item.\n"
            "[FINAL Annual Report.pdf, p48]\n"
            "Cash and cash equivalents are reported.\n"
        )

        answer = engine.answer_factual_query_from_context(
            "Which documents mention cash and cash equivalents as a reported line item?",
            context,
            [
                {"filename": "leac203.pdf", "page": 10},
                {"filename": "FINAL Annual Report.pdf", "page": 48},
                {"filename": "wipo_pub_rn2021_18e.pdf", "page": 24},
            ],
        )

        assert answer is not None
        for expected in ("FINAL Annual Report.pdf", "wipo_pub_rn2021_18e.pdf", "leac203.pdf"):
            assert expected in answer["answer"]
    finally:
        _cleanup(path)
