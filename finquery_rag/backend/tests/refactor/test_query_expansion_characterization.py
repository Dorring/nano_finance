"""Characterize current RAGEngine query expansion behavior.

These tests record the exact output of _expand_retrieval_query for various
inputs so that the extraction to QueryProcessor preserves behavior.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    """Create a minimal RAGEngine for testing query expansion."""
    return RAGEngine(
        llm_client=None,
        model_name="test",
        use_hybrid=True,
    )


class TestQueryExpansionCharacterization:
    """Record current _expand_retrieval_query behavior."""

    def test_empty_query_unchanged(self):
        engine = _make_engine()
        assert engine._expand_retrieval_query("") == ""

    def test_plain_english_query_unchanged(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What is the main idea?")
        assert result == "What is the main idea?"

    def test_title_query_adds_paper_title(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What is the title?")
        assert "paper title" in result
        assert result.startswith("What is the title?")

    def test_reporting_period_adds_fiscal_terms(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What is the reporting period?")
        assert "year ended" in result
        assert "fiscal year" in result

    def test_total_revenue_adds_revenue_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What was the total revenue?")
        assert "total revenue" in result

    def test_cash_equivalents_adds_current_assets(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What are the cash and cash equivalents?")
        assert "current assets" in result

    def test_cjk_title_adds_english_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("这篇论文的标题是什么")
        assert "paper title" in result

    def test_cjk_author_adds_english_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("作者是谁")
        assert "paper authors" in result

    def test_cjk_abstract_adds_english_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("摘要是什么")
        assert "abstract" in result

    def test_net_assets_adds_statement_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What are the net assets?")
        assert "statement of financial position" in result

    def test_credit_facilities_adds_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("Tell me about credit facilities")
        assert "revolving credit facility" in result

    def test_gross_margin_adds_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What is the gross margin?")
        assert "gross profit" in result

    def test_operating_cash_flow_adds_expansion(self):
        engine = _make_engine()
        result = engine._expand_retrieval_query("What is the operating cash flow?")
        assert "net cash" in result
        assert "operating activities" in result


class TestFrontMatterQueryCharacterization:
    """Record current _is_document_front_matter_query behavior."""

    def test_title_is_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("What is the title?") is True

    def test_author_is_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("Who is the author?") is True

    def test_abstract_is_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("What is the abstract?") is True

    def test_revenue_is_not_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("What is the total revenue?") is False

    def test_cjk_title_is_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("标题是什么") is True

    def test_cjk_author_is_front_matter(self):
        engine = _make_engine()
        assert engine._is_document_front_matter_query("作者是谁") is True


class TestNumericQueryCharacterization:
    """Record current _is_numeric_financial_query behavior."""

    def test_revenue_is_numeric(self):
        engine = _make_engine()
        assert engine._is_numeric_financial_query("What was the total revenue?") is True

    def test_title_is_not_numeric(self):
        engine = _make_engine()
        assert engine._is_numeric_financial_query("What is the title?") is False

    def test_how_much_is_numeric(self):
        engine = _make_engine()
        assert engine._is_numeric_financial_query("How much cash?") is True

    def test_how_many_is_numeric(self):
        engine = _make_engine()
        assert engine._is_numeric_financial_query("How many employees?") is True
