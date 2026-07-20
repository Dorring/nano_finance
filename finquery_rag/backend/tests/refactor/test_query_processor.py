"""Unit tests for QueryProcessor extracted from RAGEngine."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.retrieval.query_processor import QueryProcessor


@pytest.fixture
def qp():
    return QueryProcessor()


class TestQueryProcessorExpand:
    def test_empty_query(self, qp):
        assert qp.expand("") == ""

    def test_plain_query_unchanged(self, qp):
        assert qp.expand("What is the main idea?") == "What is the main idea?"

    def test_title_expansion(self, qp):
        result = qp.expand("What is the title?")
        assert "paper title" in result

    def test_reporting_period_expansion(self, qp):
        result = qp.expand("What is the reporting period?")
        assert "fiscal year" in result

    def test_total_revenue_expansion(self, qp):
        result = qp.expand("What was the total revenue?")
        assert "total revenue" in result

    def test_cash_equivalents_expansion(self, qp):
        result = qp.expand("What are the cash and cash equivalents?")
        assert "current assets" in result

    def test_cjk_title_expansion(self, qp):
        result = qp.expand("这篇论文的标题是什么")
        assert "paper title" in result

    def test_cjk_author_expansion(self, qp):
        result = qp.expand("作者是谁")
        assert "paper authors" in result

    def test_net_assets_expansion(self, qp):
        result = qp.expand("What are the net assets?")
        assert "statement of financial position" in result

    def test_credit_facilities_expansion(self, qp):
        result = qp.expand("Tell me about credit facilities")
        assert "revolving credit facility" in result

    def test_gross_margin_expansion(self, qp):
        result = qp.expand("What is the gross margin?")
        assert "gross profit" in result

    def test_operating_cash_flow_expansion(self, qp):
        result = qp.expand("What is the operating cash flow?")
        assert "net cash" in result


class TestQueryProcessorClassification:
    def test_front_matter_title(self, qp):
        assert qp.is_front_matter_query("What is the title?") is True

    def test_front_matter_author(self, qp):
        assert qp.is_front_matter_query("Who is the author?") is True

    def test_front_matter_abstract(self, qp):
        assert qp.is_front_matter_query("What is the abstract?") is True

    def test_not_front_matter_revenue(self, qp):
        assert qp.is_front_matter_query("What is the total revenue?") is False

    def test_title_query(self, qp):
        assert qp.is_title_query("What is the title?") is True

    def test_not_title_query(self, qp):
        assert qp.is_title_query("What is the revenue?") is False

    def test_numeric_query_revenue(self, qp):
        assert qp.is_numeric_query("What was the total revenue?") is True

    def test_numeric_query_how_much(self, qp):
        assert qp.is_numeric_query("How much cash?") is True

    def test_not_numeric_query_title(self, qp):
        assert qp.is_numeric_query("What is the title?") is False

    def test_not_numeric_which_documents(self, qp):
        assert qp.is_numeric_query("Which documents mention revenue?") is False


class TestQueryProcessorFollowup:
    def test_followup_it(self, qp):
        assert qp.looks_like_followup_question("What about it?") is True

    def test_followup_those(self, qp):
        assert qp.looks_like_followup_question("Tell me about those") is True

    def test_not_followup_standalone(self, qp):
        assert qp.looks_like_followup_question("What is the title?") is False

    def test_not_followup_empty(self, qp):
        assert qp.looks_like_followup_question("") is False


class TestQueryProcessorRewriteValidation:
    def test_valid_rewrite(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "What was the total revenue for 2023?") is True

    def test_too_short_rewrite(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "Hi") is False

    def test_rewrite_with_newline(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "What was\nthe revenue?") is False

    def test_rewrite_with_user_label(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "User: What was the revenue?") is False

    def test_rewrite_with_assistant_label(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "Assistant: The revenue was $100M") is False

    def test_empty_rewrite(self, qp):
        assert qp.is_valid_rewritten_query("What is the revenue?", "") is False


class TestQueryProcessorDeterministic:
    def test_should_try_numeric(self, qp):
        chunks = [{"score": 0.5}]
        assert qp.should_try_deterministic_numeric_answer("What was the total revenue?", chunks) is True

    def test_should_not_try_numeric_no_chunks(self, qp):
        assert qp.should_try_deterministic_numeric_answer("What was the total revenue?", []) is False

    def test_should_not_try_numeric_not_numeric(self, qp):
        chunks = [{"score": 0.5}]
        assert qp.should_try_deterministic_numeric_answer("What is the title?", chunks) is False

    def test_should_generate_low_confidence(self, qp):
        chunks = [{"score": 0.01}]
        assert qp.should_generate_with_low_confidence(
            "What was the revenue?", chunks,
            numeric_rrf_floor=0.008, numeric_dense_floor=0.08
        ) is True

    def test_should_not_generate_zero_score(self, qp):
        chunks = [{"score": 0.0}]
        assert qp.should_generate_with_low_confidence(
            "What was the revenue?", chunks,
            numeric_rrf_floor=0.008, numeric_dense_floor=0.08
        ) is False

    def test_should_try_factual(self, qp):
        assert qp.should_try_deterministic_factual_answer("What is the title and reporting period?") is True

    def test_should_not_try_factual(self, qp):
        assert qp.should_try_deterministic_factual_answer("What is the revenue?") is False
