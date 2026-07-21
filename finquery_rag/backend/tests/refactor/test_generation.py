"""Unit tests for Generation components."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.generation.prompt_builder import get_system_prompt, SYSTEM_PROMPT
from src.generation.response_renderer import validate_answer
from src.generation.deterministic_answers import DeterministicAnswerExtractor


class TestPromptBuilder:
    def test_system_prompt_is_string(self):
        assert isinstance(get_system_prompt(), str)

    def test_system_prompt_contains_finquery(self):
        assert "FinQuery" in get_system_prompt()

    def test_system_prompt_rules(self):
        prompt = get_system_prompt()
        assert "context" in prompt.lower()
        assert "source" in prompt.lower()

    def test_constant_matches_function(self):
        assert SYSTEM_PROMPT == get_system_prompt()


class TestResponseRenderer:
    def test_validate_empty_answer(self):
        result = validate_answer("", [])
        assert "couldn't" in result.lower()

    def test_validate_short_answer(self):
        result = validate_answer("Hi", [])
        assert "couldn't" in result.lower()

    def test_validate_normal_answer(self):
        result = validate_answer("Revenue was $100 million in 2023.", [])
        assert "Revenue" in result

    def test_validate_strips_artifacts(self):
        result = validate_answer("Revenue was $100 million.</s>", [])
        assert "</s>" not in result
        assert "Revenue" in result

    def test_validate_truncates_long_answer(self):
        long_answer = "A" * 3000
        result = validate_answer(long_answer, [], max_new_tokens=512)
        assert len(result) < 3000


class TestDeterministicAnswerExtractor:
    def test_front_matter_query_non_title(self):
        ext = DeterministicAnswerExtractor()
        result = ext.answer_front_matter_query("what is revenue?", [])
        assert result is None

    def test_front_matter_query_with_title_chunk(self):
        ext = DeterministicAnswerExtractor()
        chunks = [{
            "content": "Annual Report 2023",
            "metadata": {"type": "front_matter", "subtype": "title", "page": 1},
            "score": 0.5,
            "doc_id": "test::1",
        }]
        result = ext.answer_front_matter_query("what is the title", chunks)
        assert result is not None
        assert "Annual Report 2023" in result["answer"]

    def test_numeric_query_no_context(self):
        ext = DeterministicAnswerExtractor()
        result = ext.answer_numeric_query_from_context("revenue", "", [])
        assert result is None

    def test_factual_query_no_context(self):
        ext = DeterministicAnswerExtractor()
        result = ext.answer_factual_query_from_context("what is the title", "", [])
        assert result is None

    def test_deterministic_delegates_to_factual(self):
        ext = DeterministicAnswerExtractor()
        result = ext.answer_deterministic_query_from_context("what is the title", "", [])
        assert result is None

    def test_clean_deterministic_title(self):
        assert "Annual Report" in DeterministicAnswerExtractor._clean_deterministic_title("  Annual Report  ")

    def test_is_valid_deterministic_title_too_short(self):
        assert DeterministicAnswerExtractor._is_valid_deterministic_title("ab") is False

    def test_is_valid_deterministic_title_generic(self):
        assert DeterministicAnswerExtractor._is_valid_deterministic_title("annual report") is False

    def test_parse_context_lines(self):
        context = "[report.pdf, p1]\nRevenue was $100 million.\n---\n[report.pdf, p2]\nExpenses were $50 million."
        lines = DeterministicAnswerExtractor._parse_context_lines(context)
        assert len(lines) == 2
        assert lines[0]["source"] == "report.pdf, p1"

    def test_important_query_terms(self):
        terms = DeterministicAnswerExtractor._important_query_terms("what was the revenue in 2023")
        assert "revenue" in terms
        assert "what" not in terms

    def test_is_numeric_financial_query(self):
        assert DeterministicAnswerExtractor._is_numeric_financial_query("what was the revenue") is True
        assert DeterministicAnswerExtractor._is_numeric_financial_query("hello") is False
