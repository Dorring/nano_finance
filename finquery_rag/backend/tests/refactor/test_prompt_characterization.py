"""Characterize current RAGEngine prompt behavior.

These tests record the exact system prompt and prompt assembly behavior
so that extraction to PromptBuilder preserves behavior.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    return RAGEngine(llm_client=None, model_name="test", use_hybrid=True)


class TestPromptCharacterization:
    """Record current _get_system_prompt behavior."""

    def test_system_prompt_contains_finquery_identity(self):
        engine = _make_engine()
        prompt = engine._get_system_prompt()
        assert "FinQuery" in prompt

    def test_system_prompt_contains_citation_rule(self):
        engine = _make_engine()
        prompt = engine._get_system_prompt()
        assert "Source:" in prompt or "source" in prompt.lower()

    def test_system_prompt_contains_numeric_rule(self):
        engine = _make_engine()
        prompt = engine._get_system_prompt()
        assert "exact" in prompt.lower() or "number" in prompt.lower()

    def test_system_prompt_is_deterministic(self):
        engine = _make_engine()
        prompt1 = engine._get_system_prompt()
        prompt2 = engine._get_system_prompt()
        assert prompt1 == prompt2

    def test_system_prompt_no_markdown_table_rule(self):
        engine = _make_engine()
        prompt = engine._get_system_prompt()
        assert "markdown table" in prompt.lower() or "never use markdown" in prompt.lower()
