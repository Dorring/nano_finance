"""Unit tests for RAG Orchestrator."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.application.rag_orchestrator import RAGOrchestrator


class TestConversationalQuery:
    def test_greeting(self):
        result = RAGOrchestrator._handle_conversational_query("hello")
        assert result is not None
        assert "FinQuery" in result

    def test_identity(self):
        result = RAGOrchestrator._handle_conversational_query("what are you")
        assert result is not None
        assert "FinQuery" in result

    def test_thanks(self):
        result = RAGOrchestrator._handle_conversational_query("thank you")
        assert result is not None

    def test_goodbye(self):
        result = RAGOrchestrator._handle_conversational_query("bye")
        assert result is not None

    def test_financial_query_not_conversational(self):
        result = RAGOrchestrator._handle_conversational_query("what was the revenue in Q3")
        assert result is None

    def test_normal_query_not_conversational(self):
        result = RAGOrchestrator._handle_conversational_query("what is the total revenue")
        assert result is None

    def test_capability_question(self):
        result = RAGOrchestrator._handle_conversational_query("how does this work")
        assert result is not None

    def test_chinese_financial_not_conversational(self):
        result = RAGOrchestrator._handle_conversational_query("营收增长多少")
        assert result is None
