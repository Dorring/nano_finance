"""Phase 10A tests: deterministic query intent routing."""
import os
import sys
from unittest.mock import MagicMock

mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.intent import classify_query_intent
from services.rag_engine import RAGEngine


def test_intent_classifies_conversation_without_retrieval():
    result = classify_query_intent("hello")
    assert result["intent"] == "conversation"
    assert result["requires_retrieval"] is False


def test_intent_classifies_financial_calculation():
    result = classify_query_intent("Calculate the YoY revenue growth rate")
    assert result["intent"] == "financial_calculation"
    assert result["requires_retrieval"] is True
    assert result["confidence"] >= 0.8


def test_intent_keeps_reported_metrics_as_document_qa():
    pct = classify_query_intent("What percentage of WIPO total revenue came from PCT system fees in 2020?")
    margin = classify_query_intent("What was PDF Solutions GAAP gross margin in 2025?")
    growth = classify_query_intent("What was PDF Solutions volume-based revenue in 2025 and what was the year-over-year growth rate?")

    assert pct["intent"] == "document_qa"
    assert pct["reason"] == "reported_metric_lookup"
    assert margin["intent"] == "document_qa"
    assert growth["intent"] == "document_qa"


def test_intent_still_classifies_explicit_calculation():
    result = classify_query_intent("Calculate the growth rate from 2024 revenue to 2025 revenue")

    assert result["intent"] == "financial_calculation"


def test_intent_classifies_document_summary():
    result = classify_query_intent("Summarize the key financial metrics")
    assert result["intent"] == "document_summary"
    assert result["requires_retrieval"] is True


def test_intent_defaults_unknown_to_retrieval():
    result = classify_query_intent("What changed in the appendix?")
    assert result["intent"] == "document_qa"
    assert result["requires_retrieval"] is True
    assert result["reason"] == "default_to_retrieval"


def test_intent_classifies_clear_out_of_scope():
    result = classify_query_intent("What is the weather today?")
    assert result["intent"] == "unsupported"
    assert result["requires_retrieval"] is False


def test_query_returns_intent_metadata_for_conversation(monkeypatch):
    engine = RAGEngine(MagicMock(), use_hybrid=False)
    monkeypatch.setattr(engine, "trace_logger", MagicMock())

    import asyncio
    result = asyncio.run(engine.query("hello", user_id=1))

    assert result["intent"] == "conversation"
    assert result["intent_confidence"] >= 0.9
    assert result["searched_docs"] == []
    assert result["context_sufficient"] is True


def test_query_rejects_unsupported_without_retrieval(monkeypatch):
    engine = RAGEngine(MagicMock(), use_hybrid=False)
    monkeypatch.setattr(engine, "trace_logger", MagicMock())
    monkeypatch.setattr("services.rag_engine.list_all_documents", MagicMock(side_effect=AssertionError("should not retrieve")))

    import asyncio
    result = asyncio.run(engine.query("tell me a joke", user_id=1))

    assert result["intent"] == "unsupported"
    assert result["sources"] == []
    assert result["searched_docs"] == []
    assert result["context_sufficient"] is True
