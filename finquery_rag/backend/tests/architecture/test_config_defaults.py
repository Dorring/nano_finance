"""Verify that env var defaults match documented values.

These tests import services and check their class-level defaults and
os.getenv fallback values without requiring actual DB or model connections.
"""
import os
import inspect


def test_rag_engine_default_max_context_tokens():
    from src.services.rag_engine import RAGEngine
    assert RAGEngine.DEFAULT_MAX_CONTEXT_TOKENS == 1100


def test_rag_engine_default_max_new_tokens():
    from src.services.rag_engine import RAGEngine
    assert RAGEngine.DEFAULT_MAX_NEW_TOKENS == 512


def test_rag_engine_default_top_k():
    from src.services.rag_engine import RAGEngine
    assert RAGEngine.DEFAULT_TOP_K_CHUNKS == 3


def test_retrieval_max_search_limit():
    from src.services.retrieval import SqliteBM25Retriever
    assert 1 <= SqliteBM25Retriever.MAX_SEARCH_LIMIT <= 200


def test_session_manager_defaults():
    from src.services.session_manager import SessionManager
    assert SessionManager.DEFAULT_MAX_HISTORY == 8
    assert SessionManager.DEFAULT_TTL_SECONDS == 0
    assert SessionManager.MAX_SESSION_ID_LENGTH == 128
    assert SessionManager.MAX_CONTENT_CHARS == 20000


def test_intent_classification_default_to_retrieval():
    from src.services.intent import classify_query_intent
    result = classify_query_intent("some random unknown query text")
    assert result["intent"] == "document_qa"
    assert result["requires_retrieval"] is True


def test_intent_empty_query_returns_unsupported():
    from src.services.intent import classify_query_intent
    result = classify_query_intent("")
    assert result["intent"] == "unsupported"


def test_intent_greeting_returns_conversation():
    from src.services.intent import classify_query_intent
    result = classify_query_intent("Hello!")
    assert result["intent"] == "conversation"
    assert result["requires_retrieval"] is False


def test_intent_financial_keyword_triggers_document_qa():
    from src.services.intent import classify_query_intent
    result = classify_query_intent("What was the total revenue?")
    assert result["intent"] in ("document_qa", "financial_calculation")


def test_financial_tools_scale_factors_keys_present():
    from src.services.financial_tools import _SCALE_FACTORS
    from decimal import Decimal
    # Verify essential scale factors are present
    assert _SCALE_FACTORS.get("million") == Decimal("1000000")
    assert _SCALE_FACTORS.get("billion") == Decimal("1000000000")
    assert _SCALE_FACTORS.get("thousand") == Decimal("1000")
    # Chinese financial units
    assert _SCALE_FACTORS.get("万") == Decimal("10000")  # wan
    assert _SCALE_FACTORS.get("亿") == Decimal("100000000")  # yi
    assert _SCALE_FACTORS.get("万元") == Decimal("10000")  # wan yuan
    assert _SCALE_FACTORS.get("亿元") == Decimal("100000000")  # yi yuan


def test_answer_validation_tolerance_default():
    from src.services.answer_validation import validate_answer_calculations
    sig = inspect.signature(validate_answer_calculations)
    default = sig.parameters["tolerance_percent_points"].default
    assert str(default) == "0.05"


def test_eval_runner_n_results_bounds():
    from src.evaluation.eval_runner import EVAL_RUN_N_RESULTS_MIN, EVAL_RUN_N_RESULTS_MAX
    from src.evaluation.eval_runner import validate_n_results
    import pytest
    assert EVAL_RUN_N_RESULTS_MIN == 1
    assert EVAL_RUN_N_RESULTS_MAX == 20
    assert validate_n_results(5) == 5
    with pytest.raises(ValueError):
        validate_n_results(0)
    with pytest.raises(ValueError):
        validate_n_results(21)


def test_bm25_retriever_schema_version():
    from src.services.retrieval import SqliteBM25Retriever
    assert SqliteBM25Retriever.SCHEMA_VERSION == 2
