"""Characterize current RAGEngine trace behavior.

These tests record the exact trace fields and structure
so that extraction preserves trace compatibility.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.services.rag_engine import RAGEngine


def _make_engine():
    return RAGEngine(llm_client=None, model_name="test", use_hybrid=True)


class TestTraceCharacterization:
    """Record current trace-related behavior."""

    def test_engine_has_trace_logger(self):
        engine = _make_engine()
        assert engine.trace_logger is not None

    def test_engine_has_retrieval_debug(self):
        engine = _make_engine()
        assert hasattr(engine, "_last_retrieval_debug")
        debug = engine._last_retrieval_debug
        assert "candidate_count" in debug
        assert "returned_count" in debug

    def test_retrieval_debug_initial_values(self):
        engine = _make_engine()
        debug = engine._last_retrieval_debug
        assert debug["candidate_count"] == 0
        assert debug["returned_count"] == 0

    def test_make_retrieval_debug_keys(self):
        engine = _make_engine()
        debug = engine._make_retrieval_debug(5, 3)
        assert "candidate_count" in debug
        assert "returned_count" in debug
