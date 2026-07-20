"""Verify key services can be constructed with default parameters.

These tests verify that service constructors accept default arguments
without throwing exceptions. They do NOT require actual DB or model
connections - they test the constructor signatures and lightweight
initialization paths.
"""
import os
import tempfile


def _close_sqlite_conns(obj):
    """Close any SQLite connections held by an object before cleanup."""
    if hasattr(obj, "_local"):
        local = obj._local
        if hasattr(local, "conn") and local.conn is not None:
            try:
                local.conn.close()
            except Exception:
                pass
            local.conn = None


def test_rag_engine_constructor_accepts_defaults():
    from src.services.rag_engine import RAGEngine
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    engine = RAGEngine(llm_client=mock_client, model_name="test-model", use_hybrid=False)
    assert engine.model_name == "test-model"
    assert engine.use_hybrid is False
    assert engine.max_context_tokens == 1100
    assert engine.max_new_tokens == 512


def test_rag_engine_constructor_with_options():
    from src.services.rag_engine import RAGEngine
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    engine = RAGEngine(
        llm_client=mock_client, model_name="custom-model",
        use_hybrid=True, max_context_tokens=800, max_new_tokens=256,
    )
    assert engine.max_context_tokens == 800
    assert engine.max_new_tokens == 256


def test_session_manager_constructor():
    from src.services.session_manager import SessionManager
    db_path = os.path.join(tempfile.gettempdir(), "arch_test_sessions.db")
    try:
        manager = SessionManager(db_path=db_path)
        assert manager.max_history == 8
        assert manager.ttl_seconds == 0
        _close_sqlite_conns(manager)
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


def test_feedback_store_constructor():
    from src.services.feedback import FeedbackStore
    db_path = os.path.join(tempfile.gettempdir(), "arch_test_feedback.db")
    try:
        store = FeedbackStore(db_path=db_path)
        assert store.db_path == db_path
        _close_sqlite_conns(store)
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


def test_document_registry_constructor():
    from src.services.document_registry import DocumentRegistry
    db_path = os.path.join(tempfile.gettempdir(), "arch_test_registry.db")
    try:
        registry = DocumentRegistry(db_path=db_path)
        assert registry.db_path == db_path
        _close_sqlite_conns(registry)
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


def test_trace_logger_constructor():
    from src.services.trace import TraceLogger
    db_path = os.path.join(tempfile.gettempdir(), "arch_test_trace.db")
    try:
        logger = TraceLogger(db_path=db_path, sample_rate=0.0)
        assert logger.db_path == db_path
        assert logger.sample_rate == 0.0
        _close_sqlite_conns(logger)
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


def test_heuristic_reranker_constructor():
    from src.services.reranker import HeuristicReranker, NoopReranker

    reranker = HeuristicReranker()
    assert reranker.name == "heuristic"
    assert reranker.original_score_weight == 0.7
    assert reranker.lexical_weight == 0.3

    noop = NoopReranker()
    assert noop.name == "noop"


def test_memory_profile_store_constructor():
    from src.services.memory_profile import UserMemoryStore
    db_path = os.path.join(tempfile.gettempdir(), "arch_test_memory.db")
    try:
        store = UserMemoryStore(db_path=db_path)
        assert store.db_path == db_path
        _close_sqlite_conns(store)
    finally:
        for suffix in ("", "-shm", "-wal"):
            try:
                os.unlink(db_path + suffix)
            except OSError:
                pass


def test_build_reranker_factory():
    from src.services.reranker import build_reranker, HeuristicReranker, NoopReranker

    # Per build_reranker spec: None and "none" return None (disabled)
    assert build_reranker(None) is None
    assert build_reranker("none") is None
    assert build_reranker("off") is None
    assert build_reranker("disabled") is None

    # "noop" returns NoopReranker
    noop = build_reranker("noop")
    assert isinstance(noop, NoopReranker)

    # "heuristic" returns HeuristicReranker
    heuristic = build_reranker("heuristic")
    assert isinstance(heuristic, HeuristicReranker)


def test_query_scope_resolution():
    from src.services.query_scope import resolve_query_document_names

    names, invalid = resolve_query_document_names(None, ["doc_a", "doc_b"])
    assert names == ["doc_a", "doc_b"]
    assert invalid == []

    names, invalid = resolve_query_document_names(["doc_a"], ["doc_a", "doc_b"])
    assert names == ["doc_a"]
    assert invalid == []

    names, invalid = resolve_query_document_names(["doc_c"], ["doc_a", "doc_b"])
    assert names == []
    assert invalid == ["doc_c"]


def test_rag_engine_no_supporting_source_in_init():
    """Verify RAGEngine.__init__ does not reference supporting_source_page."""
    import inspect
    from src.services.rag_engine import RAGEngine
    source = inspect.getsource(RAGEngine.__init__)
    assert "supporting_source_page" not in source, (
        "RAGEngine.__init__ must not reference supporting_source_page metadata"
    )
