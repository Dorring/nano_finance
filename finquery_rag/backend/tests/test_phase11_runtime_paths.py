"""Phase 11A tests: runtime storage path configuration."""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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

from services.document_registry import DocumentRegistry
from services.health import collect_config_snapshot, collect_health_snapshot
from services.rag_engine import RAGEngine
from services.trace import TraceLogger


def test_document_registry_uses_env_default_path(tmp_path, monkeypatch):
    db_path = tmp_path / "registry.db"
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", str(db_path))

    registry = DocumentRegistry()

    assert registry.db_path == str(db_path)
    assert db_path.exists()


def test_trace_logger_uses_env_default_path(tmp_path, monkeypatch):
    db_path = tmp_path / "trace.db"
    monkeypatch.setenv("TRACE_DB_PATH", str(db_path))

    logger = TraceLogger()

    assert logger.db_path == str(db_path)
    assert db_path.exists()


def test_rag_engine_uses_bm25_and_trace_env_paths(tmp_path, monkeypatch):
    bm25_path = tmp_path / "bm25.db"
    trace_path = tmp_path / "trace.db"
    monkeypatch.setenv("BM25_DB_PATH", str(bm25_path))
    monkeypatch.setenv("TRACE_DB_PATH", str(trace_path))

    engine = RAGEngine(MagicMock(), use_hybrid=False)

    assert engine.bm25_db_path == str(bm25_path)
    assert engine.trace_db_path == str(trace_path)
    assert engine.bm25_retriever.db_path == str(bm25_path)
    assert engine.trace_logger.db_path == str(trace_path)


def test_explicit_rag_engine_paths_override_env(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_DB_PATH", str(tmp_path / "env_bm25.db"))
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "env_trace.db"))
    bm25_path = tmp_path / "explicit_bm25.db"
    trace_path = tmp_path / "explicit_trace.db"

    engine = RAGEngine(
        MagicMock(),
        use_hybrid=False,
        bm25_db_path=str(bm25_path),
        trace_db_path=str(trace_path),
    )

    assert engine.bm25_db_path == str(bm25_path)
    assert engine.trace_db_path == str(trace_path)


def test_health_config_reads_runtime_env_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", str(tmp_path / "registry.db"))
    monkeypatch.setenv("BM25_DB_PATH", str(tmp_path / "bm25.db"))
    monkeypatch.setenv("SESSIONS_DB_PATH", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "trace.db"))

    cfg = collect_config_snapshot()

    assert cfg["storage"]["chroma_path"] == str(tmp_path / "chroma")
    assert cfg["storage"]["document_registry_db_path"] == str(tmp_path / "registry.db")
    assert cfg["storage"]["bm25_db_path"] == str(tmp_path / "bm25.db")
    assert cfg["storage"]["sessions_db_path"] == str(tmp_path / "sessions.db")
    assert cfg["storage"]["trace_db_path"] == str(tmp_path / "trace.db")


def test_health_snapshot_uses_env_defaults_when_instances_missing(tmp_path, monkeypatch):
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    monkeypatch.setenv("DOCUMENT_REGISTRY_DB_PATH", registry.db_path)
    monkeypatch.setenv("BM25_DB_PATH", str(tmp_path / "missing_bm25.db"))
    monkeypatch.setenv("TRACE_DB_PATH", str(tmp_path / "missing_trace.db"))

    snapshot = collect_health_snapshot()

    assert snapshot["checks"]["document_registry"]["path"] == registry.db_path
    assert snapshot["checks"]["bm25"]["path"] == str(tmp_path / "missing_bm25.db")
    assert snapshot["checks"]["trace"]["path"] == str(tmp_path / "missing_trace.db")
