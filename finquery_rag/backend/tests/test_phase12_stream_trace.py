"""Phase 12B tests: stream trace_id helpers."""
import json
import os
import sys
from unittest.mock import MagicMock

mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain", "langchain.schema",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.streaming import make_stream_done_event, safe_log_query_trace


def test_make_stream_done_event_includes_trace_id():
    event = make_stream_done_event(
        sources=[],
        context_sufficient=True,
        intent="document_qa",
        intent_confidence=0.82,
        trace_id="trace-stream",
    )

    assert event.startswith("data: ")
    payload = json.loads(event.removeprefix("data: ").strip())
    assert payload["type"] == "done"
    assert payload["trace_id"] == "trace-stream"


def test_safe_log_query_trace_returns_none_on_failure():
    engine = MagicMock()
    engine.trace_logger.log.side_effect = RuntimeError("trace failed")

    assert safe_log_query_trace(engine, {"tenant_id": 1, "query_original": "Q"}) is None


def test_safe_log_query_trace_returns_logger_trace_id():
    engine = MagicMock()
    engine.trace_logger.log.return_value = "trace-ok"

    trace_id = safe_log_query_trace(engine, {"tenant_id": 1, "query_original": "Q"})

    assert trace_id == "trace-ok"
    engine.trace_logger.log.assert_called_once()



def test_make_stream_done_event_has_stable_default_fields():
    event = make_stream_done_event()
    payload = json.loads(event.removeprefix("data: ").strip())

    assert payload == {
        "type": "done",
        "sources": [],
        "confidence": None,
        "context_sufficient": None,
        "intent": None,
        "intent_confidence": None,
        "trace_id": None,
    }
