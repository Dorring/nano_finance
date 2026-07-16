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
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
import asyncio
import os

from src.services.rag_engine import RAGEngine


class _Choice:
    def __init__(self, content):
        self.message = type("Message", (), {"content": content})()


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _MockLLMClient:
    def __init__(self, rewrite_response="What was Q3 revenue?"):
        self.rewrite_response = rewrite_response
        self.calls = 0
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {"create": self._create})()

    def _create(self, **kwargs):
        self.calls += 1
        return _Response(self.rewrite_response)


def _engine(client=None, tmp_path=None):
    path = os.path.join(str(tmp_path), "bm25.db") if tmp_path else ":memory:"
    return RAGEngine(client or _MockLLMClient(), bm25_db_path=path)


def test_standalone_document_question_does_not_use_history_for_rewrite(tmp_path):
    client = _MockLLMClient("User: old question Assistant: old answer")
    engine = _engine(client, tmp_path)
    history = [
        {"role": "user", "content": "文档有几页？"},
        {"role": "assistant", "content": "[ECCV_2026.pdf, p2]"},
    ]

    rewritten = asyncio.run(engine._rewrite_query_with_context("这篇论文的标题是什么？", history))

    assert rewritten == "这篇论文的标题是什么？"
    assert client.calls == 0


def test_bad_rewrite_with_role_labels_is_rejected(tmp_path):
    client = _MockLLMClient("[ECCV_2026.pdf, p2] User: 文档有几页？ Assistant: [ECCV_2026.pdf, p2]")
    engine = _engine(client, tmp_path)
    history = [
        {"role": "user", "content": "Tell me about Q3"},
        {"role": "assistant", "content": "Q3 was strong."},
    ]

    rewritten = asyncio.run(engine._rewrite_query_with_context("What about revenue?", history))

    assert rewritten == "What about revenue?"
    assert client.calls == 1


def test_clean_followup_rewrite_is_kept(tmp_path):
    client = _MockLLMClient("What was Q3 revenue in the report?")
    engine = _engine(client, tmp_path)
    history = [
        {"role": "user", "content": "Tell me about Q3"},
        {"role": "assistant", "content": "Q3 was strong."},
    ]

    rewritten = asyncio.run(engine._rewrite_query_with_context("What about revenue?", history))

    assert rewritten == "What was Q3 revenue in the report?"


def test_rrf_two_percent_confidence_is_insufficient(tmp_path):
    engine = _engine(tmp_path=tmp_path)
    chunks = [
        {"doc_id": "d::1", "content": "weak", "metadata": {}, "score": 0.02},
        {"doc_id": "d::2", "content": "weak", "metadata": {}, "score": 0.015},
    ]

    sufficient, best, avg = engine._check_context_sufficiency(chunks)

    assert sufficient is False
    assert best == 0.02
    assert avg < 0.025


def test_rrf_three_percent_confidence_remains_sufficient(tmp_path):
    engine = _engine(tmp_path=tmp_path)
    chunks = [{"doc_id": "d::1", "content": "ok", "metadata": {}, "score": 0.03}]

    sufficient, best, avg = engine._check_context_sufficiency(chunks)

    assert sufficient is True
    assert best == 0.03