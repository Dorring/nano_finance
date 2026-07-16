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

from src.services.document_registry import DocumentRegistry
from src.services.rag_engine import RAGEngine


class _DummyLLM:
    pass


def test_chinese_title_query_expands_to_english_terms(tmp_path):
    engine = RAGEngine(_DummyLLM(), use_hybrid=False, bm25_db_path=str(tmp_path / "b.db"))

    expanded = engine._expand_retrieval_query("这篇论文的标题是什么？")

    assert "这篇论文的标题是什么？" in expanded
    assert "paper title" in expanded
    assert "title of this paper" in expanded


def test_front_matter_query_boosts_page_one_chunks(tmp_path):
    engine = RAGEngine(_DummyLLM(), use_hybrid=False, bm25_db_path=str(tmp_path / "b.db"))
    chunks = [
        {"doc_id": "d::p2", "content": "table", "metadata": {"page": 2}, "score": 0.03},
        {"doc_id": "d::p1", "content": "paper title", "metadata": {"page": 1}, "score": 0.015},
    ]

    boosted = engine._boost_front_matter_chunks("What is the title of this paper?", chunks)

    assert boosted[0]["metadata"]["page"] == 1
    assert boosted[0]["score"] == 0.035
    assert boosted[0]["front_matter_boost"] == 0.02


def test_retrieve_single_document_uses_expanded_query_and_boost(monkeypatch, tmp_path):
    captured = {}

    def fake_query_collection(query_text, doc_name, n_results, user_id):
        captured["query_text"] = query_text
        return [
            {"doc_id": "doc::p2", "content": "table", "metadata": {"page": 2}, "score": 0.03},
            {"doc_id": "doc::p1", "content": "title", "metadata": {"page": 1}, "score": 0.015},
        ]

    monkeypatch.setattr("src.services.rag_engine.query_collection", fake_query_collection)
    engine = RAGEngine(_DummyLLM(), use_hybrid=False, bm25_db_path=str(tmp_path / "b.db"))

    results = engine.retrieve_single_document("paper.pdf", "这篇论文的标题是什么？", user_id=1, n_results=2)

    assert "paper title" in captured["query_text"]
    assert results[0]["metadata"]["page"] == 1


def test_mark_ready_persists_page_count(tmp_path):
    registry = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    doc_id = "doc-1"
    registry.register(
        doc_id,
        tenant_id=1,
        filename="paper.pdf",
        file_hash="hash",
        status="indexing",
    )

    registry.mark_ready(doc_id, chunk_count=10, content_hash="content", page_count=19)
    row = registry.get_latest_version(1, "paper.pdf")

    assert row["status"] == "ready"
    assert row["chunk_count"] == 10
    assert row["page_count"] == 19