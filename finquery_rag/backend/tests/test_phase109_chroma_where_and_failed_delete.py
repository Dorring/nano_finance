import ast
from pathlib import Path

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_collection(monkeypatch):
    store = {"ids": [], "documents": [], "metadatas": []}

    def _match(meta, wf):
        if wf is None:
            return True
        if "$and" in wf:
            return all(_match(meta, clause) for clause in wf["$and"])
        if "$or" in wf:
            return any(_match(meta, clause) for clause in wf["$or"])
        for key, value in wf.items():
            if key.startswith("$"):
                continue
            if isinstance(value, dict) and "$in" in value:
                if meta.get(key) not in value["$in"]:
                    return False
            elif meta.get(key) != value:
                return False
        return True

    collection = MagicMock()

    def _upsert(ids, documents, metadatas=None):
        for idx, doc_id in enumerate(ids):
            store["ids"].append(doc_id)
            store["documents"].append(documents[idx])
            store["metadatas"].append(metadatas[idx] if metadatas else {})

    def _query(query_texts, n_results=5, where=None):
        matched = [i for i, meta in enumerate(store["metadatas"]) if _match(meta, where)][:n_results]
        return {
            "ids": [[store["ids"][i] for i in matched]],
            "documents": [[store["documents"][i] for i in matched]],
            "metadatas": [[store["metadatas"][i] for i in matched]],
            "distances": [[0.1 for _ in matched]],
        }

    def _get(include=None, where=None, limit=None):
        matched = [i for i, meta in enumerate(store["metadatas"]) if _match(meta, where)]
        if limit is not None:
            matched = matched[:limit]
        return {
            "ids": [store["ids"][i] for i in matched],
            "metadatas": [store["metadatas"][i] for i in matched],
        }

    def _delete(where=None):
        keep = [i for i, meta in enumerate(store["metadatas"]) if not _match(meta, where)]
        store["ids"] = [store["ids"][i] for i in keep]
        store["documents"] = [store["documents"][i] for i in keep]
        store["metadatas"] = [store["metadatas"][i] for i in keep]

    collection.upsert.side_effect = _upsert
    collection.query.side_effect = _query
    collection.get.side_effect = _get
    collection.delete.side_effect = _delete
    collection.count.side_effect = lambda: len(store["ids"])
    collection.name = "rag_global_knowledge_base"
    monkeypatch.setattr("src.services.vector_store.get_or_create_collection", lambda: collection)
    return collection
from src.services.vector_store import (
    _tenant_doc_where,
    _tenant_docs_where,
    add_documents,
    delete_document_collection,
    query_collection,
)


def test_chroma_single_document_filter_uses_and_operator():
    assert _tenant_doc_where(1, "report.pdf") == {
        "$and": [{"user_id": 1}, {"doc_name": "report.pdf"}]
    }


def test_chroma_multi_document_filter_uses_and_with_in_operator():
    assert _tenant_docs_where(1, ["a.pdf", "b.pdf"]) == {
        "$and": [{"user_id": 1}, {"doc_name": {"$in": ["a.pdf", "b.pdf"]}}]
    }


def test_single_document_query_and_delete_work_with_chroma_and_filter(mock_collection):
    add_documents([
        {"content": "alpha", "metadata": {"doc_id": "a1", "doc_name": "a.pdf"}},
        {"content": "beta", "metadata": {"doc_id": "b1", "doc_name": "b.pdf"}},
    ], "a.pdf", user_id=1)
    add_documents([
        {"content": "gamma", "metadata": {"doc_id": "c1", "doc_name": "a.pdf"}},
    ], "a.pdf", user_id=2)

    results = query_collection("alpha", doc_name="a.pdf", user_id=1, n_results=10)

    assert results
    assert {item["metadata"]["user_id"] for item in results} == {1}
    assert {item["metadata"]["doc_name"] for item in results} == {"a.pdf"}

    assert delete_document_collection("a.pdf", user_id=1) is True
    remaining_user_1 = query_collection("alpha", user_id=1, n_results=10)
    remaining_user_2 = query_collection("gamma", user_id=2, n_results=10)
    assert remaining_user_1 == []
    assert len(remaining_user_2) == 1


def test_delete_endpoint_allows_registry_only_failed_documents_static():
    main_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    delete_func = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "delete_document")
    block = ast.get_source_segment(source, delete_func)

    assert "registry_deleted = document_registry.delete(current_user.id, doc_name)" in block
    assert "if not vector_deleted and not registry_deleted" in block
    assert "registry_rows" in block
    assert "bm25_cleanup_attempted" in block