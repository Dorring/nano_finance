"""Phase 87 tests: BM25 operations hardening."""
import json
import os
import sqlite3
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "jieba_fast" not in sys.modules:
    jieba = types.SimpleNamespace(cut_for_search=lambda text: text.split())
    sys.modules["jieba_fast"] = jieba

from services.retrieval import SqliteBM25Retriever
from src.evaluation.eval_cli import main as eval_cli_main


def _chunk(doc_id, content="alpha revenue", doc_name="r.pdf"):
    return {
        "content": content,
        "metadata": {"doc_id": doc_id, "doc_name": doc_name, "page": 1, "type": "text"},
    }


def test_integrity_report_includes_scope_issue_and_truncation_fields(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)

    report = retriever.integrity_report(user_id=1)

    assert report["scope"] == "tenant"
    assert report["user_id"] == 1
    assert report["global_orphan_check"] is False
    assert report["issue_count"] == 0
    assert report["missing_doc_ids_truncated"] is False
    assert report["duplicate_doc_ids_truncated"] is False
    assert report["orphan_doc_ids_truncated"] is False


def test_integrity_report_truncates_long_problem_lists_but_keeps_counts(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk(f"r.pdf::{i}") for i in range(55)], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute("DELETE FROM fts_index")
        conn.commit()

    report = retriever.integrity_report(user_id=1)

    assert report["ok"] is False
    assert report["missing_fts_count"] == 55
    assert report["issue_count"] == 55
    assert len(report["missing_doc_ids"]) == 50
    assert report["missing_doc_ids_truncated"] is True


def test_rebuild_report_includes_operation_counts_global(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1"), _chunk("r.pdf::2")], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("orphan", "orphan::1"))
        conn.commit()

    report = retriever.rebuild_fts_index()

    assert report["ok"] is True
    assert report["scope"] == "global"
    assert report["rebuild"] == {
        "scope": "global",
        "user_id": None,
        "deleted_fts_rows": 3,
        "rebuilt_fts_rows": 2,
    }


def test_rebuild_report_includes_operation_counts_tenant_scope(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("a.pdf::1", doc_name="a.pdf")], user_id=1)
    retriever.add_chunks([_chunk("b.pdf::1", doc_name="b.pdf")], user_id=2)

    report = retriever.rebuild_fts_index(user_id=1)

    assert report["ok"] is True
    assert report["scope"] == "tenant"
    assert report["global_orphan_check"] is False
    assert report["rebuild"]["deleted_fts_rows"] == 1
    assert report["rebuild"]["rebuilt_fts_rows"] == 1
    assert retriever.search("revenue", user_id=2)


def test_eval_cli_bm25_rejects_invalid_optional_user_id(tmp_path, capsys):
    db_path = str(tmp_path / "bm25.db")

    assert eval_cli_main(["bm25-check", "--db", db_path, "--user-id", "0"]) == 2
    captured = capsys.readouterr()
    assert "user-id must be >= 1" in captured.err
    assert captured.out == ""

    assert eval_cli_main(["bm25-rebuild", "--db", db_path, "--user-id", "0"]) == 2
    captured = capsys.readouterr()
    assert "user-id must be >= 1" in captured.err
    assert captured.out == ""


def test_eval_cli_bm25_writes_report_atomically_with_new_fields(tmp_path, capsys):
    db_path = str(tmp_path / "bm25.db")
    out = tmp_path / "nested" / "bm25.json"
    retriever = SqliteBM25Retriever(db_path=db_path)
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)

    code = eval_cli_main(["bm25-check", "--db", db_path, "--user-id", "1", "--out", str(out)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert code == 0
    assert payload["scope"] == "tenant"
    assert payload["issue_count"] == 0
    assert json.loads(out.read_text(encoding="utf-8"))["global_orphan_check"] is False
    assert list(out.parent.glob(f".{out.name}.*.tmp")) == []
