"""Phase 14A tests: document registry lifecycle listing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry


def test_list_all_is_tenant_scoped_and_ordered(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("doc1", 1, "a.pdf", "fh1", status="pending")
    reg.transition("doc1", "parsing")
    reg.mark_indexing("doc1")
    reg.mark_ready("doc1", 2, "ch1")
    reg.register("doc2", 1, "b.pdf", "fh2", status="pending")
    reg.mark_failed("doc2", "parse failed")
    reg.register("doc3", 2, "c.pdf", "fh3", status="pending")

    rows = reg.list_all(1)

    assert [row["document_id"] for row in rows] == ["doc2", "doc1"]
    assert {row["tenant_id"] for row in rows} == {1}


def test_list_all_filters_by_status(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("ready", 1, "ready.pdf", "fh1", status="pending")
    reg.transition("ready", "parsing")
    reg.mark_indexing("ready")
    reg.mark_ready("ready", 1, "ch1")
    reg.register("failed", 1, "failed.pdf", "fh2", status="pending")
    reg.mark_failed("failed", "bad pdf")

    rows = reg.list_all(1, status="failed")

    assert len(rows) == 1
    assert rows[0]["document_id"] == "failed"
    assert rows[0]["error_message"] == "bad pdf"


def test_list_all_fail_closed_for_missing_tenant_or_invalid_status(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("doc1", 1, "a.pdf", "fh1", status="pending")

    assert reg.list_all(None) == []
    assert reg.list_all(1, status="unknown") == []


def test_status_counts_are_tenant_scoped_and_zero_filled(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("doc1", 1, "a.pdf", "fh1", status="pending")
    reg.register("doc2", 1, "b.pdf", "fh2", status="pending")
    reg.mark_failed("doc2", "bad")
    reg.register("doc3", 2, "c.pdf", "fh3", status="pending")

    counts = reg.status_counts(1)

    assert counts["pending"] == 1
    assert counts["failed"] == 1
    assert counts["ready"] == 0
    assert reg.status_counts(None) == {}
