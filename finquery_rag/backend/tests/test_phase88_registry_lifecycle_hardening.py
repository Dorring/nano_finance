"""Phase 88 tests: document registry lifecycle hardening."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry, MAX_ERROR_MESSAGE_CHARS


def _register_ready(reg, document_id, tenant_id, filename):
    reg.register(document_id, tenant_id, filename, f"fh-{document_id}", status="pending")
    reg.transition(document_id, "parsing")
    reg.mark_indexing(document_id)
    reg.mark_ready(document_id, 1, f"ch-{document_id}")


def test_count_all_matches_filter_and_is_tenant_scoped(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    _register_ready(reg, "r1", 1, "r1.pdf")
    reg.register("f1", 1, "f1.pdf", "fh-f1", status="pending")
    reg.mark_failed("f1", "bad")
    _register_ready(reg, "r2", 2, "r2.pdf")

    assert reg.count_all(1) == 2
    assert reg.count_all(1, status="ready") == 1
    assert reg.count_all(1, status="failed") == 1
    assert reg.count_all(2) == 1
    assert reg.count_all(None) == 0
    assert reg.count_all(1, status="unknown") == 0


def test_status_summary_zero_filled_and_aggregated(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("p", 1, "p.pdf", "fh-p", status="pending")
    reg.register("x", 1, "x.pdf", "fh-x", status="pending")
    reg.mark_failed("x", "bad")
    _register_ready(reg, "r", 1, "r.pdf")

    summary = reg.status_summary(1)

    assert summary["total"] == 3
    assert summary["active"] == 1
    assert summary["ready"] == 1
    assert summary["failed"] == 1
    assert summary["counts"]["pending"] == 1
    assert summary["counts"]["parsing"] == 0
    assert reg.status_summary(None) == {"counts": {}, "total": 0, "active": 0, "ready": 0, "failed": 0}


def test_failed_retry_listing_is_paginated_ordered_and_tenant_scoped(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    for i in range(3):
        reg.register(f"f{i}", 1, f"f{i}.pdf", f"fh-f{i}", status="pending")
        reg.mark_failed(f"f{i}", f"bad {i}")
    reg.register("other", 2, "other.pdf", "fh-other", status="pending")
    reg.mark_failed("other", "bad other")

    rows = reg.get_pending_for_retry(tenant_id=1, limit=2, offset=1)

    assert len(rows) == 2
    assert {row["tenant_id"] for row in rows} == {1}
    assert all(row["status"] == "failed" for row in rows)
    assert reg.get_pending_for_retry(tenant_id=1, limit="bad") == []


def test_error_messages_are_cleaned_and_truncated(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    reg.register("doc", 1, "doc.pdf", "fh", status="pending")
    reg.mark_failed("doc", "  bad\x00\n" + "x" * (MAX_ERROR_MESSAGE_CHARS + 20))

    row = reg.list_all(1, status="failed")[0]

    assert "\x00" not in row["error_message"]
    assert "\n" not in row["error_message"]
    assert len(row["error_message"]) == MAX_ERROR_MESSAGE_CHARS


def test_register_rejects_invalid_status_and_required_fields(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))

    with pytest.raises(ValueError, match="Invalid document status"):
        reg.register("doc", 1, "doc.pdf", "fh", status="unknown")
    with pytest.raises(ValueError, match="tenant_id is required"):
        reg.register("doc", None, "doc.pdf", "fh")
    with pytest.raises(ValueError, match="document_id, filename, and file_hash are required"):
        reg.register("", 1, "doc.pdf", "fh")


def test_document_registry_endpoint_exposes_real_total_and_has_more_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    block = main[main.index('@app.get("/document-registry")'):main.index('@app.get("/traces")')]

    assert "total_documents = document_registry.count_all(current_user.id, status=status)" in block
    assert '"total_documents": total_documents' in block
    assert '"has_more": normalized_offset + len(rows) < total_documents' in block
    assert '"status_summary": document_registry.status_summary(current_user.id)' in block
