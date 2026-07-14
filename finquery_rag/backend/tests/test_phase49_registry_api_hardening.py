"""Phase 49 tests: document registry listing API hardening."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry


def _register_ready(reg, document_id, tenant_id, filename):
    reg.register(document_id, tenant_id, filename, f"fh-{document_id}", status="pending")
    reg.transition(document_id, "parsing")
    reg.mark_indexing(document_id)
    reg.mark_ready(document_id, 1, f"ch-{document_id}")


def test_registry_list_all_supports_limit_and_offset(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    for i in range(5):
        _register_ready(reg, f"doc{i}", 1, f"doc{i}.pdf")

    rows = reg.list_all(1, limit=2, offset=1)

    assert len(rows) == 2
    assert [row["tenant_id"] for row in rows] == [1, 1]


def test_registry_list_all_pagination_is_tenant_scoped(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    for i in range(3):
        _register_ready(reg, f"t1-{i}", 1, f"t1-{i}.pdf")
        _register_ready(reg, f"t2-{i}", 2, f"t2-{i}.pdf")

    rows = reg.list_all(1, limit=10, offset=0)

    assert len(rows) == 3
    assert {row["tenant_id"] for row in rows} == {1}


def test_registry_list_all_fail_closed_for_bad_pagination(tmp_path):
    reg = DocumentRegistry(db_path=str(tmp_path / "registry.db"))
    _register_ready(reg, "doc1", 1, "doc1.pdf")

    assert reg.list_all(1, limit="bad") == []
    assert reg.list_all(1, offset="bad") == []
    assert reg.list_all(1, limit=-1) == []
    assert reg.list_all(1, offset=-5) == reg.list_all(1, offset=0)


def test_document_registry_endpoint_uses_normalized_pagination_static():
    root = os.path.join(os.path.dirname(__file__), "..")
    main = open(os.path.join(root, "src", "main.py"), encoding="utf-8").read()
    block = main[main.index('@app.get("/document-registry")'):main.index('@app.get("/traces")')]

    assert "limit: int = 50" in block
    assert "offset: int = 0" in block
    assert "_normalize_api_pagination(limit, offset, default_limit=50)" in block
    assert "limit=normalized_limit" in block
    assert "offset=normalized_offset" in block
    assert '"total_returned": len(rows)' in block
    assert '"limit": normalized_limit' in block
    assert '"offset": normalized_offset' in block
