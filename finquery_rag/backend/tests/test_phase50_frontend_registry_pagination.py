"""Phase 50 tests: frontend registry pagination compatibility."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_api_exposes_paginated_and_all_registry_helpers():
    content = (FRONTEND / "api.js").read_text(encoding="utf-8")

    assert "export const listDocumentRegistry = async (status = null, { limit = 100, offset = 0 } = {})" in content
    assert "const params = { limit, offset };" in content
    assert "export const listAllDocumentRegistry" in content
    assert "while (true)" in content
    assert "offset += totalReturned" in content
    assert "totalReturned < pageSize" in content


def test_dashboard_uses_all_registry_pages():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "listAllDocumentRegistry" in content
    assert "listDocumentRegistry" not in content
    assert "const registryData = await listAllDocumentRegistry();" in content


def test_dashboard_pdf_upload_check_is_case_insensitive():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "file.name.toLowerCase().endsWith('.pdf')" in content
    assert "file.name.endsWith('.pdf')" not in content
