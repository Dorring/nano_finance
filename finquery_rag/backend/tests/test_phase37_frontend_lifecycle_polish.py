"""Frontend document lifecycle UI regression checks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_dashboard_polls_while_documents_are_processing():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "hasProcessingDocuments" in content
    assert "window.setInterval" in content
    assert "fetchDocuments({ silent: true })" in content
    assert "['pending', 'parsing', 'indexing'].includes(doc.status)" in content


def test_dashboard_drops_unready_selected_documents_after_refresh():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "const readyNames = new Set" in content
    assert "setSelectedDocs((current) => current.filter((name) => readyNames.has(name)))" in content
    assert "wait until it is ready" in content


def test_sidebar_shows_lifecycle_status_summary_and_disabled_hint():
    content = (FRONTEND / "components" / "Sidebar.jsx").read_text(encoding="utf-8")

    assert "documents-status-summary" in content
    assert "processingDocuments.length" in content
    assert "failedDocuments.length" in content
    assert "aria-disabled={!isSelectable}" in content
    assert "title={isSelectable ? `Select ${doc.name}` : `${doc.name} is ${status}`}" in content


def test_lifecycle_status_summary_has_styles():
    content = (FRONTEND / "App.css").read_text(encoding="utf-8")

    assert ".documents-status-summary" in content
    assert ".documents-status-summary span" in content
