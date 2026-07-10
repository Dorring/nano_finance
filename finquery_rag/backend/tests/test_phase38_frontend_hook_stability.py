"""Frontend hook stability regression checks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_dashboard_fetch_documents_is_stable_callback_before_effects():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "useCallback" in content
    assert "const fetchDocuments = useCallback" in content
    assert content.index("const fetchDocuments = useCallback") < content.index("useEffect(() => {\n    fetchDocuments();")
    assert "}, [fetchDocuments]);" in content
    assert "}, [fetchDocuments, hasProcessingDocuments]);" in content


def test_dashboard_selected_docs_updates_are_functional():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "setSelectedDocs((current) => current.filter((name) => readyNames.has(name)))" in content
    assert "setSelectedDocs((current) => current.filter((name) => name !== docName))" in content
    assert "setSelectedDocs((current) => {" in content
    assert "return [...current, docName];" in content
    assert "setSelectedDocs(selectedDocs.filter" not in content
    assert "setSelectedDocs([...selectedDocs" not in content
