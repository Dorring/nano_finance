"""Frontend query guardrail regression checks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_dashboard_blocks_queries_without_ready_documents():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "const readyDocumentCount = documents.filter" in content
    assert "const queryDisabledReason = readyDocumentCount === 0" in content
    assert "const isQueryDisabled = isLoading || readyDocumentCount === 0" in content
    assert "if (readyDocumentCount === 0)" in content
    assert "toast.error(queryDisabledReason || 'No ready documents are available')" in content


def test_dashboard_passes_query_guardrails_to_chat_and_input():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "queryDisabled={isQueryDisabled}" in content
    assert "queryDisabledReason={queryDisabledReason}" in content
    assert "disabled={isQueryDisabled}" in content
    assert "disabledReason={queryDisabledReason}" in content


def test_chat_area_disables_example_questions_when_queries_are_disabled():
    content = (FRONTEND / "components" / "ChatArea.jsx").read_text(encoding="utf-8")

    assert "queryDisabled" in content
    assert "queryDisabledReason" in content
    assert "query-disabled-notice" in content
    assert "disabled={queryDisabled}" in content
    assert "title={queryDisabledReason || question}" in content


def test_input_bar_surfaces_disabled_reason():
    content = (FRONTEND / "components" / "InputBar.jsx").read_text(encoding="utf-8")

    assert "disabledReason" in content
    assert "input-disabled-reason" in content
    assert "placeholder={disabledReason || placeholder}" in content
    assert "title={disabledReason || 'Send question'}" in content


def test_query_guardrail_styles_exist():
    content = (FRONTEND / "App.css").read_text(encoding="utf-8")

    assert ".query-disabled-notice" in content
    assert ".input-disabled-reason" in content
    assert ".example-button:disabled" in content
