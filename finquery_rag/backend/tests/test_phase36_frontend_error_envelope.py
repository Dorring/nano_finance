"""Frontend API error envelope compatibility checks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_api_exports_error_message_parser():
    content = (FRONTEND / "api.js").read_text(encoding="utf-8")

    assert "export const getApiErrorMessage" in content
    assert "detail.message || detail.error_code" in content
    assert "error.userMessage = getApiErrorMessage(error)" in content
    assert "return getApiErrorMessage(payload, `HTTP error: ${response.status}`)" in content


def test_auth_pages_use_normalized_api_error_message():
    login = (FRONTEND / "pages" / "Login.jsx").read_text(encoding="utf-8")
    register = (FRONTEND / "pages" / "Register.jsx").read_text(encoding="utf-8")

    assert "getApiErrorMessage" in login
    assert "error.userMessage || getApiErrorMessage(error, 'Login failed')" in login
    assert "getApiErrorMessage" in register
    assert "error.userMessage || getApiErrorMessage(error, 'Registration failed')" in register
    assert "error.response?.data?.detail" not in login
    assert "error.response?.data?.detail" not in register


def test_dashboard_uses_normalized_api_error_message():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "getApiErrorMessage" in content
    assert "getApiErrorMessage(error, 'Failed to load documents')" in content
    assert "getApiErrorMessage(error, `Failed to upload ${file.name}`)" in content
    assert "getApiErrorMessage(error, `Failed to delete ${docName}`)" in content
    assert "getApiErrorMessage(error, 'Failed to get response')" in content
