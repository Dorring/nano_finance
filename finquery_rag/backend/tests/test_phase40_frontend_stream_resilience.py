"""Frontend streaming resilience regression checks."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend" / "src"


def test_stream_reader_requires_response_body_and_done_event():
    content = (FRONTEND / "api.js").read_text(encoding="utf-8")

    assert "if (!response.body)" in content
    assert "Streaming response body is not available" in content
    assert "let sawDoneEvent = false" in content
    assert "sawDoneEvent = true" in content
    assert "Streaming response ended before completion" in content


def test_stream_reader_surfaces_malformed_and_error_events():
    content = (FRONTEND / "api.js").read_text(encoding="utf-8")

    assert "throw new Error('Malformed streaming response')" in content
    assert "data.type === 'error'" in content
    assert "getApiErrorMessage(data, data.message || 'Streaming query failed')" in content
    assert "onToken(data.content || '')" in content


def test_dashboard_stream_error_path_updates_assistant_message_immutably():
    content = (FRONTEND / "pages" / "Dashboard.jsx").read_text(encoding="utf-8")

    assert "const fallback = error.message || 'Sorry, an error occurred while processing your question. Please try again.'" in content
    assert "if (lastMsg?.role !== 'assistant') return updated" in content
    assert "updated[updated.length - 1] = {" in content
    assert "streamError: fallback" in content
    assert "lastMsg.content = error.message" not in content
