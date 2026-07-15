from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import sys
import threading
from pathlib import Path


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "finquery-finance-sft1147"}]})
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                b"data: [DONE]\n\n"
            )
            return
        self._json({"choices": [{"message": {"content": "hello from adapter"}}]})

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_llm_adapter.py"
    spec = importlib.util.spec_from_file_location("check_llm_adapter", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_llm_adapter_validates_models_chat_and_stream(tmp_path):
    module = _load_script()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = tmp_path / "llm_adapter.json"
        code = module.main([
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--model",
            "finquery-finance-sft1147",
            "--api-key",
            "not-needed-for-local",
            "--stream",
            "--out",
            str(report),
        ])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["passed"] is True
    assert [check["name"] for check in payload["checks"]] == ["models", "chat", "stream"]
    assert all(check["ok"] for check in payload["checks"])