"""Smoke-check an OpenAI-compatible chat-completions adapter.

This script is intentionally dependency-light and does not import FinQuery app
state. It validates the model-service boundary used by FinQuery's OpenAI SDK
client before a backend/frontend demo.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_BASE_URL = "http://127.0.0.1:8500/v1"
DEFAULT_MODEL = "finquery-finance-sft1147"
DEFAULT_API_KEY = "not-needed-for-local"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a local OpenAI-compatible LLM adapter")
    parser.add_argument("--base-url", default=os.getenv("LLM_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--prompt", default="Answer briefly: what is revenue growth?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--stream", action="store_true", help="Also validate streaming deltas")
    parser.add_argument("--out", help="Optional JSON report path")
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    results = [
        check_models(base_url, args.api_key, args.timeout),
        check_chat_completion(
            base_url,
            args.api_key,
            args.model,
            args.prompt,
            args.max_tokens,
            args.timeout,
            stream=False,
        ),
    ]
    if args.stream:
        results.append(
            check_chat_completion(
                base_url,
                args.api_key,
                args.model,
                args.prompt,
                args.max_tokens,
                args.timeout,
                stream=True,
            )
        )

    report = {
        "passed": all(result.ok for result in results),
        "base_url": base_url,
        "model": args.model,
        "checks": [result.to_dict() for result in results],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
    return 0 if report["passed"] else 1


def check_models(base_url: str, api_key: str, timeout: float) -> CheckResult:
    try:
        status, payload = _request_json(
            "GET",
            f"{base_url}/models",
            api_key=api_key,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - detail depends on platform network errors
        return CheckResult("models", False, str(exc))
    if status < 200 or status >= 300:
        return CheckResult("models", False, f"HTTP {status}")
    data = payload.get("data")
    if not isinstance(data, list):
        return CheckResult("models", False, "response missing data list")
    return CheckResult("models", True, f"{len(data)} model(s) listed")


def check_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    *,
    stream: bool,
) -> CheckResult:
    request_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if stream:
        request_payload["stream"] = True
        return _check_streaming_chat(base_url, api_key, request_payload, timeout)
    try:
        status, payload = _request_json(
            "POST",
            f"{base_url}/chat/completions",
            api_key=api_key,
            timeout=timeout,
            payload=request_payload,
        )
    except Exception as exc:  # pragma: no cover - detail depends on platform network errors
        return CheckResult("chat", False, str(exc))
    if status < 200 or status >= 300:
        return CheckResult("chat", False, f"HTTP {status}")
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return CheckResult("chat", False, "response missing choices[0].message.content")
    if not str(content).strip():
        return CheckResult("chat", False, "empty message content")
    return CheckResult("chat", True, f"received {len(str(content))} character(s)")


def _check_streaming_chat(base_url: str, api_key: str, payload: dict[str, object], timeout: float) -> CheckResult:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            if status < 200 or status >= 300:
                return CheckResult("stream", False, f"HTTP {status}")
            saw_delta = False
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                delta = event.get("choices", [{}])[0].get("delta", {})
                if str(delta.get("content", "")).strip():
                    saw_delta = True
            if not saw_delta:
                return CheckResult("stream", False, "no choices[0].delta.content received")
            return CheckResult("stream", True, "received streaming delta content")
    except Exception as exc:  # pragma: no cover - detail depends on platform network errors
        return CheckResult("stream", False, str(exc))


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    timeout: float,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        return exc.code, payload


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))