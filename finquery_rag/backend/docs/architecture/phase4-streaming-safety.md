# Phase 4 Streaming Safety

This document describes the SSE streaming safety guarantees enforced by
the Phase 4 validation pipeline. Source of truth: the `/query/stream`
endpoint in `src/main.py` (`query_documents_stream`), the streaming
helpers in `src/services/streaming.py`, and the orchestrator in
`src/application/rag_orchestrator.py`.

## Streaming Architecture

The `/query/stream` endpoint does NOT stream raw LLM tokens to the
client. It calls `engine.query(...)` — which runs the full Phase 4
orchestrator (answerability → generation → validation → repair) — and
then emits the **final, post-validation** answer as a single SSE
`token` event, followed by a single `done` event. This is true for
calculation results (`EXECUTED`/`BLOCKED`/`FAILED`) and for non-calculation
results alike: the LLM stream is never invoked in the SSE path.

Because validation and repair complete before the first byte is sent,
the client can never receive a partial token from an answer that is
later blocked.

## Guarantees

### 1. Blocked answers are NOT streamed as partial LLM tokens

When the validation pipeline is enabled and the answer is `BLOCKED` or
`FAILED`, the safe fallback message is what `engine.query()` returns in
`result["answer"]`. The SSE handler emits that fallback as a single
`token` event. No partial LLM tokens are ever sent, because the LLM
stream is never invoked. The regression test
`test_sse_blocked_answer_emits_safe_message_as_token` asserts both that
the token content contains the safe refusal and that
`mock_engine.generate_answer_stream.assert_not_called()`.

### 2. Safe fallback is emitted as a single token event

The final answer (original, repaired, or fallback) is emitted in exactly
one SSE event:

```
data: {"type": "token", "content": "<final safe answer>"}\n\n
```

There is no token-by-token streaming of the answer in the current
implementation; the entire answer is one `token` event.

### 3. Error events never leak `str(exc)`

The streaming exception handler (`except Exception as exc:`) builds a
trace payload whose diagnostics record:

```python
"diagnostics": {
    "stream_error": True,
    "error_code": "STREAM_INTERNAL_ERROR",
    "exception_type": type(exc).__name__,
}
```

and whose `answer` and `error_message` fields are `None`. The SSE error
event sent to the client uses a generic, fixed message:

```
data: {
  "type": "error",
  "detail": {"error_code": "stream_error", "message": "Streaming query failed. Please retry."},
  "message": "Streaming query failed. Please retry.",
  "retryable": true,
  "trace_id": "..."
}\n\n
```

`str(exc)` is NEVER placed in the trace payload or in the SSE event.
The regression test `test_streaming_exception_does_not_save_str_exc`
asserts that `'"error_code": "STREAM_INTERNAL_ERROR"'` is present in
`main.py` and that `str(exc)` does not appear anywhere in the
`query_documents_stream` function body.

### 4. SSE `done` event includes `answerability`, `validation`, `repair` (when pipeline enabled)

When the validation pipeline produced these dicts, the `done` event
forwards them:

```python
if answerability_data is not None:
    done_kwargs["answerability"] = answerability_data
if validation_data is not None:
    done_kwargs["validation"] = validation_data
if repair_data is not None:
    done_kwargs["repair"] = repair_data
```

When the pipeline is disabled (`enable_validation_pipeline=False`), all
three are absent from the `done` event, preserving the legacy event
shape. The `done` event always carries the stable fields: `sources`,
`confidence`, `context_sufficient`, `intent`, `intent_confidence`,
`trace_id`, and `calculations`.

### 5. Session saves only the final safe answer (post-validation)

The SSE handler saves the assistant message to the session AFTER
`engine.query()` returns and BEFORE emitting the token event:

```python
result = await engine.query(...)
answer = result["answer"]
...
session_manager.add_message(session_id, current_user.id, "assistant", answer, ...)
```

`result["answer"]` is the post-validation answer (possibly repaired or
the safe fallback). The original LLM output — when it was blocked — is
never written to conversation history. This guarantees that subsequent
turns see only the safe answer, so a blocked answer cannot leak into
future context.

### 6. Trace does not leak blocked answer content

`engine.query()` runs the orchestrator, which sets `final_context=None`
and `answer=None` in `trace_data`. The trace stores only:

- `context_length`, `context_sha256` (first 16 hex chars)
- `answer_length`, `answer_sha256` (first 16 hex chars)
- `answerability.to_trace_dict()` (includes `best_score` /
  `average_score`, never the answer)
- `initial_validation.to_trace_dict()` and `validation.to_trace_dict()`
  (issue `message_hash` + `claim_excerpt` max 80 chars, never full
  message or full claim text)
- `repair.to_trace_dict()` (`was_repaired`, `fallback_used`,
  `repair_notes`, `answer_length`; never the answer text)

So even when the answer is blocked, neither the blocked answer text nor
the evidence context is recoverable from trace.

### 7. Internal fields never leak into the SSE `done` event

The `answerability` / `validation` / `repair` dicts forwarded to the
`done` event are produced by `to_public_dict()`:
- `answerability` omits `best_score` / `average_score`.
- `validation` issues expose only `code`, `severity`, `public_message`
  (no `message`, no `evidence_ids`).
- `repair` exposes only `was_repaired` / `fallback_used` (no
  `repair_notes`).

The regression test `test_sse_done_validation_excludes_internal_fields`
asserts `"message" not in issue`, `"evidence_ids" not in issue`,
`"best_score" not in done["answerability"]`, and
`"repair_notes" not in done["repair"]`.

## SSE Event Sequence

### Normal path (`ANSWERABLE` → `PASSED`)

```
event: token
data: {"type": "token", "content": "The revenue was..."}

event: done
data: {
  "type": "done",
  "sources": [...],
  "confidence": 0.9,
  "context_sufficient": true,
  "intent": "document_qa",
  "intent_confidence": 0.8,
  "trace_id": "...",
  "calculations": [],
  "answerability": {"status": "answerable", ...},
  "validation": {"status": "passed", ...},
  "repair": {"was_repaired": false, "fallback_used": false}
}
```

### Blocked path (`ANSWERABLE` → `BLOCKED`)

```
event: token
data: {"type": "token", "content": "I cannot provide a verified answer..."}

event: done
data: {
  "type": "done",
  "trace_id": "...",
  "answerability": {"status": "answerable", ...},
  "validation": {"status": "blocked", ...},
  "repair": {"was_repaired": false, "fallback_used": true}
}
```

The token is the safe fallback; no partial LLM tokens precede it.

### `NOT_ANSWERABLE` path (no LLM)

```
event: token
data: {"type": "token", "content": "I cannot answer this question based on the available evidence..."}

event: done
data: {
  "type": "done",
  "trace_id": "...",
  "answerability": {"status": "not_answerable", ...}
}
```

The LLM gateway is never called; the deterministic refusal is the token.

### `CALCULATION_BLOCKED` path (no LLM)

```
event: token
data: {"type": "token", "content": "The requested calculation could not be completed with the available data..."}

event: done
data: {
  "type": "done",
  "trace_id": "...",
  "answerability": {"status": "calculation_blocked", ...},
  "calculations": [{"status": "blocked", ...}]
}
```

### Internal error path

```
event: error
data: {
  "type": "error",
  "detail": {"error_code": "stream_error", "message": "Streaming query failed. Please retry."},
  "message": "Streaming query failed. Please retry.",
  "retryable": true,
  "trace_id": "..."
}
```

The trace diagnostics record `error_code: "STREAM_INTERNAL_ERROR"` and
`exception_type`; neither `str(exc)` nor the answer is stored.

## Validation Pipeline Toggle

`RAGEngine` accepts `enable_validation_pipeline: bool = True`
(`src/services/rag_engine.py`). When `False`, the orchestrator skips
answerability evaluation, response validation, and repair; the SSE
`done` event omits `answerability` / `validation` / `repair` (legacy
parity). The default is `True` in production.

## Test Coverage

- `tests/validation/test_validation_http_sse.py`: real FastAPI
  `TestClient` tests covering:
  - `done` event includes `answerability` / `validation` / `repair`
    when present.
  - `done` event omits them when the pipeline is disabled (legacy shape).
  - Internal fields (`message`, `evidence_ids`, `best_score`,
    `repair_notes`) never leak into HTTP or SSE responses.
  - Blocked answer emits the safe fallback as the single token event;
    `generate_answer_stream` is never called.
  - Token event carries the (possibly repaired) answer text.
- `tests/validation/test_trace_content_redaction.py`: asserts
  `final_context=None`, `answer=None`, `message_hash`, `claim_excerpt`
  (max 80 chars), and no `str(exc)` in streaming errors.
- `tests/validation/test_grounded_response_e2e.py`: end-to-end pipeline
  scenarios including streaming safety invariants.
