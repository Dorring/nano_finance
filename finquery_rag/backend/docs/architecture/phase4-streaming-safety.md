# Phase 4 Streaming Safety

This document describes the streaming safety guarantees enforced by the
Phase 4 validation pipeline.

## Guarantees

1. **Strict validation path does not stream partial tokens**: When the
   validation pipeline is enabled and the answer is BLOCKED or FAILED,
   the SSE stream does NOT emit partial LLM tokens. The safe fallback
   message is emitted as a single `token` event, followed by the `done`
   event.

2. **NOT_ANSWERABLE does not invoke the LLM**: When the
   AnswerabilityEvaluator returns `NOT_ANSWERABLE`, the LLM gateway is
   never called. The SSE stream emits the deterministic refusal as a
   single token event.

3. **CALCULATION_BLOCKED does not invoke the LLM**: When the calculation
   pipeline returns BLOCKED or FAILED, the LLM is bypassed. The Phase 3
   safe calculation response is emitted as the token event.

4. **Blocked answers do not write to Session**: When the answer is
   blocked or fails validation, the safe fallback message is what gets
   stored in the conversation history — not the original LLM output.

5. **Done event carries validation verdicts**: The SSE `done` event
   includes `answerability`, `validation`, and `repair` fields when the
   validation pipeline is enabled. These fields are absent when the
   pipeline is disabled (legacy parity).

6. **Internal fields never leak**: The SSE `done` event's validation
   fields use `to_public_dict()` — internal `message`, `evidence_ids`,
   `repair_notes`, and `best_score` are never exposed.

## SSE Event Sequence

### Normal path (ANSWERABLE → PASSED)

```
event: token
data: {"type": "token", "content": "The revenue was..."}

event: done
data: {
  "type": "done",
  "trace_id": "...",
  "answerability": {"status": "answerable", ...},
  "validation": {"status": "passed", ...},
  "repair": {"was_repaired": false, "fallback_used": false}
}
```

### Blocked path (ANSWERABLE → BLOCKED)

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

### NOT_ANSWERABLE path (no LLM)

```
event: token
data: {"type": "token", "content": "I cannot answer this question..."}

event: done
data: {
  "type": "done",
  "trace_id": "...",
  "answerability": {"status": "not_answerable", ...}
}
```

## Test Coverage

- `tests/validation/test_validation_http_sse.py`: 14 tests covering
  HTTP `/query` and SSE `/query/stream` with validation fields present,
  absent, and blocked scenarios.
- `tests/validation/test_grounded_response_e2e.py`: 18 e2e tests
  covering the full pipeline including streaming safety invariants.
