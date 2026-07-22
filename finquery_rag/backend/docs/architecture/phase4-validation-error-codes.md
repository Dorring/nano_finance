# Phase 4 Validation Error Codes

This document lists every validation code, its severity, the validator
that emits it, and its meaning. Source of truth: the `CODE_*` constants
and `ValidationIssue(code=...)` call sites in `src/validation/` plus
`src/domain/validation.py`.

## Answerability Statuses

`AnswerabilityStatus` (`src/domain/validation.py`):

| Status | Meaning | LLM Invoked? |
|---|---|---|
| `ANSWERABLE` | Evidence is sufficient and all requested documents are covered. | Yes |
| `PARTIALLY_ANSWERABLE` | Some requested documents are missing but enough evidence exists for a limited answer. | Yes |
| `NOT_ANSWERABLE` | Evidence is missing, empty, or below the sufficiency threshold. | No |
| `CALCULATION_BLOCKED` | Calculation pipeline returned `BLOCKED` or `FAILED`; LLM must NOT be invoked. | No |

### Answerability Reason Codes

Emitted in `AnswerabilityResult.reason_codes`
(`src/validation/answerability.py`):

| Code | Meaning |
|---|---|
| `calculation_blocked` | The calculation pipeline returned `BLOCKED`. |
| `calculation_failed` | The calculation pipeline returned `FAILED` (folded into `CALCULATION_BLOCKED` status). |
| `no_evidence` | No evidence chunks were retrieved. |
| `insufficient_evidence` | Evidence was retrieved but the sufficiency evaluator returned `False`. |
| `missing_documents` | Some requested documents have no evidence chunks (→ `PARTIALLY_ANSWERABLE`). |
| `no_retrieval_required` | The intent does not require retrieval (conversation/unsupported). |

## Validation Statuses

`ValidationStatus` (`src/domain/validation.py`):

| Status | Meaning | Action |
|---|---|---|
| `PASSED` | No `ERROR`/`CRITICAL` issues; all strict numeric claims are supported and citations are valid. | Answer returned as-is. |
| `REPAIRABLE` | Only non-blocking issues exist; a single deterministic repair is permitted. | One repair attempt, then revalidate. |
| `BLOCKED` | A `CRITICAL` issue exists (or `ERROR` on a strict path); the answer must NOT be returned. | Safe fallback. |
| `FAILED` | The validator itself could not complete (fail-closed). | Safe fallback. |
| `NOT_APPLICABLE` | Validation does not apply to this intent (conversation/unsupported, or `applies_any_validation=False`). | Answer returned as-is. |

Verdict aggregation (`ResponseValidator._determine_status`):
- Any `CRITICAL` → `BLOCKED`.
- Any `ERROR` → `BLOCKED` if `policy.strict_numeric_grounding` else `REPAIRABLE`.
- Only `WARNING`/`INFO` or no issues → `PASSED`.

## Validation Issue Codes

Severity values use the `ValidationSeverity` enum: `info`, `warning`,
`error`, `critical`. The "default" severity column below is what the
emitting validator hard-codes; some codes are policy-dependent and may
be raised or lowered (noted per code).

### Numeric grounding (`NumericClaimValidator`)

| Code | Severity (default) | Policy-dependent? | Meaning |
|---|---|---|---|
| `NUMERIC_UNGROUND` | `CRITICAL` (strict) / `ERROR` (lenient) / `WARNING` (warn) | Yes — driven by `unsupported_numeric_action` + `strict_numeric_grounding` | A numeric value in the answer could not be verified against any evidence chunk with matching metric AND period on the same chunk. |
| `NUMERIC_VALUE_MISMATCH` | (reserved) | — | Defined as a constant (`CODE_NUMERIC_VALUE_MISMATCH`) for future per-value mismatch detection. Not currently emitted by any validator. |
| `METRIC_VALUE_MISMATCH` | (reserved) | — | Defined as a constant (`CODE_METRIC_VALUE_MISMATCH`). Not currently emitted. |
| `PERIOD_VALUE_MISMATCH` | (reserved) | — | Defined as a constant (`CODE_PERIOD_VALUE_MISMATCH`). Not currently emitted. |
| `PERIOD_AMBIGUOUS` | (reserved, intended `WARNING`) | — | Defined as a constant (`CODE_PERIOD_AMBIGUOUS`). `ClaimExtractor._find_nearby_period` returns `None` when multiple distinct years sit in the window (leaving the claim periodless), but the validator does not currently emit this code. |

### Unsupported claims (`UnsupportedClaimValidator`)

| Code | Severity (default) | Meaning |
|---|---|---|
| `UNSUPPORTED_CLAIM` | `ERROR` | A numeric claim references a metric whose keyword does not appear in ANY evidence chunk (and is not the calculation's `target_metric`). Stronger than `NUMERIC_UNGROUND`: the metric itself is absent. |

### Unit / Period (`UnitPeriodValidator`)

| Code | Severity (default) | Meaning |
|---|---|---|
| `PERIOD_MISMATCH` | `ERROR` | The answer references a year/period whose years do not intersect the years present in the evidence set. |
| `CURRENCY_MISMATCH` | `ERROR` | The answer references a currency (e.g. `USD` via `$`) that does not match the single currency found in the evidence. Only flagged when the evidence has exactly one currency. |
| `UNIT_MISMATCH` | (reserved) | Defined as a constant (`CODE_UNIT_MISMATCH`). Not currently emitted; `UnitPeriodValidator` emits `CURRENCY_MISMATCH` for unit/currency deviations. |

> `PERIOD_MISMATCH` and `CURRENCY_MISMATCH` are both emitted at `ERROR`
> severity. On strict intents (`strict_numeric_grounding=True`) the
> `ResponseValidator` treats `ERROR` as `BLOCKED`; on lenient intents
> (`strict_numeric_grounding=False`, e.g. `front_matter`) `ERROR`
> produces `REPAIRABLE`.

### Citations (`CitationValidator`)

| Code | Severity (default) | Policy-dependent? | Meaning |
|---|---|---|---|
| `CITATION_MISSING` | `WARNING` (or `CRITICAL` per policy) | Yes — `CRITICAL` when `missing_citation_action="block"`, else `WARNING` | A numeric claim has no citation references at all (only emitted when `require_citations=True` and no `citation_ref` claims exist). |
| `CITATION_UNRESOLVED` | `WARNING` | No (always `WARNING` at emission; may be elevated by future policy) | A citation marker (e.g. `[1]`, `[doc.pdf, p.12]`) could not be resolved to any source. |
| `CITATION_NOT_RETRIEVED` | `CRITICAL` | No | A source object's `chunk_id` was not found in the retrieved evidence (and no document+page fallback matched). |
| `CITATION_PAGE_MISMATCH` | `ERROR` | No | A source object's `page` does not match the evidence chunk's page for the same `chunk_id`. |
| `CITATION_DOCUMENT_MISMATCH` | `ERROR` | No | A source object's `document_name` does not match the evidence chunk's document name (after lenient normalization). |
| `CITATION_CHUNK_MISSING` | `WARNING` | No | A source object has no `chunk_id` and cannot be verified against evidence. |
| `CITATION_DOES_NOT_SUPPORT_CLAIM` | `ERROR` | No | A numeric claim carries a numeric citation, but the cited source's evidence chunk does not contain the claim's value. |
| `DOCUMENT_COVERAGE_MISSING` | (reserved) | — | Defined as a constant (`CODE_DOCUMENT_COVERAGE_MISSING`). Not currently emitted; document-coverage gaps are surfaced via the `PARTIALLY_ANSWERABLE` answerability status and `missing_requirements` instead. |

### Calculation (`CalculationValidator`)

| Code | Severity (default) | Meaning |
|---|---|---|
| `CALCULATION_MISMATCH` | `CRITICAL` | A numeric claim in the answer references the calculation's `target_metric` but its value does not match the `CalculationResult.value` (within a `0.01` `Decimal` tolerance, with ratio/percent normalization). |

### Fail-closed (`ResponseValidator` / orchestrator)

| Code | Severity (default) | Emitted by | Meaning |
|---|---|---|---|
| `VALIDATOR_ERROR` | `CRITICAL` | `ResponseValidator.validate` (outer `try/except`) | The central validator raised an internal exception. Produces `ValidationStatus.FAILED`. The answer is never returned. |
| `VALIDATOR_EXCEPTION` | `CRITICAL` | `RAGOrchestrator._validate_and_repair_once` (initial-validation and revalidation `try/except` wrappers) | A validation call raised an exception inside the orchestrator. Produces `ValidationStatus.FAILED`. Distinct from `VALIDATOR_ERROR` so trace can distinguish validator-internal failure from orchestrator-side call failure. |

Both are **fail-closed**: the result is always `FAILED` (never `PASSED`),
and the safe fallback replaces the answer.

## Severity → Verdict Summary

| Severity | Strict intent (`strict_numeric_grounding=True`) | Lenient intent (`strict_numeric_grounding=False`) |
|---|---|---|
| `CRITICAL` | `BLOCKED` | `BLOCKED` |
| `ERROR` | `BLOCKED` | `REPAIRABLE` |
| `WARNING` | `PASSED` | `PASSED` |
| `INFO` | `PASSED` | `PASSED` |

## Public vs. Internal Fields

`ValidationIssue` separates public and internal serialization:

### Public (`to_public_dict`, used in API / SSE responses)

- `code` — the machine-readable code string.
- `severity` — `critical`, `error`, `warning`, or `info`.
- `public_message` — a user-safe message (no internal details).

### Internal (`to_trace_dict`, used in trace only)

- `code`, `severity`.
- `message_hash` — SHA-256 of the internal `message` (first 16 hex chars); the full `message` is NEVER stored.
- `claim_excerpt` — claim text, control-char cleaned, truncated to 80 chars; the full `claim_text` is NEVER stored.
- `evidence_ids` — the chunk IDs that were checked.

The internal `message`, full `claim_text`, `expected_value` /
`actual_value` are never placed in trace storage or public responses.
