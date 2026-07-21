# Phase 4 Validation Error Codes

This document lists all validation codes, their severity, and their
meaning.

## Answerability Statuses

| Status | Meaning | LLM Invoked? |
|---|---|---|
| ANSWERABLE | Evidence is sufficient; generation allowed. | Yes |
| PARTIALLY_ANSWERABLE | Some requested documents are missing but enough evidence exists. | Yes |
| NOT_ANSWERABLE | Evidence is missing or irrelevant; LLM must NOT be invoked. | No |
| CALCULATION_BLOCKED | Calculation pipeline returned BLOCKED/FAILED; LLM must NOT be invoked. | No |

### Answerability Reason Codes

| Code | Meaning |
|---|---|
| calculation_blocked | The calculation pipeline returned BLOCKED. |
| calculation_failed | The calculation pipeline returned FAILED. |
| no_evidence | No evidence chunks were retrieved. |
| insufficient_evidence | Evidence was retrieved but sufficiency evaluator returned False. |
| missing_documents | Some requested documents have no evidence chunks. |
| no_retrieval_required | The intent does not require retrieval (conversation/unsupported). |

## Validation Statuses

| Status | Meaning | Action |
|---|---|---|
| PASSED | All claims are grounded; no blocking issues. | Answer returned as-is. |
| REPAIRABLE | Non-blocking issues exist; answer can be repaired. | One deterministic repair attempt. |
| BLOCKED | Critical issues exist; answer must NOT be returned. | Safe fallback. |
| FAILED | Validator encountered an internal error (fail-closed). | Safe fallback. |
| NOT_APPLICABLE | Validation does not apply to this intent (conversation). | Answer returned as-is. |

## Validation Issue Codes

| Code | Severity (default) | Validator | Meaning |
|---|---|---|---|
| NUMERIC_UNGROUND | CRITICAL (strict) / ERROR (lenient) | NumericClaimValidator | A numeric value in the answer could not be verified against evidence. |
| UNSUPPORTED_CLAIM | ERROR | UnsupportedClaimValidator | The answer references a financial metric not found in evidence. |
| CITATION_MISSING | WARNING (or CRITICAL per policy) | CitationValidator | A numeric value lacks a citation to the source document. |
| CITATION_UNRESOLVED | WARNING (or CRITICAL per policy) | CitationValidator | A citation marker (e.g. [1]) could not be resolved to evidence. |
| PERIOD_MISMATCH | CRITICAL | UnitPeriodValidator | The answer references a year/period that does not match evidence. |
| CURRENCY_MISMATCH | CRITICAL | UnitPeriodValidator | The answer references a currency that does not match evidence. |
| UNIT_MISMATCH | CRITICAL | UnitPeriodValidator | The answer references a unit that does not match evidence. |
| CALCULATION_MISMATCH | CRITICAL | CalculationValidator | The answer's numeric value does not match the CalculationResult. |
| VALIDATOR_ERROR | CRITICAL | ResponseValidator (fail-closed) | The validator encountered an internal error. |

## Public vs. Internal Fields

### Public (API response)

- `code`: the machine-readable code string.
- `severity`: `critical`, `error`, or `warning`.
- `public_message`: a user-safe message (no internal details).

### Internal (trace only)

- `message`: the full internal message (may contain debug info).
- `evidence_ids`: the chunk IDs that were checked.
- `claim_text`: the extracted claim text.
- `expected_value` / `actual_value`: for numeric comparisons.
