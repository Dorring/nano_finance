# Phase 5 Failure Analysis Guide

This guide defines the failure classification taxonomy for Phase 5 evaluation.
Every case that does not pass `strict_case_pass` is classified into one or more
failure categories using a fixed priority order.

Source of truth: `src/evaluation/failure_taxonomy.py`.

---

## Overview

The failure taxonomy classifies why a case failed. A case may have multiple
failures (e.g. retrieval missed and then citation also failed), so the
classifier records:

- **Primary failure** — the highest-priority failure category detected.
- **Secondary failures** — all other failure categories detected, in priority
  order.

A case that passed `strict_case_pass` has no failures (`primary_failure =
None`, `secondary_failures = []`). A case that failed strict but triggered no
specific detector is classified as `"unclassified"`.

Classification is performed by `classify_failure(label, prediction)` and
`classify_all_failures(labels, predictions)`.

---

## Fixed Priority Order

Failures are classified in a **fixed priority order** (highest first). The
priority is determined by where in the pipeline the failure originates: a
failure early in the pipeline (system error, auth) is higher priority than a
failure late in the pipeline (citation, calculation), because early failures
may mask or cascade into later failures.

The full priority order (18 categories):

| Priority | Constant | Value |
|----------|----------|-------|
| 1 | `SYSTEM_ERROR` | `system_error` |
| 2 | `AUTH_OR_ENVIRONMENT` | `auth_or_environment` |
| 3 | `INTENT_ERROR` | `intent_error` |
| 4 | `QUERY_REWRITE_ERROR` | `query_rewrite_error` |
| 5 | `RETRIEVAL_MISS` | `retrieval_miss` |
| 6 | `RANKING_MISS` | `ranking_miss` |
| 7 | `DOCUMENT_SCOPE_ERROR` | `document_scope_error` |
| 8 | `CONTEXT_BUILD_ERROR` | `context_build_error` |
| 9 | `ANSWERABILITY_FALSE_POSITIVE` | `answerability_false_positive` |
| 10 | `ANSWERABILITY_FALSE_NEGATIVE` | `answerability_false_negative` |
| 11 | `GENERATION_ERROR` | `generation_error` |
| 12 | `UNSUPPORTED_NUMERIC_RELEASE` | `unsupported_numeric_release` |
| 13 | `UNIT_PERIOD_ERROR` | `unit_period_error` |
| 14 | `CITATION_ERROR` | `citation_error` |
| 15 | `CALCULATION_ERROR` | `calculation_error` |
| 16 | `VALIDATION_FALSE_PASS` | `validation_false_pass` |
| 17 | `VALIDATION_FALSE_BLOCK` | `validation_false_block` |
| 18 | `STREAMING_CONTRACT_ERROR` | `streaming_contract_error` |

This order is defined in `FAILURE_PRIORITY` and is frozen by the protocol.

---

## Primary vs Secondary Failures

A single case may trigger multiple failure detectors. For example, a case
might have a retrieval miss (priority 5) that caused a citation error
(priority 14). The classifier:

1. Runs all detectors in `_detect_failures()`.
2. Collects all triggered categories.
3. Sorts them by the fixed priority order.
4. Returns the first (highest-priority) as the **primary failure**.
5. Returns the rest as **secondary failures** (in priority order).

The primary failure is the root cause to investigate first. Secondary
failures are downstream effects that may resolve once the primary is fixed.

---

## How Each Category Is Detected

### 1. SYSTEM_ERROR

**Detected when:** `prediction.error_code` is not `None` AND does not start
with `auth_` or `env_`.

The blind runner catches engine exceptions and records the exception type
name as `error_code`. Any non-auth, non-environment error code is a system
error.

### 2. AUTH_OR_ENVIRONMENT

**Detected when:** `prediction.error_code` starts with `auth_` or `env_`.

Auth and environment errors are infrastructure-level and are separated from
general system errors because they indicate the run itself was compromised
(not the RAG pipeline).

### 3. INTENT_ERROR

**Detected when:** `label.expected_intent` is set AND
`prediction.intent != label.expected_intent`.

The intent router misclassified the query (e.g. a `financial_calculation`
query was routed as `document_qa`).

### 4. QUERY_REWRITE_ERROR

**Detected when:** `retrieval_debug` contains `rewrite_error` or
`query_rewrite_error`.

The query rewriter raised an error or produced an invalid rewrite. This is
checked before retrieval metrics because a rewrite error may cause all
downstream retrieval to fail.

### 5. RETRIEVAL_MISS

**Detected when:** `label.expected_sources` is set AND either:
- No retrieved chunks at all (`retrieved_chunks` is empty), OR
- Full recall (recall over all retrieved chunks) is `<= 0.0` — no expected
  source was found anywhere in the retrieved set.

### 6. RANKING_MISS

**Detected when:** `label.expected_sources` is set, some expected sources
were found (full recall > 0), BUT top-5 recall is strictly less than full
recall. This means the expected sources exist in the retrieved set but were
ranked too low to appear in the top-5.

### 7. DOCUMENT_SCOPE_ERROR

**Detected when:** `label.expected_sources` is set, some chunks were
retrieved, but none of the retrieved chunks come from any expected document
(by filename). The retriever pulled chunks from the wrong documents.

### 8. CONTEXT_BUILD_ERROR

**Detected when:** `prediction.context_sufficient` is `False`.

The context builder determined the assembled context was insufficient for
answering, even if some chunks were retrieved.

### 9. ANSWERABILITY_FALSE_POSITIVE

**Detected when:** `label.expected_answerability == "no_answer"` but the
prediction's answerability status is `"answerable"`.

The system answered a question that should have been refused.

### 10. ANSWERABILITY_FALSE_NEGATIVE

**Detected when:** `label.expected_answerability == "answerable"` but the
prediction's answerability status is `"no_answer"`,
`"not_answerable"`, or `"insufficient_context"`.

The system refused a question that should have been answered.

### 11. GENERATION_ERROR

**Detected when:** The answer is empty or whitespace-only AND
`error_code` is `None`.

The generation step produced no output, but there was no system error. This
indicates a generation failure (empty response) rather than a crash.

### 12. UNSUPPORTED_NUMERIC_RELEASE

**Detected when:** `label.expected_no_answer` is `True` AND the answer
contains numbers (extracted via regex).

The system released numeric content for a question that should have been
refused. This is a safety violation.

### 13. UNIT_PERIOD_ERROR

**Detected when:** `label.expected_calculations` is set AND for some expected
calculation with a `unit`, that unit (lowercased) does not appear in the
answer text.

The answer is missing the expected unit or scale (e.g. answering `1234`
instead of `1234 USD` or `12.3%`).

### 14. CITATION_ERROR

**Detected when:** `label.expected_sources` is set AND citation recall is
`< 1.0`.

The answer does not cite all expected sources.

### 15. CALCULATION_ERROR

**Detected when:** `label.expected_calculations` is set AND
`calculation_accuracy < 1.0`.

The prediction's calculations do not match the expected calculations (wrong
value, wrong operation, or missing calculation).

### 16. VALIDATION_FALSE_PASS

**Detected when:** `label.expected_validation_status` is in
`{blocked, rejected, failed, error}` BUT the prediction's validation status
is `"passed"`.

The validation pipeline should have blocked the answer but let it through.

### 17. VALIDATION_FALSE_BLOCK

**Detected when:** `label.expected_validation_status == "passed"` BUT the
prediction's validation status is in `{blocked, rejected, failed, error}`.

The validation pipeline wrongly blocked a correct answer.

### 18. STREAMING_CONTRACT_ERROR

**Detected when:** Any warning in `prediction.warnings` contains the
substring `"stream"` (case-insensitive).

A streaming contract violation was recorded (e.g. missing final `done` event,
malformed SSE chunk).

---

## Reporting Requirements

The failure analysis report must include:

### Per-Case Classification

For every case that failed `strict_case_pass`:

- `case_id`
- `primary_failure` — the highest-priority category (or `"unclassified"` if
  no detector triggered).
- `secondary_failures` — all other triggered categories, in priority order.
- `passed` — `False` for all reported cases.

Cases that passed are reported with `primary_failure = None` and
`secondary_failures = []`.

### Aggregate Counts

Counts per failure category, showing how many cases had each category as
their **primary** failure and how many had it as a **secondary** failure:

| Category | Primary Count | Secondary Count | Total |
|----------|---------------|------------------|-------|
| `system_error` | … | … | … |
| `auth_or_environment` | … | … | … |
| ... | ... | ... | ... |
| `streaming_contract_error` | … | … | … |

### Unclassified Cases

Any case that failed `strict_case_pass` but triggered no specific detector is
reported as `"unclassified"`. These cases require manual investigation and
may indicate a gap in the detector coverage.

### Cross-Variant Comparison (for Ablation)

For the ablation study, the failure report should compare each variant's
failure distribution against A0 (Full). This shows which components prevent
which failure types (e.g. A8/No Validation should show a spike in
`unsupported_numeric_release` and `validation_false_pass` categories).

The comparison should highlight categories where the count changed by more
than 2 cases relative to A0, as these indicate a meaningful shift attributable
to the ablated component.

---

## Phase 5 Actual Results

### Baseline (Dev Set, 10 cases)

All 10 cases failed `strict_case_pass` (0/10, 0.0%, 95% CI [0.0000, 0.2775]).

Primary failure distribution:

| Primary Failure | Count |
|-----------------|-------|
| `retrieval_miss` | 5 |
| `intent_error` | 5 |

Secondary failure distribution (most common):

| Secondary Failure | Count |
|-------------------|-------|
| `context_build_error` | 10 |
| `citation_error` | 10 |
| `answerability_false_negative` | 6 |
| `calculation_error` | 2 |
| `validation_false_block` | 1 |

**Root cause:** The dev set uses placeholder document names (e.g.
`annual_report_2025.pdf`, `balance_sheet.pdf`) that do not exist in the
ChromaDB index. All retrieval returns empty results, cascading into
context build errors, citation errors, and answerability false negatives.

### Sealed Test (5 cases)

All 5 cases failed `strict_case_pass` (0/5, 0.0%, 95% CI [0.0000, 0.4345]).

The sealed set uses different placeholder document names (e.g.
`filing_2025.pdf`, `report_2025.pdf`) that also do not exist in the
ChromaDB index, producing the same failure pattern as the baseline.

### Ablation (10 variants, 10 cases each)

All 10 ablation variants scored 0.0 `macro_strict_pass_rate`, consistent
with the baseline. Since all failures stem from missing documents in the
index (not from component differences), disabling individual components
does not change the outcome.

### Failure Separation: System Error vs Quality Failure

All failures in this evaluation are **quality failures** (retrieval miss,
intent error, citation error), not system errors. No `system_error`
or `auth_or_environment` failures were recorded. The RAG engine
initialized and processed all queries without crashes.
