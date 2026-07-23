# Phase 5 Labeling Guide

This guide defines the rules for constructing `EvaluationLabel` objects for
the Phase 5 evaluation dataset. Every label must be grounded in real document
evidence. The goal is a label set that is auditable, reproducible, and free of
the contamination that affected pre-Phase 1 results
(`eval/CONTAMINATION_NOTICE.md`).

Source of truth: `src/evaluation/schemas.py` (`EvaluationLabel`,
`ExpectedSource`, `ExpectedCalculation`).

---

## Core Principle: Evidence First

Every label field must be traceable to a specific location in a source
document. The annotator consults the document, finds the evidence, and records
the label. The model's output is never a source of truth for labels.

If the annotator cannot find evidence in the document for a label field, that
field is left empty or the case is not constructed. Do not fabricate evidence.

---

## What to Record Per Case

For each case, the annotator records the following fields. Every non-empty
field must cite a real document location.

### Expected Sources (`expected_sources`)

For each source the answer should cite or retrieve:

| Field | Required | Description |
|-------|----------|-------------|
| `filename` | Yes | The document filename (e.g. `report_2024.pdf`). |
| `page` | Yes when known | The page number in the source document. |
| `chunk_id` | Recommended | The specific chunk identifier if available. |

The `ExpectedSource.matches()` method checks `chunk_id` first (if set), then
`filename` (exact match), then `page` (string comparison). Recording all three
makes the match precise.

### Expected Numbers (`expected_numbers`)

Numeric values the correct answer must contain. Record the canonical form
(e.g. `"1234.56"`, `"12.5%"`). Numbers are normalized by removing commas and
trailing `%` before comparison.

### Expected Calculations (`expected_calculations`)

For `financial_calculation` intent cases, record each expected calculation:

| Field | Required | Description |
|-------|----------|-------------|
| `calc_id` / `id` | Yes | A unique identifier for the calculation. |
| `operation` | Yes | The formula operation (e.g. `growth_rate`, `margin`, `sum`). |
| `args` | Yes | The operation arguments (may include `period`, `metric`, operands). |
| `expected_value` | Yes | The expected result as a string. |
| `tolerance` | No (default `"0"`) | Allowed absolute deviation as a `Decimal` string. |
| `unit` | No | The expected unit/scale (e.g. `USD`, `%`, `CNY`). |

### Metric / Period / Unit

These are captured within the `expected_calculations` structure:

- **metric** — the financial metric keyword (e.g. `revenue`, `gross margin`,
  `net income`, `EBITDA`, `EPS`). Must be a term from the fixed metric
  lexicon in `src/finance/metric_lexicon.py`.
- **period** — the reporting period (e.g. `FY2024`, `Q3 2024`, bare `2024`).
- **unit** — the unit/scale the answer should express (e.g. `USD`, `%`, `CNY`,
  `EUR`).

### Value / Formula

- **value** — the expected numeric value, recorded in `expected_value`.
- **formula** — the `operation` field. This determines which formula version
  is used. The `formula_version_accuracy` metric checks that the predicted
  calculation uses the correct operation.

### Should-Refuse (`expected_no_answer`)

Set `expected_no_answer = True` when the question cannot be answered from the
document set. This includes:

- The information is genuinely absent from the corpus.
- The question asks for a metric/period not covered by the documents.
- The question is an adversarial trap (e.g. `wrong_period_trap`,
  `wrong_unit_trap`, `unsupported_numeric_trap`).

When `expected_no_answer` is `True`, the answer must not contain any numbers
(checked by `unsupported_numeric_release_rate`).

### Answer Variants

Record acceptable answer phrasings via:

- `required_answer_terms` — terms that must appear in the answer.
- `forbidden_answer_terms` — terms that must NOT appear in the answer.

These are checked by substring match (case-insensitive, after whitespace
normalization). Use them to pin key terms without over-constraining phrasing.

### Intent / Answerability / Validation

| Field | Description |
|-------|-------------|
| `expected_intent` | The expected intent (e.g. `document_qa`, `financial_calculation`). |
| `expected_answerability` | `answerable` or `no_answer`. |
| `expected_validation_status` | The expected validation outcome: `passed`, `blocked`, `rejected`, `failed`, or `error`. |

### Slice Tags (`slice_tags`)

Assign slice tags from `SLICE_CATEGORIES` in `src/evaluation/slices.py`. A
case should carry at least one tag from each relevant category (Intent,
Language, SourceType, Difficulty, Safety).

---

## Forbidden Practices

The following practices are explicitly forbidden. Any label constructed
using them is invalid and must be discarded.

1. **Reverse-engineering labels from model output.** Do not run the model,
   take its answer, and record it as the golden label. The model is the
   system under test; its output is not ground truth.

2. **Copying production sources as golden.** Do not use production trace data,
   user feedback, or live query logs as the source of expected values.
   Production data may reflect retrieval bugs, contamination, or user
   preferences rather than document evidence.

3. **Guessing pages from filenames.** Do not infer a page number from the
   document filename or title. The page must be verified by opening the
   document and locating the evidence. The contamination incident was caused
   in part by hardcoded doc-to-page mappings (`_fallback_pages_for_query` and
   related methods, now removed).

4. **Using hardcoded answers.** Do not write a fixed answer string and treat
   it as correct without document backing. Each expected number, metric, and
   period must be cited to a specific document location.

5. **Using an LLM to decide golden labels.** Do not ask a language model
   (including the system under test or any other model) to generate or
   verify expected values, expected calculations, or expected sources. All
   label decisions must be made by the human annotator reading the source
   documents. An LLM may hallucinate values that do not exist in the corpus.

---

## Dual Review Process

Each label passes through two independent passes:

### Pass 1 — Primary Annotation

The primary annotator constructs the full `EvaluationLabel` by reading the
source document(s). Every field is filled from document evidence. The
annotator records the document name, page, and (where applicable) chunk_id
for each expected source and calculation.

### Pass 2 — Independent Review

A second reviewer independently re-derives the expected values from the
source documents **without** seeing the primary annotator's labels. The
reviewer records their own values. The two passes are then compared:

- **Agreement** — the label is accepted as-is.
- **Disagreement** — the two annotators discuss the discrepancy, consult the
  source document together, and agree on the correct value. The label is
  corrected and the reason for the change is recorded in the dataset notes.

The second pass is **independent**: the reviewer does not see Pass 1 labels
until after recording their own derivation. This catches annotation errors
that a single annotator would propagate.

---

## Single Annotator Disclosure

There is **one primary annotator** for the Phase 5 dataset. This is a
disclosed limitation:

- Inter-annotator agreement cannot be measured with a single primary
  annotator (the second pass is a review, not a parallel independent
  annotation with agreement statistics).
- Systematic biases of a single annotator may affect the label set.
- This limitation is recorded in
  [phase5-known-limitations.md](phase5-known-limitations.md), point 3.

The independent second-pass review mitigates — but does not eliminate — this
risk. Any future dataset revision should add a second primary annotator and
report inter-annotator agreement (e.g. Cohen's kappa on a held-out subset).
