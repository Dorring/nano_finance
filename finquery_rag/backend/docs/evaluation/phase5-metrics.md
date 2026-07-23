# Phase 5 Metrics Reference

This document is the complete reference for all Phase 5 evaluation metrics.
Every metric is deterministic and offline — no network calls, no model
inference during scoring.

Source of truth: `src/evaluation/metrics.py`. Aggregation is performed by
`compute_all_metrics()`, which returns a flat dictionary of all metrics.

---

## Retrieval Metrics

Retrieval metrics measure whether the correct source chunks were retrieved.
They are averaged over cases that have defined `expected_sources`. When there
are no expected sources, the metric returns `1.0` (vacuously satisfied).

### recall_at_k

**Function:** `recall_at_k(expected_sources, retrieved_chunks, k)`

Fraction of expected sources found in the top-k retrieved chunks. Returns
`1.0` when there are no expected sources. Returns `0.0` when `k <= 0`.

Aggregated as `recall_at_1`, `recall_at_3`, `recall_at_5` in
`compute_all_metrics`.

### precision_at_k

**Function:** `precision_at_k(expected_sources, retrieved_chunks, k)`

Fraction of top-k retrieved chunks that match an expected source. Returns
`0.0` when `k <= 0` or no chunks are retrieved.

Aggregated as `precision_at_5`.

### mrr

**Function:** `mrr(expected_sources, retrieved_chunks)`

Mean reciprocal rank of the first matching retrieved chunk. The reciprocal
rank is `1 / rank` of the first chunk that matches any expected source.
Returns `1.0` when there are no expected sources. Returns `0.0` when no
match is found.

### ndcg_at_k

**Function:** `ndcg_at_k(expected_sources, retrieved_chunks, k)`

Normalized discounted cumulative gain with binary relevance. A chunk is
relevant (1.0) if it matches any expected source, irrelevant (0.0)
otherwise. DCG is summed as `rel / log2(rank + 1)`. IDCG is computed over
the ideal ranking (all relevant chunks first). Returns `1.0` when there are
no expected sources.

Aggregated as `ndcg_at_5`.

### document_coverage

**Function:** `document_coverage(expected_sources, retrieved_chunks)`

Fraction of expected documents (by filename) found in retrieved chunks.
Matches on `filename` / `doc_name` only (ignores page and chunk_id).
Returns `1.0` when no expected sources have a filename.

### expected_page_recall

**Function:** `expected_page_recall(expected_sources, retrieved_chunks)`

Fraction of expected (filename, page) pairs found in retrieved chunks. Only
expected sources with both `filename` and `page` set are considered. Returns
`1.0` when no expected sources have both fields set.

---

## Grounding Metrics

Grounding metrics measure whether the answer is supported by correct evidence
and citations.

### citation_precision

**Function:** `citation_precision(expected_sources, sources)`

Fraction of cited sources (in the prediction) that match an expected source.
Returns `1.0` when there are no expected sources and no citations. Returns
`0.0` when there are expected sources but no citations.

### citation_recall

**Function:** `citation_recall(expected_sources, sources)`

Fraction of expected sources that appear in the cited sources. Returns `1.0`
when there are no expected sources.

### citation_f1

**Function:** `citation_f1(precision, recall)`

Harmonic mean of citation precision and recall. Returns `0.0` when
`precision + recall == 0`.

### numeric_accuracy

**Function:** `numeric_accuracy(answer, expected_numbers)`

Fraction of expected numbers found in the answer. Numbers are extracted from
the answer via regex and compared as `Decimal` values with exact equality
(after normalizing commas and trailing `%`). Returns `1.0` when there are no
expected numbers.

### metric_value_accuracy

**Function:** `metric_value_accuracy(answer, expected_calculations)`

Fraction of expected calculation values found in the answer text. Each
expected value is compared against numbers extracted from the answer, within
the calculation's `tolerance` (as `Decimal` absolute deviation). Returns
`1.0` when there are no expected calculations.

### period_value_accuracy

**Function:** `period_value_accuracy(answer, expected_calculations)`

Fraction of period-specific expected values found in the answer. Only
calculations with a `period` key in their `args` are considered. Returns
`1.0` when no period-specific calculations exist.

### unit_scale_accuracy

**Function:** `unit_scale_accuracy(answer, expected_calculations)`

Fraction of calculations where the expected `unit`/scale appears (as a
case-insensitive substring) in the answer. Only calculations with a `unit`
defined are considered. Returns `1.0` when no calculations have a unit.

### calculation_accuracy

**Function:** `calculation_accuracy(expected_calculations, prediction_calculations)`

Fraction of expected calculations matched by prediction calculations.
Matching is by `calc_id` (falling back to `operation`), with the predicted
value within the expected `tolerance` and the predicted unit matching (when
both are defined). Returns `1.0` when there are no expected calculations.
Returns `0.0` when there are expected calculations but no prediction
calculations.

### answer_calculation_consistency

**Function:** `answer_calculation_consistency(answer, prediction_calculations)`

Fraction of prediction calculations whose values appear in the answer text.
Checks whether each calculation's value (as `Decimal`) is among the numbers
extracted from the answer. Returns `1.0` when there are no prediction
calculations.

### formula_version_accuracy

**Function:** `formula_version_accuracy(expected_calculations, prediction_calculations)`

Fraction of expected calculations using the correct formula/operation. Checks
that the matched prediction calculation has the same `operation` as the
expected calculation. Returns `1.0` when there are no expected calculations.
Returns `0.0` when there are expected calculations but no prediction
calculations.

---

## Answerability & Safety Metrics

Safety metrics measure whether the system correctly refuses unanswerable
questions and avoids releasing unsupported, invalid, or unsafe content.

### answerability_accuracy

**Function:** `answerability_accuracy(expected_answerability, prediction_answerability)`

Returns `1.0` if the prediction's answerability status matches the expected
status, `0.0` otherwise. Returns `1.0` when no expected answerability is
defined.

### answerability_macro_f1

**Function:** `answerability_macro_f1(labels, predictions)`

Macro-averaged F1 over all answerability status labels (e.g. `answerable`,
`no_answer`, `not_answerable`, `insufficient_context`). For each status, F1 is
computed from precision and recall, then averaged across statuses. Returns
`1.0` when no answerability labels are defined.

### no_answer_precision

**Function:** `no_answer_precision(labels, predictions)`

Precision for the 'no answer' class. Of cases predicted as no-answer, the
fraction that were truly no-answer. Returns `0.0` when no cases are
predicted as no-answer.

### no_answer_recall

**Function:** `no_answer_recall(labels, predictions)`

Recall for the 'no answer' class. Of true no-answer cases, the fraction
predicted as no-answer. Returns `0.0` when there are no true no-answer
cases.

### no_answer_f1

**Function:** `no_answer_f1(labels, predictions)`

F1 for the 'no answer' class. Computed as the harmonic mean of
`no_answer_precision` and `no_answer_recall`.

### unsupported_numeric_release_rate

**Function:** `unsupported_numeric_release_rate(labels, predictions)`

Fraction of no-answer cases where a number was released in the answer. A
case is "no-answer" when `expected_no_answer` is `True`. Returns `0.0` when
there are no no-answer cases.

**Lower is better.** Constraint: must be `0.0` on safety-trap slices.

### invalid_citation_release_rate

**Function:** `invalid_citation_release_rate(labels, predictions)`

Fraction of cases with at least one invalid citation. A citation is invalid
when it does not match any expected source (and expected sources are
defined). Returns `0.0` when there are no labels.

**Lower is better.** Constraint: must not exceed baseline.

### calculation_mismatch_release_rate

**Function:** `calculation_mismatch_release_rate(labels, predictions)`

Fraction of calculation cases where a wrong value was released. A case is a
"calculation case" when it has `expected_calculations`. A mismatch occurs
when `calculation_accuracy < 1.0` and the answer is non-empty and not a
refusal. Returns `0.0` when there are no calculation cases.

**Lower is better.**

### false_block_rate

**Function:** `false_block_rate(labels, predictions)`

Fraction of answerable cases that were wrongly blocked by validation. A case
is "answerable" when `expected_no_answer` is `False` and
`expected_answerability != "no_answer"`. A case is "blocked" when the
validation status is in `{blocked, rejected, failed, error}` or the answer is
empty with no error code. Returns `0.0` when there are no answerable cases.

**Lower is better.** Constraint: must not exceed baseline.

### unsafe_answer_rate

**Function:** `unsafe_answer_rate(labels, predictions)`

Overall fraction of cases with any safety violation. A case is unsafe if any
of the following holds:
- It is a no-answer case and the answer contains numbers.
- It has expected sources and the answer contains a citation that does not
  match any expected source.
- The answer contains a forbidden term.
- It is an answerable case that was wrongly blocked.

Returns `0.0` when there are no labels.

**Lower is better.** Constraint: must not exceed baseline.

### validator_fail_closed_rate

**Function:** `validator_fail_closed_rate(predictions)`

Fraction of cases where the validator failed closed (blocked). Counts
predictions where the validation status is in `{blocked, rejected, failed,
error}` or the answer is empty with no error code. Returns `0.0` when there
are no predictions.

---

## Utility Metrics

Utility metrics measure the end-to-end usefulness of the answer.

### strict_case_pass

**Function:** `strict_case_pass(label, prediction) -> bool`

Returns `True` **only if ALL** of the following conditions hold:

1. **No system error** — `prediction.error_code` is `None`.
2. **Intent correct** — if `label.expected_intent` is set, the prediction's
   intent matches.
3. **Retrieval satisfied** — if `expected_sources` is set, recall over all
   retrieved chunks is `1.0` (every expected source is found).
4. **Citation satisfied** — if `expected_sources` is set, citation recall is
   `1.0` (every expected source is cited).
5. **Expected numbers correct** — if `expected_numbers` is set,
   `numeric_accuracy` is `1.0`.
6. **Calculation correct** — if `expected_calculations` is set,
   `calculation_accuracy` is `1.0`.
7. **Answerability correct** — if `expected_answerability` is set,
   `answerability_accuracy` is `1.0`.
8. **Validation status correct** — if `expected_validation_status` is set,
   the prediction's validation status matches.
9. **No forbidden content** — if `forbidden_answer_terms` is set, none of the
   forbidden terms appear in the answer (case-insensitive).

If all applicable conditions pass, the case passes. If any one fails, the
case fails. Conditions that are not applicable (no expected value set) are
skipped (vacuously true).

### macro_strict_pass_rate

**Function:** `macro_strict_pass_rate(labels, predictions) -> float`

**This is the primary metric.**

The average of the per-slice strict pass rate over all slice tags — **not** a
flat average over individual cases.

Algorithm:
1. Collect all slice tags from `label.slice_tags` across all labels.
2. For each slice tag, compute the strict pass rate over the subset of cases
   bearing that tag.
3. Return the mean of all per-tag pass rates.

If no slice tags are present, falls back to the overall flat pass rate.

This macro-averaging prevents a high-volume slice from dominating the
headline number and ensures that small-but-important slices (e.g.
`adversarial_no_answer`, `wrong_unit_trap`) are weighted equally.

### supported_answer_coverage

**Function:** `supported_answer_coverage(labels, predictions)`

Fraction of answerable cases that got a supported answer. A supported answer
is: non-empty, not a refusal, no system error, and fully cites expected
sources (citation recall `1.0`). Returns `0.0` when there are no answerable
cases.

### partial_answer_utility

**Function:** `partial_answer_utility(label, prediction) -> float`

Weighted partial credit across all evaluation dimensions. Weights:

| Dimension | Weight |
|-----------|--------|
| Intent | 0.10 |
| Retrieval | 0.20 |
| Citation | 0.20 |
| Numbers | 0.20 |
| Calculation | 0.15 |
| Answerability | 0.10 |
| Safety | 0.05 |

When a dimension has no expected value, it contributes its full weight
(vacuously satisfied). Safety is `0.0` if there is a system error or a
forbidden term. Returns a float in `[0.0, 1.0]`.

### correct_refusal_rate

**Function:** `correct_refusal_rate(labels, predictions)`

Fraction of no-answer cases correctly refused. A correct refusal is a
no-answer prediction (by status or answer text) with no numbers in the
answer. Returns `0.0` when there are no no-answer cases.

### answered_case_rate

**Function:** `answered_case_rate(predictions)`

Fraction of cases that produced a non-empty, non-refusal answer. Returns
`0.0` when there are no predictions.

---

## System Metrics

System metrics measure operational characteristics: latency, resource usage,
and error rates.

### p50_latency / p95_latency

**Functions:** `p50_latency(predictions)`, `p95_latency(predictions)`

The 50th (median) and 95th percentile latency in milliseconds, computed via
linear interpolation over sorted latency values. Cases with `latency_ms =
None` are excluded.

### avg_retrieved_chunks

**Function:** `avg_retrieved_chunks(predictions)`

Average number of retrieved chunks per prediction.

### avg_context_tokens

**Function:** `avg_context_tokens(predictions)`

Approximate average context tokens, derived from answer length using a
heuristic of ~4 characters per token.

### avg_sources

**Function:** `avg_sources(predictions)`

Average number of cited sources per prediction.

### llm_call_rate

**Function:** `llm_call_rate(predictions)`

Fraction of cases that made an LLM call. Detected from `retrieval_debug`
fields: `llm_called`, `used_llm_rewrite`, or `llm_rewrite`.

### validation_block_rate

**Function:** `validation_block_rate(predictions)`

Fraction of cases blocked by validation (status in
`{blocked, rejected, failed, error}` or empty answer with no error code).

### calculation_bypass_rate

**Function:** `calculation_bypass_rate(predictions)`

Fraction of `financial_calculation`-intent cases that bypassed the
calculation pipeline (no structured calculations produced). Returns `0.0`
when there are no calculation-intent cases.

### system_error_rate

**Function:** `system_error_rate(predictions)`

Fraction of cases with a system error (`error_code` is not `None`).

---

## Aggregation

All metrics are aggregated by `compute_all_metrics(labels, predictions)`,
which:

1. Pairs labels with predictions by `case_id`.
2. Computes retrieval metrics averaged over cases with `expected_sources`.
3. Computes grounding metrics over relevant subsets.
4. Computes safety, utility, and system metrics over all predictions.
5. Returns a flat dictionary with `total_cases` and `scored_cases`.

Per-slice aggregation is performed by `compute_slice_metrics()` in
`src/evaluation/slices.py`, which scores each slice tag independently and
includes its `sample_count`.
