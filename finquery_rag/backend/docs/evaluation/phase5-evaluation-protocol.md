# Phase 5 Evaluation Protocol

**Protocol Version:** 1.0

**Baseline Commit:** `49eb681a9cff586134136dd720fc2fdbbb71f2fe`

**Random Seed:** `20260723`

This document is the pre-registered evaluation protocol for Phase 5 of the
FinQuery RAG evaluation system. Once submitted, the protocol is frozen and only
the explicitly enumerated mutable artifacts may change (see
[What Cannot Be Modified](#what-cannot-be-modified-after-protocol-submission)).

---

## Primary Metric

The single primary metric is **`macro_strict_pass_rate`**.

It is the average of the per-slice strict pass rate over all defined slice tags
(not a flat average over individual cases). A case passes `strict_case_pass`
only when every applicable strict condition holds: intent correct, retrieval
satisfied, citation satisfied, expected numbers correct, calculation correct,
answerability correct, validation status correct, no forbidden content, and no
system error. The macro average prevents a single high-volume slice from
dominating the headline number.

The primary metric is computed by `macro_strict_pass_rate()` in
`src/evaluation/metrics.py` and aggregated in `compute_all_metrics()`.

---

## Metric Groups

All metrics are implemented in `src/evaluation/metrics.py`. They are
deterministic and offline — no network, no model calls.

### Safety Metrics

Safety metrics are reported as **rates that must stay below** their constraint
thresholds. Lower is better.

| Metric | Description |
|--------|-------------|
| `unsupported_numeric_release_rate` | Fraction of no-answer cases where a number was released. |
| `invalid_citation_release_rate` | Fraction of cases with at least one citation that does not match any expected source. |
| `calculation_mismatch_release_rate` | Fraction of calculation cases where a wrong value was released. |
| `false_block_rate` | Fraction of answerable cases wrongly blocked by validation. |
| `unsafe_answer_rate` | Overall fraction of cases with any safety violation (numeric release, invalid citation, forbidden term, or false block). |
| `validator_fail_closed_rate` | Fraction of cases where the validator failed closed (blocked). |

### Utility Metrics

Utility metrics measure the end-to-end usefulness of the answer.

| Metric | Description |
|--------|-------------|
| `macro_strict_pass_rate` | **Primary metric.** Macro-averaged strict pass rate over slices. |
| `strict_pass_rate` | Flat strict pass rate over all cases. |
| `supported_answer_coverage` | Fraction of answerable cases that got a supported, non-empty, non-refusal answer with full citations. |
| `partial_answer_utility` | Weighted partial credit across all evaluation dimensions. |
| `correct_refusal_rate` | Fraction of no-answer cases correctly refused without numbers. |
| `answered_case_rate` | Fraction of cases that produced a non-empty, non-refusal answer. |

### Retrieval Metrics

Averaged over cases with defined `expected_sources`.

| Metric | Description |
|--------|-------------|
| `recall_at_1` / `recall_at_3` / `recall_at_5` | Recall of expected sources in the top-k retrieved chunks. |
| `precision_at_5` | Fraction of top-5 retrieved chunks that match an expected source. |
| `mrr` | Mean reciprocal rank of the first matching retrieved chunk. |
| `ndcg_at_5` | Normalized discounted cumulative gain with binary relevance. |
| `document_coverage` | Fraction of expected documents (by filename) found in retrieved chunks. |
| `expected_page_recall` | Fraction of expected (filename, page) pairs found in retrieved chunks. |

### Grounding Metrics

| Metric | Description |
|--------|-------------|
| `citation_precision` / `citation_recall` / `citation_f1` | Citation precision, recall, and F1. |
| `numeric_accuracy` | Fraction of expected numbers found in the answer. |
| `metric_value_accuracy` | Fraction of expected calculation values found in the answer. |
| `period_value_accuracy` | Fraction of period-specific expected values found in the answer. |
| `unit_scale_accuracy` | Fraction of calculations where the expected unit/scale appears in the answer. |
| `calculation_accuracy` | Fraction of expected calculations matched by prediction calculations. |
| `answer_calculation_consistency` | Fraction of prediction calculations whose values appear in the answer. |
| `formula_version_accuracy` | Fraction of expected calculations using the correct formula/operation. |

### System Metrics

| Metric | Description |
|--------|-------------|
| `p50_latency_ms` / `p95_latency_ms` | Median and 95th percentile latency in milliseconds. |
| `avg_retrieved_chunks` | Average number of retrieved chunks per prediction. |
| `avg_context_tokens` | Approximate average context tokens. |
| `avg_sources` | Average number of cited sources per prediction. |
| `llm_call_rate` | Fraction of cases that made an LLM call. |
| `validation_block_rate` | Fraction of cases blocked by validation. |
| `calculation_bypass_rate` | Fraction of calculation-intent cases that bypassed calculation. |
| `system_error_rate` | Fraction of cases with a system error. |

---

## Slice Definitions

Slices are defined in `src/evaluation/slices.py` under `SLICE_CATEGORIES`. Each
case may carry multiple slice tags across categories. The
`macro_strict_pass_rate` averages the strict pass rate over the union of all
slice tags.

### Intent

- `document_qa`
- `financial_calculation`
- `document_summary`
- `multi_document_comparison`
- `front_matter`
- `conversation`
- `unknown`

### Language

- `chinese`
- `english`
- `mixed`

### SourceType

- `narrative_paragraph`
- `table`
- `front_matter`
- `multi_page`
- `multi_document`

### Difficulty

- `direct`
- `paraphrased`
- `multi_hop`
- `ambiguous`
- `adversarial_no_answer`

### Safety

- `expected_answer`
- `expected_no_answer`
- `calculation_blocked`
- `unsupported_numeric_trap`
- `wrong_period_trap`
- `wrong_unit_trap`
- `wrong_citation_trap`

---

## Calibration Search Space

Only the following runtime retrieval parameters may be varied during
calibration (on the **calibration partition only**, never the sealed
partition). All other knobs are frozen.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_results` | Number of retrieved chunks per query. | — |
| `min_score_threshold` | Discards chunks below this score. | `0.0` |
| `numeric_rrf_floor` | Minimum RRF score for numeric candidate inclusion. | `0.008` |
| `numeric_dense_floor` | Minimum dense score for numeric candidate inclusion. | `0.08` |
| `max_context_tokens` | Maximum context tokens passed to the generator. | `1100` |
| `rrf_sufficiency_threshold` | RRF sufficiency threshold. | `0.025` |
| `dense_sufficiency_threshold` | Dense sufficiency threshold. | `0.15` |
| `document_coverage` | Target document coverage (reporting target, not a runtime knob in all builds). | — |

Parameters NOT in this list (formulas, metric lexicon, prompts, validator error
codes, per-document rules) are **not calibratable**. See
[phase5-calibration.md](phase5-calibration.md) for the full guide.

---

## Candidate Selection Rule

Candidate selection is **constraint-based**, not "maximize primary metric."
Safety constraints are applied first; only candidates that satisfy every
constraint are eligible for utility comparison. The selection proceeds in
8 steps:

1. **Run all candidates** on the calibration partition under identical
   seed, model, corpus, and index.
2. **Filter on safety constraints.** A candidate is eligible only if every
   safety metric is at or below its constraint threshold:
   - `unsupported_numeric_release_rate` == 0.0 on safety-trap slices.
   - `invalid_citation_release_rate` <= baseline + 0.0 (no regression).
   - `false_block_rate` <= baseline.
   - `unsafe_answer_rate` <= baseline.
3. **Filter on system health.** `system_error_rate` must be 0.0.
4. **Rank eligible candidates by `macro_strict_pass_rate`** (descending).
5. **Apply the one-SE rule.** If the top candidate's lead over the runner-up
   is within one standard error (bootstrap), prefer the candidate with fewer
   changed parameters (closer to baseline) to avoid overfitting.
6. **Tie-break by latency.** If two candidates are within one SE on the
   primary metric, prefer the lower `p95_latency_ms`.
7. **Tie-break by simplicity.** If still tied, prefer the candidate that
   changed fewer calibration parameters from their defaults.
8. **Record the selected candidate's full parameter vector** in the
   calibration report. This vector becomes the frozen sealed-run config.

Only the single selected candidate proceeds to the sealed run.

---

## Ablation Variants

The ablation plan uses a One-Factor-at-a-Time (OFAT) design with 10 variants
(`A0`–`A9`). Full details are in
[phase5-ablation-plan.md](phase5-ablation-plan.md).

| Variant | Name | Component Removed |
|---------|------|--------------------|
| A0 | Full | None (reference configuration) |
| A1 | Dense Only | BM25/sparse retrieval disabled |
| A2 | BM25 Only | Dense retrieval disabled |
| A3 | No Reranker | Reranking step disabled |
| A4 | No Query Rewrite | Query rewriting disabled |
| A5 | No Hierarchical Context | Hierarchical parent-context expansion disabled |
| A6 | No Calculator | Calculation pipeline disabled |
| A7 | No Answerability | Answerability gating disabled |
| A8 | No Validation | Response validation pipeline disabled |
| A9 | No Citation Validation | Citation validation disabled |

Each variant changes exactly one component. The model, tokenizer, corpus,
index, question order, and random seed are held constant across all variants.
Variant `A8` (No Validation) is an **evaluation-only** diagnostic; it must
never become the production default.

---

## Sealed Run Policy

The sealed partition is scored exactly once per candidate configuration. The
sealed runbook is documented in
[phase5-sealed-runbook.md](phase5-sealed-runbook.md).

Key rules:

- The blind runner sees only `EvaluationQuery` objects (questions, document
  names, tags, metadata). It never imports or loads any `expected_*` field.
- Predictions are generated first, then SHA256-sealed, then scored
  independently against the labels by the sealed scorer.
- The sealed scorer is a pure, offline, deterministic function of its inputs.
  It never calls the RAG engine and never modifies the predictions file.
- Allowed reruns are limited to infrastructure failures (auth, service down,
  disk, network) that occur **before** any prediction is produced.
- Disallowed reruns include poor metrics, partial case failures, bad style,
  or retriever issues.
- A scorer bug fix keeps the existing predictions and re-scores all variants
  without re-running the RAG engine.

---

## Random Seed

The fixed random seed for all Phase 5 runs is **`20260723`**.

This seed is recorded in every `RunManifest` (see `src/evaluation/manifests.py`)
and must be identical across the baseline run, calibration candidates, the
sealed run, and all ablation variants.

---

## What Cannot Be Modified After Protocol Submission

After this protocol is submitted, the following are **frozen** and may not be
modified:

1. **The protocol version** (`1.0`) and this document.
2. **The primary metric** (`macro_strict_pass_rate`) and its computation.
3. **The baseline commit** (`49eb681a9cff586134136dd720fc2fdbbb71f2fe`).
4. **The random seed** (`20260723`).
5. **The slice definitions** and the `SLICE_CATEGORIES` mapping.
6. **The metric formulas** in `src/evaluation/metrics.py`.
7. **The failure taxonomy** and its priority order in
   `src/evaluation/failure_taxonomy.py`.
8. **The ablation variant list** (`A0`–`A9`) and the OFAT design.
9. **The candidate selection rule** (8-step constraint-based process).
10. **The sealed run policy** (allowed/disallowed reruns, scorer bug procedure).
11. **The sealed partition's questions and labels.** Their SHA256 hashes are
    recorded in the `RunManifest` and must not change.
12. **The calibration search space.** No parameter outside the enumerated
    list may be tuned.

The only artifacts that may change after submission are:

- The **calibration parameter vector** (the selected candidate's values for
  `n_results`, thresholds, and floors) — frozen only after calibration
  completes and before the sealed run.
- **Scorer bug fixes** that do not alter the metric definitions or the
  predictions — applied uniformly to all variants and re-scored.
- **Infrastructure** (machine, OS, dependency versions) — recorded in the
  `RunManifest`, but must not alter retrieval/validation behavior.
