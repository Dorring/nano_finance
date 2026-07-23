# Phase 5 Sealed Run Runbook

This runbook defines the procedure for running the sealed evaluation partition.
The sealed run is scored exactly once per candidate configuration. The
procedure is designed so that the RAG engine never sees any label information,
and scoring is performed independently on sealed predictions.

Source of truth: `src/evaluation/blind_runner.py` (blind run),
`src/evaluation/sealed_scorer.py` (independent scoring),
`src/evaluation/manifests.py` (run manifest and SHA256 verification).

---

## Pre-Run Checklist

Before starting the sealed run, verify every item below. If any check fails,
do not proceed — fix the issue and restart the checklist.

### 1. Worktree Is Clean

```bash
git status --porcelain
```

The output must be empty. There must be no uncommitted changes, untracked
files, or staged modifications. A dirty worktree means the running code does
not match the recorded commit, which breaks reproducibility.

### 2. Commit Matches the Release Candidate (RC)

```bash
git rev-parse HEAD
```

The output must match the sealed-run RC commit recorded in the protocol. This
is the commit that calibration selected and that the protocol froze. The
`RunManifest.git_commit` field records this value and `RunManifest.git_dirty`
must be `false`.

### 3. Questions Hash Matches

Compute the SHA256 of the sealed `questions.jsonl`:

```bash
python -m src.evaluation.eval_cli seal-hash --file eval_data/phase5/sealed/questions.jsonl
```

The hash must match the `questions_sha256` recorded in the sealed
`DatasetManifest`. If the questions file has changed, the sealed partition is
invalid.

### 4. Config Hash Matches

Compute the SHA256 of the frozen configuration file:

```bash
python -m src.evaluation.eval_cli seal-hash --file <frozen_config.json>
```

The hash must match the `config_hash` in the `RunManifest`. The configuration
must be the calibration-selected candidate vector — no ad-hoc parameter
changes are permitted.

### 5. Model, Corpus, and Index Are Frozen

Verify that `model_checkpoint_sha256`, `tokenizer_sha256`,
`corpus_manifest_hash`, `vector_index_hash`, and `bm25_index_manifest` in the
`RunManifest` match the artifacts on disk. If the corpus or index has been
modified since calibration, the results are not comparable.

---

## Blind Run Procedure

The blind run follows four stages: **questions-only → predictions → SHA256
seal → independent scoring**. The RAG engine sees only questions; labels are
never loaded during prediction.

### Stage 1: Questions-Only

The blind runner (`run_blind_jsonl` in `src/evaluation/blind_runner.py`) loads
`EvaluationQuery` objects from the sealed `questions.jsonl`. Each query
contains only:

- `case_id`
- `question`
- `document_names`
- `tags`
- `metadata`

The `EvaluationQuery.from_dict()` constructor **rejects** any key starting
with `expected_`. This is a structural guard: if a label field accidentally
appears in the questions file, the loader raises `ValueError` and the run
aborts.

### Stage 2: Predictions

The runner calls `rag_engine.query()` for each question sequentially, passing
only `question`, `doc_names`, `user_id`, and `n_results`. The result is
captured as an `EvaluationPrediction` with all Phase 3/4 envelope fields:

- `answer`, `sources`, `retrieved_chunks`, `calculations`
- `answerability`, `validation`, `warnings`
- `intent`, `intent_confidence`, `context_sufficient`
- `retrieval_debug`, `trace_id`, `latency_ms`, `error_code`

Engine exceptions are caught and recorded as `error_code` (the exception type
name) so a single failing query never crashes the whole run.

Predictions are written atomically to `predictions.jsonl`.

### Stage 3: SHA256 Seal

After all predictions are written, compute the SHA256 of `predictions.jsonl`
using `compute_jsonl_sha256()` (canonical JSON with sorted keys). This hash
is recorded in the `RunManifest.predictions_sha256`.

The predictions file is now **sealed**. It must not be modified after this
point. The sealed scorer will reject any predictions file whose hash does
not match the manifest.

### Stage 4: Independent Scoring

The sealed scorer (`score_sealed_predictions` in
`src/evaluation/sealed_scorer.py`) runs **after** predictions are sealed.
It:

1. Loads the `RunManifest` (protocol) and reads `predictions_sha256` and
   `labels_sha256`.
2. Recomputes the predictions SHA256 and verifies it matches the manifest.
   If it does not match, scoring aborts with `ValueError`.
3. Recomputes the labels SHA256 and verifies it matches.
4. Loads predictions and labels, and verifies a 1:1 `case_id` correspondence
   (no missing, no extra predictions). Any mismatch aborts scoring.
5. Scores each prediction against its label using deterministic checks.
6. Writes the scoring report atomically.

The scorer never calls the RAG engine and never modifies the predictions
file.

---

## Allowed Reruns

A sealed run may be rerun **only** when the failure is infrastructure, not
algorithmic, and the failure occurs **before any prediction is produced**.
Allowed rerun reasons:

1. **Auth failure** — the RAG engine could not authenticate (e.g. invalid API
   key, expired token). Rerun after fixing credentials.
2. **Service down** — the LLM backend, embedding service, or vector store was
   unreachable. Rerun after the service is restored.
3. **Disk failure** — the predictions file could not be written due to disk
   errors. Rerun after disk is repaired.
4. **Network failure before any prediction** — a network outage occurred
   before the first prediction was generated. Rerun after connectivity is
   restored.

In all cases, the rerun must start from the beginning (Stage 1) with a fresh
`predictions.jsonl`. Partial reruns (re-running only failed cases) are not
permitted because they would mix predictions from different runs.

---

## Disallowed Reruns

The following are **not** valid reasons to rerun the sealed partition:

1. **Poor metrics** — the results are worse than expected. This is a finding,
   not a rerun trigger.
2. **Some case failures** — a subset of cases failed. Failures are reported
   and analyzed; they do not justify a rerun.
3. **Bad style** — the answers are grammatically or stylistically poor.
   Style is not a sealed-test pass/fail criterion.
4. **Retriever issues** — retrieval returned suboptimal results. Retrieval
   behavior is part of the system under test; rerunning would not change the
   deterministic outcome under the same seed and config.

If the system produces poor results, the correct response is to investigate,
fix the underlying issue in a new commit, recalibrate on the calibration set,
and run a **new** sealed evaluation — not to rerun the existing sealed
partition.

---

## Scorer Bug Fix Procedure

If a bug is found in the scorer (not in the RAG engine or the metrics
definitions), the fix follows this procedure:

1. **Keep the existing predictions.** Do not re-run the RAG engine. The
   sealed predictions are valid; only the scoring logic was buggy.
2. **Do not re-run the RAG.** The predictions SHA256 is unchanged.
3. **Re-score all variants.** The fixed scorer is applied uniformly to every
   variant (baseline, ablation variants A0–A9). No variant is selectively
   re-scored.
4. **Record the fix.** The scorer version and the bug description are
   recorded in the post-run report. The original (buggy) report is preserved
   for audit.
5. **Re-verify hashes.** The predictions and labels SHA256 values must still
   match the `RunManifest`. Only the scoring report changes.

A scorer bug fix must not alter the metric definitions or the failure
taxonomy. If the bug is in a metric formula, that is a protocol change, not a
scorer bug fix — see
[phase5-evaluation-protocol.md](phase5-evaluation-protocol.md), "What Cannot
Be Modified."

---

## Post-Run Report

The post-run report must include:

### All Metrics

The full metric suite from `compute_all_metrics()`:

- Retrieval: `recall_at_1`, `recall_at_3`, `recall_at_5`, `precision_at_5`,
  `mrr`, `ndcg_at_5`, `document_coverage`, `expected_page_recall`.
- Grounding: `citation_precision`, `citation_recall`, `citation_f1`,
  `numeric_accuracy`, `metric_value_accuracy`, `period_value_accuracy`,
  `unit_scale_accuracy`, `calculation_accuracy`,
  `answer_calculation_consistency`, `formula_version_accuracy`.
- Safety: `answerability_macro_f1`, `no_answer_precision`,
  `no_answer_recall`, `no_answer_f1`, `unsupported_numeric_release_rate`,
  `invalid_citation_release_rate`, `calculation_mismatch_release_rate`,
  `false_block_rate`, `unsafe_answer_rate`, `validator_fail_closed_rate`.
- Utility: `macro_strict_pass_rate`, `strict_pass_rate`,
  `supported_answer_coverage`, `partial_answer_utility`,
  `correct_refusal_rate`, `answered_case_rate`.
- System: `p50_latency_ms`, `p95_latency_ms`, `avg_retrieved_chunks`,
  `avg_context_tokens`, `avg_sources`, `llm_call_rate`,
  `validation_block_rate`, `calculation_bypass_rate`, `system_error_rate`.
- Counts: `total_cases`, `scored_cases`.

### Per-Slice Metrics

`compute_slice_metrics()` output: for each slice tag in `SLICE_CATEGORIES`,
the full metric suite and `sample_count`. Slices with 0 cases are reported
with `sample_count=0`.

### Failures

Failure classification from `classify_all_failures()`: for each case, the
`primary_failure` category, `secondary_failures` list, and `passed` boolean.
Aggregate counts per failure category are reported.

### Confidence Intervals (CIs)

Bootstrap confidence intervals are reported for the primary metric
(`macro_strict_pass_rate`) and key safety metrics. CIs are computed by
resampling the sealed cases with replacement (using the fixed seed
`20260723`) and recording the 2.5th and 97.5th percentiles of the resampled
metric distribution.

The report must state:

- The number of bootstrap resamples.
- The CI method (percentile bootstrap).
- The seed used for resampling.
- The CI for `macro_strict_pass_rate` (overall and per slice where
  `sample_count >= 20`).

### Run Manifest

The full `RunManifest` is attached to the report, recording the git commit,
model/tokenizer/corpus/index hashes, config hash, seed, and all file SHA256
values. This enables independent verification of reproducibility.
