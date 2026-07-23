# Phase 5 Calibration Guide

This guide defines what may and may not be calibrated during Phase 5, how
candidate configurations are selected, and the anti-overfitting measures
enforced. Calibration is performed on the **calibration partition only**.

Source of truth: `src/services/rag_engine.py` (parameter defaults),
`src/retrieval/context_builder.py`, and
[phase5-evaluation-protocol.md](phase5-evaluation-protocol.md) (search space
and selection rule).

---

## Golden Rule: Calibrate Only on the Calibration Set

**Never calibrate on the sealed partition.** The sealed partition's labels
are hidden until predictions are generated and SHA256-sealed. Using sealed
labels for parameter tuning invalidates the evaluation and is treated as
contamination (see [phase5-dataset-card.md](phase5-dataset-card.md),
Contamination Prevention Rules).

The `dev` partition may be inspected for debugging and sanity checks, but it
must not be used for candidate selection. Only the `calibration` partition
(80+ cases) drives parameter search.

---

## Searchable Parameters

Only the following runtime retrieval parameters may be varied during
calibration. All other knobs are frozen.

| Parameter | Where Defined | Default | Description |
|-----------|---------------|---------|-------------|
| `n_results` | `RAGEngine.__init__` | — | Number of retrieved chunks per query passed to the blind runner. |
| `min_score_threshold` | `context_builder.py` | `0.0` | Chunks below this score are discarded before context assembly. |
| `numeric_rrf_floor` | `rag_engine.py` (`RAG_NUMERIC_RRF_FLOOR`) | `0.008` | Minimum RRF score for a numeric candidate to be included. |
| `numeric_dense_floor` | `rag_engine.py` (`RAG_NUMERIC_DENSE_FLOOR`) | `0.08` | Minimum dense score for a numeric candidate to be included. |
| `max_context_tokens` | `rag_engine.py` (`DEFAULT_MAX_CONTEXT_TOKENS`) | `1100` | Maximum context tokens passed to the generator. |
| `rrf_sufficiency_threshold` | `rag_engine.py` (`RAG_RRF_SUFFICIENCY_THRESHOLD`) | `0.025` | RRF sufficiency threshold for context build. |
| `dense_sufficiency_threshold` | `rag_engine.py` (`RAG_DENSE_SUFFICIENCY_THRESHOLD`) | `0.15` | Dense sufficiency threshold for context build. |
| `document_coverage` | Reporting target | — | Target document coverage. Used as a reporting constraint, not a direct runtime knob in all builds. |

These parameters control retrieval depth, score filtering, and context
budget. They are the only variables in the calibration search space.

---

## Not Calibratable

The following are **frozen** and may not be tuned, adjusted, or rewritten
during calibration:

1. **Formulas** — The calculation operations in
   `src/finance/calculation_registry.py` and `src/finance/primitive_tools.py`.
   The math is fixed.

2. **Metric lexicon** — The `_METRIC_KEYWORDS` / `_METRIC_CANONICAL` mappings
   in `src/finance/metric_lexicon.py`. These define which financial terms are
   recognized.

3. **Prompts** — The prompt templates in `src/generation/prompt_builder.py`.
   The generation prompt is not a calibration variable.

4. **Validator error codes** — The error code vocabulary in
   `src/validation/` and `docs/architecture/phase4-validation-error-codes.md`.
   Error codes are a stable contract.

5. **Per-document rules** — No document-specific or query-specific hardcoding
   is permitted. The contamination incident was caused by
   `_fallback_pages_for_query` and related methods (now removed). Per-document
   rules must never be reintroduced.

6. **Metric definitions** — The metric formulas in
   `src/evaluation/metrics.py` are frozen by the protocol.

7. **Slice definitions** — The `SLICE_CATEGORIES` in
   `src/evaluation/slices.py` are frozen by the protocol.

8. **Failure taxonomy** — The `FAILURE_PRIORITY` order in
   `src/evaluation/failure_taxonomy.py` is frozen by the protocol.

---

## Constraint-Based Selection: Safety First, Then Utility

Candidate selection is **not** "pick the highest `macro_strict_pass_rate`."
It is a constraint-based process: safety constraints are applied first, and
only eligible candidates compete on utility. The full 8-step rule is in
[phase5-evaluation-protocol.md](phase5-evaluation-protocol.md#candidate-selection-rule).

Summary:

1. Run all candidates on the calibration partition under identical seed,
   model, corpus, and index.
2. **Safety filter.** A candidate is eligible only if every safety metric is
   at or below its constraint threshold:
   - `unsupported_numeric_release_rate` == 0.0 on safety-trap slices.
   - `invalid_citation_release_rate` <= baseline (no regression).
   - `false_block_rate` <= baseline.
   - `unsafe_answer_rate` <= baseline.
3. **System health filter.** `system_error_rate` must be 0.0.
4. Rank eligible candidates by `macro_strict_pass_rate` (descending).
5. Apply the one-SE (standard error) rule: if the top candidate's lead is
   within one bootstrap SE of the runner-up, prefer the candidate closer to
   baseline.
6. Tie-break by `p95_latency_ms` (lower wins).
7. Tie-break by simplicity (fewer changed parameters wins).
8. Record the selected candidate's full parameter vector.

The single selected candidate proceeds to the sealed run.

---

## Anti-Overfitting Measures

Calibration on a finite dataset risks overfitting to the calibration
partition. The following measures mitigate this risk:

1. **Constraint-based, not metric-maximizing.** Safety constraints eliminate
   unsafe candidates before any utility comparison, preventing the calibrator
   from chasing utility at the expense of safety.

2. **One-SE rule.** When the top candidate's lead is within one bootstrap
   standard error of the runner-up, the calibrator must prefer the candidate
   closer to the baseline (fewer changed parameters). This avoids selecting a
   candidate whose apparent advantage is noise.

3. **Default preference.** When parameters do not produce a statistically
   significant improvement, they remain at their defaults. The burden of
   proof is on the candidate to justify a change.

4. **Simultaneous simplicity and latency tie-breaks.** Even after the one-SE
   rule, remaining ties are broken by lower latency and fewer changed
   parameters, both of which reduce overfitting risk.

5. **Single candidate to sealed.** Only one candidate configuration proceeds
   to the sealed partition. The calibrator does not get to "try again" on the
   sealed set.

6. **Frozen search space.** Only the enumerated parameters may be tuned. No
   new parameter may be introduced during calibration. This prevents the
   calibrator from searching an expanding space.

7. **Document-level isolation.** The calibration and sealed partitions share
   no documents, so a parameter that overfits to calibration-specific document
   quirks will not benefit on the sealed set.

---

## All Candidates Must Be Reported

The calibration report must include **every candidate** that was evaluated,
not just the winner. For each candidate, record:

- The full parameter vector (values of all searchable parameters).
- All safety metrics (constraint pass/fail status).
- All utility metrics (`macro_strict_pass_rate`, `strict_pass_rate`,
  `supported_answer_coverage`, etc.).
- All retrieval and grounding metrics.
- All system metrics (`p50_latency_ms`, `p95_latency_ms`, etc.).
- Whether the candidate was eligible (passed safety constraints).
- Whether the candidate was selected and why (or why not).

Candidates that failed the safety filter are reported with their constraint
violations. Candidates that were eliminated by the one-SE rule are reported
with their SE comparison against the winner. No candidate may be silently
dropped from the report.
