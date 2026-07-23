# Phase 5 Ablation Plan

This document defines the ablation study for Phase 5. The goal is to measure
the marginal contribution of each pipeline component by removing it one at a
time and observing the effect on safety and utility metrics.

Source of truth: the Phase 5 evaluation protocol
([phase5-evaluation-protocol.md](phase5-evaluation-protocol.md)) and the
pipeline modules in `src/`.

---

## Design: One-Factor-at-a-Time (OFAT)

The ablation uses a **One-Factor-at-a-Time** design. Each variant removes
exactly one component from the reference configuration (A0). This isolates the
marginal effect of that component.

OFAT is chosen over full factorial design because:

- The goal is component-level attribution, not interaction modeling.
- The number of variants (10) is manageable and reproducible.
- Full factorial would require 2^9 = 512 runs, which is infeasible for a
  sealed-test evaluation scored once per variant.

A known limitation of OFAT is that it does not capture interactions between
components. If component interactions are suspected, future work may add a
small focused interaction study.

---

## Held-Constant Factors

The following are identical across all 10 variants. Only the single ablated
component differs.

| Factor | Value |
|--------|-------|
| Model checkpoint | Fixed (recorded in `RunManifest.model_checkpoint_sha256`) |
| Tokenizer | Fixed (recorded in `RunManifest.tokenizer_sha256`) |
| Corpus | Fixed (recorded in `RunManifest.corpus_manifest_hash`) |
| Vector index | Fixed (recorded in `RunManifest.vector_index_hash`) |
| BM25 index | Fixed (recorded in `RunManifest.bm25_index_manifest`) |
| Question order | Fixed (identical `questions.jsonl` for every variant) |
| Random seed | `20260723` (recorded in `RunManifest.random_seed`) |
| Configuration (non-ablated params) | Frozen calibration-selected vector |

---

## The 10 Variants

| Variant | Name | Component Removed | What Changes |
|---------|------|--------------------|--------------|
| **A0** | Full | None | Reference configuration. All components active. |
| **A1** | Dense Only | BM25/sparse retrieval | Only dense (embedding) retrieval runs. BM25 disabled. Tests sparse contribution. |
| **A2** | BM25 Only | Dense retrieval | Only BM25 (sparse) retrieval runs. Dense disabled. Tests dense contribution. |
| **A3** | No Reranker | Reranking step | Retrieval candidates are not reranked. Tests reranker contribution. |
| **A4** | No Query Rewrite | Query rewriting | The original question is passed directly to retrieval without rewriting/expansion. |
| **A5** | No Hierarchical Context | Hierarchical parent-context expansion | Child chunks are returned without expanding to their parent section/page excerpt. |
| **A6** | No Calculator | Calculation pipeline | The Phase 3 calculation pipeline is bypassed. `financial_calculation` intent falls back to generation. |
| **A7** | No Answerability | Answerability gating | The answerability check is disabled. No-answer cases are not gated. |
| **A8** | No Validation | Response validation pipeline | The Phase 4 validation pipeline is fully disabled. No claim extraction, no citation/numeric/calculation validation, no repair. |
| **A9** | No Citation Validation | Citation validation | Only citation validation is disabled. All other validators remain active. |

### Component Map (for reference)

- **BM25/sparse** — `src/services/retrieval.py`, `rag_bm25.db`
- **Dense** — `src/services/vector_store.py`, Chroma collection
- **Reranker** — `src/services/reranker.py` (heuristic or cross-encoder)
- **Query rewrite** — `src/retrieval/query_processor.py`
- **Hierarchical context** — `src/retrieval/context_builder.py`
- **Calculator** — `src/finance/calculation_pipeline.py`
- **Answerability** — `src/validation/answerability.py`
- **Validation** — `src/validation/validation_pipeline.py`
- **Citation validation** — `src/validation/citation_validator.py`

---

## Only One Component Changed Per Variant

This is the defining constraint of the OFAT design. Each variant differs
from A0 (and from every other variant) in exactly one component. No variant
ablates two components simultaneously.

Before scoring, the ablation runner must verify that:

1. The variant's `RunManifest` matches A0 on every held-constant factor
   (model, tokenizer, corpus, index, seed, question set).
2. The variant's configuration differs from A0 only in the ablated
   component's parameters.
3. No other retrieval, generation, or validation parameter was changed.

---

## A8 ("No Validation") Is Evaluation-Only

Variant **A8 (No Validation)** disables the entire Phase 4 validation
pipeline. This is a diagnostic ablation to measure how much safety the
validation layer provides. It is **never** the production default.

- A8 must not be deployed to production under any circumstance.
- A8 results may not be cited as a recommended configuration.
- A8 exists solely to quantify the validation pipeline's contribution to
  safety metrics.

The same restriction applies, to a lesser degree, to A7 (No Answerability):
disabling answerability gating removes a safety layer and must not become a
production default.

---

## Reporting Requirements

Each ablation variant report must include:

### Strict Pass Metrics

- `macro_strict_pass_rate` (the primary metric).
- `strict_pass_rate` (flat).
- Per-slice `strict_pass_rate` for every slice tag in `SLICE_CATEGORIES`.

### Safety Metric Changes

**Do not report only strict pass.** The ablation report must include the
delta (Δ) of every safety metric relative to A0:

| Safety Metric | A0 Value | Variant Value | Δ | Direction |
|---------------|----------|---------------|---|-----------|
| `unsupported_numeric_release_rate` | … | … | … | higher = worse |
| `invalid_citation_release_rate` | … | … | … | higher = worse |
| `calculation_mismatch_release_rate` | … | … | … | higher = worse |
| `false_block_rate` | … | … | … | higher = worse |
| `unsafe_answer_rate` | … | … | … | higher = worse |
| `validator_fail_closed_rate` | … | … | … | — |

A component may improve strict pass while worsening safety (e.g. removing
validation will increase strict pass on blocked-answerable cases but will
also increase `unsupported_numeric_release_rate`). Both effects must be
reported.

### Full Metric Suite

In addition to strict pass and safety, each variant report includes:

- All retrieval metrics (recall@k, precision@k, MRR, nDCG, coverage, page recall).
- All grounding metrics (citation P/R/F1, numeric accuracy, calculation accuracy, etc.).
- All utility metrics (supported coverage, partial utility, correct refusal, answered rate).
- All system metrics (p50/p95 latency, avg chunks/tokens/sources, LLM call rate, validation block, calculation bypass, system error).
- Failure classification counts per category from `classify_all_failures()`.
- `RunManifest` for the variant, verifying held-constant factors match A0.

### Failure Analysis

Each variant's failures are classified using the fixed priority taxonomy in
`src/evaluation/failure_taxonomy.py`. The ablation report should highlight
which failure categories shifted relative to A0 (see
[phase5-failure-analysis.md](phase5-failure-analysis.md)).
