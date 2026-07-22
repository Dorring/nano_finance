# Phase 5 Dataset Card

This card describes the Phase 5 evaluation dataset for the FinQuery RAG
system. The dataset is split into three partitions with strict document-level
isolation to prevent data leakage and label contamination.

Source of truth: `src/evaluation/schemas.py` (partition and case definitions),
`scripts/check_phase5_dataset_overlap.py` (overlap checks), and
`eval/CONTAMINATION_NOTICE.md` (contamination history).

---

## Partitions

The dataset uses three partitions, identified by the `partition` field in
`DatasetManifest`. Valid values are defined in `VALID_PARTITIONS`:
`("dev", "calibration", "sealed")`.

| Partition | Minimum Size | Purpose |
|-----------|--------------|---------|
| `dev` | 40+ cases | Open development and debugging. May be inspected freely. |
| `calibration` | 80+ cases | Parameter search and candidate selection. Labels visible to the calibrator but never used for tuning after selection. |
| `sealed` | 120+ cases | Scored exactly once per candidate. Labels hidden until predictions are sealed. |

Minimum sizes are floors. A partition may exceed its minimum but may never
fall below it after the dataset is finalized.

---

## Document-Level Isolation

**No document may appear in more than one partition.** Isolation is enforced
at the document (filename) level, not the chunk level. This means:

- A financial report used to construct `dev` cases cannot also be the source
  for `calibration` or `sealed` cases, even if different pages are referenced.
- If a document is shared across partitions, the overlap checker flags it as
  a `chunk_id` overlap error (see below).
- Document-to-partition assignment is recorded at dataset construction time
  and verified before any run.

This rule exists because retrieval and generation can leak information across
pages of the same document (e.g. via hierarchical parent-context expansion).

---

## Slice Requirements

Each partition must cover the slice tags defined in `SLICE_CATEGORIES`
(`src/evaluation/slices.py`). For the **sealed partition**, every slice must
contain **>= 20 items** so that per-slice metrics are statistically
meaningful and the `macro_strict_pass_rate` is not dominated by tiny slices.

The five slice categories are:

1. **Intent** — `document_qa`, `financial_calculation`, `document_summary`,
   `multi_document_comparison`, `front_matter`, `conversation`, `unknown`.
2. **Language** — `chinese`, `english`, `mixed`.
3. **SourceType** — `narrative_paragraph`, `table`, `front_matter`,
   `multi_page`, `multi_document`.
4. **Difficulty** — `direct`, `paraphrased`, `multi_hop`, `ambiguous`,
   `adversarial_no_answer`.
5. **Safety** — `expected_answer`, `expected_no_answer`, `calculation_blocked`,
   `unsupported_numeric_trap`, `wrong_period_trap`, `wrong_unit_trap`,
   `wrong_citation_trap`.

A case may carry multiple slice tags across categories (e.g. a case can be
`document_qa` + `english` + `table` + `direct` + `expected_answer`).

---

## Data Independence Rules

1. **No case_id reuse across partitions.** Every `case_id` is globally unique.
2. **No identical questions across partitions.** Exact and normalized
   (lowercase, stripped, whitespace-collapsed) question duplicates are
   forbidden.
3. **No shared (document, page, metric, period) tuples.** A combination of
   document + page + metric + period may not appear in two partitions.
4. **No shared expected-number sets.** Two partitions must not contain the
   same sorted tuple of expected numbers.
5. **No chunk_id overlap.** The set of `chunk_id` values referenced by one
   partition must be disjoint from every other partition.

These rules are enforced by `scripts/check_phase5_dataset_overlap.py`, which
returns exit code 1 if any violation is found. High-similarity questions
(Jaccard >= 0.8 on word sets) produce warnings but do not fail the check.

---

## What Constitutes a Case

A single evaluation case is the union of two structurally isolated objects:

### EvaluationQuery

What the **blind runner** sees. Defined in `src/evaluation/schemas.py`. It
contains **no** `expected_*` fields — the constructor explicitly rejects any
key starting with `expected_`.

```python
EvaluationQuery(
    case_id: str,
    question: str,
    document_names: tuple[str, ...],
    tags: tuple[str, ...],
    metadata: Mapping[str, Any],
)
```

### EvaluationLabel

What the **sealed scorer** sees. It is never passed to the RAG engine. It is
loaded only after predictions are generated and sealed.

```python
EvaluationLabel(
    case_id: str,
    expected_sources: tuple[ExpectedSource, ...],
    expected_numbers: tuple[str, ...],
    expected_calculations: tuple[ExpectedCalculation, ...],
    expected_intent: str | None,
    expected_answerability: str | None,
    expected_validation_status: str | None,
    expected_no_answer: bool,
    required_answer_terms: tuple[str, ...],
    forbidden_answer_terms: tuple[str, ...],
    slice_tags: tuple[str, ...],
)
```

A `DatasetManifest` records each partition's `questions_sha256` (public) and
`labels_sha256` (omitted from the sealed public manifest). The sealed scorer
recomputes and verifies both hashes before scoring.

---

## Annotation Process

Annotation follows a **single primary annotator with independent second-pass
review** model:

1. **Primary annotation.** A single annotator constructs the `EvaluationLabel`
   for each case by consulting the source documents directly. Every label
   field must be traceable to a specific document, page, and chunk (see
   [phase5-labeling-guide.md](phase5-labeling-guide.md)).

2. **Independent second-pass review.** A second reviewer independently
   re-derives a subset of labels from the source documents without seeing the
   primary annotator's labels. Discrepancies are resolved by discussion and
   the label is corrected with a recorded reason.

3. **Single annotator disclosure.** Because there is only one primary
   annotator, this is a disclosed limitation (see
   [phase5-known-limitations.md](phase5-known-limitations.md), point 3).

---

## Contamination Prevention Rules

The Phase 5 dataset is constructed to avoid the contamination that affected
pre-Phase 1 results (documented in `eval/CONTAMINATION_NOTICE.md`).

1. **No hardcoded document-to-page mappings.** The deprecated methods
   (`_fallback_pages_for_query`, `_supporting_page_coverage_for_query`,
   `_force_supporting_page_coverage`, `_augment_with_page_fallbacks`,
   `_ensure_supporting_sources`) were removed in Phase 1 and must never be
   restored.

2. **No eval-specific source injection.** Retrieval must use the same code
   path as production. No `supporting_source_page` metadata flag.

3. **No reverse-engineering labels from model output.** Labels come from
   document evidence, not from running the model and treating its output as
   ground truth.

4. **No copying production sources as golden.** Production trace data is not
   used as a golden label source.

5. **Cross-partition overlap is checked.** The `check_phase5_dataset_overlap.py`
   script runs in CI and must pass before any partition is scored.

6. **Sealed labels are not used for tuning.** Once the sealed partition is
   published, its labels may not be used to tune the system or improve
   metrics. Sealed labels that leak into tuning invalidate the results.

7. **Smoke fixtures remain for CI only.** The `eval/golden_smoke.jsonl`
   fixture is for CI validation and smoke tests, not for resume, README, or
   project quality claims. Only sealed-test metrics may be cited externally.
