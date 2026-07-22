# Phase 5 Known Limitations

This document explicitly lists the known limitations of the Phase 5 evaluation
system. These are design boundaries, not bugs — each limitation exists for a
specific reason (determinism, auditability, cost, or feasibility). Source of
truth: the evaluation modules in `src/evaluation/` and the protocol in
[phase5-evaluation-protocol.md](phase5-evaluation-protocol.md).

---

## 1. Sealed Sample Size Limitations

The sealed partition contains 120+ cases. While this meets the minimum
threshold for per-slice reporting (each slice >= 20 items), it is a small
sample for drawing high-confidence statistical conclusions.

- Per-slice metrics on slices with exactly 20 cases have wide confidence
  intervals. A single case flip changes the slice pass rate by 5 percentage
  points.
- The `macro_strict_pass_rate` averages over slices, which reduces variance
  compared to a flat average, but the per-slice variance remains high.
- Rare failure modes (e.g. `wrong_unit_trap`) may be under-represented. A
  failure that occurs in 1 of 20 cases may or may not reflect a real
  systematic issue.
- Bootstrap confidence intervals are reported, but they reflect sampling
  variance only — they do not account for label noise or model
  non-determinism.

The results should be interpreted as directional, not as precise point
estimates of production performance.

## 2. Document Domain Coverage Limitations

The evaluation corpus covers a limited set of financial documents. The
documents were selected to exercise specific retrieval, calculation, and
validation paths, not to be representative of all financial reporting
domains.

- Metrics measured on this corpus may not transfer to documents with
  different structures (e.g. insurance filings, banking regulatory reports,
  non-English financial statements with different formatting conventions).
- Slice coverage (Intent, SourceType, Difficulty, Safety) is designed for
  this corpus. A different corpus may produce slices with zero cases.
- The metric lexicon in `src/finance/metric_lexicon.py` covers common
  financial metrics (revenue, margin, EBITDA, EPS, etc.) but does not cover
  every possible financial term. Metrics outside the lexicon are not
  validated.

Results are specific to the tested document domain and should not be
generalized without re-evaluation on the target domain.

## 3. Annotator Count (Single Annotator)

There is **one primary annotator** for the Phase 5 dataset, with an
independent second-pass review (see
[phase5-labeling-guide.md](phase5-labeling-guide.md)).

- Inter-annotator agreement (e.g. Cohen's kappa) cannot be computed with a
  single primary annotator. The second pass is a review, not a parallel
  independent annotation.
- Systematic biases of the single annotator (e.g. consistently recording a
  tolerance that is too tight or too loose) may affect the label set without
  being detected.
- The second-pass review catches individual errors but cannot detect biases
  that affect both passes (if the reviewer shares the annotator's
  assumptions).

Future dataset revisions should add a second primary annotator and report
inter-annotator agreement on a held-out subset.

## 4. Deterministic Metrics Are Not Complete Natural-Language Quality

All Phase 5 metrics are deterministic: they use regex extraction, `Decimal`
comparison, and string matching. They do not assess semantic correctness,
readability, coherence, or helpfulness of the natural-language answer.

- An answer that contains the correct numbers and citations but is poorly
  written, confusing, or misleading will pass `strict_case_pass`.
- An answer that is semantically correct but uses a different number format
  than expected (e.g. `1.2 billion` instead of `1,200,000,000`) may fail
  `numeric_accuracy` despite being equivalent.
- The `partial_answer_utility` metric provides weighted partial credit, but
  its weights are fixed and may not reflect the relative importance of each
  dimension for every use case.

Deterministic metrics are a strong safety net but are not a substitute for
human judgment of answer quality.

## 5. No LLM Judge

Phase 5 deliberately does **not** use an LLM-as-Judge. No model is asked to
evaluate another model's output for semantic correctness, fluency, or
helpfulness.

- All validators and metrics are deterministic Python. This keeps scoring
  auditable, reproducible, and free of additional model cost/latency.
- The cost is that semantic errors a judge model might detect (e.g. a wrong
  company name, a subtle logical error, a misleading causal claim) are not
  caught if they do not trigger a specific numeric/unit/period/citation/
  calculation check.
- An LLM Judge would introduce its own biases and non-determinism, which
  would compromise the sealed-test guarantee of byte-identical reproducibility.

This is a conscious trade-off: auditability and determinism over semantic
coverage.

## 6. Answer Phrasing Quality Has Only Limited Rule-Based Scoring

Answer phrasing quality is assessed only through:

- `required_answer_terms` — substring presence (case-insensitive).
- `forbidden_answer_terms` — substring absence (case-insensitive).
- `no_answer` markers — a fixed list of refusal phrases (e.g. "couldn't
  find", "无法回答").

These are blunt instruments:

- They cannot assess whether the answer is well-structured, concise, or
  appropriate in tone.
- They cannot detect paraphrased correct answers that use synonyms not in
  the required terms list.
- The `no_answer` marker list is finite; a novel refusal phrasing may not be
  recognized as a refusal, causing a `correct_refusal_rate` miss.

Free-form answer quality (style, clarity, completeness of explanation) is not
scored. A human review pass would be needed for that.

## 7. Sealed Labels Cannot Be Used for Tuning After Publication

Once the sealed partition is published and scored, its labels are revealed.
From that point on, the sealed labels **must not** be used to tune the
system, improve metrics, or select configurations.

- Using sealed labels for tuning is contamination: it turns the held-out
  test set into training data.
- If sealed labels are used for any optimization, the sealed results are
  invalidated and a new sealed partition must be constructed.
- The sealed labels may be used for analysis, failure investigation, and
  reporting, but never as a tuning signal.

This rule is absolute. There is no "small" or "indirect" use of sealed labels
that is permitted for tuning. The calibration partition is the only set on
which parameter search may occur.

## 8. Model Checkpoint and Corpus Changes Require New Evaluation

Phase 5 results are valid only for the specific model checkpoint, tokenizer,
corpus, and index recorded in the `RunManifest`. If any of these change, the
results are no longer comparable and a new evaluation is required.

- **Model checkpoint change** — a new fine-tune, a different base model, or
  even a re-export of the same weights with different quantization changes
  generation behavior. A new sealed run is required.
- **Corpus change** — adding, removing, or modifying documents changes
  retrieval and grounding. Even adding documents that are not directly
  referenced by sealed questions can change retrieval rankings (e.g. by
  introducing near-duplicate chunks).
- **Index change** — rebuilding the vector or BM25 index changes chunk IDs,
  scores, and rankings. A new index requires a new sealed run.
- **Tokenizer change** — changes token boundaries, affecting context window
  budget and generation.

The `RunManifest` records all of these as SHA256 hashes. If any hash
changes, the previous sealed results are not valid for the new configuration.

## 9. Results Cannot Be Extrapolated to All Financial Tasks

The Phase 5 evaluation covers a specific set of financial question types
(`document_qa`, `financial_calculation`, `document_summary`,
`multi_document_comparison`, `front_matter`) on a specific corpus. The
results cannot be extrapolated to:

- **Other financial domains** — tax accounting, derivatives pricing, actuarial
  modeling, regulatory capital calculation, etc. may require different
  formulas, metrics, and validation rules.
- **Other question types** — open-ended analysis, forecasting, scenario
  modeling, or multi-turn advisory conversations are not covered by the
  current slice taxonomy.
- **Other languages** — while `chinese`, `english`, and `mixed` slices
  exist, the coverage of financial terminology in Chinese may differ from
  English, and mixed-language edge cases may not generalize.
- **Other document formats** — HTML filings, XBRL data, scanned images, or
  hand-written tables are not covered.

A claim that the system "works for financial questions" based on Phase 5
results is valid only within the scope of the tested slices, corpus, and
document types.

## 10. Does Not Claim to Fully Eliminate Hallucination

Phase 5 significantly reduces hallucination by combining retrieval grounding,
citation validation, numeric claim validation, and calculation validation.
However, it does **not** claim to fully eliminate hallucination.

- **Non-numeric hallucinations** — a fabricated event description, a wrong
  company name, or an incorrect causal claim may pass validation if it does
  not trigger a specific numeric/unit/period/citation check.
- **Conversation-intent gaps** — `conversation` and `unsupported` intents
  skip validation (`applies_any_validation == False`). A fabricated number in
  a conversational reply is not blocked.
- **Evidence interpretation errors** — if retrieval returns incorrect
  evidence, validation will ground the answer in that incorrect evidence and
  may pass an answer that is grounded-but-wrong.
- **Semantic errors** — subtle semantic errors (e.g. attributing a value to
  the wrong entity, or confusing gross and net figures when both appear in
  the evidence) are not caught by regex-based claim extraction.

The system is designed to be a strong deterministic safety net that blocks
the most common and dangerous forms of numeric hallucination. It is not a
complete correctness guarantee.
