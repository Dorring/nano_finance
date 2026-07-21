# Phase 4 Known Limitations

This document explicitly lists the known limitations of the Phase 4
grounding and validation pipeline. These are design boundaries, not
bugs — each limitation exists to keep validation deterministic and
auditable. Source of truth: the validator modules in `src/validation/`
and the domain types in `src/domain/validation.py`.

---

## 1. Cannot verify arbitrary natural-language facts

Phase 4 does **not** perform complete natural-language fact
verification. The validators check only specific, regex-extractable
claim types:

- **Numbers** — `amount`, `percent`, `ratio` (via `ClaimExtractor`).
- **Units / currencies** — `$`, `¥`, `€`, `£` mapping to USD/CNY/EUR/GBP.
- **Years / periods** — `FY2025`, `Q3 2024`, bare `2024`.
- **Metrics** — a fixed keyword lexicon (`_METRIC_KEYWORDS` /
  `_METRIC_CANONICAL`) covering revenue, gross margin, net income,
  EBITDA, debt ratio, EPS, etc.
- **Citations** — bracket references `[1]` and `[doc.pdf, p.12]`.
- **Calculations** — final values from a Phase 3 `EXECUTED`
  `CalculationResult`.

Free-form factual propositions (e.g. "the company expanded to Europe in
2024", "the CEO resigned in Q2") are NOT extracted as claims and are NOT
verified beyond checking whether an associated metric keyword appears in
the retrieved text. A wrong company name, wrong event description, or
subtle semantic error may pass validation if it does not trigger a
specific numeric/unit/period/citation/calculation validator.

## 2. Sealed Test belongs to Phase 5

No sealed (golden) test set is constructed in Phase 4. Sealed-test
construction and long-term regression protection against model/output
drift belong to Phase 5. Phase 4's own tests are open characterization
and regression tests, not a sealed fixture.

## 3. ClaimExtractor does not parse complex table structures

`ClaimExtractor` is purely regex-based. It extracts numeric values,
percentages, ratios, years, quarters, and citation markers from flat
answer text. It does **not** parse tabular structures (e.g. markdown
tables, multi-row financial statements) into structured claims. A number
inside a table cell is extracted as a standalone claim with metric/period
inferred from a fixed-character proximity window (`_find_nearby_metric`
window = 60 chars, `_find_nearby_period` window = 80 chars), which may
mis-attribute the metric or period when a table cell is dense or
multi-column. Complex table reasoning is not validated step-by-step.

## 4. Unit/Period Validator uses evidence years set when no claim-level binding exists

`UnitPeriodValidator._validate_periods` extracts ALL years mentioned
across the entire evidence set into a single `evidence_years` set, then
checks that each period claim's years intersect that set. When a numeric
claim has no claim-level period binding (i.e. `ClaimExtractor` could not
attach a period), the period check operates on the aggregate evidence
years, not on the specific chunk that grounded the value. This means a
period claim can pass if the year appears anywhere in the evidence, even
if the grounded value came from a different chunk for a different year.
The `NumericClaimValidator`'s same-chunk requirement (value + metric +
period on one chunk) compensates for numeric claims, but pure period
claims are checked only at the evidence-set level.

## 5. Document name matching is lenient

`CitationValidator._doc_names_match` normalizes document names by:
1. Lowercasing.
2. Removing common file extensions (`pdf`, `txt`, `csv`, `xlsx`, `docx`,
   `html`, `md`, `json`).
3. Stripping `user_<id>_` prefixes (e.g. `user_123_paper` → `paper`).
4. Comparing the resulting stems with substring containment
   (`na == nb or na in nb or nb in na`).

This leniency avoids false positives when the context builder strips
extensions/prefixes, but it can produce false negatives (accepting a
mismatched source) when two distinct documents share a stem substring
(e.g. `report_2024.pdf` and `report_2024_summary.pdf`).

## 6. Replay from trace is not possible by default

Phase 4 redacts content from trace storage:
- `final_context=None` and `answer=None` in `trace_data`.
- `ValidationIssue.to_trace_dict()` stores `message_hash` (not `message`)
  and `claim_excerpt` (max 80 chars, not full `claim_text`).
- `ValidationResult.to_trace_dict()` stores `repaired` as a boolean
  (not the repaired answer).
- `RepairResult.to_trace_dict()` stores `answer_length` (not the answer).

Because the full context, answer, claim text, and internal messages are
not recoverable from trace, replaying a query from its trace is NOT
possible by default. Phase 5 will use **independent Sealed Fixtures**
(stored separately from trace) to enable deterministic replay and
long-term regression protection, rather than relying on trace content.

## 7. Conversation intent skips validation (`NOT_APPLICABLE`)

`conversation` and `unsupported` intents have
`applies_any_validation == False`. `ResponseValidator._validate_inner`
returns `ValidationStatus.NOT_APPLICABLE` immediately — no claims are
extracted, no validators run. This means a conversational response that
happens to contain a fabricated number will NOT be blocked by Phase 4
validation. This is intentional (conversational replies should not be
rejected for lacking document evidence), but it is a coverage gap: the
numeric grounding guarantee applies only to document-grounded intents
(`document_qa`, `document_summary`, `multi_document_comparison`,
`financial_calculation`, `front_matter`).

## 8. No LLM-as-Judge, no multi-agent, no threshold tuning

Phase 4 deliberately avoids three techniques sometimes used in
grounding systems:

- **No LLM-as-Judge** — no model is asked to evaluate another model's
  output. All validators are deterministic Python using regex, `Decimal`
  comparison, and string matching. This keeps validation auditable and
  free of additional model cost/latency, at the cost of not catching
  semantic errors a judge model might detect.
- **No multi-agent** — there is no agent debate, no critic agent, no
  self-reflection loop. The pipeline is single-pass: answerability →
  generation → validation → (at most one) repair → revalidation.
- **No threshold tuning** — validation thresholds are fixed at
  conservative defaults: the `CalculationValidator` tolerance is
  `0.01` `Decimal`; the period/currency matchers use exact set
  intersection; the citation resolver requires exact document-name AND
  page match for `[doc.pdf, p.12]`-style citations. Calibration of these
  thresholds against real evaluation data belongs to Phase 5.

## 9. Retrieval errors still affect results

Phase 4 validates the answer against whatever evidence the retrieval
pipeline returned. It does NOT independently verify the correctness or
relevance of retrieval. If retrieval returns incorrect or irrelevant
evidence, validation will ground the answer in that incorrect evidence
and may pass an answer that is grounded-but-wrong. Phase 4 is a
grounding safety net, not a retrieval-correctness guarantee.

## 10. Multi-hop reasoning is limited

Validators operate on single-hop claims extracted from the answer text.
Multi-hop reasoning chains (e.g. "A increased by 10%, B is twice A, so
B is...") are not fully decomposed and validated step-by-step. The
`CalculationValidator` checks final values against a
`CalculationResult`, but free-form multi-hop numeric reasoning in LLM
text is only checked at the individual claim level (each numeric value
must be grounded in evidence on its own).

## 11. No claim to fully eliminate hallucination

Phase 4 significantly reduces numeric hallucination by blocking
ungrounded numeric claims and enforcing citations. However, it does
**not** claim to fully eliminate hallucination. Non-numeric
hallucinations, subtle semantic errors, errors in evidence
interpretation, and conversation-intent numeric claims may still occur.
The system is designed to be a strong deterministic safety net, not a
complete correctness guarantee.
