# Phase 4 Known Limitations

This document explicitly lists the known limitations of the Phase 4
grounding and validation pipeline. These are design boundaries, not
bugs — each limitation exists to keep validation deterministic and
auditable.

---

## 1. No Complete Natural Language Fact Verification

Phase 4 does **not** perform complete natural language fact
verification. The validators check specific claim types (numeric,
unit, period, citation, calculation) using regex extraction and
deterministic comparison. Free-form factual claims (e.g., "the company
expanded to Europe in 2024") are not verified against evidence beyond
checking whether the metric appears in the retrieved text.

## 2. Primary Validation Targets

The pipeline primarily validates:
- **Numbers**: numeric values in the answer must appear in evidence.
- **Units**: currency and unit must match evidence.
- **Years/Periods**: year/period references must match evidence.
- **Metrics**: financial metrics referenced must exist in evidence.
- **Citations**: numeric claims must cite retrievable evidence.

## 3. No LLM Judge

Phase 4 does **not** use LLM-as-Judge. All validators are deterministic
Python modules using regex, Decimal comparison, and string matching.
No model is asked to evaluate another model's output.

## 4. Non-Numeric Claims May Only Produce Warnings

Claims that cannot be parsed as numeric, unit, period, or citation
issues may only produce a `WARNING` severity — never `CRITICAL`. This
means non-numeric factual errors (e.g., wrong company name, wrong
event description) may pass validation if they do not trigger a
specific validator.

## 5. Retrieval Errors Still Affect Results

If the retrieval pipeline returns incorrect or irrelevant evidence,
the validation pipeline will validate against that incorrect evidence.
Phase 4 does not independently verify the correctness of retrieval — it
only checks that the answer is grounded in whatever evidence was
retrieved.

## 6. Sealed Test Belongs to Phase 5

No sealed (golden) test set is constructed in Phase 4. Sealed test
construction and long-term regression protection belong to Phase 5.

## 7. Threshold Calibration Belongs to Phase 5

The validation thresholds (e.g., numeric tolerance, citation
strictness) are fixed at conservative defaults. Threshold calibration
against real evaluation data belongs to Phase 5.

## 8. Multi-Hop Complex Reasoning Still Limited

The validators operate on single-hop claims extracted from the answer
text. Multi-hop reasoning chains (e.g., "A increased by 10%, B is
twice A, so B is...") are not fully decomposed and validated
step-by-step. The calculation validator checks final values against
`CalculationResult`, but free-form multi-hop numeric reasoning in LLM
text is only checked at the claim level.

## 9. No Claim to Fully Eliminate Hallucination

Phase 4 significantly reduces numeric hallucination by blocking
ungrounded numeric claims and enforcing citations. However, it does
**not** claim to fully eliminate hallucination. Non-numeric
hallucinations, subtle semantic errors, and errors in evidence
interpretation may still occur. The system is designed to be a strong
deterministic safety net, not a complete correctness guarantee.
