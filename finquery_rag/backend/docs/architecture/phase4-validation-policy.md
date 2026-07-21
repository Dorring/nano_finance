# Phase 4 Validation Policy

This document describes the intent-aware validation policies used by the
Phase 4 grounding and validation pipeline.

## Design

Each intent has a frozen `ValidationPolicy` that controls how strictly
answers are validated. The policy is selected by
`get_policy_for_intent(intent_string)`.

Not all intents require the same rules: a financial calculation demands
strict numeric grounding and citation for every operand, while a plain
conversation must not be rejected for lacking document evidence.

## Policy Matrix

| Intent | require_evidence | require_citations | validate_numeric | validate_units | validate_periods | strict_numeric_grounding | unsupported_numeric_action | missing_citation_action |
|---|---|---|---|---|---|---|---|---|
| financial_calculation | True | True | True | True | True | True | block | block |
| document_qa | True | True | True | True | True | True | block | warn |
| document_summary | True | False | True | True | True | True | block | warn |
| multi_document_comparison | True | True | True | True | True | True | block | block |
| front_matter | True | True | False | False | False | False | warn | warn |
| conversation | False | False | False | False | False | False | warn | warn |
| unsupported | False | False | False | False | False | False | warn | warn |

Unknown / None intents fall back to the `document_qa` policy (conservative).

## Action Semantics

- **block**: the issue is CRITICAL; the answer must NOT be returned.
  The validator sets the issue severity to `CRITICAL` and the
  `ResponseValidator` aggregates it into a `BLOCKED` verdict.
- **warn**: the issue is WARNING; the answer may still pass. The
  validator sets the issue severity to `WARNING` and the
  `ResponseValidator` records it but does not block.

## Policy Properties

- `require_evidence`: if True, the absence of any retrieved evidence
  forces `NOT_ANSWERABLE` (no LLM invocation).
- `require_citations`: if True, numeric/calculation claims must cite
  evidence that actually contains the value.
- `validate_numeric_claims`: if True, run `NumericClaimValidator`.
- `validate_units`: if True, run `UnitPeriodValidator` (unit checks).
- `validate_periods`: if True, period/year mismatches are blocking.
- `strict_numeric_grounding`: if True, ANY unsupported numeric claim is
  CRITICAL (BLOCK); if False, unsupported numerics are ERROR and may be
  repairable.
- `unsupported_numeric_action`: `block` or `warn`.
- `missing_citation_action`: `block` or `warn`.
- `applies_any_validation`: True if at least one validator should run.

## Conversation Special Case

Conversation intents return `ValidationStatus.NOT_APPLICABLE` —
validation is skipped entirely. This ensures conversational responses
are not rejected for lacking document evidence.
