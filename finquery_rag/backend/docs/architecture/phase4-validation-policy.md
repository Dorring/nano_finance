# Phase 4 Validation Policy

This document describes the intent-aware validation policies used by the
Phase 4 grounding and validation pipeline. Source of truth:
`src/validation/validation_policy.py`.

## Design

Each intent has a frozen `ValidationPolicy` dataclass that controls how
strictly answers are validated. The policy is selected by
`get_policy_for_intent(intent_string)`.

Not all intents require the same rules: a financial calculation demands
strict numeric grounding and a citation for every operand, while a plain
conversation must not be rejected for lacking document evidence.

`ValidationPolicy` is frozen (`@dataclass(frozen=True)`) and validated
in `__post_init__`: `unsupported_numeric_action` and
`missing_citation_action` must each be `"block"` or `"warn"` (constants
`ACTION_BLOCK` / `ACTION_WARN`), otherwise a `ValueError` is raised at
construction time.

## Policy Matrix

The table below reflects the 7 policies constructed in
`validation_policy.py` exactly. Column names use the short policy
language requested by the task; the mapping to dataclass fields is:

- `require_evidence` → `require_evidence`
- `require_citations` → `require_citations`
- `strict_numeric` → `strict_numeric_grounding`
- `unsupported` → `unsupported_numeric_action`
- `missing_citation` → `missing_citation_action`

| Intent | require_evidence | require_citations | strict_numeric | unsupported | missing_citation |
|---|---|---|---|---|---|
| `financial_calculation` | True | True | True | block | block |
| `document_qa` | True | True | True | block | warn |
| `document_summary` | True | False | True | block | warn |
| `multi_document_comparison` | True | True | True | block | block |
| `front_matter` | True | True | False | warn | warn |
| `conversation` | False | False | False | warn | warn |
| `unsupported` | False | False | False | warn | warn |

### Full dataclass fields

For completeness, every field of each `ValidationPolicy`:

| Intent | require_evidence | require_citations | validate_numeric_claims | validate_units | validate_periods | strict_numeric_grounding | unsupported_numeric_action | missing_citation_action |
|---|---|---|---|---|---|---|---|---|
| `financial_calculation` | True | True | True | True | True | True | block | block |
| `document_qa` | True | True | True | True | True | True | block | warn |
| `document_summary` | True | False | True | True | True | True | block | warn |
| `multi_document_comparison` | True | True | True | True | True | True | block | block |
| `front_matter` | True | True | False | False | False | False | warn | warn |
| `conversation` | False | False | False | False | False | False | warn | warn |
| `unsupported` | False | False | False | False | False | False | warn | warn |

## Default Fallback

Unknown / `None` intents fall back to the `document_qa` policy
(`_DEFAULT_POLICY = _DOCUMENT_QA`). This is the conservative choice: it
requires evidence, requires citations, runs all numeric/unit/period
validators, and blocks on unsupported numeric claims. A typo or a new
intent string therefore never weakens validation.

```python
def get_policy_for_intent(intent: str | None) -> ValidationPolicy:
    if not intent:
        return _DEFAULT_POLICY
    return _POLICY_BY_INTENT.get(intent, _DEFAULT_POLICY)
```

## Action Semantics

The two action strings control how a validator sets issue severity:

- **`block`** — the issue is `CRITICAL`; the answer must NOT be returned.
  - `unsupported_numeric_action="block"` + `strict_numeric_grounding=True`
    → `NumericClaimValidator` emits `NUMERIC_UNGROUND` at `CRITICAL`.
  - `missing_citation_action="block"` → `CitationValidator` emits
    `CITATION_MISSING` at `CRITICAL`.
  - The `ResponseValidator` aggregates any `CRITICAL` issue into a
    `BLOCKED` verdict.
- **`warn`** — the issue is `WARNING`; the answer may still pass.
  - `unsupported_numeric_action="warn"` → `NUMERIC_UNGROUND` at `WARNING`
    regardless of `strict_numeric_grounding`.
  - `missing_citation_action="warn"` → `CITATION_MISSING` at `WARNING`.
  - `WARNING`/`INFO`-only issue sets aggregate to `PASSED`.

The intermediate `strict_numeric_grounding` flag only matters when the
action is `block` AND the claim is not strictly warn-able: when
`strict_numeric_grounding=False` and the action is `block`, an
unsupported numeric claim is downgraded from `CRITICAL` to `ERROR`,
which the `ResponseValidator` treats as `REPAIRABLE` (lenient) instead
of `BLOCKED` (strict).

## Policy Properties

- `require_evidence`: if `True`, the absence of any retrieved evidence
  forces `NOT_ANSWERABLE` (no LLM invocation). If `False`
  (conversation/unsupported), the answerability evaluator returns
  `ANSWERABLE` with `reason_codes=("no_retrieval_required",)`.
- `require_citations`: if `True`, numeric/calculation claims must cite
  evidence; otherwise `CITATION_MISSING` is emitted per numeric claim.
  `document_summary` sets this to `False` (summaries aggregate many
  facts, so per-claim citations are not enforced).
- `validate_numeric_claims`: if `True`, `NumericClaimValidator` and
  `UnsupportedClaimValidator` run.
- `validate_units`: if `True`, `UnitPeriodValidator` runs currency
  checks.
- `validate_periods`: if `True`, `UnitPeriodValidator` runs period
  checks. (Both unit and period checks are gated together in practice;
  `front_matter` and `conversation` disable both.)
- `strict_numeric_grounding`: if `True`, ANY unsupported numeric claim
  is `CRITICAL` (`BLOCK`); if `False`, unsupported numerics are `ERROR`
  and may be repairable (lenient path).
- `unsupported_numeric_action`: `block` or `warn`.
- `missing_citation_action`: `block` or `warn`.
- `applies_any_validation` (computed property): `True` if at least one
  of `validate_numeric_claims`, `validate_units`, `validate_periods`, or
  `require_citations` is `True`. When `False`, `ResponseValidator`
  short-circuits to `ValidationStatus.NOT_APPLICABLE`.

## Conversation / Unsupported Special Case

`conversation` and `unsupported` intents have
`applies_any_validation == False`. `ResponseValidator._validate_inner`
returns `ValidationStatus.NOT_APPLICABLE` immediately — no validators
run, no claims are extracted. This ensures conversational responses are
not rejected for lacking document evidence. The helper
`validation_status_for_conversation()` returns
`ValidationStatus.NOT_APPLICABLE` for the same purpose.

## How Policies Are Consumed

1. **`AnswerabilityEvaluator`** reads `require_evidence` to decide
   whether empty/insufficient evidence blocks generation.
2. **`ResponseValidator`** reads `applies_any_validation` to decide
   whether to short-circuit to `NOT_APPLICABLE`, then passes the policy
   to each sub-validator.
3. **`NumericClaimValidator`** reads `validate_numeric_claims`,
   `unsupported_numeric_action`, and `strict_numeric_grounding`.
4. **`UnitPeriodValidator`** reads `validate_units` and
   `validate_periods`.
5. **`CitationValidator`** reads `require_citations` and
   `missing_citation_action`.
6. **`UnsupportedClaimValidator`** reads `validate_numeric_claims`.
7. **`ResponseValidator._determine_status`** reads
   `strict_numeric_grounding` to decide whether `ERROR` issues produce
   `BLOCKED` (strict) or `REPAIRABLE` (lenient).

## Test Coverage

`tests/validation/test_validation_policy.py` covers: each intent maps to
its expected frozen policy; unknown/`None` intents fall back to
`document_qa`; the policy is frozen (immutable); invalid action strings
raise `ValueError`; `applies_any_validation` is correct per intent; and
the conversation special-case helper returns `NOT_APPLICABLE`.
