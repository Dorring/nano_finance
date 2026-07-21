# Phase 4: Grounding and Validation

## Overview

Phase 4 adds a two-stage trustworthy answer mechanism to the RAG
orchestrator: **pre-generation answerability evaluation** and
**post-generation response validation with deterministic repair**.

The goal is to prevent the LLM from producing ungrounded numeric claims,
missing citations, period mismatches, and unsupported metrics — without
retraining the model, without LLM-as-Judge, and without adding new
calculation formulas.

Key design decisions:
- **No model retraining**: validation is purely deterministic.
- **No LLM-as-Judge**: all validators use regex extraction and `Decimal`
  comparison; no model is asked to judge another model's output.
- **No new formulas**: Phase 3's calculation formulas remain unchanged.
- **Fail-closed**: when a validator cannot complete, the result is
  `FAILED` (never `PASSED`); the answer is replaced with a safe fallback.
- **At-most-once repair**: a single deterministic repair attempt strips
  sentences containing ungrounded numeric claims. If repair fails, a safe
  fallback is used.
- **Additive API**: `AnswerResult` gains `answerability`, `validation`,
  and `repair` optional fields (default `None`). They are emitted only
  when non-`None` (`response_model_exclude_none`).

## Architecture

### Layer Dependency

```
domain ← validation ← application ← services/api
```

- `src/domain/` — pure data objects (frozen dataclasses, stdlib only)
- `src/validation/` — validators, pipeline, repair (depends on `domain` only)
- `src/application/` — orchestrator (depends on `domain`, `validation`,
  `retrieval`, `generation`)
- `src/services/` — facade and HTTP/SSE wiring (depends on `application`)

`src/validation/` must NOT import from `src.services` or `src.application`.

### The 12 Validation Modules

| # | Module | Role |
|---|---|---|
| 1 | `validation/answerability.py` → `AnswerabilityEvaluator` | Pre-generation gate. Inspects sufficiency, calculation, evidence, and requested documents to emit one of four `AnswerabilityStatus` verdicts. Never calls the LLM. |
| 2 | `validation/claim_extractor.py` → `ClaimExtractor` | Regex-based extraction of 5 deterministic claim types (`amount`, `percent`, `ratio`, `period`, `citation_ref`) from the generated answer. Free-form propositions are NOT extracted. |
| 3 | `validation/numeric_claim_validator.py` → `NumericClaimValidator` | Checks that each numeric claim's value + metric + period all appear on the SAME evidence chunk (`NUMERIC_UNGROUND`). |
| 4 | `validation/unit_period_validator.py` → `UnitPeriodValidator` | Checks period and currency consistency against evidence (`PERIOD_MISMATCH`, `CURRENCY_MISMATCH`). |
| 5 | `validation/citation_validator.py` → `CitationValidator` | Validates source objects (chunk_id / document_name / page), citation presence, resolvability, and claim support. |
| 6 | `validation/calculation_validator.py` → `CalculationValidator` | Checks that answer values for a calculated metric match the `CalculationResult` (`CALCULATION_MISMATCH`). |
| 7 | `validation/unsupported_claim_validator.py` → `UnsupportedClaimValidator` | Flags numeric claims whose metric keyword is absent from all evidence (`UNSUPPORTED_CLAIM`). |
| 8 | `validation/response_validator.py` → `ResponseValidator` | Central aggregator that runs all validators, suppresses `NUMERIC_UNGROUND` for calculation-supported claims, and produces a single `ValidationResult`. Fail-closed on exception. |
| 9 | `validation/response_repair.py` → `ResponseRepair` | Deterministic repair (strip ungrounded-claim sentences) + safe fallback messages. Never calls the LLM. |
| 10 | `validation/validation_pipeline.py` → `GroundedValidationPipeline` | Top-level facade combining `AnswerabilityEvaluator` + `ResponseValidator` + `GroundedResponseResult` aggregation. |
| 11 | `validation/validation_policy.py` → `ValidationPolicy` | 7 frozen per-intent policies + `get_policy_for_intent` factory (unknown intents fall back to `document_qa`). |
| 12 | `domain/validation.py` | Dependency-free typed boundary: `AnswerabilityStatus`, `ValidationStatus`, `ValidationSeverity`, `ValidationIssue`, `AnswerabilityResult`, `ValidationResult`, `GroundedResponseResult`, `ExtractedClaim`. |

> The `UnsupportedClaimValidator` (module 7) is counted separately from the
> `NumericClaimValidator` (module 3) because it checks *metric presence*
> rather than *value grounding*. Together with `ClaimExtractor` and the
> domain types, the pipeline is composed of 12 modules.

### Module Structure

```
src/
├── domain/
│   └── validation.py           # AnswerabilityStatus, ValidationStatus,
│                               # ValidationSeverity, AnswerabilityResult,
│                               # ValidationIssue, ValidationResult,
│                               # GroundedResponseResult, ExtractedClaim
├── validation/
│   ├── validation_policy.py    # 7 per-intent frozen policies
│   ├── answerability.py        # Pre-generation AnswerabilityEvaluator
│   ├── claim_extractor.py      # Regex-based claim extraction (5 types)
│   ├── numeric_claim_validator.py   # NUMERIC_UNGROUND detection
│   ├── unit_period_validator.py     # PERIOD/CURRENCY_MISMATCH
│   ├── citation_validator.py        # CITATION_* codes
│   ├── calculation_validator.py     # CALCULATION_MISMATCH
│   ├── unsupported_claim_validator.py  # UNSUPPORTED_CLAIM
│   ├── response_validator.py        # Central ResponseValidator (fail-closed)
│   ├── response_repair.py           # Deterministic repair + safe fallback
│   └── validation_pipeline.py       # GroundedValidationPipeline facade
├── application/
│   └── rag_orchestrator.py     # Integrated answerability + validation + repair
├── services/
│   └── rag_engine.py           # enable_validation_pipeline=True by default
├── models/
│   └── schemas.py              # AnswerabilityResponse, ValidationResponse,
│                               # RepairResponse, QueryResponse fields
└── main.py                     # /query + /query/stream wiring
```

## The 4 Answerability Statuses

`AnswerabilityEvaluator.evaluate()` returns exactly one of
(`src/domain/validation.py`):

| Status | Meaning | LLM Invoked? | Orchestrator Action |
|---|---|---|---|
| `ANSWERABLE` | Evidence is sufficient and all requested documents are covered. | Yes | Proceed to generation, then validate. |
| `PARTIALLY_ANSWERABLE` | Some requested documents are missing but enough evidence exists for a limited answer. | Yes | Proceed to generation, apply restricted prefix/suffix, then validate. |
| `NOT_ANSWERABLE` | Evidence is missing, empty, or below the sufficiency threshold. | **No** | Return deterministic refusal; bypass LLM. |
| `CALCULATION_BLOCKED` | The Phase 3 calculation pipeline returned `BLOCKED` or `FAILED`. | **No** | Return Phase 3 safe calculation response; bypass LLM. |

Evaluation order inside `AnswerabilityEvaluator.evaluate()`:
1. Calculation `BLOCKED` / `FAILED` → `CALCULATION_BLOCKED` (FAILED
   calculations are folded into `CALCULATION_BLOCKED` to avoid
   reintroducing numeric hallucinations via LLM free-form text).
2. Intent with `require_evidence=False` (conversation/unsupported) →
   `ANSWERABLE` (`reason_codes=("no_retrieval_required",)`).
3. No evidence at all → `NOT_ANSWERABLE` (`reason_codes=("no_evidence",)`).
4. Sufficiency evaluator returns `False` → `NOT_ANSWERABLE`
   (`reason_codes=("insufficient_evidence",)`).
5. Some requested documents have no evidence chunks →
   `PARTIALLY_ANSWERABLE` (`reason_codes=("missing_documents",)`).
6. Otherwise → `ANSWERABLE`.

## Validation Pipeline Flow

```
User Question
     │
     ▼
┌─────────────────────┐
│  Intent Classifier   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Retrieval + Context │
│  Sufficiency Eval    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐    BLOCKED/FAILED
│  Calculation Pipeline├──────────────────────┐
│  (Phase 3)           │ EXECUTED             │
└─────────┬───────────┘ ─────────────────────┘
          │
          ▼
┌─────────────────────────────┐  NOT_ANSWERABLE ──→ Safe refusal (no LLM)
│  PRE-GENERATION:            │  CALCULATION_BLOCKED ──→ Phase 3 safe response
│  AnswerabilityEvaluator     │  PARTIALLY_ANSWERABLE ──→ LLM + restricted prefix
│  (Phase 4)                  │  ANSWERABLE ──→ continue
└─────────┬───────────────────┘
          │  (LLM bypassed on NOT_ANSWERABLE / CALCULATION_BLOCKED)
          ▼
┌─────────────────────┐
│  LLM Generation      │
│  (or deterministic)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────┐  PASSED ──→ answer as-is
│  POST-GENERATION:           │  REPAIRABLE ──→ one deterministic repair, revalidate
│  ResponseValidator          │  BLOCKED ──→ safe fallback
│  (Phase 4)                  │  FAILED ──→ safe fallback (fail-closed)
│  + ResponseRepair           │  NOT_APPLICABLE ──→ answer as-is (conversation)
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────┐
│  AnswerResult        │  + answerability / validation / repair dicts
│  (additive fields)   │
└─────────┬───────────┘
          │
          ▼
   HTTP /query or SSE /query/stream
   (response_model_exclude_none)
```

### ALL Retrieval Paths Go Through Validation

The orchestrator applies `_validate_and_repair_once` to every generation
path that is not already answerability-blocked:

| Path | Answerability | Validation |
|---|---|---|
| Calculation `EXECUTED` | `ANSWERABLE` (calc does not block) | Yes — rendered calculation answer is validated. |
| Calculation `BLOCKED` | `CALCULATION_BLOCKED` | Bypassed (deterministic safe response; validation NOT_APPLICABLE). |
| Calculation `FAILED` | `CALCULATION_BLOCKED` | Bypassed (deterministic safe response; validation NOT_APPLICABLE). |
| Front Matter | Evaluated with `intent="front_matter"` | Yes — `_validate_and_repair_once` runs on the front-matter answer. |
| Deterministic context answer | `ANSWERABLE` / `PARTIALLY_ANSWERABLE` | Yes. |
| LLM answer | `ANSWERABLE` / `PARTIALLY_ANSWERABLE` | Yes. |
| `NOT_ANSWERABLE` | `NOT_ANSWERABLE` | Bypassed (safe refusal). |
| Conversation / unsupported | `ANSWERABLE` | `NOT_APPLICABLE` (policy disables all validators). |

## The `_validate_and_repair_once` Pattern

Implemented in `RAGOrchestrator._validate_and_repair_once`
(`src/application/rag_orchestrator.py`). This is the single unified
validation + repair orchestration used by every generation path.

```
1. Initial validation  (ResponseValidator.validate, fail-closed)
       │
       ├── PASSED / NOT_APPLICABLE ──→ return answer as-is (no repair)
       ├── BLOCKED / FAILED ──→ safe fallback immediately (NO repair)
       └── REPAIRABLE ──→ continue to step 2
2. ONE repair  (ResponseRepair.repair — strips ungrounded-claim sentences)
       │
       ├── fallback_used (empty / unrepairable) ──→ return fallback
       └── repaired answer ──→ continue to step 3
3. Revalidate the repaired answer  (at most ONE revalidation, fail-closed)
       │
       ├── PASSED / NOT_APPLICABLE ──→ return repaired answer
       └── BLOCKED / FAILED / REPAIRABLE ──→ safe fallback (NO second repair)
```

Invariants enforced by this method:
- **At most ONE repair** — there is no second repair branch.
- **At most ONE revalidation** — the repaired answer is validated exactly
  once; a still-failing result goes straight to fallback.
- **No LLM in repair** — `ResponseRepair.repair` is pure string surgery.
- **Fail-closed on validator exception** — both the initial validation
  and the revalidation wrap `validate_response` in `try/except`; any
  exception yields `ValidationStatus.FAILED` with a single
  `VALIDATOR_EXCEPTION` CRITICAL issue (never `PASSED`).
- The API `validation` field reflects the **revalidation** result when
  repair was attempted, otherwise the initial validation result. The
  **initial** validation result is preserved separately in trace as
  `initial_validation`.

## PARTIALLY_ANSWERABLE Restricted Prefix

When `AnswerabilityEvaluator` returns `PARTIALLY_ANSWERABLE`, the
orchestrator wraps the generated answer with a fixed Chinese prefix and
suffix (`_apply_partial_prefix`) before validation, so a partial answer
can never be mistaken for a complete one:

```
根据当前检索到的资料，只能确认以下部分：
{answer}
未找到或无法验证：{missing_requirements}
```

- `missing_requirements` is the `"; "`-joined
  `AnswerabilityResult.missing_requirements` tuple (e.g.
  `"document: peer_report.pdf"`).
- If `missing_requirements` is empty, the suffix uses the fixed string
  `"部分请求的文档或数据"`.

## Trace Privacy

Phase 4 redacts content from trace storage to prevent answer / context /
internal-message leakage (`src/domain/validation.py` and
`src/application/rag_orchestrator.py`):

- **`final_context=None`** and **`answer=None`** in `trace_data`. The
  orchestrator stores only `context_length`, `context_sha256`,
  `answer_length`, and `answer_sha256` (truncated SHA-256, first 16 hex
  chars).
- `ValidationIssue.to_trace_dict()` omits the internal `message` and full
  `claim_text`. It emits:
  - `message_hash` — SHA-256 of the internal message (first 16 hex chars).
  - `claim_excerpt` — the claim text, control-char cleaned, **truncated to
    80 characters max**.
  - `code`, `severity`, `evidence_ids`.
- `ValidationResult.to_trace_dict()` stores `repaired` as a **boolean
  flag**, never the repaired answer text.
- `AnswerabilityResult.to_trace_dict()` includes `best_score` /
  `average_score` for debugging but the **public** dict omits them.
- `RepairResult.to_trace_dict()` includes `repair_notes` and
  `answer_length`, never the answer text.

Replay-from-trace is therefore NOT possible by default (content is
redacted); see `phase4-known-limitations.md`.

## Citation Validation Against Real Sources

`CitationValidator` validates four dimensions, using the `sources` tuple
(the source objects returned to the API consumer) plus the evidence set:

1. **Source object validity** — each source's `chunk_id` must exist in
   the retrieved evidence; `document_name` and `page` must match the
   evidence chunk's fields. Document name matching is lenient: it
   lowercases, strips file extensions, and strips `user_<id>_` prefixes.
2. **Citation presence** — when `require_citations=True`, numeric claims
   with no citation references at all are flagged `CITATION_MISSING`.
3. **Citation resolvability** — `[1]` resolves to `sources[0]`;
   `[doc.pdf, p.12]` resolves only if both document name AND page match
   a source. Unresolved citations are flagged.
4. **Claim support** — for a numeric claim with a numeric citation, the
   cited source's evidence chunk must contain the claim's value;
   otherwise `CITATION_DOES_NOT_SUPPORT_CLAIM`.

## Numeric Grounding: Same-Evidence-Chunk Requirement

`NumericClaimValidator._find_supporting_evidence` requires that a
claim's **value + metric + period** ALL appear on the **same evidence
chunk**:

1. The claim's value (in some textual representation — plain,
   comma-formatted, or scale-suffixed) must appear in the chunk text.
2. If the claim has a metric, the same chunk must contain that metric
   keyword (or a known alias from `_METRIC_CANONICAL`).
3. If the claim has a period, the same chunk must contain that year
   (regex `20[0-2]\d | 19[89]\d`).

A correct value appearing in a *different* chunk from its metric or
period is NOT grounded — this prevents cross-contamination where a
number for the wrong year/metric would otherwise pass.

Claims whose metric and value match an `EXECUTED` `CalculationResult`'s
`target_metric` and `value` (within a `0.01` `Decimal` tolerance, with
ratio/percent normalization) are exempt from `NUMERIC_UNGROUND` — the
deterministic calculation IS the evidence.

## Key Invariants

1. **No LLM in repair** — `ResponseRepair.repair` performs only
   regex-based sentence stripping and fallback selection.
2. **At most ONE repair** per answer.
3. **At most ONE revalidation** of the repaired answer.
4. **Fail-closed on validator exception** — both `ResponseValidator`
   (emits `VALIDATOR_ERROR`) and the orchestrator wrapper
   (`_validate_and_repair_once` emits `VALIDATOR_EXCEPTION`) convert any
   raised exception into `ValidationStatus.FAILED` with a single
   CRITICAL issue. The answer is never returned.
5. **NOT_ANSWERABLE / CALCULATION_BLOCKED never invoke the LLM.**
6. **Blocked / failed answers are never streamed as partial LLM tokens**
   — see `phase4-streaming-safety.md`.
7. **Trace never stores full context or answer text** — only hashes and
   lengths.
8. **Public API responses never expose** internal `message`,
   `evidence_ids`, `repair_notes`, `best_score`, or `average_score`.

## Testing

- **Validation tests**: `tests/validation/` (answerability, claim/numeric
  validation, citation + calculation validation, metric/period grounding,
  partial answerability, repair + revalidation, response validation
  pipeline, source object validation, trace content redaction,
  validation policy, validation domain, validation artifacts, front
  matter wiring, calculation validation wiring, grounded response e2e,
  validation HTTP/SSE, phase 4 baseline characterization).
- **Phase 3 regression**: finance tests remain green.
- **HTTP/SSE**: real FastAPI `TestClient` tests with validation fields
  present, absent, and blocked scenarios.
- **Trace redaction**: regression tests asserting `final_context=None`,
  `answer=None`, `message_hash`, `claim_excerpt` (max 80 chars), and no
  `str(exc)` in streaming errors.
