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
- **No LLM-as-Judge**: all validators use regex extraction and Decimal
  comparison; no model is asked to judge another model's output.
- **No new formulas**: Phase 3's 9 calculation formulas remain unchanged.
- **Fail-closed**: when a validator cannot complete, the result is
  ``FAILED`` (never ``PASSED``); the answer is replaced with a safe
  fallback.
- **At-most-once repair**: a single deterministic repair attempt strips
  sentences containing ungrounded numeric claims. If repair fails, a safe
  fallback is used.
- **Additive API**: ``AnswerResult`` gains ``answerability``,
  ``validation``, and ``repair`` optional fields (default ``None``).
  ``to_legacy_dict()`` emits them only when non-``None``.

- **Branch**: `feat/nf-04-grounding-and-validation`
- **Base commit**: `0da5898` (Phase 3 merge to master)
- **Commits**: 12

## Architecture

### Layer Dependency

```
domain ← validation ← application ← services/api
```

- `src/domain/` — pure data objects (frozen dataclasses, stdlib only)
- `src/validation/` — validators, pipeline, repair (depends on `domain` only)
- `src/application/` — orchestrator (depends on `domain`, `validation`, `retrieval`, `generation`)
- `src/services/` — facade and HTTP/SSE wiring (depends on `application`)

`src/validation/` must NOT import from `src.services` or `src.application`.

### Module Structure

```
src/
├── domain/
│   └── validation.py           # AnswerabilityStatus, ValidationStatus,
│                               # ValidationSeverity, AnswerabilityResult,
│                               # ValidationIssue, ValidationResult,
│                               # GroundedResponseResult
├── validation/
│   ├── validation_policy.py    # 7 per-intent frozen policies
│   ├── answerability.py        # Pre-generation AnswerabilityEvaluator
│   ├── claim_extractor.py      # Regex-based claim extraction (5 types)
│   ├── numeric_claim_validator.py   # NUMERIC_UNGROUND detection
│   ├── unit_period_validator.py     # PERIOD/CURRENCY/UNIT_MISMATCH
│   ├── citation_validator.py        # CITATION_MISSING/UNRESOLVED
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

### Pipeline Flow

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
│  Calculation Pipeline├──────────────────────┐  (Phase 3 bypass)
│  (Phase 3)           │ EXECUTED             │
└─────────┬───────────┘ ─────────────────────┘
          │ NOT_APPLICABLE
          ▼
┌─────────────────────────────┐
│  PRE-GENERATION:             │  NOT_ANSWERABLE ──→ Safe refusal (no LLM)
│  AnswerabilityEvaluator      │  CALCULATION_BLOCKED ──→ Phase 3 safe response
│  (Phase 4)                   │  ANSWERABLE ──→ continue
└─────────┬───────────────────┘
          │
          ▼
┌─────────────────────┐
│  LLM Generation      │
│  (or deterministic)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────┐
│  POST-GENERATION:            │  PASSED ──→ answer as-is
│  ResponseValidator           │  REPAIRABLE ──→ one deterministic repair
│  (Phase 4)                   │  BLOCKED ──→ safe fallback
│  + ResponseRepair            │  FAILED ──→ safe fallback (fail-closed)
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

### Orchestrator Insertion Points

1. **Pre-generation** (after context build, before LLM):
   `AnswerabilityEvaluator.evaluate()` runs. If `NOT_ANSWERABLE`, the LLM
   is bypassed and a deterministic refusal is returned.

2. **Post-generation** (after LLM/deterministic answer, before trace):
   `ResponseValidator.validate()` runs on the generated answer. If
   `BLOCKED` or `FAILED`, `ResponseRepair.repair()` replaces the answer
   with a safe fallback. If `REPAIRABLE`, a single deterministic repair
   is attempted (strip ungrounded numeric sentences).

### Calculation-Supported Claim Suppression

Claims whose metric and value match an `EXECUTED` `CalculationResult`'s
`target_metric` and `value` are exempt from `NUMERIC_UNGROUND` issues —
the calculation IS the evidence.

## Testing

- **Validation tests**: `tests/validation/` (189 tests)
- **Baseline characterization**: 12 pre-validation invariants
- **E2E regression**: 18 grounded response scenarios
- **HTTP/SSE**: 14 endpoint tests with real FastAPI TestClient
- **Phase 3 regression**: all 404 finance tests remain green
- **Full suite**: 1383 passed, 0 failed, 0 errors
