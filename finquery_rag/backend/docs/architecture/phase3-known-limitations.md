# Phase 3 Known Limitations

This document explicitly lists the known limitations of the Phase 3
deterministic financial calculation pipeline. These are design boundaries,
not bugs — each limitation exists to keep calculations deterministic and
auditable.

---

## 1. Model Capabilities

- **No Tool Calling SFT**: The model has not been fine-tuned for tool
  calling / function calling. It cannot decide which operation to invoke
  or generate structured tool-call arguments.
- **Not Native Function Calling**: The current implementation is a
  system-orchestrated Tool-Augmented RAG, not model-native Function
  Calling. The operation router, evidence extractor, and executor are
  deterministic Python modules — the model is never asked to choose an
  operation or produce calculation code.

## 2. Calculation Scope

- **Fixed formulas only**: The pipeline supports 9 fixed formula
  operations (difference, growth_rate, percentage_share, sum, average,
  gross_margin, net_margin, debt_ratio, scale_conversion). Each has a
  single versioned implementation.
- **No free-form Python**: The pipeline does not execute arbitrary
  model-generated Python code. `eval`, `exec`, and `compile` are
  prohibited. All calculations use `decimal.Decimal` with fixed formulas.
- **No currency conversion**: Scale conversion handles unit scale only
  (e.g., 万 → 亿, million → billion). Currency conversion (e.g., USD → CNY)
  is explicitly rejected with `CURRENCY_NOT_SUPPORTED`.
- **ROE not implemented**: Return on Equity (`net_income / equity`) is
  not in the v1 registry. Deferred to a future phase.
- **CAGR not implemented**: Compound Annual Growth Rate is not in the v1
  registry. Deferred to a future phase.

## 3. Scale Conversion

- **Fully implemented (Option A)**: `SCALE_CONVERSION` is registered and
  executable. The router extracts the target scale from the question,
  the extractor infers the source scale from evidence, and the executor
  calls `convert_scale()`.
- **Missing scale → BLOCKED**: If the source or target scale cannot be
  determined, the calculation is blocked with `UNIT_AMBIGUOUS` (no LLM
  fallback).
- **Currency → BLOCKED**: If a currency keyword is detected (e.g., "美元",
  "rmb", "usd"), the calculation is blocked with
  `CURRENCY_NOT_SUPPORTED`.
- **Percentage rejected**: Percentage values are not accepted for scale
  conversion.

## 4. Evidence and Ambiguity

- **Ambiguous evidence → BLOCKED**: If the evidence extractor cannot
  find a required operand role, or finds ambiguous matches, the plan is
  blocked with `INSUFFICIENT_OPERANDS` or a role-specific reason. The LLM
  is not asked to "guess" the missing value.
- **Single-document extraction**: The extractor operates on the merged
  evidence set from retrieval. Cross-table multi-step calculations
  (e.g., extracting a value from a balance sheet and another from an
  income statement in different documents, then combining them) are not
  supported in v1.

## 5. LLM Interaction

- **FAILED does not fall back to LLM**: When an internal calculation
  error occurs (e.g., primitive exception, unknown operation, registry
  failure), the pipeline returns a `FAILED` result with a safe failure
  message. The LLM is **not** asked to recompute. This prevents
  numerical hallucination through the LLM backdoor.
- **Only NOT_APPLICABLE continues to LLM**: When the router determines
  the question is not a calculation (e.g., a qualitative question), the
  result is `NOT_APPLICABLE` and the normal RAG/LLM flow continues
  unchanged.

## 6. Answer Validation

- **No sealed-test validation**: Complete answer validation against
  sealed test sets belongs to Phase 5. Phase 3 validates formula
  correctness through unit, integration, and contract tests only.
- **No Phase 4 integration**: Phase 4 (full answer validation and
  scoring) has not been started. Phase 3 does not depend on Phase 4
  components.

## 7. Testing Boundaries

- **12 skipped tests**: 11 tests are skipped because the `jose` library
  is not installed in the test environment (JWT-related auth tests).
  1 test is skipped because it requires a full application stack
  (database, vector store, embedding model) that is not available in
  the unit test environment.
- **Sealed Test belongs to Phase 5**: No sealed (golden) test is
  constructed in Phase 3.

## 8. Future Phases

- **Phase 4** (not started): Full answer validation, scoring, and
  end-to-end quality gates.
- **Phase 5** (not started): Sealed test construction and long-term
  regression protection.
