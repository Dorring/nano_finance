# Phase 2 Deferred Findings

This document records issues identified during Phase 2 that were NOT modified
in this phase. They are documented here for traceability and will be addressed
in subsequent phases.

## 1. Intent Rule Limitations

**Location**: `src/services/intent.py`, `src/retrieval/query_processor.py`

The current intent classifier uses keyword-based rules. Limitations:
- No ML-based classification; financial vs. conversational disambiguation
  relies on a static keyword list.
- `requires_retrieval` is determined by keyword absence, not semantic analysis.
- CJK financial terms are hardcoded; new languages require code changes.

**Phase 3+ action**: Consider pluggable intent provider protocol.

## 2. Sufficiency Threshold Issues

**Location**: `src/retrieval/context_builder.py::EvidenceSufficiencyEvaluator`

- `is_sufficient` uses a fixed score threshold (no per-query adaptation).
- `confidence` computation is a simple average of chunk scores.
- No calibration against actual answer quality.

**Phase 3+ action**: Replace with `Answerability` domain object that
incorporates calculation requirements.

## 3. Deterministic Numeric Extraction Limitations

**Location**: `src/generation/deterministic_answers.py`

- Regex-based numeric extraction misses formatted numbers (e.g., "1,234.56").
- No unit normalization (millions vs. billions).
- Table-cell extraction does not handle merged cells.

**Phase 3 action**: Replace with `financial_tools.py` calculation pipeline.

## 4. Model Parameter Annotation Inconsistency

**Location**: `src/generation/llm_gateway.py`

- `max_new_tokens` is a constructor param but not consistently used.
- `temperature` is hardcoded in some call sites.
- No `top_p` / `frequency_penalty` configuration.

**Phase 3+ action**: Unify generation parameters in a `GenerationConfig` dataclass.

## 5. Sealed Test Suite Not Established

**Location**: `tests/`

Golden compatibility tests exist for response shape, but:
- No sealed test corpus of (question, expected_answer) pairs.
- Characterization tests capture current behavior, not desired behavior.
- No regression threshold for semantic drift.

**Phase 6 action**: Build sealed eval set as part of evaluation framework.

## 6. Financial Tools Not Integrated

**Status**: Deferred to Phase 3 (per plan).

`src/services/financial_tools.py` exists but is not called from the
orchestrator. Phase 3 will insert it between retrieval and generation.

## 7. Answer Validation Not Integrated

**Status**: Deferred to Phase 4 (per plan).

`src/services/answer_validation.py` exists but is not called from the
orchestrator. Phase 4 will insert it after generation.

## 8. Orchestrator Accesses RetrievalPipeline Private Field

**Location**: `src/application/rag_orchestrator.py:243`

```python
retrieval_debug=dict(self._retrieval_pipeline._last_retrieval_debug),
```

The orchestrator reads `_retrieval_pipeline._last_retrieval_debug` (a private
attribute) to populate `retrieval_debug` in the answer. This is a minor
encapsulation leak.

**Phase 3 action**: Add a public `last_debug` property to `RetrievalPipeline`.

## 9. QueryProcessor Static Methods Still Instantiated Per-Call

**Location**: `src/services/rag_engine.py:390, 406`

```python
return QueryProcessor().should_try_deterministic_factual_answer(query)
return QueryProcessor().is_numeric_query(query)
```

Two static helper methods are called via `QueryProcessor()` instantiation
rather than as `@staticmethod` or module-level functions. This creates
unnecessary object allocation.

**Phase 3 action**: Convert to `@staticmethod` or move to module-level functions.

## 10. Long Function: RAGOrchestrator.answer

**Location**: `src/application/rag_orchestrator.py:79-247`

The `answer` method is ~170 lines. While the early-return branches are
extracted to static helpers, the main FULL path is still a single long
function with nested conditionals.

**Phase 3 action**: Extract FULL path into `_execute_full_retrieval` method
once calculation steps are inserted (avoiding premature abstraction).
