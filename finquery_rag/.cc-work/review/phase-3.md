# Phase 3: Answer Reliability

## Commit
- SHA: `c28dcc9`
- Branch: `cc/rag-production-review`
- Title: `WIP: rag phase 3 answer reliability`

## Modified Files
- `backend/src/services/rag_engine.py` — answer validation, context sufficiency, confidence scoring
- `backend/tests/test_phase3.py` — 23 tests for Phase 3

## Design Decisions

### Answer Validation (`_validate_answer`)
- Strips model artifacts: `<|end|>`, `</s>`, `[END]`, `[/INST]`
- Returns refusal message if answer is empty or near-empty (< 10 chars after cleanup)
- Truncates overly long answers at `max_new_tokens * 4` chars (word-boundary safe)
- Called in `generate_answer()` after LLM returns, before returning to caller

### Context Sufficiency Check (`_check_context_sufficiency`)
- Returns `(is_sufficient: bool, best_score: float, avg_score: float)`
- Sufficiency threshold: `best_score >= 0.15`
- Called in `query()` after retrieval, before context building
- Provides signal to callers for conditional answer handling

### Confidence Scoring (`_compute_confidence`)
- Returns float in [0.0, 1.0]
- Formula: `0.7 * best_score + 0.3 * avg_score` (weighted blend)
- Bounded to [0.0, 1.0] via `min(max(...))`
- Included in query result dict as `confidence` and `context_sufficient`

### Query Result Enrichment
- `query()` now returns `confidence` and `context_sufficient` fields
- Backward-compatible: callers that don't read these keys are unaffected
- Conversational queries (greetings, etc.) do not include these fields

## Test Results
- 436 tests collected, all passed (includes all phases 0-3)
- 23 new Phase 3 tests across 5 classes:
  - TestAnswerValidation (7 tests): empty, None, artifacts, near-empty, normal, truncation
  - TestContextSufficiency (5 tests): empty, high, low, boundary, just-below
  - TestConfidenceScore (5 tests): empty, high, low, bounded, weighted
  - TestGenerateAnswerValidation (3 tests): empty context, normal, artifact stripping
  - TestQueryReturnsConfidence (1 test): conversational query structure

## Pre-commit Checks
- ✅ pytest: 436/436 passed
- ✅ AST parse: all files OK
- ✅ git diff --check: clean
- ✅ staged files: only finquery_rag/

## Known Risks
- Sufficiency threshold (0.15) is a heuristic — may need tuning for specific embedding models
- Confidence formula weights (0.7/0.3) are initial defaults — may need calibration
- `generate_answer_stream` does not apply `_validate_answer` (streaming constraint)
- Long answer truncation at word boundary may break mid-sentence for non-space-delimited languages
