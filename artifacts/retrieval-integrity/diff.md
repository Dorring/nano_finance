# Phase 1: Retrieval Integrity Behavioral Diff

## Before (contaminated)
- 5 hardcoded methods in rag_engine.py mapping doc names to page numbers
- supporting_source_page used as ranking signal in 3 methods
- _ensure_supporting_sources propagated eval flags to final citations
- ~2020 lines in rag_engine.py

## After (clean)
- 5 methods removed (replaced with Phase 1 comments)
- supporting_source_page: 0 references in production code
- Oracle context isolated in src/evaluation/oracle_context.py
- ~1846 lines in rag_engine.py (174 lines removed)

## Expected Behavioral Changes
- Retrieval results for known eval documents will DIFFER
- Recall@K may DECREASE (was artificially boosted)
- Citation precision may DECREASE (pages no longer force-injected)
- These changes are CORRECT - the old metrics were contaminated

## Test Updates
- 7 tests converted from testing removed methods to verifying removal
- All 583 existing tests pass
- 2 new test directories: tests/integrity/ (4 tests), tests/architecture/ (from Phase 0)

## New Capabilities
- Leakage scanner: scripts/check_eval_leakage.py (CI-ready)
- Oracle context: src/evaluation/oracle_context.py (offline-only)
- Contamination notice: eval/CONTAMINATION_NOTICE.md

## WARNING
Do not restore any removed methods to improve metrics.
Clean retrieval metrics are expected to be lower than contaminated metrics.
