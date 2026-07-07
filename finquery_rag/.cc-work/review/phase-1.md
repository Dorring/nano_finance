# Phase 1 Review Record

## Commit
- SHA: 99a77eb
- Title: WIP: rag phase 1 lifecycle observability
- Branch: cc/rag-production-review

## Modified Files
- backend/src/services/document_registry.py (NEW) - document lifecycle registry with state machine
- backend/src/services/trace.py (NEW) - structured query tracing with sanitization
- backend/src/main.py - integrated registry into upload flow (dedup, state tracking)
- backend/tests/test_phase1.py (NEW) - 31 tests covering registry and trace

## Design Decisions
1. DocumentRegistry uses standalone SQLite (not shared with BM25) for isolation
2. State machine: pending -> parsing -> indexing -> ready/failed, with retry from failed
3. File hash dedup skips re-processing identical files; content hash dedup catches same-content-different-filename
4. TraceLogger sanitizes phone numbers/SSNs by default; supports sampling rate
5. Upload flow: compute hash -> check dedup -> register -> process -> mark ready

## Test Results
```
70 passed in 4.11s (39 Phase 0 + 31 Phase 1)
AST parse: all 4 files OK
git diff --check: EXIT: 0
```

## Known Risks
- Embedding model not changed (still all-MiniLM-L6-v2, weak on Chinese)
- Trace logger not yet integrated into query endpoint (Phase 2+)
- No cross-encoder reranker yet
