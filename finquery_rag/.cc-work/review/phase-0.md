# Phase 0 Review Record

## Commit
- SHA: cb7b640
- Title: WIP: rag phase 0 safety baseline
- Branch: cc/rag-production-review

## Modified Files
- backend/src/services/chunk_id.py (NEW) - shared chunk ID generation/validation
- backend/src/services/retrieval.py - BM25 FTS5 standalone, doc_name column, fail-closed
- backend/src/services/vector_store.py - fail-closed user_id, scoped IDs at storage boundary
- backend/src/services/ingest.py - uses make_chunk_id
- backend/src/main.py - await, HTTPException preservation, tempfile, clear_all_for_user
- backend/tests/test_phase0.py (NEW) - 39 tests
- backend/conftest.py (NEW) - pytest ignore cache dirs
- backend/pyproject.toml - pytest config added

## Design Decisions
1. Chunk IDs scoped as `user_{user_id}_{doc_name}::{suffix}` for tenant isolation
2. BM25 FTS5 changed from content-backed to standalone table for proper delete semantics
3. All query/delete/list operations fail-closed when user_id is None
4. Schema migration adds doc_name column and backfills from metadata_json

## Test Results
```
39 passed in 2.56s
compileall: EXIT: 0
git diff --check: EXIT: 0
```

## Known Risks
- Phase 0 uses standalone BM25 table; Phase 1 document registry will need to coordinate deletes
- No structured tracing yet (deferred to Phase 1)
