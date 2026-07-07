# Phase 4 Review — Session Memory (会话能力)

- **Commit:** `65cf3b7`
- **Date:** 2026-07-07
- **Branch:** `cc/rag-production-review`

## Scope

Phase 4 implements short-term conversation memory for multi-turn dialogue, following the principle that historical model answers NEVER enter retrieval context as financial facts.

### Included

1. **`SessionManager`** (`session_manager.py`) — SQLite-backed per-tenant session storage
   - Thread-local connections with WAL mode
   - `add_message()`, `get_recent_messages()`, `clear_session()`, `get_session_count()`
   - Configurable `max_history` (default 8 pairs, i.e. 16 messages)
   - Fail-closed: rejects `None` user_id, empty session_id, unknown roles
   - Schema versioned for future migration

2. **Query Rewriting** (`rag_engine.py`) — `_rewrite_query_with_context()`
   - Async LLM call to rewrite follow-up questions as standalone queries
   - Uses last 4 messages (2 pairs) only, truncated to 200 chars each
   - Max tokens: 100 for rewrite response
   - Graceful fallback: returns original question on LLM error or short response

3. **API Changes** (`main.py`)
   - `POST /query`: loads conversation history when `session_id` provided, saves user+assistant messages after query
   - `POST /sessions/clear`: clears a session's message history
   - `GET /sessions/{session_id}`: retrieves session history for frontend restore

4. **Schema Changes** (`schemas.py`)
   - `QueryRequest.session_id: str | None` — session identifier
   - `QueryResponse.session_id: str | None` — echoed back
   - `QueryResponse.rewritten_question: str | None` — standalone question after rewrite

### Explicitly Out of Scope
- GraphRAG — no evaluation evidence
- Autonomous Agent — no external actions
- Writing conversation history into the vector store
- Long-term user preference storage — requires clear product requirement

## Modified Files

| File | Change |
|------|--------|
| `backend/src/services/session_manager.py` | NEW — 156 lines |
| `backend/src/services/rag_engine.py` | MODIFIED — `_rewrite_query_with_context()` (54 lines), `query()` early-return fix, `rewritten_question` in return dict |
| `backend/src/main.py` | MODIFIED — import `SessionManager`, `/query` session handling, 2 new session endpoints |
| `backend/src/models/schemas.py` | MODIFIED — `session_id` + `rewritten_question` fields |
| `backend/tests/test_phase4.py` | NEW — 267 lines, 21 tests |

## Test Results

- **21 new tests** (7 classes): SessionManager CRUD, tenant isolation, query rewriting, rewritten_question in response, AST invariants
- **129 total tests** — all pass
- AST parse: clean (all source files)
- git diff --check: clean (CRLF warning only, expected on Windows)

## Design Decisions

1. **Thread-local connections.** `SessionManager` uses `threading.local()` to avoid SQLite connection conflicts in multi-threaded FastAPI workers.

2. **Rewrite prompt is minimal.** Only 200 chars per message, max 100 output tokens. Designed for a 2B model; the rewrite is a lightweight `f"Given this conversation:..."` prompt, not an elaborate system message.

3. **Rewrite happens BEFORE retrieval.** The standalone question is used for vector/BM25 search, while the original question is saved to session history. This keeps retrieval clean of conversational cruft.

4. **rewritten_question is null when no history.** Clear signal to frontend: if `rewritten_question` is not None, the user sees their original question but the system searched with the rewritten version.

5. **Conversation history lives in SQLite, NOT in ChromaDB.** Irreversible: no conversation data can leak into the vector store via the `query()` path. Enforced by AST-level tests.

## Known Risks

- **No session expiration.** Sessions grow unbounded in SQLite. A cleanup cron job or TTL-based pruning should be added for production.
- **Rewrite prompt may not work well for complex multi-hop follow-ups.** The 2B model may produce degraded rewrites. A/B comparison or model upgrade recommended before production use.
- **Stream endpoint not yet session-aware.** `/query/stream` does not save messages to session or perform query rewriting.
- **No session-level `rewritten_question` returned in stream.** Frontend cannot show rewritten question in streaming mode.` 
- **`session_manager` DB path is hardcoded** (`sessions.db`). Should be configurable via env var.
