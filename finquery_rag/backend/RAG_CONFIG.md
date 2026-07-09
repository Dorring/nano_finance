# FinQuery RAG configuration

Runtime configuration is read from environment variables.

## Reranking

Reranking is disabled by default.

```bash
RAG_RERANKER=heuristic
RAG_CANDIDATE_MULTIPLIER=2
```

Available rerankers:

- unset / `none` / `off` — disabled, preserves existing retrieval behavior.
- `heuristic` — dependency-free lexical fallback for local experiments.
- `cross-encoder` — optional model-backed reranker.

Cross-encoder reranking must also set a model name or local path:

```bash
RAG_RERANKER=cross-encoder
RAG_RERANKER_MODEL=/path/to/local/cross-encoder-model
RAG_CANDIDATE_MULTIPLIER=4
```

Use a local model path for production-like runs. If `RAG_RERANKER_MODEL` is
missing, initialization fails explicitly instead of silently downloading a model
or changing retrieval behavior.

For any reranker change, run an eval report before merging:

```bash
python -m src.eval_cli run --cases <cases.jsonl> --out <candidate.jsonl> --user-id <id>
python -m src.eval_cli score --cases <cases.jsonl> --predictions <candidate.jsonl> --out <candidate_report.json>
python -m src.eval_cli compare --baseline <baseline_report.json> --candidate <candidate_report.json>
```

## Health and readiness probes

The backend exposes two unauthenticated operational probes:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

- `/healthz` is a lightweight liveness check for process availability.
- `/readyz` returns a non-secret RAG dependency snapshot and uses HTTP 503
  when required local stores are unavailable or runtime configuration is
  invalid.

`/readyz` is intentionally cheap: it does not call the LLM, does not request
embeddings, and does not read tenant document content.

## Intent routing

FinQuery includes a deterministic intent router before retrieval. It is
conservative by design:

- clear greetings, thanks, and product-help questions bypass retrieval;
- clear out-of-scope general questions bypass retrieval with a refusal;
- financial QA, document summaries, and financial calculations continue through
  the RAG pipeline;
- unknown domain-specific wording defaults to retrieval to avoid suppressing
  valid document questions.

The router exposes `intent` and `intent_confidence` in `/query` responses and
the final `/query/stream` event. It does not call an LLM and has no external
dependencies.

## Runtime storage paths

Use explicit paths in production so the app, health checks, eval tooling, and
background scripts read and write the same local stores:

```bash
CHROMA_PATH=/var/lib/finquery/chroma_db
DOCUMENT_REGISTRY_DB_PATH=/var/lib/finquery/document_registry.db
BM25_DB_PATH=/var/lib/finquery/rag_bm25.db
SESSIONS_DB_PATH=/var/lib/finquery/sessions.db
TRACE_DB_PATH=/var/lib/finquery/trace_log.db
```

When unset, FinQuery keeps the existing development defaults under the backend
working directory.

## BM25 maintenance

The sparse index stores canonical chunk rows in `chunk_store` and query rows in
the FTS5 table `fts_index`. Use the maintenance commands after manual database
changes, interrupted writes, or before production cutovers:

```bash
python -m src.eval_cli bm25-check --db "$BM25_DB_PATH"
python -m src.eval_cli bm25-rebuild --db "$BM25_DB_PATH"
```

Both commands accept `--user-id` for tenant-scoped checks/rebuilds. A global
rebuild also removes orphan FTS rows that can no longer be attributed to a
tenant.

## Trace IDs

Non-streaming `/query` responses and `/query/stream` final `done` events include
`trace_id` when structured tracing is successfully written. Use this ID with
trace export/query tooling to locate the exact request path during debugging or
replay preparation. If trace persistence fails, the answer path still succeeds
and `trace_id` is `null`.

## Trace retention

Trace logs are stored in SQLite and can grow over time. Use explicit retention
for production or long-running demos:

```bash
TRACE_TTL_SECONDS=2592000
python -m src.eval_cli traces-cleanup --db "$TRACE_DB_PATH" --ttl-seconds "$TRACE_TTL_SECONDS"
```

Pass `--tenant-id` to clean one tenant only. Without `--tenant-id`, cleanup is
global and should be treated as an operator/admin action.
