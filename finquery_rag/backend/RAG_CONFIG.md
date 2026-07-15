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

Run the same non-secret checks from a shell before deployment or after moving
runtime data paths:

```bash
python -m src.eval_cli doctor   --bm25-db "$BM25_DB_PATH"   --out /tmp/finquery_doctor.json
```

`doctor` returns `0` when required stores are ready, `1` when readiness is
degraded, and still writes a JSON snapshot suitable for support/debugging. Use
`--warn-only` for non-blocking diagnostics in ad-hoc maintenance scripts.

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
FEEDBACK_DB_PATH=/var/lib/finquery/feedback.db
```

When unset, FinQuery keeps the existing development defaults under the backend
working directory.

## Deployment preflight

Before a server cutover, run the aggregate preflight command. It combines
readiness checks, migration audit, fixture audit, eval gate, baseline comparison,
and retrieval diagnostics into one JSON report without calling LLMs or embedding
models:

```bash
python -m src.eval_cli preflight   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --baseline eval/baseline_smoke_report.json   --bm25-db "$BM25_DB_PATH"   --registry-db "$DOCUMENT_REGISTRY_DB_PATH"   --chroma-path "$CHROMA_PATH"   --trace-db "$TRACE_DB_PATH"   --feedback-db "$FEEDBACK_DB_PATH"   --out /tmp/finquery_preflight.json
```

Use `--warn-only` for exploratory runs on partially initialized environments.
A normal non-zero exit means at least one section failed and should be reviewed
before serving production traffic.

## Migration readiness audit

Before deploying code that relies on tenant-scoped chunk IDs against existing
local data, run a non-content migration audit. It inspects identifiers, table
shape, and aggregate counts only; it does not read chunk text or document
content:

```bash
python -m src.eval_cli migration-audit   --bm25-db "$BM25_DB_PATH"   --registry-db "$DOCUMENT_REGISTRY_DB_PATH"   --chroma-path "$CHROMA_PATH"   --out /tmp/finquery_migration_audit.json
```

The command returns `1` when high-risk legacy patterns are found, such as
unscoped BM25 `doc_id` values, missing `user_id`, mismatched user prefixes, or
unscoped Chroma embedding IDs. Use `--warn-only` for exploratory audits that
should not fail a maintenance script. High-risk findings usually mean the BM25
and/or Chroma indexes should be rebuilt from tenant-scoped chunks before serving
production queries.

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

Authenticated users can inspect only their own traces through the API:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/traces
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/traces/<trace_id>
```

Trace API responses omit `tenant_id` and decode stored JSON columns such as
sources, candidates, and filter conditions.

## Answer feedback

Authenticated users can submit feedback for their own traced answers. The API
validates that the `trace_id` belongs to the current user before storing
feedback:

```bash
curl -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"trace_id":"<trace_id>","rating":"down","comment":"missing citation"}' \
  http://127.0.0.1:8000/feedback

curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/feedback
```

Feedback responses omit `tenant_id`. Ratings are constrained to `up` or `down`;
comments are optional and capped at 2000 characters.

Export down-rated traced answers into replay cases for triage and regression
testing:

```bash
python -m src.eval_cli feedback-to-replay \
  --feedback-db "$FEEDBACK_DB_PATH" \
  --trace-db "$TRACE_DB_PATH" \
  --tenant-id 1 \
  --rating down \
  --out eval/feedback_replay.jsonl
```

The output is regular evaluation-case JSONL. Feedback fields are preserved in
case metadata; missing trace rows are skipped.

## Trace retention

Trace logs are stored in SQLite and can grow over time. Use explicit retention
for production or long-running demos:

```bash
TRACE_TTL_SECONDS=2592000
python -m src.eval_cli traces-cleanup --db "$TRACE_DB_PATH" --ttl-seconds "$TRACE_TTL_SECONDS"
```

Pass `--tenant-id` to clean one tenant only. Without `--tenant-id`, cleanup is
global and should be treated as an operator/admin action.

## Document lifecycle registry

The upload path records document lifecycle state in `document_registry`.
Use the authenticated endpoint below to inspect current user's registry rows,
including failed or in-progress documents that may not appear in vector-store
document listings:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/document-registry
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8000/document-registry?status=failed"
```

Responses intentionally omit file and content hashes.
