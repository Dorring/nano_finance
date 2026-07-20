# Upstream Attribution

This document delineates the boundaries between upstream nanochat, the original FinQuery project, and personal additions made in this repository.

## 1. Upstream: nanochat (Karpathy)

Source: https://github.com/karpathy/nanochat
License: MIT

### Files from upstream (with modifications noted)

Unmodified upstream files:
- nanochat/checkpoint_manager.py, common.py, core_eval.py, loss_eval.py
- nanochat/execution.py, flash_attention.py, fp8.py, report.py, ui.html
- scripts/chat_rl.py, chat_web.py, chat_cli.py, chat_eval.py
- runs/speedrun.sh, miniseries.sh, scaling_laws.sh, runcpu.sh
- tasks/arc.py, gsm8k.py, mmlu.py, humaneval.py, smoltalk.py, spellingbee.py, common.py, customjson.py

Modified upstream files:
- nanochat/gpt.py - Extended for training metadata
- nanochat/engine.py - Adapted for finance generation
- nanochat/tokenizer.py - Custom 65K vocab BPE
- nanochat/dataloader.py, dataset.py, optim.py - Training tuning
- nanochat/chat_format.py - Extended chat format
- scripts/base_train.py, base_eval.py, chat_sft.py - Custom config
- scripts/tok_train.py, tok_eval.py - Custom tokenizer

## 2. Upstream: FinQuery (datalordstephen)

Source: https://github.com/datalordstephen/finquery
Original purpose: RAG-based financial document QA system.

### Original FinQuery components retained (with modifications)

- Frontend React app structure - User-facing chat and document upload UI
- FastAPI backend skeleton - API framework and CORS configuration
- auth.py - JWT-based user authentication
- ingest.py - PyMuPDF + Camelot table extraction pipeline
- vector_store.py - ChromaDB embedding storage and retrieval
- retrieval.py - SQLite FTS5-based sparse retrieval
- rag_engine.py - Hybrid retrieval + RRF fusion (heavily modified, ~2019 lines)
- models/schemas.py - API request/response Pydantic models
- models/user.py - SQLAlchemy user ORM
- database.py - PostgreSQL connection setup
- process_tables.py - Camelot table extraction + NVIDIA API enhancement

## 3. Personal Additions

### 3.1 Financial Domain Training

- nanochat/finance_eval.py - Financial task metrics: numeric QA, table QA, NER, RE, sentiment, summarization
- nanochat/dialogue_eval.py - Dialogue quality regression metrics
- nanochat/training_metadata.py - Training run metadata and model manifest
- scripts/finance_eval.py - CLI for scoring finance evaluation predictions
- scripts/finance_generate.py - Generate model predictions for finance eval
- scripts/dialogue_generate.py / dialogue_compare.py - Dialogue regression tooling
- scripts/analyze_finance_failures.py - Failure analysis for finance tasks
- scripts/merge_sft_data.py - SFT data mixture and merging
- SFT_V2_OPTIMIZATION.md - SFT v2 data mixture strategy
- AI_APPLICATION_IMPLEMENTATION_PLAN.md - Project implementation roadmap

### 3.2 OpenAI-Compatible API Adapter

- scripts/chat_openai_compat.py - Full OpenAI /v1/chat/completions server with GPU worker pool, abuse prevention, multi-GPU support

### 3.3 FinQuery RAG Extensions

- src/services/intent.py - Deterministic query intent classification (7 intent types)
- src/services/financial_tools.py - Deterministic financial calculation helpers (Decimal-based, 7 operations)
- src/services/answer_validation.py - Post-generation answer consistency checks
- src/services/evaluation.py - Offline RAG eval scoring (CPU-safe, 14+ metrics)
- src/services/eval_runner.py - Eval execution harness (in-process + HTTP modes)
- src/services/eval_cli.py - Full CLI: score, run, run-http, compare, gate, diagnostics, interview-report, failure-analysis, audit-fixtures
- src/services/trace.py - Structured retrieval trace logging (SQLite, schema v2)
- src/services/session_manager.py - SQLite conversation session store (WAL mode, TTL support)
- src/services/document_registry.py - Document lifecycle state machine
- src/services/feedback.py - User feedback store (thumbs up/down)
- src/services/health.py - Health/diagnostics snapshot
- src/services/preflight.py - Deployment preflight checks
- src/services/migration_audit.py - Chroma/BM25 index integrity checks
- src/services/reranker.py - Heuristic + cross-encoder reranker factory
- src/services/query_scope.py - Tenant-scoped document filter resolution
- src/services/streaming.py - SSE streaming event helpers
- src/services/retrieval_config.py - Env-var based retrieval configuration
- src/services/sqlite_migrations.py - Schema migration utilities
- src/services/chunk_id.py - Chunk ID generation and scoping
- src/services/memory_profile.py - User memory profile for personalization

### 3.4 Test Infrastructure

- tests/test_finance_eval.py - Finance evaluation tests
- finquery_rag/backend/tests/ (75+ test files) - Phase-organized FinQuery tests

### 3.5 CI/CD

- .github/workflows/finquery-rag-backend.yml - Backend CI: tests + eval gate + preflight
- .github/workflows/finquery-rag-frontend.yml - Frontend CI: lint + build
- finquery_rag/backend/scripts/ci_eval_gate.py - CI eval gate script
- finquery_rag/backend/scripts/ci_preflight_smoke.py - CI preflight smoke test

### 3.6 Evaluation Fixtures

- finquery_rag/backend/eval/golden_smoke.jsonl - 12 smoke test cases
- finquery_rag/backend/eval/predictions_smoke.jsonl - Deterministic predictions
- finquery_rag/backend/eval/baseline_smoke_report.json - Checked scorer output
- finquery_rag/backend/eval/real_eval_template.jsonl - Real PDF eval template
- finquery_rag/backend/eval/real_eval_labeling_template.csv - Labeling template (15 cases)

### 3.7 Documentation

- CLAUDE.md - Claude Code guidance
- docs/architecture/* - Architecture documentation (this Phase 0)
- report.md - Project report
- finquery_rag/backend/RAG_CONFIG.md - RAG configuration reference
- finquery_rag/backend/FINANCIAL_TOOLS.md - Financial tools documentation
- finquery_rag/backend/eval/README.md - Evaluation fixture documentation
