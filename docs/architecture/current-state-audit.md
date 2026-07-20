# Current State Audit

This document catalogs the implementation status of every subsystem in the nano_finance repository as of the baseline audit (2026-07-20). Each component is classified into one of six status categories.

## Status Categories

| Status | Meaning |
|--------|---------|
| FULL | Fully implemented and in active use |
| NOT_WIRED | Implemented but not wired into the main pipeline |
| STUB | Interface/placeholder only, no real implementation |
| DOC_CLAIM | Claimed in documentation but not verifiable in code |
| CONTAM_RISK | Evaluation contamination risk identified |
| NEEDS_DATA | Requires manual experimental data |

---

## 1. Tokenizer

Entry points: scripts/tok_train.py, scripts/tok_eval.py, nanochat/tokenizer.py

| Component | Status | Detail |
|-----------|--------|--------|
| BPE Tokenizer training | FULL | Custom 65K vocab trained on 1.5GB financial corpus |
| Tokenizer evaluation | FULL | tok_eval.py computes compression rates vs GPT-2, cl100k |
| Special tokens | FULL | Control tokens, think plain-text strategy |
| Tokenizer benchmark artifacts | NEEDS_DATA | scripts/benchmark_tokenizers.py does not exist; 59.5% reduction claim needs CSV/JSON evidence |

---

## 2. Base Pretraining

Entry points: scripts/base_train.py, nanochat/gpt.py, nanochat/dataloader.py, nanochat/dataset.py, nanochat/optim.py

| Component | Status | Detail |
|-----------|--------|--------|
| GPT model architecture | FULL | Depth-driven config; all hyperparameters auto-derived |
| Training loop | FULL | Gradient accumulation, checkpoint resume, wandb logging |
| Precision management | FULL | COMPUTE_DTYPE via nanochat/common.py with env var override |
| Data loading | FULL | Distributed tokenizing dataloader |
| Training metadata | FULL | nanochat/training_metadata.py tracks run params |
| Corpus documentation | NEEDS_DATA | 17.68B token claim needs verification |
| Checkpoint lineage | NEEDS_DATA | Step 7060, 28000; need SHA256 hashes and config hashes |
| Training metrics export | STUB | scripts/export_model_manifest.py does not exist |

Key config: --depth drives model size; trained on 8XH100 node; ~2.76 hrs to GPT-2 capability.

## 3. SFT / CoT

Entry points: scripts/chat_sft.py, nanochat/chat_format.py, tasks/

| Component | Status | Detail |
|-----------|--------|--------|
| SFT training pipeline | FULL | Multi-task SFT with checkpoint resume |
| Chat format encoding | FULL | System prompt merged into user message (no separate system role) |
| Data mixture (v1) | FULL | 39,801 samples: 91.4% finance, 3.1% R1 CoT, 5.5% general |
| Data mixture (v2) | NOT_WIRED | SFT_V2_OPTIMIZATION.md describes 1M token rebalancing; build_sft_v2.py not found in repo |
| CoT filtering | FULL | Full CoT sample filtering in SFT data |
| Assistant-only loss masking | FULL | Training only on assistant tokens |
| SFT checkpoints | NEEDS_DATA | SFT800 (best val_bpb 0.4783), SFT1147 (0.4842, better finance scores) |
| Task definitions | FULL | FinQA, TAT-QA, FinER, FinRED, FinSen, FiQA, ECTSum, Finance R1 |

---

## 4. Model Evaluation

Entry points: scripts/finance_eval.py, scripts/finance_generate.py, nanochat/finance_eval.py, nanochat/dialogue_eval.py, nanochat/core_eval.py, scripts/chat_eval.py

| Component | Status | Detail |
|-----------|--------|--------|
| Finance task metrics | FULL | Numeric QA, table QA, NER, RE, sentiment, summarization, instruction following |
| Dialogue regression | FULL | Quality checks: CJK, word count, refusals, suspicious patterns |
| CORE metric (DCLM) | FULL | nanochat/core_eval.py |
| Bits-per-byte eval | FULL | nanochat/loss_eval.py |
| Finance eval CLI | FULL | scripts/finance_eval.py - ID-keyed scoring with numeric tolerance |
| Model generation for eval | FULL | scripts/finance_generate.py |
| Model manifest | STUB | scripts/export_model_manifest.py does not exist |
| Sealed test set | STUB | No sealed test split; current eval uses validation data only |

Key metrics: Finance macro primary 0.4432 (SFT1147), Validation BPB 0.4783 (SFT800), CORE 0.2339 (Base 28000).

---

## 5. Inference Service

Entry points: scripts/chat_openai_compat.py, nanochat/engine.py, scripts/chat_web.py, scripts/chat_cli.py

| Component | Status | Detail |
|-----------|--------|--------|
| OpenAI-compatible API | FULL | Full /v1/chat/completions with streaming, multi-GPU worker pool |
| Abuse prevention | FULL | Message limits, length limits, temperature/top_k bounds |
| GPU worker pool | FULL | Async request distribution across GPUs |
| Multi-byte UTF-8 streaming | FULL | Safe Chinese/emoji output |
| Web chat UI | FULL | scripts/chat_web.py + nanochat/ui.html |
| CLI chat | FULL | scripts/chat_cli.py |
| Model discovery | FULL | /v1/models endpoint |
| Serving benchmarks | STUB | scripts/benchmark_serving.py does not exist |
| Deployment config | STUB | No docker-compose.yml or unified service config |

Key config: Port 8998 (default), FRP maps to cloud 8500; model loaded from SFT checkpoint.

---

## 6. PDF / Table Processing

Entry points: finquery_rag/backend/src/services/ingest.py, finquery_rag/backend/src/services/process_tables.py

| Component | Status | Detail |
|-----------|--------|--------|
| PDF text extraction | FULL | PyMuPDF with font hierarchy analysis, markdown reconstruction |
| Table extraction | FULL | Camelot (stream + lattice fallback) + NVIDIA API enhancement |
| Hierarchical chunking | FULL | Markdown header splitting + recursive sub-splitting |
| Front matter extraction | FULL | Title extraction from page 1 with font-size signals |
| Parent-child indexing | FULL | Child chunks with parent excerpts, section paths |
| Table context enhancement | FULL | NVIDIA API optional; falls back gracefully |
| Chunk sizing | FULL | chunk_size=350, overlap=50 for 2048 context window |

---

## 7. Vector Retrieval (ChromaDB)

Entry points: finquery_rag/backend/src/services/vector_store.py

| Component | Status | Detail |
|-----------|--------|--------|
| Collection management | FULL | Per-user collections, add/query/delete/list |
| Embedding model | FULL | Configurable via EMBEDDING_MODEL_NAME (default: all-MiniLM-L6-v2) |
| Metadata filtering | FULL | User-scoped queries with ChromaDB where clauses |
| Front matter retrieval | FULL | get_front_matter_chunks() for title/abstract lookups |
| Page-level retrieval | FULL | get_page_chunks() for page-specific fallback |

---

## 8. BM25 (Sparse Retrieval)

Entry points: finquery_rag/backend/src/services/retrieval.py

| Component | Status | Detail |
|-----------|--------|--------|
| SQLite FTS5 retriever | FULL | WAL mode, schema v2, jieba_fast Chinese tokenization |
| Content indexing | FULL | Chunk store with FTS5 external content table |
| User-scoped search | FULL | Queries filtered by user_id |
| Document name scoping | FULL | doc_name column with migration |
| Schema migration | FULL | Automatic FTS5 rebuild on schema version change |

---

## 9. RRF (Reciprocal Rank Fusion)

Entry points: finquery_rag/backend/src/services/retrieval.py (rrf function), rag_engine.py

| Component | Status | Detail |
|-----------|--------|--------|
| RRF algorithm | FULL | Dense + sparse fusion with configurable k parameter |
| Sufficiency thresholds | FULL | RRF and dense sufficiency thresholds, numeric floors |

---

## 10. Reranker

Entry points: finquery_rag/backend/src/services/reranker.py

| Component | Status | Detail |
|-----------|--------|--------|
| Reranker protocol | FULL | Reranker Protocol class |
| Noop reranker | FULL | Default pass-through |
| Heuristic reranker | FULL | Dependency-free lexical reranker (0.7 original + 0.3 lexical weight) |
| Cross-encoder reranker | NOT_WIRED | Configurable but requires explicit model path; not default |
| Reranker factory | FULL | build_reranker() in reranker.py |

## 11. RAG Orchestrator

Entry points: finquery_rag/backend/src/services/rag_engine.py (~2019 lines)

| Component | Status | Detail |
|-----------|--------|--------|
| Query intent routing | FULL | classify_query_intent -> retrieval vs conversation vs unsupported |
| Hybrid retrieval | FULL | Dense + BM25 with RRF fusion |
| Context assembly | FULL | Token-budgeted context with hierarchical parent expansion |
| LLM generation | FULL | Async OpenAI client call with streaming support |
| Citation building | FULL | Source tracking with chunk_id, parent_id, section_path |
| Multi-document coverage | FULL | _ensure_multi_doc_coverage() guarantees per-doc representation |
| Query expansion | FULL | _expand_retrieval_query() adds finance-specific retrieval terms |
| Token budgeting | FULL | 1100 max context tokens, 512 max new tokens for 2048 window |
| Streaming support | FULL | SSE streaming with error events |
| Page fallback coverage | CONTAM_RISK | _ensure_page_fallback_coverage() uses supporting_source_page metadata flag |
| Supporting page injection | CONTAM_RISK | _force_supporting_page_coverage() calls _supporting_pages_for_query() |
| Hardcoded page rules | CONTAM_RISK | _fallback_pages_for_query() hardcodes filename->page mappings for "final annual report", "wipo", "leac" |
| Supporting pages for query | CONTAM_RISK | _supporting_pages_for_query() maps specific doc names + query keywords to specific pages |
| Augment page fallbacks | CONTAM_RISK | _augment_with_page_fallbacks() injects known pages with score floors |

CRITICAL FINDING: Lines 317-413 of rag_engine.py contain hardcoded document-name-to-page-number mappings that effectively encode evaluation ground truth into the production retrieval path. Specific filenames ("final annual report", "wipo", "leac") and query terms ("record revenue", "cash and cash equivalents", "gross margin", "pct", "madrid", "black swan", "sunfill", etc.) are mapped to specific page numbers. These mappings must be removed in Phase 1.

## 12. Financial Tools

Entry points: finquery_rag/backend/src/services/financial_tools.py

| Component | Status | Detail |
|-----------|--------|--------|
| Number parsing | FULL | parse_financial_number() - Decimal-based, handles commas, parens, scales |
| growth_rate | FULL | (current - previous) / previous with zero-division protection |
| percentage_share | FULL | part / total |
| sum_values | FULL | Sum with Decimal precision |
| verify_sum | FULL | Component sum vs reported total within tolerance |
| convert_scale | FULL | million/billion/wan/yi conversions |
| format_ratio_percent | FULL | Ratio to percent formatting |
| Unit scales | FULL | English (k, m, bn) + Chinese (wan, yi) |

STATUS: NOT_WIRED - These tools are fully implemented and tested, but the RAG orchestrator (rag_engine.py) does not import or call them. The financial calculation pipeline (Phase 3) must wire them into the query flow.

---

## 13. Answer Validation

Entry points: finquery_rag/backend/src/services/answer_validation.py

| Component | Status | Detail |
|-----------|--------|--------|
| Percent value extraction | FULL | Regex-based extraction from generated answers |
| Calculation consistency | FULL | validate_answer_calculations() compares percent claims vs deterministic calc |
| Missing calculation detection | FULL | Identifies expected calc values missing from answer |
| Unsupported value detection | FULL | Identifies answer percentages not backed by calculations |
| Tolerance configuration | FULL | Configurable percentage-point tolerance (default 0.05) |

STATUS: NOT_WIRED - The validator is implemented and tested but not called by the main RAG query flow.

---

## 14. Trace

Entry points: finquery_rag/backend/src/services/trace.py

| Component | Status | Detail |
|-----------|--------|--------|
| SQLite trace store | FULL | Schema v2, WAL mode, per-tenant indexing |
| Query lifecycle logging | FULL | Original/rewritten query, intent, candidates, context, answer, sources |
| Content redaction | FULL | redact_content=True by default |
| Sampling | FULL | Configurable sample_rate |
| Trace replay export | FULL | trace_to_replay_case() for eval |
| Diagnostics JSON | FULL | Per-trace diagnostics payload |
| Trace API endpoints | FULL | GET /traces, GET /traces/{id}, GET /replay/traces |
| Schema migration | FULL | ensure_column, run_component_migrations |
| Staged timing | STUB | No per-stage latency breakdown |

---

## 15. Real Evaluation & CI

Entry points: finquery_rag/backend/src/services/evaluation.py, eval_runner.py, eval_cli.py, .github/workflows/

| Component | Status | Detail |
|-----------|--------|--------|
| Offline eval scoring | FULL | 14+ metrics: answer, citation, retrieval, calculation, no-answer |
| Eval runner (in-process) | FULL | run_jsonl_cases() |
| Eval runner (HTTP) | FULL | run_jsonl_cases_http() with auth preflight |
| Eval CLI | FULL | 10 subcommands |
| Smoke fixtures | FULL | 12 deterministic cases with predictions and baseline report |
| CI workflow (backend) | FULL | Tests + eval gate + preflight on PR/push |
| CI workflow (frontend) | FULL | Lint + build |
| JUnit output | FULL | Gate command produces JUnit XML for CI annotations |
| Fixture audit | FULL | Tag coverage, intent coverage, quality checks |
| Retrieval diagnostics | FULL | Recall@K, MRR, missed sources, worst cases |
| Oracle context path | STUB | src/evaluation/oracle_context.py does not exist |
| Leakage scan script | STUB | scripts/check_eval_leakage.py does not exist |
| Sealed test set | STUB | No document-isolated test split |
| Real evaluation dataset | STUB | Only smoke fixtures; no 80-case real eval set |
| Ablation framework | STUB | No retrieval or system ablation tooling |

CRITICAL FINDING: The eval infrastructure is comprehensive but relies entirely on synthetic smoke fixtures. The eval README explicitly states smoke fixtures are "not intended to represent product quality." No real financial document evaluation dataset exists.

---

## Summary

| # | Subsystem | Status |
|---|-----------|--------|
| 1 | Tokenizer | FULL (needs benchmark artifacts) |
| 2 | Base Pretraining | FULL (needs training docs) |
| 3 | SFT/CoT | FULL (needs data docs) |
| 4 | Model Evaluation | FULL (needs sealed test + manifest) |
| 5 | Inference Service | FULL (needs benchmarks + deploy) |
| 6 | PDF/Table Processing | FULL |
| 7 | Vector Retrieval | FULL |
| 8 | BM25 | FULL |
| 9 | RRF | FULL |
| 10 | Reranker | FULL |
| 11 | RAG Orchestrator | CONTAM_RISK |
| 12 | Financial Tools | NOT_WIRED |
| 13 | Answer Validation | NOT_WIRED |
| 14 | Trace | FULL (needs staged timing) |
| 15 | Real Eval & CI | FULL (needs real dataset + leakage scan) |
