# Implementation Map

This document maps each audit finding to specific files and lines that need changes in Phases 1-8.

## Phase 1: Retrieval Integrity (fix/nf-01-retrieval-integrity)

### Files to modify

**rag_engine.py** (CRITICAL - contamination removal):
- Lines 317-353: Remove _fallback_pages_for_query() - hardcoded doc->page mappings
- Lines 355-385: Remove _supporting_pages_for_query() - eval-specific page injection
- Lines 266-315: Remove _force_supporting_page_coverage() - forces eval pages into top-k
- Lines 387-414: Remove or refactor _augment_with_page_fallbacks() - removes score floor injection
- Lines 145-197: Refactor _ensure_page_fallback_coverage() - remove supporting_source_page priority
- Lines 162, 175, 189, 242, 256: Remove all supporting_source_page metadata flag references
- Lines 309, 409, 531-532, 569-576: Remove all supporting_source_page metadata writes
- Lines 591, 606, 612: Remove calls to _force_supporting_page_coverage()

**Files to audit (read-only verification):**
- retrieval.py: Verify no gold-label leakage in BM25 indexing
- vector_store.py: Verify no expected-page injection in vector queries
- reranker.py: Verify no query-specific reranker rules

### New files to create

- tests/integrity/test_no_golden_import_in_production.py
- tests/integrity/test_no_query_specific_page_rules.py
- tests/integrity/test_filename_invariance.py
- tests/integrity/test_oracle_not_imported_by_api.py
- finquery_rag/backend/src/evaluation/oracle_context.py (Oracle-only evidence provider)
- scripts/check_eval_leakage.py (CI source scanner)

---

## Phase 2: RAG Modularization (refactor/nf-02-rag-orchestration)

### Files to create

Target directory structure:
```
finquery_rag/backend/src/application/rag_orchestrator.py
finquery_rag/backend/src/application/response_pipeline.py
finquery_rag/backend/src/domain/query.py
finquery_rag/backend/src/domain/evidence.py
finquery_rag/backend/src/domain/calculation.py
finquery_rag/backend/src/domain/answer.py
finquery_rag/backend/src/retrieval/query_processor.py
finquery_rag/backend/src/retrieval/dense_retriever.py
finquery_rag/backend/src/retrieval/sparse_retriever.py
finquery_rag/backend/src/retrieval/fusion.py
finquery_rag/backend/src/retrieval/reranking.py
finquery_rag/backend/src/retrieval/context_builder.py
finquery_rag/backend/src/retrieval/sufficiency.py
finquery_rag/backend/src/generation/prompt_builder.py
finquery_rag/backend/src/generation/llm_client.py
finquery_rag/backend/src/generation/citation_builder.py
finquery_rag/backend/src/generation/answer_renderer.py
```

### Files to modify

- rag_engine.py: Reduce to ~400-600 line Facade, delegate to RAGOrchestrator
- main.py: Update imports if needed; maintain API compatibility

### Domain objects to define

- QueryRequest (frozen dataclass)
- EvidenceItem (frozen dataclass)
- CalculationOperand (frozen dataclass)
- CalculationPlan (frozen dataclass)
- CalculationResult (frozen dataclass)
- AnswerBundle (frozen dataclass)

---

## Phase 3: Financial Calculation Pipeline (feat/nf-03-financial-calculation-pipeline)

### Files to create

```
finquery_rag/backend/src/finance/metric_lexicon.py
finquery_rag/backend/src/finance/operation_router.py
finquery_rag/backend/src/finance/evidence_extractor.py
finquery_rag/backend/src/finance/unit_normalizer.py
finquery_rag/backend/src/finance/calculation_registry.py
finquery_rag/backend/src/finance/calculation_executor.py
finquery_rag/backend/src/finance/calculation_verifier.py
finquery_rag/backend/src/finance/calculation_answer_renderer.py
```

### Files to modify

- financial_tools.py: Extend with new operations (difference, average, gross_margin, net_margin, debt_ratio, roe, cagr)
- intent.py: Extend intent types (add table_calculation, multi_document_comparison)
- rag_engine.py / rag_orchestrator.py: Wire calculation pipeline into query flow

### Registry operations (v1)

difference, growth_rate, percentage_share, sum, average, gross_margin, net_margin, debt_ratio, roe, cagr

---

## Phase 4: Grounding & Validation (feat/nf-04-grounding-and-validation)

### Files to modify

- answer_validation.py: Extend with validate_numeric_claims(), validate_units(), validate_periods(), validate_citations(), validate_operand_provenance(), validate_unsupported_claims()
- rag_engine.py / rag_orchestrator.py: Add answerability assessment, safe fallback routing

### New answerability states

ANSWERABLE, PARTIALLY_ANSWERABLE, NOT_ANSWERABLE, CALCULATION_BLOCKED

---

## Phase 5: Real Evaluation (test/nf-05-real-evaluation-suite)

### Files to create

```
finquery_rag/backend/eval/real_eval_sealed_test.jsonl (~80 cases)
finquery_rag/backend/eval/real_eval_development.jsonl
finquery_rag/backend/src/evaluation/ablation.py
```

### Files to modify

- evaluation.py: Add calculation metrics, ablation support
- eval_cli.py: Add ablation subcommand
- eval_runner.py: Add ablation runner

---

## Phase 6: Training Evidence (docs/nf-06-training-reproducibility)

### Files to create

```
MODEL_CARD.md
DATA_CARD.md
docs/training/tokenizer-report.md
docs/training/pretraining-report.md
docs/training/sft-cot-report.md
docs/training/checkpoint-lineage.md
docs/training/distributed-training-failures.md
docs/evaluation/model-evaluation.md
scripts/benchmark_tokenizers.py
scripts/export_model_manifest.py
artifacts/tokenizer/tokenizer_benchmark.csv
artifacts/tokenizer/tokenizer_benchmark.json
```

### Human input required

- Exact training corpus composition (English/Chinese/finance ratios)
- Checkpoint SHA256 hashes
- Training GPU-hours and peak VRAM
- Distributed training failure logs
- Exact SFT data composition per source

---

## Phase 7: Serving & Observability (perf/nf-07-serving-observability)

### Files to create

```
config/model.yaml
config/retrieval.yaml
config/calculation.yaml
config/evaluation.yaml
scripts/benchmark_serving.py
scripts/benchmark_rag_latency.py
scripts/preflight.py
scripts/start_local.sh
docker-compose.yml
.env.example
```

### Files to modify

- trace.py: Add per-stage latency fields
- main.py: Add /health endpoint enhancements

---

## Phase 8: Showcase (docs/nf-08-resume-demo)

### Files to modify

- README.md: Complete rewrite for nano_finance project
- Frontend: Add calculation result display with collapsible sections

### Demo scenarios (7 required)

1. Single-document metric lookup
2. Table value query
3. YoY growth calculation
4. Multi-document financial comparison
5. Unit conversion
6. Evidence-insufficient refusal
7. Model-only vs RAG vs RAG+Calculator comparison
