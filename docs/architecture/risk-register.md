# Risk Register

Catalog of identified risks from the baseline audit. Each risk includes severity, affected files, and the phase where it will be addressed.

## Risk Severity Levels

| Level | Definition |
|-------|-----------|
| CRITICAL | Blocks production deployment; must fix before any metric claims |
| HIGH | Significantly impacts reliability or metric trustworthiness |
| MEDIUM | Code quality or maintainability concern |
| LOW | Nice-to-have improvement |

---

## CRITICAL Risks

### R-001: Production Retrieval Contains Hardcoded Eval Page Mappings

Severity: CRITICAL
Affected: rag_engine.py lines 317-413
Detail: The methods _fallback_pages_for_query() and _supporting_pages_for_query() contain hardcoded mappings from specific document filenames ("final annual report", "wipo", "leac") and query keywords to specific page numbers. These effectively encode evaluation ground truth into the production retrieval path.
Impact: All RAG retrieval metrics are contaminated; cannot distinguish real retrieval quality from hardcoded rules.
Phase: 1
Verification: test_no_query_specific_page_rules.py, test_filename_invariance.py

### R-002: supporting_source_page Metadata Flag Used as Ranking Signal

Severity: CRITICAL
Affected: rag_engine.py lines 145-197, 226-264, 266-315, 387-414
Detail: The supporting_source_page metadata flag is used to prioritize chunks in ranking and to set higher score floors. This flag is set based on _supporting_pages_for_query() which uses eval-specific knowledge.
Impact: Retrieval ranking is artificially boosted for chunks that match expected eval answers.
Phase: 1

### R-003: No Sealed Test Set Exists

Severity: CRITICAL
Affected: Entire evaluation framework
Detail: All current evaluation uses smoke fixtures (synthetic, 12 cases) which are explicitly documented as "not intended to represent product quality." No real financial document evaluation dataset exists with document-level train/test isolation.
Impact: Cannot make any credible claims about RAG quality.
Phase: 5

## HIGH Risks

### R-004: Financial Tools Not Wired Into Main Pipeline

Severity: HIGH
Affected: financial_tools.py, rag_engine.py
Detail: financial_tools.py implements 7 deterministic calculation operations with full Decimal precision. However, rag_engine.py never imports or calls these tools. All financial calculations currently rely on the LLM to perform arithmetic.
Impact: Financial calculations are not deterministic; LLM arithmetic errors are not caught.
Phase: 3

### R-005: Answer Validation Not Wired Into Main Pipeline

Severity: HIGH
Affected: answer_validation.py, rag_engine.py
Detail: answer_validation.py can detect missing and unsupported percentage claims in generated answers, but is never called by the production query flow.
Impact: Generated answers with incorrect financial numbers are returned without validation.
Phase: 4

### R-006: rag_engine.py is ~2019 Lines - Single Point of Failure

Severity: HIGH
Affected: rag_engine.py
Detail: The RAG orchestrator is a monolithic ~2019-line file containing retrieval, fusion, reranking, context building, prompt construction, answer generation, citation building, source tracking, and page fallback logic all in one class.
Impact: Difficult to test in isolation; changes to any subsystem risk breaking others; contamination fixes are spread across many methods.
Phase: 2

### R-007: Model Metrics Not Verifiable From Artifacts

Severity: HIGH
Affected: Multiple
Detail: Key model claims (parameter count, training tokens, tokenizer compression rate) exist in documentation but cannot be independently verified from committed artifacts. No model manifest export script exists.
Impact: Resume/interview claims lack evidentiary support.
Phase: 6

### R-008: No Oracle Context Path for Upper-Bound Measurement

Severity: HIGH
Affected: evaluation.py, eval_runner.py
Detail: There is no way to measure "what would the model generate if given perfect evidence?" This makes it impossible to distinguish retrieval failures from generation failures.
Impact: Cannot properly diagnose whether poor answers are caused by retrieval misses or model limitations.
Phase: 1 (oracle context), Phase 5 (ablation)

---

## MEDIUM Risks

### R-009: Intent Classification is Keyword-Only

Severity: MEDIUM
Affected: intent.py
Detail: classify_query_intent() uses only keyword matching. Edge cases: "How much did revenue grow?" contains "how much" (document_lookup pattern) AND "grow" (calculation keyword). The current priority order may misclassify ambiguous queries.
Impact: Some financial calculation queries may be routed to document_qa instead.
Phase: 3

### R-010: No Staged Latency Tracking in Trace

Severity: MEDIUM
Affected: trace.py
Detail: Trace records total latency but does not break down by stage (query rewrite, dense retrieval, BM25, fusion, rerank, evidence extraction, calculation, generation, validation).
Impact: Cannot identify performance bottlenecks without manual instrumentation.
Phase: 7

### R-011: No Serving Benchmarks

Severity: MEDIUM
Affected: scripts/ directory (missing benchmark_serving.py)
Detail: No TTFT, tokens/s, p50/p95 latency, or peak VRAM measurements exist for the inference service.
Impact: Cannot make performance claims or capacity plan.
Phase: 7

### R-012: SFT V2 Data Mixture Script Missing

Severity: MEDIUM
Affected: SFT_V2_OPTIMIZATION.md
Detail: SFT_V2_OPTIMIZATION.md references build_sft_v2.py for a 1M assistant-token finance set, but this script is not found in the repository. The v2 mixture strategy exists only as documentation.
Impact: SFT v2 training is not reproducible from committed code.
Phase: 6

### R-013: No Docker or Unified Deployment Config

Severity: MEDIUM
Affected: None (missing)
Detail: No docker-compose.yml, unified service config, or single-command startup exists. The inference server, FinQuery backend, and frontend must be started separately.
Impact: Difficult to demo or deploy consistently.
Phase: 7

---

## LOW Risks

### R-014: NVIDIA API Dependency in Table Processing

Severity: LOW
Affected: process_tables.py
Detail: Table enhancement uses an optional NVIDIA API call. When the API is unavailable, the system falls back gracefully to raw table markdown.
Impact: Table summaries may be lower quality without the API key, but ingestion does not fail.
Phase: No action required (graceful fallback exists)

### R-015: No GPU Monitoring Script in CI

Severity: LOW
Affected: scripts/gpu_monitor.py
Detail: gpu_monitor.py exists but is not integrated into CI or health checks.
Impact: GPU health issues may go undetected.
Phase: 7

---

## Risk Remediation Schedule

| Phase | Risks Addressed |
|-------|----------------|
| Phase 1 | R-001, R-002, R-008 (oracle path) |
| Phase 2 | R-006 |
| Phase 3 | R-004, R-009 |
| Phase 4 | R-005 |
| Phase 5 | R-003, R-008 (ablation) |
| Phase 6 | R-007, R-012 |
| Phase 7 | R-010, R-011, R-013, R-015 |
| Phase 8 | None (documentation/showcase) |
