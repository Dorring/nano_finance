# Phase 2: RAG Orchestration Modularization

## Overview

Phase 2 decomposed the monolithic `RAGEngine` (1000+ lines pre-refactor) into
a layered architecture with typed domain boundaries. The refactor preserves
the exact legacy dict-shaped API response while introducing `QueryRequest`
and `AnswerResult` as the production call boundary.

- **Baseline commit**: `3d08498` (Phase 1 merge)
- **Phase 2 branch**: `refactor/nf-02-rag-orchestration`
- **Starting commit**: `83d0cc5` (initial modularization)
- **Closure commits**: typed-boundary integration, test isolation, cycle fixes

## Architecture (Before → After)

### Before: Monolithic RAGEngine

```
src/services/rag_engine.py (~1000 lines)
  ├── query rewriting
  ├── intent classification
  ├── retrieval (single/multi/hybrid)
  ├── context building & dedup
  ├── sufficiency evaluation
  ├── deterministic answering
  ├── LLM generation
  ├── trace logging
  └── API response dict construction
```

### After: Layered Modules

```
src/
├── domain/                      # Typed boundary (no heavy deps)
│   ├── query.py                 # QueryRequest
│   ├── evidence.py              # EvidenceItem (metadata-aware)
│   └── answer.py                # AnswerResult, AnswerPath, RetrievalResult
├── retrieval/                   # Retrieval pipeline (no services dep)
│   ├── query_processor.py       # QueryProcessor (expand, rewrite, classify)
│   ├── retrieval_pipeline.py    # RetrievalPipeline (single/multi/hybrid)
│   ├── candidate_fusion.py      # rrf, normalize, dedupe, summarize
│   └── context_builder.py       # ContextBuilder, SufficiencyEvaluator
├── generation/                  # LLM gateway & answer rendering
│   ├── prompt_builder.py        # System prompt construction
│   ├── llm_gateway.py           # LLMGateway (generate, rewrite_query)
│   ├── deterministic_answers.py # Front-matter & context deterministic answers
│   └── response_renderer.py     # Answer validation
├── application/                 # Orchestration (depends on domain/retrieval/generation)
│   └── rag_orchestrator.py      # RAGOrchestrator.answer(request) -> AnswerResult
└── services/                    # Facade & legacy compat (depends on application)
    └── rag_engine.py            # RAGEngine (440 lines, Facade)
```

## Module Responsibilities

| Module | Responsibility | Depends On |
|--------|---------------|------------|
| `domain.query` | `QueryRequest` immutable input | (none) |
| `domain.evidence` | `EvidenceItem` with nested-metadata support | (none) |
| `domain.answer` | `AnswerResult`, `AnswerPath`, `RetrievalResult` | (none) |
| `retrieval.query_processor` | Query expansion, rewrite, classification | (none) |
| `retrieval.retrieval_pipeline` | Single/multi-doc retrieval orchestration | `candidate_fusion` |
| `retrieval.candidate_fusion` | RRF fusion, dedup, score normalization | (none) |
| `retrieval.context_builder` | Context assembly, sufficiency eval | (none) |
| `generation.prompt_builder` | System prompt templates | (none) |
| `generation.llm_gateway` | LLM generation & query rewriting | `query_processor` (optional) |
| `generation.deterministic_answers` | Front-matter & numeric extraction | (none) |
| `generation.response_renderer` | Answer validation | (none) |
| `application.rag_orchestrator` | Full pipeline coordination | `domain`, `retrieval`, `generation` |
| `services.rag_engine` | Facade: builds `QueryRequest`, delegates to orchestrator | `application`, `services.memory_profile` |

## Data Flow: QueryRequest → AnswerResult

```
HTTP /query
    │
    ▼
RAGEngine.query(question, doc_names, user_id, n_results, conversation_history, memory_profile)
    │
    │  1. Construct QueryRequest(question, document_names, user_id,
    │     conversation_history, memory_profile)
    │
    ▼
RAGOrchestrator.answer(request, *, n_results)
    │
    │  2. If conversation_history: gateway.rewrite_query(...)
    │  3. classify_intent(question)
    │  4. _handle_conversational_query(question)  → CONVERSATIONAL branch
    │  5. If not requires_retrieval              → NO_RETRIEVAL branch
    │  6. List documents; if empty               → NO_DOCUMENTS branch
    │  7. Retrieve chunks (front-matter → single/multi)
    │  8. Deterministic front-matter answer?
    │  9. Sufficiency eval → context build → deterministic context answer?
    │ 10. LLM generation (if sufficient)          → FULL branch
    │ 11. Trace logging
    │
    ▼
AnswerResult (immutable, 14 fields + path + had_conversation_history)
    │
    │  to_legacy_dict() reproduces exact pre-refactor dict shape per branch
    │
    ▼
HTTP response (dict)
```

## Facade Boundary

`RAGEngine` is now a thin Facade:
- Constructs `QueryRequest` from positional args
- Delegates to `RAGOrchestrator.answer()`
- Converts `AnswerResult` via `to_legacy_dict()`
- No per-query private dependency sync (dependencies injected once at `__init__`)

## Dependency Injection

All orchestrator dependencies are injected once at construction:

```python
RAGOrchestrator(
    query_processor=QueryProcessor(...),
    retrieval_pipeline=RetrievalPipeline(...),
    context_builder=ContextBuilder(...),
    sufficiency_evaluator=EvidenceSufficiencyEvaluator(...),
    llm_gateway=LLMGateway(...),
    deterministic_extractor=DeterministicAnswerExtractor(...),
    trace_logger=TraceLogger(...),
    intent_classifier=classify_query_intent,
    list_all_documents_fn=list_all_documents,
    get_front_matter_chunks_fn=get_front_matter_chunks,
)
```

`RAGEngine.__init__` constructs these and passes them to `RAGOrchestrator`.
No per-query reassignment of `_orchestrator._*` attributes.

## Legacy API Compatibility

`AnswerResult.to_legacy_dict()` reproduces the exact dict shape that the
pre-refactor `RAGOrchestrator.query` returned, including:

- Field set per branch (FULL / CONVERSATIONAL / NO_RETRIEVAL / NO_DOCUMENTS)
- `rewritten_question` inclusion rules:
  - FULL: always present (None if no conversation history)
  - Other branches: present iff `had_conversation_history=True`
- `None` preservation for nullable fields
- `list` vs `tuple` conversion (legacy API uses lists)
- `warnings` only present when non-empty

Verified by:
- `tests/test_phase4.py::TestQueryReturnsRewrittenQuestion`
- `tests/refactor/test_response_characterization.py`
- `tests/architecture/test_api_contract.py`
- `tests/test_phase12_trace_id.py`, `test_phase24`, `test_phase31`

## Phase 3 Insertion Points

Phase 3 (financial calculation pipeline) will extend the typed boundary:

- `CalculationPlan` — inserted between retrieval and generation
- `CalculationResult` — inserted after deterministic/LLM answering
- `Answerability` — replaces ad-hoc `context_sufficient` boolean
- `warnings` / `calculations` fields on `AnswerResult`

The `application.rag_orchestrator.RAGOrchestrator.answer` method is the
single integration point. Phase 3 should:
1. Add new domain objects in `src/domain/`
2. Extend `AnswerResult` with `calculations: tuple[CalculationResult, ...]`
3. Insert calculation steps between steps 8-10 in the data flow above
4. Not modify retrieval algorithm, thresholds, or prompts
