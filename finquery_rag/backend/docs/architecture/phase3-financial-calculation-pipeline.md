# Phase 3: System-Orchestrated Financial Calculation Pipeline

## Overview

Phase 3 adds a deterministic financial calculation pipeline to the RAG
orchestrator. Instead of relying on model-native Function Calling or SFT
tool-use, the system routes calculation questions through a fixed
deterministic pipeline that executes Decimal-based financial formulas on
evidence-bound operands extracted from retrieved chunks.

Key design decisions:
- **No model retraining**: the calculation pipeline is purely deterministic.
- **No eval/exec**: all formulas are fixed Python functions using `Decimal`.
- **No model-generated code execution**: the model never produces code.
- **Conservative routing**: "毛利率是多少" (document QA) vs "根据收入和营业成本计算毛利率" (calculation pipeline).
- **LLM bypass**: `EXECUTED` and `BLOCKED` calculations skip the LLM entirely.
- **Additive API**: `AnswerResult.calculations` is `()` by default; `to_legacy_dict` emits it only when non-empty.

- **Branch**: `feat/nf-03-financial-calculation-pipeline`
- **Base commit**: `739949b` (Phase 2 merge)
- **Commits**: 10 (baseline characterization → domain objects → primitives → router → extraction → registry/executor → renderer → orchestrator integration → e2e tests → docs)

## Architecture

### Layer Dependency

```
domain → finance → application → services
```

- `src/domain/` — pure data objects (no heavy deps, stdlib only)
- `src/finance/` — calculation primitives, pipeline, registry (depends on `domain` only)
- `src/application/` — orchestrator (depends on `domain`, `finance`, `retrieval`, `generation`)
- `src/services/` — facade and legacy compat (depends on `application`)

`src/finance/` must NOT import from `src.services` or `src.application`.

### Module Structure

```
src/
├── domain/
│   ├── calculation.py          # CalculationOperation, CalculationStatus,
│   │                           # CalculationOperand, CalculationPlan,
│   │                           # CalculationResult, NOT_APPLICABLE_RESULT
│   ├── evidence.py             # EvidenceItem (from_chunk / to_chunk)
│   └── answer.py               # AnswerResult (+ calculations field)
├── finance/
│   ├── primitive_tools.py      # Decimal-based financial primitives
│   ├── unit_normalizer.py      # Scale word normalization (千万/百万/万/亿/...)
│   ├── evidence_extractor.py   # Extract CalculationOperands from EvidenceItems
│   ├── operation_router.py     # Route question → CalculationOperation
│   ├── metric_lexicon.py       # Metric keyword → operation mapping
│   ├── calculation_registry.py # OperationEntry registry (9 operations)
│   ├── calculation_executor.py # execute_plan(plan) → CalculationResult
│   ├── calculation_renderer.py # render_calculation_result(result) → str
│   ├── calculation_pipeline.py # CalculationPipeline.try_calculate()
│   └── __init__.py             # Public API exports
└── application/
    └── rag_orchestrator.py     # RAGOrchestrator (+ calculation_pipeline param)
```

## Calculation Flow

```
Question + Intent + Evidence
    │
    ▼
┌─────────────────┐
│ operation_router│  3-gate routing:
│ route_calc()    │  1. intent == financial_calculation?
│                 │  2. explicit calculation verb? (计算/计算/Compute/...)
│                 │  3. metric or operation keyword match?
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
 NOT_APPL  READY
    │         │
    ▼         ▼
 bypass   ┌──────────────────┐
          │ evidence_extractor│  Extract operands from EvidenceItems:
          │ extract_operands()│  - parse financial numbers
          │                  │  - bind to metric roles (revenue, cogs, etc.)
          │                  │  - normalize units (千万, 百万, million, ...)
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ _build_plan()    │  Create CalculationPlan:
          │                  │  - validate required roles present
          │                  │  - BLOCKED if missing roles
          │                  │  - BLOCKED if generic op has no operands
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ calculation_     │  Execute plan:
          │ executor         │  - look up OperationEntry in registry
          │ execute_plan()   │  - validate operand count
          │                  │  - call primitive function
          │                  │  - EXECUTED / BLOCKED / FAILED
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ calculation_     │  Render result:
          │ renderer         │  - EXECUTED → multi-line answer
          │ render_result()  │    (metric label, value, formula, citations)
          │                  │  - BLOCKED → single-line refusal
          └────────┬─────────┘
                   │
                   ▼
              AnswerResult
              (+ calculations field)
```

## CalculationOperation Enum (9 operations)

| Operation | Formula | Version | Min Operands | Roles |
|-----------|---------|---------|-------------|-------|
| `DIFFERENCE` | `a - b` | `difference.v1` | 2 | (none, generic) |
| `GROWTH_RATE` | `(current - previous) / previous` | `growth_rate.v1` | 2 | current, previous |
| `PERCENTAGE_SHARE` | `part / total` | `percentage_share.v1` | 2 | part, total |
| `SUM` | `sum(values)` | `sum.v1` | 1 | (none, generic) |
| `AVERAGE` | `sum(values) / count` | `average.v1` | 1 | (none, generic) |
| `GROSS_MARGIN` | `(revenue - cogs) / revenue` | `gross_margin.v1` | 2 | revenue, cogs |
| `NET_MARGIN` | `net_income / revenue` | `net_margin.v1` | 2 | revenue, net_income |
| `DEBT_RATIO` | `total_liabilities / total_assets` | `debt_ratio.v1` | 2 | total_liabilities, total_assets |
| `SCALE_CONVERSION` | (v1: always declines) | `scale_conversion.v1` | 1 | (none, generic) |

**Note**: `ROE` and `CAGR` are intentionally NOT included in v1. They will be added in a future phase.

## CalculationStatus State Machine

```
                  ┌────────────────┐
                  │ NOT_APPLICABLE │  (routing did not match)
                  └───────┬────────┘
                          │
                    (bypass LLM,
                     no calculations)
                          │
                          ▼
                  ┌───────────────┐
    ┌─────────────│     READY     │←──────────────┐
    │             └───────┬───────┘               │
    │                     │                       │
    │              (execute plan)                 │
    │                     │                       │
    │            ┌────────┴────────┐              │
    │            │                 │              │
    │            ▼                 ▼              │
    │  ┌──────────────┐  ┌──────────────┐        │
    │  │   EXECUTED   │  │   BLOCKED    │        │
    │  │ (success)    │  │ (declined)   │        │
    │  └──────┬───────┘  └──────┬───────┘        │
    │         │                 │                │
    │   (bypass LLM,      (bypass LLM,           │
    │    calculations)     calculations)         │
    │         │                 │                │
    │         └────────┬────────┘                │
    │                  │                         │
    │                  ▼                         │
    │          ┌──────────────┐                  │
    └──────────│   FAILED     │  (exception)     │
               └──────┬───────┘                  │
                      │                         │
                (continue to LLM,               │
                 no calculations)               │
                      │                         │
                      └─────────────────────────┘
```

## Orchestrator Integration

### Insertion Point

The calculation pipeline runs **after context build, before the deterministic context answer extractor / LLM**:

```python
# 1. Retrieve chunks
# 2. Build context
context, sources = self._context_builder.build(chunks)

# 3. Deterministic calculation pipeline (Phase 3)
if self._calculation_pipeline is not None:
    evidence = tuple(EvidenceItem.from_chunk(c) for c in chunks)
    calculation_result = self._calculation_pipeline.try_calculate(
        question, intent, evidence,
    )
    if calculation_result.status is not CalculationStatus.NOT_APPLICABLE:
        calculation_answer = render_calculation_result(calculation_result)

# 4. LLM bypass on EXECUTED / BLOCKED
if calculation_result.status in (EXECUTED, BLOCKED):
    answer = calculation_answer
    # skip LLM
else:
    # continue: deterministic context answer → LLM
    answer = await self._llm_gateway.generate(context, question)
```

### LLM Bypass Logic

| Status | LLM Called? | calculations field |
|--------|------------|-------------------|
| `NOT_APPLICABLE` | Yes | `()` (empty) |
| `READY` | Yes | `()` (empty) |
| `EXECUTED` | **No** | `(calc_dict,)` |
| `BLOCKED` | **No** | `(calc_dict,)` |
| `FAILED` | Yes | `()` (empty) |

### Trace Diagnostics

When the pipeline runs (status != `NOT_APPLICABLE`), trace_data includes:

```json
{
  "diagnostics": {
    "calculation": {
      "status": "executed",
      "operation": "gross_margin",
      "formula_version": "gross_margin.v1",
      "operand_count": 2,
      "error_code": null
    }
  }
}
```

When `NOT_APPLICABLE`, `diagnostics.calculation` is `null`.

## AnswerResult.calculations (Additive Field)

```python
@dataclass(frozen=True)
class AnswerResult:
    # ... existing fields unchanged ...
    calculations: tuple[dict[str, Any], ...] = ()
```

`to_legacy_dict` emits `calculations` only when non-empty:

```python
if self.calculations:
    result["calculations"] = list(self.calculations)
```

Existing API consumers are unaffected: when the pipeline is not configured (`calculation_pipeline=None`) or the question is not a calculation (`NOT_APPLICABLE`), `calculations` is `()` and absent from the legacy dict.

## Evidence-Bound Operands

Every `CalculationOperand` binds to the `EvidenceItem` it was extracted from:

```python
@dataclass(frozen=True)
class CalculationOperand:
    value: Decimal
    raw_text: str           # original text span (e.g. "$1,000,000")
    role: str | None        # metric role (e.g. "revenue", "cogs")
    evidence_chunk_id: str  # source chunk ID
    document_name: str | None
    page: int | None
    source_text: str        # full evidence content for citation
```

This ensures every calculation operand is traceable to a retrieved document chunk.

## Security Constraints

- **No `eval` / `exec` / `compile`**: verified by `scripts/check_eval_leakage.py`.
- **No model-generated code execution**: the model never produces code; all formulas are fixed Python functions.
- **No `__import__` or dynamic import**: all imports are static.
- **Decimal-only arithmetic**: `Decimal` with `ROUND_HALF_UP`, explicit precision; NaN and Infinity are rejected.
- **No access to evaluation labels / oracle / support pages / standard answers**: production retrieval is unchanged.

## Test Coverage

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_calculation_baseline.py` | ~30 | Intent routing baseline, AnswerResult field baseline, orchestrator param baseline, finance package migration |
| `test_primitive_tools.py` | ~50 | Primitive correctness, NaN rejection, layer purity |
| `test_calculation_domain.py` | ~40 | Domain object immutability, serialization, status transitions |
| `test_operation_router.py` | ~40 | 3-gate routing, metric lexicon, Chinese/English keywords |
| `test_unit_normalizer.py` | ~30 | Scale word normalization (千万/百万/万/亿/million/billion/...) |
| `test_evidence_extractor.py` | ~50 | Operand extraction, role binding, multi-chunk evidence |
| `test_calculation_registry.py` | ~20 | Registry completeness, formula version, frozen dataclass |
| `test_calculation_executor.py` | ~53 | Execute plan, status mapping, operand validation, error codes |
| `test_calculation_renderer.py` | ~19 | EXECUTED/BLOCKED rendering, percentage format, citations |
| `test_calculation_pipeline.py` | ~20 | Pipeline integration, NOT_APPLICABLE/EXECUTED/BLOCKED paths |
| `test_e2e_calculation.py` | 17 | End-to-end orchestrator integration, LLM bypass, trace diagnostics |

**Total**: 1091 passed, 12 skipped, 0 failures.

## Commit History

| # | SHA | Message |
|---|-----|---------|
| 1 | `3be0b11` | test: characterize financial calculation baseline |
| 2 | `fa4a877` | feat: add financial calculation domain objects |
| 3 | `9e9bff1` | refactor: move deterministic financial primitives into finance package |
| 4 | `4045c54` | feat: add operation router and financial metric lexicon |
| 5 | `f2c2fc0` | feat: add unit normalization and evidence extraction |
| 6 | `80850c4` | feat: add calculation registry and executor |
| 7 | `0856875` | feat: add deterministic calculation renderer |
| 8 | `e82a0e9` | feat: integrate calculation pipeline into RAG orchestrator |
| 9 | `29040aa` | test: add end-to-end RAG calculator regression suite |
| 10 | (this commit) | docs: document orchestrated financial calculation pipeline |

## Future Work (Not in Phase 3)

- `ROE` and `CAGR` operations (require additional operand roles)
- `SCALE_CONVERSION` with explicit from/to scale params (v1 always declines)
- Multi-step calculation chains (e.g. "compute gross margin, then compare to last year")
- Calculation result caching for repeated queries
- Phase 4: answer validation for calculation results
- Phase 5: sealed test suite for calculation accuracy
