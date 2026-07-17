# FinQuery RAG interview demo guide

This guide is for presenting FinQuery as an AI application / RAG engineering
project. It focuses on what to show, how to run it, and which metrics are
defensible in an interview.

## One-minute project pitch

FinQuery is a local-first financial document Q&A system. Users upload PDFs, the
backend builds a tenant-scoped RAG index, and answers are generated from
retrieved evidence with page-level citations.

The system is split into separate layers:

- ingestion: PDF parsing, table handling, lifecycle registry, de-duplication;
- retrieval: dense Chroma search, BM25, RRF, reranker hooks;
- context building: section-aware parent-child expansion and token budgeting;
- generation: OpenAI-compatible local/remote LLM client;
- reliability: no-answer gate, answer validation, trace logging, replay, eval;
- memory: session history and editable preference profile for query rewriting.

This separation is the main design point: the generator is not expected to solve
retrieval, citation, memory, and evaluation by itself.

## Architecture summary

```text
PDF upload
  -> safe filename + tenant auth
  -> DocumentRegistry lifecycle
  -> PyMuPDF/Camelot extraction
  -> structured chunks + tables + front matter
  -> section-aware parent-child metadata
  -> Chroma dense index + SQLite BM25/FTS5

Query
  -> intent routing
  -> optional session/profile rewrite
  -> dense + BM25 retrieval
  -> RRF + optional reranker
  -> parent context expansion
  -> context sufficiency / no-answer gate
  -> LLM generation
  -> validation + citations + trace
```

## What to show live

Use one clean PDF and a fixed question script. Re-upload the PDF after any
ingestion/indexing change so newly added metadata, such as `parent_id` and
`section_path`, is present.

Recommended demo questions:

1. Front matter / deterministic answer

   ```text
   What is the title of this paper/report?
   ```

2. Document QA

   ```text
   What problem does this document address?
   ```

3. Section-level summary

   ```text
   Summarize the main contribution or key financial highlights.
   ```

4. Number/table question

   ```text
   What was the reported revenue / margin / cash balance?
   ```

5. No-answer gate

   ```text
   What was the CEO's salary?
   ```

   Use a question whose answer is not present in the uploaded document. The
   desired behavior is a refusal, not a hallucinated answer.

6. Follow-up memory

   ```text
   Summarize the FY2024 revenue trend.
   What about margin?
   ```

   If a memory profile is configured, explain that it is used only to resolve
   ambiguity in query rewriting, not as factual evidence.

## Local model integration

FinQuery talks to the generator through an OpenAI-compatible client. The current
local SFT baseline can be exposed by the adapter from the model training side.

Backend environment:

```bash
export LLM_API_BASE_URL=http://127.0.0.1:8500/v1
export LLM_MODEL_NAME=finquery-finance-sft1147
export LLM_API_KEY=not-needed-for-local
```

This does not conflict with embedding or reranker models. The generator answers
from retrieved context; embedding/reranker models control evidence retrieval and
ordering.

## Retrieval model configuration

Defaults are small and CI-safe:

```bash
export EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
export RAG_RERANKER=heuristic
export RAG_CANDIDATE_MULTIPLIER=2
```

Optional production-like alternatives:

```bash
export EMBEDDING_MODEL_NAME=/local/path/to/bge-small-en-v1.5
export RAG_RERANKER=cross-encoder
export RAG_RERANKER_MODEL=/local/path/to/bge-reranker-base
```

Changing the embedding model requires rebuilding/re-uploading the document
index, because existing vectors were produced by the previous model.

## Runbook

Backend:

```bash
cd finquery_rag/backend
uv sync
export LLM_API_BASE_URL=http://127.0.0.1:8500/v1
export LLM_MODEL_NAME=finquery-finance-sft1147
export LLM_API_KEY=not-needed-for-local
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd finquery_rag/frontend
npm install
npm run dev -- --host 0.0.0.0
```

Operational checks:

```bash
cd finquery_rag/backend
python -m src.eval_cli doctor --warn-only --out /tmp/finquery_doctor.json
python -m src.eval_cli preflight --warn-only --out /tmp/finquery_preflight.json
```

## Evaluation workflow

There are two evaluation layers.

### Commit-safe smoke eval

These fixtures are synthetic and deterministic. They validate scoring, replay,
gate, diagnostics, and report tooling. They are useful for CI but should not be
presented as real product accuracy.

```bash
cd finquery_rag/backend
python -m src.eval_cli retrieval-eval-bundle \
  --cases eval/golden_smoke.jsonl \
  --predictions eval/predictions_smoke.jsonl \
  --k 1 --k 3 --k 5 \
  --out-dir /tmp/finquery_smoke_eval
```

### Real PDF eval

For interview metrics, build a small golden set from real uploaded PDFs:

- 20-30 cases total;
- include title/front matter, factual QA, table/number, section summary,
  follow-up, and no-answer cases;
- every answerable case should include expected sources;
- no-answer cases should set `expected_no_answer: true`.

Template:

```bash
cp eval/real_eval_template.jsonl /tmp/finquery_real_eval.jsonl
# edit expected answers and sources after manually inspecting the PDF
python -m src.eval_cli run \
  --cases /tmp/finquery_real_eval.jsonl \
  --out /tmp/finquery_real_predictions.jsonl \
  --user-id <your_user_id>

python -m src.eval_cli retrieval-eval-bundle \
  --cases /tmp/finquery_real_eval.jsonl \
  --predictions /tmp/finquery_real_predictions.jsonl \
  --k 1 --k 3 --k 5 \
  --out-dir /tmp/finquery_real_eval_report
```

Use the generated `interview_report.json` `resume_metrics` block only after you
have verified the real golden cases.

## Metrics that are safe to put on a resume

Only report metrics with the dataset name and size. Example format:

```text
Built an offline eval/replay suite for a 30-case financial PDF golden set,
tracking answer pass rate, citation recall, no-answer accuracy, Recall@5, and
MRR across retrieval/reranker changes.
```

Avoid vague claims such as "95% accurate" unless the golden set, scoring method,
and document domain are stated.

## Interview talking points

- Why hybrid retrieval: dense search handles semantic similarity; BM25 handles
  exact financial terms, numbers, and tickers.
- Why reranking: first-stage retrieval is optimized for recall; reranker improves
  evidence ordering before context construction.
- Why parent-child context: small chunks retrieve precisely, but parent section
  expansion gives the generator enough local context.
- Why deterministic front matter: simple facts such as title should not depend
  on LLM generation when metadata can answer them.
- Why no-answer gate: production RAG should refuse when retrieval confidence is
  low instead of letting the generator invent facts.
- Why memory is limited: session/profile memory resolves query ambiguity but is
  not used as financial evidence.
- Why eval/replay: retrieval, prompt, model, and reranker changes need regression
  checks, not just manual demos.

## Known limitations

- PDF parsing quality still depends on document layout and table extraction.
- Changing embedding models requires index rebuilds.
- Cross-encoder reranking can improve quality but adds latency and model
  deployment complexity.
- Small local SFT models should be kept grounded by retrieval and no-answer
  gates; they should not answer from parametric memory alone.
