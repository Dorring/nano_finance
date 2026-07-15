# FinQuery SFT1147 RAG integration runbook

This runbook connects FinQuery RAG to the locally trained finance SFT model through the OpenAI-compatible nanochat adapter. It is intended for demo, interview, and server smoke-test workflows.

## Model baseline

Use SFT1147 as the current generation baseline for FinQuery RAG. Do not keep tuning RAG internals before this integration is working end-to-end.

- checkpoint: `/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_final_mixdata/model_001147.pt`
- metadata: `/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_final_mixdata/meta_001147.json`
- source: `sft`
- model tag: `d24_final_mixdata`
- step: `1147`
- OpenAI-compatible model name: `finquery-finance-sft1147`
- context length: `2048` tokens
- recommended temperature: `0`
- recommended max tokens: `512`
- API key: local dummy value is acceptable, for example `not-needed-for-local`

## Start the model adapter

Run this on the GPU server from the nanochat repository root:

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat
source /home/mxf/projects/Qhhhhhhaaa/nanochat/nano/bin/activate

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=7

python -m scripts.chat_openai_compat \
  --source sft \
  --model-tag d24_final_mixdata \
  --step 1147 \
  --model-name finquery-finance-sft1147 \
  --port 8500 \
  --host 0.0.0.0 \
  --temperature 0 \
  --max-tokens 512
```

The adapter exposes:

- `GET /v1/models`
- `POST /v1/chat/completions`
- non-streaming responses at `choices[].message.content`
- streaming deltas at `choices[].delta.content`

It accepts `system`, `user`, and `assistant` messages. The adapter merges system content into the user prompt to match the nanochat training format and applies prompt-budget truncation for long RAG contexts.

## Configure FinQuery backend

FinQuery already uses the OpenAI SDK client. Switch the backend to SFT1147 with environment variables:

```bash
export LLM_API_BASE_URL=http://127.0.0.1:8500/v1
export LLM_MODEL_NAME=finquery-finance-sft1147
export LLM_API_KEY=not-needed-for-local
```

These variables are the model-service switching point. Do not introduce a second `FINQUERY_LLM_*` naming scheme unless the backend is explicitly changed.

For persistent runtime paths on the server, also set:

```bash
export CHROMA_PATH=/var/lib/finquery/chroma_db
export DOCUMENT_REGISTRY_DB_PATH=/var/lib/finquery/document_registry.db
export BM25_DB_PATH=/var/lib/finquery/rag_bm25.db
export SESSIONS_DB_PATH=/var/lib/finquery/sessions.db
export TRACE_DB_PATH=/var/lib/finquery/trace_log.db
export FEEDBACK_DB_PATH=/var/lib/finquery/feedback.db
```

## Smoke checks before starting the UI

Check the model adapter first:

```bash
curl http://127.0.0.1:8500/v1/models

curl -s http://127.0.0.1:8500/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer not-needed-for-local' \
  -d '{"model":"finquery-finance-sft1147","messages":[{"role":"user","content":"Answer briefly: what is revenue growth?"}],"temperature":0,"max_tokens":128}'
```

Then run backend-only checks from `finquery_rag/backend`:

```bash
python scripts/ci_eval_gate.py
python scripts/ci_preflight_smoke.py
python -m src.eval_cli doctor --bm25-db "$BM25_DB_PATH" --out /tmp/finquery_doctor.json
```

`ci_eval_gate.py` and `ci_preflight_smoke.py` do not call the model. They verify that the deterministic eval, baseline, preflight, migration audit, and retrieval diagnostics still work.

## Start FinQuery

Backend:

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/backend
uv sync
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd /mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/frontend
npm install
npm run dev -- --host 0.0.0.0
```

Operational probes:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

`/readyz` should show the configured model name under the redacted LLM config snapshot. It does not call the model.

## End-to-end interview demo script

1. Start the SFT1147 OpenAI-compatible adapter.
2. Start the FinQuery backend with `LLM_API_BASE_URL`, `LLM_MODEL_NAME`, and `LLM_API_KEY` set as above.
3. Start the frontend.
4. Register or log in.
5. Upload one financial report PDF.
6. Ask one factual question, for example: `What was Q3 revenue?`
7. Ask one calculation question, for example: `What was revenue growth from 100 to 120?`
8. Ask one no-answer question, for example: `What dividend did the company declare in Q3?`
9. Show citations/sources in the answer UI.
10. Show trace diagnostics, feedback, eval gate artifacts, and preflight output.

## Model capability boundary

Use SFT1147 as the RAG answer generator, not as an unaided financial oracle.

- Good fit: grounded financial QA after RAG context injection, simple summaries, basic calculation explanations.
- Risky: answering from parametric memory without retrieved context.
- Citations: the model can follow citation prompts, but FinQuery source/citation logic remains authoritative.
- No-answer: the model's own refusal behavior is not the safety boundary; keep FinQuery `context_sufficient` and no-answer gates in control.
- Structured outputs: keep post-processing validation for citations, calculations, confidence, and traceability.

## Next evaluation step

After the demo path is stable, create a 20-30 case real-document regression set comparing:

- SFT1147 + FinQuery RAG
- default or placeholder model + FinQuery RAG

Label expected sources/pages, no-answer cases, calculations, and intent. Keep private documents out of Git; commit only sanitized fixtures or generated reports that contain no customer content.