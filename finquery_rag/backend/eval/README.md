# FinQuery RAG evaluation fixtures

This directory contains small, commit-safe fixtures for the offline RAG evaluation layer.
They are not intended to represent product quality; they only validate that the scoring,
reporting, and comparison pipeline keeps working in CI.

## Files

- `golden_smoke.jsonl` — minimal golden/replay cases.
- `predictions_smoke.jsonl` — deterministic predictions matching the smoke cases.

Do not commit real customer documents, trace databases, ChromaDB data, model outputs with
sensitive content, or large generated reports here.

## Local commands

From `finquery_rag/backend`:

```bash
python -m src.eval_cli score \
  --cases eval/golden_smoke.jsonl \
  --predictions eval/predictions_smoke.jsonl \
  --out /tmp/finquery_eval_report.json

python -m src.eval_cli compare \
  --baseline /tmp/finquery_eval_report.json \
  --candidate /tmp/finquery_eval_report.json \
  --out /tmp/finquery_eval_compare.json
```

For real evaluation:

1. Create a project-specific golden JSONL with expected sources, phrases, numbers, and no-answer cases.
2. Run `python -m src.eval_cli run --cases <cases.jsonl> --out <predictions.jsonl> --user-id <id>`.
3. Score the predictions.
4. Compare candidate reports against a checked baseline before merging retrieval or prompt changes.


## Calculation consistency

When a prediction includes `calculations`, the eval scorer also checks whether
percentage claims in `answer` are consistent with those deterministic
calculation outputs. The aggregate metric is `answer_calculation_consistency`.
