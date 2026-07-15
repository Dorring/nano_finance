# FinQuery RAG evaluation fixtures

This directory contains small, commit-safe fixtures for the offline RAG evaluation layer.
They are not intended to represent product quality; they validate that scoring,
reporting, comparison, and CI gate behavior keep working.

## Files

- `golden_smoke.jsonl` - minimal golden/replay cases.
- `predictions_smoke.jsonl` - deterministic predictions matching the smoke cases.
- `baseline_smoke_report.json` - checked scorer output used by the smoke regression gate.

Do not commit real customer documents, trace databases, ChromaDB data, model outputs with
sensitive content, or large generated reports here.

## Local commands

From `finquery_rag/backend`:

```bash
python -m src.eval_cli score   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --out /tmp/finquery_eval_report.json

python -m src.eval_cli compare   --baseline /tmp/finquery_eval_report.json   --candidate /tmp/finquery_eval_report.json   --out /tmp/finquery_eval_compare.json
```

## CI quality gate

For the commit-safe smoke gate, run:

```bash
python scripts/ci_eval_gate.py
```

Set `FINQUERY_EVAL_ARTIFACT_DIR=/path/to/artifacts` when CI should collect the
generated report, comparison, and JUnit XML. Without it, artifacts are written to
the system temp directory. The backend GitHub Actions workflow sets this variable
and uploads the generated files as the `finquery-eval-gate` artifact.

Use `gate` directly when a retrieval, prompt, reranker, or answer-validation change needs a
custom pass/fail signal in CI:

```bash
python -m src.eval_cli gate   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --min-pass-rate 1.0   --max-missing 0   --out /tmp/finquery_eval_report.json   --junit-out /tmp/finquery_eval_gate.xml
```

For regression checks against a checked baseline report:

```bash
python -m src.eval_cli gate   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --baseline eval/baseline_smoke_report.json   --tolerance 0.01   --min-pass-rate 0.95   --max-missing 0   --comparison-out /tmp/finquery_eval_compare.json   --junit-out /tmp/finquery_eval_gate.xml
```

Exit codes:

- `0` - gate passed.
- `1` - valid inputs, but thresholds or baseline comparison failed.
- `2` - invalid inputs or malformed fixtures.

The JUnit output is intentionally small: one testcase per gate check. This makes
GitHub Actions and other CI systems annotate the failed threshold instead of only
showing a generic command failure.

## Real evaluation workflow

1. Create a project-specific golden JSONL with expected sources, phrases, numbers,
   intent labels, calculations, and no-answer cases.
2. Run `python -m src.eval_cli run --cases <cases.jsonl> --out <predictions.jsonl> --user-id <id>`.
3. Score the predictions or run the `gate` command directly.
4. Compare candidate reports against a checked baseline before merging retrieval or
   prompt changes.

## Calculation consistency

When a prediction includes `calculations`, the eval scorer also checks whether
percentage claims in `answer` are consistent with those deterministic calculation
outputs. The aggregate metric is `answer_calculation_consistency`.
