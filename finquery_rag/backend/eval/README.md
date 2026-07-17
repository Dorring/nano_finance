# FinQuery RAG evaluation fixtures

This directory contains small, commit-safe fixtures for the offline RAG evaluation layer.
They are not intended to represent product quality; they validate that scoring,
reporting, comparison, and CI gate behavior keep working.

## Files

- `golden_smoke.jsonl` - expanded deterministic golden/replay cases.
- `predictions_smoke.jsonl` - deterministic predictions matching all smoke cases.
- `baseline_smoke_report.json` - checked scorer output used by the smoke regression gate.
- `real_eval_template.jsonl` - copy/edit template for real PDF evaluation. It
  contains placeholders and must not be scored directly.

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
generated fixture audit, report, comparison, retrieval diagnostics, and JUnit XML. Without it,
artifacts are written to the system temp directory. The backend GitHub Actions
workflow sets this variable and uploads the generated files as the
`finquery-eval-gate` artifact.

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

## Fixture audit

Before a golden set becomes a merge gate, audit its coverage and taxonomy:

```bash
python -m src.eval_cli audit-fixtures   --cases eval/golden_smoke.jsonl   --min-cases 12   --required-tag smoke   --required-tag citation   --required-tag no_answer   --required-tag calculation   --require-expected-intent   --out /tmp/finquery_fixture_audit.json
```

The audit reports tag counts, intent counts, coverage rates, missing required
tags, and per-case quality issues. It exits with `1` when configured fixture
policy fails, and `2` when the JSONL itself is malformed.

## Retrieval diagnostics

When answer quality drops, run retrieval diagnostics before changing prompts. The
report is answer-independent and shows whether expected sources appeared in the
candidate list, including Recall@K, MRR, missed sources, and worst cases:

```bash
python -m src.eval_cli retrieval-diagnostics   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --k 1   --k 3   --k 5   --out /tmp/finquery_retrieval_diagnostics.json
```

By default diagnostics inspect `retrieved_chunks`. Use `--candidate-field sources`
when you need to debug final cited sources instead of raw retrieval candidates.

## Interview/demo report

For resume and interview demos, generate a compact report that groups the same
offline metrics into answer quality, citation grounding, retrieval quality,
no-answer behavior, and weak cases:

```bash
python -m src.eval_cli interview-report \
  --cases eval/golden_smoke.jsonl \
  --predictions eval/predictions_smoke.jsonl \
  --k 1 --k 3 --k 5 \
  --out /tmp/finquery_interview_report.json
```

Use the `resume_metrics` block for defensible project claims, for example:

- Golden answer pass rate
- Citation recall
- Retrieval Recall@5 / MRR
- No-answer accuracy

These numbers are only meaningful for the dataset named in the report. For a
real demo, create a small project-specific golden set from uploaded financial
documents and regenerate this report after every retrieval, reranker, prompt, or
model-service change.

## Real evaluation workflow

1. Create a project-specific golden JSONL with expected sources, phrases, numbers,
   intent labels, calculations, and no-answer cases.
2. Run `python -m src.eval_cli run --cases <cases.jsonl> --out <predictions.jsonl> --user-id <id>`.
3. Run fixture audit whenever the golden set changes.
4. Run retrieval diagnostics to confirm expected sources are in the candidate set.
5. Score the predictions or run the `gate` command directly.
6. Compare candidate reports against a checked baseline before merging retrieval or
   prompt changes.

Start from the template when building interview metrics:

```bash
cp eval/real_eval_template.jsonl /tmp/finquery_real_eval.jsonl
```

Replace every `REPLACE_WITH_*` placeholder by manually inspecting the uploaded
PDF. Do not report resume/interview metrics from unedited template fixtures or
from synthetic smoke fixtures.

## Calculation consistency

When a prediction includes `calculations`, the eval scorer also checks whether
percentage claims in `answer` are consistent with those deterministic calculation
outputs. The aggregate metric is `answer_calculation_consistency`.
