# Phase 5 Evaluation Data

This directory holds the evaluation dataset for Phase 5 of the FinQuery RAG
evaluation system. The data is split into three partitions with strict
label-isolation rules so that the blind runner never sees expected fields and
the sealed partition's labels are never committed to version control.

## Three Partitions

| Partition    | Purpose                                             | Min Cases |
|--------------|----------------------------------------------------|-----------|
| `dev`        | Visible development set for smoke tests and debugging. Questions **and** labels are committed. | 40+ |
| `calibration`| Used to search the calibration parameter space. Questions and labels are committed.            | 80+ |
| `sealed`     | Scored exactly once per candidate. Only **questions** are committed; labels live locally in `.sealed/`. | 120+ |

The minimum case counts above are the targets the dataset must reach before
the sealed run is unsealed. The current placeholder sets are intentionally
smaller (10 dev, 5 sealed) to demonstrate the format.

## Questions and Labels Are Separated

Each partition stores its data in two separate files:

- `questions.jsonl` — one `EvaluationQuery` per line. Contains **only** the
  question, document names, tags, and metadata. It must never contain any
  `expected_*` field. This is the only file the blind runner reads.
- `labels.jsonl` — one `EvaluationLabel` per line. Contains all `expected_*`
  fields (expected sources, numbers, calculations, intent, answerability,
  validation status, no-answer flag, required/forbidden terms, slice tags).
  This file is only read by the sealed scorer after predictions are sealed.

The separation enforces label isolation structurally: a question file can be
shared, inspected, or fed to the blind runner without leaking any answer
signal.

## Sealed Labels Must NOT Be Committed

The sealed partition's `labels.jsonl` is stored **locally only** under
`../../.sealed/` (the `.sealed/` directory at the backend root). That
directory is listed in `.gitignore` and must never be committed to git.

Only `questions.jsonl` and `manifest.public.json` are committed for the
sealed partition. The public manifest intentionally carries
`labels_sha256: null` so the label hash is not leaked.

A `.gitkeep` file lives in `.sealed/` so the directory exists in fresh
checkouts. Operators place the real `labels.jsonl` there manually before
scoring (see `docs/evaluation/phase5-sealed-runbook.md`).

## Schemas

JSON Schema definitions for the three record types live in `schemas/`:

- `schemas/evaluation-query.schema.json` — `EvaluationQuery` (no `expected_*`)
- `schemas/evaluation-label.schema.json` — `EvaluationLabel` (all `expected_*`)
- `schemas/evaluation-manifest.schema.json` — `DatasetManifest`

The canonical Python dataclasses are in
`src/evaluation/schemas.py`; these JSON Schema files document the on-disk
JSONL shape and are used by validation tooling.

## Manifests

Each partition has a `manifest.json` (or `manifest.public.json` for sealed)
recording the partition name, case count, SHA256 of the questions/labels
files, creation timestamp, and the slice tags covered. SHA256 values are
placeholders here and are recomputed by the generation scripts.
