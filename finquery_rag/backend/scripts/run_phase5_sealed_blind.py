#!/usr/bin/env python3
"""Run the Phase 5 sealed blind prediction step (no scoring).

This script ONLY produces predictions. It:

- Loads questions from ``eval_data/phase5/sealed/questions.jsonl`` (NO
  labels are ever accessed).
- Runs the RC-frozen RAG engine against each question via
  ``run_blind_queries()``.
- Writes predictions atomically to
  ``artifacts/evaluation/phase5/sealed-v2/predictions.jsonl``.
- Computes and writes both raw and canonical SHA256 hashes.
- Writes a ``predictions.jsonl.complete.json`` marker when done.
- Writes a run manifest capturing the full environment state.

Atomic write protocol:

1. Write to ``predictions.jsonl.tmp``.
2. ``os.fsync()`` the file.
3. ``os.replace()`` to ``predictions.jsonl``.
4. Compute raw SHA256 of the final file.
5. Compute canonical SHA256 (``compute_jsonl_sha256``).
6. Write ``predictions.jsonl.sha256`` (raw hash).
7. Write ``predictions.jsonl.canonical.sha256`` (canonical hash).
8. Write ``predictions.jsonl.complete.json`` marker.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed_blind.py
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed_blind.py --dry-run

Environment:
    Model server must be running at http://localhost:8500.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries  # noqa: E402
from src.evaluation.manifests import (  # noqa: E402
    compute_file_sha256,
    compute_jsonl_sha256,
    create_manifest,
)

SEALED_QUESTIONS_PATH = (
    BACKEND_DIR / "eval_data" / "phase5" / "sealed" / "questions.jsonl"
)
OUTPUT_DIR = BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "sealed-v2"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
PREDICTIONS_TMP_PATH = OUTPUT_DIR / "predictions.jsonl.tmp"
RAW_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.sha256"
CANONICAL_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.canonical.sha256"
COMPLETE_MARKER_PATH = OUTPUT_DIR / "predictions.jsonl.complete.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run-manifest.json"

MODEL_SERVER_ENDPOINT = "http://localhost:8500/v1"
MODEL_NAME = "finquery-finance-sft1147"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _init_rag_engine() -> Any:
    """Initialize the RC-frozen RAG engine against the local model server."""
    from openai import OpenAI

    from src.services.rag_engine import RAGEngine

    client = OpenAI(api_key="sk-placeholder", base_url=MODEL_SERVER_ENDPOINT)
    return RAGEngine(llm_client=client, model_name=MODEL_NAME)


def _write_text_atomic(path: Path, content: str) -> None:
    """Write text to ``path`` via a temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with open(  # noqa: PTH123 - need a raw handle for fsync
            path.parent / f".{path.name}.tmp",
            "w",
            encoding="utf-8",
        ) as fh:
            tmp_name = fh.name
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


async def _run_blind() -> dict[str, Any]:
    """Load questions, run the engine, and write predictions atomically."""
    print("Loading sealed questions (NO labels accessed)...")
    queries = load_queries(str(SEALED_QUESTIONS_PATH))
    print(f"Loaded {len(queries)} sealed questions")

    engine = _init_rag_engine()
    print("RAG engine initialized successfully")

    print("Running blind evaluation on sealed set...")
    started_at = _now_iso()
    predictions = await run_blind_queries(queries, engine, user_id=1, n_results=3)
    ended_at = _now_iso()
    print(f"Generated {len(predictions)} predictions")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: write predictions to the .tmp file.
    with PREDICTIONS_TMP_PATH.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred.to_dict(), ensure_ascii=False, default=str) + "\n")
        # Step 2: fsync the file.
        fh.flush()
        os.fsync(fh.fileno())

    # Step 3: atomically replace the final file.
    os.replace(PREDICTIONS_TMP_PATH, PREDICTIONS_PATH)
    print(f"Predictions saved to: {PREDICTIONS_PATH}")

    # Step 4 & 5: compute raw and canonical SHA256 of the final file.
    raw_sha = compute_file_sha256(PREDICTIONS_PATH)
    canonical_sha = compute_jsonl_sha256(PREDICTIONS_PATH)
    print(f"Predictions raw SHA256:      {raw_sha}")
    print(f"Predictions canonical SHA256: {canonical_sha}")

    # Step 6: write raw hash.
    _write_text_atomic(RAW_SHA_PATH, raw_sha + "\n")

    # Step 7: write canonical hash.
    _write_text_atomic(CANONICAL_SHA_PATH, canonical_sha + "\n")

    # Step 8: write the completion marker.
    marker = {
        "completed": True,
        "completed_at": _now_iso(),
        "raw_sha256": raw_sha,
        "canonical_sha256": canonical_sha,
        "case_count": len(predictions),
    }
    _write_text_atomic(
        COMPLETE_MARKER_PATH,
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    # Write the run manifest with all metadata.
    manifest = create_manifest(
        run_id=f"sealed-v2-{started_at}",
        run_type="sealed",
        n_results=3,
        case_count=len(predictions),
        run_started_at=started_at,
        run_ended_at=ended_at,
        questions_path=SEALED_QUESTIONS_PATH,
        predictions_path=PREDICTIONS_PATH,
        repo_path=BACKEND_DIR,
    )
    _write_text_atomic(
        RUN_MANIFEST_PATH,
        json.dumps(
            manifest.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )
    print(f"Run manifest saved to: {RUN_MANIFEST_PATH}")

    return marker


def _dry_run() -> int:
    """Validate inputs without running the engine."""
    print("=== Dry run: validating inputs ===")
    if not SEALED_QUESTIONS_PATH.is_file():
        print(f"ERROR: sealed questions not found at {SEALED_QUESTIONS_PATH}")
        return 1
    queries = load_queries(str(SEALED_QUESTIONS_PATH))
    print(f"OK: {len(queries)} sealed questions parsed successfully")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OK: output directory writable at {OUTPUT_DIR}")
    print("Dry run passed. No predictions were generated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 sealed blind predictions (no scoring)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running the RAG engine.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Sealed Blind Prediction (v2)")
    print("=" * 60)

    if args.dry_run:
        return _dry_run()

    marker = asyncio.run(_run_blind())
    print("\nBlind prediction completed successfully.")
    print(f"  Case count: {marker['case_count']}")
    print(f"  Raw SHA256: {marker['raw_sha256']}")
    print(f"  Canonical SHA256: {marker['canonical_sha256']}")
    print("Score later with: python3 scripts/score_phase5_sealed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
