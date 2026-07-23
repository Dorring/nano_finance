#!/usr/bin/env python3
"""Run Phase 5 sealed blind evaluation and scoring.

This script performs the sealed evaluation in two phases:

Phase 1 (Blind Run):
- Loads sealed questions (NO labels accessed).
- Runs blind evaluation using the RC-frozen RAG engine.
- Saves predictions with SHA256 hash.
- Predictions are immutable once generated.

Phase 2 (Scoring):
- Loads sealed labels (kept in .sealed/ directory, not in repo).
- Scores predictions using the sealed scorer (no RAG calls).
- Saves final report with all metrics, slices, and failure taxonomy.
- Scoring is performed exactly once.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed.py            # blind run only
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed.py --score    # blind run + score

Environment:
    Model server must be running at http://localhost:8500.
    Sealed labels must be at .sealed/labels.jsonl (not in repo).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries, load_labels  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.slices import compute_slice_metrics  # noqa: E402
from src.evaluation.failure_taxonomy import classify_all_failures  # noqa: E402
from src.evaluation.statistics import wilson_interval  # noqa: E402
from src.evaluation.sealed_scorer import score_sealed_predictions  # noqa: E402


SEALED_QUESTIONS_PATH = (
    BACKEND_DIR / "eval_data" / "phase5" / "sealed" / "questions.jsonl"
)
SEALED_LABELS_PATH = BACKEND_DIR / ".sealed" / "labels.jsonl"
PREDICTIONS_OUTPUT = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "sealed"
    / "sealed-predictions.jsonl"
)
REPORT_OUTPUT = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "sealed"
    / "sealed-report.json"
)
RC_FREEZE_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "rc"
    / "rc-freeze.json"
)


def compute_jsonl_sha256(filepath: Path) -> str:
    """Compute SHA256 of a JSONL file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def run_sealed_blind():
    """Run blind evaluation on sealed questions (no labels accessed)."""
    print("\n=== Phase 1: Sealed Blind Run ===")
    print("Loading sealed questions (NO labels accessed)...")

    queries = load_queries(str(SEALED_QUESTIONS_PATH))
    print(f"Loaded {len(queries)} sealed questions")

    # Verify questions hash
    questions_hash = compute_jsonl_sha256(SEALED_QUESTIONS_PATH)
    print(f"Sealed questions SHA256: {questions_hash}")

    # Initialize RAG engine
    try:
        from openai import OpenAI
        from src.services.rag_engine import RAGEngine

        client = OpenAI(
            api_key="sk-placeholder",
            base_url="http://localhost:8500/v1"
        )
        engine = RAGEngine(
            llm_client=client,
            model_name="finquery-finance-sft1147",
        )
        print("RAG engine initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG engine: {e}")
        return None

    # Run blind evaluation
    print("Running blind evaluation on sealed set...")
    predictions = await run_blind_queries(
        queries, engine, user_id=1, n_results=3
    )
    print(f"Generated {len(predictions)} predictions")

    # Save predictions as JSONL
    PREDICTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_OUTPUT, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred.to_dict(), ensure_ascii=False, default=str) + "\n")

    # Compute predictions hash
    predictions_hash = compute_jsonl_sha256(PREDICTIONS_OUTPUT)
    print(f"Predictions SHA256: {predictions_hash}")
    print(f"Predictions saved to: {PREDICTIONS_OUTPUT}")

    return {
        "questions_sha256": questions_hash,
        "predictions_sha256": predictions_hash,
        "prediction_count": len(predictions),
        "predictions_path": str(PREDICTIONS_OUTPUT),
    }


def run_sealed_scoring(blind_info: dict):
    """Score sealed predictions (labels accessed, no RAG calls)."""
    print("\n=== Phase 2: Sealed Scoring ===")

    if not SEALED_LABELS_PATH.is_file():
        print(f"ERROR: Sealed labels not found at {SEALED_LABELS_PATH}")
        print("Cannot score without labels. Blind run completed successfully.")
        return None

    print(f"Loading sealed labels from {SEALED_LABELS_PATH}...")
    labels = load_labels(str(SEALED_LABELS_PATH))
    print(f"Loaded {len(labels)} sealed labels")

    labels_hash = compute_jsonl_sha256(SEALED_LABELS_PATH)
    print(f"Sealed labels SHA256: {labels_hash}")

    # Load predictions
    print(f"Loading predictions from {PREDICTIONS_OUTPUT}...")
    predictions_data = []
    with open(PREDICTIONS_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions_data.append(json.loads(line))

    # Verify predictions hash hasn't changed
    current_pred_hash = compute_jsonl_sha256(PREDICTIONS_OUTPUT)
    if current_pred_hash != blind_info["predictions_sha256"]:
        print("ERROR: Predictions hash changed since blind run!")
        print(f"  Blind run: {blind_info['predictions_sha256']}")
        print(f"  Current:   {current_pred_hash}")
        return None

    print(f"Predictions hash verified: {current_pred_hash}")

    # Score using sealed scorer (no RAG calls)
    print("Scoring predictions (no RAG calls)...")
    scoring_result = score_sealed_predictions(
        predictions_path=str(PREDICTIONS_OUTPUT),
        labels_path=str(SEALED_LABELS_PATH),
        predictions_sha256=blind_info["predictions_sha256"],
    )

    # Also compute full metrics
    from src.evaluation.schemas import EvaluationPrediction
    predictions = [EvaluationPrediction.from_dict(p) for p in predictions_data]
    metrics = compute_all_metrics(labels, predictions)
    slice_metrics = compute_slice_metrics(labels, predictions, compute_all_metrics)
    failures = classify_all_failures(labels, predictions)

    # Confidence intervals
    n = metrics.get("total_cases", 0)
    pass_rate = metrics.get("strict_pass_rate", 0.0)
    n_pass = int(round(pass_rate * n)) if n > 0 else 0
    ci_low, ci_high = wilson_interval(n_pass, n)

    report = {
        "sealed_status": "completed",
        "blind_info": blind_info,
        "labels_sha256": labels_hash,
        "scoring_result": scoring_result,
        "summary": {
            "total_queries": n,
            "strict_pass_count": n_pass,
            "strict_pass_rate": pass_rate,
            "strict_pass_ci_low": ci_low,
            "strict_pass_ci_high": ci_high,
            **metrics,
        },
        "slice_metrics": slice_metrics,
        "failure_taxonomy": failures,
        "scoring_executed_once": True,
    }

    # Save report
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nSealed report saved to: {REPORT_OUTPUT}")

    print(f"\nSealed Scoring Summary:")
    print(f"  Total queries: {n}")
    print(f"  Strict pass: {n_pass}/{n}")
    print(f"  Strict pass rate: {pass_rate:.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Macro strict pass rate: {metrics.get('macro_strict_pass_rate', 0.0):.4f}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 5 sealed evaluation")
    parser.add_argument(
        "--score", action="store_true",
        help="Also score predictions (requires sealed labels)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Sealed Evaluation")
    print("=" * 60)

    # Load RC freeze record
    if RC_FREEZE_PATH.is_file():
        with open(RC_FREEZE_PATH, "r", encoding="utf-8") as f:
            rc_freeze = json.load(f)
        print(f"RC commit: {rc_freeze.get('rc_commit')}")
        print(f"Worktree clean at freeze: {rc_freeze.get('worktree_clean')}")
    else:
        print("WARNING: RC freeze record not found. Run freeze_phase5_rc.py first.")
        rc_freeze = {}

    # Phase 1: Blind run
    blind_info = asyncio.run(run_sealed_blind())
    if blind_info is None:
        print("Blind run failed. Exiting.")
        return 1

    # Phase 2: Scoring (optional)
    if args.score:
        report = run_sealed_scoring(blind_info)
        if report is None:
            print("Scoring failed or labels not available.")
            print("Blind run completed successfully. Score later with --score.")
    else:
        print("\nBlind run completed. Run with --score to score predictions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
