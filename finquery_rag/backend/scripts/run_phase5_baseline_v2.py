#!/usr/bin/env python3
"""Phase 5 v2 baseline evaluation with partition-specific indexes.

Runs blind evaluation on a specified partition (dev, calibration, or
sealed) using the Phase 5 v2 per-partition ChromaDB and BM25 indexes
built by ``scripts/build_phase5_eval_index.py``.

Outputs:
- ``{partition}-report.json``  — metrics, slices, failure taxonomy
- ``{partition}-predictions.jsonl`` — raw predictions (for calibration
  Stage 1 replay input when partition=calibration)

Usage::

    # Dev partition (baseline evaluation)
    HF_HUB_OFFLINE=1 python scripts/run_phase5_baseline_v2.py --partition dev

    # Calibration partition (generates predictions for Stage 1)
    HF_HUB_OFFLINE=1 python scripts/run_phase5_baseline_v2.py --partition calibration

Environment:
    Model server must be running at http://localhost:8500.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.slices import compute_slice_metrics  # noqa: E402
from src.evaluation.failure_taxonomy import classify_all_failures  # noqa: E402
from src.evaluation.statistics import wilson_interval  # noqa: E402

PARTITION_USER_IDS: dict[str, int] = {
    "dev": 9001,
    "calibration": 9002,
    "sealed": 9003,
}

OUTPUT_BASE = BACKEND_DIR / "artifacts" / "evaluation" / "phase5"


def compute_sha256(filepath: str | Path) -> str | None:
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_sha256(dirpath: str | Path) -> str | None:
    p = Path(dirpath)
    if not p.is_dir():
        return None
    h = hashlib.sha256()
    files = sorted(p.rglob("*"))
    for fp in files:
        if fp.is_file():
            rel = str(fp.relative_to(p))
            h.update(rel.encode())
            h.update(b"\0")
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            h.update(b"\0")
    return h.hexdigest()


def compute_manifest(partition: str) -> dict:
    """Compute resource hashes for the baseline manifest."""
    index_dir = BACKEND_DIR / "indexes" / "phase5" / partition
    hashes: dict = {
        "partition": partition,
        "user_id": PARTITION_USER_IDS[partition],
        "chroma_db_path": str(index_dir / "chroma"),
        "chroma_db_sha256": compute_dir_sha256(index_dir / "chroma"),
        "bm25_db_path": str(index_dir / "rag_bm25.db"),
        "bm25_db_sha256": compute_sha256(index_dir / "rag_bm25.db"),
        "chunk_manifest_sha256": compute_sha256(index_dir / "chunk-manifest.json"),
    }

    # Model server info
    hashes["model_server_endpoint"] = "http://localhost:8500"
    hashes["model_server_name"] = "finquery-finance-sft1147"

    # Git commit
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(BACKEND_DIR),
        )
        hashes["baseline_commit"] = result.stdout.strip()
    except Exception:
        hashes["baseline_commit"] = None

    return hashes


def setup_partition_index(partition: str) -> None:
    """Set environment variables for the partition-specific index."""
    index_dir = BACKEND_DIR / "indexes" / "phase5" / partition
    chroma_path = index_dir / "chroma"
    bm25_path = str(index_dir / "rag_bm25.db")

    os.environ["CHROMA_PATH"] = str(chroma_path)
    os.environ["BM25_DB_PATH"] = bm25_path

    # Reset ChromaDB client singleton
    import src.services.vector_store as vs

    vs._chroma_client = None


async def run_partition_evaluation(partition: str) -> dict | None:
    """Run blind evaluation on the specified partition."""
    questions_path = (
        BACKEND_DIR / "eval_data" / "phase5" / partition / "questions.jsonl"
    )
    labels_path = BACKEND_DIR / "eval_data" / "phase5" / partition / "labels.jsonl"

    queries, labels = load_queries_and_labels(str(questions_path), str(labels_path))
    print(f"Loaded {len(queries)} {partition} queries, {len(labels)} labels")

    # Set up partition-specific index
    setup_partition_index(partition)
    user_id = PARTITION_USER_IDS[partition]

    # Initialize RAG engine
    try:
        from openai import OpenAI

        from src.services.rag_engine import RAGEngine

        client = OpenAI(
            api_key="sk-placeholder",
            base_url="http://localhost:8500/v1",
        )
        engine = RAGEngine(
            llm_client=client,
            model_name="finquery-finance-sft1147",
        )
        print(f"RAG engine initialized (user_id={user_id})")
    except Exception as exc:
        print(f"Failed to initialize RAG engine: {exc}")
        return None

    # Run blind evaluation
    print(f"Running blind evaluation on {partition} set...")
    predictions = await run_blind_queries(queries, engine, user_id=user_id, n_results=3)
    print(f"Generated {len(predictions)} predictions")

    # Save predictions JSONL (needed for calibration Stage 1)
    output_dir = OUTPUT_BASE / partition
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / f"{partition}-predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {preds_path}")

    # Compute metrics
    metrics = compute_all_metrics(labels, predictions)
    slice_metrics = compute_slice_metrics(labels, predictions, compute_all_metrics)
    failures = classify_all_failures(labels, predictions)

    n = metrics.get("total_cases", 0)
    pass_rate = metrics.get("strict_pass_rate", 0.0)
    n_pass = int(round(pass_rate * n)) if n > 0 else 0
    ci_low, ci_high = wilson_interval(n_pass, n)

    result = {
        "manifest": compute_manifest(partition),
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
        "predictions_path": str(preds_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 v2 baseline evaluation with partition indexes"
    )
    parser.add_argument(
        "--partition",
        choices=["dev", "calibration", "sealed"],
        default="dev",
        help="Partition to evaluate (default: dev)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Phase 5 v2 Baseline Evaluation — {args.partition}")
    print("=" * 60)

    result = asyncio.run(run_partition_evaluation(args.partition))

    if result is None:
        print("Evaluation failed — RAG engine could not be initialized.")
        output = {
            "manifest": compute_manifest(args.partition),
            "evaluation_status": "failed",
        }
    else:
        output = result
        output["evaluation_status"] = "completed"

    # Save report
    output_dir = OUTPUT_BASE / args.partition
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.partition}-report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nReport saved to: {report_path}")

    if output.get("evaluation_status") == "completed":
        summary = output.get("summary", {})
        print(f"\n{args.partition.title()} Summary:")
        print(f"  Total queries: {summary.get('total_queries', 0)}")
        print(
            f"  Strict pass: "
            f"{summary.get('strict_pass_count', 0)}/"
            f"{summary.get('total_queries', 0)}"
        )
        print(f"  Strict pass rate: {summary.get('strict_pass_rate', 0):.4f}")
        print(
            f"  95% CI: [{summary.get('strict_pass_ci_low', 0):.4f}, "
            f"{summary.get('strict_pass_ci_high', 0):.4f}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
