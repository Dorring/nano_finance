#!/usr/bin/env python3
"""Run Phase 5 baseline evaluation on the dev set.

This script:
1. Computes SHA256 hashes of the model checkpoint, tokenizer, corpus/index, and config.
2. Initializes the RAG engine with the running model server.
3. Runs blind evaluation on the dev set.
4. Computes metrics and saves the baseline report.

Usage:
    python3 scripts/run_phase5_baseline.py

Environment:
    Model server must be running at http://localhost:8500 with model name
    "finquery-finance-sft1147".
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

# Ensure src is importable
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

# Phase 5 imports
from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.slices import compute_slice_metrics  # noqa: E402
from src.evaluation.failure_taxonomy import classify_all_failures  # noqa: E402
from src.evaluation.statistics import wilson_interval  # noqa: E402


def compute_sha256(filepath: str | Path) -> str | None:
    """Compute SHA256 of a file. Returns None if file doesn't exist."""
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dir_sha256(dirpath: str | Path) -> str | None:
    """Compute a deterministic hash of a directory's contents."""
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


def compute_hashes() -> dict:
    """Compute hashes of all resources for the baseline manifest."""
    hashes = {}

    # Model checkpoint - look for the SFT checkpoint
    sft_checkpoint_dirs = [
        "/mnt/disk/mxf/.cache/nanochat/chatsft_checkpoints/d24_final_mixdata",
        "/mnt/disk/mxf/.cache/nanochat/chatsft_checkpoints/d24_finance_v2_lr010",
        "/mnt/disk/mxf/.cache/nanochat/chatsft_checkpoints/d24_finance_v2_lr005",
    ]
    model_checkpoint_path = None
    model_checkpoint_hash = None
    for d in sft_checkpoint_dirs:
        p = Path(d)
        if p.is_dir():
            # Find the best checkpoint (from best.json or highest step)
            best_file = p / "best.json"
            if best_file.is_file():
                try:
                    best = json.loads(best_file.read_text())
                    step = best.get("step")
                    ckpt = best.get("checkpoint", f"model_{step:06d}.pt")
                    ckpt_path = p / ckpt
                    if ckpt_path.is_file():
                        model_checkpoint_path = str(ckpt_path)
                        model_checkpoint_hash = compute_sha256(ckpt_path)
                        break
                except (json.JSONDecodeError, KeyError):
                    pass
            # Fallback: use directory hash
            model_checkpoint_path = str(p)
            model_checkpoint_hash = compute_dir_sha256(p)
            break

    hashes["model_checkpoint_path"] = model_checkpoint_path
    hashes["model_checkpoint_sha256"] = model_checkpoint_hash
    hashes["model_server_endpoint"] = "http://localhost:8500"
    hashes["model_server_name"] = "finquery-finance-sft1147"
    hashes["model_tag"] = "d24_final_mixdata"
    hashes["model_step"] = 1147

    # Tokenizer
    tokenizer_path = "/mnt/disk/mxf/.cache/nanochat/tokenizer/tokenizer.pkl"
    tokenizer_hash = compute_sha256(tokenizer_path)
    if tokenizer_hash is None:
        tokenizer_path = "/mnt/disk/mxf/.cache/nanochat/tokenizer"
        tokenizer_hash = compute_dir_sha256(tokenizer_path)
    hashes["tokenizer_path"] = tokenizer_path
    hashes["tokenizer_sha256"] = tokenizer_hash

    # Corpus/Index - ChromaDB
    chroma_path = str(BACKEND_DIR / "chroma_db")
    hashes["chroma_db_path"] = chroma_path
    hashes["chroma_db_sha256"] = compute_dir_sha256(chroma_path)

    # BM25
    bm25_path = str(BACKEND_DIR / "rag_bm25.db")
    hashes["bm25_db_path"] = bm25_path
    hashes["bm25_db_sha256"] = compute_sha256(bm25_path)

    # Config
    config_path = str(BACKEND_DIR / "pyproject.toml")
    hashes["config_path"] = config_path
    hashes["config_sha256"] = compute_sha256(config_path)

    # Baseline commit
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(BACKEND_DIR)
        )
        hashes["baseline_commit"] = result.stdout.strip()
    except Exception:
        hashes["baseline_commit"] = None

    return hashes


async def run_baseline_evaluation():
    """Run the baseline evaluation on the dev set."""
    dev_questions = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "questions.jsonl"
    dev_labels = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "labels.jsonl"

    queries, labels = load_queries_and_labels(
        str(dev_questions), str(dev_labels)
    )
    print(f"Loaded {len(queries)} dev queries and {len(labels)} labels")

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
    print("Running blind evaluation on dev set...")
    predictions = await run_blind_queries(
        queries, engine, user_id=1, n_results=3
    )
    print(f"Generated {len(predictions)} predictions")

    # Compute metrics
    metrics = compute_all_metrics(labels, predictions)
    slice_metrics = compute_slice_metrics(labels, predictions, compute_all_metrics)
    failures = classify_all_failures(labels, predictions)

    # Compute confidence intervals from aggregate metrics
    n = metrics.get("total_cases", 0)
    pass_rate = metrics.get("strict_pass_rate", 0.0)
    n_pass = int(round(pass_rate * n)) if n > 0 else 0
    ci_low, ci_high = wilson_interval(n_pass, n)

    result = {
        "manifest": compute_hashes(),
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
    }

    return result


def main():
    print("=" * 60)
    print("Phase 5 Baseline Evaluation")
    print("=" * 60)

    # Compute hashes
    print("\n1. Computing resource hashes...")
    hashes = compute_hashes()
    for k, v in hashes.items():
        print(f"  {k}: {v}")

    # Run baseline evaluation
    print("\n2. Running baseline evaluation...")
    result = asyncio.run(run_baseline_evaluation())

    if result is None:
        print("Baseline evaluation failed - RAG engine could not be initialized.")
        print("Recording manifest only.")
        output = {"manifest": hashes, "evaluation_status": "failed"}
    else:
        output = result
        output["evaluation_status"] = "completed"

    # Save baseline report
    output_dir = BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline-report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nBaseline report saved to: {output_path}")

    # Print summary
    if output.get("evaluation_status") == "completed":
        summary = output.get("summary", {})
        print("\nBaseline Summary:")
        print(f"  Total queries: {summary.get('total_queries', 0)}")
        print(f"  Strict pass: {summary.get('strict_pass_count', 0)}/{summary.get('total_queries', 0)}")
        print(f"  Strict pass rate: {summary.get('strict_pass_rate', 0):.4f}")
        print(f"  95% CI: [{summary.get('strict_pass_ci_low', 0):.4f}, {summary.get('strict_pass_ci_high', 0):.4f}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
