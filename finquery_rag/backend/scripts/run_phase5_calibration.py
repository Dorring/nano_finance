#!/usr/bin/env python3
"""Run Phase 5 threshold calibration search on the calibration set.

This script:
1. Loads calibration queries and labels.
2. Initializes the RAG engine and runs blind evaluation on the calibration set.
3. Loads the baseline metrics from the baseline report.
4. Searches the pre-registered calibration parameter space.
5. Applies the protocol's selection rule to pick the best candidate.
6. Saves all candidates and the winner.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_calibration.py

Environment:
    Model server must be running at http://localhost:8500.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.calibration import (  # noqa: E402
    search_calibration_space,
    select_best_candidate,
    eliminate_unsafe_candidates,
)
from src.evaluation.statistics import wilson_interval  # noqa: E402


PROTOCOL_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "protocol"
    / "phase5-evaluation-protocol.json"
)
BASELINE_REPORT_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "baseline"
    / "baseline-report.json"
)
OUTPUT_DIR = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "calibration"
)


def load_protocol() -> dict:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_metrics() -> dict:
    with open(BASELINE_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("summary", {})


async def run_calibration_evaluation():
    """Run calibration evaluation and search."""
    cal_questions = BACKEND_DIR / "eval_data" / "phase5" / "calibration" / "questions.jsonl"
    cal_labels = BACKEND_DIR / "eval_data" / "phase5" / "calibration" / "labels.jsonl"

    queries, labels = load_queries_and_labels(
        str(cal_questions), str(cal_labels)
    )
    print(f"Loaded {len(queries)} calibration queries and {len(labels)} labels")

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

    # Run blind evaluation on calibration set
    print("Running blind evaluation on calibration set...")
    predictions = await run_blind_queries(
        queries, engine, user_id=1, n_results=3
    )
    print(f"Generated {len(predictions)} predictions")

    # Load baseline metrics
    baseline_metrics = load_baseline_metrics()
    print(f"Loaded baseline metrics ({len(baseline_metrics)} keys)")

    # Load search space from protocol
    protocol = load_protocol()
    search_space = protocol.get("calibration_search_space", {})
    print(f"Search space: {len(search_space)} parameters")

    # Compute total combinations
    total_combos = 1
    for k, v in search_space.items():
        total_combos *= len(v)
    print(f"Total parameter combinations: {total_combos}")

    # Search calibration space
    print("Searching calibration space...")
    candidates = search_calibration_space(
        labels, predictions, search_space, baseline_metrics
    )
    print(f"Evaluated {len(candidates)} candidates")

    safe_candidates = eliminate_unsafe_candidates(candidates, baseline_metrics)
    print(f"Safe candidates: {len(safe_candidates)} / {len(candidates)}")

    # Select best candidate
    winner = select_best_candidate(candidates, baseline_metrics)
    print(f"Winner params: {winner.get('params', {})}")
    winner_metrics = winner.get("metrics", {})
    print(f"Winner macro_strict_pass_rate: {winner_metrics.get('macro_strict_pass_rate', 0.0):.4f}")

    # Compute confidence intervals for winner
    n = winner_metrics.get("total_cases", 0)
    pass_rate = winner_metrics.get("strict_pass_rate", 0.0)
    n_pass = int(round(pass_rate * n)) if n > 0 else 0
    ci_low, ci_high = wilson_interval(n_pass, n)

    result = {
        "search_space": search_space,
        "total_combinations": total_combos,
        "total_candidates": len(candidates),
        "safe_candidates": len(safe_candidates),
        "baseline_macro_strict_pass_rate": baseline_metrics.get("macro_strict_pass_rate", 0.0),
        "winner": {
            "params": winner.get("params", {}),
            "metrics": winner_metrics,
            "safe": winner.get("safe", False),
            "violations": winner.get("violations", []),
            "strict_pass_ci_low": ci_low,
            "strict_pass_ci_high": ci_high,
        },
        "all_candidates_summary": [
            {
                "params": c.get("params", {}),
                "macro_strict_pass_rate": c.get("metrics", {}).get("macro_strict_pass_rate", 0.0),
                "strict_pass_rate": c.get("metrics", {}).get("strict_pass_rate", 0.0),
                "safe": c.get("safe", False),
                "violations": c.get("violations", []),
            }
            for c in candidates
        ],
    }

    return result


def main():
    print("=" * 60)
    print("Phase 5 Threshold Calibration Search")
    print("=" * 60)

    result = asyncio.run(run_calibration_evaluation())

    if result is None:
        print("Calibration evaluation failed.")
        output = {"calibration_status": "failed"}
    else:
        output = result
        output["calibration_status"] = "completed"

    # Save calibration report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "calibration-report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nCalibration report saved to: {output_path}")

    if output.get("calibration_status") == "completed":
        print("\nCalibration Summary:")
        print(f"  Total combinations: {output.get('total_combinations', 0)}")
        print(f"  Total candidates: {output.get('total_candidates', 0)}")
        print(f"  Safe candidates: {output.get('safe_candidates', 0)}")
        winner = output.get("winner", {})
        print(f"  Winner params: {winner.get('params', {})}")
        wm = winner.get("metrics", {})
        print(f"  Winner strict_pass_rate: {wm.get('strict_pass_rate', 0.0):.4f}")
        print(f"  Winner macro_strict_pass_rate: {wm.get('macro_strict_pass_rate', 0.0):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
