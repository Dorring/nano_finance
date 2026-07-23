#!/usr/bin/env python3
"""Phase 5 v2 two-stage threshold calibration.

Implements the two-stage calibration protocol defined in
``artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json``:

Stage 1 — Replay Calibration (fast, no RAG engine):
    Load existing calibration predictions from a prior baseline run,
    apply each parameter combination via ``apply_params_to_prediction()``,
    compute metrics, and select the best safe candidate using the v2
    selection rule (safety release metrics as gate, false_block_rate as
    utility, safe=0 → baseline).

Stage 2 — End-to-End Rerun (verifies replay matches reality):
    Re-run the RAG engine on the calibration partition with the winning
    configuration from Stage 1. Compare the rerun macro_strict_pass_rate
    against the replay macro_strict_pass_rate. If the difference exceeds
    the parity threshold (default 0.05), flag as parity failure and fall
    back to baseline.

Usage::

    # Stage 1 only (no model server required)
    python scripts/run_phase5_calibration_v2.py --stage-1-only \\
        --predictions artifacts/evaluation/phase5/baseline/calibration-predictions.jsonl

    # Full two-stage (requires model server at localhost:8500)
    python scripts/run_phase5_calibration_v2.py --full \\
        --predictions artifacts/evaluation/phase5/baseline/calibration-predictions.jsonl

    # Stage 2 only (requires Stage 1 winner already saved)
    python scripts/run_phase5_calibration_v2.py --stage-2-only

Outputs (under ``artifacts/evaluation/phase5/calibration-v2/``):
    stage1-replay-report.json  — all candidates + winner
    stage2-rerun-report.json   — rerun metrics + parity check
    calibration-v2-report.json — merged final report
    selected-config.json       — the final selected configuration
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.calibration import (  # noqa: E402
    apply_params_to_prediction,
    eliminate_unsafe_candidates,
    search_calibration_space,
    select_best_candidate,
)
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.schemas import EvaluationPrediction  # noqa: E402
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
    / "calibration-v2"
)
CAL_QUESTIONS = (
    BACKEND_DIR / "eval_data" / "phase5" / "calibration" / "questions.jsonl"
)
CAL_LABELS = (
    BACKEND_DIR / "eval_data" / "phase5" / "calibration" / "labels.jsonl"
)

PARITY_THRESHOLD = 0.05
CALIBRATION_USER_ID = 9002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_protocol() -> dict[str, Any]:
    with open(PROTOCOL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline_metrics() -> dict[str, Any]:
    with open(BASELINE_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("summary", {})


def load_predictions(path: str | Path) -> list[EvaluationPrediction]:
    """Load predictions from a JSONL file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Predictions file not found: {p}")
    preds: list[EvaluationPrediction] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        preds.append(EvaluationPrediction.from_dict(json.loads(line)))
    return preds


def load_labels() -> tuple[list, list]:
    """Load calibration queries and labels."""
    return load_queries_and_labels(str(CAL_QUESTIONS), str(CAL_LABELS))


# ---------------------------------------------------------------------------
# Stage 1: Replay Calibration
# ---------------------------------------------------------------------------


def run_stage1_replay(
    predictions: list[EvaluationPrediction],
    labels: list,
    search_space: dict[str, list[Any]],
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Stage 1: Replay calibration on existing predictions.

    Applies each parameter combination to the predictions via
    ``apply_params_to_prediction()``, computes metrics, and selects the
    best safe candidate. No RAG engine is required.
    """
    print("[Stage 1] Replay calibration starting...")
    print(f"  Predictions: {len(predictions)}")
    print(f"  Labels: {len(labels)}")

    total_combos = 1
    for values in search_space.values():
        total_combos *= len(values)
    print(f"  Parameter combinations: {total_combos}")

    candidates = search_calibration_space(
        labels, predictions, search_space, baseline_metrics
    )
    print(f"  Evaluated {len(candidates)} candidates")

    safe_candidates = eliminate_unsafe_candidates(candidates, baseline_metrics)
    print(f"  Safe candidates: {len(safe_candidates)} / {len(candidates)}")

    winner = select_best_candidate(candidates, baseline_metrics)

    # Determine effective config (winner or baseline fallback)
    if winner.get("status") == "no_safe_candidate":
        effective_config = "baseline"
        effective_params: dict[str, Any] = {}
        tuning_applied = False
        print("  No safe candidate — selecting baseline (safe=0 rule)")
    else:
        effective_config = "stage1_winner"
        effective_params = winner.get("params", {})
        tuning_applied = True
        wm = winner.get("metrics", {})
        print(
            f"  Winner macro_strict_pass_rate: "
            f"{wm.get('macro_strict_pass_rate', 0.0):.4f}"
        )

    # Compute baseline replay metrics (params = no-op) for comparison
    baseline_replay_metrics = compute_all_metrics(labels, predictions)

    # Compute winner replay metrics
    if tuning_applied and effective_params:
        winner_preds = [
            apply_params_to_prediction(p, effective_params) for p in predictions
        ]
        winner_replay_metrics = compute_all_metrics(labels, winner_preds)
    else:
        winner_replay_metrics = baseline_replay_metrics

    n = winner_replay_metrics.get("total_cases", 0)
    pass_rate = winner_replay_metrics.get("strict_pass_rate", 0.0)
    n_pass = int(round(pass_rate * n)) if n > 0 else 0
    ci_low, ci_high = wilson_interval(n_pass, n)

    report = {
        "stage": 1,
        "stage_name": "replay_calibration",
        "requires_rag_engine": False,
        "total_combinations": total_combos,
        "total_candidates": len(candidates),
        "safe_candidates": len(safe_candidates),
        "baseline_macro_strict_pass_rate": baseline_metrics.get(
            "macro_strict_pass_rate", 0.0
        ),
        "baseline_replay_macro_strict_pass_rate": baseline_replay_metrics.get(
            "macro_strict_pass_rate", 0.0
        ),
        "winner": {
            "status": winner.get("status", "selected"),
            "params": effective_params,
            "metrics": winner_replay_metrics,
            "safe": winner.get("safe", False),
            "violations": winner.get("violations", []),
            "strict_pass_ci_low": ci_low,
            "strict_pass_ci_high": ci_high,
        },
        "effective_config": effective_config,
        "tuning_applied": tuning_applied,
        "all_candidates_summary": [
            {
                "params": c.get("params", {}),
                "macro_strict_pass_rate": c.get("metrics", {}).get(
                    "macro_strict_pass_rate", 0.0
                ),
                "strict_pass_rate": c.get("metrics", {}).get(
                    "strict_pass_rate", 0.0
                ),
                "safe": c.get("safe", False),
                "violations": c.get("violations", []),
            }
            for c in candidates
        ],
    }
    print(f"[Stage 1] Complete. Effective config: {effective_config}")
    return report


# ---------------------------------------------------------------------------
# Stage 2: End-to-End Rerun
# ---------------------------------------------------------------------------


def _build_engine_with_params(
    params: dict[str, Any],
) -> Any | None:
    """Build a RAG engine configured with the winner calibration params."""
    try:
        from openai import OpenAI

        from src.services.rag_engine import RAGEngine

        client = OpenAI(
            api_key="sk-placeholder",
            base_url="http://localhost:8500/v1",
        )
        n_results = int(params.get("n_results", 3))
        engine = RAGEngine(
            llm_client=client,
            model_name="finquery-finance-sft1147",
            n_results=n_results,
        )
        return engine
    except Exception as exc:  # noqa: BLE001
        print(f"  Failed to build RAG engine: {exc}")
        return None


async def run_stage2_rerun(
    winner_params: dict[str, Any],
    labels: list,
    queries: list,
    baseline_metrics: dict[str, Any],
    stage1_replay_macro: float,
    parity_threshold: float = PARITY_THRESHOLD,
) -> dict[str, Any]:
    """Stage 2: End-to-end rerun with winner config to verify parity.

    Re-runs the RAG engine on the calibration partition with the winning
    configuration from Stage 1. Compares the rerun macro_strict_pass_rate
    against the Stage 1 replay value. If the difference exceeds the
    parity threshold, flags as parity failure and recommends baseline.
    """
    print("[Stage 2] End-to-end rerun starting...")
    print(f"  Winner params: {winner_params}")

    if not winner_params:
        print("  No winner params — skipping rerun (baseline selected)")
        return {
            "stage": 2,
            "stage_name": "end_to_end_rerun",
            "requires_rag_engine": True,
            "rerun_status": "skipped_no_winner",
            "parity_check": "not_applicable",
            "rerun_metrics": {},
            "parity_passed": True,
            "recommendation": "baseline",
        }

    engine = _build_engine_with_params(winner_params)
    if engine is None:
        print("  RAG engine unavailable — skipping rerun")
        return {
            "stage": 2,
            "stage_name": "end_to_end_rerun",
            "requires_rag_engine": True,
            "rerun_status": "engine_unavailable",
            "parity_check": "not_applicable",
            "rerun_metrics": {},
            "parity_passed": False,
            "recommendation": "baseline",
            "note": "RAG engine could not be initialized; fall back to baseline",
        }

    from src.evaluation.blind_runner import run_blind_queries

    n_results = int(winner_params.get("n_results", 3))
    print(f"  Running {len(queries)} calibration queries with n_results={n_results}...")

    # Set the partition index env vars for the calibration partition
    cal_index_dir = BACKEND_DIR / "indexes" / "phase5" / "calibration"
    os.environ["CHROMA_PATH"] = str(cal_index_dir / "chroma")
    os.environ["BM25_DB_PATH"] = str(cal_index_dir / "rag_bm25.db")

    # Reset ChromaDB client singleton
    import src.services.vector_store as vs

    vs._chroma_client = None

    predictions = await run_blind_queries(
        queries, engine, user_id=CALIBRATION_USER_ID, n_results=n_results
    )
    print(f"  Generated {len(predictions)} rerun predictions")

    rerun_metrics = compute_all_metrics(labels, predictions)

    rerun_macro = rerun_metrics.get("macro_strict_pass_rate", 0.0)
    macro_diff = abs(rerun_macro - stage1_replay_macro)
    parity_passed = macro_diff <= parity_threshold

    print(f"  Replay macro: {stage1_replay_macro:.4f}")
    print(f"  Rerun macro:  {rerun_macro:.4f}")
    print(f"  Diff:         {macro_diff:.4f} (threshold: {parity_threshold})")

    recommendation = "stage1_winner" if parity_passed else "baseline"

    return {
        "stage": 2,
        "stage_name": "end_to_end_rerun",
        "requires_rag_engine": True,
        "rerun_status": "completed",
        "parity_check": {
            "replay_macro_strict_pass_rate": stage1_replay_macro,
            "rerun_macro_strict_pass_rate": rerun_macro,
            "absolute_difference": macro_diff,
            "threshold": parity_threshold,
            "passed": parity_passed,
        },
        "rerun_metrics": rerun_metrics,
        "parity_passed": parity_passed,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def save_report(data: dict[str, Any], filename: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"  Saved: {path}")
    return path


def build_final_report(
    stage1: dict[str, Any],
    stage2: dict[str, Any] | None,
    baseline_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Merge Stage 1 and Stage 2 into the final calibration report."""
    winner = stage1.get("winner", {})
    stage1_macro = winner.get("metrics", {}).get("macro_strict_pass_rate", 0.0)

    if stage2 is None:
        final_config = stage1.get("effective_config", "baseline")
        final_params = winner.get("params", {})
        final_status = "stage1_only"
        parity = None
    else:
        parity = stage2.get("parity_check")
        if stage2.get("rerun_status") == "completed" and not stage2.get(
            "parity_passed", False
        ):
            final_config = "baseline"
            final_params = {}
            final_status = "parity_failed_baseline_fallback"
        elif stage2.get("rerun_status") in (
            "skipped_no_winner",
            "engine_unavailable",
        ):
            final_config = stage2.get("recommendation", "baseline")
            final_params = winner.get("params", {}) if final_config != "baseline" else {}
            final_status = stage2.get("rerun_status", "unknown")
        else:
            final_config = "stage1_winner"
            final_params = winner.get("params", {})
            final_status = "confirmed"

    return {
        "calibration_version": "v2",
        "protocol_version": "2.0",
        "baseline_macro_strict_pass_rate": baseline_metrics.get(
            "macro_strict_pass_rate", 0.0
        ),
        "stage1_replay_macro_strict_pass_rate": stage1_macro,
        "stage1": stage1,
        "stage2": stage2,
        "final_config": final_config,
        "final_params": final_params,
        "final_status": final_status,
        "parity_check": parity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 v2 two-stage threshold calibration"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--stage-1-only",
        action="store_true",
        help="Run only Stage 1 (replay, no RAG engine required)",
    )
    mode.add_argument(
        "--stage-2-only",
        action="store_true",
        help="Run only Stage 2 (rerun, requires Stage 1 winner saved)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        default=True,
        help="Run both stages (default)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="Path to calibration predictions JSONL (Stage 1 input)",
    )
    parser.add_argument(
        "--parity-threshold",
        type=float,
        default=PARITY_THRESHOLD,
        help=f"Parity threshold for Stage 2 (default: {PARITY_THRESHOLD})",
    )
    args = parser.parse_args()

    parity_threshold = args.parity_threshold

    print("=" * 60)
    print("Phase 5 v2 Two-Stage Calibration")
    print("=" * 60)

    protocol = load_protocol()
    search_space = protocol.get("calibration_search_space", {})
    baseline_metrics = load_baseline_metrics()

    print(f"\nBaseline macro_strict_pass_rate: "
          f"{baseline_metrics.get('macro_strict_pass_rate', 0.0):.4f}")
    print(f"Search space: {len(search_space)} parameters")

    # Load calibration queries and labels
    queries, labels = load_labels()
    print(f"Loaded {len(queries)} calibration queries, {len(labels)} labels")

    stage2_report: dict[str, Any] | None = None

    # ---- Stage 1 ----
    if not args.stage_2_only:
        if not args.predictions:
            print("\nERROR: --predictions is required for Stage 1")
            print("Run the baseline script first to generate calibration predictions,")
            print("or use --stage-2-only if Stage 1 winner is already saved.")
            return 1

        predictions = load_predictions(args.predictions)
        stage1_report = run_stage1_replay(
            predictions, labels, search_space, baseline_metrics
        )
        save_report(stage1_report, "stage1-replay-report.json")

        winner_params = stage1_report.get("winner", {}).get("params", {})
    else:
        # Load Stage 1 winner from saved report
        stage1_path = OUTPUT_DIR / "stage1-replay-report.json"
        if not stage1_path.is_file():
            print(f"\nERROR: Stage 1 report not found at {stage1_path}")
            print("Run Stage 1 first.")
            return 1
        with open(stage1_path, "r", encoding="utf-8") as f:
            stage1_report = json.load(f)
        winner_params = stage1_report.get("winner", {}).get("params", {})
        print(f"\nLoaded Stage 1 winner params: {winner_params}")

    # ---- Stage 2 ----
    if not args.stage_1_only:
        stage1_macro = stage1_report.get("winner", {}).get("metrics", {}).get(
            "macro_strict_pass_rate", 0.0
        )
        stage2_report = asyncio.run(
            run_stage2_rerun(
                winner_params,
                labels,
                queries,
                baseline_metrics,
                stage1_macro,
                parity_threshold=parity_threshold,
            )
        )
        save_report(stage2_report, "stage2-rerun-report.json")

    # ---- Final merged report ----
    final_report = build_final_report(stage1_report, stage2_report, baseline_metrics)
    save_report(final_report, "calibration-v2-report.json")

    # Save selected config
    selected_config = {
        "config": final_report["final_config"],
        "params": final_report["final_params"],
        "status": final_report["final_status"],
        "source": "calibration_v2_two_stage",
    }
    save_report(selected_config, "selected-config.json")

    print("\n" + "=" * 60)
    print("Calibration v2 Summary:")
    print(f"  Final config: {final_report['final_config']}")
    print(f"  Final status: {final_report['final_status']}")
    print(f"  Final params: {final_report['final_params']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
