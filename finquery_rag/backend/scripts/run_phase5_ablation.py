#!/usr/bin/env python3
"""Run Phase 5 preregistered ablation study.

Runs all 10 ablation variants (A0-A9) from the pre-registered protocol.
Each variant disables exactly one component. The ablation is run on the
dev set using the same blind runner as the baseline.

Variant A0 (Full System) is the baseline. Variants A1-A7 are production-safe
(disabling a component). Variants A8-A9 disable validation and are NOT
production-safe.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_ablation.py

Environment:
    Model server must be running at http://localhost:8500.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.ablation import (  # noqa: E402
    ABLATION_VARIANTS,
    get_ablation_config,
    validate_ablation_config,
    ablation_report,
    is_production_safe,
)
from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.slices import compute_slice_metrics  # noqa: E402


OUTPUT_DIR = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "ablation"
)

# Base production config
BASE_CONFIG = {
    "use_hybrid": True,
    "enable_calculation_pipeline": True,
    "enable_validation_pipeline": True,
    "reranker_name": None,
    "max_context_tokens": 1100,
    "max_new_tokens": 512,
}


def build_engine_config(variant_id: str) -> dict:
    """Build RAGEngine kwargs for a given ablation variant."""
    variant_config = get_ablation_config(BASE_CONFIG, variant_id)
    engine_kwargs = {
        "model_name": "finquery-finance-sft1147",
        "use_hybrid": variant_config.get("use_hybrid", True),
        "enable_calculation_pipeline": variant_config.get("enable_calculation_pipeline", True),
        "enable_validation_pipeline": variant_config.get("enable_validation_pipeline", True),
        "max_context_tokens": variant_config.get("max_context_tokens", 1100),
        "max_new_tokens": variant_config.get("max_new_tokens", 512),
    }
    # Handle disable flags
    diff = variant_config
    if diff.get("disable_bm25"):
        engine_kwargs["use_hybrid"] = False
    if diff.get("disable-calculation-pipeline"):
        engine_kwargs["enable_calculation_pipeline"] = False
    if diff.get("disable-validation-pipeline"):
        engine_kwargs["enable_validation_pipeline"] = False
    return engine_kwargs


async def run_ablation_variant(variant_id: str, variant_name: str, queries, labels) -> dict:
    """Run a single ablation variant and return its metrics."""
    print(f"\n  Running variant {variant_id}: {variant_name}...")

    # Validate config
    variant = next(v for v in ABLATION_VARIANTS if v["id"] == variant_id)
    errors = validate_ablation_config(variant_id, variant["config_diff"])
    if errors:
        print(f"    Config validation errors: {errors}")
        return {"error": "config_validation_failed", "errors": errors}

    # Build engine config
    engine_kwargs = build_engine_config(variant_id)
    print(f"    Engine config: {engine_kwargs}")

    try:
        from openai import OpenAI
        from src.services.rag_engine import RAGEngine

        client = OpenAI(
            api_key="sk-placeholder",
            base_url="http://localhost:8500/v1"
        )
        engine = RAGEngine(llm_client=client, **engine_kwargs)
    except Exception as e:
        print(f"    Failed to initialize engine: {e}")
        return {"error": f"engine_init_failed: {e}"}

    # Run blind evaluation
    predictions = await run_blind_queries(
        queries, engine, user_id=1, n_results=3
    )
    print(f"    Generated {len(predictions)} predictions")

    # Compute metrics
    metrics = compute_all_metrics(labels, predictions)
    slice_m = compute_slice_metrics(labels, predictions, compute_all_metrics)

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "production_safe": is_production_safe(variant_id),
        "config_diff": variant["config_diff"],
        "metrics": metrics,
        "slice_metrics": slice_m,
    }


async def run_all_ablations():
    """Run all ablation variants on the dev set."""
    dev_questions = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "questions.jsonl"
    dev_labels = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "labels.jsonl"

    queries, labels = load_queries_and_labels(
        str(dev_questions), str(dev_labels)
    )
    print(f"Loaded {len(queries)} dev queries and {len(labels)} labels")
    print(f"Running {len(ABLATION_VARIANTS)} ablation variants...")

    results = {}
    for variant in ABLATION_VARIANTS:
        vid = variant["id"]
        vname = variant["name"]
        result = await run_ablation_variant(vid, vname, queries, labels)
        results[vid] = result

    return results


def main():
    print("=" * 60)
    print("Phase 5 Ablation Study")
    print("=" * 60)

    results = asyncio.run(run_all_ablations())

    # Build ablation report
    metrics_by_variant = {
        vid: r.get("metrics", {}) for vid, r in results.items()
        if "error" not in r
    }
    report = ablation_report(metrics_by_variant)

    # Save full results
    output = {
        "ablation_status": "completed",
        "total_variants": len(ABLATION_VARIANTS),
        "successful_variants": len(metrics_by_variant),
        "failed_variants": len(results) - len(metrics_by_variant),
        "report": report,
        "variants": {},
    }

    for vid, result in results.items():
        if "error" in result:
            output["variants"][vid] = {
                "variant_id": vid,
                "status": "failed",
                "error": result["error"],
            }
        else:
            m = result.get("metrics", {})
            output["variants"][vid] = {
                "variant_id": vid,
                "variant_name": result.get("variant_name", ""),
                "status": "completed",
                "production_safe": result.get("production_safe", True),
                "config_diff": result.get("config_diff", {}),
                "macro_strict_pass_rate": m.get("macro_strict_pass_rate", 0.0),
                "strict_pass_rate": m.get("strict_pass_rate", 0.0),
                "citation_recall": m.get("citation_recall", 0.0),
                "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                "false_block_rate": m.get("false_block_rate", 0.0),
                "unsafe_answer_rate": m.get("unsafe_answer_rate", 0.0),
                "total_cases": m.get("total_cases", 0),
            }

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ablation-report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nAblation report saved to: {output_path}")

    # Print summary
    print(f"\nAblation Summary:")
    print(f"  Total variants: {output['total_variants']}")
    print(f"  Successful: {output['successful_variants']}")
    print(f"  Failed: {output['failed_variants']}")
    print(f"\n  Variant Results:")
    for vid in sorted(output["variants"].keys()):
        v = output["variants"][vid]
        status = "OK" if v["status"] == "completed" else "FAIL"
        macro = v.get("macro_strict_pass_rate", 0.0)
        safe = v.get("production_safe", True)
        print(f"    {vid}: [{status}] macro_strict={macro:.4f} safe={safe}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
