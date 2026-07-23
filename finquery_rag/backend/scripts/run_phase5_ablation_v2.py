#!/usr/bin/env python3
"""Phase 5 v2 ablation study with partition-specific indexes.

Runs all 10 ablation variants (A0-A9) from the pre-registered protocol
on the dev partition using the Phase 5 v2 per-partition ChromaDB and
BM25 indexes (user_id=9001).

Each variant uses :func:`get_variant_feature_flags` to obtain an
:class:`EvaluationFeatureFlags` object, which is then injected into the
RAGEngine via :func:`build_evaluation_engine`. All 9 feature flags
truly reach the engine components (constructor kwargs + runtime patches),
and the application is verified by :func:`assert_feature_flags_enforced`.

Variant A0 (Full System) is the baseline. Variants A1-A7 are
production-safe. Variants A8-A9 disable validation and are NOT
production-safe.

Usage::

    HF_HUB_OFFLINE=1 python scripts/run_phase5_ablation_v2.py

Environment:
    Model server must be running at http://localhost:8500.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.ablation import (  # noqa: E402
    ABLATION_VARIANTS,
    ablation_report,
    get_variant_feature_flags,
    is_production_safe,
    validate_ablation_config,
)
from src.evaluation.blind_runner import run_blind_queries  # noqa: E402
from src.evaluation.dataset_loader import load_queries_and_labels  # noqa: E402
from src.evaluation.engine_factory import build_evaluation_engine  # noqa: E402
from src.evaluation.feature_flag_injection import (  # noqa: E402
    assert_feature_flags_enforced,
)
from src.evaluation.metrics import compute_all_metrics  # noqa: E402
from src.evaluation.slices import compute_slice_metrics  # noqa: E402

OUTPUT_DIR = BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "ablation-v2"


async def run_ablation_variant(
    variant_id: str, variant_name: str, queries, labels
) -> dict:
    """Run a single ablation variant and return its metrics."""
    print(f"\n  Running variant {variant_id}: {variant_name}...")

    variant = next(v for v in ABLATION_VARIANTS if v["id"] == variant_id)
    errors = validate_ablation_config(variant_id, variant["config_diff"])
    if errors:
        print(f"    Config validation errors: {errors}")
        return {"error": "config_validation_failed", "errors": errors}

    flags = get_variant_feature_flags(variant_id)
    print(f"    Feature flags: {flags.to_dict()}")

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key="sk-placeholder",
            base_url="http://localhost:8500/v1",
        )
        engine, engine_record = build_evaluation_engine(
            client,
            partition="dev",
            feature_flags=flags,
            model_name=os.getenv("LLM_MODEL_NAME", "finquery-finance-sft1147"),
            run_sentinel=True,
        )
    except Exception as exc:
        print(f"    Failed to initialize engine: {exc}")
        return {"error": f"engine_init_failed: {exc}"}

    # Verify all 9 flags are actually enforced on the engine
    violations = assert_feature_flags_enforced(flags, engine)
    if violations:
        print(f"    Feature flag enforcement violations: {violations}")
        return {
            "error": "feature_flag_enforcement_failed",
            "violations": violations,
        }
    print("    All 9 feature flags verified enforced")

    user_id = engine_record.partition_user_id
    n_results = engine_record.calibration.n_results or 3

    predictions = await run_blind_queries(
        queries, engine, user_id=user_id, n_results=n_results
    )
    print(f"    Generated {len(predictions)} predictions")

    metrics = compute_all_metrics(labels, predictions)
    slice_m = compute_slice_metrics(labels, predictions, compute_all_metrics)

    return {
        "variant_id": variant_id,
        "variant_name": variant_name,
        "production_safe": is_production_safe(variant_id),
        "config_diff": variant["config_diff"],
        "feature_flags": flags.to_dict(),
        "feature_flag_application": {
            "constructor_kwargs": engine_record.feature_flags.constructor_kwargs,
            "runtime_patches": engine_record.feature_flags.runtime_patches,
            "noops": engine_record.feature_flags.noops,
        },
        "sentinel_query_passed": engine_record.sentinel_query_passed,
        "sentinel_query_result_count": engine_record.sentinel_query_result_count,
        "metrics": metrics,
        "slice_metrics": slice_m,
    }


async def run_all_ablations() -> dict:
    """Run all ablation variants on the dev set."""
    dev_questions = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "questions.jsonl"
    dev_labels = BACKEND_DIR / "eval_data" / "phase5" / "dev" / "labels.jsonl"

    queries, labels = load_queries_and_labels(str(dev_questions), str(dev_labels))
    print(f"Loaded {len(queries)} dev queries and {len(labels)} labels")
    print(f"Running {len(ABLATION_VARIANTS)} ablation variants...")

    results: dict[str, dict] = {}
    for variant in ABLATION_VARIANTS:
        vid = variant["id"]
        vname = variant["name"]
        result = await run_ablation_variant(vid, vname, queries, labels)
        results[vid] = result

    return results


def main() -> int:
    print("=" * 60)
    print("Phase 5 v2 Ablation Study (partition indexes + feature flags)")
    print("=" * 60)

    results = asyncio.run(run_all_ablations())

    metrics_by_variant = {
        vid: r.get("metrics", {}) for vid, r in results.items() if "error" not in r
    }
    report = ablation_report(metrics_by_variant)

    output = {
        "ablation_version": "v2",
        "ablation_status": "completed",
        "feature_flag_injection": "all_9_flags_reach_components_via_engine_factory",
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
                "feature_flags": result.get("feature_flags", {}),
                "feature_flag_application": result.get(
                    "feature_flag_application", {}
                ),
                "sentinel_query_passed": result.get("sentinel_query_passed", False),
                "sentinel_query_result_count": result.get(
                    "sentinel_query_result_count", 0
                ),
                "macro_strict_pass_rate": m.get("macro_strict_pass_rate", 0.0),
                "strict_pass_rate": m.get("strict_pass_rate", 0.0),
                "citation_recall": m.get("citation_recall", 0.0),
                "p95_latency_ms": m.get("p95_latency_ms", 0.0),
                "false_block_rate": m.get("false_block_rate", 0.0),
                "unsupported_numeric_release_rate": m.get(
                    "unsupported_numeric_release_rate", 0.0
                ),
                "calculation_mismatch_release_rate": m.get(
                    "calculation_mismatch_release_rate", 0.0
                ),
                "invalid_citation_release_rate": m.get(
                    "invalid_citation_release_rate", 0.0
                ),
                "unsafe_content_release_rate": m.get(
                    "unsafe_content_release_rate", 0.0
                ),
                "total_cases": m.get("total_cases", 0),
            }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "ablation-v2-report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    print(f"\nAblation report saved to: {output_path}")

    print("\nAblation Summary:")
    print(f"  Total variants: {output['total_variants']}")
    print(f"  Successful: {output['successful_variants']}")
    print(f"  Failed: {output['failed_variants']}")
    for vid in sorted(output["variants"]):
        v = output["variants"][vid]
        status = v.get("status", "unknown")
        macro = v.get("macro_strict_pass_rate", 0.0)
        print(f"  {vid}: {status} macro={macro:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
