#!/usr/bin/env python3
"""Run the Phase 5 synthetic held-out blind prediction step (no scoring).

This script ONLY produces predictions. It:

- Enforces RC freeze pre-flight checks via :func:`verify_rc_freeze` (git
  HEAD, clean worktree, selected config, questions hash, protocol hash).
- Loads the winner calibration params from ``selected-config.json`` and
  truly injects them into the RAGEngine via
  :func:`build_evaluation_engine`.
- Runs a sentinel query to verify index wiring before producing
  predictions (fails fast if 0 results).
- Loads questions from ``eval_data/phase5/sealed/questions.jsonl`` (NO
  labels are ever accessed).
- Runs the RAG engine against each question via ``run_blind_queries()``.
- Writes predictions atomically to
  ``artifacts/evaluation/phase5/sealed-v2/predictions.jsonl``.
- Computes and writes both raw and canonical SHA256 hashes.
- Writes a ``predictions.jsonl.complete.json`` marker when done.
- Writes a run manifest + RC freeze report capturing the full state.

Note: This dataset is classified as ``synthetic_held_out``, not a true
sealed evaluation. See the protocol's ``dataset_classification`` field.

Usage:
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed_blind.py
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed_blind.py --dry-run
    HF_HUB_OFFLINE=1 python3 scripts/run_phase5_sealed_blind.py --skip-rc-freeze

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
from src.evaluation.engine_factory import build_evaluation_engine  # noqa: E402
from src.evaluation.manifests import (  # noqa: E402
    compute_file_sha256,
    compute_jsonl_sha256,
    create_manifest,
)
from src.evaluation.rc_freeze import (  # noqa: E402
    RCFreezeReport,
    RCFreezeViolation,
    verify_rc_freeze,
)

SEALED_QUESTIONS_PATH = (
    BACKEND_DIR / "eval_data" / "phase5" / "sealed" / "questions.jsonl"
)
PROTOCOL_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "protocol"
    / "phase5-evaluation-protocol.json"
)
SELECTED_CONFIG_PATH = (
    BACKEND_DIR
    / "artifacts"
    / "evaluation"
    / "phase5"
    / "calibration-v2"
    / "selected-config.json"
)
RC_FREEZE_MANIFEST_PATH = (
    BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "rc-freeze-manifest.json"
)
OUTPUT_DIR = BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "sealed-v2"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
PREDICTIONS_TMP_PATH = OUTPUT_DIR / "predictions.jsonl.tmp"
RAW_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.sha256"
CANONICAL_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.canonical.sha256"
COMPLETE_MARKER_PATH = OUTPUT_DIR / "predictions.jsonl.complete.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run-manifest.json"
RC_FREEZE_REPORT_PATH = OUTPUT_DIR / "rc-freeze-report.json"

MODEL_SERVER_ENDPOINT = "http://localhost:8500/v1"
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "finquery-finance-sft1147")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _load_rc_freeze_manifest() -> dict[str, Any]:
    """Load the RC freeze manifest if it exists (may contain expected hashes)."""
    if not RC_FREEZE_MANIFEST_PATH.is_file():
        return {}
    try:
        return json.loads(RC_FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_selected_config() -> dict[str, Any]:
    """Load the selected config (winner calibration params)."""
    if not SELECTED_CONFIG_PATH.is_file():
        print(f"WARNING: selected-config.json not found at {SELECTED_CONFIG_PATH}")
        print("  Running with default params (no calibration applied)")
        return {}
    try:
        return json.loads(SELECTED_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"WARNING: Failed to load selected-config.json: {exc}")
        return {}


def _run_rc_freeze_checks(skip: bool = False) -> RCFreezeReport | None:
    """Run RC freeze pre-flight checks.

    Returns the RCFreezeReport if checks were run, or None if skipped.
    If checks fail, RCFreezeViolation is raised (caller must handle).
    """
    if skip:
        print("RC freeze checks SKIPPED (--skip-rc-freeze)")
        return None

    print("Running RC freeze pre-flight checks...")
    rc_manifest = _load_rc_freeze_manifest()

    expected_commit = rc_manifest.get("expected_commit")
    expected_questions_sha = rc_manifest.get("questions_sha256")
    expected_protocol_sha = rc_manifest.get("protocol_sha256")

    try:
        report = verify_rc_freeze(
            BACKEND_DIR,
            expected_commit=expected_commit,
            selected_config_path=SELECTED_CONFIG_PATH,
            questions_path=SEALED_QUESTIONS_PATH,
            expected_questions_sha256=expected_questions_sha,
            protocol_path=PROTOCOL_PATH,
            expected_protocol_sha256=expected_protocol_sha,
            require_clean_worktree=True,
        )
        print(f"  RC freeze PASSED (git_head={report.git_head[:12]}...)")
        return report
    except RCFreezeViolation as exc:
        print("  RC freeze FAILED:")
        for v in exc.args[0].split("\n"):
            print(f"    {v}")
        raise


async def _run_blind(calibration_params: dict[str, Any]) -> dict[str, Any]:
    """Load questions, run the engine, and write predictions atomically."""
    print("Loading sealed questions (NO labels accessed)...")
    queries = load_queries(str(SEALED_QUESTIONS_PATH))
    print(f"Loaded {len(queries)} sealed questions")

    # Build engine via unified factory (handles partition index + sentinel
    # query + calibration param injection)
    try:
        from openai import OpenAI

        client = OpenAI(api_key="sk-placeholder", base_url=MODEL_SERVER_ENDPOINT)
        engine, engine_record = build_evaluation_engine(
            client,
            partition="sealed",
            calibration_params=calibration_params,
            model_name=MODEL_NAME,
            run_sentinel=True,
        )
        user_id = engine_record.partition_user_id
        n_results = engine_record.calibration.n_results or 3
        print(
            f"RAG engine initialized (user_id={user_id}, "
            f"sentinel_passed={engine_record.sentinel_query_passed}, "
            f"sentinel_count={engine_record.sentinel_query_result_count}, "
            f"n_results={n_results})"
        )
        print(
            f"Calibration params applied: "
            f"{engine_record.calibration.applied}"
        )
    except Exception as exc:
        print(f"Failed to initialize RAG engine: {exc}")
        raise

    print("Running blind evaluation on held-out set...")
    started_at = _now_iso()
    predictions = await run_blind_queries(
        queries, engine, user_id=user_id, n_results=n_results
    )
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
        "calibration_params_applied": engine_record.calibration.applied,
        "sentinel_query_passed": engine_record.sentinel_query_passed,
        "sentinel_query_result_count": engine_record.sentinel_query_result_count,
    }
    _write_text_atomic(
        COMPLETE_MARKER_PATH,
        json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    # Write the run manifest with all metadata.
    manifest = create_manifest(
        run_id=f"sealed-v2-{started_at}",
        run_type="sealed",
        n_results=n_results,
        case_count=len(predictions),
        run_started_at=started_at,
        run_ended_at=ended_at,
        questions_path=SEALED_QUESTIONS_PATH,
        predictions_path=PREDICTIONS_PATH,
        config_path=SELECTED_CONFIG_PATH,
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
    if SELECTED_CONFIG_PATH.is_file():
        print(f"OK: selected config found at {SELECTED_CONFIG_PATH}")
    else:
        print(f"WARNING: selected config not found at {SELECTED_CONFIG_PATH}")
    print("Dry run passed. No predictions were generated.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 synthetic held-out blind predictions (no scoring)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without running the RAG engine.",
    )
    parser.add_argument(
        "--skip-rc-freeze",
        action="store_true",
        help="Skip RC freeze pre-flight checks (NOT recommended for real runs).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Synthetic Held-out Blind Prediction (v2)")
    print("=" * 60)

    if args.dry_run:
        return _dry_run()

    # 1. RC freeze pre-flight (fail-closed)
    rc_report: RCFreezeReport | None = None
    try:
        rc_report = _run_rc_freeze_checks(skip=args.skip_rc_freeze)
    except RCFreezeViolation:
        print("\nRC freeze violation — aborting held-out run.")
        print("Fix the violations above or use --skip-rc-freeze for testing.")
        return 1

    # Save RC freeze report
    if rc_report is not None:
        _write_text_atomic(
            RC_FREEZE_REPORT_PATH,
            json.dumps(rc_report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        )
        print(f"RC freeze report saved to: {RC_FREEZE_REPORT_PATH}")

    # 2. Load selected config (winner calibration params)
    selected_config = _load_selected_config()
    calibration_params = selected_config.get("params", {})
    if calibration_params:
        print(f"Loaded calibration params from selected-config.json: {calibration_params}")
    else:
        print("No calibration params loaded — using engine defaults")

    # 3. Run blind prediction
    try:
        marker = asyncio.run(_run_blind(calibration_params))
    except Exception as exc:
        print(f"\nBlind prediction failed: {exc}")
        return 1

    print("\nBlind prediction completed successfully.")
    print(f"  Case count: {marker['case_count']}")
    print(f"  Raw SHA256: {marker['raw_sha256']}")
    print(f"  Canonical SHA256: {marker['canonical_sha256']}")
    print(f"  Sentinel passed: {marker.get('sentinel_query_passed', 'N/A')}")
    print("Score later with: python3 scripts/score_phase5_sealed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
