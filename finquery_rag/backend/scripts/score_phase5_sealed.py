#!/usr/bin/env python3
"""Score Phase 5 sealed predictions (no RAG engine calls).

This script ONLY scores predictions produced by
``run_phase5_sealed_blind.py``. It:

- Verifies the predictions file exists and has a ``.complete.json`` marker.
- Verifies the raw SHA256 matches ``predictions.jsonl.sha256``.
- Verifies the canonical SHA256 matches ``predictions.jsonl.canonical.sha256``.
- Calls ``score_sealed_predictions()`` from the sealed scorer.
- Writes the scoring report to
  ``artifacts/evaluation/phase5/sealed-v2/scoring-report.json``.
- Appends an entry to the scoring ledger.

Usage:
    python3 scripts/score_phase5_sealed.py
    python3 scripts/score_phase5_sealed.py --labels .sealed/labels.jsonl

Environment:
    Sealed labels must be available (default: ``.sealed/labels.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from src.evaluation.manifests import (  # noqa: E402
    compute_file_sha256,
    compute_jsonl_sha256,
)
from src.evaluation.scoring_ledger import (  # noqa: E402
    ScoringLedgerEntry,
    append_ledger_entry,
)
from src.evaluation.sealed_scorer import score_sealed_predictions  # noqa: E402

OUTPUT_DIR = BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "sealed-v2"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
RAW_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.sha256"
CANONICAL_SHA_PATH = OUTPUT_DIR / "predictions.jsonl.canonical.sha256"
COMPLETE_MARKER_PATH = OUTPUT_DIR / "predictions.jsonl.complete.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run-manifest.json"
SCORING_PROTOCOL_PATH = OUTPUT_DIR / "scoring-protocol.json"
SCORING_REPORT_PATH = OUTPUT_DIR / "scoring-report.json"
LEDGER_PATH = OUTPUT_DIR / "scoring-ledger.json"

DEFAULT_LABELS_PATH = BACKEND_DIR / ".sealed" / "labels.jsonl"

SCORER_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _verify_integrity(labels_path: Path) -> dict[str, Any]:
    """Verify the predictions file, marker, and dual hashes.

    Returns the completion marker dict on success. Raises ``SystemExit``
    on any integrity failure.
    """
    if not PREDICTIONS_PATH.is_file():
        raise SystemExit(f"ERROR: predictions not found at {PREDICTIONS_PATH}")
    if not COMPLETE_MARKER_PATH.is_file():
        raise SystemExit(
            f"ERROR: completion marker not found at {COMPLETE_MARKER_PATH}. "
            "Run scripts/run_phase5_sealed_blind.py first."
        )
    if not RAW_SHA_PATH.is_file():
        raise SystemExit(f"ERROR: raw SHA256 file not found at {RAW_SHA_PATH}")
    if not CANONICAL_SHA_PATH.is_file():
        raise SystemExit(
            f"ERROR: canonical SHA256 file not found at {CANONICAL_SHA_PATH}"
        )

    marker = json.loads(_read_text(COMPLETE_MARKER_PATH))
    expected_raw = _read_text(RAW_SHA_PATH).strip()
    expected_canonical = _read_text(CANONICAL_SHA_PATH).strip()

    actual_raw = compute_file_sha256(PREDICTIONS_PATH)
    actual_canonical = compute_jsonl_sha256(PREDICTIONS_PATH)

    if actual_raw != expected_raw:
        raise SystemExit(
            "ERROR: raw SHA256 mismatch.\n"
            f"  expected (predictions.jsonl.sha256): {expected_raw}\n"
            f"  actual:                              {actual_raw}"
        )
    if actual_canonical != expected_canonical:
        raise SystemExit(
            "ERROR: canonical SHA256 mismatch.\n"
            f"  expected (predictions.jsonl.canonical.sha256): {expected_canonical}\n"
            f"  actual:                                         {actual_canonical}"
        )
    if actual_raw != marker.get("raw_sha256"):
        raise SystemExit(
            "ERROR: raw SHA256 does not match completion marker.\n"
            f"  marker: {marker.get('raw_sha256')}\n"
            f"  actual: {actual_raw}"
        )
    if actual_canonical != marker.get("canonical_sha256"):
        raise SystemExit(
            "ERROR: canonical SHA256 does not match completion marker.\n"
            f"  marker: {marker.get('canonical_sha256')}\n"
            f"  actual: {actual_canonical}"
        )

    print(f"Predictions raw SHA256 verified:       {actual_raw}")
    print(f"Predictions canonical SHA256 verified: {actual_canonical}")
    print(f"Case count from marker:                {marker.get('case_count')}")

    if not labels_path.is_file():
        raise SystemExit(f"ERROR: sealed labels not found at {labels_path}")
    labels_sha = compute_jsonl_sha256(labels_path)
    print(f"Sealed labels canonical SHA256:        {labels_sha}")

    return {
        "marker": marker,
        "raw_sha256": actual_raw,
        "canonical_sha256": actual_canonical,
        "labels_sha256": labels_sha,
    }


def _write_protocol(
    *,
    canonical_pred_sha: str,
    labels_sha: str,
    case_count: int,
) -> str:
    """Write the scoring protocol JSON for the sealed scorer.

    Returns the ``run_id`` recorded in the protocol.
    """
    run_id = "sealed-v2-score"
    if RUN_MANIFEST_PATH.is_file():
        manifest = json.loads(_read_text(RUN_MANIFEST_PATH))
        run_id = str(manifest.get("run_id", run_id))
    protocol = {
        "run_id": run_id,
        "run_type": "sealed",
        "predictions_sha256": canonical_pred_sha,
        "labels_sha256": labels_sha,
        "case_count": case_count,
    }
    SCORING_PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORING_PROTOCOL_PATH.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Scoring protocol saved to: {SCORING_PROTOCOL_PATH}")
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score Phase 5 sealed predictions (no RAG engine calls)."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS_PATH,
        help=f"Path to sealed labels JSONL (default: {DEFAULT_LABELS_PATH}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Sealed Scoring (v2)")
    print("=" * 60)

    integrity = _verify_integrity(args.labels)

    run_id = _write_protocol(
        canonical_pred_sha=integrity["canonical_sha256"],
        labels_sha=integrity["labels_sha256"],
        case_count=integrity["marker"]["case_count"],
    )

    print("Scoring predictions (no RAG calls)...")
    report = score_sealed_predictions(
        predictions_path=PREDICTIONS_PATH,
        labels_path=args.labels,
        protocol_path=SCORING_PROTOCOL_PATH,
        output_path=SCORING_REPORT_PATH,
    )
    print(f"Scoring report saved to: {SCORING_REPORT_PATH}")

    summary = report.get("summary", {})
    entry = ScoringLedgerEntry(
        run_id=str(report.get("run_id", run_id)),
        scored_at=_now_iso(),
        predictions_sha256=str(
            report.get("predictions_sha256", integrity["canonical_sha256"])
        ),
        labels_sha256=str(report.get("labels_sha256", integrity["labels_sha256"])),
        case_count=int(summary.get("total", 0)),
        passed=int(summary.get("passed", 0)),
        failed=int(summary.get("failed", 0)),
        pass_rate=float(summary.get("pass_rate", 0.0)),
        scorer_version=SCORER_VERSION,
    )
    append_ledger_entry(LEDGER_PATH, entry)
    print(f"Scoring ledger appended at: {LEDGER_PATH}")

    print("\nSealed Scoring Summary:")
    print(f"  Total cases: {entry.case_count}")
    print(f"  Passed:      {entry.passed}")
    print(f"  Failed:      {entry.failed}")
    print(f"  Pass rate:   {entry.pass_rate:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
