#!/usr/bin/env python3
"""
generate_phase5_artifacts.py

Generates the Phase 5 evaluation artifact JSON files under
``artifacts/evaluation/phase5/`` for the finquery_rag backend.

The script reads the **frozen evaluation protocol** from
``artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json`` and
emits deterministic per-run-type artifacts into three subdirectories:

    artifacts/evaluation/phase5/
        protocol/                       <-- input (frozen, read-only)
            phase5-evaluation-protocol.json
        baseline/                       <-- generated
            baseline-config.json
        calibration/                    <-- generated
            calibration-search-space.json
        sealed/                         <-- generated
            sealed-run-policy.json

Determinism contract
--------------------
The output is a pure function of its inputs. To guarantee that two runs
produce byte-identical output:

1. ``generated_commit`` is read from the protocol's ``baseline_commit`` —
   NEVER from a live ``git rev-parse HEAD``.
2. ``generated_at`` is read from the dev partition's ``manifest.json``
   ``created_at`` field — NEVER from the system clock.
3. All JSON is written with ``sort_keys=True``, ``indent=2``,
   ``ensure_ascii=False`` and a trailing newline.

Fail-fast contract
------------------
Every evidence path referenced by the artifacts is validated to exist
*before* any file is written. If any referenced path is missing the script
prints an error to stderr and returns exit code 1 without writing anything.

Exit codes
----------
    0  - all artifacts generated successfully
    1  - an unrecoverable error occurred (missing protocol, missing
         evidence path, write failure)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # .../backend

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts", "evaluation", "phase5")
PROTOCOL_DIR = os.path.join(ARTIFACTS_DIR, "protocol")
PROTOCOL_PATH = os.path.join(PROTOCOL_DIR, "phase5-evaluation-protocol.json")

BASELINE_DIR = os.path.join(ARTIFACTS_DIR, "baseline")
CALIBRATION_DIR = os.path.join(ARTIFACTS_DIR, "calibration")
SEALED_DIR = os.path.join(ARTIFACTS_DIR, "sealed")

DEV_MANIFEST_PATH = os.path.join(
    ROOT_DIR, "eval_data", "phase5", "dev", "manifest.json"
)

GENERATED_BY = "scripts/generate_phase5_artifacts.py"
SCHEMA_VERSION = "1.0"

# Fixed fallback timestamp used only when the dev manifest cannot be read.
# This keeps the script deterministic even without the manifest present.
FALLBACK_GENERATED_AT = "1970-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Deterministic metadata helpers
# ---------------------------------------------------------------------------
def _read_json(path: str) -> Dict[str, Any]:
    """Read a JSON object from ``path``. Raises on missing/invalid file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _deterministic_meta(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Build the standard metadata block using ONLY deterministic sources.

    ``generated_commit`` comes from the protocol's ``baseline_commit`` —
    never from a live git command. ``generated_at`` comes from the dev
    partition manifest's ``created_at`` — never from the system clock.
    """
    commit = str(protocol.get("baseline_commit", ""))
    generated_at = FALLBACK_GENERATED_AT
    try:
        manifest = _read_json(DEV_MANIFEST_PATH)
        manifest_created = manifest.get("created_at")
        if isinstance(manifest_created, str) and manifest_created:
            generated_at = manifest_created
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {
        "generated_by": GENERATED_BY,
        "generated_commit": commit,
        "generated_at": generated_at,
        "schema_version": SCHEMA_VERSION,
        "source_protocol": "artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json",
    }


# ---------------------------------------------------------------------------
# Fail-fast validation
# ---------------------------------------------------------------------------
def validate_evidence_paths(paths: List[str]) -> List[str]:
    """Validate that all referenced evidence paths exist.

    Returns a list of error strings. An empty list means all paths exist.
    """
    errors: List[str] = []
    seen = set()
    for rel in paths:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        candidate = os.path.join(ROOT_DIR, rel.replace("/", os.sep))
        if not os.path.exists(candidate):
            errors.append(f"evidence path does not exist: {rel}")
    return errors


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------
def build_baseline_artifact(
    protocol: Dict[str, Any], meta: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """baseline/baseline-config.json — the frozen baseline run configuration."""
    evidence = [
        "artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json",
        "eval_data/phase5/dev/questions.jsonl",
        "eval_data/phase5/dev/labels.jsonl",
        "src/evaluation/blind_runner.py",
        "src/evaluation/sealed_scorer.py",
    ]
    data = dict(meta)
    data["run_type"] = "baseline"
    data["baseline_commit"] = str(protocol.get("baseline_commit", ""))
    data["primary_metric"] = str(protocol.get("primary_metric", ""))
    data["random_seed"] = int(protocol.get("random_seed", 0))
    data["ablation_variants"] = [
        {"id": str(v.get("id", "")), "name": str(v.get("name", ""))}
        for v in protocol.get("ablation_variants", [])
    ]
    return data, evidence


def build_calibration_artifact(
    protocol: Dict[str, Any], meta: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """calibration/calibration-search-space.json — the calibratable parameter space."""
    evidence = [
        "artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json",
        "docs/evaluation/phase5-calibration.md",
        "src/evaluation/calibration.py",
    ]
    data = dict(meta)
    data["run_type"] = "calibration"
    data["calibration_search_space"] = dict(
        protocol.get("calibration_search_space", {})
    )
    data["candidate_selection_rule"] = dict(
        protocol.get("candidate_selection_rule", {})
    )
    return data, evidence


def build_sealed_artifact(
    protocol: Dict[str, Any], meta: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """sealed/sealed-run-policy.json — the sealed run policy."""
    evidence = [
        "artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json",
        "eval_data/phase5/sealed/questions.jsonl",
        "eval_data/phase5/sealed/manifest.public.json",
        "docs/evaluation/phase5-sealed-runbook.md",
        "src/evaluation/sealed_scorer.py",
    ]
    data = dict(meta)
    data["run_type"] = "sealed"
    data["held_out_run_policy"] = dict(protocol.get("held_out_run_policy", {}))
    data["public_manifest"] = "eval_data/phase5/sealed/manifest.public.json"
    data["sealed_labels_local_path"] = ".sealed/labels.jsonl"
    return data, evidence


# ---------------------------------------------------------------------------
# Atomic, deterministic writer
# ---------------------------------------------------------------------------
def write_artifact(rel_dir: str, filename: str, data: Dict[str, Any]) -> str:
    """Write ``data`` as deterministic pretty JSON to ``<rel_dir>/<filename>``.

    JSON is serialized with ``sort_keys=True`` so that re-running the script
    on identical input produces byte-identical output. A trailing newline is
    always written. Returns the absolute path written.
    """
    out_dir = os.path.join(ROOT_DIR, rel_dir.replace("/", os.sep))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    """Generate Phase 5 evaluation artifacts.

    Returns 0 on success and 1 on any unrecoverable failure.
    """
    # --- Fail Fast: protocol must exist and be valid JSON ---
    if not os.path.isfile(PROTOCOL_PATH):
        print(
            f"ERROR: protocol file not found: {PROTOCOL_PATH}", file=sys.stderr
        )
        return 1
    try:
        protocol = _read_json(PROTOCOL_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read protocol: {exc}", file=sys.stderr)
        return 1

    # --- Build all artifacts in memory first (so we can validate evidence) ---
    meta = _deterministic_meta(protocol)
    builders = [
        (
            "artifacts/evaluation/phase5/baseline",
            "baseline-config.json",
            build_baseline_artifact,
        ),
        (
            "artifacts/evaluation/phase5/calibration",
            "calibration-search-space.json",
            build_calibration_artifact,
        ),
        (
            "artifacts/evaluation/phase5/sealed",
            "sealed-run-policy.json",
            build_sealed_artifact,
        ),
    ]

    all_artifacts: List[Tuple[str, str, Dict[str, Any], List[str]]] = []
    try:
        for rel_dir, filename, builder in builders:
            data, evidence = builder(protocol, meta)
            all_artifacts.append((rel_dir, filename, data, evidence))
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to build artifacts: {exc}", file=sys.stderr)
        return 1

    # --- Fail Fast: validate all evidence paths exist BEFORE writing ---
    all_refs: List[str] = []
    for _, _, _, evidence in all_artifacts:
        all_refs.extend(evidence)
    # The protocol path is always required.
    all_refs.append(
        "artifacts/evaluation/phase5/protocol/phase5-evaluation-protocol.json"
    )
    path_errors = validate_evidence_paths(all_refs)
    if path_errors:
        for err in path_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("Generating Phase 5 evaluation artifacts...")
    print(f"  protocol    : {PROTOCOL_PATH}")
    print(f"  commit      : {meta['generated_commit']}")
    print(f"  generated_at: {meta['generated_at']}")
    print()

    # --- Write all artifacts ---
    written: List[str] = []
    try:
        for rel_dir, filename, data, _ in all_artifacts:
            path = write_artifact(rel_dir, filename, data)
            written.append(path)
            print(
                f"  [OK] {os.path.relpath(path, ROOT_DIR)}  "
                f"({os.path.getsize(path)} bytes)"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: failed to write artifacts: {exc}", file=sys.stderr)
        return 1

    print()
    print(
        f"Successfully generated {len(written)} of {len(builders)} artifacts."
    )
    print("All evidence references validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
