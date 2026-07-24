#!/usr/bin/env python3
"""生成统一发布 manifest (Generate unified release manifest).

聚合所有其他 manifest，生成统一发布 manifest。
输出到 artifacts/release/phase6/release-manifest.json。
"""

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.path.expanduser("~/.cache/nanochat"))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

SCHEMA_VERSION = "1.0"

RELEASE_ID = "nano-finance-d24-sft-v1"

# This release provides documentation and an evidence classification framework.
# It does NOT include a verifiable, downloadable model weight package.
# The step 150 checkpoint is a failed experiment (smoke checkpoint), not a
# production release model. The historical production checkpoint (sft1147)
# is not available for verification.
RELEASE_TYPE = "documentation_and_evidence"
RELEASE_MODEL_CHECKPOINT = None
EVALUATION_SMOKE_CHECKPOINT = "d24_finance_v2_lr010/step_150"
HISTORICAL_PRODUCTION_CHECKPOINT = "sft1147"
HISTORICAL_PRODUCTION_CHECKPOINT_STATUS = "unavailable_unverified"

# Known identifiers for linking
BASE_ARCHITECTURE_ID = "nanochat-d24-1.4b"
TOKENIZER_ID = "nanochat-bpe-65k"
PRETRAINING_RUN_ID = "d24_final_mixdata"
SFT_RUN_ID = "d24_finance_v2_lr010"
CHECKPOINT_ID = "d24_finance_v2_lr010/step_150"
RAG_RELEASE_ID = "finquery-rag-v1"
EVALUATION_RELEASE_ID = "nano-finance-eval-v1"

LICENSE = "MIT (Copyright 2025 Andrej Karpathy)"

# Manifests whose full data should NOT be embedded in release-manifest.json
# (to avoid prohibited claim text or large data leaking into the release manifest)
MANIFESTS_METADATA_ONLY = {"claim_evidence_map"}


def sanitize_string(s: str) -> str:
    """Sanitize a string by removing absolute paths and sensitive path segments."""
    if not isinstance(s, str):
        return s
    import re
    s = re.sub(r"/home/[a-zA-Z0-9._-]+/\.cache/nanochat/", "~/.cache/nanochat/", s)
    s = re.sub(r"/home/[a-zA-Z0-9._-]+/", "~/", s)
    s = re.sub(r"/mnt/[a-zA-Z0-9._/-]+", "", s)
    s = re.sub(r"finance-data-process/data/processed/sft/", "", s)
    s = re.sub(r"/data/[a-zA-Z0-9._/-]+", "", s)
    return s


def sanitize_paths_in_obj(obj):
    """Recursively sanitize all string values in a dict or list."""
    if isinstance(obj, dict):
        return {k: sanitize_paths_in_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_paths_in_obj(item) for item in obj]
    elif isinstance(obj, str):
        return sanitize_string(obj)
    return obj


def load_json_safe(path: Path):
    """Load JSON file, return None on failure."""
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def file_exists(path: Path) -> bool:
    """Check if a file exists."""
    return path.exists() and path.is_file()


def get_git_commit() -> str:
    """Get current git commit SHA."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return None


def collect_sub_manifests() -> dict:
    """Collect all sub-manifests with existence status."""
    manifest_files = {
        "checkpoint_manifest": "checkpoint-manifest.json",
        "model_lineage": "model-lineage.json",
        "pretraining_data_manifest": "pretraining-data-manifest.json",
        "sft_data_manifest": "sft-data-manifest.json",
        "tokenizer_manifest": "tokenizer-manifest.json",
        "training_evidence": "training-evidence.json",
        "training_runs": "training-runs.json",
        "evaluation_evidence": "evaluation-evidence.json",
        "dependency_manifest": "dependency-manifest.json",
        "license_inventory": "license-inventory.json",
        "claim_evidence_map": "claim-evidence-map.json",
    }
    collected = {}
    for key, fname in sorted(manifest_files.items()):
        fpath = OUTPUT_DIR / fname
        collected[key] = {
            "available": file_exists(fpath),
            "path": f"artifacts/release/phase6/{fname}" if file_exists(fpath) else None,
            "data": load_json_safe(fpath),
        }
    return collected


def build_release_manifest() -> dict:
    """Build the unified release manifest."""
    sub_manifests = collect_sub_manifests()

    manifest = {
        "base_architecture_id": BASE_ARCHITECTURE_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "component_manifests": {
            key: {
                "available": info["available"],
                "path": info["path"],
            }
            for key, info in sorted(sub_manifests.items())
        },
        "evaluation_release_id": EVALUATION_RELEASE_ID,
        "evaluation_smoke_checkpoint": EVALUATION_SMOKE_CHECKPOINT,
        "git_commit": get_git_commit(),
        "historical_production_checkpoint": HISTORICAL_PRODUCTION_CHECKPOINT,
        "historical_production_checkpoint_status": HISTORICAL_PRODUCTION_CHECKPOINT_STATUS,
        "license": LICENSE,
        "manifest_type": "release",
        "pretraining_run_id": PRETRAINING_RUN_ID,
        "rag_release_id": RAG_RELEASE_ID,
        "release_id": RELEASE_ID,
        "release_model_checkpoint": RELEASE_MODEL_CHECKPOINT,
        "release_type": RELEASE_TYPE,
        "schema_version": SCHEMA_VERSION,
        "sft_run_id": SFT_RUN_ID,
        "sub_manifest_data": {
            key: (info["data"] if key not in MANIFESTS_METADATA_ONLY else None)
            for key, info in sorted(sub_manifests.items())
        },
        "tokenizer_id": TOKENIZER_ID,
    }

    return manifest


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_release_manifest()
    manifest = sanitize_paths_in_obj(manifest)
    output_path = OUTPUT_DIR / "release-manifest.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
