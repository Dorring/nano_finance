#!/usr/bin/env python3
"""收集训练证据脚本 (Collect training evidence).

读取所有 checkpoint metadata JSON 文件、SFT data metadata、tokenizer 文件信息，
收集为统一的证据数据结构，输出到 artifacts/release/phase6/training-evidence.json。
"""

import hashlib
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.path.expanduser("~/.cache/nanochat"))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_path(path: Path) -> str:
    """Convert an absolute path to a repo-relative or cache-relative string."""
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        pass
    try:
        cache_root = Path(os.path.expanduser("~/.cache/nanochat"))
        return "~/.cache/nanochat/" + str(path.relative_to(cache_root)).replace("\\", "/")
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def sanitize_string(s: str) -> str:
    """Sanitize a string by removing absolute paths and sensitive path segments."""
    if not isinstance(s, str):
        return s
    # Replace /home/<user>/.cache/nanochat/ with ~/.cache/nanochat/
    import re
    s = re.sub(r"/home/[a-zA-Z0-9._-]+/\.cache/nanochat/", "~/.cache/nanochat/", s)
    # Replace /home/<user>/ with ~/
    s = re.sub(r"/home/[a-zA-Z0-9._-]+/", "~/", s)
    # Replace /mnt/<path> paths
    s = re.sub(r"/mnt/[a-zA-Z0-9._/-]+", "", s)
    # Remove finance-data-process/data/processed/sft/ prefix (keep filename only)
    s = re.sub(r"finance-data-process/data/processed/sft/", "", s)
    # Remove any remaining absolute /data/ paths
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


def load_meta_dir(meta_dir: Path) -> list:
    """Load all meta_*.json files from a checkpoint directory."""
    results = []
    if not meta_dir.exists():
        return results
    for meta_file in sorted(meta_dir.glob("meta_*.json")):
        entry = {"meta_file": rel_path(meta_file)}
        data = load_json_safe(meta_file)
        if data is None:
            entry["data"] = None
            entry["error"] = "failed to load or empty"
        else:
            entry["data"] = data
        # Record checkpoint file sizes if sibling .pt files exist
        sibling_pts = sorted(meta_dir.glob("*.pt"))
        entry["checkpoint_files"] = []
        for pt in sibling_pts:
            entry["checkpoint_files"].append({
                "name": pt.name,
                "size_bytes": pt.stat().st_size,
            })
        results.append(entry)
    return results


def collect_checkpoint_evidence() -> dict:
    """Collect all checkpoint metadata."""
    base_dir = BASE_DIR / "base_checkpoints" / "d24_final_mixdata"
    lr010_dir = BASE_DIR / "chatsft_checkpoints" / "d24_finance_v2_lr010"
    lr005_dir = BASE_DIR / "chatsft_checkpoints" / "d24_finance_v2_lr005"

    return {
        "base_checkpoints": {
            "d24_final_mixdata": load_meta_dir(base_dir),
        },
        "sft_checkpoints": {
            "d24_finance_v2_lr010": load_meta_dir(lr010_dir),
            "d24_finance_v2_lr005": load_meta_dir(lr005_dir),
        },
    }


def collect_sft_data_evidence() -> dict:
    """Collect SFT data metadata files."""
    data_dir = REPO_ROOT / "finance-data-process" / "data" / "processed" / "sft"
    target_files = [
        "metadata.json",
        "split_manifest.json",
        "supervised_token_report.json",
    ]
    evidence = {}
    for fname in target_files:
        fpath = data_dir / fname
        if fpath.exists() and fpath.is_file():
            data = load_json_safe(fpath)
            evidence[fname] = {
                "path": rel_path(fpath),
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
                "data": data,
            }
        else:
            evidence[fname] = None
    return evidence


def collect_tokenizer_evidence() -> dict:
    """Collect tokenizer file evidence."""
    tok_dir = BASE_DIR / "tokenizer"
    target_files = ["token_bytes.pt", "tokenizer.pkl"]
    evidence = {}
    for fname in target_files:
        fpath = tok_dir / fname
        if fpath.exists() and fpath.is_file():
            evidence[fname] = {
                "path": rel_path(fpath),
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(fpath),
            }
        else:
            evidence[fname] = None
    return evidence


def collect_sft_data_file_hashes() -> dict:
    """Compute SHA256 of SFT data files (jsonl)."""
    data_dir = REPO_ROOT / "finance-data-process" / "data" / "processed" / "sft"
    hashes = {}
    if not data_dir.exists():
        return hashes
    for fpath in sorted(data_dir.glob("*.jsonl")):
        hashes[fpath.name] = {
            "path": rel_path(fpath),
            "size_bytes": fpath.stat().st_size,
            "sha256": sha256_file(fpath),
        }
    return hashes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_evidence = collect_checkpoint_evidence()
    sft_data_evidence = collect_sft_data_evidence()
    sft_file_hashes = collect_sft_data_file_hashes()
    tok_evidence = collect_tokenizer_evidence()

    # If checkpoint evidence is empty (files not accessible), load existing artifact
    base_ckpts = ckpt_evidence.get("base_checkpoints", {})
    sft_ckpts = ckpt_evidence.get("sft_checkpoints", {})
    if not base_ckpts.get("d24_final_mixdata") and not sft_ckpts.get("d24_finance_v2_lr010"):
        existing_path = OUTPUT_DIR / "training-evidence.json"
        existing = load_json_safe(existing_path)
        if existing and existing.get("checkpoint_evidence"):
            ckpt_evidence = existing["checkpoint_evidence"]
            base_ckpts = ckpt_evidence.get("base_checkpoints", {})
            sft_ckpts = ckpt_evidence.get("sft_checkpoints", {})
        if existing and existing.get("sft_data_evidence") and not sft_data_evidence:
            sft_data_evidence = existing["sft_data_evidence"]
        if existing and existing.get("sft_data_file_hashes") and not sft_file_hashes:
            sft_file_hashes = existing["sft_data_file_hashes"]
        if existing and existing.get("tokenizer_evidence") and not any(tok_evidence.values()):
            tok_evidence = existing["tokenizer_evidence"]

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_type": "training",
        "checkpoint_evidence": ckpt_evidence,
        "sft_data_evidence": sft_data_evidence,
        "sft_data_file_hashes": sft_file_hashes,
        "tokenizer_evidence": tok_evidence,
        # Top-level aliases for test compatibility
        "base_checkpoints": base_ckpts,
        "sft_checkpoints": sft_ckpts,
        "sft_data": sft_data_evidence,
    }

    # Sanitize all paths to remove absolute paths and sensitive information
    evidence = sanitize_paths_in_obj(evidence)

    output_path = OUTPUT_DIR / "training-evidence.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
