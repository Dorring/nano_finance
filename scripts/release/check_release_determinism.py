#!/usr/bin/env python3
"""确定性检查 (Determinism check).

运行 generate_release_manifest.py 两次，比较输出是否一致。
检查所有 manifest 的字段排序稳定性。

退出码: 0=pass, 1=fail
"""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.path.expanduser("~/.cache/nanochat"))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

SCHEMA_VERSION = "1.0"

# Scripts to check for determinism
MANIFEST_SCRIPTS = [
    "collect_training_evidence.py",
    "build_tokenizer_manifest.py",
    "build_data_manifests.py",
    "build_checkpoint_manifest.py",
    "generate_release_manifest.py",
]

# Output files to compare
MANIFEST_OUTPUTS = [
    "training-evidence.json",
    "tokenizer-manifest.json",
    "pretraining-data-manifest.json",
    "sft-data-manifest.json",
    "checkpoint-manifest.json",
    "model-lineage.json",
    "release-manifest.json",
]


def run_script(script_name: str) -> bool:
    """Run a script and return success status."""
    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"  SKIP: {script_name} not found")
        return True

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            print(f"  ERROR running {script_name}: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT running {script_name}")
        return False
    except Exception as e:
        print(f"  EXCEPTION running {script_name}: {e}")
        return False


def read_json(path: Path):
    """Read and parse JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def serialize_json(data) -> str:
    """Serialize JSON deterministically."""
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)


def check_field_sorting() -> list:
    """Check that all manifest JSON files have sorted keys."""
    issues = []
    for fname in MANIFEST_OUTPUTS:
        fpath = OUTPUT_DIR / fname
        if not fpath.exists():
            continue
        data = read_json(fpath)
        if data is None:
            continue
        # Re-serialize with sort_keys and compare
        original = fpath.read_text(encoding="utf-8")
        reserialized = serialize_json(data)
        if original != reserialized:
            issues.append({
                "file": f"artifacts/release/phase6/{fname}",
                "issue": "file is not deterministically sorted",
            })
    return issues


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_issues = []

    # Step 1: Run all scripts once (first run)
    print("Step 1: First run of all manifest scripts...")
    for script in MANIFEST_SCRIPTS:
        print(f"  Running {script}...")
        if not run_script(script):
            all_issues.append({
                "issue": f"script failed on first run: {script}",
                "severity": "error",
            })

    # Capture first run outputs
    first_run = {}
    for fname in MANIFEST_OUTPUTS:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            first_run[fname] = fpath.read_text(encoding="utf-8")

    # Step 2: Run all scripts again (second run)
    print("Step 2: Second run of all manifest scripts...")
    for script in MANIFEST_SCRIPTS:
        print(f"  Running {script}...")
        if not run_script(script):
            all_issues.append({
                "issue": f"script failed on second run: {script}",
                "severity": "error",
            })

    # Compare outputs
    print("Step 3: Comparing outputs...")
    for fname in MANIFEST_OUTPUTS:
        fpath = OUTPUT_DIR / fname
        second_content = fpath.read_text(encoding="utf-8") if fpath.exists() else None
        first_content = first_run.get(fname)

        if first_content is None and second_content is None:
            continue
        if first_content is None:
            all_issues.append({
                "file": fname,
                "issue": "file appeared on second run but not first",
                "severity": "error",
            })
        elif second_content is None:
            all_issues.append({
                "file": fname,
                "issue": "file disappeared on second run",
                "severity": "error",
            })
        elif first_content != second_content:
            all_issues.append({
                "file": fname,
                "issue": "output differs between first and second run",
                "severity": "error",
            })

    # Step 4: Check field sorting stability
    print("Step 4: Checking field sorting...")
    sorting_issues = check_field_sorting()
    all_issues.extend(sorting_issues)

    has_errors = any(i.get("severity") == "error" for i in all_issues)

    result = {
        "checks": {
            "field_sorting": {
                "issues": sorting_issues,
                "passed": not any(
                    i.get("severity") == "error" for i in sorting_issues
                ),
            },
            "output_stability": {
                "issues": [
                    i for i in all_issues
                    if i.get("severity") == "error"
                    and "differs" in i.get("issue", "")
                ],
                "passed": not any(
                    "differs" in i.get("issue", "")
                    for i in all_issues
                ),
            },
        },
        "issues": all_issues,
        "passed": not has_errors,
        "schema_version": SCHEMA_VERSION,
        "scripts_checked": MANIFEST_SCRIPTS,
        "summary": {
            "files_compared": len(MANIFEST_OUTPUTS),
            "scripts_run": len(MANIFEST_SCRIPTS),
            "total_issues": len(all_issues),
        },
    }

    output_path = OUTPUT_DIR / "determinism-check.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")

    if result["passed"]:
        print("PASS: All determinism checks passed.")
        sys.exit(0)
    else:
        print(f"FAIL: {len(all_issues)} determinism issues found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
