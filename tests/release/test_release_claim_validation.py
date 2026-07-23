"""Tests for validate_release_claims.py execution.

Verify validate_release_claims.py script: exit code 0,
output JSON exists, and all verified claim evidence files exist.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "release"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def validation_script() -> Path:
    script = SCRIPTS_DIR / "validate_release_claims.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    return script


@pytest.fixture
def claim_evidence_map() -> dict:
    return _load_json(ARTIFACTS_DIR / "claim-evidence-map.json")


def _run_script(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", str(script)],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT), timeout=300,
    )


def test_validate_release_claims_exit_code_zero(validation_script):
    result = _run_script(validation_script)
    assert result.returncode == 0, f"Exit {result.returncode}\nstderr: {result.stderr}"


def test_validation_output_json_exists(validation_script):
    # Verify output JSON exists
    result = _run_script(validation_script)
    if result.returncode != 0:
        pytest.skip(f"Script failed: {result.stderr}")
    output = None
    try:
        output = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        pass
    if output is None:
        for name in ["claim-validation-result.json", "validation-result.json", "claim-validation.json"]:
            candidate = ARTIFACTS_DIR / name
            if candidate.exists():
                output = json.loads(candidate.read_text(encoding="utf-8"))
                break
    assert output is not None, "No validation output JSON found"


def test_verified_claim_evidence_files_exist(validation_script, claim_evidence_map):
    # Verify all verified claim evidence files exist
    claims = claim_evidence_map.get("claims", [])
    if isinstance(claims, dict):
        claims = list(claims.values())
    verified = [c for c in claims if isinstance(c, dict) and c.get("status") == "verified"]
    if not verified:
        pytest.skip("No verified claims found")
    for claim in verified:
        evidence = claim.get("evidence", [])
        for ev in evidence:
            if isinstance(ev, dict):
                path_str = ev.get("path") or ev.get("file") or ev.get("artifact")
            elif isinstance(ev, str):
                path_str = ev
            else:
                continue
            if path_str is None:
                continue
            p = Path(path_str)
            ev_path = p if p.is_absolute() else REPO_ROOT / path_str
            cid = claim.get("id", "?")
            assert ev_path.exists(), f"Evidence not found for claim {cid}: {ev_path}"
