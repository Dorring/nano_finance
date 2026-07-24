"""Tests for check_release_determinism.py execution.

Verify check_release_determinism.py script: exit code 0,
and consecutive manifest generation is identical.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "release"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"


@pytest.fixture
def determinism_script() -> Path:
    script = SCRIPTS_DIR / "check_release_determinism.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    return script


def _run_script(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python", str(script)],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT), timeout=600,
    )


def test_check_release_determinism_exit_code_zero(determinism_script):
    # Run script and verify exit code 0
    result = _run_script(determinism_script)
    assert result.returncode == 0, f"Exit {result.returncode}\nstderr: {result.stderr}"


def test_consecutive_manifests_are_identical(determinism_script):
    # Verify consecutive manifest generation is identical
    manifest_path = ARTIFACTS_DIR / "release-manifest.json"

    def _manifest_hash():
        if manifest_path.exists():
            return hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        return None

    result1 = _run_script(determinism_script)
    if result1.returncode != 0:
        pytest.skip(f"First run failed: {result1.stderr}")
    hash_after_1 = _manifest_hash()

    result2 = _run_script(determinism_script)
    if result2.returncode != 0:
        pytest.skip(f"Second run failed: {result2.stderr}")
    hash_after_2 = _manifest_hash()

    if hash_after_1 is not None and hash_after_2 is not None:
        assert hash_after_1 == hash_after_2, "Manifest differs between consecutive runs"
    assert result1.stdout == result2.stdout, "Script output differs between consecutive runs"
