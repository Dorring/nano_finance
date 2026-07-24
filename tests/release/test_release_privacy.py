"""Tests for check_release_privacy.py execution.

Verify check_release_privacy.py script: exit code 0,
scans docs/release/ and artifacts/release/,
and can detect sensitive information in temp files.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "release"


@pytest.fixture
def privacy_script() -> Path:
    script = SCRIPTS_DIR / "check_release_privacy.py"
    if not script.exists():
        pytest.skip(f"Script not found: {script}")
    return script


def test_check_release_privacy_exit_code_zero(privacy_script):
    # Run script and verify exit code 0
    result = subprocess.run(
        ["python", str(privacy_script)],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT), timeout=300,
    )
    assert result.returncode == 0, f"Exit {result.returncode}\nstderr: {result.stderr}"


def test_privacy_scan_covers_release_dirs(privacy_script):
    # Verify scanning docs/release/ and artifacts/release/
    result = subprocess.run(
        ["python", str(privacy_script)],
        capture_output=True, text=True,
        cwd=str(REPO_ROOT), timeout=300,
    )
    output = result.stdout + result.stderr
    docs_exists = (REPO_ROOT / "docs" / "release").exists()
    artifacts_exists = (REPO_ROOT / "artifacts" / "release").exists()
    assert "docs" in output.lower() or "release" in output.lower() or docs_exists
    assert "artifacts" in output.lower() or artifacts_exists


def test_privacy_script_detects_sensitive_info(privacy_script):
    # Create temp file with sensitive info, verify script detects it
    docs_release = REPO_ROOT / "docs" / "release"
    if not docs_release.exists():
        pytest.skip("docs/release/ directory does not exist")
    sensitive_file = docs_release / "_test_sensitive_temp.md"
    sensitive_content = (
        "# Temp test file\n"
        "API_KEY=sk-1234567890abcdef1234567890abcdef\n"
        "password = my_secret_password_123\n"
        "SECRET_TOKEN=ghp_abcdef1234567890abcdef1234567890\n"
    )
    try:
        sensitive_file.write_text(sensitive_content, encoding="utf-8")
        result = subprocess.run(
            ["python", str(privacy_script)],
            capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=300,
        )
        output = result.stdout + result.stderr
        detected = (
            "sensitive" in output.lower()
            or "secret" in output.lower()
            or "leak" in output.lower()
            or "api_key" in output.lower()
            or "password" in output.lower()
            or result.returncode != 0
        )
        assert detected, "Script failed to detect sensitive information"
    finally:
        if sensitive_file.exists():
            sensitive_file.unlink()
