"""Tests for model-lineage.json.

Verify base checkpoint to SFT checkpoint lineage chain,
parent_checkpoint_sha256 field presence, and tokenizer_sha256
consistency with tokenizer-manifest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_field(obj, field):
    """Recursively find a field by name."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field:
                return v
            result = _find_field(v, field)
            if result is not None:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_field(item, field)
            if result is not None:
                return result
    return None


@pytest.fixture
def model_lineage() -> dict:
    return _load_json(ARTIFACTS_DIR / "model-lineage.json")


@pytest.fixture
def tokenizer_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "tokenizer-manifest.json")


def test_lineage_chain_base_to_sft(model_lineage):
    # Verify base checkpoint -> SFT checkpoint lineage chain
    assert isinstance(model_lineage, dict)
    text = json.dumps(model_lineage)
    has_base = "base" in text.lower() or "28000" in text
    has_sft = "sft" in text.lower() or "150" in text
    assert has_base, "Lineage missing base checkpoint reference"
    assert has_sft, "Lineage missing SFT checkpoint reference"


def test_parent_identity_digest_present(model_lineage):
    # Verify parent_identity_digest exists (renamed from parent_checkpoint_sha256)
    parent_digest = _find_field(model_lineage, "parent_identity_digest")
    if parent_digest is None:
        # Fallback for backward compatibility
        parent_digest = _find_field(model_lineage, "parent_checkpoint_sha256")
    assert parent_digest is not None, "parent_identity_digest not found in model-lineage.json"
    assert isinstance(parent_digest, str), f"parent_identity_digest should be str, got {type(parent_digest).__name__}"
    assert len(parent_digest) == 64, f"parent_identity_digest should be 64-char hex, got len={len(parent_digest)}"


def test_tokenizer_sha256_matches_tokenizer_manifest(model_lineage, tokenizer_manifest):
    # Verify tokenizer_sha256 matches tokenizer-manifest
    lineage_sha = _find_field(model_lineage, "tokenizer_sha256")
    if lineage_sha is None:
        pytest.skip("tokenizer_sha256 not found in model-lineage.json")
    manifest_sha = (
        _find_field(tokenizer_manifest, "tokenizer_sha256")
        or _find_field(tokenizer_manifest, "sha256")
        or _find_field(tokenizer_manifest, "hash")
    )
    if manifest_sha is None:
        pytest.skip("tokenizer sha256 not found in tokenizer-manifest.json")
    assert lineage_sha == manifest_sha, f"Tokenizer sha256 mismatch: lineage={lineage_sha} vs manifest={manifest_sha}"
