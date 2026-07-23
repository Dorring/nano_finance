"""Tests for checkpoint-manifest.json.

Verify Phase 6 checkpoint manifest: base checkpoint (step 28000),
SFT checkpoint (step 150) existence, SHA256 hashes, model_config,
and val_bpb being float.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

BASE_STEP = 28000
SFT_STEP = 150


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_checkpoints(manifest) -> list:
    cks = manifest.get("checkpoints")
    if cks is None:
        cks = manifest.get("checkpoints_list")
    if cks is None:
        if isinstance(manifest, list):
            cks = manifest
        elif isinstance(manifest, dict):
            vals = list(manifest.values())
            if all(isinstance(v, dict) for v in vals):
                cks = vals
    if isinstance(cks, dict):
        cks = list(cks.values())
    return cks if isinstance(cks, list) else []


def _find_checkpoint(checkpoints, step):
    for ck in checkpoints:
        if not isinstance(ck, dict):
            continue
        ck_step = ck.get("step") or ck.get("training_step") or ck.get("global_step")
        if ck_step == step:
            return ck
    return None


@pytest.fixture
def checkpoint_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "checkpoint-manifest.json")


def test_base_checkpoint_step_28000_exists(checkpoint_manifest):
    cks = _get_checkpoints(checkpoint_manifest)
    base = _find_checkpoint(cks, BASE_STEP)
    assert base is not None


def test_sft_checkpoint_step_150_exists(checkpoint_manifest):
    cks = _get_checkpoints(checkpoint_manifest)
    sft = _find_checkpoint(cks, SFT_STEP)
    assert sft is not None


def test_each_checkpoint_has_sha256(checkpoint_manifest):
    cks = _get_checkpoints(checkpoint_manifest)
    assert len(cks) > 0, "No checkpoints found"
    for ck in cks:
        if not isinstance(ck, dict):
            continue
        sha = ck.get("sha256") or ck.get("hash")
        ck_name = ck.get("name", ck.get("step", "unknown"))
        assert sha is not None, f"Missing sha256: {ck_name}"
        assert re.fullmatch(r"[0-9a-fA-F]{64}", str(sha)), f"Invalid sha256: {ck_name}"


def test_each_checkpoint_has_model_config(checkpoint_manifest):
    cks = _get_checkpoints(checkpoint_manifest)
    assert len(cks) > 0
    for ck in cks:
        if not isinstance(ck, dict):
            continue
        cfg = ck.get("model_config") or ck.get("config") or ck.get("model")
        ck_name = ck.get("name", ck.get("step", "unknown"))
        assert cfg is not None, f"Missing model_config: {ck_name}"


def test_val_bpb_is_float(checkpoint_manifest):
    cks = _get_checkpoints(checkpoint_manifest)
    assert len(cks) > 0
    found_any = False
    for ck in cks:
        if not isinstance(ck, dict):
            continue
        bpb = ck.get("val_bpb")
        if bpb is not None:
            found_any = True
            assert isinstance(bpb, float), f"val_bpb should be float, got {type(bpb).__name__}"
    if not found_any:
        pytest.skip("No val_bpb field found")
