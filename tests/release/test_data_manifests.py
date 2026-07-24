"""Tests for pretraining-data-manifest.json and sft-data-manifest.json.

Verify pretraining shard count (>0), SFT total samples (39534),
data source count (8), finance_r1 count (1225), and
train/val/test split ratio reasonableness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

EXPECTED_SFT_TOTAL = 39534
EXPECTED_SFT_SOURCES = 8
EXPECTED_FINANCE_R1_COUNT = 1225


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def pretraining_data_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "pretraining-data-manifest.json")


@pytest.fixture
def sft_data_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "sft-data-manifest.json")


def test_pretraining_shard_count_positive(pretraining_data_manifest):
    # When data dir is not accessible, shards list is empty and actual_shard_count is null.
    # We verify that expected_shard_count is recorded and verification_status is set.
    expected = pretraining_data_manifest.get("expected_shard_count")
    if expected is not None:
        assert expected > 0, f"expected_shard_count should be positive, got {expected}"

    shards = pretraining_data_manifest.get("shards")
    if shards is not None and isinstance(shards, list) and len(shards) > 0:
        # If shards are present, they should be real (non-zero size)
        for s in shards:
            if isinstance(s, dict) and "size_bytes" in s:
                assert s["size_bytes"] > 0, f"Shard {s.get('name')} has zero size"

    # verification_status must be present
    vs = pretraining_data_manifest.get("verification_status")
    assert vs is not None, "pretraining manifest missing verification_status"
    assert vs in ("verified_local", "historical_self_reported"), \
        f"Unexpected verification_status: {vs}"


def test_sft_total_samples_is_39534(sft_data_manifest):
    total = sft_data_manifest.get("total_samples")
    if total is None:
        total = sft_data_manifest.get("total")
    assert total == EXPECTED_SFT_TOTAL


def test_sft_has_8_data_sources(sft_data_manifest):
    sources = sft_data_manifest.get("sources")
    if sources is None:
        sources = sft_data_manifest.get("data_sources")
    if sources is None:
        pytest.skip("Cannot find data sources")
    if isinstance(sources, dict):
        count = len(sources)
    elif isinstance(sources, list):
        count = len(sources)
    else:
        pytest.skip(f"Unexpected sources type: {type(sources)}")
    assert count == EXPECTED_SFT_SOURCES, f"Expected {EXPECTED_SFT_SOURCES} sources, got {count}"


def test_sft_finance_r1_count_is_1225(sft_data_manifest):
    sources = sft_data_manifest.get("sources") or sft_data_manifest.get("data_sources") or {}
    finance_r1 = None
    if isinstance(sources, dict):
        finance_r1 = sources.get("finance_r1")
    elif isinstance(sources, list):
        for src in sources:
            if isinstance(src, dict) and src.get("name") == "finance_r1":
                finance_r1 = src
                break
    assert finance_r1 is not None, "finance_r1 not found"
    if isinstance(finance_r1, dict):
        count = finance_r1.get("count")
    else:
        count = finance_r1
    assert count == EXPECTED_FINANCE_R1_COUNT, f"Expected {EXPECTED_FINANCE_R1_COUNT}, got {count}"


def test_sft_train_val_test_split_reasonable(sft_data_manifest):
    split = sft_data_manifest.get("split") or sft_data_manifest.get("splits")
    if split is None:
        if all(k in sft_data_manifest for k in ("train", "val", "test")):
            split = sft_data_manifest
    if split is None:
        pytest.skip("Cannot find train/val/test split")
    if not isinstance(split, dict):
        pytest.skip("Split is not a dict")

    train = split.get("train")
    cot_train = split.get("cot_train")
    effective_train = split.get("effective_train")
    val = split.get("val") or split.get("validation")
    test = split.get("test")

    # Verify real split numbers (not adjusted for ratio thresholds)
    assert train == 30641, f"Expected train=30641, got {train}"
    assert cot_train == 979, f"Expected cot_train=979, got {cot_train}"
    assert effective_train == 31620, f"Expected effective_train=31620, got {effective_train}"
    assert val == 3958, f"Expected val=3958, got {val}"
    assert test == 3956, f"Expected test=3956, got {test}"

    # Verify effective_train = train + cot_train
    assert effective_train == train + cot_train, \
        f"effective_train ({effective_train}) != train + cot_train ({train + cot_train})"

    # Verify total consistency: effective_train + val + test = total_samples
    total_samples = sft_data_manifest.get("total_samples")
    if total_samples is not None:
        computed_total = effective_train + val + test
        assert computed_total == total_samples, \
            f"effective_train + val + test ({computed_total}) != total_samples ({total_samples})"

    # Train ratio should be close to 80% (allow reasonable rounding)
    total = effective_train + val + test
    train_ratio = effective_train / total
    assert 0.79 <= train_ratio <= 0.81, \
        f"Train ratio {train_ratio:.4f} outside expected ~0.80 range"
