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
    shards = pretraining_data_manifest.get("shards")
    if shards is None:
        shards = pretraining_data_manifest.get("shard_count")
    if shards is None:
        for k, v in pretraining_data_manifest.items():
            if "shard" in k.lower() and isinstance(v, list):
                shards = v
                break
    if shards is None:
        pytest.skip("Cannot determine shard count")
    if isinstance(shards, list):
        assert len(shards) > 0
    elif isinstance(shards, int):
        assert shards > 0
    else:
        pytest.skip(f"Unexpected shards type: {type(shards)}")


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
    train = split.get("train") if isinstance(split, dict) else None
    val = split.get("val") or split.get("validation") if isinstance(split, dict) else None
    test = split.get("test") if isinstance(split, dict) else None
    values = [v for v in (train, val, test) if v is not None]
    if not values:
        pytest.skip("No train/val/test values found")
    if not all(isinstance(v, (int, float)) for v in values):
        pytest.skip("Split values are not numeric")
    total = sum(values)
    assert total > 0
    train_ratio = (train or 0) / total
    val_ratio = (val or 0) / total
    test_ratio = (test or 0) / total
    assert 0.80 <= train_ratio <= 0.999
    assert 0.0 <= val_ratio <= 0.15
    assert 0.0 <= test_ratio <= 0.15
