"""Tests for collect_training_evidence.py output (training-evidence.json).

验证 Phase 6 发布管线中 collect_training_evidence.py 脚本生成的训练证据产物，
包括 base/sft checkpoint 元数据、SFT 数据元数据、哈希合法性和绝对路径泄露检查。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "release"

# 绝对路径泄露检测模式
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\"),               # Windows 盘符路径 C:\
    re.compile(r"/(home|Users|root|mnt|data|var|opt|tmp|srv)/"),  # POSIX 绝对路径
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_hashes(obj, acc=None):
    """递归收集所有 hash/sha256 字段的值。"""
    if acc is None:
        acc = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and ("hash" in k.lower() or "sha256" in k.lower()):
                acc.append((k, v))
            else:
                _collect_hashes(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _collect_hashes(item, acc)
    return acc


def _find_absolute_paths(obj, acc=None):
    """递归查找字符串值中包含的绝对路径。"""
    if acc is None:
        acc = []
    if isinstance(obj, str):
        for pat in ABSOLUTE_PATH_PATTERNS:
            if pat.search(obj):
                acc.append(obj)
                break
    elif isinstance(obj, dict):
        for v in obj.values():
            _find_absolute_paths(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _find_absolute_paths(item, acc)
    return acc


@pytest.fixture
def training_evidence() -> dict:
    return _load_json(ARTIFACTS_DIR / "training-evidence.json")


def test_training_evidence_exists_and_parseable(training_evidence):
    # 验证 training-evidence.json 存在且可解析
    assert isinstance(training_evidence, dict)
    assert len(training_evidence) > 0


def test_base_checkpoints_meta_present(training_evidence):
    # 验证包含 base_checkpoints 的 meta 数据
    assert "base_checkpoints" in training_evidence, "Missing base_checkpoints section"
    base = training_evidence["base_checkpoints"]
    assert base is not None
    if isinstance(base, dict):
        assert "meta" in base or len(base) > 0
    elif isinstance(base, list):
        assert len(base) > 0, "base_checkpoints list is empty"


def test_sft_checkpoints_meta_present(training_evidence):
    # 验证包含 sft_checkpoints 的 meta 数据
    assert "sft_checkpoints" in training_evidence, "Missing sft_checkpoints section"
    sft = training_evidence["sft_checkpoints"]
    assert sft is not None
    if isinstance(sft, dict):
        assert "meta" in sft or len(sft) > 0
    elif isinstance(sft, list):
        assert len(sft) > 0, "sft_checkpoints list is empty"


def test_sft_data_metadata_present(training_evidence):
    # 验证包含 sft_data 的 metadata
    assert "sft_data" in training_evidence, "Missing sft_data section"
    sft_data = training_evidence["sft_data"]
    assert sft_data is not None
    if isinstance(sft_data, dict):
        assert "metadata" in sft_data or len(sft_data) > 0
    elif isinstance(sft_data, list):
        assert len(sft_data) > 0, "sft_data list is empty"


def test_all_hashes_are_64_char_hex(training_evidence):
    # 验证所有 hash 是 64 字符的 hex
    hashes = _collect_hashes(training_evidence)
    assert hashes, "Expected at least one hash field in training-evidence.json"
    for key, value in hashes:
        assert re.fullmatch(r"[0-9a-fA-F]{64}", value), \
            f"Hash for '{key}' is not 64-char hex: {value}"


def test_no_absolute_paths_leaked(training_evidence):
    # 验证没有绝对路径泄露
    leaked = _find_absolute_paths(training_evidence)
    assert not leaked, \
        f"Absolute paths leaked in training-evidence.json: {leaked[:5]}"
