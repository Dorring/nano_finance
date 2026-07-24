"""Tests for tokenizer-manifest.json.

验证 Phase 6 发布的 tokenizer 清单产物，包括词表大小 (65000)、
special tokens 数量 (9个)、<pad> 不在 special tokens 中、
算法标识为 byte-level-bpe、tokenizer 文件 SHA256 及压缩率基准数据。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

EXPECTED_VOCAB_SIZE = 65000
EXPECTED_SPECIAL_TOKEN_COUNT = 9
NON_SPECIAL_TOKEN = "<pad>"


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def tokenizer_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "tokenizer-manifest.json")


def test_vocab_size_is_65000(tokenizer_manifest):
    # 验证 vocab_size == 65000
    assert tokenizer_manifest.get("vocab_size") == EXPECTED_VOCAB_SIZE, \
        f"Expected vocab_size={EXPECTED_VOCAB_SIZE}, got {tokenizer_manifest.get('vocab_size')}"


def test_special_tokens_count_is_9(tokenizer_manifest):
    # 验证 special_tokens 包含 9 个 token
    special = tokenizer_manifest.get("special_tokens")
    assert special is not None, "Missing special_tokens field"
    if isinstance(special, dict):
        count = len(special)
    elif isinstance(special, list):
        count = len(special)
    else:
        pytest.skip(f"Unexpected special_tokens type: {type(special)}")
    assert count == EXPECTED_SPECIAL_TOKEN_COUNT, \
        f"Expected {EXPECTED_SPECIAL_TOKEN_COUNT} special tokens, got {count}"


def test_pad_not_in_special_tokens(tokenizer_manifest):
    # 验证 <pad> 不在 special_tokens 中
    special = tokenizer_manifest.get("special_tokens", [])
    if isinstance(special, dict):
        tokens = list(special.keys()) + [str(v) for v in special.values()]
    else:
        tokens = [str(t) for t in special]
    assert NON_SPECIAL_TOKEN not in tokens, \
        f"'{NON_SPECIAL_TOKEN}' should not be a special token"


def test_algorithm_is_byte_level_bpe(tokenizer_manifest):
    # 验证 algorithm == "byte-level-bpe"
    algo = tokenizer_manifest.get("algorithm", "")
    assert algo == "byte-level-bpe", f"Expected byte-level-bpe, got {algo}"


def test_tokenizer_file_sha256_present(tokenizer_manifest):
    # 验证有 tokenizer 文件的 SHA256
    found = False

    def _search(obj):
        nonlocal found
        if isinstance(obj, dict):
            for k, v in obj.items():
                if ("sha256" in k.lower() or "hash" in k.lower()) and isinstance(v, str):
                    if re.fullmatch(r"[0-9a-fA-F]{64}", v):
                        found = True
                        return
                _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)

    _search(tokenizer_manifest)
    assert found, "No tokenizer file SHA256 (64-char hex) found in tokenizer-manifest.json"


def test_compression_benchmark_present(tokenizer_manifest):
    # 验证有压缩率 benchmark 数据
    found = False
    for key in ["compression_ratio", "compression_benchmark", "benchmark", "compression"]:
        if key in tokenizer_manifest:
            found = True
            break
    if not found:
        # 深度搜索包含 compression 或 benchmark 的键
        def _search(obj):
            nonlocal found
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if "compress" in k.lower() or "benchmark" in k.lower():
                        found = True
                        return
                    _search(v)
            elif isinstance(obj, list):
                for item in obj:
                    _search(item)

        _search(tokenizer_manifest)
    assert found, "No compression benchmark data found in tokenizer-manifest.json"
