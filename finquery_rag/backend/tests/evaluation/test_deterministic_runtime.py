"""Tests for src.evaluation.deterministic."""

from __future__ import annotations

import os

from src.evaluation.deterministic import (
    assert_deterministic_runtime,
    configure_deterministic_runtime,
)


def test_configure_sets_pythonhashseed() -> None:
    """configure_deterministic_runtime must set PYTHONHASHSEED."""
    configure_deterministic_runtime(12345)
    assert os.environ.get("PYTHONHASHSEED") == "12345"


def test_configure_sets_pythonhashseed_for_different_seeds() -> None:
    """Different seeds produce different PYTHONHASHSEED values."""
    configure_deterministic_runtime(1)
    assert os.environ.get("PYTHONHASHSEED") == "1"
    configure_deterministic_runtime(999)
    assert os.environ.get("PYTHONHASHSEED") == "999"


def test_assert_returns_correct_dict() -> None:
    """assert_deterministic_runtime returns the expected keys and types."""
    configure_deterministic_runtime(42)
    info = assert_deterministic_runtime(42)
    assert info["seed"] == 42
    assert info["pythonhashseed"] == "42"
    assert isinstance(info["torch_available"], bool)
    assert isinstance(info["numpy_available"], bool)
