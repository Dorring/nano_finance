"""Deterministic runtime configuration for Phase 5 evaluation.

Seeds all stochastic sources (Python ``random``, NumPy, PyTorch, and
``PYTHONHASHSEED``) so that evaluation runs are reproducible. This module
does NOT enable ``torch.use_deterministic_algorithms`` — that decision is
left to the caller because it can raise errors for unsupported operations.
"""

from __future__ import annotations

import os
import random
from typing import Any

__all__ = [
    "configure_deterministic_runtime",
    "assert_deterministic_runtime",
]


def configure_deterministic_runtime(seed: int) -> None:
    """Seed all random sources for deterministic evaluation runs.

    Sets ``PYTHONHASHSEED``, the stdlib ``random`` module, and (when
    available) NumPy and PyTorch RNGs. PyTorch's
    ``use_deterministic_algorithms`` is intentionally NOT toggled here —
    the caller should set it separately if needed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def assert_deterministic_runtime(seed: int) -> dict[str, Any]:
    """Verify that deterministic runtime configuration was applied.

    Returns a dict with ``seed``, ``pythonhashseed``,
    ``torch_available``, and ``numpy_available`` so callers can confirm
    the runtime state before a sealed run.
    """
    torch_available = False
    numpy_available = False
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        pass
    try:
        import numpy  # noqa: F401

        numpy_available = True
    except ImportError:
        pass
    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "torch_available": torch_available,
        "numpy_available": numpy_available,
    }
