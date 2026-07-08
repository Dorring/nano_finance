"""Small, dependency-free helpers for durable training metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


def save_best_pointer(checkpoint_dir: str | Path, step: int, val_bpb: float) -> Path:
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "best.json"
    temporary = directory / "best.json.tmp"
    payload = {
        "step": step,
        "val_bpb": val_bpb,
        "checkpoint": f"model_{step:06d}.pt",
    }
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def scale_initial_learning_rates(
    param_groups: Iterable[dict],
    scale: float,
) -> None:
    if scale <= 0:
        raise ValueError("learning-rate scale must be positive")
    for group in param_groups:
        if "initial_lr" not in group:
            raise ValueError("resumed optimizer is missing initial_lr")
        group["initial_lr"] *= scale
        group["lr"] = group["initial_lr"]
