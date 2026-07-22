"""Ablation runner for Phase 5 RAG evaluation.

Defines the ablation variants A0–A9 from the pre-registered protocol,
generates per-variant configs from a base config, validates that each
variant changes exactly one component, and produces a comparison report.

Safety guarantees:
    - The production default config is NEVER modified by any ablation.
      ``get_ablation_config`` returns a deep copy.
    - Variants that disable validation (A8, A9) are marked
      ``production_safe = False`` and must never become the production
      default.
"""
from __future__ import annotations

import copy
from typing import Any

__all__ = [
    "ABLATION_VARIANTS",
    "get_ablation_config",
    "validate_ablation_config",
    "ablation_report",
    "is_production_safe",
]

ABLATION_VARIANTS: list[dict[str, Any]] = [
    {
        "id": "A0",
        "name": "Full System",
        "config_diff": {},
        "production_safe": True,
    },
    {
        "id": "A1",
        "name": "Dense Only",
        "config_diff": {"disable_bm25": True},
        "production_safe": True,
    },
    {
        "id": "A2",
        "name": "BM25 Only",
        "config_diff": {"disable-dense": True},
        "production_safe": True,
    },
    {
        "id": "A3",
        "name": "No Reranker",
        "config_diff": {"disable-reranker": True},
        "production_safe": True,
    },
    {
        "id": "A4",
        "name": "No Query Rewrite",
        "config_diff": {"disable-query-rewrite": True},
        "production_safe": True,
    },
    {
        "id": "A5",
        "name": "No Hierarchical Context",
        "config_diff": {"disable-hierarchical-context": True},
        "production_safe": True,
    },
    {
        "id": "A6",
        "name": "No Calculator",
        "config_diff": {"disable-calculation-pipeline": True},
        "production_safe": True,
    },
    {
        "id": "A7",
        "name": "No Answerability Gate",
        "config_diff": {"disable-answerability": True},
        "production_safe": True,
    },
    {
        "id": "A8",
        "name": "No Post-generation Validation",
        "config_diff": {"disable-validation-pipeline": True},
        "production_safe": False,
    },
    {
        "id": "A9",
        "name": "No Citation Validation",
        "config_diff": {"disable-citation-validation": True},
        "production_safe": False,
    },
]

_VARIANT_BY_ID: dict[str, dict[str, Any]] = {
    v["id"]: v for v in ABLATION_VARIANTS
}


def _get_variant(variant_id: str) -> dict[str, Any]:
    """Return the variant definition, raising ``KeyError`` if not found."""
    if variant_id not in _VARIANT_BY_ID:
        raise KeyError(f"unknown ablation variant: {variant_id!r}")
    return _VARIANT_BY_ID[variant_id]


def get_ablation_config(
    base_config: dict[str, Any], variant_id: str
) -> dict[str, Any]:
    """Return a new config with the variant's ``config_diff`` applied.

    The ``base_config`` is deep-copied so the caller's production default
    is never modified.

    Args:
        base_config: The production default configuration.
        variant_id: One of ``"A0"`` … ``"A9"``.

    Returns:
        A new dict with the variant's overrides applied on top of a deep
        copy of ``base_config``.

    Raises:
        KeyError: If ``variant_id`` is not a known ablation variant.
    """
    variant = _get_variant(variant_id)
    new_config = copy.deepcopy(base_config)
    new_config.update(copy.deepcopy(variant["config_diff"]))
    return new_config


def validate_ablation_config(
    variant_id: str, config: dict[str, Any]
) -> list[str]:
    """Validate that ``config`` matches the expected variant diff.

    Checks:
        - ``variant_id`` is a known variant.
        - For A0 (Full System): ``config`` must be empty.
        - For A1–A9: ``config`` must contain exactly one key (only one
          component changed).

    Args:
        variant_id: The variant identifier (``"A0"`` … ``"A9"``).
        config: The config diff to validate.

    Returns:
        A list of error strings. An empty list means the config is valid.
    """
    errors: list[str] = []
    if variant_id not in _VARIANT_BY_ID:
        errors.append(f"unknown variant id: {variant_id!r}")
        return errors
    variant = _VARIANT_BY_ID[variant_id]
    expected = variant["config_diff"]
    expected_keys = set(expected.keys())
    actual_keys = set(config.keys())
    if actual_keys != expected_keys:
        errors.append(
            f"variant {variant_id} config keys mismatch: "
            f"expected {sorted(expected_keys)}, got {sorted(actual_keys)}"
        )
    if variant_id != "A0" and len(actual_keys) != 1:
        errors.append(
            f"variant {variant_id} must change exactly one component, "
            f"got {len(actual_keys)}"
        )
    return errors


def is_production_safe(variant_id: str) -> bool:
    """Return True if the variant is safe to use as a production default."""
    variant = _get_variant(variant_id)
    return bool(variant.get("production_safe", True))


def ablation_report(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Generate a comparison report across ablation variants.

    Args:
        results: A dict mapping variant IDs (``"A0"`` … ``"A9"``) to
            their metric dicts.

    Returns:
        A report dict with:
            - ``"variants"``: ordered list of variant summaries.
            - ``"baseline_id"``: always ``"A0"``.
            - ``"deltas"``: per-variant delta from A0 baseline.
            - ``"production_safe"``: per-variant safety flag.
    """
    baseline = results.get("A0", {})
    baseline_macro = float(baseline.get("macro_strict_pass_rate", 0.0))
    variants: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, float]] = {}
    production_safe: dict[str, bool] = {}
    for variant in ABLATION_VARIANTS:
        vid = variant["id"]
        metrics = results.get(vid, {})
        macro = float(metrics.get("macro_strict_pass_rate", 0.0))
        delta = macro - baseline_macro
        variants.append(
            {
                "id": vid,
                "name": variant["name"],
                "macro_strict_pass_rate": macro,
                "delta_from_baseline": delta,
            }
        )
        deltas[vid] = {"delta_macro_strict_pass_rate": delta}
        production_safe[vid] = is_production_safe(vid)
    return {
        "baseline_id": "A0",
        "variants": variants,
        "deltas": deltas,
        "production_safe": production_safe,
    }
