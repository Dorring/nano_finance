"""Ablation runner for Phase 5 RAG evaluation.

Defines the ablation variants A0–A9 from the pre-registered protocol,
generates per-variant configs from a base config, validates that each
variant changes exactly one component, and produces a comparison report.

v2 extends each variant with an :class:`EvaluationFeatureFlags` object
that drives runtime flag injection into the RAGEngine. The legacy
``config_diff`` (dict-based, one component per variant) is preserved for
backward compatibility with existing tooling and tests.

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

from src.evaluation.schemas import EvaluationFeatureFlags

__all__ = [
    "ABLATION_VARIANTS",
    "get_ablation_config",
    "get_variant_feature_flags",
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

_VARIANT_BY_ID: dict[str, dict[str, Any]] = {v["id"]: v for v in ABLATION_VARIANTS}

# v2: per-variant EvaluationFeatureFlags that drive runtime injection into
# the RAGEngine. The flags follow the SAME single-component ablation scheme
# as the legacy ``config_diff``: A0 is Full System (all flags True), and
# each of A1-A9 disables exactly ONE component. This guarantees that every
# variant changes exactly one thing relative to A0, so deltas are
# attributable. The ``config_diff`` and ``feature_flags`` for a given
# variant ID describe the SAME ablation in two representations.
#
#   A0  Full System                    (all True)
#   A1  Dense Only                     (bm25_enabled=False)
#   A2  BM25 Only                      (dense_enabled=False)
#   A3  No Reranker                    (reranker_enabled=False)
#   A4  No Query Rewrite               (query_rewrite_enabled=False)
#   A5  No Hierarchical Context        (hierarchical_context_enabled=False)
#   A6  No Calculator                  (calculator_enabled=False)
#   A7  No Answerability Gate          (answerability_enabled=False)
#   A8  No Post-generation Validation  (post_validation_enabled=False)
#   A9  No Citation Validation         (citation_validation_enabled=False)
#
# Note: A8 disables the entire validation pipeline at construction time
# (enable_validation_pipeline=False), so the fine-grained answerability and
# citation flags are also effectively off. A7 and A9 keep the pipeline
# constructed and use runtime stubs to disable only the targeted sub-stage.
_VARIANT_FEATURE_FLAGS: dict[str, EvaluationFeatureFlags] = {
    "A0": EvaluationFeatureFlags(),  # Full System — all defaults True
    "A1": EvaluationFeatureFlags(bm25_enabled=False),
    "A2": EvaluationFeatureFlags(dense_enabled=False),
    "A3": EvaluationFeatureFlags(reranker_enabled=False),
    "A4": EvaluationFeatureFlags(query_rewrite_enabled=False),
    "A5": EvaluationFeatureFlags(hierarchical_context_enabled=False),
    "A6": EvaluationFeatureFlags(calculator_enabled=False),
    "A7": EvaluationFeatureFlags(answerability_enabled=False),
    "A8": EvaluationFeatureFlags(
        post_validation_enabled=False,
        answerability_enabled=False,
        citation_validation_enabled=False,
    ),
    "A9": EvaluationFeatureFlags(citation_validation_enabled=False),
}


def _get_variant(variant_id: str) -> dict[str, Any]:
    """Return the variant definition, raising ``KeyError`` if not found."""
    if variant_id not in _VARIANT_BY_ID:
        raise KeyError(f"unknown ablation variant: {variant_id!r}")
    return _VARIANT_BY_ID[variant_id]


def get_variant_feature_flags(variant_id: str) -> EvaluationFeatureFlags:
    """Return the :class:`EvaluationFeatureFlags` for ``variant_id``.

    Args:
        variant_id: One of ``"A0"`` ... ``"A9"``.

    Raises:
        KeyError: If ``variant_id`` is not a known ablation variant.
    """
    _get_variant(variant_id)
    return _VARIANT_FEATURE_FLAGS[variant_id]


def get_ablation_config(
    base_config: dict[str, Any] | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    """Return an ablation config for ``variant_id``.

    Supports two calling conventions for backward compatibility:

    - **Old-style** ``get_ablation_config(base_config, variant_id)``:
      deep-copies ``base_config`` and applies the variant's legacy
      ``config_diff`` on top. The production default is never mutated.
      Returns the merged dict (no ``feature_flags`` key) so existing
      callers and assertions -- including ``config == base_config`` for
      A0 -- continue to hold.
    - **New-style** ``get_ablation_config(variant_id)``: returns a dict
      carrying the variant metadata (``id``, ``name``, ``config_diff``,
      ``production_safe``) plus a ``feature_flags`` key holding the
      :class:`EvaluationFeatureFlags` for runtime injection.

    Args:
        base_config: The production default configuration (old-style), or
            the ``variant_id`` when called new-style.
        variant_id: One of ``"A0"`` ... ``"A9"``. ``None`` in new-style calls.

    Raises:
        KeyError: If ``variant_id`` is not a known ablation variant.
    """
    if variant_id is None:
        # New-style call: get_ablation_config(variant_id)
        variant_id = base_config  # type: ignore[assignment]
        base_config = None

    variant = _get_variant(variant_id)
    feature_flags = _VARIANT_FEATURE_FLAGS[variant_id]

    if base_config is not None:
        # Old-style: merge legacy config_diff onto a deep copy of base_config.
        new_config = copy.deepcopy(base_config)
        new_config.update(copy.deepcopy(variant["config_diff"]))
        return new_config

    # New-style: return metadata dict with feature_flags.
    return {
        "id": variant["id"],
        "name": variant["name"],
        "config_diff": copy.deepcopy(variant["config_diff"]),
        "production_safe": variant["production_safe"],
        "feature_flags": feature_flags,
    }


def validate_ablation_config(variant_id: str, config: dict[str, Any]) -> list[str]:
    """Validate that ``config`` matches the expected variant diff.

    Checks:
        - ``variant_id`` is a known variant.
        - For A0 (Full System): ``config`` must be empty.
        - For A1-A9: ``config`` must contain exactly one key (only one
          component changed).

    Args:
        variant_id: The variant identifier (``"A0"`` ... ``"A9"``).
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
        results: A dict mapping variant IDs (``"A0"`` ... ``"A9"``) to
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
