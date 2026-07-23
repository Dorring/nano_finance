"""Translate ``EvaluationFeatureFlags`` into RAGEngine runtime configuration.

This module is the evaluation-only composition root that converts the
declarative feature flags defined in :mod:`src.evaluation.schemas` into the
concrete constructor parameters accepted by :class:`src.services.rag_engine.RAGEngine`.

Production code must NEVER read these flags from user input or config files.
Only the evaluation runner injects a non-default flags object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.evaluation.schemas import EvaluationFeatureFlags

if TYPE_CHECKING:
    from src.services.rag_engine import RAGEngine

__all__ = [
    "apply_feature_flags_to_engine_kwargs",
    "build_evaluation_engine",
    "assert_feature_flags_enforced",
]


def apply_feature_flags_to_engine_kwargs(
    flags: EvaluationFeatureFlags,
) -> dict[str, Any]:
    """Convert ``EvaluationFeatureFlags`` into RAGEngine constructor kwargs."""
    return {
        "use_hybrid": flags.dense_enabled and flags.bm25_enabled,
        "enable_calculation_pipeline": flags.calculator_enabled,
        "enable_validation_pipeline": flags.post_validation_enabled
        or flags.answerability_enabled
        or flags.citation_validation_enabled,
        "reranker_name": "default" if flags.reranker_enabled else None,
    }


def build_evaluation_engine(
    llm_client: Any,
    flags: EvaluationFeatureFlags,
    **overrides: Any,
) -> "RAGEngine":
    """Build a RAGEngine with evaluation feature flags applied."""
    from src.services.rag_engine import RAGEngine

    kwargs = apply_feature_flags_to_engine_kwargs(flags)
    kwargs.update(overrides)
    return RAGEngine(llm_client, **kwargs)


def assert_feature_flags_enforced(
    flags: EvaluationFeatureFlags, engine: "RAGEngine"
) -> list[str]:
    """Verify that the engine actually respects the feature flags.

    Returns a list of violation messages (empty if all flags are correctly enforced).
    """
    violations: list[str] = []
    # Check reranker
    if not flags.reranker_enabled and engine._reranker is not None:
        violations.append("reranker_enabled=False but engine has a reranker")
    if flags.reranker_enabled and engine._reranker is None:
        violations.append("reranker_enabled=True but engine has no reranker")
    # Check calculation pipeline
    if not flags.calculator_enabled and getattr(engine, "_orchestrator", None):
        if getattr(engine._orchestrator, "_calculation_pipeline", None) is not None:
            violations.append(
                "calculator_enabled=False but calculation pipeline is configured"
            )
    # Check validation pipeline
    if not flags.post_validation_enabled and getattr(engine, "_orchestrator", None):
        if getattr(engine._orchestrator, "_validation_pipeline", None) is not None:
            violations.append(
                "post_validation_enabled=False but validation pipeline is configured"
            )
    return violations
