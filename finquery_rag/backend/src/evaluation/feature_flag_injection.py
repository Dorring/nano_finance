"""Translate ``EvaluationFeatureFlags`` into RAGEngine runtime configuration.

This module is the evaluation-only composition root that converts the
declarative feature flags defined in :mod:`src.evaluation.schemas` into
concrete constructor parameters AND runtime patches on
:class:`src.services.rag_engine.RAGEngine`.

Two-phase application:

1. **Constructor phase** — flags that map directly to RAGEngine
   ``__init__`` kwargs (``use_hybrid``, ``reranker_name``,
   ``enable_calculation_pipeline``, ``enable_validation_pipeline``).

2. **Runtime patch phase** — flags that have no constructor equivalent
   are applied by patching internal attributes after construction:

   - ``dense_enabled=False``       → patch ``_retrieval_pipeline._dense_query_fn``
   - ``query_rewrite_enabled=False`` → patch ``_llm_gateway.rewrite_query``
   - ``hierarchical_context_enabled=False`` → patch ``_context_builder._merge_parent_context_chunks``
   - ``answerability_enabled=False`` (fine-grained) → stub ``_validation_pipeline._answerability``
   - ``citation_validation_enabled=False`` (fine-grained) → stub ``_validation_pipeline._response_validator._citation_validator``

Production code must NEVER read these flags from user input or config files.
Only the evaluation runner injects a non-default flags object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.evaluation.schemas import EvaluationFeatureFlags

if TYPE_CHECKING:
    from src.services.rag_engine import RAGEngine

__all__ = [
    "FeatureFlagApplicationRecord",
    "apply_feature_flags_to_engine_kwargs",
    "apply_feature_flags_runtime",
    "build_evaluation_engine",
    "assert_feature_flags_enforced",
]


# ---------------------------------------------------------------------------
# Sentinel stubs for runtime patching
# ---------------------------------------------------------------------------


class _PatchedDenseQueryFn:
    """Sentinel: dense retrieval disabled for evaluation ablation.

    Returns an empty list so RRF fusion only uses BM25 results.
    """

    _eval_patch = "disabled_dense"

    def __call__(self, **_kwargs: Any) -> list:
        return []


class _NoOpCitationValidator:
    """Stub: never reports citation issues.

    Used when ``citation_validation_enabled=False`` but other post-generation
    validation (numeric, calculation, unit/period) must still run.
    """

    _eval_patch = "disabled_citation"

    def validate(self, **_kwargs: Any) -> tuple:
        return ()


class _AlwaysAnswerable:
    """Stub: always returns ANSWERABLE.

    Used when ``answerability_enabled=False`` but post-generation validation
    must still run.
    """

    _eval_patch = "disabled_answerability"

    def evaluate(self, **kwargs: Any) -> Any:
        from src.domain.validation import AnswerabilityResult, AnswerabilityStatus

        evidence = kwargs.get("evidence", ())
        docs = kwargs.get("requested_documents", ())
        return AnswerabilityResult(
            status=AnswerabilityStatus.ANSWERABLE,
            reason_codes=(),
            evidence_count=len(evidence) if evidence else 0,
            document_count=len(docs) if docs else 0,
            best_score=None,
            average_score=None,
            missing_requirements=(),
        )


async def _no_rewrite(question: str, *_args: Any, **_kwargs: Any) -> str:
    """Belt-and-suspenders: skip query rewrite even if conversation_history is non-empty."""
    return question


# ---------------------------------------------------------------------------
# Application record
# ---------------------------------------------------------------------------


@dataclass
class FeatureFlagApplicationRecord:
    """Records how each feature flag was applied to the engine."""

    constructor_kwargs: dict[str, Any] = field(default_factory=dict)
    runtime_patches: dict[str, str] = field(default_factory=dict)
    noops: dict[str, str] = field(default_factory=dict)

    @property
    def all_flags_addressed(self) -> bool:
        """True if all 9 flags have an entry in some category."""
        seen: set[str] = set()
        seen.update(self.runtime_patches.keys())
        seen.update(self.noops.keys())
        # Constructor kwargs address flags implicitly
        return len(seen) >= 5  # At least the 5 non-constructor flags


# ---------------------------------------------------------------------------
# Phase 1: Constructor kwargs
# ---------------------------------------------------------------------------


def apply_feature_flags_to_engine_kwargs(
    flags: EvaluationFeatureFlags,
) -> dict[str, Any]:
    """Convert ``EvaluationFeatureFlags`` into RAGEngine constructor kwargs.

    Mapping:

    - ``use_hybrid`` = ``bm25_enabled`` (True → hybrid mode calls BM25;
      False → dense-only branch)
    - ``enable_calculation_pipeline`` = ``calculator_enabled``
    - ``enable_validation_pipeline`` = ``post_validation_enabled OR
      answerability_enabled OR citation_validation_enabled`` (any validation
      sub-flag being True keeps the pipeline constructed; fine-grained
      disabling is handled by runtime patches)
    - ``reranker_name`` = ``"heuristic"`` if ``reranker_enabled`` else ``None``
    """
    enable_validation = (
        flags.post_validation_enabled
        or flags.answerability_enabled
        or flags.citation_validation_enabled
    )
    return {
        "use_hybrid": flags.bm25_enabled,
        "enable_calculation_pipeline": flags.calculator_enabled,
        "enable_validation_pipeline": enable_validation,
        "reranker_name": "heuristic" if flags.reranker_enabled else None,
    }


# ---------------------------------------------------------------------------
# Phase 2: Runtime patches
# ---------------------------------------------------------------------------


def apply_feature_flags_runtime(
    engine: "RAGEngine",
    flags: EvaluationFeatureFlags,
) -> FeatureFlagApplicationRecord:
    """Apply feature flags that cannot be set via constructor kwargs.

    Patches internal engine attributes after construction. Each patch is
    recorded in the returned :class:`FeatureFlagApplicationRecord`.

    Flags handled here:

    - ``dense_enabled=False`` → patch ``_retrieval_pipeline._dense_query_fn``
      to return ``[]`` (BM25-only mode via RRF with empty dense)
    - ``query_rewrite_enabled=False`` → patch ``_llm_gateway.rewrite_query``
      (belt-and-suspenders; blind runner already passes empty history)
    - ``hierarchical_context_enabled=False`` → patch
      ``_context_builder._merge_parent_context_chunks`` to identity
    - ``answerability_enabled=False`` (when pipeline still constructed) →
      stub ``_validation_pipeline._answerability``
    - ``citation_validation_enabled=False`` (when pipeline still constructed) →
      stub ``_validation_pipeline._response_validator._citation_validator``
    """
    record = FeatureFlagApplicationRecord()
    record.constructor_kwargs = apply_feature_flags_to_engine_kwargs(flags)

    # --- dense_enabled ---
    if not flags.dense_enabled:
        pipeline = getattr(engine, "_retrieval_pipeline", None)
        if pipeline is not None:
            pipeline._dense_query_fn = _PatchedDenseQueryFn()
            record.runtime_patches["dense_enabled"] = (
                "patched _retrieval_pipeline._dense_query_fn -> _PatchedDenseQueryFn"
            )
        else:
            record.noops["dense_enabled"] = "no _retrieval_pipeline attribute"
    else:
        record.noops["dense_enabled"] = "enabled (constructor default)"

    # --- query_rewrite_enabled ---
    if not flags.query_rewrite_enabled:
        gateway = getattr(engine, "_llm_gateway", None)
        if gateway is not None:
            gateway.rewrite_query = _no_rewrite  # type: ignore[assignment]
            record.runtime_patches["query_rewrite_enabled"] = (
                "patched _llm_gateway.rewrite_query -> _no_rewrite "
                "(belt-and-suspenders; blind runner passes empty history)"
            )
        else:
            record.noops["query_rewrite_enabled"] = "no _llm_gateway attribute"
    else:
        record.noops["query_rewrite_enabled"] = "enabled (constructor default)"

    # --- hierarchical_context_enabled ---
    if not flags.hierarchical_context_enabled:
        builder = getattr(engine, "_context_builder", None)
        if builder is not None:
            builder._merge_parent_context_chunks = lambda chunks: chunks  # type: ignore[assignment]
            record.runtime_patches["hierarchical_context_enabled"] = (
                "patched _context_builder._merge_parent_context_chunks -> identity"
            )
        else:
            record.noops["hierarchical_context_enabled"] = "no _context_builder attribute"
    else:
        record.noops["hierarchical_context_enabled"] = "enabled (constructor default)"

    # --- answerability_enabled (fine-grained) ---
    # Only patch if validation pipeline is constructed (i.e., some validation
    # flag is True) but answerability specifically is False.
    validation_pipeline = getattr(engine, "_validation_pipeline", None)
    if not flags.answerability_enabled and validation_pipeline is not None:
        validation_pipeline._answerability = _AlwaysAnswerable()
        record.runtime_patches["answerability_enabled"] = (
            "stubbed _validation_pipeline._answerability -> _AlwaysAnswerable"
        )
    elif not flags.answerability_enabled and validation_pipeline is None:
        record.noops["answerability_enabled"] = (
            "entire validation pipeline is None (constructor disabled)"
        )
    else:
        record.noops["answerability_enabled"] = "enabled (constructor default)"

    # --- citation_validation_enabled (fine-grained) ---
    # Only patch if validation pipeline is constructed and post-validation
    # is still enabled (otherwise the whole pipeline is None).
    if (
        not flags.citation_validation_enabled
        and validation_pipeline is not None
        and flags.post_validation_enabled
    ):
        rv = getattr(validation_pipeline, "_response_validator", None)
        if rv is not None:
            rv._citation_validator = _NoOpCitationValidator()
            record.runtime_patches["citation_validation_enabled"] = (
                "stubbed _validation_pipeline._response_validator._citation_validator "
                "-> _NoOpCitationValidator"
            )
        else:
            record.noops["citation_validation_enabled"] = "no _response_validator"
    elif not flags.citation_validation_enabled and validation_pipeline is None:
        record.noops["citation_validation_enabled"] = (
            "entire validation pipeline is None (constructor disabled)"
        )
    elif not flags.citation_validation_enabled and not flags.post_validation_enabled:
        record.noops["citation_validation_enabled"] = (
            "post_validation_enabled=False; entire response validation skipped"
        )
    else:
        record.noops["citation_validation_enabled"] = "enabled (constructor default)"

    return record


# ---------------------------------------------------------------------------
# Combined builder
# ---------------------------------------------------------------------------


def build_evaluation_engine(
    llm_client: Any,
    flags: EvaluationFeatureFlags,
    **overrides: Any,
) -> tuple["RAGEngine", FeatureFlagApplicationRecord]:
    """Build a RAGEngine with evaluation feature flags fully applied.

    Returns a tuple of ``(engine, application_record)``. The record
    documents exactly which flags were applied via constructor, which via
    runtime patch, and which were no-ops.
    """
    from src.services.rag_engine import RAGEngine

    kwargs = apply_feature_flags_to_engine_kwargs(flags)
    kwargs.update(overrides)
    engine = RAGEngine(llm_client, **kwargs)
    record = apply_feature_flags_runtime(engine, flags)
    return engine, record


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def assert_feature_flags_enforced(
    flags: EvaluationFeatureFlags, engine: "RAGEngine"
) -> list[str]:
    """Verify that the engine actually respects all 9 feature flags.

    Returns a list of violation messages (empty if all flags are correctly
    enforced).  This function checks both constructor-level and
    runtime-patch-level enforcement.
    """
    violations: list[str] = []

    # --- bm25_enabled (constructor: use_hybrid) ---
    if not flags.bm25_enabled and engine.use_hybrid:
        violations.append("bm25_enabled=False but engine.use_hybrid=True")
    if flags.bm25_enabled and not engine.use_hybrid:
        violations.append("bm25_enabled=True but engine.use_hybrid=False")

    # --- dense_enabled (runtime patch) ---
    if not flags.dense_enabled:
        pipeline = getattr(engine, "_retrieval_pipeline", None)
        fn = getattr(pipeline, "_dense_query_fn", None) if pipeline else None
        if not hasattr(fn, "_eval_patch") or fn._eval_patch != "disabled_dense":
            violations.append(
                "dense_enabled=False but _dense_query_fn not patched to disabled"
            )

    # --- reranker_enabled (constructor: reranker_name) ---
    if not flags.reranker_enabled and engine.reranker is not None:
        violations.append("reranker_enabled=False but engine has a reranker")
    if flags.reranker_enabled and engine.reranker is None:
        violations.append("reranker_enabled=True but engine has no reranker")

    # --- query_rewrite_enabled (runtime patch) ---
    if not flags.query_rewrite_enabled:
        gateway = getattr(engine, "_llm_gateway", None)
        fn = getattr(gateway, "rewrite_query", None) if gateway else None
        if fn is not _no_rewrite:
            violations.append(
                "query_rewrite_enabled=False but _llm_gateway.rewrite_query not patched"
            )

    # --- hierarchical_context_enabled (runtime patch) ---
    if not flags.hierarchical_context_enabled:
        builder = getattr(engine, "_context_builder", None)
        merge_fn = getattr(builder, "_merge_parent_context_chunks", None) if builder else None
        # The patched version is a lambda; check it's not the original method
        if merge_fn is not None and hasattr(merge_fn, "__func__"):
            violations.append(
                "hierarchical_context_enabled=False but _merge_parent_context_chunks "
                "still appears to be the original method"
            )

    # --- calculator_enabled (constructor: enable_calculation_pipeline) ---
    if not flags.calculator_enabled:
        calc = getattr(engine, "_calculation_pipeline", None)
        if calc is not None:
            violations.append(
                "calculator_enabled=False but calculation pipeline is configured"
            )

    # --- post_validation_enabled (constructor: enable_validation_pipeline) ---
    if not flags.post_validation_enabled:
        vp = getattr(engine, "_validation_pipeline", None)
        if vp is not None:
            violations.append(
                "post_validation_enabled=False but validation pipeline is configured"
            )

    # --- answerability_enabled (runtime patch or constructor) ---
    if not flags.answerability_enabled:
        vp = getattr(engine, "_validation_pipeline", None)
        if vp is not None:
            ans = getattr(vp, "_answerability", None)
            if not hasattr(ans, "_eval_patch") or ans._eval_patch != "disabled_answerability":
                violations.append(
                    "answerability_enabled=False but _answerability not stubbed"
                )

    # --- citation_validation_enabled (runtime patch or constructor) ---
    if not flags.citation_validation_enabled:
        vp = getattr(engine, "_validation_pipeline", None)
        if vp is not None and flags.post_validation_enabled:
            rv = getattr(vp, "_response_validator", None)
            cv = getattr(rv, "_citation_validator", None) if rv else None
            if not hasattr(cv, "_eval_patch") or cv._eval_patch != "disabled_citation":
                violations.append(
                    "citation_validation_enabled=False but _citation_validator not stubbed"
                )

    return violations
