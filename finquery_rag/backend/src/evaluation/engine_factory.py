"""Unified evaluation engine factory.

This module is the single entry point for building RAG engines in Phase 5
evaluation scripts. It handles:

1. **Partition index setup** — sets ``CHROMA_PATH`` and ``BM25_DB_PATH``
   env vars for the correct partition (dev/calibration/sealed) and resets
   the ChromaDB client singleton.

2. **Calibration param injection** — truly injects the 5 runtime-applicable
   calibration params into the engine:

   - ``max_context_tokens`` → RAGEngine constructor param
   - ``min_score_threshold`` → runtime-synced via ``engine.min_score_threshold``
   - ``numeric_dense_floor`` → patched via ``engine._orchestrator._numeric_dense_floor``
   - ``numeric_rrf_floor`` → patched via ``engine._orchestrator._numeric_rrf_floor``
   - ``rrf_sufficiency_threshold`` → patched via ``engine._sufficiency_evaluator._rrf_threshold``
   - ``dense_sufficiency_threshold`` → patched via ``engine._sufficiency_evaluator._dense_threshold``

   ``n_results`` is NOT an engine param; it is passed to ``query()`` per
   call by the blind runner. The factory extracts and returns it.

3. **Feature flag injection** — delegates to
   :mod:`src.evaluation.feature_flag_injection` for all 9 ablation flags.

4. **Sentinel query verification** — runs a known query against the engine
   to verify the index is correctly wired and retrieval returns results.
   Fails fast if 0 results are returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.feature_flag_injection import (
    FeatureFlagApplicationRecord,
    apply_feature_flags_runtime,
    apply_feature_flags_to_engine_kwargs,
)
from src.evaluation.schemas import EvaluationFeatureFlags

__all__ = [
    "CalibrationParamRecord",
    "EngineApplicationRecord",
    "PARTITION_USER_IDS",
    "setup_partition_index",
    "apply_calibration_params_runtime",
    "build_evaluation_engine",
    "verify_sentinel_query",
]


# ---------------------------------------------------------------------------
# Partition configuration
# ---------------------------------------------------------------------------

PARTITION_USER_IDS: dict[str, int] = {
    "dev": 9001,
    "calibration": 9002,
    "sealed": 9003,
}

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class CalibrationParamRecord:
    """Records how each calibration param was applied."""

    applied: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    n_results: int | None = None


@dataclass
class EngineApplicationRecord:
    """Combined record for all engine configuration."""

    partition: str = ""
    partition_user_id: int = 0
    calibration: CalibrationParamRecord = field(default_factory=CalibrationParamRecord)
    feature_flags: FeatureFlagApplicationRecord = field(
        default_factory=FeatureFlagApplicationRecord
    )
    sentinel_query_passed: bool = False
    sentinel_query_result_count: int = 0


# ---------------------------------------------------------------------------
# Partition index setup
# ---------------------------------------------------------------------------


def setup_partition_index(partition: str, backend_dir: Path | None = None) -> int:
    """Set up ChromaDB and BM25 env vars for a partition.

    Sets ``CHROMA_PATH`` and ``BM25_DB_PATH`` to point at the
    partition-specific index directory, and resets the ChromaDB client
    singleton so the new path takes effect.

    Returns the user_id for the partition.
    """
    if partition not in PARTITION_USER_IDS:
        raise ValueError(
            f"Invalid partition '{partition}'. "
            f"Must be one of: {list(PARTITION_USER_IDS.keys())}"
        )

    base = backend_dir or BACKEND_DIR
    index_dir = base / "indexes" / "phase5" / partition
    chroma_path = index_dir / "chroma"
    bm25_path = index_dir / "rag_bm25.db"

    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Partition index not found: {chroma_path}. "
            f"Run the index build script first."
        )

    os.environ["CHROMA_PATH"] = str(chroma_path)
    os.environ["BM25_DB_PATH"] = str(bm25_path)

    # Reset ChromaDB client singleton so the new path takes effect
    import src.services.vector_store as vs

    vs._chroma_client = None

    return PARTITION_USER_IDS[partition]


# ---------------------------------------------------------------------------
# Calibration param injection
# ---------------------------------------------------------------------------

# The set of calibration params that can be truly applied at runtime.
# Params NOT in this set (e.g., document_coverage_threshold,
# sufficiency_best_score_threshold, sufficiency_average_score_threshold)
# do not exist at runtime and must not appear in the calibration search space.
RUNTIME_CALIBRATION_PARAMS: frozenset[str] = frozenset({
    "n_results",
    "max_context_tokens",
    "min_score_threshold",
    "numeric_dense_floor",
    "numeric_rrf_floor",
    "rrf_sufficiency_threshold",
    "dense_sufficiency_threshold",
})


def apply_calibration_params_runtime(
    engine: Any,
    params: dict[str, Any],
) -> CalibrationParamRecord:
    """Apply calibration params to the engine at runtime.

    Only params in :data:`RUNTIME_CALIBRATION_PARAMS` are applied. Any
    other params are recorded as skipped with a reason.

    ``n_results`` is extracted but NOT applied to the engine — it must be
    passed to ``query()`` per call by the caller. The value is recorded in
    ``CalibrationParamRecord.n_results``.
    """
    record = CalibrationParamRecord()

    for key, value in params.items():
        if key not in RUNTIME_CALIBRATION_PARAMS:
            record.skipped[key] = (
                f"not a runtime-applicable param (value={value})"
            )
            continue

        if key == "n_results":
            record.n_results = int(value)
            record.applied[key] = "extracted for per-call query() usage"
            continue

        if key == "max_context_tokens":
            engine.max_context_tokens = int(value)
            record.applied[key] = f"set engine.max_context_tokens={value}"
            continue

        if key == "min_score_threshold":
            engine.min_score_threshold = float(value)
            # build_context() will sync this to _context_builder
            record.applied[key] = f"set engine.min_score_threshold={value}"
            continue

        if key == "numeric_dense_floor":
            orch = getattr(engine, "_orchestrator", None)
            if orch is not None:
                orch._numeric_dense_floor = float(value)
                record.applied[key] = (
                    f"patched engine._orchestrator._numeric_dense_floor={value}"
                )
            else:
                record.skipped[key] = "no _orchestrator attribute"
            continue

        if key == "numeric_rrf_floor":
            orch = getattr(engine, "_orchestrator", None)
            if orch is not None:
                orch._numeric_rrf_floor = float(value)
                record.applied[key] = (
                    f"patched engine._orchestrator._numeric_rrf_floor={value}"
                )
            else:
                record.skipped[key] = "no _orchestrator attribute"
            continue

        if key == "rrf_sufficiency_threshold":
            evaluator = getattr(engine, "_sufficiency_evaluator", None)
            if evaluator is not None:
                evaluator._rrf_threshold = float(value)
                record.applied[key] = (
                    f"patched engine._sufficiency_evaluator._rrf_threshold={value}"
                )
            else:
                record.skipped[key] = "no _sufficiency_evaluator attribute"
            continue

        if key == "dense_sufficiency_threshold":
            evaluator = getattr(engine, "_sufficiency_evaluator", None)
            if evaluator is not None:
                evaluator._dense_threshold = float(value)
                record.applied[key] = (
                    f"patched engine._sufficiency_evaluator._dense_threshold={value}"
                )
            else:
                record.skipped[key] = "no _sufficiency_evaluator attribute"
            continue

    return record


# ---------------------------------------------------------------------------
# Unified factory
# ---------------------------------------------------------------------------


def build_evaluation_engine(
    llm_client: Any,
    *,
    partition: str,
    calibration_params: dict[str, Any] | None = None,
    feature_flags: EvaluationFeatureFlags | None = None,
    model_name: str = "finquery-finance-sft1147",
    backend_dir: Path | None = None,
    run_sentinel: bool = True,
) -> tuple[Any, EngineApplicationRecord]:
    """Build a RAG engine for evaluation with full param/flag injection.

    This is the SINGLE entry point for all evaluation scripts. It:

    1. Sets up partition-specific index env vars via :func:`setup_partition_index`
    2. Builds RAGEngine with constructor params (from feature flags +
       ``max_context_tokens`` from calibration params)
    3. Applies remaining calibration params via runtime patching
    4. Applies feature flags via runtime patching
    5. Optionally runs a sentinel query to verify index wiring

    Returns ``(engine, application_record)``. The record documents every
    param and flag application for auditability.
    """
    from src.services.rag_engine import RAGEngine

    record = EngineApplicationRecord()

    # 1. Partition index setup
    record.partition = partition
    record.partition_user_id = setup_partition_index(partition, backend_dir)

    # 2. Build constructor kwargs
    constructor_kwargs: dict[str, Any] = {"model_name": model_name}

    # Feature flags → constructor kwargs
    if feature_flags is not None:
        flag_kwargs = apply_feature_flags_to_engine_kwargs(feature_flags)
        constructor_kwargs.update(flag_kwargs)
    else:
        feature_flags = EvaluationFeatureFlags()  # all-True default

    # Calibration params → constructor kwargs (only max_context_tokens)
    cal_params = calibration_params or {}
    if "max_context_tokens" in cal_params:
        constructor_kwargs["max_context_tokens"] = int(cal_params["max_context_tokens"])

    # 3. Build engine
    engine = RAGEngine(llm_client, **constructor_kwargs)

    # 4. Apply calibration params via runtime patching
    record.calibration = apply_calibration_params_runtime(engine, cal_params)

    # 5. Apply feature flags via runtime patching
    record.feature_flags = apply_feature_flags_runtime(engine, feature_flags)

    # 6. Sentinel query verification
    if run_sentinel:
        record.sentinel_query_passed, record.sentinel_query_result_count = (
            verify_sentinel_query(engine, record.partition_user_id)
        )
        if not record.sentinel_query_passed:
            raise RuntimeError(
                f"Sentinel query returned {record.sentinel_query_result_count} "
                f"results. Index wiring failure for partition '{partition}'. "
                f"Check that indexes/phase5/{partition}/chroma contains the "
                f"correct collection and that user_id={record.partition_user_id} "
                f"matches the index build."
            )

    return engine, record


# ---------------------------------------------------------------------------
# Sentinel query verification
# ---------------------------------------------------------------------------

# A simple query that should always return results if the index is correctly
# wired. The query is generic enough to match any financial document in the
# evaluation corpus.
_SENTINEL_QUERY = "公司财务报告"
_SENTINEL_DOC_NAMES = None  # No doc filter; search all docs
_SENTINEL_N_RESULTS = 3


def verify_sentinel_query(
    engine: Any,
    user_id: int,
) -> tuple[bool, int]:
    """Run a sentinel query to verify the index is correctly wired.

    Returns ``(passed, result_count)``. Fails (returns ``False``) if the
    query returns 0 results, indicating an index wiring failure.
    """
    try:
        import asyncio

        async def _run() -> list:
            raw = engine.query(
                question=_SENTINEL_QUERY,
                doc_names=_SENTINEL_DOC_NAMES,
                user_id=user_id,
                n_results=_SENTINEL_N_RESULTS,
                conversation_history=[],
                memory_profile=None,
            )
            if hasattr(raw, "__awaitable__"):
                raw = await raw
            if isinstance(raw, dict):
                sources = raw.get("sources", [])
                chunks = raw.get("retrieved_chunks", [])
                return chunks if chunks else sources
            return []

        result = asyncio.run(_run())
        count = len(result) if result else 0
        return (count > 0, count)
    except Exception:
        return (False, 0)
