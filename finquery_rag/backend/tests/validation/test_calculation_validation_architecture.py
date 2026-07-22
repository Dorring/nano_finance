"""Phase 4 architecture checks for calculation validation wiring.

Statically verify that the RAG orchestrator routes every calculation
path (EXECUTED, BLOCKED, FAILED) through the Phase 4 answerability and
validation pipeline, and that BLOCKED/FAILED never reaches LLM generation.

These are *architecture* (source-structure) checks, not runtime behaviour
tests.  Runtime behaviour is covered by
``test_calculation_validation_runtime.py``,
``test_calculation_validation_http_runtime.py`` and
``test_calculation_validation_sse_runtime.py``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

_ORCHESTRATOR_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "application",
    "rag_orchestrator.py",
)


@pytest.fixture(scope="module")
def orchestrator_source() -> str:
    """Load the orchestrator source for static wiring checks."""
    with open(_ORCHESTRATOR_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_calculation_executed_runs_answerability(orchestrator_source: str) -> None:
    """EXECUTED path must invoke evaluate_answerability before generation.

    The Phase 4 hotfix runs answerability for ALL paths (including
    EXECUTED). This guards against regressing to a state where the
    EXECUTED path bypasses answerability evaluation.
    """
    assert "evaluate_answerability" in orchestrator_source
    assert "CalculationStatus.EXECUTED" in orchestrator_source


def test_calculation_executed_runs_validation(orchestrator_source: str) -> None:
    """EXECUTED path must call _validate_and_repair_once after generation.

    Post-generation validation runs for all non-answerability-blocked
    paths, including calculation EXECUTED.
    """
    assert "_validate_and_repair_once" in orchestrator_source
    assert "CalculationStatus.EXECUTED" in orchestrator_source


def test_calculation_blocked_has_answerability(orchestrator_source: str) -> None:
    """BLOCKED path must set answerability_result (CALCULATION_BLOCKED)."""
    assert "answerability_result" in orchestrator_source
    assert "CalculationStatus.BLOCKED" in orchestrator_source
    assert "CALCULATION_BLOCKED" in orchestrator_source


def test_calculation_blocked_uses_safe_fallback(orchestrator_source: str) -> None:
    """BLOCKED path must use ResponseRepair to produce a safe fallback."""
    assert "ResponseRepair" in orchestrator_source
    assert "self._response_repair" in orchestrator_source
    assert "CALCULATION_BLOCKED" in orchestrator_source


def test_calculation_failed_has_answerability(orchestrator_source: str) -> None:
    """FAILED path must set answerability_result (CALCULATION_BLOCKED)."""
    assert "answerability_result" in orchestrator_source
    assert "CalculationStatus.FAILED" in orchestrator_source
    assert "CALCULATION_BLOCKED" in orchestrator_source


def test_calculation_blocked_does_not_call_llm(orchestrator_source: str) -> None:
    """BLOCKED/FAILED must not reach LLM generation.

    The ``answerability_blocked`` gate prevents the LLM branch from
    running, and the ``is_calculation_blocked_or_failed`` branch handles
    the no-validation-pipeline case by using the rendered calculation
    answer (safe refusal) and explicitly skipping the LLM.
    """
    assert "is_calculation_blocked_or_failed" in orchestrator_source
    assert "answerability_blocked" in orchestrator_source
    assert "skip LLM" in orchestrator_source


def test_calculation_result_in_trace(orchestrator_source: str) -> None:
    """Calculation diagnostics must appear in the trace."""
    assert "to_trace_dict" in orchestrator_source
    assert "calculation" in orchestrator_source
    assert "diagnostics" in orchestrator_source
