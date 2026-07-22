"""Tests for candidate selection rules in src.evaluation.calibration."""
from __future__ import annotations

from src.evaluation.calibration import select_best_candidate

_BASELINE = {
    "macro_strict_pass_rate": 0.5,
    "unsupported_numeric_release_rate": 0.0,
    "invalid_citation_release_rate": 0.0,
    "calculation_mismatch_release_rate": 0.0,
    "false_block_rate": 0.0,
    "unsafe_answer_rate": 0.0,
    "validator_fail_closed_rate": 0.0,
    "citation_recall": 0.5,
    "p95_latency_ms": 100.0,
}


def _candidate(
    cid: str,
    *,
    macro: float = 0.5,
    unsupported: float = 0.0,
    mismatch: float = 0.0,
    invalid_citation: float = 0.0,
    false_block: float = 0.0,
    unsafe: float = 0.0,
    fail_closed: float = 0.0,
    citation_recall: float = 0.5,
    p95: float = 100.0,
) -> dict:
    return {
        "params": {"id": cid},
        "metrics": {
            "macro_strict_pass_rate": macro,
            "unsupported_numeric_release_rate": unsupported,
            "calculation_mismatch_release_rate": mismatch,
            "invalid_citation_release_rate": invalid_citation,
            "false_block_rate": false_block,
            "unsafe_answer_rate": unsafe,
            "validator_fail_closed_rate": fail_closed,
            "citation_recall": citation_recall,
            "p95_latency_ms": p95,
        },
        "safe": True,
        "violations": [],
    }


class TestSafetyWorseEliminated:
    def test_safety_worse_eliminated(self) -> None:
        """A candidate with any safety metric worse than baseline is not selected."""
        candidates = [
            _candidate("safe", macro=0.6),
            _candidate("unsafe", macro=0.99, false_block=0.1),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        assert best["params"]["id"] == "safe"


class TestNewUnsupportedNumericEliminated:
    def test_new_unsupported_numeric_eliminated(self) -> None:
        """New unsupported_numeric_release (baseline 0, candidate > 0) is eliminated."""
        candidates = [
            _candidate("clean", macro=0.6),
            _candidate("trap", macro=0.99, unsupported=0.01),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        assert best["params"]["id"] == "clean"


class TestNewCalculationMismatchEliminated:
    def test_new_calculation_mismatch_eliminated(self) -> None:
        """New calculation_mismatch_release is eliminated."""
        candidates = [
            _candidate("clean", macro=0.6),
            _candidate("mismatch", macro=0.99, mismatch=0.01),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        assert best["params"]["id"] == "clean"


class TestMaximizeMacroStrictPass:
    def test_maximize_macro_strict_pass(self) -> None:
        """Among safe candidates, the highest macro_strict_pass_rate wins."""
        candidates = [
            _candidate("low", macro=0.6),
            _candidate("mid", macro=0.7),
            _candidate("high", macro=0.9),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        assert best["params"]["id"] == "high"
