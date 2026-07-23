"""Tests for candidate selection rules in src.evaluation.calibration.

v2 semantics: safety *release* metrics are the safety gate.
``false_block_rate`` is a utility metric, not a safety release violation.
When safe_candidates == 0, the function returns baseline (never selects
from unsafe candidates).
"""
from __future__ import annotations

from src.evaluation.calibration import select_best_candidate

_BASELINE = {
    "macro_strict_pass_rate": 0.5,
    "unsupported_numeric_release_rate": 0.0,
    "invalid_citation_release_rate": 0.0,
    "calculation_mismatch_release_rate": 0.0,
    "unsafe_content_release_rate": 0.0,
    "false_block_rate": 0.0,
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
    unsafe_content: float = 0.0,
    false_block: float = 0.0,
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
            "unsafe_content_release_rate": unsafe_content,
            "false_block_rate": false_block,
            "citation_recall": citation_recall,
            "p95_latency_ms": p95,
        },
        "safe": True,
        "violations": [],
    }


class TestSafetyWorseEliminated:
    def test_safety_worse_eliminated(self) -> None:
        """A candidate with any safety release metric worse than baseline is not selected."""
        candidates = [
            _candidate("safe", macro=0.6),
            _candidate("unsafe", macro=0.99, unsafe_content=0.1),
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


class TestNoSafeCandidateReturnsBaseline:
    def test_no_safe_candidate_returns_baseline(self) -> None:
        """When all candidates are unsafe, return baseline (never select unsafe)."""
        candidates = [
            _candidate("bad1", macro=0.99, unsafe_content=0.1),
            _candidate("bad2", macro=0.99, unsupported=0.01),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        assert best["status"] == "no_safe_candidate"
        assert best["selected_config"] == "baseline"
        assert best["tuning_applied"] is False
        assert best["safe_candidates_count"] == 0


class TestEmptyCandidatesReturnsBaseline:
    def test_empty_candidates_returns_baseline(self) -> None:
        """When no candidates at all, return baseline."""
        best = select_best_candidate([], _BASELINE)
        assert best["status"] == "no_safe_candidate"
        assert best["selected_config"] == "baseline"


class TestFalseBlockIsUtilityNotSafety:
    def test_false_block_not_safety(self) -> None:
        """false_block_rate is utility, not safety — candidate with higher
        false_block but higher macro should still win (it is safe)."""
        candidates = [
            _candidate("conservative", macro=0.6, false_block=0.0),
            _candidate("aggressive", macro=0.9, false_block=0.2),
        ]
        best = select_best_candidate(candidates, _BASELINE)
        # aggressive is safe (no safety release regression) and has higher macro
        assert best["params"]["id"] == "aggressive"
