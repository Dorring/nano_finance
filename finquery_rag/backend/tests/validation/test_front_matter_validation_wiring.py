"""Phase 4 production regression tests for front matter validation wiring.

Statically verify that the orchestrator's front matter branch runs
answerability evaluation and response validation (with repair), and that
the front matter validation policy requires citations.
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

_POLICY_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "validation",
    "validation_policy.py",
)


def _extract_policy_block(source: str, name: str) -> str:
    """Extract a ``name = ValidationPolicy(...)`` block from source.

    Returns the substring from the assignment start through the matching
    closing paren. Used to scope substring assertions to a single policy
    so a change to ``_FRONT_MATTER`` cannot be masked by other policies.
    """
    marker = f"{name} = ValidationPolicy("
    start = source.find(marker)
    assert start != -1, f"{name} definition not found in policy source"
    depth = 0
    i = source.find("(", start)
    while i < len(source):
        ch = source[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return source[start:]


@pytest.fixture(scope="module")
def orchestrator_source() -> str:
    """Load the orchestrator source for static wiring checks."""
    with open(_ORCHESTRATOR_PATH, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="module")
def policy_source() -> str:
    """Load the validation policy source for static checks."""
    with open(_POLICY_PATH, "r", encoding="utf-8") as f:
        return f.read()


def test_front_matter_runs_answerability(orchestrator_source: str) -> None:
    """Front matter branch must call evaluate_answerability."""
    assert "evaluate_answerability" in orchestrator_source
    assert "front_matter" in orchestrator_source


def test_front_matter_runs_source_validation(orchestrator_source: str) -> None:
    """Front matter branch must call _validate_and_repair_once."""
    assert "_validate_and_repair_once" in orchestrator_source
    assert "front_matter" in orchestrator_source


def test_front_matter_missing_evidence_is_blocked(orchestrator_source: str) -> None:
    """Front matter path must check for NOT_ANSWERABLE and refuse safely."""
    assert "AnswerabilityStatus.NOT_ANSWERABLE" in orchestrator_source


def test_front_matter_uses_front_matter_intent(orchestrator_source: str) -> None:
    """Front matter path must pass intent='front_matter'."""
    assert 'intent="front_matter"' in orchestrator_source


def test_front_matter_builds_evidence(orchestrator_source: str) -> None:
    """Front matter branch must build evidence via EvidenceItem.from_chunk."""
    assert "EvidenceItem.from_chunk" in orchestrator_source


def test_front_matter_policy_requires_citations(policy_source: str) -> None:
    """The _FRONT_MATTER policy must exist and require citations."""
    assert "_FRONT_MATTER = ValidationPolicy(" in policy_source
    block = _extract_policy_block(policy_source, "_FRONT_MATTER")
    assert "require_citations=True" in block
    # The policy must also be registered under the front_matter intent key.
    assert '"front_matter": _FRONT_MATTER' in policy_source
