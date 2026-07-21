"""Phase 4 production regression tests for validation artifact files.

Verify that the Phase 4 artifact JSON files exist in the
artifacts/validation/ directory and have the expected top-level structure.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

_ARTIFACTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "artifacts",
    "validation",
)

_EXPECTED_FILES = (
    "phase4-answerability-matrix.json",
    "phase4-validation-code-matrix.json",
    "phase4-policy-matrix.json",
    "phase4-api-contract.json",
    "phase4-streaming-safety.json",
    "phase4-non-validation-parity.json",
    "phase4-acceptance.json",
)


def _load_artifact(name: str) -> dict:
    """Load an artifact JSON file as a dict."""
    path = os.path.join(_ARTIFACTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_artifacts_directory_exists() -> None:
    """The artifacts/validation/ directory must exist."""
    assert os.path.isdir(_ARTIFACTS_DIR), (
        f"artifacts directory not found: {_ARTIFACTS_DIR}"
    )


def test_all_seven_artifact_files_exist() -> None:
    """All seven Phase 4 artifact JSON files must exist."""
    assert os.path.isdir(_ARTIFACTS_DIR), (
        f"artifacts directory not found: {_ARTIFACTS_DIR}"
    )
    missing = [
        name
        for name in _EXPECTED_FILES
        if not os.path.isfile(os.path.join(_ARTIFACTS_DIR, name))
    ]
    assert not missing, f"missing artifact files: {missing}"


def test_answerability_matrix_structure() -> None:
    """answerability-matrix.json must have a 'status' key with answerability statuses."""
    data = _load_artifact("phase4-answerability-matrix.json")
    assert "status" in data
    serialized = json.dumps(data)
    assert any(
        status in serialized
        for status in (
            "ANSWERABLE",
            "NOT_ANSWERABLE",
            "PARTIALLY_ANSWERABLE",
            "CALCULATION_BLOCKED",
        )
    ), "no canonical answerability status found in answerability matrix"


def test_validation_code_matrix_structure() -> None:
    """validation-code-matrix.json must have a 'codes' key with validation error codes."""
    data = _load_artifact("phase4-validation-code-matrix.json")
    assert "codes" in data


def test_policy_matrix_structure() -> None:
    """policy-matrix.json must have a 'policies' key with intent names."""
    data = _load_artifact("phase4-policy-matrix.json")
    assert "policies" in data
    serialized = json.dumps(data)
    assert any(
        intent in serialized
        for intent in (
            "financial_calculation",
            "document_qa",
            "front_matter",
            "conversation",
        )
    ), "no canonical intent name found in policy matrix"


def test_api_contract_structure() -> None:
    """api-contract.json must have an 'endpoints' key."""
    data = _load_artifact("phase4-api-contract.json")
    assert "endpoints" in data


def test_streaming_safety_structure() -> None:
    """streaming-safety.json must have a 'safety_checks' key."""
    data = _load_artifact("phase4-streaming-safety.json")
    assert "safety_checks" in data


def test_acceptance_structure() -> None:
    """acceptance.json must have 'criteria' (list), 'total', and 'passed' counts."""
    data = _load_artifact("phase4-acceptance.json")
    assert "criteria" in data
    assert isinstance(data["criteria"], list)
    assert len(data["criteria"]) > 0
    assert "total" in data
    assert isinstance(data["total"], int)
    assert "passed" in data
    assert isinstance(data["passed"], int)
    assert data["total"] == len(data["criteria"])
    assert 0 <= data["passed"] <= data["total"]


def test_acceptance_not_all_true_blindly() -> None:
    """acceptance.json must not blindly set everything to true without description.

    Each criterion must carry a non-empty 'description' so that a passing
    acceptance record is auditable rather than a rubber stamp.
    """
    data = _load_artifact("phase4-acceptance.json")
    criteria = data["criteria"]
    assert isinstance(criteria, list)
    assert len(criteria) > 0
    for criterion in criteria:
        assert "description" in criterion, (
            f"criterion missing 'description': {criterion}"
        )
        description = criterion.get("description")
        assert isinstance(description, str) and description.strip(), (
            f"criterion has empty description: {criterion}"
        )
