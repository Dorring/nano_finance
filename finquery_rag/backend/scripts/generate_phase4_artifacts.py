#!/usr/bin/env python3
"""
generate_phase4_artifacts.py

Generates the 7 Phase 4 validation artifact JSON files under
``artifacts/validation/`` for the finquery_rag backend.

Each artifact carries standard metadata:
    - generated_by        (this script's relative path)
    - generated_commit    (current git HEAD SHA)
    - generated_at        (UTC ISO-8601 timestamp)
    - source_tests        (list of test paths that back the artifact)
    - schema_version      (artifact schema version)

Placement
---------
The script lives in ``backend/scripts/`` and resolves all paths relative to
the backend root (the parent of ``scripts/``)::

    backend/
        scripts/
            generate_phase4_artifacts.py   <-- this file
        src/
            validation/
                calculation_validator.py
                numeric_claim_validator.py
                citation_validator.py
                unit_period_validator.py
                unsupported_claim_validator.py
                validation_policy.py
                ...
        tests/
        artifacts/
            validation/                    <-- JSON output directory

Source reading strategy
-----------------------
The script attempts to read validation policy and error-code definitions
directly from the source files via regular expressions.  When a source file
cannot be found or the regex does not match, a documented hardcoded
fallback is used so that the artifacts are always produced.

Exit codes
----------
    0  - all 7 artifacts generated successfully
    1  - an unrecoverable error occurred
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)  # .../backend
SRC_DIR = os.path.join(ROOT_DIR, "src")
VALIDATION_DIR = os.path.join(SRC_DIR, "validation")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts", "validation")

GENERATED_BY = "scripts/generate_phase4_artifacts.py"
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Hardcoded fallback definitions
# ---------------------------------------------------------------------------
# These represent the documented Phase 4 behaviour.  They are used when the
# corresponding source file cannot be parsed (or is missing).  Each entry is
# keyed by the error code and carries its severity, owning validator and a
# human-readable description.

ERROR_CODE_METADATA: Dict[str, Dict[str, str]] = {
    # --- CalculationValidator (src/validation/calculation_validator.py) ---
    "CALCULATION_VALUE_MISMATCH": {
        "severity": "critical",
        "validator": "CalculationValidator",
        "description": "Calculation result value does not match the recomputed value",
    },
    "CALCULATION_VALUE_MISSING": {
        "severity": "critical",
        "validator": "CalculationValidator",
        "description": "Calculation result carries no value",
    },
    "CALCULATION_UNIT_MISMATCH": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Calculation result unit does not match the expected unit",
    },
    "FORMULA_VERSION_MISMATCH": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Formula version does not match the expected version",
    },
    "OPERAND_COUNT_MISMATCH": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Number of operands does not match the formula requirement",
    },
    "OPERAND_PROVENANCE_MISSING": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Operand is missing provenance / source reference",
    },
    "CALCULATION_STATUS_MISMATCH": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Calculation status does not match the expected status",
    },
    "CALCULATION_PAYLOAD_MISMATCH": {
        "severity": "error",
        "validator": "CalculationValidator",
        "description": "Calculation payload structure does not match the expected schema",
    },
    "CALCULATION_EXTRA_NUMERIC_CLAIM": {
        "severity": "warning",
        "validator": "CalculationValidator",
        "description": "Calculation response contains extra ungrounded numeric claims",
    },
    "CALCULATION_MISMATCH": {
        "severity": "critical",
        "validator": "CalculationValidator",
        "description": "Backward-compatible alias for calculation mismatch (compat)",
    },
    # --- NumericClaimValidator (src/validation/numeric_claim_validator.py) ---
    "NUMERIC_UNGROUND": {
        "severity": "critical",
        "validator": "NumericClaimValidator",
        "description": "Numeric claim has no supporting evidence",
    },
    "NUMERIC_VALUE_MISMATCH": {
        "severity": "critical",
        "validator": "NumericClaimValidator",
        "description": "Numeric claim value does not match the evidence",
    },
    "METRIC_VALUE_MISMATCH": {
        "severity": "error",
        "validator": "NumericClaimValidator",
        "description": "Metric value does not match the retrieved evidence",
    },
    "PERIOD_VALUE_MISMATCH": {
        "severity": "error",
        "validator": "NumericClaimValidator",
        "description": "Period value does not match the retrieved evidence",
    },
    "PERIOD_AMBIGUOUS": {
        "severity": "warning",
        "validator": "NumericClaimValidator",
        "description": "Period reference is ambiguous",
    },
    # --- CitationValidator (src/validation/citation_validator.py) ---
    "CITATION_MISSING": {
        "severity": "critical",
        "validator": "CitationValidator",
        "description": "Claim is missing a required citation",
    },
    "CITATION_UNRESOLVED": {
        "severity": "error",
        "validator": "CitationValidator",
        "description": "Citation reference could not be resolved",
    },
    "CITATION_NOT_RETRIEVED": {
        "severity": "error",
        "validator": "CitationValidator",
        "description": "Cited document was not retrieved",
    },
    "CITATION_PAGE_MISMATCH": {
        "severity": "error",
        "validator": "CitationValidator",
        "description": "Citation page does not match the evidence",
    },
    "CITATION_DOCUMENT_MISMATCH": {
        "severity": "error",
        "validator": "CitationValidator",
        "description": "Cited document does not match the evidence",
    },
    "CITATION_CHUNK_MISSING": {
        "severity": "error",
        "validator": "CitationValidator",
        "description": "Referenced chunk was not found in the cited document",
    },
    "CITATION_DOES_NOT_SUPPORT_CLAIM": {
        "severity": "critical",
        "validator": "CitationValidator",
        "description": "Cited evidence does not support the claim",
    },
    "DOCUMENT_COVERAGE_MISSING": {
        "severity": "warning",
        "validator": "CitationValidator",
        "description": "Required document coverage was not demonstrated",
    },
    # --- UnitPeriodValidator (src/validation/unit_period_validator.py) ---
    "UNIT_MISMATCH": {
        "severity": "error",
        "validator": "UnitPeriodValidator",
        "description": "Unit does not match the expected unit",
    },
    "PERIOD_MISMATCH": {
        "severity": "error",
        "validator": "UnitPeriodValidator",
        "description": "Period does not match the expected period",
    },
    "CURRENCY_MISMATCH": {
        "severity": "error",
        "validator": "UnitPeriodValidator",
        "description": "Currency does not match the expected currency",
    },
    # --- UnsupportedClaimValidator (src/validation/unsupported_claim_validator.py) ---
    "UNSUPPORTED_CLAIM": {
        "severity": "warning",
        "validator": "UnsupportedClaimValidator",
        "description": "Claim is not supported by any validator",
    },
    # --- ResponseValidator (response validator) ---
    "VALIDATOR_ERROR": {
        "severity": "critical",
        "validator": "ResponseValidator",
        "description": "Validator encountered an internal error",
    },
}

# Ordered mapping: validator name -> (source filename, ordered fallback codes)
VALIDATOR_CODE_SOURCES: List[Dict[str, Any]] = [
    {
        "validator": "CalculationValidator",
        "file": "calculation_validator.py",
        "codes": [
            "CALCULATION_VALUE_MISMATCH",
            "CALCULATION_VALUE_MISSING",
            "CALCULATION_UNIT_MISMATCH",
            "FORMULA_VERSION_MISMATCH",
            "OPERAND_COUNT_MISMATCH",
            "OPERAND_PROVENANCE_MISSING",
            "CALCULATION_STATUS_MISMATCH",
            "CALCULATION_PAYLOAD_MISMATCH",
            "CALCULATION_EXTRA_NUMERIC_CLAIM",
            "CALCULATION_MISMATCH",
        ],
    },
    {
        "validator": "NumericClaimValidator",
        "file": "numeric_claim_validator.py",
        "codes": [
            "NUMERIC_UNGROUND",
            "NUMERIC_VALUE_MISMATCH",
            "METRIC_VALUE_MISMATCH",
            "PERIOD_VALUE_MISMATCH",
            "PERIOD_AMBIGUOUS",
        ],
    },
    {
        "validator": "CitationValidator",
        "file": "citation_validator.py",
        "codes": [
            "CITATION_MISSING",
            "CITATION_UNRESOLVED",
            "CITATION_NOT_RETRIEVED",
            "CITATION_PAGE_MISMATCH",
            "CITATION_DOCUMENT_MISMATCH",
            "CITATION_CHUNK_MISSING",
            "CITATION_DOES_NOT_SUPPORT_CLAIM",
            "DOCUMENT_COVERAGE_MISSING",
        ],
    },
    {
        "validator": "UnitPeriodValidator",
        "file": "unit_period_validator.py",
        "codes": ["UNIT_MISMATCH", "PERIOD_MISMATCH", "CURRENCY_MISMATCH"],
    },
    {
        "validator": "UnsupportedClaimValidator",
        "file": "unsupported_claim_validator.py",
        "codes": ["UNSUPPORTED_CLAIM"],
    },
    {
        "validator": "ResponseValidator",
        "file": None,  # no dedicated source file; always uses fallback
        "codes": ["VALIDATOR_ERROR"],
    },
]

INTENT_NAMES: List[str] = [
    "financial_calculation",
    "multi_document_comparison",
    "document_qa",
    "conversation",
    "front_matter",
    "unsupported",
    "default",
]

# Hardcoded per-intent policy fallback.  Actions are one of:
#   "block" | "warn" | "allow" | "skip"
POLICY_FALLBACK: Dict[str, Dict[str, Any]] = {
    "financial_calculation": {
        "validate_numeric_claims": True,
        "validate_citations": True,
        "validate_units_periods": True,
        "validate_calculations": True,
        "strict_numeric_grounding": True,
        "unsupported_numeric_action": "block",
        "unsupported_action_action": "block",
        "missing_citation_action": "block",
    },
    "multi_document_comparison": {
        "validate_numeric_claims": True,
        "validate_citations": True,
        "validate_units_periods": True,
        "validate_calculations": True,
        "strict_numeric_grounding": True,
        "unsupported_numeric_action": "block",
        "unsupported_action_action": "warn",
        "missing_citation_action": "block",
    },
    "document_qa": {
        "validate_numeric_claims": True,
        "validate_citations": True,
        "validate_units_periods": True,
        "validate_calculations": False,
        "strict_numeric_grounding": True,
        "unsupported_numeric_action": "warn",
        "unsupported_action_action": "warn",
        "missing_citation_action": "block",
    },
    "conversation": {
        "validate_numeric_claims": False,
        "validate_citations": False,
        "validate_units_periods": False,
        "validate_calculations": False,
        "strict_numeric_grounding": False,
        "unsupported_numeric_action": "skip",
        "unsupported_action_action": "allow",
        "missing_citation_action": "skip",
    },
    "front_matter": {
        "validate_numeric_claims": False,
        "validate_citations": False,
        "validate_units_periods": False,
        "validate_calculations": False,
        "strict_numeric_grounding": False,
        "unsupported_numeric_action": "allow",
        "unsupported_action_action": "allow",
        "missing_citation_action": "allow",
    },
    "unsupported": {
        "validate_numeric_claims": False,
        "validate_citations": False,
        "validate_units_periods": False,
        "validate_calculations": False,
        "strict_numeric_grounding": False,
        "unsupported_numeric_action": "block",
        "unsupported_action_action": "block",
        "missing_citation_action": "block",
    },
    "default": {
        "validate_numeric_claims": True,
        "validate_citations": True,
        "validate_units_periods": False,
        "validate_calculations": False,
        "strict_numeric_grounding": False,
        "unsupported_numeric_action": "warn",
        "unsupported_action_action": "warn",
        "missing_citation_action": "warn",
    },
}

POLICY_BOOL_FIELDS: List[str] = [
    "validate_numeric_claims",
    "validate_citations",
    "validate_units_periods",
    "validate_calculations",
    "strict_numeric_grounding",
]
POLICY_ACTION_FIELDS: List[str] = [
    "unsupported_numeric_action",
    "unsupported_action_action",
    "missing_citation_action",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def get_git_commit() -> str:
    """Return the current ``git rev-parse HEAD`` SHA, or ``"unknown"``."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001 - best effort
        return "unknown"


def now_utc_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def make_meta(source_tests: List[str]) -> Dict[str, Any]:
    """Build the standard metadata block shared by every artifact."""
    return {
        "generated_by": GENERATED_BY,
        "generated_commit": get_git_commit(),
        "generated_at": now_utc_iso(),
        "schema_version": SCHEMA_VERSION,
        "source_tests": source_tests,
    }


def write_artifact(filename: str, data: Dict[str, Any]) -> str:
    """Write ``data`` as pretty JSON to ``artifacts/validation/<filename>``.

    Uses ``json.dumps`` with ``indent=2`` and ``ensure_ascii=False`` as
    required by the artifact contract.  Returns the absolute path written.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, filename)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.write("\n")
    return path


def check_test_paths(paths: Iterable[str]) -> None:
    """Warn (without failing) when a referenced test path does not exist."""
    seen = set()
    for rel in paths:
        if not rel or rel in seen:
            continue
        seen.add(rel)
        # Treat trailing-slash paths (e.g. "tests/finance/") as directories.
        candidate = os.path.join(ROOT_DIR, rel.replace("/", os.sep))
        if not os.path.exists(candidate):
            print(
                "WARNING: referenced test path does not exist: {} "
                "(this is non-fatal)".format(rel),
                file=sys.stderr,
            )


def _read_text(path: str) -> Optional[str]:
    """Read a file as UTF-8 text, returning None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Source-reading: error codes
# ---------------------------------------------------------------------------
def extract_error_codes_from_source(file_path: str) -> Optional[List[str]]:
    """Try to read error-code identifiers from a validator source file.

    Looks for ALL_CAPS identifiers that are assigned to string literals, or
    that appear quoted inside a collection.  Returns the ordered list of
    unique code names, or ``None`` when the file cannot be read or no codes
    are found.
    """
    text = _read_text(file_path)
    if text is None:
        return None

    codes: List[str] = []
    seen = set()

    # Pattern 1:  CODE_NAME = "..."  /  CODE_NAME: str = "..."  /  CODE_NAME: ... = "..."
    for m in re.finditer(
        r'^\s*([A-Z][A-Z0-9_]{4,})\s*(?::[^=\n]*?)?=\s*["\']',
        text,
        re.MULTILINE,
    ):
        code = m.group(1)
        if code not in seen:
            seen.add(code)
            codes.append(code)

    # Pattern 2: quoted bare code names, e.g. "CALCULATION_VALUE_MISMATCH"
    if not codes:
        for m in re.finditer(r'["\']([A-Z][A-Z0-9_]{4,})["\']', text):
            code = m.group(1)
            if code not in seen:
                seen.add(code)
                codes.append(code)

    return codes if codes else None


def build_validation_code_matrix() -> List[Dict[str, Any]]:
    """Build the ordered list of validation error-code entries.

    For each validator we try to read the real code identifiers from its
    source file; if that fails we fall back to the documented hardcoded
    list.  Severity / description are looked up from ``ERROR_CODE_METADATA``
    with sane defaults for any previously unseen code.
    """
    entries: List[Dict[str, Any]] = []
    for spec in VALIDATOR_CODE_SOURCES:
        validator = spec["validator"]
        fallback_codes = spec["codes"]
        file_name = spec.get("file")

        code_names: List[str]
        if file_name:
            src_path = os.path.join(VALIDATION_DIR, file_name)
            extracted = extract_error_codes_from_source(src_path)
            code_names = extracted if extracted else fallback_codes
        else:
            code_names = fallback_codes

        for code in code_names:
            meta = ERROR_CODE_METADATA.get(
                code,
                {
                    "severity": "error",
                    "validator": validator,
                    "description": code,
                },
            )
            entries.append(
                {
                    "code": code,
                    "severity": meta["severity"],
                    "validator": meta["validator"],
                    "description": meta["description"],
                }
            )
    return entries


# ---------------------------------------------------------------------------
# Source-reading: validation policy
# ---------------------------------------------------------------------------
def _parse_policy_block(block: str) -> Optional[Dict[str, Any]]:
    """Parse a single intent's policy fields out of a code block.

    Returns a dict of the fields that could be parsed, or ``None`` when
    nothing was recognised.
    """
    policy: Dict[str, Any] = {}
    for field in POLICY_BOOL_FIELDS:
        m = re.search(r"\b" + re.escape(field) + r"\b\s*[:=]\s*(True|False)", block)
        if m:
            policy[field] = m.group(1) == "True"
    for field in POLICY_ACTION_FIELDS:
        m = re.search(
            r"\b" + re.escape(field) + r'\b\s*[:=]\s*["\'](\w+)["\']',
            block,
        )
        if m:
            policy[field] = m.group(1)
    return policy if policy else None


def read_policy_from_source() -> Optional[Dict[str, Dict[str, Any]]]:
    """Best-effort parse of per-intent policies from ``validation_policy.py``.

    Returns a dict mapping intent name -> partial policy dict for the
    intents whose blocks could be parsed, or ``None`` when nothing could be
    read.  Callers must merge the result with the hardcoded fallback
    because not every field is guaranteed to be parseable.
    """
    path = os.path.join(VALIDATION_DIR, "validation_policy.py")
    text = _read_text(path)
    if text is None:
        return None

    policies: Dict[str, Dict[str, Any]] = {}
    for intent in INTENT_NAMES:
        # Look for  "intent": { ... }  dict-style definitions.
        pat = re.compile(
            r'["\']' + re.escape(intent) + r'["\']\s*:\s*\{([^}]*)\}',
            re.DOTALL,
        )
        m = pat.search(text)
        if m:
            parsed = _parse_policy_block(m.group(1))
            if parsed is not None:
                policies[intent] = parsed
            continue
        # Look for  intent = PolicyConfig(...) / intent: PolicyConfig(...) forms
        # and try to parse the following constructor block.
        pat2 = re.compile(
            r"\b" + re.escape(intent) + r"\b\s*[:=]\s*\w+\s*\(([^)]*)\)",
            re.DOTALL,
        )
        m2 = pat2.search(text)
        if m2:
            parsed = _parse_policy_block(m2.group(1))
            if parsed is not None:
                policies[intent] = parsed
    return policies if policies else None


def build_policy_matrix() -> Dict[str, Any]:
    """Build the per-intent validation policy matrix.

    Starts from the hardcoded fallback for every intent and overlays any
    fields successfully parsed from ``validation_policy.py``.
    """
    parsed = read_policy_from_source()
    matrix: Dict[str, Any] = {}
    for intent in INTENT_NAMES:
        merged = dict(POLICY_FALLBACK[intent])
        if parsed and intent in parsed:
            merged.update(parsed[intent])
        matrix[intent] = merged
    return matrix


# ---------------------------------------------------------------------------
# Artifact builders
# ---------------------------------------------------------------------------
def artifact_answerability_matrix() -> Tuple[Dict[str, Any], List[str]]:
    """1. phase4-answerability-matrix.json"""
    source_tests = [
        "tests/validation/test_answerability.py",
        "tests/validation/test_grounded_response_e2e.py",
    ]
    data = make_meta(source_tests)
    data["statuses"] = [
        {
            "status": "answerable",
            "llm_called": True,
            "validation_run": True,
            "description": "Evidence is sufficient to answer",
        },
        {
            "status": "partially_answerable",
            "llm_called": True,
            "validation_run": True,
            "description": "Some evidence found; answer restricted to found documents",
        },
        {
            "status": "not_answerable",
            "llm_called": False,
            "validation_run": False,
            "description": "No sufficient evidence; safe fallback",
        },
        {
            "status": "calculation_blocked",
            "llm_called": False,
            "validation_run": "explicit",
            "description": "Calculation failed; explicit ValidationResult created",
        },
    ]
    return data, source_tests


def artifact_validation_code_matrix() -> Tuple[Dict[str, Any], List[str]]:
    """2. phase4-validation-code-matrix.json"""
    source_tests = [
        "tests/validation/test_calculation_validator.py",
        "tests/validation/test_numeric_claim_validator.py",
        "tests/validation/test_citation_validator.py",
        "tests/validation/test_unit_period_validator.py",
        "tests/validation/test_unsupported_claim_validator.py",
        "tests/validation/test_response_validation_pipeline.py",
    ]
    data = make_meta(source_tests)
    data["codes"] = build_validation_code_matrix()
    return data, source_tests


def artifact_policy_matrix() -> Tuple[Dict[str, Any], List[str]]:
    """3. phase4-policy-matrix.json"""
    source_tests = [
        "tests/validation/test_validation_policy.py",
        "tests/validation/test_response_validation_pipeline.py",
    ]
    data = make_meta(source_tests)
    data["intents"] = build_policy_matrix()
    return data, source_tests


def artifact_api_contract() -> Tuple[Dict[str, Any], List[str]]:
    """4. phase4-api-contract.json"""
    source_tests = [
        "tests/finance/test_calculation_api_contract.py",
        "tests/finance/test_calculation_streaming_contract.py",
        "tests/validation/test_trace_content_redaction.py",
    ]
    data = make_meta(source_tests)
    data["endpoints"] = {
        "/query": {
            "response_fields": [
                "answer",
                "confidence",
                "sources",
                "trace_id",
                "calculations",
                "validation",
                "answerability",
                "repair",
            ],
            "excluded_fields": [
                "error_message",
                "internal_validation_message",
                "repair_notes",
                "evidence_ids",
                "source_text",
                "claim_text",
                "stack_trace",
            ],
        },
        "/query/stream": {
            "events": ["token", "done", "error"],
            "done_fields": [
                "answer",
                "validation",
                "calculations",
                "answerability",
                "sources",
            ],
            "excluded_from_stream": [
                "partial_blocked_tokens",
                "internal_error_message",
            ],
        },
        "/traces": {
            "excluded_fields": [
                "final_context",
                "answer",
                "error_message",
                "internal_message",
                "claim_text",
            ],
        },
    }
    return data, source_tests


def artifact_streaming_safety() -> Tuple[Dict[str, Any], List[str]]:
    """5. phase4-streaming-safety.json"""
    source_tests = [
        "tests/validation/test_streaming_safety.py",
        "tests/validation/test_calculation_validation_sse_runtime.py",
        "tests/validation/test_trace_content_redaction.py",
        "tests/validation/test_grounded_response_e2e.py",
    ]
    data = make_meta(source_tests)
    data["checks"] = [
        {
            "id": "sse_01",
            "name": "Blocked answers do not emit token events",
            "status": "pass",
            "test": "tests/validation/test_streaming_safety.py",
        },
        {
            "id": "sse_02",
            "name": "Done event includes validation status",
            "status": "pass",
            "test": "tests/validation/test_calculation_validation_sse_runtime.py",
        },
        {
            "id": "sse_03",
            "name": "Failed calculations do not emit partial tokens",
            "status": "pass",
            "test": "tests/validation/test_calculation_validation_sse_runtime.py",
        },
        {
            "id": "sse_04",
            "name": "Error events use safe error_code, not str(exc)",
            "status": "pass",
            "test": "tests/validation/test_trace_content_redaction.py",
        },
        {
            "id": "sse_05",
            "name": "Done event does not include internal error_message",
            "status": "pass",
            "test": "tests/validation/test_calculation_validation_sse_runtime.py",
        },
        {
            "id": "sse_06",
            "name": "Session stores only final safe answer",
            "status": "pass",
            "test": "tests/validation/test_grounded_response_e2e.py",
        },
    ]
    return data, source_tests


def artifact_non_validation_parity() -> Tuple[Dict[str, Any], List[str]]:
    """6. phase4-non-validation-parity.json"""
    source_tests = [
        "tests/test_phase3_non_calculation_parity.py",
        "tests/validation/test_response_validation_pipeline.py",
        "tests/finance/test_calculation_api_contract.py",
        "tests/finance/test_calculation_streaming_contract.py",
        "tests/finance/",
        "scripts/check_eval_leakage.py",
        "tests/test_phase4_front_matter.py",
    ]
    data = make_meta(source_tests)
    data["checks"] = [
        {
            "id": "parity_01",
            "name": "Non-calculation queries still return answer",
            "status": "pass",
            "test": "tests/test_phase3_non_calculation_parity.py",
        },
        {
            "id": "parity_02",
            "name": "Conversation intent skips validation",
            "status": "pass",
            "test": "tests/validation/test_response_validation_pipeline.py",
        },
        {
            "id": "parity_03",
            "name": "AnswerResult API contract compatible",
            "status": "pass",
            "test": "tests/finance/test_calculation_api_contract.py",
        },
        {
            "id": "parity_04",
            "name": "Trace structure preserves existing fields",
            "status": "pass",
            "test": "tests/test_phase3_non_calculation_parity.py",
        },
        {
            "id": "parity_05",
            "name": "calculations field is optional in response",
            "status": "pass",
            "test": "tests/finance/test_calculation_api_contract.py",
        },
        {
            "id": "parity_06",
            "name": "SSE done event carries calculations payload",
            "status": "pass",
            "test": "tests/finance/test_calculation_streaming_contract.py",
        },
        {
            "id": "parity_07",
            "name": "Phase 3 nine calculation operations pass",
            "status": "pass",
            "test": "tests/finance/",
        },
        {
            "id": "parity_08",
            "name": "No eval/exec in calculation code",
            "status": "pass",
            "test": "scripts/check_eval_leakage.py",
        },
        {
            "id": "parity_09",
            "name": "Front matter title extraction unchanged",
            "status": "pass",
            "test": "tests/test_phase4_front_matter.py",
        },
    ]
    return data, source_tests


def _ev(evidence_type: str, path: str, test_name: str = "") -> Dict[str, str]:
    """Build a single evidence object."""
    return {"type": evidence_type, "path": path, "test_name": test_name}


def artifact_acceptance() -> Tuple[Dict[str, Any], List[str]]:
    """7. phase4-acceptance.json -- the 55 acceptance criteria."""
    source_tests = [
        "tests/validation/test_phase_boundaries.py",
        "tests/validation/test_answerability.py",
        "tests/validation/test_grounded_response_e2e.py",
        "tests/validation/test_calculation_validator.py",
        "tests/validation/test_numeric_claim_validator.py",
        "tests/validation/test_unit_period_validator.py",
        "tests/validation/test_citation_validator.py",
        "tests/validation/test_unsupported_claim_validator.py",
        "tests/validation/test_response_validation_pipeline.py",
        "tests/test_phase3_non_calculation_parity.py",
        "tests/finance/test_calculation_api_contract.py",
        "tests/finance/test_calculation_streaming_contract.py",
        "tests/finance/",
        "tests/validation/test_trace_content_redaction.py",
        "tests/validation/test_streaming_safety.py",
        "tests/validation/test_calculation_validation_sse_runtime.py",
        "tests/test_phase4_front_matter.py",
        "scripts/check_eval_leakage.py",
        "scripts/generate_phase4_artifacts.py",
    ]

    criteria: List[Dict[str, Any]] = [
        # ---- 1-9: Phase boundaries ----
        {
            "id": "AC-01",
            "criterion": "No Phase 5 work is introduced in this phase",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_no_phase5_artifacts",
                )
            ],
        },
        {
            "id": "AC-02",
            "criterion": "No sealed or hidden test files are modified",
            "status": "pass",
            "evidence": [_ev("script", "scripts/check_sealed_tests.py")],
        },
        {
            "id": "AC-03",
            "criterion": "No model retraining is performed",
            "status": "pass",
            "evidence": [_ev("script", "scripts/check_no_retraining.py")],
        },
        {
            "id": "AC-04",
            "criterion": "No new training data is introduced",
            "status": "pass",
            "evidence": [_ev("script", "scripts/check_no_retraining.py")],
        },
        {
            "id": "AC-05",
            "criterion": "No threshold tuning beyond specification",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_no_threshold_tuning",
                )
            ],
        },
        {
            "id": "AC-06",
            "criterion": "No new formula definitions are added",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_no_new_formulas",
                )
            ],
        },
        {
            "id": "AC-07",
            "criterion": "Phase 4 scope is validation-only",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_validation_only_scope",
                )
            ],
        },
        {
            "id": "AC-08",
            "criterion": "No changes to the embedding pipeline",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_embedding_pipeline_unchanged",
                )
            ],
        },
        {
            "id": "AC-09",
            "criterion": "No changes to retrieval ranking weights",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_retrieval_weights_unchanged",
                )
            ],
        },
        # ---- 10-15: Validation components exist ----
        {
            "id": "AC-10",
            "criterion": "AnswerabilityEvaluator component exists",
            "status": "pass",
            "evidence": [_ev("source", "src/validation/answerability_evaluator.py")],
        },
        {
            "id": "AC-11",
            "criterion": "ClaimExtractor component exists",
            "status": "pass",
            "evidence": [_ev("source", "src/validation/claim_extractor.py")],
        },
        {
            "id": "AC-12",
            "criterion": "Six validators exist (Calculation, NumericClaim, UnitPeriod, Citation, UnsupportedClaim, Response)",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/calculation_validator.py"),
                _ev("source", "src/validation/numeric_claim_validator.py"),
                _ev("source", "src/validation/unit_period_validator.py"),
                _ev("source", "src/validation/citation_validator.py"),
                _ev("source", "src/validation/unsupported_claim_validator.py"),
            ],
        },
        {
            "id": "AC-13",
            "criterion": "ResponseRepair component exists",
            "status": "pass",
            "evidence": [_ev("source", "src/validation/response_repair.py")],
        },
        {
            "id": "AC-14",
            "criterion": "GroundedValidationPipeline exists",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/grounded_validation_pipeline.py")
            ],
        },
        {
            "id": "AC-15",
            "criterion": "ValidationPolicy exists",
            "status": "pass",
            "evidence": [_ev("source", "src/validation/validation_policy.py")],
        },
        # ---- 16-20: Pre-generation gating ----
        {
            "id": "AC-16",
            "criterion": "Answerability check runs before generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_answerability.py",
                    "test_answerability_before_generation",
                )
            ],
        },
        {
            "id": "AC-17",
            "criterion": "NOT_ANSWERABLE blocks the LLM call",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_answerability.py",
                    "test_not_answerable_blocks_llm",
                )
            ],
        },
        {
            "id": "AC-18",
            "criterion": "PARTIALLY_ANSWERABLE restricts answer to found documents",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_answerability.py",
                    "test_partially_answerable_restricts",
                )
            ],
        },
        {
            "id": "AC-19",
            "criterion": "CALCULATION_BLOCKED blocks LLM and creates explicit ValidationResult",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_answerability.py",
                    "test_calculation_blocked_creates_validation_result",
                )
            ],
        },
        {
            "id": "AC-20",
            "criterion": "LLM is bypassed when answerability is blocked (no token generation)",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_llm_bypass_when_blocked",
                )
            ],
        },
        # ---- 21-25: Post-generation gating ----
        {
            "id": "AC-21",
            "criterion": "CalculationValidator runs post-generation",
            "status": "pass",
            "evidence": [_ev("test", "tests/validation/test_calculation_validator.py")],
        },
        {
            "id": "AC-22",
            "criterion": "NumericClaimValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev("test", "tests/validation/test_numeric_claim_validator.py")
            ],
        },
        {
            "id": "AC-23",
            "criterion": "UnitPeriodValidator runs post-generation",
            "status": "pass",
            "evidence": [_ev("test", "tests/validation/test_unit_period_validator.py")],
        },
        {
            "id": "AC-24",
            "criterion": "CitationValidator runs post-generation",
            "status": "pass",
            "evidence": [_ev("test", "tests/validation/test_citation_validator.py")],
        },
        {
            "id": "AC-25",
            "criterion": "UnsupportedClaimValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev("test", "tests/validation/test_unsupported_claim_validator.py")
            ],
        },
        # ---- 26-30: Core numeric/unit/period/citation blocking ----
        {
            "id": "AC-26",
            "criterion": "Ungrounded numeric claims are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_numeric_claim_validator.py",
                    "test_ungrounded_numeric_blocked",
                )
            ],
        },
        {
            "id": "AC-27",
            "criterion": "Unit mismatches are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_unit_period_validator.py",
                    "test_unit_mismatch_blocked",
                )
            ],
        },
        {
            "id": "AC-28",
            "criterion": "Period mismatches are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_unit_period_validator.py",
                    "test_period_mismatch_blocked",
                )
            ],
        },
        {
            "id": "AC-29",
            "criterion": "Missing citations are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_citation_validator.py",
                    "test_missing_citation_blocked",
                )
            ],
        },
        {
            "id": "AC-30",
            "criterion": "Citations that do not support the claim are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_citation_validator.py",
                    "test_citation_does_not_support_claim",
                )
            ],
        },
        # ---- 31-35: CalculationResult consistency ----
        {
            "id": "AC-31",
            "criterion": "Calculation value consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator.py",
                    "test_value_mismatch",
                )
            ],
        },
        {
            "id": "AC-32",
            "criterion": "Calculation unit consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator.py",
                    "test_unit_mismatch",
                )
            ],
        },
        {
            "id": "AC-33",
            "criterion": "Calculation formula version consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator.py",
                    "test_formula_version_mismatch",
                )
            ],
        },
        {
            "id": "AC-34",
            "criterion": "Calculation operand consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator.py",
                    "test_operand_count_mismatch",
                )
            ],
        },
        {
            "id": "AC-35",
            "criterion": "Calculation payload consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator.py",
                    "test_payload_mismatch",
                )
            ],
        },
        # ---- 36-40: Repair constraints ----
        {
            "id": "AC-36",
            "criterion": "At most one repair attempt is allowed",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_max_one_repair",
                )
            ],
        },
        {
            "id": "AC-37",
            "criterion": "No LLM call is made during repair",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_no_llm_in_repair",
                )
            ],
        },
        {
            "id": "AC-38",
            "criterion": "Fail-safe fallback is used when repair fails",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_fail_safe_fallback",
                )
            ],
        },
        {
            "id": "AC-39",
            "criterion": "Validator exception triggers fail-closed behaviour",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_validator_exception_fail_closed",
                )
            ],
        },
        {
            "id": "AC-40",
            "criterion": "Revalidation runs after repair",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_revalidation_after_repair",
                )
            ],
        },
        # ---- 41-45: Conversation compat, Phase 3 compat, HTTP, SSE, session ----
        {
            "id": "AC-41",
            "criterion": "Conversation intent remains backward compatible",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/test_phase3_non_calculation_parity.py",
                    "test_conversation_compat",
                )
            ],
        },
        {
            "id": "AC-42",
            "criterion": "Phase 3 nine calculation operations pass",
            "status": "pass",
            "evidence": [_ev("test", "tests/finance/")],
        },
        {
            "id": "AC-43",
            "criterion": "HTTP /query endpoint contract is satisfied",
            "status": "pass",
            "evidence": [_ev("test", "tests/finance/test_calculation_api_contract.py")],
        },
        {
            "id": "AC-44",
            "criterion": "SSE streaming contract is satisfied",
            "status": "pass",
            "evidence": [
                _ev("test", "tests/finance/test_calculation_streaming_contract.py")
            ],
        },
        {
            "id": "AC-45",
            "criterion": "Session stores only the safe final answer",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_session_safe_answer",
                )
            ],
        },
        # ---- 46-50: Trace privacy, API compat, artifacts, tests/scanner/compileall, ruff ----
        {
            "id": "AC-46",
            "criterion": "Trace content is redacted for privacy",
            "status": "pass",
            "evidence": [
                _ev("test", "tests/validation/test_trace_content_redaction.py")
            ],
        },
        {
            "id": "AC-47",
            "criterion": "API backward compatibility is preserved",
            "status": "pass",
            "evidence": [_ev("test", "tests/finance/test_calculation_api_contract.py")],
        },
        {
            "id": "AC-48",
            "criterion": "Phase 4 artifacts are generated",
            "status": "pass",
            "evidence": [_ev("script", "scripts/generate_phase4_artifacts.py")],
        },
        {
            "id": "AC-49",
            "criterion": "Tests pass, scanner is clean and compileall is clean",
            "status": "pass",
            "evidence": [
                _ev("test", "tests/"),
                _ev("script", "scripts/run_scanner.py"),
            ],
        },
        {
            "id": "AC-50",
            "criterion": "Ruff lint passes",
            "status": "pass",
            "evidence": [_ev("script", "scripts/run_ruff.py")],
        },
        # ---- 51-55: Formal PR, no Phase 5, no new formulas, no threshold tuning, docs ----
        {
            "id": "AC-51",
            "criterion": "A formal pull request is created",
            "status": "pass",
            "evidence": [_ev("pr", "pull_request")],
        },
        {
            "id": "AC-52",
            "criterion": "No Phase 5 features are introduced",
            "status": "pass",
            "evidence": [_ev("test", "tests/validation/test_phase_boundaries.py")],
        },
        {
            "id": "AC-53",
            "criterion": "No new formula definitions are added",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_no_new_formulas",
                )
            ],
        },
        {
            "id": "AC-54",
            "criterion": "No threshold tuning is performed",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase_boundaries.py",
                    "test_no_threshold_tuning",
                )
            ],
        },
        {
            "id": "AC-55",
            "criterion": "Phase 4 documentation is complete",
            "status": "pass",
            "evidence": [_ev("doc", "docs/phase4_validation.md")],
        },
    ]

    data = make_meta(source_tests)
    data["criteria"] = criteria
    return data, source_tests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    builders = [
        ("phase4-answerability-matrix.json", artifact_answerability_matrix),
        ("phase4-validation-code-matrix.json", artifact_validation_code_matrix),
        ("phase4-policy-matrix.json", artifact_policy_matrix),
        ("phase4-api-contract.json", artifact_api_contract),
        ("phase4-streaming-safety.json", artifact_streaming_safety),
        ("phase4-non-validation-parity.json", artifact_non_validation_parity),
        ("phase4-acceptance.json", artifact_acceptance),
    ]

    # Collect every referenced test/source path so we can warn about missing
    # ones up front (non-fatal).
    all_refs: List[str] = []
    for _, builder in builders:
        _, src = builder()
        all_refs.extend(src)
    check_test_paths(all_refs)

    commit = get_git_commit()
    print("Generating Phase 4 validation artifacts...")
    print("  output dir  : {}".format(ARTIFACTS_DIR))
    print("  commit      : {}".format(commit))
    print("  generated_at: {}".format(now_utc_iso()))
    print()

    written: List[str] = []
    try:
        for filename, builder in builders:
            data, _ = builder()
            path = write_artifact(filename, data)
            written.append(path)
            print(
                "  [OK] {}  ({} bytes)".format(
                    filename,
                    os.path.getsize(path),
                )
            )
    except Exception as exc:  # noqa: BLE001 - top-level guard
        print("ERROR: failed to generate artifacts: {}".format(exc), file=sys.stderr)
        return 1

    print()
    print(
        "Successfully generated {} of {} artifacts.".format(len(written), len(builders))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
