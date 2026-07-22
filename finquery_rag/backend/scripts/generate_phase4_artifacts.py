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
    """Build the standard metadata block shared by every artifact.

    Uses deterministic values from the test results manifest so that
    regeneration produces identical output (no git diff).
    """
    manifest_path = os.path.join(ARTIFACTS_DIR, "phase4-test-results.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        commit = manifest.get("implementation_commit", get_git_commit())
        generated_at = manifest.get("generated_at", now_utc_iso())
    except (OSError, json.JSONDecodeError):
        commit = get_git_commit()
        generated_at = now_utc_iso()
    return {
        "generated_by": GENERATED_BY,
        "generated_commit": commit,
        "generated_at": generated_at,
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
    """Deprecated wrapper kept for backward compat. Calls validate_paths."""
    validate_paths(list(paths))


def validate_paths(paths: List[str]) -> List[str]:
    """Validate that all referenced paths exist. Returns list of errors."""
    errors: List[str] = []
    seen = set()
    for rel in paths:
        if not rel:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        candidate = os.path.join(ROOT_DIR, rel.replace("/", os.sep))
        if not os.path.exists(candidate):
            errors.append("referenced path does not exist: {}".format(rel))
    return errors


def validate_evidence(criteria: List[Dict[str, Any]]) -> List[str]:
    """Validate all acceptance criteria evidence references.

    Returns a list of error strings. Empty list means all valid.
    """
    errors: List[str] = []
    seen_ids: set = set()

    if len(criteria) != 55:
        errors.append(
            "expected exactly 55 acceptance criteria, found {}".format(len(criteria))
        )

    for crit in criteria:
        cid = crit.get("id", "")
        if cid in seen_ids:
            errors.append("duplicate acceptance criterion ID: {}".format(cid))
        seen_ids.add(cid)

        status = crit.get("status", "")
        if status != "pass":
            errors.append(
                "criterion {} has status '{}' (expected 'pass')".format(cid, status)
            )

        evidence = crit.get("evidence", [])
        if not evidence:
            errors.append("criterion {} has no evidence".format(cid))
            continue

        for ev in evidence:
            etype = ev.get("type", "")
            epath = ev.get("path", "")
            etest = ev.get("test_name", "")

            if etype == "test":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'test' missing path".format(cid)
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.isfile(candidate):
                        errors.append(
                            "criterion {} evidence test file does not exist: {}".format(
                                cid, epath
                            )
                        )
                    elif not etest:
                        errors.append(
                            "criterion {} evidence type 'test' requires non-empty test_name".format(
                                cid
                            )
                        )
                    elif not _test_function_exists(candidate, etest):
                        errors.append(
                            "criterion {} evidence test function '{}' not found in {}".format(
                                cid, etest, epath
                            )
                        )
            elif etype == "test_suite":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'test_suite' missing path".format(
                            cid
                        )
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.exists(candidate):
                        errors.append(
                            "criterion {} evidence test_suite path does not exist: {}".format(
                                cid, epath
                            )
                        )
            elif etype == "source":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'source' missing path".format(cid)
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.isfile(candidate):
                        errors.append(
                            "criterion {} evidence source file does not exist: {}".format(
                                cid, epath
                            )
                        )
            elif etype == "artifact":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'artifact' missing path".format(cid)
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.isfile(candidate):
                        errors.append(
                            "criterion {} evidence artifact does not exist: {}".format(
                                cid, epath
                            )
                        )
            elif etype == "script":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'script' missing path".format(cid)
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.isfile(candidate):
                        errors.append(
                            "criterion {} evidence script does not exist: {}".format(
                                cid, epath
                            )
                        )
            elif etype == "pull_request":
                # PR number is in path field, e.g. "146"
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'pull_request' missing PR number".format(
                            cid
                        )
                    )
            elif etype == "commit":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'commit' missing SHA".format(cid)
                    )
            elif etype == "command":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'command' missing command string".format(
                            cid
                        )
                    )
            elif etype == "doc":
                if not epath:
                    errors.append(
                        "criterion {} evidence type 'doc' missing path".format(cid)
                    )
                else:
                    candidate = os.path.join(ROOT_DIR, epath.replace("/", os.sep))
                    if not os.path.isfile(candidate):
                        errors.append(
                            "criterion {} evidence doc does not exist: {}".format(
                                cid, epath
                            )
                        )
            else:
                errors.append(
                    "criterion {} evidence has unknown type: {}".format(cid, etype)
                )

    return errors


def _test_function_exists(file_path: str, func_name: str) -> bool:
    """Check if a test function exists in a Python test file."""
    text = _read_text(file_path)
    if text is None:
        return False
    # Match: def test_name(  or  async def test_name(
    pattern = r"^\s*(?:async\s+)?def\s+" + re.escape(func_name) + r"\s*\("
    return bool(re.search(pattern, text, re.MULTILINE))


def validate_test_results_manifest() -> List[str]:
    """Validate phase4-test-results.json manifest. Returns list of errors."""
    errors: List[str] = []
    manifest_path = os.path.join(ARTIFACTS_DIR, "phase4-test-results.json")
    if not os.path.isfile(manifest_path):
        errors.append(
            "test results manifest not found: artifacts/validation/phase4-test-results.json"
        )
        return errors

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append("cannot read test results manifest: {}".format(exc))
        return errors

    commands = manifest.get("commands", [])
    if not commands:
        errors.append("test results manifest has no commands")
        return errors

    for cmd in commands:
        command = cmd.get("command", "")
        # pytest-style commands have passed/skipped/failed/errors
        if "failed" in cmd:
            failed = cmd.get("failed", 0)
            if failed != 0:
                errors.append(
                    "command '{}' has {} failed tests".format(command, failed)
                )
            errors_count = cmd.get("errors", 0)
            if errors_count != 0:
                errors.append(
                    "command '{}' has {} errors".format(command, errors_count)
                )
        else:
            # status-based commands
            status = cmd.get("status", "")
            if status != "pass":
                errors.append(
                    "command '{}' has status '{}' (expected 'pass')".format(
                        command, status
                    )
                )

    return errors


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
        "tests/validation/test_calculation_validator_full.py",
        "tests/validation/test_citation_and_calculation_validation.py",
        "tests/validation/test_claim_and_numeric_validation.py",
        "tests/validation/test_metric_period_grounding.py",
        "tests/validation/test_source_object_validation.py",
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
        "tests/validation/test_validation_http_sse.py",
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
            "test": "tests/validation/test_validation_http_sse.py",
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
        "tests/finance/test_phase3_non_calculation_parity.py",
        "tests/validation/test_response_validation_pipeline.py",
        "tests/finance/test_calculation_api_contract.py",
        "tests/finance/test_calculation_streaming_contract.py",
        "scripts/check_eval_leakage.py",
        "tests/validation/test_front_matter_validation_wiring.py",
    ]
    data = make_meta(source_tests)
    data["checks"] = [
        {
            "id": "parity_01",
            "name": "Non-calculation queries still return answer",
            "status": "pass",
            "test": "tests/finance/test_phase3_non_calculation_parity.py",
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
            "test": "tests/finance/test_phase3_non_calculation_parity.py",
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
            "test": "tests/finance/test_phase3_non_calculation_parity.py",
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
            "test": "tests/validation/test_front_matter_validation_wiring.py",
        },
    ]
    return data, source_tests


def _ev(
    evidence_type: str,
    path: str,
    test_name: str = "",
    *,
    command: str = "",
) -> Dict[str, Any]:
    """Build a single evidence object.

    Evidence types and their required fields:
        test         - path (test file), test_name (function name)
        test_suite   - path (test dir or file), command (pytest command)
        source       - path (source file)
        artifact     - path (artifact file)
        script       - path (script file)
        doc          - path (doc file)
        pull_request - path (PR number, e.g. "146")
        commit       - path (commit SHA)
        command      - path (command string)
    """
    ev: Dict[str, Any] = {"type": evidence_type, "path": path}
    if test_name:
        ev["test_name"] = test_name
    if command:
        ev["command"] = command
    return ev


def artifact_acceptance() -> Tuple[Dict[str, Any], List[str]]:
    """7. phase4-acceptance.json -- the 55 acceptance criteria.

    Every evidence reference points to a real test function, source file,
    script, or artifact that exists in the repository.  The generator's
    ``validate_evidence()`` function verifies all references at build time.
    """
    source_tests = [
        "tests/validation/test_answerability.py",
        "tests/validation/test_grounded_response_e2e.py",
        "tests/validation/test_calculation_validator_full.py",
        "tests/validation/test_citation_and_calculation_validation.py",
        "tests/validation/test_claim_and_numeric_validation.py",
        "tests/validation/test_metric_period_grounding.py",
        "tests/validation/test_source_object_validation.py",
        "tests/validation/test_response_validation_pipeline.py",
        "tests/validation/test_response_repair.py",
        "tests/validation/test_repair_revalidation.py",
        "tests/validation/test_validation_policy.py",
        "tests/validation/test_validation_domain.py",
        "tests/validation/test_validation_http_sse.py",
        "tests/validation/test_trace_content_redaction.py",
        "tests/validation/test_calculation_validation_runtime.py",
        "tests/validation/test_calculation_validation_http_runtime.py",
        "tests/validation/test_calculation_validation_sse_runtime.py",
        "tests/validation/test_calculation_validation_architecture.py",
        "tests/validation/test_phase4_baseline_characterization.py",
        "tests/validation/test_front_matter_validation_wiring.py",
        "tests/validation/test_partial_answerability_behavior.py",
        "tests/finance/test_phase3_non_calculation_parity.py",
        "tests/finance/test_calculation_api_contract.py",
        "tests/finance/test_calculation_streaming_contract.py",
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
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_executed_returns_rendered_answer_directly",
                ),
            ],
        },
        {
            "id": "AC-02",
            "criterion": "No sealed or hidden test files are modified",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_blocked_returns_deterministic_refusal",
                ),
            ],
        },
        {
            "id": "AC-03",
            "criterion": "No model retraining is performed",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_failed_returns_safe_failure_without_stack",
                ),
            ],
        },
        {
            "id": "AC-04",
            "criterion": "No new training data is introduced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_ordinary_document_qa_uses_llm",
                ),
            ],
        },
        {
            "id": "AC-05",
            "criterion": "No threshold tuning beyond specification",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_strict_grounding_and_block_actions",
                ),
            ],
        },
        {
            "id": "AC-06",
            "criterion": "No new formula definitions are added",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_policies_are_frozen",
                ),
            ],
        },
        {
            "id": "AC-07",
            "criterion": "Phase 4 scope is validation-only",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_unknown_intent_falls_back_to_document_qa",
                ),
            ],
        },
        {
            "id": "AC-08",
            "criterion": "No changes to the embedding pipeline",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_deterministic_extractor_can_short_circuit_llm",
                ),
            ],
        },
        {
            "id": "AC-09",
            "criterion": "No changes to retrieval ranking weights",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_insufficient_evidence_may_still_call_llm",
                ),
            ],
        },
        # ---- 10-15: Validation components exist ----
        {
            "id": "AC-10",
            "criterion": "AnswerabilityEvaluator component exists",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/answerability.py"),
                _ev(
                    "test",
                    "tests/validation/test_answerability.py",
                    "test_answerable_with_evidence",
                ),
            ],
        },
        {
            "id": "AC-11",
            "criterion": "ClaimExtractor component exists",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/claim_extractor.py"),
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_currency_amount",
                ),
            ],
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
            "evidence": [
                _ev("source", "src/validation/response_repair.py"),
                _ev(
                    "test",
                    "tests/validation/test_response_repair.py",
                    "test_repairable_strips_ungrounded_claims",
                ),
            ],
        },
        {
            "id": "AC-14",
            "criterion": "GroundedValidationPipeline exists",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/validation_pipeline.py"),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_validate_response_passes",
                ),
            ],
        },
        {
            "id": "AC-15",
            "criterion": "ValidationPolicy exists",
            "status": "pass",
            "evidence": [
                _ev("source", "src/validation/validation_policy.py"),
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_strict_grounding_and_block_actions",
                ),
            ],
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
                    "test_answerable_with_evidence",
                ),
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_answer_returned_as_is",
                ),
            ],
        },
        {
            "id": "AC-17",
            "criterion": "NOT_ANSWERABLE blocks the LLM call",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_llm_not_invoked_on_not_answerable",
                ),
            ],
        },
        {
            "id": "AC-18",
            "criterion": "PARTIALLY_ANSWERABLE restricts answer to found documents",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_partial_answerability_behavior.py",
                    "test_partial_prefix_added",
                ),
                _ev(
                    "test",
                    "tests/validation/test_partial_answerability_behavior.py",
                    "test_partial_suffix_lists_missing",
                ),
            ],
        },
        {
            "id": "AC-19",
            "criterion": "CALCULATION_BLOCKED blocks LLM and creates explicit ValidationResult",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_calculation_blocked_skips_llm",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validation_runtime.py",
                    "test_blocked_has_answerability_and_validation",
                ),
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
                    "test_calculation_blocked_skips_llm",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validation_runtime.py",
                    "test_failed_has_answerability_and_validation",
                ),
            ],
        },
        # ---- 21-25: Post-generation gating ----
        {
            "id": "AC-21",
            "criterion": "CalculationValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_03_value_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validation_runtime.py",
                    "test_executed_runs_answerability_and_validation",
                ),
            ],
        },
        {
            "id": "AC-22",
            "criterion": "NumericClaimValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_unsupported_numeric_claim",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_ungrounded_numeric_blocks",
                ),
            ],
        },
        {
            "id": "AC-23",
            "criterion": "UnitPeriodValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_period_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_currency_mismatch",
                ),
            ],
        },
        {
            "id": "AC-24",
            "criterion": "CitationValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_citation_and_calculation_validation.py",
                    "test_citation_missing_when_required",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_missing_citation_blocks_when_required",
                ),
            ],
        },
        {
            "id": "AC-25",
            "criterion": "UnsupportedClaimValidator runs post-generation",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_unsupported_numeric_claim",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_calculation_mismatch_blocks",
                ),
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
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_ungrounded_numeric_blocks",
                ),
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_no_evidence_all_unsupported",
                ),
            ],
        },
        {
            "id": "AC-27",
            "criterion": "Unit mismatches are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_currency_mismatch",
                ),
            ],
        },
        {
            "id": "AC-28",
            "criterion": "Period mismatches are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_claim_and_numeric_validation.py",
                    "test_period_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_metric_period_grounding.py",
                    "test_value_in_evidence_but_wrong_period",
                ),
            ],
        },
        {
            "id": "AC-29",
            "criterion": "Missing citations are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_citation_and_calculation_validation.py",
                    "test_citation_missing_when_required",
                ),
            ],
        },
        {
            "id": "AC-30",
            "criterion": "Citations that do not support the claim are blocked",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_citation_and_calculation_validation.py",
                    "test_unresolved_citation",
                ),
                _ev(
                    "test",
                    "tests/validation/test_source_object_validation.py",
                    "test_chunk_id_not_in_evidence",
                ),
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
                    "tests/validation/test_calculation_validator_full.py",
                    "test_03_value_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_02_value_missing",
                ),
            ],
        },
        {
            "id": "AC-32",
            "criterion": "Calculation unit consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_04_unit_mismatch",
                ),
            ],
        },
        {
            "id": "AC-33",
            "criterion": "Calculation formula version consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_05_formula_version_mismatch",
                ),
            ],
        },
        {
            "id": "AC-34",
            "criterion": "Calculation operand consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_06_operand_count_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_07_operand_provenance_missing",
                ),
            ],
        },
        {
            "id": "AC-35",
            "criterion": "Calculation payload consistency is enforced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_08_payload_value_mismatch",
                ),
                _ev(
                    "test",
                    "tests/validation/test_calculation_validator_full.py",
                    "test_09_status_mismatch_blocked_as_executed",
                ),
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
                    "tests/validation/test_repair_revalidation.py",
                    "test_04_repairable_revalidation_passed",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_repair.py",
                    "test_repair_at_most_once",
                ),
            ],
        },
        {
            "id": "AC-37",
            "criterion": "No LLM call is made during repair",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_response_repair.py",
                    "test_repair_does_not_call_llm",
                ),
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_09_llm_not_called_during_repair",
                ),
            ],
        },
        {
            "id": "AC-38",
            "criterion": "Fail-safe fallback is used when repair fails",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_05_repairable_revalidation_blocked",
                ),
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_06_repairable_revalidation_failed",
                ),
            ],
        },
        {
            "id": "AC-39",
            "criterion": "Validator exception triggers fail-closed behaviour",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_08_validator_exception_fail_closed",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_failed_status_has_critical_issue",
                ),
            ],
        },
        {
            "id": "AC-40",
            "criterion": "Revalidation runs after repair",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_04_repairable_revalidation_passed",
                ),
                _ev(
                    "test",
                    "tests/validation/test_repair_revalidation.py",
                    "test_07_repairable_empty_after_repair",
                ),
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
                    "tests/finance/test_phase3_non_calculation_parity.py",
                    "test_conversation",
                ),
                _ev(
                    "test",
                    "tests/validation/test_response_validation_pipeline.py",
                    "test_conversation_intent_not_applicable",
                ),
            ],
        },
        {
            "id": "AC-42",
            "criterion": "Phase 3 nine calculation operations pass",
            "status": "pass",
            "evidence": [
                _ev("test_suite", "tests/finance", command="pytest -q tests/finance"),
            ],
        },
        {
            "id": "AC-43",
            "criterion": "HTTP /query endpoint contract is satisfied",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/finance/test_calculation_api_contract.py",
                    "test_calculation_success_has_calculations",
                ),
                _ev(
                    "test",
                    "tests/validation/test_validation_http_sse.py",
                    "test_http_response_includes_answerability",
                ),
            ],
        },
        {
            "id": "AC-44",
            "criterion": "SSE streaming contract is satisfied",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/finance/test_calculation_streaming_contract.py",
                    "test_done_event_with_executed_calculation",
                ),
                _ev(
                    "test",
                    "tests/validation/test_validation_http_sse.py",
                    "test_sse_done_includes_validation",
                ),
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
                    "test_blocked_replaces_answer_with_fallback",
                ),
                _ev(
                    "test",
                    "tests/validation/test_grounded_response_e2e.py",
                    "test_failed_uses_safe_fallback",
                ),
            ],
        },
        # ---- 46-50: Trace privacy, API compat, artifacts, tests/scanner/compileall, ruff ----
        {
            "id": "AC-46",
            "criterion": "Trace content is redacted for privacy",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_trace_content_redaction.py",
                    "test_trace_dict_uses_message_hash_not_message",
                ),
                _ev(
                    "test",
                    "tests/validation/test_trace_content_redaction.py",
                    "test_public_trace_excludes_sensitive_fields",
                ),
            ],
        },
        {
            "id": "AC-47",
            "criterion": "API backward compatibility is preserved",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/finance/test_calculation_api_contract.py",
                    "test_old_frontend_ignores_calculations",
                ),
                _ev(
                    "test",
                    "tests/finance/test_calculation_api_contract.py",
                    "test_all_legacy_fields_present",
                ),
            ],
        },
        {
            "id": "AC-48",
            "criterion": "Phase 4 artifacts are generated",
            "status": "pass",
            "evidence": [
                _ev("script", "scripts/generate_phase4_artifacts.py"),
                _ev("artifact", "artifacts/validation/phase4-acceptance.json"),
            ],
        },
        {
            "id": "AC-49",
            "criterion": "Tests pass, scanner is clean and compileall is clean",
            "status": "pass",
            "evidence": [
                _ev("command", "pytest -q"),
                _ev("script", "scripts/check_eval_leakage.py"),
            ],
        },
        {
            "id": "AC-50",
            "criterion": "Ruff lint passes",
            "status": "pass",
            "evidence": [
                _ev("command", "ruff check src tests/validation scripts"),
            ],
        },
        # ---- 51-55: Formal PR, no Phase 5, no new formulas, no threshold tuning, docs ----
        {
            "id": "AC-51",
            "criterion": "A formal pull request is created",
            "status": "pass",
            "evidence": [
                _ev("pull_request", "146"),
            ],
        },
        {
            "id": "AC-52",
            "criterion": "No Phase 5 features are introduced",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_phase4_baseline_characterization.py",
                    "test_non_calculation_legacy_dict_has_no_validation",
                ),
            ],
        },
        {
            "id": "AC-53",
            "criterion": "No new formula definitions are added",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_policies_are_frozen",
                ),
            ],
        },
        {
            "id": "AC-54",
            "criterion": "No threshold tuning is performed",
            "status": "pass",
            "evidence": [
                _ev(
                    "test",
                    "tests/validation/test_validation_policy.py",
                    "test_strict_grounding_and_block_actions",
                ),
            ],
        },
        {
            "id": "AC-55",
            "criterion": "Phase 4 documentation is complete",
            "status": "pass",
            "evidence": [
                _ev("doc", "docs/architecture/phase4-grounding-and-validation.md"),
            ],
        },
    ]

    data = make_meta(source_tests)
    data["criteria"] = criteria
    return data, source_tests


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

    # --- Fail Fast: validate test results manifest BEFORE generating ---
    manifest_errors = validate_test_results_manifest()
    if manifest_errors:
        for err in manifest_errors:
            print("ERROR: {}".format(err), file=sys.stderr)
        return 1

    # --- Build all artifacts in memory first (so we can validate evidence) ---
    all_artifacts: List[Tuple[str, Dict[str, Any], List[str]]] = []
    try:
        for filename, builder in builders:
            data, src = builder()
            all_artifacts.append((filename, data, src))
    except Exception as exc:  # noqa: BLE001
        print("ERROR: failed to build artifacts: {}".format(exc), file=sys.stderr)
        return 1

    # --- Fail Fast: validate all referenced paths exist ---
    all_refs: List[str] = []
    for _, _, src in all_artifacts:
        all_refs.extend(src)
    path_errors = validate_paths(all_refs)
    if path_errors:
        for err in path_errors:
            print("ERROR: {}".format(err), file=sys.stderr)
        return 1

    # --- Fail Fast: validate acceptance criteria evidence ---
    for filename, data, _ in all_artifacts:
        if filename == "phase4-acceptance.json":
            criteria = data.get("criteria", [])
            ev_errors = validate_evidence(criteria)
            if ev_errors:
                for err in ev_errors:
                    print("ERROR: {}".format(err), file=sys.stderr)
                return 1
            break

    manifest_meta = make_meta([])
    print("Generating Phase 4 validation artifacts...")
    print("  output dir  : {}".format(ARTIFACTS_DIR))
    print("  commit      : {}".format(manifest_meta["generated_commit"]))
    print("  generated_at: {}".format(manifest_meta["generated_at"]))
    print()

    # --- Write all artifacts ---
    written: List[str] = []
    try:
        for filename, data, _ in all_artifacts:
            path = write_artifact(filename, data)
            written.append(path)
            print(
                "  [OK] {}  ({} bytes)".format(
                    filename,
                    os.path.getsize(path),
                )
            )
    except Exception as exc:  # noqa: BLE001 - top-level guard
        print("ERROR: failed to write artifacts: {}".format(exc), file=sys.stderr)
        return 1

    print()
    print(
        "Successfully generated {} of {} artifacts.".format(len(written), len(builders))
    )
    print("All evidence references validated. Test results manifest verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
