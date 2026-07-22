#!/usr/bin/env python3
"""Sealed evaluation leakage scanner for Phase 5.

Scans the codebase to ensure sealed test data never leaks into production
code. The scanner checks:

1. Production source modules under ``src/`` (excluding ``src/evaluation/``)
   must NOT import ``src.evaluation`` or ``evaluation`` via
   ``from src.evaluation ...``, ``import src.evaluation ...`` or
   ``from evaluation ...``.
2. Sealed Case IDs (``sealed_*`` prefix, or Case IDs loaded from
   ``eval_data/phase5/sealed/questions.jsonl`` when present) must not
   appear in production code.
3. Sealed complete questions (loaded from the sealed questions file) must
   not appear verbatim in production code.
4. Expected-number / expected-source / golden-page keywords must not
   appear in production code.
5. Label file path markers (``.sealed/``, ``labels.jsonl``) must not
   appear in production code.
6. Evaluation-specific document aliases must not appear in production code.

Scanned directories:
    - src/**           (excluding src/evaluation/**)
    - config/**
    - prompts/**
    - scripts/         (excluding evaluation scripts)

Allowed locations where sealed data *may* appear:
    - src/evaluation/**
    - tests/evaluation/**
    - eval_data/**
    - artifacts/evaluation/**
    - docs/evaluation/**

Exit codes:
    0 = no leakage detected
    1 = leakage detected
    2 = configuration error
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BACKEND_DIR / "src"
CONFIG_DIR = BACKEND_DIR / "config"
PROMPTS_DIR = BACKEND_DIR / "prompts"
SCRIPTS_DIR = BACKEND_DIR / "scripts"
SEALED_QUESTIONS_FILE = (
    BACKEND_DIR / "eval_data" / "phase5" / "sealed" / "questions.jsonl"
)

# Allowed sealed-data directories (relative to backend root).
ALLOWED_SEALED_DIRS = [
    "src/evaluation",
    "tests/evaluation",
    "eval_data",
    "artifacts/evaluation",
    "docs/evaluation",
]

# Pre-existing imports of evaluation utilities in production code.
#
# These modules import general-purpose utility functions (write_jsonl,
# evaluate_predictions, compare_reports, etc.) from src.evaluation.evaluation.
# They do NOT access sealed data, Phase 5 labels, or sealed case IDs.
# They predate Phase 5 and are allowed for backward compatibility.
#
# The allowlist maps relative file paths (relative to src/) to the set of
# import module names that are permitted.
IMPORT_ALLOWLIST: dict[str, set[str]] = {
    "main.py": {"evaluation.evaluation", "src.evaluation.evaluation"},
    "services/preflight.py": {"src.evaluation.evaluation"},
    "services/trace.py": {"src.evaluation.evaluation"},
}

# Keywords that indicate sealed/expected answer data.
SEALED_KEYWORDS = [
    "expected_pages",
    "expected_sources",
    "expected_source",
    "expected_number",
    "expected_numbers",
    "golden_page",
    "golden_pages",
    "oracle_page",
    "oracle_pages",
    "support_page",
    "supporting_source_page",
]

# Label file path markers.
LABEL_PATH_MARKERS = [
    ".sealed/",
    "labels.jsonl",
    "seal_labels.jsonl",
    "expected_labels.jsonl",
]

# Evaluation-specific document aliases that must not appear in production.
EVAL_DOC_ALIASES = [
    "sealed_eval_doc",
    "eval_benchmark_doc",
    "phase5_eval_corpus",
]

# Regex for sealed Case IDs (``sealed_`` prefix).
SEALED_CASE_ID_PATTERN = re.compile(r"sealed_[A-Za-z0-9_\-]+")


class SourceParseError(RuntimeError):
    """Raised when a production source file cannot be parsed."""


def _python_files(directory: Path) -> list[Path]:
    """Return all Python files under ``directory`` (skipping __pycache__)."""
    if not directory.is_dir():
        return []
    result: list[Path] = []
    for fp in directory.rglob("*.py"):
        if "__pycache__" in fp.name:
            continue
        result.append(fp)
    return result


def _is_eval_script(path: Path) -> bool:
    """Return True if the script file is an evaluation script.

    Evaluation scripts are identified by ``eval`` or ``phase5`` in the
    filename (case-insensitive). These scripts may legitimately reference
    sealed data and are excluded from the scripts/ scan.
    """
    name = path.name.lower()
    return "eval" in name or "phase5" in name


def _is_inside_eval_dir(filepath: Path, src_dir: Path) -> bool:
    """Return True if ``filepath`` lives under ``src_dir/evaluation/``."""
    eval_dir = (src_dir / "evaluation").resolve()
    try:
        filepath.resolve().relative_to(eval_dir)
        return True
    except ValueError:
        return False


def _scan_imports(filepath: Path) -> set[str]:
    """Return the set of module names imported by the given Python file."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SourceParseError(f"{filepath}: cannot read: {exc}") from exc
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourceParseError(
            f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"
        ) from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def _load_sealed_questions(sealed_file: Path) -> list[str] | None:
    """Load sealed questions from ``questions.jsonl``.

    Returns ``None`` if the file is absent. Returns a list of question
    strings (deduplicated, order preserved) when the file exists.
    """
    if not sealed_file.is_file():
        return None
    questions: list[str] = []
    seen: set[str] = set()
    try:
        text = sealed_file.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        q = obj.get("question") or obj.get("query")
        if isinstance(q, str) and q.strip():
            qv = q.strip()
            if qv not in seen:
                seen.add(qv)
                questions.append(qv)
    return questions if questions else None


def _imports_evaluation(imp: str) -> bool:
    """Return True if the import string references the evaluation module."""
    if imp in ("src.evaluation", "evaluation"):
        return True
    return imp.startswith("src.evaluation.") or imp.startswith("evaluation.")


def check_production_imports_evaluation(src_dir: Path) -> list[str]:
    """Check that production modules don't import evaluation.

    Production modules under ``src/`` (excluding ``src/evaluation/``) must
    not import ``src.evaluation`` or ``evaluation``. Returns a list of
    violation strings (empty if clean).
    """
    violations: list[str] = []
    if not src_dir.is_dir():
        return violations
    for fp in _python_files(src_dir):
        if _is_inside_eval_dir(fp, src_dir):
            continue
        try:
            imports = _scan_imports(fp)
        except SourceParseError as exc:
            violations.append(str(exc))
            continue
        flagged = {imp for imp in imports if _imports_evaluation(imp)}
        if not flagged:
            continue
        try:
            rel = fp.resolve().relative_to(src_dir.resolve())
        except ValueError:
            rel = fp
        rel_str = str(rel).replace("\\", "/")
        allowed = IMPORT_ALLOWLIST.get(rel_str, set())
        flagged -= allowed
        if not flagged:
            continue
        for imp in sorted(flagged):
            violations.append(f"{rel}: imports '{imp}'")
    return violations


def _scan_text_for_sealed(
    text: str, rel_path: Path, sealed_questions: list[str] | None,
    prefix: str = "",
) -> list[str]:
    """Scan source text for sealed patterns. Returns violation strings."""
    violations: list[str] = []
    sealed_qs_set = {q for q in (sealed_questions or []) if len(q) >= 6}
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for m in SEALED_CASE_ID_PATTERN.finditer(stripped):
            token = m.group(0)
            violations.append(
                f"{prefix}{rel_path}:{i}: sealed Case ID pattern '{token}'"
            )
        for kw in SEALED_KEYWORDS:
            if kw in stripped:
                violations.append(
                    f"{prefix}{rel_path}:{i}: sealed keyword '{kw}'"
                )
        for marker in LABEL_PATH_MARKERS:
            if marker in stripped:
                violations.append(
                    f"{prefix}{rel_path}:{i}: label path marker '{marker}'"
                )
        for alias in EVAL_DOC_ALIASES:
            if alias in stripped:
                violations.append(
                    f"{prefix}{rel_path}:{i}: eval document alias '{alias}'"
                )
        for q in sealed_qs_set:
            if q in line:
                violations.append(
                    f"{prefix}{rel_path}:{i}: sealed question verbatim"
                )
    return violations


def check_sealed_patterns(
    src_dir: Path, sealed_questions: list[str] | None,
) -> list[str]:
    """Check for sealed question/case_id patterns in production code.

    Scans all Python files under ``src/`` (excluding ``src/evaluation/``).
    Returns a list of violation strings (empty if clean).
    """
    violations: list[str] = []
    if not src_dir.is_dir():
        return violations
    for fp in _python_files(src_dir):
        if _is_inside_eval_dir(fp, src_dir):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = fp.resolve().relative_to(src_dir.resolve())
        except ValueError:
            rel = fp
        violations.extend(
            _scan_text_for_sealed(text, rel, sealed_questions, prefix="src/")
        )
    return violations


def _check_sealed_in_directory(
    d: Path, sealed_questions: list[str] | None,
) -> list[str]:
    """Check a non-src directory (config, prompts, scripts) for sealed data."""
    violations: list[str] = []
    if not d.is_dir():
        return violations
    for fp in _python_files(d):
        if d == SCRIPTS_DIR and _is_eval_script(fp):
            continue
        try:
            text = fp.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            rel = fp.resolve().relative_to(d.resolve())
        except ValueError:
            rel = fp
        violations.extend(
            _scan_text_for_sealed(
                text, rel, sealed_questions, prefix=f"{d.name}/"
            )
        )
    return violations


def main() -> int:
    """Run all checks. Return 0 if clean, 1 if leakage found."""
    if not SRC_DIR.is_dir():
        print(f"ERROR: src dir not found: {SRC_DIR}", file=sys.stderr)
        return 2

    sealed_questions = _load_sealed_questions(SEALED_QUESTIONS_FILE)

    all_violations: list[str] = []
    all_violations.extend(check_production_imports_evaluation(SRC_DIR))
    all_violations.extend(check_sealed_patterns(SRC_DIR, sealed_questions))
    all_violations.extend(_check_sealed_in_directory(CONFIG_DIR, sealed_questions))
    all_violations.extend(_check_sealed_in_directory(PROMPTS_DIR, sealed_questions))
    all_violations.extend(_check_sealed_in_directory(SCRIPTS_DIR, sealed_questions))

    if all_violations:
        print(
            f"FAIL: {len(all_violations)} sealed-leakage violation(s) detected.",
            file=sys.stderr,
        )
        for v in all_violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    if sealed_questions:
        print(
            f"PASS: No sealed evaluation leakage detected "
            f"(checked {len(sealed_questions)} sealed questions)."
        )
    else:
        print(
            "PASS: No sealed evaluation leakage detected "
            "(no sealed questions file; checked import rules and patterns)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
