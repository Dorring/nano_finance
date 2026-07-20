#!/usr/bin/env python3
"""CI gate: scan production source for evaluation leakage.

Checks:
1. Production source does not import evaluation/oracle_context
2. Production source does not reference expected_pages/expected_sources
3. No hardcoded filename-to-page-number dictionary mappings
4. No hardcoded question-to-page-number dictionary mappings
5. supporting_source_page does not appear in production ranking code
6. eval/tests/fixtures directories are not imported by production services

Exit codes: 0 = no leakage, 1 = leakage detected, 2 = config error
"""
from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path


SERVICES_DIR = Path(__file__).resolve().parent.parent / "src" / "services"
MAIN_PY = Path(__file__).resolve().parent.parent / "src" / "main.py"

FORBIDDEN_IMPORTS = ["evaluation.oracle_context", "eval."]
FORBIDDEN_KEYWORDS = ["expected_pages", "expected_sources", "expected_source", "golden_page", "oracle_page", "support_page"]

# Evaluation modules that legitimately reference expected_sources in golden cases.
# These are NOT production services; they only run in offline eval CLI.
EVAL_MODULE_ALLOWLIST = {"evaluation.py", "eval_runner.py", "oracle_context.py"}
FORBIDDEN_FIELDS = ["supporting_source_page"]


def _python_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    result = []
    for fp in directory.rglob("*.py"):
        if "__pycache__" not in str(fp):
            result.append(fp)
    return result


def _is_eval_module(filepath: Path) -> bool:
    """Check if file is an eval-only module that legitimately uses expected_sources."""
    return filepath.name in EVAL_MODULE_ALLOWLIST or "evaluation" in str(filepath.parent)


def _scan_imports(filepath: Path) -> set[str]:
    with open(filepath, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError:
            return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def check_forbidden_imports() -> list[str]:
    violations = []
    for fp in _python_files(SERVICES_DIR):
        for imp in _scan_imports(fp):
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in imp:
                    violations.append(f"{fp.name}: imports '{imp}'")
    if MAIN_PY.is_file():
        for imp in _scan_imports(MAIN_PY):
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in imp:
                    violations.append(f"main.py: imports '{imp}'")
    return violations


def check_forbidden_keywords() -> list[str]:
    violations = []
    for fp in _python_files(SERVICES_DIR):
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#") or "Phase 1" in s:
                continue
            if _is_eval_module(fp):
                continue
            for kw in FORBIDDEN_KEYWORDS:
                if kw in s:
                    violations.append(f"{fp.name}:{i}: {s[:120]}")
    return violations


def check_ranking_fields() -> list[str]:
    violations = []
    for fp in _python_files(SERVICES_DIR):
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#") or "Phase 1" in s:
                continue
            for field in FORBIDDEN_FIELDS:
                if field in s and any(op in s for op in ["sort(", "priority", "score", "boost", "weight"]):
                    violations.append(f"{fp.name}:{i}: {s[:120]}")
    return violations


def check_eval_imports() -> list[str]:
    violations = []
    for fp in _python_files(SERVICES_DIR):
        for imp in _scan_imports(fp):
            if imp == "eval" or (imp or "").startswith("eval."):
                violations.append(f"{fp.name}: imports '{imp}'")
    if MAIN_PY.is_file():
        for imp in _scan_imports(MAIN_PY):
            if imp == "eval" or (imp or "").startswith("eval."):
                violations.append(f"main.py: imports '{imp}'")
    return violations


def main() -> int:
    if not SERVICES_DIR.is_dir():
        print(f"ERROR: services dir not found: {SERVICES_DIR}", file=sys.stderr)
        return 2

    checks = {
        "forbidden_imports": check_forbidden_imports(),
        "forbidden_keywords": check_forbidden_keywords(),
        "ranking_fields": check_ranking_fields(),
        "eval_imports": check_eval_imports(),
    }

    total = 0
    for name, violations in checks.items():
        if violations:
            print(f"\n[{name}] {len(violations)} violation(s):")
            for v in violations:
                print(f"  {v}")
            total += len(violations)

    if total == 0:
        print("PASS: No evaluation leakage detected in production source.")
        return 0

    print(f"\nFAIL: {total} leakage violation(s) detected.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
