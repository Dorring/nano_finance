#!/usr/bin/env python3
"""CI gate: scan production source for evaluation leakage.

Checks:
1. Production source does not import evaluation/oracle_context
2. Production source does not reference expected_pages/expected_sources
3. No hardcoded filename-to-page-number dictionary mappings
4. No hardcoded question-to-page-number dictionary mappings
5. supporting_source_page / page_fallback do not appear in production ranking code
6. eval/tests/fixtures directories are not imported by production services
7. No hardcoded filename-to-answer mappings
8. No benchmark entity names in production code
9. SyntaxError in production source is a scan failure

Exit codes: 0 = no leakage, 1 = leakage detected, 2 = config error
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
EVAL_DIR_PATH = SRC_DIR / "evaluation"

FORBIDDEN_IMPORTS = [
    "evaluation.oracle_context",
    "eval.",
    "tests.fixtures",
]
FORBIDDEN_KEYWORDS = [
    "expected_pages",
    "expected_sources",
    "expected_source",
    "golden_page",
    "oracle_page",
    "support_page",
]
FORBIDDEN_FIELDS = ["supporting_source_page", "page_fallback"]

# Benchmark document names that must not appear in production code
BENCHMARK_ENTITY_NAMES = [
    "FINAL Annual Report.pdf",
    "wipo_pub_rn2021_18e.pdf",
    "leac203.pdf",
]

# Patterns for hardcoded filename-to-page mappings
FILENAME_PAGE_PATTERN = re.compile(
    r'"(?:' + "|".join(re.escape(n) for n in BENCHMARK_ENTITY_NAMES) + r')".*\[\d+',
    re.IGNORECASE,
)

# Patterns for hardcoded filename-to-answer mappings
FILENAME_ANSWER_PATTERN = re.compile(
    r'"(?:' + "|".join(re.escape(n) for n in BENCHMARK_ENTITY_NAMES) + r')".*:\s*\d+[\d,]*',
    re.IGNORECASE,
)


class SourceParseError(RuntimeError):
    """Raised when a production source file has a SyntaxError."""
    pass


def _python_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    result = []
    for fp in directory.rglob("*.py"):
        if "__pycache__" not in str(fp):
            result.append(fp)
    return result


def _is_eval_module(filepath: Path) -> bool:
    """Only src/evaluation/ can legitimately use expected_sources."""
    try:
        filepath.resolve().relative_to(EVAL_DIR_PATH.resolve())
        return True
    except ValueError:
        return False


def production_python_files() -> list[Path]:
    """All production Python files under src/, excluding src/evaluation/."""
    return [
        path
        for path in _python_files(SRC_DIR)
        if not _is_eval_module(path)
    ]


def _scan_imports(filepath: Path) -> set[str]:
    with open(filepath, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError as exc:
            raise SourceParseError(
                f"{filepath}:{exc.lineno}: SyntaxError: {exc.msg}"
            ) from exc
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    return imports


def _matches_forbidden(imp: str, forbidden: str) -> bool:
    """Check if import matches forbidden module using boundary matching.

    Uses module-boundary matching so 'eval.' does not match 'retrieval.'.
    A match requires the forbidden string to be a complete module path
    prefix, not an arbitrary substring.
    """
    if not forbidden:
        return False
    # Exact match
    if imp == forbidden:
        return True
    # Module path prefix: 'eval' matches 'eval.foo' but not 'retrieval'
    # Strip trailing '.' from forbidden for clean prefix matching
    prefix = forbidden.rstrip(".")
    return imp == prefix or imp.startswith(prefix + ".")


def check_forbidden_imports() -> list[str]:
    violations = []
    for fp in production_python_files():
        for imp in _scan_imports(fp):
            for forbidden in FORBIDDEN_IMPORTS:
                if _matches_forbidden(imp, forbidden):
                    violations.append(f"{fp.relative_to(SRC_DIR)}: imports '{imp}'")
    return violations


def check_forbidden_keywords() -> list[str]:
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for kw in FORBIDDEN_KEYWORDS:
                if kw in s:
                    violations.append(f"{fp.relative_to(SRC_DIR)}:{i}: {s[:120]}")
    return violations


def check_ranking_fields() -> list[str]:
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for field in FORBIDDEN_FIELDS:
                if field in s and any(op in s for op in ["sort(", "priority", "score", "boost", "weight"]):
                    violations.append(f"{fp.relative_to(SRC_DIR)}:{i}: {s[:120]}")
    return violations


def check_eval_imports() -> list[str]:
    violations = []
    for fp in production_python_files():
        for imp in _scan_imports(fp):
            if imp == "eval" or (imp or "").startswith("eval."):
                violations.append(f"{fp.relative_to(SRC_DIR)}: imports '{imp}'")
    return violations


def check_benchmark_entities() -> list[str]:
    """Check for benchmark document names in production code."""
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for name in BENCHMARK_ENTITY_NAMES:
                if name in s:
                    violations.append(f"{fp.relative_to(SRC_DIR)}:{i}: benchmark entity '{name}'")
    return violations


def check_hardcoded_answers() -> list[str]:
    """Check for hardcoded numeric answers in production code."""
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if FILENAME_ANSWER_PATTERN.search(s):
                violations.append(f"{fp.relative_to(SRC_DIR)}:{i}: hardcoded filename-answer: {s[:120]}")
    return violations


def check_filename_page_mappings() -> list[str]:
    """Check for hardcoded filename-to-page-number mappings."""
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            lines = fh.readlines()
        for i, line in enumerate(lines, 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if FILENAME_PAGE_PATTERN.search(s):
                violations.append(f"{fp.relative_to(SRC_DIR)}:{i}: filename-page mapping: {s[:120]}")
    return violations


def check_syntax_errors() -> list[str]:
    """Check for SyntaxErrors in production source files."""
    violations = []
    for fp in production_python_files():
        with open(fp, encoding="utf-8") as fh:
            try:
                ast.parse(fh.read())
            except SyntaxError as exc:
                violations.append(f"{fp.relative_to(SRC_DIR)}:{exc.lineno}: SyntaxError: {exc.msg}")
    return violations


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"ERROR: src dir not found: {SRC_DIR}", file=sys.stderr)
        return 2

    prod_files = production_python_files()
    print(f"Scanning {len(prod_files)} production Python files under {SRC_DIR}")

    checks = {
        "forbidden_imports": check_forbidden_imports(),
        "forbidden_keywords": check_forbidden_keywords(),
        "ranking_fields": check_ranking_fields(),
        "eval_imports": check_eval_imports(),
        "benchmark_entities": check_benchmark_entities(),
        "hardcoded_answers": check_hardcoded_answers(),
        "filename_page_mappings": check_filename_page_mappings(),
        "syntax_errors": check_syntax_errors(),
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
