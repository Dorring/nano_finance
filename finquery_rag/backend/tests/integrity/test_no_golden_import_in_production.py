"""Verify production modules do not import golden labels, expected sources,
oracle modules, or eval fixtures.

Uses AST import scanning and keyword searches to detect contamination.
"""
import ast
import os
import sys


PRODUCTION_DIRS = [
    "src/services",
    "src/models",
]
PRODUCTION_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

FORBIDDEN_IMPORTS = [
    "evaluation.oracle_context",
    "eval.golden_smoke",
    "eval.predictions_smoke",
    "eval.baseline_smoke_report",
    "eval.real_eval_template",
    "eval.real_eval_labeling_template",
]

FORBIDDEN_KEYWORDS = [
    "expected_pages",
    "expected_sources",
    "expected_source",
    "golden_page",
    "oracle_page",
    "support_page",
]


def _python_files(root: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".py") and not fn.startswith("test_"):
                paths.append(os.path.join(dirpath, fn))
    return paths


def _scan_imports(filepath: str) -> list[str]:
    """Extract all import statements from a Python file using AST."""
    with open(filepath, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError:
            return ["<syntax_error>"]

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")
    return imports


def _scan_content(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as fh:
        return fh.read()


def test_no_oracle_import_in_services():
    """Production services must not import oracle_context."""
    services_dir = os.path.join(PRODUCTION_ROOT, "src", "services")
    if not os.path.isdir(services_dir):
        return  # Skip if not running from backend dir

    violations = []
    for filepath in _python_files(services_dir):
        imports = _scan_imports(filepath)
        content = _scan_content(filepath)
        for forbidden in FORBIDDEN_IMPORTS:
            if any(forbidden in imp for imp in imports):
                violations.append(f"{os.path.basename(filepath)}: imports {forbidden}")
            if forbidden in content:
                # Check if it's just in a comment
                lines = [l for l in content.split("\n") if forbidden in l and not l.strip().startswith("#")]
                if lines:
                    violations.append(f"{os.path.basename(filepath)}: references {forbidden}")

    assert not violations, (
        f"Production services must not import oracle/eval modules:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_golden_keywords_in_services():
    """Production services must not reference expected_sources/expected_pages etc."""
    services_dir = os.path.join(PRODUCTION_ROOT, "src", "services")
    if not os.path.isdir(services_dir):
        return

    violations = []
    for filepath in _python_files(services_dir):
        content = _scan_content(filepath)
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in content:
                # Check if it's only in comments
                lines = [l for l in content.split("\n") if keyword in l and not l.strip().startswith("#")]
                if lines:
                    violations.append(
                        f"{os.path.basename(filepath)}: line {content.split(chr(10)).index(lines[0]) + 1}: {keyword}"
                    )

    assert not violations, (
        f"Production services must not reference eval-specific keywords:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_supporting_source_page_in_ranking():
    """supporting_source_page must not appear in any production service."""
    services_dir = os.path.join(PRODUCTION_ROOT, "src", "services")
    if not os.path.isdir(services_dir):
        return

    violations = []
    for filepath in _python_files(services_dir):
        content = _scan_content(filepath)
        if "supporting_source_page" in content:
            lines = [
                i + 1 for i, l in enumerate(content.split("\n"))
                if "supporting_source_page" in l
                and not l.strip().startswith("#")
                and "Phase 1" not in l
            ]
            if lines:
                violations.append(
                    f"{os.path.basename(filepath)}: lines {lines}"
                )

    assert not violations, (
        f"supporting_source_page must not appear in production services:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_eval_fixture_import_in_production():
    """Production code must not import from eval/ fixture directory."""
    services_dir = os.path.join(PRODUCTION_ROOT, "src", "services")
    if not os.path.isdir(services_dir):
        return

    violations = []
    for filepath in _python_files(services_dir):
        imports = _scan_imports(filepath)
        for imp in imports:
            if imp.startswith("eval.") or imp == "eval":
                violations.append(f"{os.path.basename(filepath)}: imports {imp}")

    assert not violations, (
        f"Production code must not import eval fixtures:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
