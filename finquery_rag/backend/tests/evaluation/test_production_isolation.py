"""Tests that verify Phase 5 evaluation code does not pollute production
deployment.

These tests guard against accidentally introducing imports of
``src.evaluation`` (Phase 5 evaluation modules) into production source
code, copying sealed artifacts into the production Docker image, or
shipping evaluation-only dependencies (pytest, ruff, etc.) as production
requirements.

Pre-existing legacy imports of ``src.evaluation.evaluation`` (the
general-purpose evaluation utilities module that predates Phase 5) are
allow-listed for backward compatibility, matching the allowlist in
``scripts/check_sealed_eval_leakage.py``. Any *new* import of Phase 5
specific evaluation modules (calibration, ablation, sealed_scorer,
case_scorer, label_validator, etc.) by production code is a violation.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BACKEND_DIR / "src"
EVAL_DIR = SRC_DIR / "evaluation"

# Pre-existing legacy imports of the general-purpose evaluation module
# (``src.evaluation.evaluation``). These predate Phase 5 and are allowed
# for backward compatibility. Map relative path (relative to ``src/``)
# to the set of permitted module names (both relative and absolute
# forms).
IMPORT_ALLOWLIST: dict[str, set[str]] = {
    "main.py": {"evaluation.evaluation", "src.evaluation.evaluation"},
    "services/preflight.py": {"src.evaluation.evaluation"},
    "services/trace.py": {"src.evaluation.evaluation"},
}

# Production service directories that must never import Phase 5 eval.
PRODUCTION_SERVICE_DIRS = (
    "services",
    "application",
    "finance",
    "validation",
    "retrieval",
    "generation",
    "domain",
)

# Packages that are evaluation/dev-only and must not appear in production
# requirements (pyproject.toml [project.dependencies] or requirements.txt).
EVAL_ONLY_PACKAGES = {
    "pytest",
    "pytest-asyncio",
    "pytest-asyncio",
    "ruff",
    "mypy",
    "pytest-cov",
    "pytest-mock",
    "pytest-xdist",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_files(directory: Path) -> list[Path]:
    """Return all ``*.py`` files under ``directory`` (skipping __pycache__)."""
    if not directory.is_dir():
        return []
    files: list[Path] = []
    for path in directory.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        files.append(path)
    return files


def _is_inside_eval(filepath: Path) -> bool:
    """Return True if ``filepath`` lives under ``src/evaluation/``."""
    try:
        filepath.resolve().relative_to(EVAL_DIR.resolve())
        return True
    except ValueError:
        return False


def _scan_imports(filepath: Path) -> set[str]:
    """Return the set of module names imported by ``filepath`` via AST.

    Handles both ``import X`` and ``from X import Y`` forms. For relative
    imports (``from .X import Y``) the module is returned as-is (the
    caller must interpret level appropriately).
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            if level > 0:
                # Relative import: resolve against the file's package.
                # We only care whether it ultimately touches evaluation.
                # We store the raw module plus the relative form so the
                # caller can match either way.
                imports.add(module)  # e.g. "evaluation.evaluation"
            else:
                imports.add(module)
    return imports


def _resolve_relative_import(filepath: Path, module: str, level: int) -> str:
    """Resolve a relative import to an absolute module path.

    For example, a file ``src/main.py`` with ``from .evaluation.evaluation
    import X`` (level=1, module="evaluation.evaluation") resolves to
    ``src.evaluation.evaluation``.
    """
    if level == 0:
        return module
    # Walk up from the file's directory ``level`` times.
    parts = list(filepath.relative_to(SRC_DIR).parts)
    # Drop the filename.
    if parts:
        parts = parts[:-1]
    for _ in range(level - 1):
        if parts:
            parts = parts[:-1]
    prefix = ".".join(["src"] + [p for p in parts if p])
    if module:
        return f"{prefix}.{module}" if prefix else module
    return prefix


def _imports_evaluation_module(
    filepath: Path, imp_module: str, level: int
) -> str | None:
    """Return the matched evaluation module name, or None.

    Considers both the raw import string and the resolved absolute form
    against the evaluation package. Returns the matched module string
    (for allowlist comparison) or None if the import does not touch
    ``src.evaluation`` / ``evaluation``.
    """
    if level == 0:
        if imp_module in ("src.evaluation", "evaluation"):
            return imp_module
        if imp_module.startswith("src.evaluation.") or imp_module.startswith(
            "evaluation."
        ):
            return imp_module
        return None
    # Relative import — resolve and check.
    resolved = _resolve_relative_import(filepath, imp_module, level)
    if resolved in ("src.evaluation", "evaluation"):
        return resolved
    if resolved.startswith("src.evaluation.") or resolved.startswith("evaluation."):
        return resolved
    # Also check the raw module form for relative imports.
    # e.g. ``from .evaluation import X`` → module="evaluation", level=1
    if imp_module == "evaluation" or imp_module.startswith("evaluation."):
        return imp_module
    return None


def _scan_file_for_evaluation_imports(
    filepath: Path,
) -> list[tuple[str, str]]:
    """Return a list of (matched_module, raw_import) violations for file.

    Respects ``IMPORT_ALLOWLIST`` — allowlisted imports are not violations.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    try:
        rel = str(filepath.resolve().relative_to(SRC_DIR.resolve())).replace("\\", "/")
    except ValueError:
        rel = filepath.name
    allowed = IMPORT_ALLOWLIST.get(rel, set())
    violations: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                matched = _imports_evaluation_module(filepath, alias.name, 0)
                if matched and matched not in allowed:
                    violations.append((matched, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            level = node.level or 0
            matched = _imports_evaluation_module(filepath, module, level)
            if matched and matched not in allowed:
                raw = (
                    f"from {'.' * level}{module} import "
                    f"{', '.join(a.name for a in node.names)}"
                )
                violations.append((matched, raw))
    return violations


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProductionIsolation:
    """Verify production code is isolated from Phase 5 evaluation code."""

    def test_production_modules_do_not_import_evaluation(self) -> None:
        """No production module (outside src/evaluation/) may import from
        ``src.evaluation`` except for the legacy allowlisted files.

        Uses AST parsing (not string matching) to detect imports.
        """
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found — not running from backend")
        violations: list[str] = []
        files_scanned = 0
        for filepath in _python_files(SRC_DIR):
            if _is_inside_eval(filepath):
                continue
            files_scanned += 1
            for matched, raw in _scan_file_for_evaluation_imports(filepath):
                try:
                    rel = str(
                        filepath.resolve().relative_to(SRC_DIR.resolve())
                    ).replace("\\", "/")
                except ValueError:
                    rel = filepath.name
                violations.append(f"{rel}: {raw} (matches '{matched}')")
        assert files_scanned > 0, "no production Python files were scanned"
        assert not violations, (
            "Production modules must not import from src.evaluation "
            "(except legacy allowlisted files):\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_docker_production_excludes_eval_data(self) -> None:
        """If a Dockerfile exists, it must not COPY eval_data/ or sealed
        artifacts into the production image.

        Skips gracefully if no Dockerfile is present.
        """
        dockerfile = BACKEND_DIR / "Dockerfile"
        if not dockerfile.is_file():
            pytest.skip("no Dockerfile present")
        content = dockerfile.read_text(encoding="utf-8")
        forbidden_patterns = [
            "eval_data/",
            "eval_corpus/",
            ".sealed/",
            "artifacts/evaluation/",
            "indexes/phase5/",
        ]
        violations: list[str] = []
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not stripped.upper().startswith("COPY"):
                continue
            for pattern in forbidden_patterns:
                if pattern in stripped:
                    violations.append(
                        f"Dockerfile:{i}: COPY references '{pattern}': {stripped}"
                    )
        assert not violations, (
            "Dockerfile must not COPY eval/sealed data into production image:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_production_requirements_exclude_eval_only_packages(self) -> None:
        """Production requirements (pyproject.toml [project.dependencies]
        or requirements.txt) must not include evaluation-only packages
        like pytest, pytest-asyncio, or ruff.

        A separate requirements-dev.txt or similar is fine and is not
        checked here.
        """
        prod_deps = _collect_production_dependencies()
        if not prod_deps:
            pytest.skip(
                "no production requirements found (no pyproject.toml or "
                "requirements.txt)"
            )
        violations: list[str] = []
        for dep in prod_deps:
            # dep is normalized: lowercase, no version specifier, no extras.
            if dep in EVAL_ONLY_PACKAGES:
                violations.append(dep)
        assert not violations, (
            "Production requirements must not include evaluation-only "
            "packages:\n" + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_evaluation_imports_in_services(self) -> None:
        """Files under specific production service directories must NOT
        import from ``src.evaluation`` (except legacy allowlisted files).
        """
        if not SRC_DIR.is_dir():
            pytest.skip("src/ directory not found — not running from backend")
        violations: list[str] = []
        files_scanned = 0
        for sub in PRODUCTION_SERVICE_DIRS:
            sub_dir = SRC_DIR / sub
            if not sub_dir.is_dir():
                continue
            for filepath in _python_files(sub_dir):
                files_scanned += 1
                for matched, raw in _scan_file_for_evaluation_imports(filepath):
                    try:
                        rel = str(
                            filepath.resolve().relative_to(SRC_DIR.resolve())
                        ).replace("\\", "/")
                    except ValueError:
                        rel = filepath.name
                    violations.append(f"{rel}: {raw} (matches '{matched}')")
        assert files_scanned > 0, "no production service files were scanned"
        assert not violations, (
            "Production service modules must not import from src.evaluation:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_evaluation_feature_flags_default_true(self) -> None:
        """``EvaluationFeatureFlags`` must have exactly 9 boolean flags,
        all defaulting to ``True``.
        """
        try:
            from src.evaluation.schemas import EvaluationFeatureFlags
        except ImportError as exc:
            pytest.skip(f"cannot import EvaluationFeatureFlags: {exc}")
        fields = dataclasses.fields(EvaluationFeatureFlags)
        bool_fields = [f for f in fields if f.type is bool or f.type == "bool"]
        assert len(bool_fields) == 9, (
            f"expected 9 boolean flags, found {len(bool_fields)}: "
            f"{[f.name for f in bool_fields]}"
        )
        defaults = EvaluationFeatureFlags()
        for field in bool_fields:
            value = getattr(defaults, field.name)
            assert value is True, (
                f"flag '{field.name}' must default to True, got {value!r}"
            )


# ---------------------------------------------------------------------------
# Requirements parsing helpers
# ---------------------------------------------------------------------------


def _normalize_dep_name(raw: str) -> str:
    """Normalize a PEP 508 dependency specifier to a lowercase package name.

    Strips extras, version specifiers, markers, and whitespace.
    """
    name = raw.strip()
    # Strip environment markers (everything after ';').
    if ";" in name:
        name = name.split(";", 1)[0]
    # Strip extras (everything in [...] before any version).
    if "[" in name:
        name = name.split("[", 1)[0]
    # Strip version specifiers.
    for sep in ("==", ">=", "<=", "!=", "~=", ">", "<", "==="):
        if sep in name:
            name = name.split(sep, 1)[0]
            break
    return name.strip().lower().replace("_", "-")


def _collect_production_dependencies() -> list[str]:
    """Return normalized production dependency names.

    Reads ``pyproject.toml`` ``[project.dependencies]`` and/or
    ``requirements.txt``. Does NOT read ``requirements-dev.txt`` or
    optional dependency groups.
    """
    deps: list[str] = []
    pyproject = BACKEND_DIR / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            text = ""
        deps.extend(_parse_pyproject_dependencies(text))
    requirements = BACKEND_DIR / "requirements.txt"
    if requirements.is_file():
        try:
            text = requirements.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            normalized = _normalize_dep_name(line)
            if normalized:
                deps.append(normalized)
    return deps


def _parse_pyproject_dependencies(text: str) -> list[str]:
    """Extract ``[project.dependencies]`` entries from pyproject.toml text.

    Uses simple line-based parsing to avoid a tomllib dependency on older
    Python versions; the section is recognised by the header
    ``[project]`` followed by a ``dependencies = [...]`` block.
    """
    deps: list[str] = []
    lines = text.splitlines()
    in_project = False
    in_dependencies = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            in_dependencies = False
            continue
        if not in_project:
            continue
        if in_dependencies:
            if stripped.startswith("]"):
                in_dependencies = False
                continue
            # Remove trailing comma and surrounding quotes.
            item = stripped.rstrip(",").strip()
            if (item.startswith('"') and item.endswith('"')) or (
                item.startswith("'") and item.endswith("'")
            ):
                item = item[1:-1]
            if item:
                deps.append(_normalize_dep_name(item))
        elif stripped.startswith("dependencies") and "=" in stripped:
            rhs = stripped.split("=", 1)[1].strip()
            if rhs.startswith("["):
                # Inline list or multi-line list.
                inline = rhs[1:].strip()
                if inline.startswith("]"):
                    continue
                # Parse inline list items.
                inline = inline.rstrip("]").rstrip(",")
                for item in _split_inline_list(inline):
                    if item:
                        deps.append(_normalize_dep_name(item))
                if rhs.rstrip().endswith("]"):
                    in_dependencies = False
                else:
                    in_dependencies = True
    return deps


def _split_inline_list(text: str) -> list[str]:
    """Split a comma-separated list of quoted strings."""
    items: list[str] = []
    for token in text.split(","):
        token = token.strip()
        if (token.startswith('"') and token.endswith('"')) or (
            token.startswith("'") and token.endswith("'")
        ):
            token = token[1:-1]
        if token:
            items.append(token)
    return items
