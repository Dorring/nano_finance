"""Verify critical service modules are importable and free of heavy deps.

Tests are split into pure-Python checks (always run) and import checks
(only run when optional dependencies like jose are available).
"""
import importlib
import os
import pytest


# Modules that should be importable without jose/openai/chromadb
PURE_MODULES = [
    "src.services.intent",
    "src.services.financial_tools",
    "src.services.answer_validation",
    "src.services.chunk_id",
    "src.services.query_scope",
    "src.services.retrieval_config",
    "src.services.streaming",
    "src.services.sqlite_migrations",
]

# Modules that depend on heavier stack (jose, openai, chromadb, pymupdf, etc.)
FULL_STACK_MODULES = [
    "src.services.auth",
    "src.services.ingest",
    "src.services.vector_store",
    "src.services.rag_engine",
    "src.services.retrieval",
    "src.services.reranker",
    "src.services.evaluation",
    "src.services.eval_runner",
    "src.services.trace",
    "src.services.session_manager",
    "src.services.document_registry",
    "src.services.feedback",
    "src.services.health",
    "src.services.preflight",
    "src.services.migration_audit",
    "src.services.memory_profile",
    "src.services.process_tables",
    "src.models.schemas",
    "src.models.user",
    "src.database",
    "src.eval_cli",
]

ALL_MODULES = PURE_MODULES + FULL_STACK_MODULES


def _source_path(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec and spec.origin:
        return spec.origin
    parts = module_name.split(".")
    return os.path.join("src", *parts[1:]) + ".py"


def test_all_module_files_exist():
    """Every critical module must have a corresponding .py file on disk."""
    missing = []
    for module_name in ALL_MODULES:
        path = _source_path(module_name)
        if not os.path.isfile(path):
            alt_path = path.replace("src/", "finquery_rag/backend/src/")
            if not os.path.isfile(alt_path):
                missing.append(f"{module_name} -> {path}")
    assert not missing, (
        f"{len(missing)} module file(s) not found:\n"
        + "\n".join(f"  {m}" for m in missing)
    )


def test_pure_modules_import():
    """Modules without heavy deps must be importable."""
    failures = []
    for module_name in PURE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, f"{type(exc).__name__}: {exc}"))
    assert not failures, (
        f"{len(failures)} pure module(s) failed to import:\n"
        + "\n".join(f"  {name}: {err}" for name, err in failures)
    )


@pytest.mark.slow
def test_full_stack_modules_import():
    """Modules with deps must be importable when venv is available."""
    try:
        importlib.import_module("jose")
        have_jose = True
    except ImportError:
        have_jose = False

    if not have_jose:
        pytest.skip("Full-stack deps (jose) not available in this environment")

    failures = []
    for module_name in FULL_STACK_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            failures.append((module_name, f"{type(exc).__name__}: {exc}"))
    assert not failures, (
        f"{len(failures)} full-stack module(s) failed to import:\n"
        + "\n".join(f"  {name}: {err}" for name, err in failures)
    )


def test_evaluation_no_heavy_deps_in_source():
    """evaluation.py must not import torch or chromadb at module level."""
    path = _source_path("src.services.evaluation")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    forbidden = ["import torch", "from torch", "import chromadb", "from chromadb"]
    for token in forbidden:
        assert token not in content, (
            f"evaluation.py must not import heavy deps, found: {token}"
        )


def test_financial_tools_no_llm_deps_in_source():
    """financial_tools.py must remain pure (no openai, torch imports)."""
    path = _source_path("src.services.financial_tools")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    forbidden = ["import openai", "from openai", "import torch", "from torch"]
    for token in forbidden:
        assert token not in content, (
            f"financial_tools.py must remain pure, found: {token}"
        )


def test_eval_runner_no_heavy_deps_in_source():
    """eval_runner.py must not import model-heavy libs at module level."""
    path = _source_path("src.services.eval_runner")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    forbidden = [
        "import chromadb",
        "import sentence_transformers",
        "import camelot",
    ]
    for token in forbidden:
        assert token not in content, (
            f"eval_runner.py must not import heavy deps, found: {token}"
        )


def test_intent_no_heavy_deps_in_source():
    """intent.py must remain pure dependency-free."""
    path = _source_path("src.services.intent")
    with open(path, encoding="utf-8") as fh:
        content = fh.read()
    forbidden = ["import torch", "import openai", "import chromadb"]
    for token in forbidden:
        assert token not in content, (
            f"intent.py must remain pure, found: {token}"
        )
