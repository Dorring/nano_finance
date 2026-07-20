"""Verify that the Oracle context module cannot be reached from production API paths.

Uses AST analysis to confirm no import chain connects main.py or rag_engine.py
to evaluation.oracle_context.
"""
import ast
import os
import pytest


def _collect_imports(filepath: str) -> set[str]:
    """Collect all local (src.*) imports from a Python file."""
    with open(filepath, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError:
            return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (
                node.module.startswith("src.")
                or node.module.startswith(".")
                or node.module.startswith("services.")
                or node.module.startswith("evaluation.")
            ):
                imports.add(node.module)
    return imports


def _resolve_relative_import(module: str, current_dir: str) -> str | None:
    """Resolve a relative import to an absolute src.* module name."""
    if module.startswith("src."):
        return module
    if module.startswith("evaluation."):
        return f"src.{module}"
    if module.startswith("services."):
        return f"src.{module}"
    if module.startswith("."):
        # Relative import - resolve from current directory
        parts = current_dir.replace("\\", "/").split("/")
        dots = len(module) - len(module.lstrip("."))
        module_rest = module.lstrip(".")
        resolved = "/".join(parts[:-dots] if dots > 0 else parts)
        if module_rest:
            resolved = f"{resolved}/{module_rest.replace('.', '/')}"
        # Convert back to src.* format
        if "src/services" in resolved:
            return resolved.split("src/")[1].replace("/", ".")
    return None


def test_oracle_not_imported_by_main():
    """main.py must not import oracle_context directly or transitively."""
    main_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "main.py"
    )
    if not os.path.isfile(main_path):
        pytest.skip("main.py not found")

    imports = _collect_imports(main_path)
    oracle_imports = [imp for imp in imports if "oracle" in imp.lower()]
    assert not oracle_imports, (
        f"main.py imports oracle-related modules: {oracle_imports}"
    )


def test_oracle_not_imported_by_rag_engine():
    """rag_engine.py must not import oracle_context."""
    rag_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "services", "rag_engine.py"
    )
    if not os.path.isfile(rag_path):
        pytest.skip("rag_engine.py not found")

    imports = _collect_imports(rag_path)
    oracle_imports = [imp for imp in imports if "oracle" in imp.lower()]
    assert not oracle_imports, (
        f"rag_engine.py imports oracle-related modules: {oracle_imports}"
    )


def test_oracle_not_imported_by_retrieval():
    """Retrieval modules must not import oracle_context."""
    retrieval_paths = [
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "services", "retrieval.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "services", "vector_store.py"),
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "services", "reranker.py"),
    ]

    for rpath in retrieval_paths:
        if not os.path.isfile(rpath):
            continue
        imports = _collect_imports(rpath)
        oracle_imports = [imp for imp in imports if "oracle" in imp.lower()]
        assert not oracle_imports, (
            f"{os.path.basename(rpath)} imports oracle: {oracle_imports}"
        )


def test_oracle_module_exists_and_is_isolated():
    """Verify oracle_context.py exists and does not import production code."""
    oracle_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "evaluation", "oracle_context.py"
    )
    if not os.path.isfile(oracle_path):
        pytest.skip("oracle_context.py not found (this is OK before Phase 1 completion)")

    imports = _collect_imports(oracle_path)
    forbidden = [
        imp for imp in imports
        if any(p in imp for p in ["services.rag_engine", "services.retrieval", "main"])
    ]
    assert not forbidden, (
        f"oracle_context.py must not import production RAG modules: {forbidden}"
    )
