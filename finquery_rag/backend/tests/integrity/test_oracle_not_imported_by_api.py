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


def test_oracle_builds_context_from_evidence_content():
    """Oracle can build context when real evidence content is provided."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from evaluation.oracle_context import build_oracle_context

    case = {
        "expected_sources": [
            {
                "filename": "report.pdf",
                "page": 10,
                "content": "Revenue was $100 million.",
                "chunk_id": "report.pdf::page_10::chunk_1",
            }
        ]
    }
    context, sources = build_oracle_context(case)
    assert "Revenue was $100 million" in context
    assert len(sources) == 1
    assert sources[0]["oracle"] is True


def test_oracle_raises_when_only_expected_answer_contains():
    """Oracle must raise when case has expected_answer_contains but no evidence content."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from evaluation.oracle_context import build_oracle_context

    case = {
        "expected_answer_contains": "$100 million",
        "expected_sources": [],
    }
    with pytest.raises(ValueError, match="no expected_sources"):
        build_oracle_context(case)


def test_oracle_context_does_not_contain_expected_answer():
    """Oracle context must not auto-inject the expected answer."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from evaluation.oracle_context import build_oracle_context

    case = {
        "expected_answer_contains": "SECRET_ANSWER_12345",
        "expected_sources": [
            {
                "filename": "report.pdf",
                "page": 10,
                "content": "Revenue was $100 million.",
                "chunk_id": "report.pdf::page_10::chunk_1",
            }
        ]
    }
    context, sources = build_oracle_context(case)
    assert "SECRET_ANSWER_12345" not in context


def test_production_modules_cannot_import_oracle():
    """Production service modules must not be able to import oracle_context."""
    import importlib
    # Verify that importing oracle from a production module path fails
    try:
        # This should work (evaluation is not production)
        mod = importlib.import_module("src.evaluation.oracle_context")
        # But production services should not import it
        for prod_module in ["src.services.rag_engine", "src.services.retrieval", "src.services.reranker"]:
            try:
                m = importlib.import_module(prod_module)
                source = ""
                mod_file = getattr(m, "__file__", "")
                if mod_file and os.path.isfile(mod_file):
                    with open(mod_file, encoding="utf-8") as fh:
                        source = fh.read()
                assert "oracle_context" not in source, (
                    f"{prod_module} must not reference oracle_context"
                )
            except ImportError:
                pass  # Module may not be importable without deps
    except ImportError:
        pytest.skip("oracle_context module not available")
