"""Verify the system does not have query-keyword-specific page rules."""
import pytest
import os


def test_removed_methods_gone():
    """All leaked methods must be fully removed from RAGEngine."""
    from src.services.rag_engine import RAGEngine
    for method in [
        "_fallback_pages_for_query", "_supporting_pages_for_query",
        "_force_supporting_page_coverage", "_augment_with_page_fallbacks",
        "_ensure_supporting_sources", "_ensure_page_fallback_coverage",
        "answer_multi_doc_query_from_context",
    ]:
        assert not hasattr(RAGEngine, method), f"RAGEngine.{method} must not exist"


def test_no_hardcoded_page_lists_in_source():
    """No hardcoded page number lists tied to document filenames."""
    services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "services")
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")

    with open(os.path.join(services_dir, "rag_engine.py"), encoding="utf-8") as f:
        content = f.read()

    forbidden_docs = ["final annual report", "wipo_pub_rn2021", "leac203", "pdf solutions"]
    lines = content.split("\n")
    violations = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#") or "Phase 1" in line:
            continue
        for doc in forbidden_docs:
            if doc in line.lower():
                violations.append(f"Line {i}: {line.strip()[:120]}")
                break
    
    assert not violations, (
        f"Benchmark document names found in rag_engine.py:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_no_benchmark_answer_values():
    """No hardcoded benchmark answer values in production code."""
    services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "services")
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")

    with open(os.path.join(services_dir, "rag_engine.py"), encoding="utf-8") as f:
        content = f.read()

    forbidden_values = ["143,540", "387,063", "468,272", "42.2 million"]
    lines = content.split("\n")
    violations = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("#") or "Phase 1" in line:
            continue
        clean_line = line.replace(",", "").replace(" ", "").lower()
        for val in forbidden_values:
            if val.replace(",", "").replace(" ", "").lower() in clean_line:
                violations.append(f"Line {i}: {line.strip()[:120]}")
                break
    
    assert not violations, (
        f"Hardcoded benchmark values found in rag_engine.py:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
