"""Verify that document filename changes do not alter retrieval results."""
import pytest
import os


def test_removed_methods_not_accessible():
    """All leaked methods must be fully removed from RAGEngine."""
    from src.services.rag_engine import RAGEngine
    removed = [
        "_fallback_pages_for_query",
        "_supporting_pages_for_query",
        "_force_supporting_page_coverage",
        "_augment_with_page_fallbacks",
        "_ensure_supporting_sources",
        "_ensure_page_fallback_coverage",
        "answer_multi_doc_query_from_context",
    ]
    for method in removed:
        assert not hasattr(RAGEngine, method), f"RAGEngine.{method} must not exist after Phase 1"


def test_no_filename_based_page_rules():
    """Verify no code path selects pages based on document filename alone."""
    import re
    services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "services")
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")
    
    violations = []
    for dirpath, _, filenames in os.walk(services_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    stripped = line.strip()
                    if stripped.startswith("#") or "Phase 1" in stripped:
                        continue
                    # Check for hardcoded filename-to-page-number patterns
                    if re.search(r'"(?:FINAL Annual Report|wipo_pub_rn2021|leac203)\.pdf".*\[\d+', stripped, re.IGNORECASE):
                        violations.append(f"{fn}:{i}: {stripped[:120]}")
    
    assert not violations, (
        f"Hardcoded filename-to-page mappings found:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_supporting_source_page_not_in_production():
    """supporting_source_page must not appear in any production code path."""
    services_dir = os.path.join(os.path.dirname(__file__), "..", "..", "src", "services")
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")

    violations = []
    for dirpath, _, filenames in os.walk(services_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fn)
            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if "supporting_source_page" in line:
                        stripped = line.strip()
                        if stripped.startswith("#") or "Phase 1" in stripped:
                            continue
                        violations.append(f"{fn}:{i}: {stripped[:100]}")

    assert not violations, (
        f"supporting_source_page found in production code:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
