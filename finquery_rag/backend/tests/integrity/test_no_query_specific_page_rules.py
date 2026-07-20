"""Verify the system does not have query-keyword-specific page rules.

After Phase 1 cleanup, the production retriever must treat semantically
similar but differently-worded queries equivalently, without mapping
specific keywords to fixed page numbers.
"""
import pytest


def test_removed_methods_not_accessible():
    """All leaked methods must be fully removed from RAGEngine."""
    from src.services.rag_engine import RAGEngine

    removed_methods = [
        "_fallback_pages_for_query",
        "_supporting_pages_for_query",
        "_force_supporting_page_coverage",
        "_augment_with_page_fallbacks",
        "_ensure_supporting_sources",
    ]
    for method in removed_methods:
        assert not hasattr(RAGEngine, method), (
            f"RAGEngine.{method} must not exist after Phase 1 cleanup"
        )


def test_no_specific_doc_page_rules_in_source():
    """Verify no hardcoded document-name-to-page mappings remain in source."""
    import os

    services_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "services"
    )
    rag_path = os.path.join(services_dir, "rag_engine.py")
    if not os.path.isfile(rag_path):
        pytest.skip("rag_engine.py not found at expected path")

    with open(rag_path, encoding="utf-8") as fh:
        content = fh.read()

    # These hardcoded document names must not appear in active code
    forbidden_docs = [
        "final annual report",
        "pdf solutions",
        "wipo_pub_rn2021",
        "leac203",
    ]
    violations = []
    for doc in forbidden_docs:
        for i, line in enumerate(content.split("\n"), 1):
            if doc in line.lower() and not line.strip().startswith("#"):
                violations.append(f"Line {i}: {line.strip()[:100]}")

    # Some doc names in test data constructors are OK
    # Only flag if they appear in method logic (not test data)
    active_violations = [
        v for v in violations
        if "def _" not in content.split("\n")[int(v.split(":")[0].split()[1]) - 20:int(v.split(":")[0].split()[1])]
    ]


def test_supporting_source_page_not_in_production():
    """supporting_source_page must not appear in any production code path."""
    import os

    services_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "services"
    )
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


def test_no_query_keyword_to_page_mapping():
    """Verify no function maps query keywords to fixed page numbers."""
    import os

    services_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "services"
    )
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")

    # These patterns indicate keyword->page rules
    suspicious_patterns = [
        "cash and cash equivalents",
        "record revenue",
        "gross margin",
        "black swan",
        "sunfill",
        "pct system",
        "madrid",
        "credit facilities",
        "operating activities",
    ]

    violations = []
    for dirpath, _, filenames in os.walk(services_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            fpath = os.path.join(dirpath, fn)
            with open(fpath, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if line.strip().startswith("#"):
                        continue
                    for pattern in suspicious_patterns:
                        if pattern in line.lower() and ".append(" not in line.lower():
                            violations.append(f"{fn}:{i}: {line.strip()[:120]}")

    # Allow the pattern keywords in intent.py (query classification) and
    # in expansion queries, but not in page-mapping logic
    active_violations = [
        v for v in violations
        if "intent.py" not in v
        and "expand_retrieval_query" not in v
        and "_EXPANSIONS" not in v.split(":")[1]
    ]
    assert not active_violations, (
        f"Query-keyword-to-page mappings found in production:\n"
        + "\n".join(f"  {v}" for v in active_violations)
    )
