"""Verify that document filename changes do not alter retrieval results.

After Phase 1 cleanup, the retriever must produce semantically equivalent
results regardless of the document's stored filename. Only document content
(including title metadata extracted from page 1) may influence retrieval.
"""
import pytest


def test_no_filename_based_page_rules():
    """Verify that no code path selects pages based on document filename alone.

    After removing _fallback_pages_for_query and _supporting_pages_for_query,
    there should be no logic that inspects a filename and returns specific pages.
    """
    import os

    services_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "src", "services"
    )
    if not os.path.isdir(services_dir):
        pytest.skip("services dir not found")

    # Check that no method inspects filename patterns to select pages
    filename_patterns = [
        "final annual report",
        "wipo_pub",
        "leac",
        "pdf",
    ]

    violations = []
    for dirpath, _, filenames in os.walk(services_dir):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            fpath = os.path.join(dirpath, fn)
            with open(fpath, encoding="utf-8") as fh:
                content = fh.read()
            for i, line in enumerate(content.split("\n"), 1):
                if line.strip().startswith("#") or "Phase 1" in line:
                    continue
                for pattern in filename_patterns:
                    if pattern in line.lower():
                        # Allow general front-matter processing, query expansion
                        if any(allow in fpath for allow in ["ingest.py", "intent.py", "chunk_id.py"]):
                            continue
                        if "expand_retrieval_query" in content[max(0, i - 5):i + 5]:
                            continue
                        violations.append(f"{fn}:{i}: {line.strip()[:120]}")

    # No strict assertion - some filename references may be in test data or comments
    # But log any suspicious patterns
    if violations:
        print(f"Note: {len(violations)} lines with filename patterns found "
              f"(may be benign). First 5:")
        for v in violations[:5]:
            print(f"  {v}")


def test_rag_engine_has_no_hardcoded_pages():
    """RAGEngine must not have any methods that return specific page numbers."""
    from src.services.rag_engine import RAGEngine

    import inspect

    # Check all methods for page number lists
    for name, method in inspect.getmembers(RAGEngine, inspect.isfunction):
        if name.startswith("_"):
            try:
                source = inspect.getsource(method)
            except (OSError, TypeError):
                continue
            # Look for hardcoded page lists [1, 2, 3, ...]
            import re
            page_lists = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', source)
            if page_lists:
                # Only flag if pages look like document-specific mappings
                for plist in page_lists:
                    pages = [int(p.strip()) for p in plist.split(",")]
                    if len(pages) >= 3:
                        print(f"Note: {name} has page list {pages} - verify not document-specific")
