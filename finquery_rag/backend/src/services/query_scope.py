"""Helpers for resolving tenant-scoped query document filters."""
from __future__ import annotations


def _unique_names(names):
    """Return non-empty names once, preserving request order."""
    seen = set()
    unique = []
    for name in names or []:
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def resolve_query_document_names(requested_doc_names, ready_doc_names, fallback_doc_names=None):
    """Resolve a query document filter against ready documents.

    Returns (resolved_names, invalid_names). When the registry has no ready
    documents, fallback_doc_names may be used for legacy indexes that predate
    lifecycle tracking.
    """
    ready_names = _unique_names(ready_doc_names)
    fallback_names = _unique_names(fallback_doc_names)
    available_names = ready_names or fallback_names
    available = set(available_names)

    if requested_doc_names is None:
        return available_names, []

    requested = _unique_names(requested_doc_names)
    invalid = [name for name in requested if name not in available]
    if invalid:
        return [], invalid
    return requested, []
