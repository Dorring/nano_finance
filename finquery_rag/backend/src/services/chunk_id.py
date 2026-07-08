"""Shared chunk ID generation and validation for tenant isolation."""

SCOPE_PREFIX = "user_"


def make_chunk_id(user_id, doc_name, suffix):
    """
    Generate a tenant-scoped chunk ID.

    Args:
        user_id: Must not be None. user_id=0 is valid.
        doc_name: Original filename.
        suffix: e.g. "page_1::chunk_0" or "page_1::table_1".

    Returns:
        Scoped ID string: "user_{user_id}_{doc_name}::{suffix}"

    Raises:
        ValueError: if user_id is None.
    """
    if user_id is None:
        raise ValueError("user_id is required for chunk ID generation")
    return f"{SCOPE_PREFIX}{user_id}_{doc_name}::{suffix}"


def is_scoped_chunk_id(chunk_id: str, user_id=None) -> bool:
    """
    Check whether a chunk ID already has the correct tenant scope prefix.
    If user_id is provided, also verifies the ID belongs to that user.
    """
    if not isinstance(chunk_id, str) or not chunk_id.startswith(SCOPE_PREFIX):
        return False
    if user_id is None:
        return True
    expected = f"{SCOPE_PREFIX}{user_id}_"
    return chunk_id.startswith(expected)


def ensure_scoped_chunk_id(chunk_id: str, user_id, doc_name: str) -> str:
    """
    Return chunk_id if already correctly scoped; otherwise wrap it.
    Idempotent: calling twice does not double-prefix.
    Raises ValueError if the chunk_id has a DIFFERENT user prefix.
    """
    if user_id is None:
        raise ValueError("user_id is required for chunk ID generation")
    if is_scoped_chunk_id(chunk_id, user_id):
        return chunk_id
    if is_scoped_chunk_id(chunk_id):
        raise ValueError(
            f"chunk_id {chunk_id!r} belongs to a different tenant, expected user_{user_id}_"
        )
    return f"{SCOPE_PREFIX}{user_id}_{chunk_id}"
