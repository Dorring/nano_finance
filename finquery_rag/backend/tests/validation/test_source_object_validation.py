"""Phase 4 production regression tests for CitationValidator source objects.

Verifies that the ``CitationValidator`` correctly validates the ``sources``
tuple against the retrieved evidence: chunk_id presence, document-name
consistency (with extension / user-prefix normalization), page consistency,
and source deduplication. Uses the ``front_matter`` policy which requires
citations.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.evidence import EvidenceItem
from src.validation.citation_validator import (
    CODE_CITATION_CHUNK_MISSING,
    CODE_CITATION_DOCUMENT_MISMATCH,
    CODE_CITATION_NOT_RETRIEVED,
    CODE_CITATION_PAGE_MISMATCH,
    CitationValidator,
)
from src.validation.validation_policy import get_policy_for_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
    page: int = 12,
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content="Revenue for 2024 was 125 million.",
        document_name=document_name,
        page=page,
        content_type="text",
        score=0.9,
        rerank_score=None,
        metadata={},
    )


def _front_matter_policy():
    return get_policy_for_intent("front_matter")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_source_no_issues():
    """A source that matches the evidence produces no issues."""
    ev = _evidence(chunk_id="c1", document_name="annual_report.pdf", page=12)
    source = {"chunk_id": "c1", "document_name": "annual_report.pdf", "page": 12}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(source,),
    )
    assert issues == ()


def test_chunk_id_missing_flagged():
    """A source with no chunk_id is flagged CITATION_CHUNK_MISSING."""
    ev = _evidence()
    source = {"chunk_id": "", "document_name": "annual_report.pdf", "page": 12}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(source,),
    )
    codes = [i.code for i in issues]
    assert CODE_CITATION_CHUNK_MISSING in codes


def test_chunk_id_not_in_evidence():
    """A source chunk_id absent from evidence is flagged CITATION_NOT_RETRIEVED."""
    ev = _evidence(chunk_id="c1", document_name="annual_report.pdf", page=12)
    # chunk_id, document_name, AND page all differ so the fallback match
    # also fails.
    source = {"chunk_id": "c_missing", "document_name": "other.pdf", "page": 99}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(source,),
    )
    codes = [i.code for i in issues]
    assert CODE_CITATION_NOT_RETRIEVED in codes


def test_document_name_mismatch():
    """A source doc_name that doesn't match evidence is flagged CITATION_DOCUMENT_MISMATCH."""
    ev = _evidence(chunk_id="c1", document_name="annual_report.pdf", page=12)
    source = {"chunk_id": "c1", "document_name": "different.pdf", "page": 12}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(source,),
    )
    codes = [i.code for i in issues]
    assert CODE_CITATION_DOCUMENT_MISMATCH in codes


def test_page_mismatch():
    """A source page that doesn't match evidence is flagged CITATION_PAGE_MISMATCH."""
    ev = _evidence(chunk_id="c1", document_name="annual_report.pdf", page=12)
    source = {"chunk_id": "c1", "document_name": "annual_report.pdf", "page": 99}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(source,),
    )
    codes = [i.code for i in issues]
    assert CODE_CITATION_PAGE_MISMATCH in codes


def test_doc_names_match_strips_extension():
    """_doc_names_match treats 'report.pdf' and 'report' as the same document."""
    assert CitationValidator._doc_names_match("report.pdf", "report") is True
    assert CitationValidator._doc_names_match("report", "report.pdf") is True
    assert CitationValidator._doc_names_match("annual.pdf", "annual") is True


def test_doc_names_match_strips_user_prefix():
    """_doc_names_match strips the user_N_ prefix from document names."""
    assert CitationValidator._doc_names_match("user_123_report.pdf", "report") is True
    assert CitationValidator._doc_names_match("user_7_report", "report.pdf") is True
    assert CitationValidator._doc_names_match("user_123_report.pdf", "annual.pdf") is False


def test_duplicate_sources_normalized():
    """Duplicate sources are deduplicated so each issue is reported once."""
    ev = _evidence()
    # Two identical sources missing a chunk_id: only one CITATION_CHUNK_MISSING.
    bad_source = {"chunk_id": "", "document_name": "annual_report.pdf", "page": 12}
    issues = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(bad_source, bad_source),
    )
    chunk_missing = [i for i in issues if i.code == CODE_CITATION_CHUNK_MISSING]
    assert len(chunk_missing) == 1

    # Two identical valid sources produce no issues.
    good_source = {"chunk_id": "c1", "document_name": "annual_report.pdf", "page": 12}
    issues_valid = CitationValidator().validate(
        claims=(),
        evidence=(ev,),
        policy=_front_matter_policy(),
        sources=(good_source, good_source),
    )
    assert issues_valid == ()
