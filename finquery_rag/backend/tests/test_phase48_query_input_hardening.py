"""Phase 48 tests: query request input hardening."""
import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.schemas import QueryRequest
from services.health import collect_config_snapshot
from services.query_scope import resolve_query_document_names


def test_query_request_trims_question_and_rejects_blank_or_too_long():
    request = QueryRequest(question="  What is revenue?  ")
    assert request.question == "What is revenue?"

    with pytest.raises(ValidationError):
        QueryRequest(question="   ")

    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 4001)


def test_query_request_normalizes_document_names():
    request = QueryRequest(
        question="What is revenue?",
        document_names=[" report.pdf ", "report.pdf", "summary.pdf"],
    )

    assert request.document_names == ["report.pdf", "summary.pdf"]


@pytest.mark.parametrize("document_names", [
    [""],
    ["   "],
    ["bad/name.pdf"],
    ["bad\\name.pdf"],
    ["bad\x00.pdf"],
    ["x" * 181 + ".pdf"],
    [f"d{i}.pdf" for i in range(21)],
])
def test_query_request_rejects_invalid_document_names(document_names):
    with pytest.raises(ValidationError):
        QueryRequest(question="What is revenue?", document_names=document_names)


def test_query_scope_trims_and_deduplicates_requested_names():
    resolved, invalid = resolve_query_document_names(
        [" report.pdf ", "report.pdf", "summary.pdf"],
        ["report.pdf", "summary.pdf"],
    )

    assert resolved == ["report.pdf", "summary.pdf"]
    assert invalid == []


def test_query_scope_ignores_non_string_internal_names():
    resolved, invalid = resolve_query_document_names(
        ["report.pdf", None, 123],
        ["report.pdf"],
    )

    assert resolved == ["report.pdf"]
    assert invalid == []


def test_health_config_reports_query_input_limits():
    cfg = collect_config_snapshot()

    assert cfg["limits"]["query_question_max_chars"] == 4000
    assert cfg["limits"]["query_document_names_max_items"] == 20
    assert cfg["limits"]["query_document_name_max_chars"] == 180
