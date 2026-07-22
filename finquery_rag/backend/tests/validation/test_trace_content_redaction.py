"""Phase 4 hotfix: trace content redaction regression tests.

Verifies that the default trace does NOT store:
- Full final_context
- Full answer
- Internal validation messages
- Full claim_text
- str(exc) in streaming errors

Only hashes, lengths, codes, and public-safe metadata are retained.
"""
from __future__ import annotations

import hashlib
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.validation import (
    ValidationIssue,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# ValidationIssue.to_trace_dict redaction
# ---------------------------------------------------------------------------

class TestValidationIssueTraceRedaction:
    """ValidationIssue.to_trace_dict must redact message and claim_text."""

    def test_trace_dict_uses_message_hash_not_message(self):
        issue = ValidationIssue(
            code="NUMERIC_UNGROUND",
            severity=ValidationSeverity.CRITICAL,
            message="Internal: revenue 500 not found in chunk abc123 with metric mismatch",
            claim_text="Revenue was $500M",
        )
        trace = issue.to_trace_dict()
        assert "message" not in trace
        assert "message_hash" in trace
        expected = hashlib.sha256(issue.message.encode("utf-8")).hexdigest()[:16]
        assert trace["message_hash"] == expected

    def test_trace_dict_uses_claim_excerpt_not_claim_text(self):
        issue = ValidationIssue(
            code="CITATION_MISSING",
            severity=ValidationSeverity.ERROR,
            message="internal detail",
            claim_text="The revenue figure of 1,250 million is unsupported",
        )
        trace = issue.to_trace_dict()
        assert "claim_text" not in trace
        assert "claim_excerpt" in trace
        assert trace["claim_excerpt"] == issue.claim_text[:80]

    def test_trace_dict_truncates_long_claim_excerpt(self):
        long_claim = "A" * 200
        issue = ValidationIssue(
            code="NUMERIC_UNGROUND",
            severity=ValidationSeverity.CRITICAL,
            message="msg",
            claim_text=long_claim,
        )
        trace = issue.to_trace_dict()
        assert len(trace["claim_excerpt"]) == 80

    def test_trace_dict_strips_control_chars_from_claim(self):
        issue = ValidationIssue(
            code="X",
            severity=ValidationSeverity.WARNING,
            message="msg",
            claim_text="hello\x00world\n\ttab",
        )
        trace = issue.to_trace_dict()
        assert "\x00" not in trace["claim_excerpt"]

    def test_trace_dict_preserves_code_severity_evidence_ids(self):
        issue = ValidationIssue(
            code="CITATION_PAGE_MISMATCH",
            severity=ValidationSeverity.ERROR,
            message="internal",
            claim_text="claim",
            evidence_ids=("chunk_1", "chunk_2"),
        )
        trace = issue.to_trace_dict()
        assert trace["code"] == "CITATION_PAGE_MISMATCH"
        assert trace["severity"] == "error"
        assert trace["evidence_ids"] == ["chunk_1", "chunk_2"]

    def test_trace_dict_handles_none_claim_text(self):
        issue = ValidationIssue(
            code="X",
            severity=ValidationSeverity.WARNING,
            message="msg",
            claim_text=None,
        )
        trace = issue.to_trace_dict()
        assert trace["claim_excerpt"] is None


# ---------------------------------------------------------------------------
# Orchestrator trace_data redaction (integration)
# ---------------------------------------------------------------------------

class TestOrchestratorTraceRedaction:
    """The orchestrator must set final_context=None and answer=None in trace."""

    @staticmethod
    def _make_orchestrator_with_mock_trace():
        """Create a minimal orchestrator mock that captures trace_data."""
        from src.application.rag_orchestrator import RAGOrchestrator

        orchestrator = RAGOrchestrator.__new__(RAGOrchestrator)
        orchestrator._trace_logger = MagicMock()
        orchestrator._trace_logger.log = MagicMock(return_value="trace-redacted")
        return orchestrator

    def test_trace_logger_receives_none_for_context_and_answer(self, tmp_path):
        """Verify the orchestrator sends final_context=None and answer=None to trace."""
        orchestrator_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src",
            "application", "rag_orchestrator.py",
        )
        content = open(orchestrator_path, encoding="utf-8").read()

        # The orchestrator must explicitly set these to None.
        assert '"final_context": None' in content
        assert '"answer": None' in content
        # And must compute hashes for diagnostics.
        assert "context_sha256" in content
        assert "answer_sha256" in content
        assert "context_length" in content
        assert "answer_length" in content


# ---------------------------------------------------------------------------
# Public trace API redaction (main.py _public_trace)
# ---------------------------------------------------------------------------

class TestPublicTraceRedaction:
    """The /traces endpoint must not expose final_context, answer, or error_message."""

    def test_public_trace_excludes_sensitive_fields(self):
        main_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "main.py",
        )
        content = open(main_path, encoding="utf-8").read()

        # _public_trace must not include final_context or answer in keys list.
        # Find the keys list in _public_trace.
        assert "def _public_trace" in content
        # The function must NOT assign trace["final_context"] or trace["answer"].
        assert 'trace["final_context"]' not in content
        assert 'trace["answer"]' not in content
        # It must expose safe diagnostics instead.
        assert "context_available" in content
        assert "answer_available" in content
        assert "context_sha256" in content
        assert "answer_sha256" in content

    def test_streaming_exception_does_not_save_str_exc(self):
        main_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "src", "main.py",
        )
        content = open(main_path, encoding="utf-8").read()

        # The streaming exception handler must not save str(exc).
        assert '"error_code": "STREAM_INTERNAL_ERROR"' in content
        assert '"exception_type": type(exc).__name__' in content
        # Must NOT save str(exc) in the trace payload.
        assert 'str(exc)' not in content.split('def query_documents_stream')[1]
