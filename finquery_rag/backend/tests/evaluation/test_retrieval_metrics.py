"""Tests for retrieval metrics in src.evaluation.metrics."""
from __future__ import annotations

from src.evaluation.metrics import (
    document_coverage,
    expected_page_recall,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.schemas import ExpectedSource


def _src(filename: str, page: int | None = None, chunk_id: str | None = None) -> ExpectedSource:
    """Build an ExpectedSource conveniently."""
    return ExpectedSource(filename=filename, page=page, chunk_id=chunk_id)


def _chunk(filename: str, page: int | None = None, chunk_id: str | None = None) -> dict:
    """Build a retrieved chunk dict."""
    d: dict = {}
    if filename:
        d["filename"] = filename
    if page is not None:
        d["page"] = page
    if chunk_id is not None:
        d["chunk_id"] = chunk_id
    return d


class TestRecallAtK:
    def test_perfect_recall_at_5(self) -> None:
        """All expected sources found in top-5 retrieved chunks."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [
            _chunk("a.pdf", page=1),
            _chunk("b.pdf", page=2),
            _chunk("c.pdf", page=3),
        ]
        assert recall_at_k(expected, retrieved, 5) == 1.0

    def test_zero_recall(self) -> None:
        """No expected sources in retrieved chunks."""
        expected = [_src("a.pdf", page=1)]
        retrieved = [_chunk("c.pdf", page=3), _chunk("d.pdf", page=4)]
        assert recall_at_k(expected, retrieved, 5) == 0.0

    def test_partial_recall(self) -> None:
        """Half of expected sources found."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [_chunk("a.pdf", page=1), _chunk("c.pdf", page=3)]
        assert recall_at_k(expected, retrieved, 5) == 0.5

    def test_no_expected_sources_returns_one(self) -> None:
        assert recall_at_k([], [_chunk("a.pdf")], 5) == 1.0


class TestMRR:
    def test_mrr_first_position(self) -> None:
        """First chunk matches → MRR = 1.0."""
        expected = [_src("a.pdf", page=1)]
        retrieved = [_chunk("a.pdf", page=1), _chunk("b.pdf", page=2)]
        assert mrr(expected, retrieved) == 1.0

    def test_mrr_not_found(self) -> None:
        """No match → MRR = 0.0."""
        expected = [_src("a.pdf", page=1)]
        retrieved = [_chunk("b.pdf", page=2), _chunk("c.pdf", page=3)]
        assert mrr(expected, retrieved) == 0.0

    def test_mrr_second_position(self) -> None:
        expected = [_src("a.pdf", page=1)]
        retrieved = [_chunk("b.pdf", page=2), _chunk("a.pdf", page=1)]
        assert mrr(expected, retrieved) == 0.5


class TestNDCG:
    def test_ndcg_perfect(self) -> None:
        """All expected sources in top positions → NDCG = 1.0."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [
            _chunk("a.pdf", page=1),
            _chunk("b.pdf", page=2),
            _chunk("c.pdf", page=3),
        ]
        assert ndcg_at_k(expected, retrieved, 5) == 1.0

    def test_ndcg_no_match(self) -> None:
        expected = [_src("a.pdf", page=1)]
        retrieved = [_chunk("b.pdf", page=2), _chunk("c.pdf", page=3)]
        assert ndcg_at_k(expected, retrieved, 5) == 0.0


class TestDocumentCoverage:
    def test_document_coverage(self) -> None:
        """All expected documents found in retrieved chunks."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [_chunk("a.pdf", page=1), _chunk("b.pdf", page=2)]
        assert document_coverage(expected, retrieved) == 1.0

    def test_document_coverage_partial(self) -> None:
        expected = [_src("a.pdf"), _src("b.pdf")]
        retrieved = [_chunk("a.pdf"), _chunk("c.pdf")]
        assert document_coverage(expected, retrieved) == 0.5


class TestPageRecall:
    def test_page_recall(self) -> None:
        """Expected (filename, page) pairs found in retrieved."""
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [_chunk("a.pdf", page=1), _chunk("b.pdf", page=2)]
        assert expected_page_recall(expected, retrieved) == 1.0

    def test_page_recall_partial(self) -> None:
        expected = [_src("a.pdf", page=1), _src("b.pdf", page=2)]
        retrieved = [_chunk("a.pdf", page=1), _chunk("b.pdf", page=99)]
        assert expected_page_recall(expected, retrieved) == 0.5
