import os
import tempfile
import time
import sys
from unittest.mock import MagicMock


mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "camelot", "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "pymupdf", "langchain_core", "langchain_core.documents",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["langchain_core.documents"].Document = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_text_splitters"].MarkdownHeaderTextSplitter = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["camelot"].read_pdf = lambda *args, **kwargs: []

from src.services.ingest import (
    _chunk_content_with_section,
    _hierarchy_metadata,
    _section_path_from_metadata,
)
from src.services.rag_engine import RAGEngine


class _DummyLLM:
    def __init__(self):
        self.chat = self

    def create(self, **kwargs):
        class _Response:
            choices = [type("Choice", (), {"message": type("Msg", (), {"content": "ok"})()})()]
        return _Response()


def _make_engine(**kwargs):
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return RAGEngine(_DummyLLM(), bm25_db_path=tmp.name, **kwargs), tmp.name


def _cleanup(path):
    for _ in range(3):
        try:
            os.unlink(path)
            return
        except PermissionError:
            time.sleep(0.05)


def _chunk(doc_id, content, metadata, score=0.3):
    return {
        "doc_id": doc_id,
        "content": content,
        "metadata": metadata,
        "score": score,
    }


def test_hierarchy_metadata_tracks_section_path_and_bounded_parent_excerpt():
    metadata = {
        "Header 1": "1 Introduction",
        "Header 2": "Motivation",
        "Header 4": "PAGE_MARKER_2",
    }
    parent_content = "This section explains the benchmark. " * 80

    hierarchy = _hierarchy_metadata(
        metadata,
        user_id=7,
        doc_name="paper.pdf",
        page=2,
        chunk_idx=4,
        parent_content=parent_content,
    )

    assert _section_path_from_metadata(metadata) == "1 Introduction > Motivation"
    assert hierarchy["section_path"] == "1 Introduction > Motivation"
    assert hierarchy["section_title"] == "Motivation"
    assert hierarchy["parent_id"] == "user_7_paper.pdf::page_2::parent_4"
    assert hierarchy["parent_page"] == 2
    assert hierarchy["parent_child"] is True
    assert len(hierarchy["parent_excerpt"]) <= 1400


def test_chunk_content_with_section_prepends_retrieval_context_once():
    first = _chunk_content_with_section("Revenue increased.", "MD&A > Revenue")
    second = _chunk_content_with_section(first, "MD&A > Revenue")

    assert first.startswith("Section: MD&A > Revenue\n")
    assert second == first


def test_build_context_merges_child_hits_into_one_parent_excerpt():
    engine, path = _make_engine()
    try:
        parent_excerpt = (
            "Section context: revenue increased because enterprise demand improved. "
            "Operating margin also expanded."
        )
        base_meta = {
            "type": "text",
            "page": 3,
            "parent_id": "user_1_report.pdf::page_3::parent_9",
            "parent_page": 3,
            "parent_excerpt": parent_excerpt,
            "section_path": "MD&A > Revenue",
        }
        c1 = _chunk(
            "user_1_report.pdf::page_3::chunk_9_0",
            "enterprise demand improved",
            dict(base_meta),
            score=0.1,
        )
        c2 = _chunk(
            "user_1_report.pdf::page_3::chunk_9_1",
            "operating margin expanded",
            dict(base_meta),
            score=0.7,
        )

        context, sources = engine.build_context([c1, c2])

        assert context.count(parent_excerpt) == 1
        assert "Matched child evidence:" in context
        assert "enterprise demand improved" in context
        assert "operating margin expanded" in context
        assert len(sources) == 1
        assert sources[0]["parent_id"] == "user_1_report.pdf::page_3::parent_9"
        assert sources[0]["section_path"] == "MD&A > Revenue"
        assert sources[0]["child_hit_count"] == 2
        assert sources[0]["score"] == 0.7
        assert sources[0]["chunk_id"] == "user_1_report.pdf::page_3::chunk_9_0"
    finally:
        _cleanup(path)


def test_build_context_keeps_non_hierarchical_chunks_unchanged():
    engine, path = _make_engine()
    try:
        chunk = _chunk(
            "user_1_report.pdf::page_1::chunk_1",
            "Standalone context.",
            {"type": "text", "page": 1},
        )

        context, sources = engine.build_context([chunk])

        assert "Standalone context." in context
        assert len(sources) == 1
        assert sources[0]["parent_id"] is None
        assert sources[0]["section_path"] is None
    finally:
        _cleanup(path)
