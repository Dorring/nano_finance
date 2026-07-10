"""Phase 23 tests: typed source response schema exposes exact chunk provenance."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.schemas import QueryResponse, SourceInfo


def test_query_response_sources_are_typed_and_keep_chunk_id():
    response = QueryResponse(
        answer="A",
        sources=[
            {
                "filename": "report.pdf",
                "page": 3,
                "type": "text",
                "score": 0.91,
                "chunk_id": "user_1_report.pdf::page_3::chunk_7",
            }
        ],
        question="Q",
        searched_docs=["report.pdf"],
    )

    source = response.sources[0]
    assert isinstance(source, SourceInfo)
    assert source.chunk_id == "user_1_report.pdf::page_3::chunk_7"
    assert source.filename == "report.pdf"
    assert source.page == 3


def test_source_info_schema_documents_chunk_id():
    schema = QueryResponse.model_json_schema()
    source_ref = schema["properties"]["sources"]["items"]["$ref"].split("/")[-1]
    source_schema = schema["$defs"][source_ref]

    assert "chunk_id" in source_schema["properties"]
    assert "score" in source_schema["properties"]



def test_main_api_pagination_helper_rejects_negative_values_static():
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert "def _normalize_api_pagination" in content
    assert "limit must be >= 1" in content
    assert "offset must be >= 0" in content
    assert "created_after must be <= created_before" in content
    assert "normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=20)" in content
    assert "normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=50)" in content
