import sys
from unittest.mock import MagicMock

mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain_core", "langchain_core.documents", "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["langchain_core.documents"].Document = MagicMock()
sys.modules["langchain_text_splitters"].RecursiveCharacterTextSplitter = MagicMock()
sys.modules["langchain_text_splitters"].MarkdownHeaderTextSplitter = MagicMock()
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]

from src.services.ingest import _extract_title_from_first_page
from src.services.rag_engine import RAGEngine


class _FakePage:
    def __init__(self, lines, height=1000):
        self._lines = lines
        self.rect = type("Rect", (), {"height": height})()

    def get_text(self, mode, *args, **kwargs):
        assert mode == "dict"
        blocks = []
        for text, size, y0 in self._lines:
            blocks.append({
                "type": 0,
                "lines": [{
                    "spans": [{
                        "text": text,
                        "size": size,
                        "bbox": [0, y0, 100, y0 + 10],
                    }]
                }]
            })
        return {"blocks": blocks}


class _DummyLLM:
    pass


def test_extract_title_merges_multiline_front_matter_title():
    page = _FakePage([
        ("001", 8, 20),
        ("Rethinking Crack Segmentation: A", 22, 80),
        ("Semantic-Visual Co-refinement Paradigm with", 22, 120),
        ("the MSCrack30K Benchmark", 22, 160),
        ("Anonymous ECCV 2026 Submission", 14, 260),
        ("Abstract. Crack segmentation remains challenging", 11, 380),
    ])

    title = _extract_title_from_first_page(page)

    assert title == "Rethinking Crack Segmentation: A Semantic-Visual Co-refinement Paradigm with the MSCrack30K Benchmark"


def test_front_matter_title_answer_strips_title_prefix(tmp_path):
    engine = RAGEngine(_DummyLLM(), bm25_db_path=str(tmp_path / "b.db"))
    chunks = [{
        "doc_id": "paper::title",
        "content": "Title: Rethinking Crack Segmentation",
        "metadata": {"type": "front_matter", "subtype": "title", "page": 1, "doc_name": "paper.pdf"},
        "score": 0.02,
    }]

    answer = engine.answer_front_matter_query("What is the title of this paper?", chunks)

    assert answer["answer"] == 'The title of the paper is "Rethinking Crack Segmentation".'
    assert answer["diagnostic"] == "front_matter_title"