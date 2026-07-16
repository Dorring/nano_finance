"""LangChain import compatibility checks for backend startup."""
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock


def test_ingest_imports_document_from_langchain_core(monkeypatch):
    documents = ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents.Document = Document
    langchain_core = ModuleType("langchain_core")
    langchain_core.documents = documents

    for name in [
        "camelot",
        "pymupdf",
        "langchain_text_splitters",
        "src.services.ingest",
    ]:
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "camelot", MagicMock())
    monkeypatch.setitem(sys.modules, "pymupdf", MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", documents)

    splitters = ModuleType("langchain_text_splitters")
    splitters.RecursiveCharacterTextSplitter = MagicMock(return_value=MagicMock())
    splitters.MarkdownHeaderTextSplitter = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "langchain_text_splitters", splitters)

    ingest = importlib.import_module("src.services.ingest")

    assert ingest.Document is Document