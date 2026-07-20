"""Shared fixtures for refactor characterization tests.

These tests need RAGEngine but may not have chromadb/tiktoken/jieba installed.
We mock the heavy dependencies at import time.
"""
import sys
from unittest.mock import MagicMock


def _install_mocks():
    """Install mock modules for heavy dependencies if not available."""
    heavy_modules = {
        "chromadb": ["config", "utils", "utils.embedding_functions",
                      "utils.embedding_functions.SentenceTransformerEmbeddingFunction"],
        "tiktoken": [],
        "sentence_transformers": [],
        "openai": [],
        "jieba_fast": [],
        "jieba_fast.jieba": [],
        "pymupdf": [],
        "fitz": [],
        "jose": ["jwt", "jws", "jwe"],
        "bcrypt": [],
        "sqlalchemy": ["orm", "ext", "ext.declarative", "Column", "String", "Integer",
                        "Float", "Boolean", "Text", "DateTime", "ForeignKey",
                        "create_engine", "Column"],
    }

    for mod_name, submods in heavy_modules.items():
        if mod_name not in sys.modules:
            mock = MagicMock()
            sys.modules[mod_name] = mock
            for sub in submods:
                full = f"{mod_name}.{sub}"
                sys.modules[full] = MagicMock()


_install_mocks()
