"""Shared fixtures for refactor characterization tests.

Only modules that are NOT installed in the CI environment are mocked.
Available dependencies (bcrypt, fastapi, sqlalchemy, openai, tiktoken) are
deliberately NOT mocked — mocking them replaces real types with ``MagicMock``
and breaks forward references in type annotations (e.g. ``src.services.auth``),
which causes SyntaxErrors in ``tests/architecture/test_api_contract.py`` and
``tests/architecture/test_imports.py``.

jose and passlib are also NOT mocked even though they are unavailable:
mocking them causes MagicMock forward-reference SyntaxErrors in
``src.services.auth``. Tests that need jose/passlib should use
``pytest.importorskip`` or ``_have_api_deps()`` guards instead.

``pytest_configure`` installs mocks for unavailable modules before collection
so that module-level ``from src.services.rag_engine import RAGEngine`` succeeds.
``pytest_unconfigure`` removes them after the session so they do not leak.
"""
import sys
from unittest.mock import MagicMock


# Modules that are NOT installed in the CI environment and must be mocked.
UNAVAILABLE_MODULES: dict[str, list[str]] = {
    "chromadb": ["config", "utils", "utils.embedding_functions"],
    "sentence_transformers": [],
    "pymupdf": [],
    "fitz": [],
    "jieba_fast": ["jieba"],
    "camelot": [],
    "langchain": [],
    "langchain_core": ["documents"],
    "langchain_text_splitters": [],
}

# Tracks which modules we inserted so ``pytest_unconfigure`` can remove them
# without deleting modules that were already present before our session.
_installed_by_us: list[str] = []


def _install_mocks() -> None:
    """Install mock modules for unavailable dependencies if not present."""
    for mod_name, submods in UNAVAILABLE_MODULES.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()
            _installed_by_us.append(mod_name)
        for sub in submods:
            full = f"{mod_name}.{sub}"
            if full not in sys.modules:
                sys.modules[full] = MagicMock()
                _installed_by_us.append(full)
    # chromadb embedding function needs a callable class
    ef = sys.modules.get("chromadb.utils.embedding_functions")
    if ef is not None and not hasattr(ef, "_configured_by_conftest"):
        ef.SentenceTransformerEmbeddingFunction.return_value = MagicMock()
        ef._configured_by_conftest = True
    # jieba_fast.cut_for_search returns list of tokens
    jf = sys.modules.get("jieba_fast")
    if jf is not None and not hasattr(jf, "_cut_configured"):
        jf.cut_for_search = lambda text: [text]
        jf._cut_configured = True
    # langchain splitters need callable classes
    lts = sys.modules.get("langchain_text_splitters")
    if lts is not None and not isinstance(lts, type) and not hasattr(lts, "_splitters_configured"):
        lts.RecursiveCharacterTextSplitter = MagicMock()
        lts.MarkdownHeaderTextSplitter = MagicMock()
        lts._splitters_configured = True
    lcd = sys.modules.get("langchain_core.documents")
    if lcd is not None and not isinstance(lcd, type) and not hasattr(lcd, "_doc_configured"):
        lcd.Document = MagicMock()
        lcd._doc_configured = True


def _uninstall_mocks() -> None:
    """Remove mock modules installed by ``_install_mocks``."""
    for mod_name in _installed_by_us:
        sys.modules.pop(mod_name, None)
    _installed_by_us.clear()


def pytest_configure(config):
    """Pytest hook: install unavailable-dep mocks before test collection."""
    _install_mocks()


def pytest_unconfigure(config):
    """Pytest hook: remove unavailable-dep mocks after all tests complete."""
    _uninstall_mocks()
