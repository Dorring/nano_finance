import ast
from pathlib import Path

from src.services.health import collect_config_snapshot


def test_health_config_reports_heuristic_reranker_by_default(monkeypatch):
    monkeypatch.delenv("RAG_RERANKER", raising=False)

    cfg = collect_config_snapshot()

    assert cfg["retrieval"]["reranker"] == "heuristic"


def test_health_config_respects_reranker_off(monkeypatch):
    monkeypatch.setenv("RAG_RERANKER", "off")

    cfg = collect_config_snapshot()

    assert cfg["retrieval"]["reranker"] == "off"


def test_fastapi_runtime_defaults_to_heuristic_reranker_static():
    main_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    source = main_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    get_engine = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_rag_engine")
    block = ast.get_source_segment(source, get_engine)

    assert 'os.getenv("RAG_RERANKER", "heuristic")' in block
    assert 'os.getenv("RAG_RERANKER_MODEL")' in block


def test_rag_config_documents_default_heuristic_reranker():
    config_path = Path(__file__).resolve().parents[1] / "RAG_CONFIG.md"
    content = config_path.read_text(encoding="utf-8")

    assert "default `heuristic` reranker" in content
    assert "`none` / `off`" in content
    assert "simple Chinese demos" in content