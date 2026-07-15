from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sft1147_runbook_documents_openai_compatible_adapter():
    doc = (ROOT / "docs" / "SFT1147_RAG_DEMO.md").read_text(encoding="utf-8")

    required = [
        "python -m scripts.chat_openai_compat",
        "--model-name finquery-finance-sft1147",
        "LLM_API_BASE_URL=http://127.0.0.1:8500/v1",
        "LLM_MODEL_NAME=finquery-finance-sft1147",
        "LLM_API_KEY=not-needed-for-local",
        "GET /v1/models",
        "POST /v1/chat/completions",
        "context_sufficient",
        "ci_preflight_smoke.py",
    ]
    for needle in required:
        assert needle in doc


def test_rag_config_links_sft1147_switching_envs():
    config = (ROOT / "RAG_CONFIG.md").read_text(encoding="utf-8", errors="ignore")

    assert "LLM backend: local SFT1147 adapter" in config
    assert "LLM_API_BASE_URL=http://127.0.0.1:8500/v1" in config
    assert "LLM_MODEL_NAME=finquery-finance-sft1147" in config
    assert "docs/SFT1147_RAG_DEMO.md" in config