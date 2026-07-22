"""Tests for the sealed evaluation leakage scanner (Phase 5).

These tests verify that :mod:`check_sealed_eval_leakage` correctly detects
sealed data leaking into production code, while allowing evaluation modules
to reference sealed data legitimately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_sealed_eval_leakage as scanner  # noqa: E402


def _write(path: Path, content: str) -> Path:
    """Write a Python file with the given content (creating parents)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_production_imports_evaluation_detected(tmp_path: Path) -> None:
    """A production module importing src.evaluation must be flagged."""
    src = tmp_path / "src"
    eval_dir = src / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "__init__.py").write_text("", encoding="utf-8")
    _write(
        src / "services" / "bad.py",
        "from src.evaluation import eval_runner\n",
    )
    violations = scanner.check_production_imports_evaluation(src)
    assert any("src.evaluation" in v for v in violations), violations


def test_evaluation_module_imports_allowed(tmp_path: Path) -> None:
    """src/evaluation/ modules importing evaluation must NOT be flagged."""
    src = tmp_path / "src"
    eval_dir = src / "evaluation"
    _write(eval_dir / "__init__.py", "")
    _write(
        eval_dir / "runner.py",
        "from src.evaluation import schemas\nfrom evaluation import schemas as s2\n",
    )
    violations = scanner.check_production_imports_evaluation(src)
    assert violations == [], violations


def test_clean_codebase_passes(tmp_path: Path) -> None:
    """A clean production module must produce no violations."""
    src = tmp_path / "src"
    eval_dir = src / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "__init__.py").write_text("", encoding="utf-8")
    _write(
        src / "services" / "clean.py",
        '"""Clean production module."""\n\n\ndef run(query: str) -> str:\n'
        '    return query\n',
    )
    assert scanner.check_production_imports_evaluation(src) == []
    assert scanner.check_sealed_patterns(src, None) == []


def test_scanner_exit_code_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() returns 0 when the codebase is clean."""
    backend = tmp_path
    src = backend / "src"
    eval_dir = src / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "__init__.py").write_text("", encoding="utf-8")
    _write(
        src / "services" / "clean.py",
        '"""Clean production module."""\n\n\ndef run() -> int:\n    return 0\n',
    )
    monkeypatch.setattr(scanner, "BACKEND_DIR", backend)
    monkeypatch.setattr(scanner, "SRC_DIR", src)
    monkeypatch.setattr(scanner, "CONFIG_DIR", backend / "config")
    monkeypatch.setattr(scanner, "PROMPTS_DIR", backend / "prompts")
    monkeypatch.setattr(scanner, "SCRIPTS_DIR", backend / "scripts")
    monkeypatch.setattr(
        scanner,
        "SEALED_QUESTIONS_FILE",
        backend / "eval_data" / "phase5" / "sealed" / "questions.jsonl",
    )
    assert scanner.main() == 0


def test_scanner_exit_code_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() returns 1 when leakage is detected."""
    backend = tmp_path
    src = backend / "src"
    eval_dir = src / "evaluation"
    eval_dir.mkdir(parents=True)
    (eval_dir / "__init__.py").write_text("", encoding="utf-8")
    _write(
        src / "services" / "bad.py",
        "from src.evaluation import eval_runner\n",
    )
    monkeypatch.setattr(scanner, "BACKEND_DIR", backend)
    monkeypatch.setattr(scanner, "SRC_DIR", src)
    monkeypatch.setattr(scanner, "CONFIG_DIR", backend / "config")
    monkeypatch.setattr(scanner, "PROMPTS_DIR", backend / "prompts")
    monkeypatch.setattr(scanner, "SCRIPTS_DIR", backend / "scripts")
    monkeypatch.setattr(
        scanner,
        "SEALED_QUESTIONS_FILE",
        backend / "eval_data" / "phase5" / "sealed" / "questions.jsonl",
    )
    assert scanner.main() == 1
