"""Deployment preflight report for FinQuery RAG.

The preflight report aggregates existing non-secret checks into one JSON payload:
health readiness, migration readiness, fixture audit, eval gate, and retrieval
diagnostics. It does not call LLMs or embeddings.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.evaluation import (
    audit_evaluation_fixtures,
    compare_reports,
    diagnose_retrieval,
    evaluate_predictions,
    load_jsonl_cases,
    load_jsonl_predictions,
)
from .health import collect_health_snapshot
from .migration_audit import audit_migration_readiness


DEFAULT_CASES_PATH = "eval/golden_smoke.jsonl"
DEFAULT_PREDICTIONS_PATH = "eval/predictions_smoke.jsonl"
DEFAULT_BASELINE_PATH = "eval/baseline_smoke_report.json"


def build_preflight_report(
    *,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    baseline_path: str | Path | None = DEFAULT_BASELINE_PATH,
    bm25_db_path: str | None = None,
    registry_db_path: str | None = None,
    chroma_path: str | None = None,
    trace_db_path: str | None = None,
    feedback_db_path: str | None = None,
    min_pass_rate: float = 1.0,
    max_missing: int = 0,
    regression_tolerance: float = 0.0,
    min_cases: int = 1,
    required_tags: tuple[str, ...] = (),
    require_expected_intent: bool = False,
) -> dict[str, Any]:
    """Build a deterministic deployment preflight report."""
    _validate_preflight_thresholds(
        min_pass_rate=min_pass_rate,
        max_missing=max_missing,
        regression_tolerance=regression_tolerance,
        min_cases=min_cases,
    )
    health = collect_health_snapshot(
        bm25_db_path=bm25_db_path,
        trace_db_path=trace_db_path,
        feedback_db_path=feedback_db_path,
    )
    migration = audit_migration_readiness(
        bm25_db_path=bm25_db_path,
        registry_db_path=registry_db_path,
        chroma_path=chroma_path,
    )
    cases = load_jsonl_cases(cases_path)
    predictions = load_jsonl_predictions(predictions_path)
    fixture_audit = audit_evaluation_fixtures(
        cases,
        min_cases=min_cases,
        required_tags=required_tags,
        require_expected_intent=require_expected_intent,
    )
    eval_report = evaluate_predictions(cases, predictions)
    eval_gate = _build_eval_gate_summary(
        eval_report,
        min_pass_rate=min_pass_rate,
        max_missing=max_missing,
    )
    comparison = None
    if baseline_path:
        baseline = _load_json_object(baseline_path, "baseline")
        comparison = compare_reports(
            baseline,
            eval_report,
            regression_tolerance=regression_tolerance,
        )
    retrieval = diagnose_retrieval(cases, predictions, ks=(1, 3, 5))

    sections = {
        "health": bool(health.get("ready")),
        "migration": bool(migration.get("passed")),
        "fixture_audit": bool(fixture_audit.get("passed")),
        "eval_gate": bool(eval_gate.get("passed")),
        "baseline_comparison": True if comparison is None else bool(comparison.get("passed")),
    }
    passed = all(sections.values())
    return {
        "passed": passed,
        "sections": sections,
        "summary": {
            "failed_sections": [name for name, ok in sections.items() if not ok],
            "health_status": health.get("status"),
            "migration_high_risks": migration.get("summary", {}).get("high_risk_count", 0),
            "eval_pass_rate": eval_report.get("summary", {}).get("pass_rate"),
            "eval_missing_predictions": eval_report.get("summary", {}).get("missing_predictions"),
            "retrieval_recall_at_5": retrieval.get("summary", {}).get("recall_at_k", {}).get("5"),
        },
        "health": health,
        "migration": migration,
        "fixture_audit": fixture_audit,
        "eval_gate": eval_gate,
        "eval_report": eval_report,
        "baseline_comparison": comparison,
        "retrieval_diagnostics": retrieval,
    }


def _validate_preflight_thresholds(*, min_pass_rate: float, max_missing: int, regression_tolerance: float, min_cases: int) -> None:
    if min_pass_rate < 0 or min_pass_rate > 1:
        raise ValueError("min_pass_rate must be between 0 and 1")
    if max_missing < 0:
        raise ValueError("max_missing must be >= 0")
    if regression_tolerance < 0:
        raise ValueError("regression_tolerance must be >= 0")
    if min_cases < 0:
        raise ValueError("min_cases must be >= 0")


def _build_eval_gate_summary(report: dict[str, Any], *, min_pass_rate: float, max_missing: int) -> dict[str, Any]:
    summary = report.get("summary") or {}
    pass_rate = float(summary.get("pass_rate") or 0.0)
    missing = int(summary.get("missing_predictions") or 0)
    checks = [
        {
            "name": "min_pass_rate",
            "passed": pass_rate >= min_pass_rate,
            "actual": pass_rate,
            "expected": min_pass_rate,
        },
        {
            "name": "max_missing_predictions",
            "passed": missing <= max_missing,
            "actual": missing,
            "expected": max_missing,
        },
    ]
    failed = [item for item in checks if not item["passed"]]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
    }


def _load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} report must be a JSON object")
    return payload
