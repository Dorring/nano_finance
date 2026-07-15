"""Run the commit-safe FinQuery evaluation smoke gate.

This script is intentionally dependency-light so GitHub Actions can call one
stable entrypoint instead of duplicating eval_cli arguments in workflow YAML.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_cli import main as eval_cli_main  # noqa: E402


def main() -> int:
    artifact_root = os.getenv("FINQUERY_EVAL_ARTIFACT_DIR")
    artifacts = Path(artifact_root) if artifact_root else Path(tempfile.gettempdir()) / "finquery_eval_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cases = ROOT / "eval" / "golden_smoke.jsonl"
    predictions = ROOT / "eval" / "predictions_smoke.jsonl"
    audit_code = eval_cli_main([
        "audit-fixtures",
        "--cases",
        str(cases),
        "--min-cases",
        "12",
        "--required-tag",
        "smoke",
        "--required-tag",
        "citation",
        "--required-tag",
        "no_answer",
        "--required-tag",
        "calculation",
        "--require-expected-intent",
        "--out",
        str(artifacts / "smoke_fixture_audit.json"),
    ])
    if audit_code != 0:
        return audit_code
    gate_code = eval_cli_main([
        "gate",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--baseline",
        str(ROOT / "eval" / "baseline_smoke_report.json"),
        "--min-pass-rate",
        "1.0",
        "--max-missing",
        "0",
        "--tolerance",
        "0.0",
        "--out",
        str(artifacts / "smoke_report.json"),
        "--comparison-out",
        str(artifacts / "smoke_comparison.json"),
        "--junit-out",
        str(artifacts / "smoke_gate.xml"),
    ])
    if gate_code != 0:
        return gate_code
    return eval_cli_main([
        "retrieval-diagnostics",
        "--cases",
        str(cases),
        "--predictions",
        str(predictions),
        "--k",
        "1",
        "--k",
        "3",
        "--k",
        "5",
        "--out",
        str(artifacts / "smoke_retrieval_diagnostics.json"),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
