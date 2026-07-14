"""Phase 82 tests: eval compare quality-gate input hardening."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import compare_reports
from src.eval_cli import main as eval_cli_main


def _write_report(path, pass_rate):
    path.write_text(
        json.dumps({"summary": {"pass_rate": pass_rate}, "cases": []}),
        encoding="utf-8",
    )


def test_compare_reports_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="regression_tolerance must be >= 0"):
        compare_reports(
            {"summary": {"pass_rate": 1.0}, "cases": []},
            {"summary": {"pass_rate": 0.9}, "cases": []},
            regression_tolerance=-0.1,
        )


def test_eval_cli_compare_rejects_negative_tolerance(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out = tmp_path / "comparison.json"
    _write_report(baseline, 1.0)
    _write_report(candidate, 0.9)

    code = eval_cli_main([
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--tolerance",
        "-0.01",
        "--out",
        str(out),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "tolerance must be >= 0" in captured.err
    assert not out.exists()


def test_eval_cli_compare_rejects_malformed_json_report(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text("{not json", encoding="utf-8")
    _write_report(candidate, 1.0)

    code = eval_cli_main([
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "baseline report must be valid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_eval_cli_compare_rejects_non_object_report(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text("[]", encoding="utf-8")
    _write_report(candidate, 1.0)

    code = eval_cli_main([
        "compare",
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
    ])
    captured = capsys.readouterr()

    assert code == 2
    assert "baseline report must be a JSON object" in captured.err
    assert captured.out == ""
