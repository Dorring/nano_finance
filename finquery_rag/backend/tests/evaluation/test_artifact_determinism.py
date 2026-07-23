"""Tests for artifact determinism in the report builder."""
from __future__ import annotations

import json
import re

from src.evaluation.report_builder import (
    build_final_report,
    build_run_manifest_output,
    validate_report_completeness,
)


def _sample_report() -> dict:
    """Return a minimal valid report for determinism checks."""
    manifest = build_run_manifest_output(
        git_commit="abc123",
        git_dirty=False,
        predictions_sha256="pred_hash",
        questions_sha256="q_hash",
        labels_sha256="l_hash",
        case_count=5,
        run_type="sealed",
        n_results=5,
        random_seed=42,
    )
    return build_final_report(
        metrics={"macro_strict_pass_rate": 0.8},
        slices={"intent": {"document_qa": 0.9}},
        failures={"categories": {}},
        ablations={"baseline_id": "A0"},
        calibration={"selected": {"n_results": 5}},
        confidence_intervals={"macro_strict_pass_rate": [0.7, 0.9]},
        manifest=manifest,
    )


class TestReportDeterministic:
    def test_report_deterministic(self) -> None:
        """Same inputs → byte-identical JSON output."""
        r1 = _sample_report()
        r2 = _sample_report()
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2


class TestReportNoTimestampInCoreFields:
    def test_report_no_timestamp_in_core_fields(self) -> None:
        """Core report fields must not contain timestamp-like keys."""
        report = _sample_report()
        forbidden_patterns = (
            "timestamp",
            "created_at",
            "generated_at",
            "updated_at",
            "run_time",
            "wall_clock",
        )

        def _check(obj: object) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    key_lower = str(key).lower()
                    for pattern in forbidden_patterns:
                        assert pattern not in key_lower, (
                            f"forbidden key '{key}' contains '{pattern}'"
                        )
                    _check(val)
            elif isinstance(obj, list):
                for item in obj:
                    _check(item)

        _check(report)


class TestReportNoUsername:
    def test_report_no_username(self) -> None:
        """Report must not contain username/user/author fields."""
        report = _sample_report()
        forbidden_patterns = ("username", "user", "author", "operator")
        text = json.dumps(report, sort_keys=True).lower()
        for pattern in forbidden_patterns:
            assert pattern not in text, f"report contains '{pattern}'"


class TestReportNoAbsolutePaths:
    def test_report_no_absolute_paths(self) -> None:
        """Report must not contain absolute file paths."""
        report = _sample_report()
        text = json.dumps(report, sort_keys=True)
        # Match Unix absolute paths /home/... or /mnt/... or /usr/...
        # and Windows absolute paths C:\... or Y:\...
        unix_pattern = r"(?<![\w])/(?:home|mnt|usr|tmp|var|opt|root|etc)/[^\s\"']+"
        windows_pattern = r"[A-Z]:\\[^\s\"']+"
        assert not re.search(unix_pattern, text), "report contains Unix absolute path"
        assert not re.search(windows_pattern, text), (
            "report contains Windows absolute path"
        )


class TestValidateReportCompleteness:
    def test_complete_report_passes(self) -> None:
        """A report with all required fields has no missing fields."""
        report = _sample_report()
        missing = validate_report_completeness(report)
        assert missing == []

    def test_incomplete_report_reports_missing(self) -> None:
        """A report missing fields lists them."""
        missing = validate_report_completeness({"metrics": {}})
        assert "slices" in missing
        assert "manifest" in missing
