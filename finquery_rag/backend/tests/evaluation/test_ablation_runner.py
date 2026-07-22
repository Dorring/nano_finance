"""Tests for src.evaluation.ablation."""
from __future__ import annotations

import copy

from src.evaluation.ablation import (
    ABLATION_VARIANTS,
    ablation_report,
    get_ablation_config,
    is_production_safe,
    validate_ablation_config,
)

_BASE_CONFIG = {
    "n_results": 5,
    "min_score_threshold": 0.1,
    "disable_bm25": False,
    "disable-dense": False,
    "disable-reranker": False,
    "disable-query-rewrite": False,
    "disable-hierarchical-context": False,
    "disable-calculation-pipeline": False,
    "disable-answerability": False,
    "disable-validation-pipeline": False,
    "disable-citation-validation": False,
}


class TestEachVariantChangesOneComponent:
    def test_each_variant_changes_one_component(self) -> None:
        """A0 changes nothing; A1–A9 change exactly one key."""
        for variant in ABLATION_VARIANTS:
            vid = variant["id"]
            diff = variant["config_diff"]
            errors = validate_ablation_config(vid, diff)
            assert errors == [], f"{vid}: {errors}"
            if vid == "A0":
                assert len(diff) == 0
            else:
                assert len(diff) == 1


class TestProductionDefaultNotModified:
    def test_production_default_not_modified(self) -> None:
        """get_ablation_config must not mutate the base_config."""
        for variant in ABLATION_VARIANTS:
            base_copy = copy.deepcopy(_BASE_CONFIG)
            _ = get_ablation_config(_BASE_CONFIG, variant["id"])
            assert _BASE_CONFIG == base_copy, f"base modified by {variant['id']}"


class TestNoValidationNotProductionDefault:
    def test_no_validation_not_production_default(self) -> None:
        """A8 and A9 (validation-disabling) must not be production-safe."""
        assert is_production_safe("A8") is False
        assert is_production_safe("A9") is False
        assert is_production_safe("A0") is True
        assert is_production_safe("A1") is True


class TestAllVariantsGenerateConfig:
    def test_all_variants_generate_config(self) -> None:
        """Every variant produces a valid, non-identical config (except A0)."""
        for variant in ABLATION_VARIANTS:
            vid = variant["id"]
            config = get_ablation_config(_BASE_CONFIG, vid)
            assert isinstance(config, dict)
            assert config is not _BASE_CONFIG
            for key, val in variant["config_diff"].items():
                assert config[key] == val
            if vid == "A0":
                assert config == _BASE_CONFIG


class TestAblationReport:
    def test_ablation_report_structure(self) -> None:
        """Report has expected keys and baseline is A0."""
        results = {
            "A0": {"macro_strict_pass_rate": 0.8},
            "A1": {"macro_strict_pass_rate": 0.7},
            "A8": {"macro_strict_pass_rate": 0.9},
        }
        report = ablation_report(results)
        assert report["baseline_id"] == "A0"
        assert len(report["variants"]) == len(ABLATION_VARIANTS)
        assert report["production_safe"]["A8"] is False
        assert abs(
            report["deltas"]["A1"]["delta_macro_strict_pass_rate"] - (-0.1)
        ) < 1e-9
