"""Tests for feature-flag injection and the v2 ablation variant mapping."""

from __future__ import annotations

from src.evaluation.ablation import (
    ABLATION_VARIANTS,
    get_ablation_config,
    get_variant_feature_flags,
)
from src.evaluation.feature_flag_injection import (
    apply_feature_flags_to_engine_kwargs,
    assert_feature_flags_enforced,
)
from src.evaluation.schemas import EvaluationFeatureFlags

_ALL_FLAG_FIELDS = (
    "dense_enabled",
    "bm25_enabled",
    "reranker_enabled",
    "query_rewrite_enabled",
    "hierarchical_context_enabled",
    "calculator_enabled",
    "answerability_enabled",
    "post_validation_enabled",
    "citation_validation_enabled",
)


def _flags_tuple(flags: EvaluationFeatureFlags) -> tuple[bool, ...]:
    return tuple(getattr(flags, name) for name in _ALL_FLAG_FIELDS)


class TestAllVariantsHaveDistinctFlags:
    def test_all_variants_have_distinct_flags(self) -> None:
        """All 10 variants (A0-A9) produce distinct EvaluationFeatureFlags."""
        seen: dict[tuple[bool, ...], str] = {}
        for variant in ABLATION_VARIANTS:
            vid = variant["id"]
            flags = get_variant_feature_flags(vid)
            key = _flags_tuple(flags)
            assert key not in seen, (
                f"variant {vid} duplicates the flags of variant {seen[key]}"
            )
            seen[key] = vid
        assert len(seen) == 10


class TestA0FullSystemAllTrue:
    def test_a0_full_system_all_true(self) -> None:
        """A0 (Full System) has all 9 flags set to True."""
        flags = get_variant_feature_flags("A0")
        for name in _ALL_FLAG_FIELDS:
            assert getattr(flags, name) is True, f"A0.{name} should be True"


class TestA1DenseOnly:
    def test_a1_dense_only_disables_bm25(self) -> None:
        """A1 (Dense Only) has bm25_enabled=False, all others True."""
        flags = get_variant_feature_flags("A1")
        assert flags.bm25_enabled is False
        for name in _ALL_FLAG_FIELDS:
            if name == "bm25_enabled":
                continue
            assert getattr(flags, name) is True, f"A1.{name} should be True"


class TestApplyFeatureFlagsToEngineKwargs:
    def test_full_flags_produce_hybrid_with_reranker_and_pipelines(self) -> None:
        flags = EvaluationFeatureFlags()  # all True
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["use_hybrid"] is True
        assert kwargs["enable_calculation_pipeline"] is True
        assert kwargs["enable_validation_pipeline"] is True
        assert kwargs["reranker_name"] == "default"

    def test_dense_only_disables_hybrid(self) -> None:
        flags = EvaluationFeatureFlags(
            dense_enabled=True,
            bm25_enabled=False,
        )
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["use_hybrid"] is False

    def test_bm25_only_keeps_hybrid_for_bm25_path(self) -> None:
        # BM25-only needs use_hybrid=True so the hybrid retrieval path
        # actually calls BM25; dense is then disabled via runtime patch
        # (not via use_hybrid, which would disable BM25 instead).
        flags = EvaluationFeatureFlags(
            dense_enabled=False,
            bm25_enabled=True,
        )
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["use_hybrid"] is True

    def test_reranker_disabled_passes_none(self) -> None:
        flags = EvaluationFeatureFlags(reranker_enabled=False)
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["reranker_name"] is None

    def test_calculator_disabled(self) -> None:
        flags = EvaluationFeatureFlags(calculator_enabled=False)
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["enable_calculation_pipeline"] is False

    def test_validation_disabled_only_when_all_three_off(self) -> None:
        # post_validation alone keeps the pipeline on (answerability/citation on)
        flags = EvaluationFeatureFlags(
            post_validation_enabled=False,
            answerability_enabled=True,
            citation_validation_enabled=True,
        )
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["enable_validation_pipeline"] is True

        flags = EvaluationFeatureFlags(
            post_validation_enabled=False,
            answerability_enabled=False,
            citation_validation_enabled=False,
        )
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert kwargs["enable_validation_pipeline"] is False

    def test_kwargs_only_contain_expected_keys(self) -> None:
        flags = EvaluationFeatureFlags()
        kwargs = apply_feature_flags_to_engine_kwargs(flags)
        assert set(kwargs.keys()) == {
            "use_hybrid",
            "enable_calculation_pipeline",
            "enable_validation_pipeline",
            "reranker_name",
        }


class TestAblationConfigBackwardCompatible:
    def test_ablation_config_backward_compatible(self) -> None:
        """get_ablation_config("A0") returns a dict (not just flags) with expected keys."""
        config = get_ablation_config("A0")
        assert isinstance(config, dict)
        # Old config-format keys preserved
        assert "id" in config
        assert "name" in config
        assert "config_diff" in config
        assert "production_safe" in config
        # New feature_flags key present
        assert "feature_flags" in config
        assert isinstance(config["feature_flags"], EvaluationFeatureFlags)
        assert config["id"] == "A0"

    def test_old_style_two_arg_call_still_works(self) -> None:
        """Two-arg call returns the merged base config (no feature_flags key)."""
        base = {"n_results": 5, "disable_bm25": False}
        config = get_ablation_config(base, "A1")
        assert isinstance(config, dict)
        assert config["disable_bm25"] is True
        assert config["n_results"] == 5
        assert "feature_flags" not in config

    def test_old_style_does_not_mutate_base(self) -> None:
        base = {"disable_bm25": False}
        original = dict(base)
        _ = get_ablation_config(base, "A1")
        assert base == original

    def test_new_style_returns_feature_flags_for_each_variant(self) -> None:
        for variant in ABLATION_VARIANTS:
            vid = variant["id"]
            config = get_ablation_config(vid)
            assert isinstance(config["feature_flags"], EvaluationFeatureFlags)


class TestFeatureFlagsDefaultAllTrue:
    def test_feature_flags_default_all_true(self) -> None:
        """Fresh EvaluationFeatureFlags() has all 9 flags True."""
        flags = EvaluationFeatureFlags()
        for name in _ALL_FLAG_FIELDS:
            assert getattr(flags, name) is True, f"default {name} should be True"
        assert len(_ALL_FLAG_FIELDS) == 9


class TestA5CalculatorEnabled:
    def test_a5_calculator_enabled(self) -> None:
        """A5 has calculator_enabled=True."""
        flags = get_variant_feature_flags("A5")
        assert flags.calculator_enabled is True


class TestA6NoCalculator:
    def test_a6_no_calculator(self) -> None:
        """A6 (No Calculator) has calculator_enabled=False, all others True."""
        flags = get_variant_feature_flags("A6")
        assert flags.calculator_enabled is False
        for name in _ALL_FLAG_FIELDS:
            if name == "calculator_enabled":
                continue
            assert getattr(flags, name) is True, f"A6.{name} should be True"


class TestA8NoPostGenerationValidation:
    def test_a8_disables_full_validation_pipeline(self) -> None:
        """A8 (No Post-generation Validation) disables the whole validation
        pipeline at construction time, so all three validation flags are
        False (post_validation, answerability, citation_validation)."""
        flags = get_variant_feature_flags("A8")
        assert flags.post_validation_enabled is False
        assert flags.answerability_enabled is False
        assert flags.citation_validation_enabled is False
        # Non-validation flags stay True.
        for name in _ALL_FLAG_FIELDS:
            if name in {
                "post_validation_enabled",
                "answerability_enabled",
                "citation_validation_enabled",
            }:
                continue
            assert getattr(flags, name) is True, f"A8.{name} should be True"


class TestAssertFeatureFlagsEnforced:
    def test_returns_empty_list_for_well_formed_fake_engine(self) -> None:
        """When flags match the engine state, no violations are reported."""

        class _FakeOrchestrator:
            _calculation_pipeline = None
            _validation_pipeline = None

        class _FakeEngine:
            use_hybrid = True  # bm25_enabled=True (default)
            reranker = None  # reranker_enabled=False
            _orchestrator = _FakeOrchestrator()
            _retrieval_pipeline = None
            _llm_gateway = None
            _context_builder = None
            _calculation_pipeline = None
            _validation_pipeline = None

        flags = EvaluationFeatureFlags(
            reranker_enabled=False,
            calculator_enabled=False,
            post_validation_enabled=False,
        )
        assert assert_feature_flags_enforced(flags, _FakeEngine()) == []

    def test_detects_reranker_when_disabled(self) -> None:
        class _FakeEngine:
            use_hybrid = True
            reranker = object()  # truthy — should not be present when disabled
            _orchestrator = None
            _retrieval_pipeline = None
            _llm_gateway = None
            _context_builder = None
            _validation_pipeline = None

        flags = EvaluationFeatureFlags(reranker_enabled=False)
        violations = assert_feature_flags_enforced(flags, _FakeEngine())
        assert any("reranker" in v for v in violations)

    def test_detects_missing_reranker_when_enabled(self) -> None:
        class _FakeEngine:
            use_hybrid = True
            reranker = None  # missing when should be present
            _orchestrator = None
            _retrieval_pipeline = None
            _llm_gateway = None
            _context_builder = None
            _validation_pipeline = None

        flags = EvaluationFeatureFlags(reranker_enabled=True)
        violations = assert_feature_flags_enforced(flags, _FakeEngine())
        assert any("reranker" in v for v in violations)
