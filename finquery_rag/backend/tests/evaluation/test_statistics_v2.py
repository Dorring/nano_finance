"""Tests for new statistics functions in src.evaluation.statistics."""

from __future__ import annotations

from src.evaluation.statistics import (
    cluster_bootstrap_ci_by_document,
    exact_mcnemar_test,
    holm_correction,
    proportion_ci,
)


class TestClusterBootstrapByDocument:
    def test_returns_valid_ci(self) -> None:
        """Cluster bootstrap returns a valid interval."""
        values = [1.0, 0.0, 1.0, 0.0]
        doc_ids = ["doc_a", "doc_a", "doc_b", "doc_b"]
        low, high = cluster_bootstrap_ci_by_document(values, doc_ids, n_resamples=500)
        assert 0.0 <= low <= high <= 1.0

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        values = [0.5, 0.3, 0.8, 0.1]
        doc_ids = ["a", "a", "b", "b"]
        ci1 = cluster_bootstrap_ci_by_document(values, doc_ids, n_resamples=200)
        ci2 = cluster_bootstrap_ci_by_document(values, doc_ids, n_resamples=200)
        assert ci1 == ci2

    def test_empty_returns_zero(self) -> None:
        """Empty input returns (0.0, 0.0)."""
        assert cluster_bootstrap_ci_by_document([], []) == (0.0, 0.0)

    def test_mismatched_lengths_returns_zero(self) -> None:
        """Mismatched lengths return (0.0, 0.0)."""
        assert cluster_bootstrap_ci_by_document([1.0, 2.0], ["a"]) == (0.0, 0.0)


class TestExactMcNemar:
    def test_uses_exact_for_small_samples(self) -> None:
        """Small samples (b+c < 25) use the exact binomial method."""
        result = exact_mcnemar_test(3, 1)
        assert result["method"] == "exact"
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["statistic"] == 2.0  # |3-1|

    def test_uses_chi_square_for_large_samples(self) -> None:
        """Large samples (b+c >= 25) use the chi-square approximation."""
        result = exact_mcnemar_test(20, 5)
        assert result["method"] == "chi_square"
        assert 0.0 <= result["p_value"] <= 1.0

    def test_no_discordant(self) -> None:
        """No discordant pairs → p_value 1.0."""
        result = exact_mcnemar_test(0, 0)
        assert result["p_value"] == 1.0
        assert result["method"] == "exact"

    def test_symmetric(self) -> None:
        """Swapping b and c gives the same p-value."""
        r1 = exact_mcnemar_test(5, 2)
        r2 = exact_mcnemar_test(2, 5)
        assert r1["p_value"] == r2["p_value"]


class TestHolmCorrection:
    def test_returns_corrected_pvalues(self) -> None:
        """Holm correction multiplies smallest p by m."""
        p_values = [0.01, 0.02, 0.03]
        adjusted = holm_correction(p_values)
        assert len(adjusted) == 3
        assert all(0.0 <= p <= 1.0 for p in adjusted)
        # Smallest p (0.01) is multiplied by m=3 → 0.03
        assert abs(adjusted[0] - 0.03) < 1e-9

    def test_preserves_order(self) -> None:
        """Adjusted p-values are returned in the original order."""
        p_values = [0.03, 0.01, 0.02]
        adjusted = holm_correction(p_values)
        # 0.01 is smallest → 0.01*3 = 0.03 at index 1
        assert abs(adjusted[1] - 0.03) < 1e-9

    def test_monotonicity(self) -> None:
        """Adjusted p-values are non-decreasing in sorted order."""
        p_values = [0.001, 0.005, 0.01, 0.04]
        adjusted = holm_correction(p_values)
        sorted_adj = [adjusted[i] for i in sorted(range(4), key=lambda i: p_values[i])]
        for i in range(len(sorted_adj) - 1):
            assert sorted_adj[i] <= sorted_adj[i + 1] + 1e-12

    def test_empty(self) -> None:
        """Empty input returns empty list."""
        assert holm_correction([]) == []

    def test_caps_at_one(self) -> None:
        """Adjusted p-values are capped at 1.0."""
        p_values = [0.5, 0.6, 0.7]
        adjusted = holm_correction(p_values)
        assert all(p <= 1.0 for p in adjusted)


class TestProportionCI:
    def test_returns_null_when_total_zero(self) -> None:
        """total=0 returns the null sentinel."""
        result = proportion_ci(0, 0)
        assert result == {"value": None, "n_applicable": 0}

    def test_returns_correct_ci_for_normal_case(self) -> None:
        """50/100 gives a CI containing 0.5."""
        result = proportion_ci(50, 100)
        assert result["numerator"] == 50
        assert result["denominator"] == 100
        assert abs(result["point_estimate"] - 0.5) < 1e-9
        assert 0.0 <= result["ci_lower"] <= 0.5 <= result["ci_upper"] <= 1.0

    def test_all_successes(self) -> None:
        """All successes → CI near 1.0."""
        result = proportion_ci(100, 100)
        assert result["point_estimate"] == 1.0
        assert result["ci_lower"] > 0.9
        assert result["ci_upper"] == 1.0

    def test_all_failures(self) -> None:
        """All failures → CI near 0.0."""
        result = proportion_ci(0, 100)
        assert result["point_estimate"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] < 0.1
