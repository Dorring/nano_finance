"""Tests for src.evaluation.statistics."""
from __future__ import annotations

from src.evaluation.statistics import (
    bootstrap_ci,
    mcnemar_test,
    paired_bootstrap_difference,
    wilson_interval,
)


class TestWilsonInterval:
    def test_wilson_interval_all_successes(self) -> None:
        """All successes → interval should be near 1.0."""
        low, high = wilson_interval(100, 100)
        assert low > 0.9
        assert abs(high - 1.0) < 1e-9
        assert low <= high

    def test_wilson_interval_all_failures(self) -> None:
        """All failures → interval should be near 0.0."""
        low, high = wilson_interval(0, 100)
        assert low == 0.0
        assert high < 0.1
        assert low <= high

    def test_wilson_interval_half(self) -> None:
        """Half successes → interval should contain 0.5."""
        low, high = wilson_interval(50, 100)
        assert low < 0.5 < high
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0


class TestBootstrapCI:
    def test_bootstrap_ci_deterministic(self) -> None:
        """Same seed → identical result."""
        data = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
        ci1 = bootstrap_ci(data, n_bootstrap=500, seed=123)
        ci2 = bootstrap_ci(data, n_bootstrap=500, seed=123)
        assert ci1 == ci2

    def test_bootstrap_ci_different_seeds_may_differ(self) -> None:
        """Different seeds can produce different (but valid) intervals."""
        data = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
        ci1 = bootstrap_ci(data, n_bootstrap=500, seed=1)
        ci2 = bootstrap_ci(data, n_bootstrap=500, seed=999)
        assert ci1[0] <= ci1[1]
        assert ci2[0] <= ci2[1]

    def test_bootstrap_ci_empty(self) -> None:
        """Empty data → (0.0, 0.0)."""
        assert bootstrap_ci([]) == (0.0, 0.0)


class TestPairedBootstrapDifference:
    def test_paired_bootstrap_difference(self) -> None:
        """Paired bootstrap of identical systems → interval near 0."""
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.5, 0.6, 0.7, 0.8, 0.9]
        low, high = paired_bootstrap_difference(a, b, n_bootstrap=500, seed=42)
        assert low <= 0.0 <= high

    def test_paired_bootstrap_deterministic(self) -> None:
        """Same seed → identical result."""
        a = [0.5, 0.6, 0.7, 0.8, 0.9]
        b = [0.4, 0.5, 0.6, 0.7, 0.8]
        ci1 = paired_bootstrap_difference(a, b, n_bootstrap=500, seed=7)
        ci2 = paired_bootstrap_difference(a, b, n_bootstrap=500, seed=7)
        assert ci1 == ci2

    def test_paired_bootstrap_mismatched_lengths(self) -> None:
        """Mismatched lengths → (0.0, 0.0)."""
        assert paired_bootstrap_difference([1.0, 2.0], [1.0]) == (0.0, 0.0)


class TestMcNemarTest:
    def test_mcnemar_test_no_discordance(self) -> None:
        """No discordant pairs → statistic 0, p_value 1.0."""
        a = [True, True, False, False, True]
        b = [True, True, False, False, True]
        result = mcnemar_test(a, b)
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["n_discordant"] == 0

    def test_mcnemar_test_with_discordance(self) -> None:
        """Discordant pairs → statistic > 0, p_value < 1.0."""
        a = [True, True, True, True, True, False, False, False, False, False]
        b = [False, False, False, False, False, True, True, True, True, True]
        result = mcnemar_test(a, b)
        assert result["n_discordant"] == 10
        assert result["statistic"] > 0.0
        assert 0.0 <= result["p_value"] < 1.0

    def test_mcnemar_test_partial_discordance(self) -> None:
        """Partial discordance is correctly counted."""
        a = [True, False, True, False]
        b = [False, True, True, True]
        result = mcnemar_test(a, b)
        assert result["n_discordant"] == 3
