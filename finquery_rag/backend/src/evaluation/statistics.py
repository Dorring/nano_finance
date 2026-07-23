"""Statistical helpers for Phase 5 RAG evaluation.

All functions are deterministic given the same seed and fully offline.
No external dependencies beyond the Python standard library are used.

Functions:
    wilson_interval: Wilson score confidence interval for a proportion.
    bootstrap_ci: Non-parametric bootstrap CI for the mean of a sample.
    paired_bootstrap_difference: Paired bootstrap CI for the mean
        difference between two systems on the same cases.
    mcnemar_test: McNemar's test for paired binary outcomes.
    cluster_bootstrap_ci_by_document: Cluster bootstrap CI that resamples
        documents (clusters) rather than individual cases.
    exact_mcnemar_test: McNemar's test with exact binomial test for small
        samples and chi-square approximation for large samples.
    holm_correction: Holm-Bonferroni correction for multiple comparisons.
    proportion_ci: Wilson score CI for a proportion with full metadata.
    metric_with_ci: Wrapper returning the standard metric output format.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Any

__all__ = [
    "wilson_interval",
    "bootstrap_ci",
    "paired_bootstrap_difference",
    "mcnemar_test",
    "cluster_bootstrap_ci_by_document",
    "exact_mcnemar_test",
    "holm_correction",
    "proportion_ci",
    "metric_with_ci",
]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score confidence interval for a proportion.

    Args:
        successes: Number of observed successes (must be >= 0).
        total: Number of trials (must be > 0).
        z: Z-score for the desired confidence level (default 1.96 for 95%).

    Returns:
        A ``(low, high)`` tuple with the lower and upper bounds of the
        interval. Both bounds are clamped to ``[0.0, 1.0]``. When
        ``total <= 0`` the function returns ``(0.0, 0.0)``.
    """
    if total <= 0:
        return 0.0, 0.0
    if successes < 0:
        successes = 0
    if successes > total:
        successes = total
    n = total
    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)) / denominator
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def bootstrap_ci(
    data: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for the mean of ``data``.

    Resamples the data with replacement ``n_bootstrap`` times, computes
    the mean of each resample, and returns the percentile interval at the
    requested confidence level.

    Args:
        data: List of observed values.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level in ``(0, 1)``.
        seed: Random seed for reproducibility.

    Returns:
        A ``(low, high)`` tuple. Returns ``(0.0, 0.0)`` when ``data``
        is empty.
    """
    if not data:
        return 0.0, 0.0
    n = len(data)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return _percentile_interval(means, confidence)


def paired_bootstrap_difference(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a paired bootstrap CI for the mean difference ``a - b``.

    Both lists must have the same length and represent paired
    observations (e.g., the metric for the same evaluation case under
    two different systems). The per-case differences are resampled with
    replacement.

    Args:
        a: Values from system A.
        b: Values from system B (same length as ``a``).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level in ``(0, 1)``.
        seed: Random seed for reproducibility.

    Returns:
        A ``(low, high)`` tuple for the mean difference. Returns
        ``(0.0, 0.0)`` when the inputs are empty or mismatched in
        length.
    """
    if not a or len(a) != len(b):
        return 0.0, 0.0
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return _percentile_interval(means, confidence)


def mcnemar_test(a_correct: list[bool], b_correct: list[bool]) -> dict[str, Any]:
    """McNemar's test for paired binary outcomes.

    Compares two systems' correctness on the same set of cases. The
    test focuses on the discordant pairs:

        b_count: cases where A is correct but B is not.
        c_count: cases where B is correct but A is not.

    Uses the Edwards continuity correction
    ``(|b - c| - 1)^2 / (b + c)`` and derives the p-value from the
    chi-square survival function with 1 degree of freedom
    (``erfc(sqrt(statistic / 2))``).

    Args:
        a_correct: Per-case correctness of system A.
        b_correct: Per-case correctness of system B.

    Returns:
        A dict with keys ``"statistic"``, ``"p_value"``, and
        ``"n_discordant"``.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(
            "a_correct and b_correct must have the same length, "
            f"got {len(a_correct)} and {len(b_correct)}"
        )
    b_count = sum(1 for i in range(len(a_correct)) if a_correct[i] and not b_correct[i])
    c_count = sum(1 for i in range(len(a_correct)) if not a_correct[i] and b_correct[i])
    n_discordant = b_count + c_count
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_discordant": 0}
    abs_diff = abs(b_count - c_count)
    statistic = ((abs_diff - 1) ** 2) / n_discordant
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "n_discordant": n_discordant,
    }


def _percentile_interval(
    sorted_values: list[float], confidence: float
) -> tuple[float, float]:
    """Return the percentile interval from a sorted list of bootstrap means."""
    n = len(sorted_values)
    if n == 0:
        return 0.0, 0.0
    alpha = (1.0 - confidence) / 2.0
    low_idx = int(math.floor(alpha * n))
    high_idx = int(math.ceil((1.0 - alpha) * n)) - 1
    low_idx = min(max(low_idx, 0), n - 1)
    high_idx = min(max(high_idx, 0), n - 1)
    return sorted_values[low_idx], sorted_values[high_idx]


def cluster_bootstrap_ci_by_document(
    values: list[float],
    document_ids: list[str],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Cluster bootstrap CI for the mean, resampling documents not cases.

    Cases sharing a ``document_id`` are treated as a cluster. On each
    resample, documents are drawn with replacement and ALL cases from
    each drawn document are included. This correctly accounts for
    within-document correlation and avoids underestimating variance.

    Args:
        values: Per-case metric values.
        document_ids: Per-case document cluster identifiers (same length
            as ``values``).
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level in ``(0, 1)``.
        seed: Random seed for reproducibility.

    Returns:
        A ``(low, high)`` tuple. Returns ``(0.0, 0.0)`` when ``values``
        is empty or the lengths mismatch.
    """
    if not values or len(values) != len(document_ids):
        return 0.0, 0.0
    doc_to_values: dict[str, list[float]] = {}
    for doc_id, val in zip(document_ids, values):
        doc_to_values.setdefault(doc_id, []).append(val)
    unique_docs = list(doc_to_values.keys())
    n_docs = len(unique_docs)
    if n_docs == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        count = 0
        for _ in range(n_docs):
            doc = unique_docs[rng.randrange(n_docs)]
            for v in doc_to_values[doc]:
                total += v
                count += 1
        means.append(total / count if count > 0 else 0.0)
    means.sort()
    return _percentile_interval(means, confidence)


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Exact McNemar test for paired binary outcomes.

    Uses the exact binomial test when ``b + c < 25`` (small samples) and
    the chi-square approximation with Edwards continuity correction
    otherwise.

    Args:
        b: Discordant pairs where system A is correct and B is not.
        c: Discordant pairs where system B is correct and A is not.

    Returns:
        A dict with ``"statistic"``, ``"p_value"``, and ``"method"``
        (``"exact"`` or ``"chi_square"``).
    """
    n = b + c
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "method": "exact"}
    if n < 25:
        # Under the null, b ~ Binomial(n, 0.5). Two-sided p-value.
        k = min(b, c)
        p_one_sided = sum(math.comb(n, i) * (0.5**n) for i in range(k + 1))
        p_value = min(1.0, 2.0 * p_one_sided)
        return {
            "statistic": float(abs(b - c)),
            "p_value": p_value,
            "method": "exact",
        }
    statistic = ((abs(b - c) - 1) ** 2) / n
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "method": "chi_square",
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction for multiple comparisons.

    Adjusts p-values to control the family-wise error rate. The
    adjusted p-values are returned in the SAME order as the input.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of adjusted p-values in the original order, each in
        ``[0.0, 1.0]``.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adj = min(1.0, max(adj, prev))
        adjusted[orig_idx] = adj
        prev = adj
    return adjusted


def proportion_ci(
    successes: int, total: int, confidence: float = 0.95
) -> dict[str, Any]:
    """Wilson score confidence interval for a proportion.

    Args:
        successes: Number of observed successes.
        total: Number of trials.
        confidence: Confidence level in ``(0, 1)``.

    Returns:
        A dict with ``numerator``, ``denominator``, ``point_estimate``,
        ``ci_lower``, and ``ci_upper``. When ``total == 0`` returns
        ``{"value": None, "n_applicable": 0}``.
    """
    if total == 0:
        return {"value": None, "n_applicable": 0}
    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    low, high = wilson_interval(successes, total, z=z)
    return {
        "numerator": successes,
        "denominator": total,
        "point_estimate": successes / total,
        "ci_lower": low,
        "ci_upper": high,
    }


def metric_with_ci(successes: int, total: int) -> dict[str, Any]:
    """Return the full metric output format with a 95% Wilson CI.

    When ``total == 0`` returns ``{"value": None, "n_applicable": 0}``.
    """
    if total == 0:
        return {"value": None, "n_applicable": 0}
    return proportion_ci(successes, total, confidence=0.95)
