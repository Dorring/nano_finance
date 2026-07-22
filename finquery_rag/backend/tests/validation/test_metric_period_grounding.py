"""Phase 4 production regression tests for numeric-metric-period grounding.

Verifies that ``NumericClaimValidator._find_supporting_evidence`` requires
the value, the metric (or a known alias), and the period (year) to all
appear on the SAME evidence chunk. A correct value bound to the wrong
period or wrong metric must be flagged as ungrounded, even if the value
exists elsewhere in the evidence set. Uses the ``financial_calculation``
policy which enforces strict numeric grounding.
"""
from __future__ import annotations

import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.domain.evidence import EvidenceItem
from src.domain.validation import ExtractedClaim
from src.validation.numeric_claim_validator import (
    CODE_NUMERIC_UNGROUND,
    NumericClaimValidator,
)
from src.validation.validation_policy import get_policy_for_intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence(
    content: str,
    *,
    chunk_id: str = "c1",
    document_name: str = "annual_report.pdf",
) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        content=content,
        document_name=document_name,
        page=12,
        content_type="text",
        score=0.9,
        rerank_score=None,
        metadata={},
    )


def _calc_policy():
    return get_policy_for_intent("financial_calculation")


def _claim(
    *,
    value: Decimal,
    metric: str = "revenue",
    period: str = "2024",
    raw_text: str = "125",
    claim_id: str = "claim_001",
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=claim_id,
        claim_type="amount",
        raw_text=raw_text,
        metric=metric,
        value=value,
        unit="base",
        scale=None,
        currency=None,
        period=period,
        citation_refs=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_value_in_evidence_but_wrong_period():
    """Value 125 exists for 2024, but the claim binds it to 2023 -> ungrounded."""
    claim = _claim(value=Decimal("125"), metric="revenue", period="2023")
    ev_2023 = _evidence("Revenue for 2023 was 100.", chunk_id="c1")
    ev_2024 = _evidence("Revenue for 2024 was 125.", chunk_id="c2")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev_2023, ev_2024),
        policy=_calc_policy(),
    )
    codes = [i.code for i in issues]
    assert CODE_NUMERIC_UNGROUND in codes


def test_value_in_evidence_but_wrong_metric():
    """Value 125 exists for assets, but the claim binds it to revenue -> ungrounded."""
    claim = _claim(value=Decimal("125"), metric="revenue", period="2024")
    ev_assets = _evidence("Total assets for 2024 were 125.", chunk_id="c1")
    ev_revenue = _evidence("Revenue for 2024 was 100.", chunk_id="c2")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev_assets, ev_revenue),
        policy=_calc_policy(),
    )
    codes = [i.code for i in issues]
    assert CODE_NUMERIC_UNGROUND in codes


def test_value_in_other_chunk_not_cited_chunk():
    """The value exists in one chunk, but the chunk with the matching
    metric+period does not contain the value -> ungrounded.

    This guards the same-chunk invariant: value, metric, and period must
    co-occur on a single evidence item.
    """
    claim = _claim(value=Decimal("125"), metric="revenue", period="2024")
    # Has the value 125, but neither the revenue metric nor the 2024 period.
    ev_value_only = _evidence("The discount rate was 125 basis points.", chunk_id="c1")
    # Has revenue + 2024, but the value is 100, not 125.
    ev_metric_period = _evidence("Revenue for 2024 was 100 million.", chunk_id="c2")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev_value_only, ev_metric_period),
        policy=_calc_policy(),
    )
    codes = [i.code for i in issues]
    assert CODE_NUMERIC_UNGROUND in codes


def test_correct_value_metric_period_passes():
    """Value, metric, and period all match on the same chunk -> no issue."""
    claim = _claim(value=Decimal("125"), metric="revenue", period="2024")
    ev = _evidence("Revenue for 2024 was 125 million.", chunk_id="c1")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev,),
        policy=_calc_policy(),
    )
    assert issues == ()


def test_metric_aliases_recognized():
    """Multiple keywords map to the canonical 'revenue' metric and are recognized."""
    aliases = NumericClaimValidator._metric_aliases("revenue")
    # revenue / total revenue / net revenue / sales / net sales all map to
    # the canonical 'revenue' metric.
    assert "revenue" in aliases
    assert "total revenue" in aliases
    assert "net revenue" in aliases
    assert "sales" in aliases
    assert "net sales" in aliases
    # All aliases share the same canonical metric.
    from src.validation.claim_extractor import _METRIC_CANONICAL

    for alias in aliases:
        assert _METRIC_CANONICAL[alias] == "revenue"

    # End-to-end: an evidence chunk using the 'total revenue' alias supports
    # a claim whose metric is the canonical 'revenue'.
    claim = _claim(value=Decimal("125"), metric="revenue", period="2024")
    ev = _evidence("Total revenue for 2024 was 125.", chunk_id="c1")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev,),
        policy=_calc_policy(),
    )
    assert issues == ()


def test_value_representations_include_comma_format():
    """The value 1250000 is represented as '1,250,000' and matched in evidence."""
    claim = _claim(
        value=Decimal("1250000"),
        metric="revenue",
        period="2024",
        raw_text="1250000",
    )
    reps = NumericClaimValidator._value_representations(claim)
    assert "1,250,000" in reps
    assert "1250000" in reps

    # End-to-end: evidence using the comma-formatted value is recognized.
    ev = _evidence("Revenue for 2024 was $1,250,000.", chunk_id="c1")
    issues = NumericClaimValidator().validate(
        claims=(claim,),
        evidence=(ev,),
        policy=_calc_policy(),
    )
    assert issues == ()
