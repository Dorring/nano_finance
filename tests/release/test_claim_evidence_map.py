"""Tests for claim-evidence-map.json.

Verify Phase 6 claim evidence mapping: each claim has status field,
status values are valid, verified claims have evidence array,
prohibited claims not in allowed_surfaces, 0/54 not quality metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

VALID_STATUSES = {"verified", "partially_verified", "unverified", "prohibited"}


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _get_claims(manifest) -> list:
    claims = manifest.get("claims")
    if claims is None:
        claims = manifest.get("claim_evidence_map")
    if claims is None:
        if isinstance(manifest, dict):
            vals = list(manifest.values())
            if all(isinstance(v, dict) for v in vals):
                claims = vals
    if isinstance(claims, dict):
        claims = list(claims.values())
    return claims if isinstance(claims, list) else []


@pytest.fixture
def claim_evidence_map() -> dict:
    return _load_json(ARTIFACTS_DIR / "claim-evidence-map.json")


def test_each_claim_has_status(claim_evidence_map):
    claims = _get_claims(claim_evidence_map)
    assert len(claims) > 0
    for i, claim in enumerate(claims):
        assert isinstance(claim, dict), f"Claim {i} is not a dict"
        assert "status" in claim, f"Claim {i} missing status field"


def test_status_values_are_valid(claim_evidence_map):
    claims = _get_claims(claim_evidence_map)
    assert len(claims) > 0
    for i, claim in enumerate(claims):
        status = claim.get("status")
        assert status in VALID_STATUSES, f"Claim {i} invalid status '{status}'"


def test_verified_claims_have_evidence_array(claim_evidence_map):
    claims = _get_claims(claim_evidence_map)
    verified = [c for c in claims if isinstance(c, dict) and c.get("status") == "verified"]
    if not verified:
        pytest.skip("No verified claims found")
    for i, claim in enumerate(verified):
        evidence = claim.get("evidence")
        cid = claim.get("id", "?")
        assert evidence is not None, f"Verified claim {i} (id={cid}) missing evidence"
        assert isinstance(evidence, list), f"Verified claim {i} evidence should be list"
        assert len(evidence) > 0, f"Verified claim {i} (id={cid}) empty evidence"


def test_prohibited_claims_not_in_allowed_surfaces(claim_evidence_map):
    claims = _get_claims(claim_evidence_map)
    prohibited = [c for c in claims if isinstance(c, dict) and c.get("status") == "prohibited"]
    if not prohibited:
        pytest.skip("No prohibited claims found")
    for i, claim in enumerate(prohibited):
        allowed = claim.get("allowed_surfaces", [])
        cid = claim.get("id", "?")
        assert isinstance(allowed, list)
        assert len(allowed) == 0, f"Prohibited claim {i} (id={cid}) has allowed_surfaces: {allowed}"


def test_zero_54_not_marked_as_quality_metric(claim_evidence_map):
    claims = _get_claims(claim_evidence_map)
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_text = json.dumps(claim)
        if "54" in claim_text or "synthetic" in claim_text.lower() or "held_out" in claim_text.lower():
            is_quality = claim.get("is_quality_metric") or claim.get("quality_metric")
            if is_quality is not None:
                cid = claim.get("id", "?")
                assert not is_quality, f"0/54 should not be quality metric (id={cid})"
            tags = claim.get("tags", []) or claim.get("categories", [])
            if tags:
                cid = claim.get("id", "?")
                assert "quality_metric" not in tags, f"0/54 not quality_metric tag (id={cid})"
