#!/usr/bin/env python3
"""验证发布声明 (Validate release claims).

读取 claim-evidence-map.json，检查每个 verified claim 的 evidence artifact 存在，
检查 unverified claim 不出现在 allowed_surfaces 中，检查 prohibited claims 不存在。
输出到 artifacts/release/phase6/claim-validation.json。

退出码: 0=pass, 1=fail
"""

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.path.expanduser("~/.cache/nanochat"))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"
DOCS_RELEASE_DIR = REPO_ROOT / "docs" / "release"

SCHEMA_VERSION = "1.0"

# Path to the claim-evidence map
CLAIM_MAP_PATH = REPO_ROOT / "artifacts" / "release" / "phase6" / "claim-evidence-map.json"

# Evidence status → maximum claim status allowed.
# Only verified_local evidence can support a "verified" claim.
# placeholder evidence is forbidden entirely.
EVIDENCE_STATUS_TO_MAX_CLAIM = {
    "verified_local": "verified",
    "external_attestation": "partially_verified",
    "historical_self_reported": "unverified",
    "historical_unavailable": "unverified",
    "fallback": "unverified",
    "placeholder": None,  # forbidden as evidence
}

# Claim status rank for comparison (higher = stronger)
_CLAIM_STATUS_RANK = {
    "verified": 3,
    "partially_verified": 2,
    "unverified": 1,
    "prohibited": 0,
}


def load_json_safe(path: Path):
    """Load JSON file, return None on failure."""
    if not path.exists() or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def file_exists(path: Path) -> bool:
    """Check if a file exists."""
    return path.exists() and path.is_file()


def check_verified_claims(claim_map: dict) -> list:
    """Check that each verified claim has evidence artifacts that exist."""
    issues = []
    verified = claim_map.get("verified_claims", [])
    if not isinstance(verified, list):
        return [{"issue": "verified_claims is not a list", "severity": "error"}]

    for i, claim_entry in enumerate(verified):
        if not isinstance(claim_entry, dict):
            issues.append({
                "claim_index": i,
                "issue": "claim entry is not a dict",
                "severity": "error",
            })
            continue

        claim_text = claim_entry.get("claim", f"<unknown_claim_{i}>")
        evidence = claim_entry.get("evidence")
        evidence_artifacts = claim_entry.get("evidence_artifacts", [])

        # If evidence is a string path, check it exists
        if isinstance(evidence, str):
            evidence_path = REPO_ROOT / evidence
            if not file_exists(evidence_path):
                issues.append({
                    "claim": claim_text,
                    "issue": f"evidence file not found: {evidence}",
                    "severity": "error",
                })

        # Check evidence_artifacts list
        if isinstance(evidence_artifacts, list):
            for art_path in evidence_artifacts:
                if isinstance(art_path, str):
                    full_path = REPO_ROOT / art_path
                    if not file_exists(full_path):
                        issues.append({
                            "claim": claim_text,
                            "issue": f"evidence artifact not found: {art_path}",
                            "severity": "error",
                        })

    return issues


def check_evidence_status_gate(claim_map: dict) -> list:
    """Enforce evidence quality gate: only verified_local evidence can support
    a 'verified' claim. Claims with weaker evidence must be downgraded.

    Returns a list of issues. An issue with severity 'error' indicates a claim
    whose status exceeds what its evidence allows.
    """
    issues = []
    claims = claim_map.get("claims", [])
    if not isinstance(claims, list):
        return [{"issue": "claims is not a list", "severity": "error"}]

    for i, claim_entry in enumerate(claims):
        if not isinstance(claim_entry, dict):
            continue

        claim_status = claim_entry.get("status")
        if claim_status is None:
            continue

        claim_text = claim_entry.get("claim", f"<unknown_claim_{i}>")
        cid = claim_entry.get("id", f"claim_{i}")
        evidence_list = claim_entry.get("evidence", [])

        if not isinstance(evidence_list, list):
            continue

        # If claim has no evidence at all, skip (handled by other checks)
        if not evidence_list:
            continue

        # Determine the strongest evidence status for this claim
        strongest_max = None
        has_placeholder = False
        for ev in evidence_list:
            if not isinstance(ev, dict):
                continue
            ev_status = ev.get("verification_status")
            if ev_status == "placeholder":
                has_placeholder = True
                continue
            if ev_status is None:
                # If no verification_status field, treat as unknown — cannot
                # support verified status
                continue
            max_claim = EVIDENCE_STATUS_TO_MAX_CLAIM.get(ev_status)
            if max_claim is None:
                continue
            if strongest_max is None:
                strongest_max = max_claim
            else:
                if _CLAIM_STATUS_RANK.get(max_claim, 0) > _CLAIM_STATUS_RANK.get(strongest_max, 0):
                    strongest_max = max_claim

        # placeholder evidence is forbidden
        if has_placeholder:
            issues.append({
                "claim": claim_text,
                "claim_id": cid,
                "issue": "placeholder evidence is forbidden",
                "severity": "error",
            })
            continue

        if strongest_max is None:
            # No recognized evidence status at all
            if claim_status == "verified":
                issues.append({
                    "claim": claim_text,
                    "claim_id": cid,
                    "issue": "verified claim has no evidence with recognized verification_status",
                    "severity": "error",
                })
            continue

        # Check if claim status exceeds what evidence allows
        claim_rank = _CLAIM_STATUS_RANK.get(claim_status, 0)
        max_rank = _CLAIM_STATUS_RANK.get(strongest_max, 0)
        if claim_rank > max_rank:
            issues.append({
                "claim": claim_text,
                "claim_id": cid,
                "issue": f"claim status '{claim_status}' exceeds evidence max '{strongest_max}'",
                "severity": "error",
            })

    return issues


def check_unverified_claims(claim_map: dict) -> list:
    """Check that unverified claims do not appear in allowed_surfaces."""
    issues = []
    unverified = claim_map.get("unverified_claims", [])
    if not isinstance(unverified, list):
        return [{"issue": "unverified_claims is not a list", "severity": "warning"}]

    # Collect all allowed surfaces text to search
    allowed_surfaces = claim_map.get("allowed_surfaces", [])
    surface_texts = {}
    for surface in allowed_surfaces:
        if isinstance(surface, dict):
            surface_name = surface.get("name", "unknown")
            surface_path = surface.get("path")
            if surface_path:
                full_path = REPO_ROOT / surface_path
                if file_exists(full_path):
                    try:
                        surface_texts[surface_name] = full_path.read_text(
                            encoding="utf-8"
                        ).lower()
                    except OSError:
                        surface_texts[surface_name] = None

    for i, claim_entry in enumerate(unverified):
        if not isinstance(claim_entry, dict):
            issues.append({
                "claim_index": i,
                "issue": "unverified claim entry is not a dict",
                "severity": "warning",
            })
            continue

        claim_text = claim_entry.get("claim", "")
        if not claim_text:
            continue

        # Check if unverified claim text appears in any allowed surface
        claim_lower = claim_text.lower()
        for surface_name, text in surface_texts.items():
            if text and claim_lower in text:
                issues.append({
                    "claim": claim_text,
                    "issue": f"unverified claim found in allowed surface: {surface_name}",
                    "severity": "error",
                })

    return issues


def check_prohibited_claims(claim_map: dict) -> list:
    """Check that prohibited claims do not exist in any release docs."""
    issues = []
    prohibited = claim_map.get("prohibited_claims", [])
    if not isinstance(prohibited, list):
        return [{"issue": "prohibited_claims is not a list", "severity": "warning"}]

    # Files to exclude from scan (self-referential files that define/list prohibited claims,
    # or aggregator manifests that embed sub-manifest data containing prohibited claim references)
    EXCLUDE_FILES = {
        "claim-evidence-map.json",
        "claim-validation.json",
        "release-manifest.json",
        "evaluation-evidence.json",
    }

    # Scan all release docs
    doc_texts = {}
    if DOCS_RELEASE_DIR.exists():
        for fpath in sorted(DOCS_RELEASE_DIR.rglob("*")):
            if fpath.is_file() and fpath.suffix in (".md", ".json", ".txt", ".html"):
                if fpath.name in EXCLUDE_FILES:
                    continue
                try:
                    doc_texts[str(fpath.relative_to(REPO_ROOT))] = (
                        fpath.read_text(encoding="utf-8").lower()
                    )
                except (OSError, UnicodeDecodeError):
                    pass

    # Also scan release artifacts (excluding self-referential files)
    if OUTPUT_DIR.exists():
        for fpath in sorted(OUTPUT_DIR.rglob("*.json")):
            if fpath.name in EXCLUDE_FILES:
                continue
            try:
                doc_texts[str(fpath.relative_to(REPO_ROOT))] = (
                    fpath.read_text(encoding="utf-8").lower()
                )
            except (OSError, UnicodeDecodeError):
                pass

    for i, claim_entry in enumerate(prohibited):
        if isinstance(claim_entry, dict):
            claim_text = claim_entry.get("claim", "")
        elif isinstance(claim_entry, str):
            claim_text = claim_entry
        else:
            continue

        if not claim_text:
            continue

        claim_lower = claim_text.lower()
        for doc_path, text in doc_texts.items():
            if text and claim_lower in text:
                issues.append({
                    "claim": claim_text,
                    "doc_path": doc_path,
                    "issue": "prohibited claim found in release document",
                    "severity": "error",
                })

    return issues


def validate() -> dict:
    """Run all validation checks."""
    claim_map = load_json_safe(CLAIM_MAP_PATH)

    if claim_map is None:
        return {
            "errors": [
                {
                    "issue": f"claim-evidence-map.json not found at {CLAIM_MAP_PATH.relative_to(REPO_ROOT)}",
                    "severity": "error",
                }
            ],
            "passed": False,
            "schema_version": SCHEMA_VERSION,
        }

    verified_issues = check_verified_claims(claim_map)
    unverified_issues = check_unverified_claims(claim_map)
    prohibited_issues = check_prohibited_claims(claim_map)
    evidence_gate_issues = check_evidence_status_gate(claim_map)

    all_issues = verified_issues + unverified_issues + prohibited_issues + evidence_gate_issues
    has_errors = any(
        issue.get("severity") == "error" for issue in all_issues
    )

    return {
        "checks": {
            "evidence_status_gate": {
                "issues": evidence_gate_issues,
                "passed": not any(
                    i.get("severity") == "error" for i in evidence_gate_issues
                ),
            },
            "prohibited_claims": {
                "issues": prohibited_issues,
                "passed": not any(
                    i.get("severity") == "error" for i in prohibited_issues
                ),
            },
            "unverified_claims": {
                "issues": unverified_issues,
                "passed": not any(
                    i.get("severity") == "error" for i in unverified_issues
                ),
            },
            "verified_claims": {
                "issues": verified_issues,
                "passed": not any(
                    i.get("severity") == "error" for i in verified_issues
                ),
            },
        },
        "claim_map_path": "artifacts/release/phase6/claim-evidence-map.json",
        "errors": all_issues,
        "passed": not has_errors,
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "prohibited_claims_count": len(
                claim_map.get("prohibited_claims", []) if claim_map else []
            ),
            "total_issues": len(all_issues),
            "unverified_claims_count": len(
                claim_map.get("unverified_claims", []) if claim_map else []
            ),
            "verified_claims_count": len(
                claim_map.get("verified_claims", []) if claim_map else []
            ),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = validate()
    output_path = OUTPUT_DIR / "claim-validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")

    if result["passed"]:
        print("PASS: All claim validations passed.")
        sys.exit(0)
    else:
        print("FAIL: Claim validation failed. See issues in claim-validation.json.")
        sys.exit(1)


if __name__ == "__main__":
    main()
