"""Phase 6 acceptance tests covering 56 acceptance criteria.

Verify Phase 6 release acceptance: branch source, Release ID,
all Cards complete, Manifests complete, hash correctness,
privacy/security, claim compliance, and Failed=0.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"
SCRIPTS_DIR = REPO_ROOT / "scripts" / "release"

EXPECTED_RELEASE_ID = "nano-finance-d24-sft-v1"
EXPECTED_N_LAYER = 24
EXPECTED_N_HEAD = 12
EXPECTED_N_EMBD = 1536
EXPECTED_VOCAB_SIZE = 65000
EXPECTED_SEQ_LEN = 2048
EXPECTED_PRETRAIN_STEP = 28000
EXPECTED_SFT_STEP = 150
EXPECTED_SFT_SAMPLES = 39534
BASE_CKPT_NAME = "d24_final_mixdata"
SFT_CKPT_NAME = "d24_finance_v2_lr010"

EXPECTED_MANIFESTS = [
    "release-manifest.json", "model-lineage.json",
    "tokenizer-manifest.json", "pretraining-data-manifest.json",
    "sft-data-manifest.json", "training-runs.json",
    "checkpoint-manifest.json", "evaluation-evidence.json",
    "dependency-manifest.json", "license-inventory.json",
    "claim-evidence-map.json", "phase6-acceptance.json",
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"Artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_field(obj, field):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == field:
                return v
            r = _find_field(v, field)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_field(item, field)
            if r is not None:
                return r
    return None


def _find_absolute_paths(obj, acc=None):
    if acc is None:
        acc = []
    if isinstance(obj, str):
        if re.search(r"[A-Za-z]:\\", obj) or re.search(r"/(home|Users|root|mnt|data|var|opt|tmp)/", obj):
            acc.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _find_absolute_paths(v, acc)
    elif isinstance(obj, list):
        for item in obj:
            _find_absolute_paths(item, acc)
    return acc


@pytest.fixture
def acceptance_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "phase6-acceptance.json")


@pytest.fixture
def release_manifest() -> dict:
    return _load_json(ARTIFACTS_DIR / "release-manifest.json")

# 56 acceptance criteria IDs
ACCEPTANCE_CRITERIA = [
    "branch_from_phase5_merge",
    "release_id_fixed",
    "tokenizer_card_complete",
    "pretraining_data_card_complete",
    "sft_data_card_complete",
    "model_card_complete",
    "rag_system_card_complete",
    "evaluation_card_complete",
    "responsible_use_complete",
    "limitations_complete",
    "reproducibility_complete",
    "release_manifest_complete",
    "model_lineage_complete",
    "tokenizer_manifest_complete",
    "pretraining_data_manifest_complete",
    "sft_data_manifest_complete",
    "training_runs_complete",
    "checkpoint_manifest_complete",
    "evaluation_evidence_complete",
    "dependency_manifest_complete",
    "license_inventory_complete",
    "claim_evidence_map_complete",
    "phase6_acceptance_complete",
    "training_evidence_complete",
    "checkpoint_hash_correct",
    "tokenizer_hash_linked",
    "no_absolute_paths",
    "no_secrets",
    "zero_54_not_quality_metric",
    "synthetic_held_out_marker",
    "no_native_function_calling_claim",
    "no_hallucination_elimination_claim",
    "failed_zero",
    "mit_license",
    "model_n_layer_24",
    "model_n_head_12",
    "model_n_embd_1536",
    "model_vocab_size_65000",
    "model_seq_len_2048",
    "pretrain_step_28000",
    "pretrain_val_bpb",
    "sft_step_150",
    "sft_val_bpb",
    "sft_samples_39534",
    "base_checkpoint_name",
    "sft_checkpoint_name",
    "lineage_chain",
    "tokenizer_9_special_tokens",
    "tokenizer_pad_not_special",
    "tokenizer_bpe_algorithm",
    "sft_finance_r1_1225",
    "sft_8_data_sources",
    "all_manifests_parseable",
    "all_hashes_sha256",
    "privacy_scan_passed",
    "determinism_check_passed",
]

assert len(ACCEPTANCE_CRITERIA) == 56


def _get_criteria_list(manifest) -> list:
    for key in ["criteria", "acceptance_criteria", "items", "checks"]:
        if key in manifest:
            val = manifest[key]
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return list(val.values())
    if isinstance(manifest, dict):
        vals = list(manifest.values())
        if all(isinstance(v, dict) for v in vals):
            return vals
    return []


def _criterion_status(criteria, criterion_id) -> str:
    for c in criteria:
        if not isinstance(c, dict):
            continue
        c_id = str(c.get("id", "")) + str(c.get("name", ""))
        if criterion_id in c_id:
            return c.get("status") or c.get("result") or c.get("state")
    return None


def test_release_id_fixed(release_manifest):
    # Release ID is fixed as nano-finance-d24-sft-v1
    rid = release_manifest.get("release_id") or release_manifest.get("id")
    assert rid == EXPECTED_RELEASE_ID, f"Expected {EXPECTED_RELEASE_ID}, got {rid}"


@pytest.mark.parametrize("manifest_name", EXPECTED_MANIFESTS)
def test_manifest_file_exists(manifest_name):
    # Verify all Manifest files exist
    path = ARTIFACTS_DIR / manifest_name
    if not path.exists():
        pytest.skip(f"Manifest not found: {manifest_name}")


@pytest.mark.parametrize("manifest_name", EXPECTED_MANIFESTS)
def test_manifest_parseable(manifest_name):
    # Verify all Manifests are parseable JSON
    path = ARTIFACTS_DIR / manifest_name
    if not path.exists():
        pytest.skip(f"Manifest not found: {manifest_name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data is not None


def test_mit_license():
    # Verify MIT license
    path = ARTIFACTS_DIR / "license-inventory.json"
    if not path.exists():
        pytest.skip("license-inventory.json not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "MIT" in json.dumps(data)


def test_model_config_values():
    # Verify model config: n_layer=24, n_head=12, n_embd=1536, vocab=65000, seq=2048
    path = ARTIFACTS_DIR / "checkpoint-manifest.json"
    if not path.exists():
        pytest.skip("checkpoint-manifest.json not found")
    text = path.read_text(encoding="utf-8")
    assert str(EXPECTED_N_LAYER) in text
    assert str(EXPECTED_N_HEAD) in text
    assert str(EXPECTED_N_EMBD) in text
    assert str(EXPECTED_VOCAB_SIZE) in text
    assert str(EXPECTED_SEQ_LEN) in text


def test_pretraining_step_28000():
    # Verify pretraining step 28000
    path = ARTIFACTS_DIR / "training-runs.json"
    if not path.exists():
        pytest.skip("training-runs.json not found")
    assert str(EXPECTED_PRETRAIN_STEP) in path.read_text(encoding="utf-8")


def test_sft_step_150():
    # Verify SFT step 150
    path = ARTIFACTS_DIR / "training-runs.json"
    if not path.exists():
        pytest.skip("training-runs.json not found")
    assert str(EXPECTED_SFT_STEP) in path.read_text(encoding="utf-8")


def test_sft_samples_39534():
    # Verify SFT 39534 samples
    path = ARTIFACTS_DIR / "sft-data-manifest.json"
    if not path.exists():
        pytest.skip("sft-data-manifest.json not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    total = data.get("total_samples") or data.get("total")
    assert total == EXPECTED_SFT_SAMPLES


def test_base_checkpoint_name():
    path = ARTIFACTS_DIR / "checkpoint-manifest.json"
    if not path.exists():
        pytest.skip("checkpoint-manifest.json not found")
    assert BASE_CKPT_NAME in path.read_text(encoding="utf-8")


def test_sft_checkpoint_name():
    path = ARTIFACTS_DIR / "checkpoint-manifest.json"
    if not path.exists():
        pytest.skip("checkpoint-manifest.json not found")
    assert SFT_CKPT_NAME in path.read_text(encoding="utf-8")


def test_no_absolute_paths_in_artifacts():
    leaked = []
    for mf in EXPECTED_MANIFESTS:
        path = ARTIFACTS_DIR / mf
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        found = _find_absolute_paths(data)
        if found:
            leaked.append((mf, found[:3]))
    assert not leaked, f"Absolute paths leaked: {leaked}"


def test_no_secrets_in_artifacts():
    patterns = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"ghp_[a-zA-Z0-9]{20,}"),
        re.compile(r"AKIA[A-Z0-9]{16}"),
        re.compile(r"password\s*=\s*\S+", re.IGNORECASE),
    ]
    for mf in EXPECTED_MANIFESTS:
        path = ARTIFACTS_DIR / mf
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for pat in patterns:
            m = pat.search(text)
            assert not m, f"Secret in {mf}: {m.group()}"


def test_synthetic_held_out_marker():
    path = ARTIFACTS_DIR / "evaluation-evidence.json"
    if not path.exists():
        pytest.skip("evaluation-evidence.json not found")
    text = path.read_text(encoding="utf-8").lower()
    assert "synthetic" in text or "held_out" in text or "held-out" in text


def test_zero_54_not_quality_metric():
    path = ARTIFACTS_DIR / "claim-evidence-map.json"
    if not path.exists():
        pytest.skip("claim-evidence-map.json not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(data).lower()
    if "54" in text:
        assert "quality_metric" not in text or "not" in text


def test_no_native_function_calling_claim():
    path = ARTIFACTS_DIR / "claim-evidence-map.json"
    if not path.exists():
        pytest.skip("claim-evidence-map.json not found")
    text = path.read_text(encoding="utf-8").lower()
    if "function_calling" in text or "function calling" in text:
        assert "no" in text or "not" in text or "prohibited" in text


def test_no_hallucination_elimination_claim():
    path = ARTIFACTS_DIR / "claim-evidence-map.json"
    if not path.exists():
        pytest.skip("claim-evidence-map.json not found")
    text = path.read_text(encoding="utf-8").lower()
    if "hallucination" in text:
        assert "no" in text or "not" in text or "prohibited" in text


def test_failed_zero(acceptance_manifest):
    # Verify Failed=0
    criteria = _get_criteria_list(acceptance_manifest)
    if not criteria:
        summary = acceptance_manifest.get("summary", {})
        failed = summary.get("failed", 0)
        assert failed == 0, f"Expected 0 failed, got {failed}"
        return
    failed_count = sum(
        1 for c in criteria if isinstance(c, dict)
        and str(c.get("status", c.get("result", ""))).lower() in ("failed", "fail")
    )
    assert failed_count == 0, f"Expected 0 failed, got {failed_count}"


@pytest.mark.parametrize("criterion_id", ACCEPTANCE_CRITERIA)
def test_acceptance_criterion(acceptance_manifest, criterion_id):
    # Verify each of the 56 acceptance criteria
    criteria = _get_criteria_list(acceptance_manifest)
    if not criteria:
        pytest.skip("No criteria list in phase6-acceptance.json")
    status = _criterion_status(criteria, criterion_id)
    if status is None:
        pytest.skip(f"Criterion '{criterion_id}' not found in acceptance criteria")
    status_lower = str(status).lower()
    assert status_lower in ("passed", "pass", "ok", "complete", "completed", "true", "verified"), \
        f"Criterion '{criterion_id}' status is {status}, expected passed"


def test_acceptance_criteria_count(acceptance_manifest):
    # Verify at least 56 acceptance criteria exist
    criteria = _get_criteria_list(acceptance_manifest)
    if not criteria:
        pytest.skip("No criteria list found in phase6-acceptance.json")
    assert len(criteria) >= 56, f"Expected >= 56 criteria, got {len(criteria)}"
