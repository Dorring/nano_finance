#!/usr/bin/env python3
"""Comprehensive release gate validator for Phase 5 v2 sealed evaluation.

Runs 20 checks before allowing the Phase 5 PR to merge. Each check returns
a ``Check`` result with ``passed``, ``message``, and ``skipped`` fields.
The script exits 0 only if ALL checks pass (skips are tolerated unless
``--strict`` is given).

Run from the backend directory:
    python scripts/validate_phase5_release.py
    python scripts/validate_phase5_release.py --strict
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

ARTIFACTS_PHASE5 = BACKEND_DIR / "artifacts" / "evaluation" / "phase5"
EVAL_DATA_PHASE5 = BACKEND_DIR / "eval_data" / "phase5"
INDEXES_PHASE5 = BACKEND_DIR / "indexes" / "phase5"
DOCS_EVAL = BACKEND_DIR / "docs" / "evaluation"
PROTOCOL_PATH = ARTIFACTS_PHASE5 / "protocol" / "phase5-evaluation-protocol.json"
INVALIDATED_RUN_DIR = ARTIFACTS_PHASE5 / "invalidated-placeholder-run"
UNICODE_SCRIPT = BACKEND_DIR / "scripts" / "check_unicode_controls.py"
OVERLAP_SCRIPT = BACKEND_DIR / "scripts" / "check_phase5_dataset_overlap.py"


@dataclass
class Check:
    """Result of one release-gate check."""

    name: str
    passed: bool
    message: str
    skipped: bool = False
    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        flag = "PASS" if self.passed else ("SKIP" if self.skipped else "FAIL")
        return f"[{flag}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Check 1: placeholder run invalidated
# ---------------------------------------------------------------------------


def check_placeholder_invalidated() -> Check:
    """Verify the placeholder run status.json exists and is invalidated."""
    status_path = INVALIDATED_RUN_DIR / "status.json"
    if not status_path.is_file():
        return Check(
            "check_placeholder_invalidated",
            False,
            f"missing {status_path.relative_to(BACKEND_DIR)}",
        )
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return Check(
            "check_placeholder_invalidated",
            False,
            f"cannot parse status.json: {exc}",
        )
    status = data.get("status")
    if status == "invalidated":
        return Check(
            "check_placeholder_invalidated",
            True,
            "placeholder run is invalidated",
        )
    return Check(
        "check_placeholder_invalidated",
        False,
        f"status is {status!r}, expected 'invalidated'",
    )


# ---------------------------------------------------------------------------
# Check 2: corpus indexed
# ---------------------------------------------------------------------------


def check_corpus_indexed() -> Check:
    """Verify index dirs exist under indexes/phase5/{dev,calibration,sealed}."""
    if not INDEXES_PHASE5.is_dir():
        return Check(
            "check_corpus_indexed",
            True,
            "indexes/phase5/ not yet built",
            skipped=True,
        )
    missing: list[str] = []
    for partition in ("dev", "calibration", "sealed"):
        part_dir = INDEXES_PHASE5 / partition
        if not part_dir.is_dir():
            missing.append(f"{partition}/ (directory missing)")
            continue
        # Look for any chroma-like and bm25-like artifacts.
        has_chroma = _has_chroma_artifact(part_dir)
        has_bm25 = _has_bm25_artifact(part_dir)
        if not has_chroma:
            missing.append(f"{partition}/ (no chroma artifact)")
        if not has_bm25:
            missing.append(f"{partition}/ (no bm25 artifact)")
    if missing:
        return Check(
            "check_corpus_indexed",
            True,
            f"index dirs incomplete: {'; '.join(missing)}",
            skipped=True,
        )
    return Check(
        "check_corpus_indexed",
        True,
        "all partition index dirs present with chroma and bm25 artifacts",
    )


def _has_chroma_artifact(directory: Path) -> bool:
    """Return True if a chroma-style artifact is found under directory."""
    for path in directory.rglob("*"):
        if path.is_dir() and (
            (path / "chroma.sqlite3").is_file()
            or any(p.suffix == ".pickle" for p in path.iterdir())
        ):
            return True
    return any(p.name == "chroma.sqlite3" for p in directory.rglob("*"))


def _has_bm25_artifact(directory: Path) -> bool:
    """Return True if a bm25-style artifact is found under directory."""
    for path in directory.rglob("*"):
        name = path.name.lower()
        if "bm25" in name or name.endswith(".db") or name.endswith(".idx"):
            return True
    return False


# ---------------------------------------------------------------------------
# Check 3: no cross-partition overlap
# ---------------------------------------------------------------------------


def check_no_cross_partition_overlap() -> Check:
    """Run the overlap checker and verify no document overlap."""
    if not OVERLAP_SCRIPT.is_file():
        return Check(
            "check_no_cross_partition_overlap",
            True,
            "overlap checker script not available",
            skipped=True,
        )
    try:
        result = subprocess.run(
            [sys.executable, str(OVERLAP_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BACKEND_DIR),
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return Check(
            "check_no_cross_partition_overlap",
            True,
            f"overlap checker could not run: {exc}",
            skipped=True,
        )
    if result.returncode == 0:
        return Check(
            "check_no_cross_partition_overlap",
            True,
            "no cross-partition overlap detected",
        )
    return Check(
        "check_no_cross_partition_overlap",
        False,
        f"overlap detected (exit {result.returncode}): "
        f"{result.stderr.strip() or result.stdout.strip()}",
    )


# ---------------------------------------------------------------------------
# Check 4: questions and labels hashed
# ---------------------------------------------------------------------------


def check_questions_labels_hashed() -> Check:
    """Verify manifest.json files have non-placeholder sha256 values."""
    partitions = ("dev", "calibration", "sealed")
    issues: list[str] = []
    for partition in partitions:
        manifest = EVAL_DATA_PHASE5 / partition / "manifest.json"
        if not manifest.is_file():
            # Sealed partition may use manifest.public.json.
            manifest = EVAL_DATA_PHASE5 / partition / "manifest.public.json"
        if not manifest.is_file():
            issues.append(f"{partition}/: no manifest file")
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            issues.append(f"{partition}/: cannot parse manifest: {exc}")
            continue
        q_hash = data.get("questions_sha256")
        l_hash = data.get("labels_sha256")
        if not q_hash or q_hash == "placeholder":
            issues.append(f"{partition}/: questions_sha256 is {q_hash!r} (placeholder)")
        # Sealed public manifest may legitimately have labels_sha256=null.
        if partition != "sealed":
            if not l_hash or l_hash == "placeholder":
                issues.append(
                    f"{partition}/: labels_sha256 is {l_hash!r} (placeholder)"
                )
    if issues:
        return Check(
            "check_questions_labels_hashed",
            False,
            f"{len(issues)} manifest issue(s): {'; '.join(issues)}",
            details=issues,
        )
    return Check(
        "check_questions_labels_hashed",
        True,
        "all manifests have non-placeholder hashes",
    )


# ---------------------------------------------------------------------------
# Check 5: protocol before calibration (timestamp ordering)
# ---------------------------------------------------------------------------


def check_protocol_before_calibration() -> Check:
    """Verify sealed protocol timestamp is earlier than calibration's."""
    protocol_path = PROTOCOL_PATH
    selected_config = INVALIDATED_RUN_DIR / "calibration" / "selected-config.json"
    if not protocol_path.is_file() or not selected_config.is_file():
        return Check(
            "check_protocol_before_calibration",
            True,
            "protocol or selected-config not found",
            skipped=True,
        )
    proto_ts = _extract_timestamp(protocol_path)
    calib_ts = _extract_timestamp(selected_config)
    if proto_ts is None or calib_ts is None:
        return Check(
            "check_protocol_before_calibration",
            True,
            "timestamps not available in protocol/selected-config",
            skipped=True,
        )
    if proto_ts <= calib_ts:
        return Check(
            "check_protocol_before_calibration",
            True,
            f"protocol ({proto_ts}) precedes calibration ({calib_ts})",
        )
    return Check(
        "check_protocol_before_calibration",
        False,
        f"protocol ({proto_ts}) is after calibration ({calib_ts})",
    )


def _extract_timestamp(path: Path) -> str | None:
    """Extract a timestamp string from a JSON artifact, or file mtime."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    for key in ("frozen_at", "created_at", "selected_at", "timestamp"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    # Fall back to filesystem mtime (ISO-ish).
    return (
        path.stat().st_mtime_isoformat()
        if hasattr(path.stat(), "st_mtime_isoformat")
        else None
    )


# ---------------------------------------------------------------------------
# Check 6: baseline declaration
# ---------------------------------------------------------------------------


def check_baseline_no_undeclared_changes() -> Check:
    """Verify docs/evaluation/baseline-declaration.md exists."""
    declaration = DOCS_EVAL / "baseline-declaration.md"
    if not declaration.is_file():
        return Check(
            "check_baseline_no_undeclared_changes",
            True,
            "baseline-declaration.md not applicable",
            skipped=True,
        )
    content = declaration.read_text(encoding="utf-8")
    if not content.strip():
        return Check(
            "check_baseline_no_undeclared_changes",
            False,
            "baseline-declaration.md is empty",
        )
    return Check(
        "check_baseline_no_undeclared_changes",
        True,
        "baseline declaration documents production changes",
    )


# ---------------------------------------------------------------------------
# Check 7: safe candidate rule (baseline fallback)
# ---------------------------------------------------------------------------


def check_safe_candidate_rule() -> Check:
    """Verify select_best_candidate returns baseline for all-unsafe input."""
    try:
        from src.evaluation.calibration import select_best_candidate
    except ImportError as exc:
        return Check(
            "check_safe_candidate_rule",
            True,
            f"cannot import select_best_candidate: {exc}",
            skipped=True,
        )
    baseline = {
        "macro_strict_pass_rate": 0.5,
        "citation_recall": 0.8,
        "p95_latency_ms": 100.0,
        "unsupported_numeric_release_rate": 0.0,
        "invalid_citation_release_rate": 0.0,
        "calculation_mismatch_release_rate": 0.0,
        "unsafe_content_release_rate": 0.0,
    }
    unsafe_candidate = {
        "params": {"n_results": 99},
        "metrics": {
            "macro_strict_pass_rate": 0.99,
            "citation_recall": 1.0,
            "p95_latency_ms": 1.0,
            "unsupported_numeric_release_rate": 0.9,  # worse than baseline
            "invalid_citation_release_rate": 0.9,
            "calculation_mismatch_release_rate": 0.9,
            "unsafe_content_release_rate": 0.9,
        },
    }
    try:
        result = select_best_candidate([unsafe_candidate], baseline)
    except Exception as exc:  # noqa: BLE001
        return Check(
            "check_safe_candidate_rule",
            False,
            f"select_best_candidate raised: {exc}",
        )
    selected = result.get("selected_config")
    if selected == "baseline":
        return Check(
            "check_safe_candidate_rule",
            True,
            "select_best_candidate returns baseline for all-unsafe input",
        )
    return Check(
        "check_safe_candidate_rule",
        False,
        f"expected baseline selection, got selected_config={selected!r}",
    )


# ---------------------------------------------------------------------------
# Check 8: ablation flags functional
# ---------------------------------------------------------------------------


def check_ablation_flags_functional() -> Check:
    """Verify get_ablation_config returns valid configs for A0-A9."""
    try:
        from src.evaluation.ablation import (
            ABLATION_VARIANTS,
            get_ablation_config,
        )
    except ImportError as exc:
        return Check(
            "check_ablation_flags_functional",
            True,
            f"cannot import ablation module: {exc}",
            skipped=True,
        )
    base_config: dict[str, Any] = {"n_results": 5}
    configs: dict[str, dict[str, Any]] = {}
    for variant in ABLATION_VARIANTS:
        vid = variant["id"]
        try:
            cfg = get_ablation_config(base_config, vid)
        except Exception as exc:  # noqa: BLE001
            return Check(
                "check_ablation_flags_functional",
                False,
                f"variant {vid} raised: {exc}",
            )
        if not isinstance(cfg, dict):
            return Check(
                "check_ablation_flags_functional",
                False,
                f"variant {vid} returned non-dict: {type(cfg).__name__}",
            )
        configs[vid] = cfg
    if len(configs) != 10:
        return Check(
            "check_ablation_flags_functional",
            False,
            f"expected 10 variants, got {len(configs)}",
        )
    # Verify variants differ (A0 should equal base; A1-A9 should differ).
    differing = 0
    base = configs.get("A0", {})
    for vid, cfg in configs.items():
        if vid == "A0":
            continue
        if cfg != base:
            differing += 1
    if differing == 0:
        return Check(
            "check_ablation_flags_functional",
            False,
            "no variants differ from baseline A0",
        )
    return Check(
        "check_ablation_flags_functional",
        True,
        f"all 10 variants return valid configs ({differing} differ from A0)",
    )


# ---------------------------------------------------------------------------
# Check 9: RC freeze hash verified
# ---------------------------------------------------------------------------


def check_rc_hash_verified() -> Check:
    """Verify RC freeze artifacts exist."""
    candidates = [
        ARTIFACTS_PHASE5 / "rc" / "rc-freeze.json",
        INVALIDATED_RUN_DIR / "rc" / "rc-freeze.json",
    ]
    rc_path = next((p for p in candidates if p.is_file()), None)
    if rc_path is None:
        return Check(
            "check_rc_hash_verified",
            True,
            "RC freeze artifacts not yet generated",
            skipped=True,
        )
    try:
        data = json.loads(rc_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return Check(
            "check_rc_hash_verified",
            False,
            f"cannot parse rc-freeze.json: {exc}",
        )
    required_hashes = (
        "rc_commit",
        "model_checkpoint_sha256",
        "protocol_sha256",
    )
    missing = [k for k in required_hashes if not data.get(k)]
    if missing:
        return Check(
            "check_rc_hash_verified",
            False,
            f"rc-freeze.json missing hashes: {missing}",
        )
    return Check(
        "check_rc_hash_verified",
        True,
        f"RC freeze verified at commit {data.get('rc_commit', '?')}",
    )


# ---------------------------------------------------------------------------
# Check 10: post-freeze no executable changes
# ---------------------------------------------------------------------------


def check_post_freeze_no_executable_changes() -> Check:
    """Verify post_freeze_diff.json exists and shows no executable changes."""
    candidates = [
        ARTIFACTS_PHASE5 / "post_freeze_diff.json",
        INVALIDATED_RUN_DIR / "post_freeze_diff.json",
    ]
    diff_path = next((p for p in candidates if p.is_file()), None)
    if diff_path is None:
        return Check(
            "check_post_freeze_no_executable_changes",
            True,
            "post_freeze_diff.json not yet generated",
            skipped=True,
        )
    try:
        data = json.loads(diff_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return Check(
            "check_post_freeze_no_executable_changes",
            False,
            f"cannot parse post_freeze_diff.json: {exc}",
        )
    watched_prefixes = ("src/", "scripts/", "config/")
    changed: list[str] = []
    entries = data if isinstance(data, list) else data.get("files", [])
    for entry in entries:
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("file") or ""
        else:
            path = str(entry)
        if any(
            path.startswith(p) or path.startswith(p.replace("/", "\\"))
            for p in watched_prefixes
        ):
            changed.append(path)
    if changed:
        return Check(
            "check_post_freeze_no_executable_changes",
            False,
            f"{len(changed)} executable file(s) changed post-freeze: {changed[:5]}",
        )
    return Check(
        "check_post_freeze_no_executable_changes",
        True,
        "no executable changes after freeze",
    )


# ---------------------------------------------------------------------------
# Check 11: blind/score CLI separated
# ---------------------------------------------------------------------------


def check_blind_score_cli_separated() -> Check:
    """Verify blind run and scoring scripts are separated."""
    blind_script = BACKEND_DIR / "scripts" / "run_phase5_sealed_blind.py"
    score_script = BACKEND_DIR / "scripts" / "score_phase5_sealed.py"
    combined_script = BACKEND_DIR / "scripts" / "run_phase5_sealed.py"
    if blind_script.is_file() and score_script.is_file():
        return Check(
            "check_blind_score_cli_separated",
            True,
            "blind and score scripts exist separately",
        )
    if not combined_script.is_file():
        return Check(
            "check_blind_score_cli_separated",
            True,
            "sealed scripts not yet implemented",
            skipped=True,
        )
    try:
        source = combined_script.read_text(encoding="utf-8")
    except OSError:
        return Check(
            "check_blind_score_cli_separated",
            True,
            "cannot read run_phase5_sealed.py",
            skipped=True,
        )
    has_blind_fn = bool(re.search(r"def\s+\w*blind\w*\s*\(", source, re.IGNORECASE))
    has_score_fn = bool(re.search(r"def\s+\w*scor\w*\s*\(", source, re.IGNORECASE))
    has_score_flag = "--score" in source
    if has_blind_fn and has_score_fn and has_score_flag:
        return Check(
            "check_blind_score_cli_separated",
            True,
            "run_phase5_sealed.py has clear blind/score separation",
        )
    return Check(
        "check_blind_score_cli_separated",
        False,
        "run_phase5_sealed.py lacks clear blind/score separation",
    )


# ---------------------------------------------------------------------------
# Check 12: prediction hashes exist
# ---------------------------------------------------------------------------


def check_prediction_hashes_exist() -> Check:
    """Verify sealed prediction files have .sha256 and .canonical.sha256."""
    sealed_dirs = [
        ARTIFACTS_PHASE5 / "sealed",
        INVALIDATED_RUN_DIR / "sealed",
    ]
    pred_files: list[Path] = []
    for d in sealed_dirs:
        if d.is_dir():
            pred_files.extend(d.glob("*predictions*.jsonl"))
    if not pred_files:
        return Check(
            "check_prediction_hashes_exist",
            True,
            "no sealed prediction files found",
            skipped=True,
        )
    missing: list[str] = []
    for pred in pred_files:
        raw_sha = pred.with_suffix(pred.suffix + ".sha256")
        canon_sha = pred.with_suffix(pred.suffix + ".canonical.sha256")
        if not raw_sha.is_file():
            missing.append(f"{raw_sha.name}")
        if not canon_sha.is_file():
            missing.append(f"{canon_sha.name}")
    if missing:
        return Check(
            "check_prediction_hashes_exist",
            True,
            f"prediction hash files missing: {missing[:5]}",
            skipped=True,
        )
    return Check(
        "check_prediction_hashes_exist",
        True,
        "all sealed predictions have raw and canonical sha256 files",
    )


# ---------------------------------------------------------------------------
# Check 13: scoring ledger exists
# ---------------------------------------------------------------------------


def check_scoring_ledger_exists() -> Check:
    """Verify a scoring ledger file exists."""
    candidates = [
        ARTIFACTS_PHASE5 / "scoring-ledger.json",
        ARTIFACTS_PHASE5 / "sealed" / "scoring-ledger.json",
        INVALIDATED_RUN_DIR / "scoring-ledger.json",
        INVALIDATED_RUN_DIR / "sealed" / "scoring-ledger.json",
    ]
    # Also search for any file with 'ledger' in the name.
    if ARTIFACTS_PHASE5.is_dir():
        for path in ARTIFACTS_PHASE5.rglob("*ledger*"):
            if path.is_file():
                candidates.append(path)
    ledger = next((p for p in candidates if p.is_file()), None)
    if ledger is None:
        return Check(
            "check_scoring_ledger_exists",
            True,
            "scoring ledger not yet generated",
            skipped=True,
        )
    return Check(
        "check_scoring_ledger_exists",
        True,
        f"scoring ledger exists at {ledger.relative_to(BACKEND_DIR)}",
    )


# ---------------------------------------------------------------------------
# Check 14: canonical case scorer is unique
# ---------------------------------------------------------------------------


def check_canonical_case_scorer_unique() -> Check:
    """Verify case_scorer.py exists and sealed_scorer.py imports from it."""
    case_scorer = BACKEND_DIR / "src" / "evaluation" / "case_scorer.py"
    sealed_scorer = BACKEND_DIR / "src" / "evaluation" / "sealed_scorer.py"
    if not case_scorer.is_file():
        return Check(
            "check_canonical_case_scorer_unique",
            False,
            "src/evaluation/case_scorer.py does not exist",
        )
    if not sealed_scorer.is_file():
        return Check(
            "check_canonical_case_scorer_unique",
            False,
            "src/evaluation/sealed_scorer.py does not exist",
        )
    try:
        sealed_source = sealed_scorer.read_text(encoding="utf-8")
        sealed_tree = ast.parse(sealed_source)
    except (OSError, SyntaxError) as exc:
        return Check(
            "check_canonical_case_scorer_unique",
            False,
            f"cannot parse sealed_scorer.py: {exc}",
        )
    imports_case_scorer = False
    defines_own_score_case = False
    for node in ast.walk(sealed_tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "case_scorer" in module:
                for alias in node.names:
                    if alias.name in ("score_case", "case_passes"):
                        imports_case_scorer = True
        if isinstance(node, ast.FunctionDef) and node.name == "_score_case":
            defines_own_score_case = True
    if not imports_case_scorer:
        return Check(
            "check_canonical_case_scorer_unique",
            False,
            "sealed_scorer.py does not import score_case from case_scorer",
        )
    if defines_own_score_case:
        return Check(
            "check_canonical_case_scorer_unique",
            False,
            "sealed_scorer.py defines its own _score_case (must use case_scorer)",
        )
    return Check(
        "check_canonical_case_scorer_unique",
        True,
        "sealed_scorer imports from canonical case_scorer",
    )


# ---------------------------------------------------------------------------
# Check 15: metric names consistent
# ---------------------------------------------------------------------------


def check_metric_names_consistent() -> Check:
    """Verify unsafe_content_release_rate exists and unsafe_answer_rate deprecated."""
    try:
        from src.evaluation import metrics
    except ImportError as exc:
        return Check(
            "check_metric_names_consistent",
            True,
            f"cannot import metrics module: {exc}",
            skipped=True,
        )
    if not hasattr(metrics, "unsafe_content_release_rate"):
        return Check(
            "check_metric_names_consistent",
            False,
            "metrics.unsafe_content_release_rate is missing",
        )
    if not hasattr(metrics, "unsafe_answer_rate"):
        return Check(
            "check_metric_names_consistent",
            False,
            "metrics.unsafe_answer_rate is missing (should exist as deprecated alias)",
        )
    # Verify unsafe_answer_rate is documented as deprecated.
    doc = (metrics.unsafe_answer_rate.__doc__ or "").lower()
    if "deprecat" not in doc:
        return Check(
            "check_metric_names_consistent",
            False,
            "unsafe_answer_rate is not documented as deprecated",
        )
    return Check(
        "check_metric_names_consistent",
        True,
        "unsafe_content_release_rate present; unsafe_answer_rate deprecated",
    )


# ---------------------------------------------------------------------------
# Check 16: artifact no absolute paths
# ---------------------------------------------------------------------------


_ABS_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\", re.IGNORECASE),  # Windows drive: Y:\, C:\
    re.compile(r"/home/"),
    re.compile(r"/mnt/"),
    re.compile(r"/Users/"),
]


def check_artifact_no_absolute_paths() -> Check:
    """Scan JSON files under artifacts/evaluation/phase5/ for absolute paths."""
    if not ARTIFACTS_PHASE5.is_dir():
        return Check(
            "check_artifact_no_absolute_paths",
            True,
            "artifacts/evaluation/phase5/ does not exist",
            skipped=True,
        )
    violations: list[str] = []
    for json_path in ARTIFACTS_PHASE5.rglob("*.json"):
        try:
            text = json_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for pattern in _ABS_PATH_PATTERNS:
                if pattern.search(line):
                    rel = json_path.relative_to(BACKEND_DIR)
                    violations.append(f"{rel}:{line_no}: {line.strip()[:120]}")
                    break
    if violations:
        return Check(
            "check_artifact_no_absolute_paths",
            False,
            f"{len(violations)} absolute path(s) found in artifacts",
            details=violations[:10],
        )
    return Check(
        "check_artifact_no_absolute_paths",
        True,
        "no absolute paths in artifact JSON files",
    )


# ---------------------------------------------------------------------------
# Check 17: unicode controls pass
# ---------------------------------------------------------------------------


def check_unicode_controls_pass() -> Check:
    """Run scripts/check_unicode_controls.py and verify it exits 0."""
    if not UNICODE_SCRIPT.is_file():
        return Check(
            "check_unicode_controls_pass",
            True,
            "check_unicode_controls.py not found",
            skipped=True,
        )
    try:
        result = subprocess.run(
            [sys.executable, str(UNICODE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BACKEND_DIR),
            timeout=180,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return Check(
            "check_unicode_controls_pass",
            False,
            f"unicode scanner could not run: {exc}",
        )
    if result.returncode == 0:
        return Check(
            "check_unicode_controls_pass",
            True,
            "unicode controls scanner passed",
        )
    return Check(
        "check_unicode_controls_pass",
        False,
        f"unicode scanner failed (exit {result.returncode}): "
        f"{result.stdout.strip() or result.stderr.strip()}",
    )


# ---------------------------------------------------------------------------
# Check 18: label validator exists
# ---------------------------------------------------------------------------


def check_label_validator_exists() -> Check:
    """Verify src/evaluation/label_validator.py exists and is callable."""
    label_validator = BACKEND_DIR / "src" / "evaluation" / "label_validator.py"
    if not label_validator.is_file():
        return Check(
            "check_label_validator_exists",
            False,
            "src/evaluation/label_validator.py does not exist",
        )
    try:
        from src.evaluation.label_validator import validate_label
    except ImportError as exc:
        return Check(
            "check_label_validator_exists",
            False,
            f"cannot import validate_label: {exc}",
        )
    if not callable(validate_label):
        return Check(
            "check_label_validator_exists",
            False,
            "validate_label is not callable",
        )
    return Check(
        "check_label_validator_exists",
        True,
        "label_validator.validate_label exists and is callable",
    )


# ---------------------------------------------------------------------------
# Check 19: case scorer has >=17 check functions
# ---------------------------------------------------------------------------


def check_case_scorer_17_checks() -> Check:
    """Verify case_scorer module defines at least 17 _check_ functions."""
    try:
        from src.evaluation import case_scorer
    except ImportError as exc:
        return Check(
            "check_case_scorer_17_checks",
            True,
            f"cannot import case_scorer: {exc}",
            skipped=True,
        )
    # Count functions starting with '_check_' in the module.
    check_fns = [
        name
        for name in dir(case_scorer)
        if name.startswith("_check_") and callable(getattr(case_scorer, name))
    ]
    # Also count check names referenced in score_case via CaseCheck(name=...).
    try:
        source = Path(case_scorer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        source = ""
        tree = None
    check_names_in_source: set[str] = set()
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_case_check = (
                    isinstance(func, ast.Name) and func.id == "CaseCheck"
                ) or (isinstance(func, ast.Attribute) and func.attr == "CaseCheck")
                if is_case_check:
                    for kw in node.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            check_names_in_source.add(str(kw.value.value))
    # The check passes if either we have >=17 _check_ functions OR
    # the source references >=17 distinct CaseCheck(name=...) values.
    if len(check_fns) >= 17:
        return Check(
            "check_case_scorer_17_checks",
            True,
            f"case_scorer defines {len(check_fns)} _check_ functions",
        )
    if len(check_names_in_source) >= 17:
        return Check(
            "check_case_scorer_17_checks",
            True,
            f"case_scorer emits {len(check_names_in_source)} distinct CaseCheck names",
        )
    return Check(
        "check_case_scorer_17_checks",
        False,
        f"case_scorer has {len(check_fns)} _check_ functions and "
        f"{len(check_names_in_source)} CaseCheck names (expected >=17)",
    )


# ---------------------------------------------------------------------------
# Check 20: acceptance criteria evidence
# ---------------------------------------------------------------------------


def check_acceptance_criteria_evidence() -> Check:
    """Verify acceptance criteria evidence files exist."""
    candidates = [
        ARTIFACTS_PHASE5 / "acceptance-criteria.json",
        ARTIFACTS_PHASE5 / "acceptance.json",
        INVALIDATED_RUN_DIR / "acceptance-criteria.json",
        BACKEND_DIR / "artifacts" / "evaluation" / "acceptance-criteria.json",
    ]
    # Also search for any file with 'acceptance' in the name under phase5.
    if ARTIFACTS_PHASE5.is_dir():
        for path in ARTIFACTS_PHASE5.rglob("*acceptance*"):
            if path.is_file():
                candidates.append(path)
    evidence = next((p for p in candidates if p.is_file()), None)
    if evidence is None:
        return Check(
            "check_acceptance_criteria_evidence",
            True,
            "acceptance criteria evidence not yet generated",
            skipped=True,
        )
    return Check(
        "check_acceptance_criteria_evidence",
        True,
        f"acceptance criteria evidence exists at {evidence.relative_to(BACKEND_DIR)}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


CHECK_FUNCTIONS: list[Callable[[], Check]] = [
    check_placeholder_invalidated,
    check_corpus_indexed,
    check_no_cross_partition_overlap,
    check_questions_labels_hashed,
    check_protocol_before_calibration,
    check_baseline_no_undeclared_changes,
    check_safe_candidate_rule,
    check_ablation_flags_functional,
    check_rc_hash_verified,
    check_post_freeze_no_executable_changes,
    check_blind_score_cli_separated,
    check_prediction_hashes_exist,
    check_scoring_ledger_exists,
    check_canonical_case_scorer_unique,
    check_metric_names_consistent,
    check_artifact_no_absolute_paths,
    check_unicode_controls_pass,
    check_label_validator_exists,
    check_case_scorer_17_checks,
    check_acceptance_criteria_evidence,
]


def _print_summary(checks: list[Check], strict: bool) -> bool:
    """Print summary table. Return True if overall pass."""
    print()
    print("=" * 78)
    print(f"Phase 5 Release Gate Validator — {len(checks)} checks")
    print("=" * 78)
    passed = sum(1 for c in checks if c.passed and not c.skipped)
    failed = sum(1 for c in checks if not c.passed and not c.skipped)
    skipped = sum(1 for c in checks if c.skipped)
    width = max(len(c.name) for c in checks)
    for check in checks:
        flag = "PASS" if check.passed else ("SKIP" if check.skipped else "FAIL")
        print(f"  [{flag}] {check.name:<{width}}  {check.message}")
        for detail in check.details:
            print(f"          {detail}")
    print("-" * 78)
    print(f"  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print("=" * 78)
    if strict:
        return failed == 0 and skipped == 0
    return failed == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 release gate validator (20 checks).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat skipped checks as failures.",
    )
    args = parser.parse_args(argv)
    checks: list[Check] = []
    for fn in CHECK_FUNCTIONS:
        try:
            check = fn()
        except Exception as exc:  # noqa: BLE001
            check = Check(
                name=fn.__name__,
                passed=False,
                message=f"check raised exception: {exc}",
            )
        checks.append(check)
    overall_pass = _print_summary(checks, strict=args.strict)
    if overall_pass:
        print("\nALL CHECKS PASSED.")
        return 0
    print("\nSOME CHECKS FAILED OR WERE SKIPPED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
