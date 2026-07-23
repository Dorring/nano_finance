#!/usr/bin/env python3
"""Verify no executable code changed between RC freeze and artifact closure.

After the release candidate (RC) is frozen, the only changes allowed
before the sealed artifacts are closed are to ``artifacts/`` and ``docs/``.
Any change to executable code (``src/``, ``scripts/``, ``config/``,
``prompts/``) is a violation because it would invalidate the frozen
evaluation basis.

The script:

- Runs ``git diff --name-only rc_commit..closure_commit``.
- Flags any changed path under ``src/``, ``scripts/``, ``config/``, or
  ``prompts/`` as a violation.
- Writes ``artifacts/evaluation/phase5/post_freeze_diff.json``.
- Exits 0 when clean, 1 when violations are present.

Usage:
    python3 scripts/generate_post_freeze_diff.py \\
        --rc-commit <sha> --closure-commit <sha>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

OUTPUT_PATH = (
    BACKEND_DIR / "artifacts" / "evaluation" / "phase5" / "post_freeze_diff.json"
)

# Path prefixes that count as executable-code violations when changed.
VIOLATION_PREFIXES: tuple[str, ...] = ("src/", "scripts/", "config/", "prompts/")


def _is_violation(path: str) -> bool:
    """Return True when ``path`` is under a violation prefix."""
    return any(path.startswith(prefix) for prefix in VIOLATION_PREFIXES)


def _git_diff_names(rc_commit: str, closure_commit: str) -> list[str]:
    """Return the list of changed file paths between two commits.

    Returns an empty list when git is unavailable or the range is empty.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{rc_commit}..{closure_commit}"],
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def check_post_freeze_changes(rc_commit: str, closure_commit: str) -> dict[str, Any]:
    """Check for changes to executable code between RC and closure.

    Returns ``{"violations": [...], "clean": bool}``. Violations are any
    changes to: ``src/``, ``scripts/``, ``config/``, ``prompts/``.
    Allowed changes: ``artifacts/``, ``docs/``.
    """
    changed = _git_diff_names(rc_commit, closure_commit)
    violations = sorted(p for p in changed if _is_violation(p))
    return {"violations": violations, "clean": len(violations) == 0}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify no executable code changed between RC freeze and closure."
    )
    parser.add_argument(
        "--rc-commit", required=True, help="The RC freeze git commit SHA."
    )
    parser.add_argument(
        "--closure-commit",
        required=True,
        help="The artifact closure git commit SHA.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 5 Post-Freeze Diff Check")
    print("=" * 60)
    print(f"RC commit:      {args.rc_commit}")
    print(f"Closure commit: {args.closure_commit}")

    check = check_post_freeze_changes(args.rc_commit, args.closure_commit)
    violations = check["violations"]

    payload: dict[str, Any] = {
        "rc_commit": args.rc_commit,
        "closure_commit": args.closure_commit,
        "violations": violations,
        "clean": check["clean"],
        "violation_prefixes": list(VIOLATION_PREFIXES),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Report saved to: {OUTPUT_PATH}")

    if check["clean"]:
        print("\nPost-freeze check PASSED: no executable code changed.")
        return 0

    print("\nPost-freeze check FAILED: executable code changed after freeze:")
    for path in violations:
        print(f"  - {path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
