#!/usr/bin/env python3
"""Scan source files for hidden Unicode bidirectional and control characters.

Trojan Source attacks (CVE-2021-42574) and similar supply-chain attacks exploit
invisible Unicode bidi/control characters to hide malicious code in plain
sight. This scanner rejects the following characters whenever they appear in
source files under the watched directories:

    U+202A  LEFT-TO-RIGHT EMBEDDING
    U+202B  RIGHT-TO-LEFT EMBEDDING
    U+202C  POP DIRECTIONAL FORMATTING
    U+202D  LEFT-TO-RIGHT OVERRIDE
    U+202E  RIGHT-TO-LEFT OVERRIDE
    U+2066  LEFT-TO-RIGHT ISOLATE
    U+2067  RIGHT-TO-LEFT ISOLATE
    U+2068  FIRST STRONG ISOLATE
    U+2069  POP DIRECTIONAL ISOLATE
    U+200B  ZERO WIDTH SPACE
    U+200E  LEFT-TO-RIGHT MARK
    U+200F  RIGHT-TO-LEFT MARK
    U+FEFF  BYTE ORDER MARK (only rejected in the *middle* of a file)

Normal Chinese characters, English text, and regular Unicode punctuation
(em dashes, smart quotes, etc.) are always allowed.

Scanned directories (relative to the backend root):
    src/  scripts/  tests/  eval_data/  docs/  config/

Skipped directories/files:
    .git/  __pycache__/  *.pyc  node_modules/  .venv/  venv/

Usage:
    python scripts/check_unicode_controls.py            # scan only
    python scripts/check_unicode_controls.py --fix      # remove violations

Exit codes:
    0 = clean (or all violations fixed)
    1 = one or more violations found (or remaining after --fix)
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent

SCAN_DIRS = ("src", "scripts", "tests", "eval_data", "docs", "config")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

# Forbidden control/bidi characters: codepoint -> short reason.
# U+FEFF is handled specially (only flagged in the middle of a file).
FORBIDDEN_CONTROLS: dict[int, str] = {
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0x200B: "ZERO WIDTH SPACE",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
}

BOM_CODEPOINT = 0xFEFF

# This script's own path (never fix itself).
SELF_PATH = Path(__file__).resolve()


def _char_name(codepoint: int) -> str:
    """Return the Unicode name for a codepoint, or a fallback string."""
    try:
        return unicodedata.name(chr(codepoint))
    except ValueError:
        return f"U+{codepoint:04X} (no Unicode name)"


def _should_skip_dir(dirname: str) -> bool:
    """Return True if a directory entry should be skipped."""
    return dirname in SKIP_DIRS


def _iter_source_files(root: Path) -> list[Path]:
    """Yield all candidate source files under ``root``.

    Skips the ``SKIP_DIRS`` directories and ``*.pyc`` files. Files are
    read lazily by the caller; binary files are filtered at read time.
    """
    if not root.is_dir():
        return []
    results: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".pyc":
            continue
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in parts):
            continue
        results.append(path)
    return results


def _read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text. Return None for binary/unreadable files."""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def scan_text(text: str) -> list[tuple[int, int, int, str]]:
    """Scan text for forbidden control characters.

    Returns a list of ``(line_no, col_no, codepoint, name)`` tuples.
    Line numbers are 1-based; column numbers are 0-based character offsets
    within the line.

    U+FEFF (BOM) is only flagged when it appears *after* the first
    character of the file — a leading BOM is a legitimate byte-order mark.
    """
    violations: list[tuple[int, int, int, str]] = []
    line_no = 1
    col = 0
    for idx, ch in enumerate(text):
        if ch == "\n":
            line_no += 1
            col = 0
            continue
        codepoint = ord(ch)
        if codepoint == BOM_CODEPOINT:
            # Leading BOM (idx == 0) is allowed; mid-file BOM is not.
            if idx != 0:
                violations.append((line_no, col, codepoint, _char_name(codepoint)))
        elif codepoint in FORBIDDEN_CONTROLS:
            violations.append((line_no, col, codepoint, _char_name(codepoint)))
        col += 1
    return violations


def fix_text(text: str) -> str:
    """Return ``text`` with all forbidden control characters removed.

    A leading U+FEFF BOM is preserved. All other forbidden characters
    (including mid-file BOMs) are stripped.
    """
    out: list[str] = []
    for idx, ch in enumerate(text):
        codepoint = ord(ch)
        if codepoint == BOM_CODEPOINT:
            if idx == 0:
                out.append(ch)
            # else: drop mid-file BOM
            continue
        if codepoint in FORBIDDEN_CONTROLS:
            continue
        out.append(ch)
    return "".join(out)


def scan_file(path: Path) -> list[tuple[int, int, int, str]]:
    """Scan one file and return its violation list (empty if clean)."""
    text = _read_text(path)
    if text is None:
        return []
    return scan_text(text)


def fix_file(path: Path) -> int:
    """Remove forbidden characters from ``path``. Return count removed."""
    text = _read_text(path)
    if text is None:
        return 0
    before = len(scan_text(text))
    if before == 0:
        return 0
    fixed = fix_text(text)
    path.write_text(fixed, encoding="utf-8")
    after = len(scan_text(fixed))
    return before - after


def _format_violation(
    path: Path, line_no: int, col: int, codepoint: int, name: str
) -> str:
    """Format a single violation line for human-readable output."""
    rel = _relative_to_backend(path)
    return f"{rel}:{line_no}:{col}: U+{codepoint:04X} {name}"


def _relative_to_backend(path: Path) -> str:
    """Return ``path`` relative to BACKEND_DIR, or the absolute path."""
    try:
        return str(path.resolve().relative_to(BACKEND_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scan source files for hidden Unicode bidi/control chars.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove offending characters in place (except in this script).",
    )
    args = parser.parse_args(argv)

    total_violations = 0
    fixed_count = 0

    for subdir in SCAN_DIRS:
        root = BACKEND_DIR / subdir
        if not root.is_dir():
            continue
        for path in _iter_source_files(root):
            if args.fix and path.resolve() == SELF_PATH:
                continue
            violations = scan_file(path)
            if not violations:
                continue
            if args.fix:
                removed = fix_file(path)
                fixed_count += removed
                remaining = scan_file(path)
                for line_no, col, codepoint, name in remaining:
                    print(
                        _format_violation(path, line_no, col, codepoint, name),
                        file=sys.stderr,
                    )
                    total_violations += 1
            else:
                for line_no, col, codepoint, name in violations:
                    print(_format_violation(path, line_no, col, codepoint, name))
                    total_violations += 1

    if args.fix:
        if fixed_count:
            print(f"Removed {fixed_count} forbidden character(s).")
        if total_violations:
            print(
                f"FAIL: {total_violations} violation(s) remain after --fix.",
                file=sys.stderr,
            )
            return 1
        print("PASS: no forbidden Unicode control characters remain.")
        return 0

    if total_violations:
        print(
            f"FAIL: {total_violations} forbidden Unicode control character(s) found.",
            file=sys.stderr,
        )
        return 1
    print("PASS: no forbidden Unicode control characters found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
