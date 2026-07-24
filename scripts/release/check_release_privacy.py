#!/usr/bin/env python3
"""隐私和安全扫描 (Privacy and security scan).

扫描 docs/release/ 和 artifacts/release/ 目录，检查是否存在:
- Windows 盘符路径 (C:\, D:\, Y:\)
- /home/<user> 路径
- 服务器 IP (10.x.x.x, 192.168.x.x)
- SSH 路径 (user@host:)
- token, API key
- 邮箱地址
- 绝对 checkpoint 路径

发现即 fail (exit 1)。
"""

import json
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.path.expanduser("~/.cache/nanochat"))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "release" / "phase6"

SCHEMA_VERSION = "1.0"

# Directories to scan
SCAN_DIRS = [
    REPO_ROOT / "docs" / "release",
    REPO_ROOT / "artifacts" / "release",
]

# File extensions to scan
SCAN_EXTENSIONS = {".md", ".json", ".txt", ".html", ".yaml", ".yml", ".csv", ".tsv"}

# Files to exclude from scan (self-referential output files)
EXCLUDE_FILES = {
    "privacy-scan.json",
    "claim-validation.json",
}

# Privacy patterns
PATTERNS = {
    "absolute_checkpoint_path": {
        "description": "Absolute checkpoint path leakage",
        "regex": re.compile(
            r"(?:/home|/mnt|/data|/opt|/var|/root)/[^\s\"']+\.(?:pt|ckpt|safetensors|bin)",
            re.IGNORECASE,
        ),
        "severity": "error",
    },
    "email": {
        "description": "Email address leakage",
        "regex": re.compile(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
        ),
        "severity": "warning",
    },
    "server_ip_10": {
        "description": "Internal server IP (10.x.x.x)",
        "regex": re.compile(
            r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
        ),
        "severity": "error",
    },
    "server_ip_192": {
        "description": "Internal server IP (192.168.x.x)",
        "regex": re.compile(
            r"\b192\.168\.\d{1,3}\.\d{1,3}\b"
        ),
        "severity": "error",
    },
    "server_ip_172": {
        "description": "Internal server IP (172.16-31.x.x)",
        "regex": re.compile(
            r"\b172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}\b"
        ),
        "severity": "error",
    },
    "ssh_path": {
        "description": "SSH path (user@host:)",
        "regex": re.compile(
            r"\b[a-zA-Z][a-zA-Z0-9._-]*@[a-zA-Z0-9.-]+:"
        ),
        "severity": "error",
    },
    "windows_drive_c": {
        "description": "Windows drive path C:\\",
        "regex": re.compile(r"[Cc]:[\\/]", re.IGNORECASE),
        "severity": "error",
    },
    "windows_drive_d": {
        "description": "Windows drive path D:\\",
        "regex": re.compile(r"[Dd]:[\\/]", re.IGNORECASE),
        "severity": "error",
    },
    "windows_drive_y": {
        "description": "Windows drive path Y:\\",
        "regex": re.compile(r"[Yy]:[\\/]", re.IGNORECASE),
        "severity": "error",
    },
    "home_path": {
        "description": "Absolute /home/<user> path",
        "regex": re.compile(
            r"/home/[a-zA-Z0-9._-]+/[^\s\"']+"
        ),
        "severity": "error",
    },
    "api_key": {
        "description": "API key leakage",
        "regex": re.compile(
            r"(?:api[_-]?key|secret[_-]?key|access[_-]?token|bearer)\s*[=:]\s*['\"]?[a-zA-Z0-9_\-]{16,}",
            re.IGNORECASE,
        ),
        "severity": "error",
    },
}


def scan_file(path: Path) -> list:
    """Scan a single file for privacy violations."""
    findings = []
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return findings

    lines = content.splitlines()
    for line_num, line in enumerate(lines, 1):
        for pattern_name, pattern_info in sorted(PATTERNS.items()):
            matches = pattern_info["regex"].finditer(line)
            for match in matches:
                try:
                    rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                except ValueError:
                    rel = str(path).replace("\\", "/")
                findings.append({
                    "column": match.start(),
                    "description": pattern_info["description"],
                    "line": line_num,
                    "matched_text": match.group(0),
                    "pattern": pattern_name,
                    "path": rel,
                    "severity": pattern_info["severity"],
                })
    return findings


def collect_scan_files() -> list:
    """Collect all files to scan."""
    files = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            continue
        for fpath in sorted(scan_dir.rglob("*")):
            if fpath.is_file() and fpath.suffix in SCAN_EXTENSIONS:
                if fpath.name in EXCLUDE_FILES:
                    continue
                files.append(fpath)
    return files


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_scan_files()
    all_findings = []
    for fpath in files:
        findings = scan_file(fpath)
        all_findings.extend(findings)

    # Sort findings for deterministic output
    all_findings.sort(key=lambda x: (x.get("path", ""), x.get("line", 0), x.get("column", 0), x.get("pattern", "")))

    has_errors = any(f.get("severity") == "error" for f in all_findings)

    result = {
        "files_scanned": len(files),
        "findings": all_findings,
        "passed": not has_errors,
        "scan_dirs": [str(d.relative_to(REPO_ROOT)).replace("\\", "/") for d in SCAN_DIRS],
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "errors": sum(1 for f in all_findings if f.get("severity") == "error"),
            "total_findings": len(all_findings),
            "warnings": sum(1 for f in all_findings if f.get("severity") == "warning"),
        },
    }

    output_path = OUTPUT_DIR / "privacy-scan.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")

    if result["passed"]:
        print("PASS: No privacy violations found.")
        sys.exit(0)
    else:
        print(f"FAIL: {result['summary']['errors']} privacy violations found.")
        sys.exit(1)


if __name__ == "__main__":
    main()
