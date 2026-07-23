"""Append-only scoring ledger for Phase 5 sealed evaluations.

The ledger records every scoring run as an immutable ``ScoringLedgerEntry``.
It is stored as a JSON list of objects and grows monotonically: each
``append_ledger_entry`` call reads the existing list, appends the new
entry, and rewrites the file atomically. This makes the scoring history
auditable without modifying prior records.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScoringLedgerEntry:
    """Immutable record of a single sealed scoring run."""

    run_id: str
    scored_at: str
    predictions_sha256: str
    labels_sha256: str
    case_count: int
    passed: int
    failed: int
    pass_rate: float
    scorer_version: str

    @classmethod
    def from_dict(cls, data: dict) -> "ScoringLedgerEntry":
        """Reconstruct a ``ScoringLedgerEntry`` from a JSON-compatible dict."""
        return cls(
            run_id=str(data["run_id"]),
            scored_at=str(data["scored_at"]),
            predictions_sha256=str(data["predictions_sha256"]),
            labels_sha256=str(data["labels_sha256"]),
            case_count=int(data["case_count"]),
            passed=int(data["passed"]),
            failed=int(data["failed"]),
            pass_rate=float(data["pass_rate"]),
            scorer_version=str(data["scorer_version"]),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict with stable key ordering."""
        return asdict(self)


def append_ledger_entry(ledger_path: Path, entry: ScoringLedgerEntry) -> None:
    """Append an entry to the scoring ledger JSON file atomically.

    The ledger is a JSON array of ``ScoringLedgerEntry`` dicts. The file is
    rewritten in full via a temp-file + ``os.replace`` so a crashed write
    never leaves a partial ledger on disk.
    """
    ledger_path = Path(ledger_path)
    entries = read_ledger(ledger_path)
    entries.append(entry)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            [e.to_dict() for e in entries], ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n"
    )
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=ledger_path.parent,
            prefix=f".{ledger_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, ledger_path)
    except Exception:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def read_ledger(ledger_path: Path) -> list[ScoringLedgerEntry]:
    """Read all entries from the scoring ledger.

    Returns an empty list when the ledger file does not exist yet. Raises
    ``ValueError`` when the file exists but is not a JSON array.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.is_file():
        return []
    data = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            f"scoring ledger at {ledger_path} must be a JSON array, "
            f"got {type(data).__name__}"
        )
    return [ScoringLedgerEntry.from_dict(item) for item in data]
