"""Tests for the append-only scoring ledger."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.evaluation.scoring_ledger import (
    ScoringLedgerEntry,
    append_ledger_entry,
    read_ledger,
)


def _make_entry(**overrides: object) -> ScoringLedgerEntry:
    base: dict[str, object] = {
        "run_id": "run-1",
        "scored_at": "2026-07-23T00:00:00Z",
        "predictions_sha256": "abc",
        "labels_sha256": "def",
        "case_count": 10,
        "passed": 8,
        "failed": 2,
        "pass_rate": 0.8,
        "scorer_version": "1.0.0",
    }
    base.update(overrides)
    return ScoringLedgerEntry(**base)  # type: ignore[arg-type]


def test_append_and_read(tmp_path: Path) -> None:
    """Append an entry and read it back unchanged."""
    ledger = tmp_path / "scoring-ledger.json"
    entry = _make_entry()

    append_ledger_entry(ledger, entry)

    entries = read_ledger(ledger)
    assert len(entries) == 1
    assert entries[0] == entry
    assert entries[0].run_id == "run-1"
    assert entries[0].pass_rate == 0.8


def test_ledger_is_append_only(tmp_path: Path) -> None:
    """Multiple appends accumulate; prior entries are preserved."""
    ledger = tmp_path / "scoring-ledger.json"
    e1 = _make_entry(run_id="run-1", pass_rate=0.8)
    e2 = _make_entry(run_id="run-2", pass_rate=0.9)

    append_ledger_entry(ledger, e1)
    append_ledger_entry(ledger, e2)

    entries = read_ledger(ledger)
    assert len(entries) == 2
    assert entries[0].run_id == "run-1"
    assert entries[1].run_id == "run-2"
    assert entries[0].pass_rate == 0.8
    assert entries[1].pass_rate == 0.9


def test_entry_immutability() -> None:
    """ScoringLedgerEntry is frozen and cannot be mutated."""
    entry = _make_entry(run_id="run-1")
    with pytest.raises(FrozenInstanceError):
        entry.run_id = "run-2"  # type: ignore[misc]
