"""Phase 13A tests: trace retention cleanup."""
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.trace import TraceLogger
from src.evaluation.eval_cli import main as eval_cli_main


def _set_created_at(db_path, trace_id, created_at):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE trace_log SET created_at = ? WHERE trace_id = ?",
            (created_at, trace_id),
        )
        conn.commit()


def test_cleanup_older_than_is_tenant_scoped(tmp_path):
    db_path = str(tmp_path / "trace.db")
    logger = TraceLogger(db_path=db_path, sample_rate=1.0, redact_content=False)
    old_t1 = logger.log(tenant_id=1, query_original="old tenant 1")
    old_t2 = logger.log(tenant_id=2, query_original="old tenant 2")
    new_t1 = logger.log(tenant_id=1, query_original="new tenant 1")
    _set_created_at(db_path, old_t1, 100)
    _set_created_at(db_path, old_t2, 100)
    _set_created_at(db_path, new_t1, 300)

    deleted = logger.cleanup_older_than(200, tenant_id=1)

    assert deleted == 1
    assert logger.get_trace(old_t1) is None
    assert logger.get_trace(old_t2) is not None
    assert logger.get_trace(new_t1) is not None


def test_cleanup_older_than_global_removes_all_old_rows(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=False)
    old_t1 = logger.log(tenant_id=1, query_original="old tenant 1")
    old_t2 = logger.log(tenant_id=2, query_original="old tenant 2")
    _set_created_at(logger.db_path, old_t1, 100)
    _set_created_at(logger.db_path, old_t2, 100)

    deleted = logger.cleanup_older_than(200)

    assert deleted == 2
    assert logger.query_traces(tenant_id=1) == []
    assert logger.query_traces(tenant_id=2) == []


def test_cleanup_by_ttl_returns_metadata(tmp_path):
    logger = TraceLogger(db_path=str(tmp_path / "trace.db"), sample_rate=1.0, redact_content=False)
    old_trace = logger.log(tenant_id=1, query_original="old")
    new_trace = logger.log(tenant_id=1, query_original="new")
    _set_created_at(logger.db_path, old_trace, 100)
    _set_created_at(logger.db_path, new_trace, 250)

    report = logger.cleanup_by_ttl(ttl_seconds=100, tenant_id=1, now=250)

    assert report["deleted"] == 1
    assert report["tenant_id"] == 1
    assert report["ttl_seconds"] == 100
    assert report["cutoff_created_at"] == 150
    assert logger.get_trace(old_trace) is None
    assert logger.get_trace(new_trace) is not None


def test_eval_cli_traces_cleanup(tmp_path, capsys):
    db_path = str(tmp_path / "trace.db")
    logger = TraceLogger(db_path=db_path, sample_rate=1.0, redact_content=False)
    old_trace = logger.log(tenant_id=1, query_original="old")
    new_trace = logger.log(tenant_id=1, query_original="new")
    now = time.time()
    _set_created_at(db_path, old_trace, now - 200)
    _set_created_at(db_path, new_trace, now)

    code = eval_cli_main([
        "traces-cleanup",
        "--db",
        db_path,
        "--tenant-id",
        "1",
        "--ttl-seconds",
        "100",
    ])

    assert code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["deleted"] == 1
    remaining = TraceLogger(db_path=db_path).query_traces(tenant_id=1)
    assert [row["trace_id"] for row in remaining] == [new_trace]
