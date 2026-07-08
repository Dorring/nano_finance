"""Phase 8A tests: session TTL and cleanup behavior."""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.session_manager import SessionManager


def _close(mgr):
    try:
        mgr.close()
    except Exception:
        pass


def _set_created_at(db_path, content, created_at):
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE conversations SET created_at = ? WHERE content = ?", (created_at, content))
    conn.commit()
    conn.close()


def test_cleanup_expired_disabled_by_default(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db)
    try:
        mgr.add_message("s1", 1, "user", "old")
        _set_created_at(db, "old", 0)
        assert mgr.cleanup_expired(now=1000) == 0
        assert mgr.get_session_count("s1", 1) == 1
    finally:
        _close(mgr)


def test_cleanup_expired_deletes_old_messages(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db, ttl_seconds=60)
    try:
        mgr.add_message("s1", 1, "user", "old")
        mgr.add_message("s1", 1, "assistant", "new")
        _set_created_at(db, "old", 100)
        _set_created_at(db, "new", 190)
        assert mgr.cleanup_expired(now=200) == 1
        mgr.ttl_seconds = 0  # avoid a second cleanup using real wall-clock time
        msgs = mgr.get_recent_messages("s1", 1)
        assert [m["content"] for m in msgs] == ["new"]
    finally:
        _close(mgr)


def test_cleanup_is_tenant_safe_but_global_retention(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db, ttl_seconds=60)
    try:
        mgr.add_message("s1", 1, "user", "u1_old")
        mgr.add_message("s1", 2, "user", "u2_new")
        _set_created_at(db, "u1_old", 100)
        _set_created_at(db, "u2_new", 190)
        assert mgr.cleanup_expired(now=200) == 1
        mgr.ttl_seconds = 0  # avoid a second cleanup using real wall-clock time
        assert mgr.get_session_count("s1", 1) == 0
        assert mgr.get_session_count("s1", 2) == 1
    finally:
        _close(mgr)


def test_get_recent_messages_runs_cleanup(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db, ttl_seconds=60)
    try:
        mgr.add_message("s1", 1, "user", "old")
        _set_created_at(db, "old", 0)
        assert mgr.get_recent_messages("s1", 1) == []
    finally:
        _close(mgr)


def test_prune_session_history_deletes_oldest_messages(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db)
    try:
        for i in range(5):
            mgr.add_message("s1", 1, "user" if i % 2 == 0 else "assistant", f"m{i}")
        deleted = mgr.prune_session_history("s1", 1, max_messages=2)
        assert deleted == 3
        msgs = mgr.get_recent_messages("s1", 1, n_pairs=10)
        assert [m["content"] for m in msgs] == ["m3", "m4"]
    finally:
        _close(mgr)


def test_prune_session_history_is_tenant_scoped(tmp_path):
    db = str(tmp_path / "sessions.db")
    mgr = SessionManager(db_path=db)
    try:
        for user_id in (1, 2):
            for i in range(3):
                mgr.add_message("s1", user_id, "user", f"u{user_id}_{i}")
        assert mgr.prune_session_history("s1", 1, max_messages=1) == 2
        assert mgr.get_session_count("s1", 1) == 1
        assert mgr.get_session_count("s1", 2) == 3
    finally:
        _close(mgr)


def test_ttl_from_environment(tmp_path, monkeypatch):
    db = str(tmp_path / "sessions.db")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "123")
    mgr = SessionManager(db_path=db)
    try:
        assert mgr.ttl_seconds == 123
    finally:
        _close(mgr)
