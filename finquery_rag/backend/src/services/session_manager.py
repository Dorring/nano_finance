import sqlite3
import time
import json
import threading

from .sqlite_migrations import ensure_column, run_component_migrations


class SessionManager:
    """
    SQLite-backed conversation session manager.

    Stores conversation history per session with tenant isolation.
    Designed for short-term memory: recent messages for query rewriting context.
    Historical answers are NEVER used as financial facts in retrieval context.

    Schema version: 2

    Retention is opt-in and disabled by default for backwards compatibility.
    """

    SCHEMA_VERSION = 2
    DEFAULT_MAX_HISTORY = 8  # max message pairs to keep
    DEFAULT_TTL_SECONDS = 0  # disabled; set env SESSION_TTL_SECONDS to enable
    MAX_SESSION_ID_LENGTH = 128
    MAX_CONTENT_CHARS = 20000
    MAX_METADATA_JSON_CHARS = 20000
    MAX_QUERY_LIMIT = 1000

    def __init__(self, db_path: str = None, max_history: int = None, ttl_seconds: int = None):
        import os
        if db_path is None:
            db_path = os.getenv("SESSIONS_DB_PATH", "sessions.db")
        if ttl_seconds is None:
            ttl_seconds = int(os.getenv("SESSION_TTL_SECONDS", str(self.DEFAULT_TTL_SECONDS)))
        self.db_path = db_path
        self.max_history = max(1, int(max_history or self.DEFAULT_MAX_HISTORY))
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection with WAL mode."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self):
        """Initialize database schema with idempotent table creation."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations(session_id, user_id, created_at);
            CREATE TABLE IF NOT EXISTS schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
        """)
        run_component_migrations(
            conn,
            "session_manager",
            self.SCHEMA_VERSION,
            {
                2: self._migrate_to_v2,
            },
        )

    @staticmethod
    def _migrate_to_v2(conn):
        ensure_column(conn, "conversations", "metadata_json", "metadata_json TEXT")

    def _is_valid_session_id(self, session_id: str) -> bool:
        return isinstance(session_id, str) and 0 < len(session_id) <= self.MAX_SESSION_ID_LENGTH

    def _bounded_content(self, content: str) -> str:
        content = str(content or "")
        if len(content) <= self.MAX_CONTENT_CHARS:
            return content
        return content[: self.MAX_CONTENT_CHARS] + "\n[truncated]"

    def _bounded_metadata_json(self, metadata: dict = None) -> str | None:
        if not metadata:
            return None
        metadata_json = json.dumps(metadata)
        if len(metadata_json) <= self.MAX_METADATA_JSON_CHARS:
            return metadata_json
        return json.dumps({"truncated": True})

    def add_message(self, session_id: str, user_id: int, role: str, content: str, metadata: dict = None):
        """
        Store a conversation message.

        Args:
            session_id: Unique session identifier
            user_id: Tenant ID for isolation
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional UI/debug metadata such as sources and diagnostics
        """
        if not self._is_valid_session_id(session_id) or user_id is None:
            return  # fail closed: reject missing identifiers or oversized session IDs
        if role not in ("user", "assistant"):
            return  # only allow known roles

        conn = self._get_conn()
        metadata_json = self._bounded_metadata_json(metadata)
        conn.execute(
            "INSERT INTO conversations (session_id, user_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, user_id, role, self._bounded_content(content), metadata_json, time.time()),
        )
        conn.commit()
        self.prune_session_history(session_id, user_id)

    def cleanup_expired(self, now: float = None) -> int:
        """Delete expired session messages and return deleted row count.

        TTL is disabled when ttl_seconds <= 0.
        """
        if self.ttl_seconds <= 0:
            return 0
        if now is None:
            now = time.time()
        cutoff = now - self.ttl_seconds
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE created_at < ?",
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount

    def prune_session_history(self, session_id: str, user_id: int, max_messages: int = None) -> int:
        """Trim oldest messages for one session and return deleted row count."""
        if not self._is_valid_session_id(session_id) or user_id is None:
            return 0
        if max_messages is None:
            max_messages = self.max_history * 2
        try:
            max_messages = max(0, min(int(max_messages), self.MAX_QUERY_LIMIT))
        except (TypeError, ValueError):
            return 0
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id FROM conversations
               WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC LIMIT -1 OFFSET ?""",
            (session_id, user_id, max_messages),
        ).fetchall()
        ids = [row["id"] for row in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        cursor = conn.execute(
            f"DELETE FROM conversations WHERE id IN ({placeholders})",
            ids,
        )
        conn.commit()
        return cursor.rowcount

    def get_recent_messages(self, session_id: str, user_id: int, n_pairs: int = None) -> list:
        """
        Retrieve recent conversation messages for a session.

        Returns messages in chronological order, limited to n_pairs * 2 messages.
        Each pair is (user_message, assistant_message).

        Args:
            session_id: Session to retrieve
            user_id: Tenant ID for isolation
            n_pairs: Max number of Q&A pairs to return (default: self.max_history)

        Returns:
            List of dicts: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        if not self._is_valid_session_id(session_id) or user_id is None:
            return []

        self.cleanup_expired()

        if n_pairs is None:
            n_pairs = self.max_history
        try:
            n_pairs = max(0, min(int(n_pairs), self.MAX_QUERY_LIMIT // 2))
        except (TypeError, ValueError):
            return []

        limit = n_pairs * 2
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT role, content, metadata_json FROM conversations
               WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, user_id, limit),
        ).fetchall()

        # Reverse to chronological order
        messages = []
        for row in reversed(rows):
            metadata = {}
            if row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except (TypeError, json.JSONDecodeError):
                    metadata = {}
            messages.append({
                "role": row["role"],
                "content": row["content"],
                "metadata": metadata,
            })
        return messages

    def clear_session(self, session_id: str, user_id: int) -> bool:
        """
        Delete all messages in a session.

        Args:
            session_id: Session to clear
            user_id: Tenant ID (required, fail closed)

        Returns:
            True if messages were deleted, False if no messages found
        """
        if not self._is_valid_session_id(session_id) or user_id is None:
            return False

        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def get_session_count(self, session_id: str, user_id: int) -> int:
        """Return the number of messages in a session."""
        if not self._is_valid_session_id(session_id) or user_id is None:
            return 0
        self.cleanup_expired()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return row["cnt"]

    def list_sessions(self, user_id: int, limit: int = 50, offset: int = 0) -> list:
        """List session summaries for one tenant ordered by latest activity."""
        if user_id is None:
            return []
        self.cleanup_expired()
        try:
            normalized_limit = max(1, min(int(limit or 50), self.MAX_QUERY_LIMIT))
            normalized_offset = max(0, int(offset or 0))
        except (TypeError, ValueError):
            return []
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT session_id, COUNT(*) AS message_count,
                      MIN(created_at) AS created_at, MAX(created_at) AS updated_at
               FROM conversations
               WHERE user_id = ?
               GROUP BY session_id
               ORDER BY updated_at DESC
               LIMIT ? OFFSET ?""",
            (user_id, normalized_limit, normalized_offset),
        ).fetchall()
        return [
            {
                "session_id": row["session_id"],
                "message_count": int(row["message_count"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def count_sessions(self, user_id: int) -> int:
        """Return number of sessions with at least one message for one tenant."""
        if user_id is None:
            return 0
        self.cleanup_expired()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM (SELECT 1 FROM conversations WHERE user_id = ? GROUP BY session_id)",
            (user_id,),
        ).fetchone()
        return int(row["cnt"] or 0)

    def clear_all_for_user(self, user_id: int) -> int:
        """Delete all session messages for one tenant and return deleted rows."""
        if user_id is None:
            return 0
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
        conn.commit()
        return cursor.rowcount

    def storage_summary(self, user_id: int = None) -> dict:
        """Return compact storage diagnostics without message content."""
        self.cleanup_expired()
        conn = self._get_conn()
        if user_id is None:
            row = conn.execute(
                "SELECT COUNT(*) AS messages, COUNT(DISTINCT session_id) AS sessions FROM conversations"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS messages, COUNT(DISTINCT session_id) AS sessions FROM conversations WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return {
            "user_id": user_id,
            "sessions": int(row["sessions"] or 0),
            "messages": int(row["messages"] or 0),
            "ttl_seconds": self.ttl_seconds,
            "max_history_pairs": self.max_history,
        }

    def close(self):
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
