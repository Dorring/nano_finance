import sqlite3
import time
import json
import threading


class SessionManager:
    """
    SQLite-backed conversation session manager.

    Stores conversation history per session with tenant isolation.
    Designed for short-term memory: recent messages for query rewriting context.
    Historical answers are NEVER used as financial facts in retrieval context.

    Schema version: 1
    """

    SCHEMA_VERSION = 1
    DEFAULT_MAX_HISTORY = 8  # max message pairs to keep

    def __init__(self, db_path: str = "sessions.db", max_history: int = None):
        self.db_path = db_path
        self.max_history = max_history or self.DEFAULT_MAX_HISTORY
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
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversations(session_id, user_id, created_at);
            CREATE TABLE IF NOT EXISTS schema_version (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
        """)
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (component, version) VALUES (?, ?)",
            ("session_manager", self.SCHEMA_VERSION),
        )
        conn.commit()

    def add_message(self, session_id: str, user_id: int, role: str, content: str):
        """
        Store a conversation message.

        Args:
            session_id: Unique session identifier
            user_id: Tenant ID for isolation
            role: 'user' or 'assistant'
            content: Message content
        """
        if not session_id or user_id is None:
            return  # fail closed: reject missing identifiers
        if role not in ("user", "assistant"):
            return  # only allow known roles

        conn = self._get_conn()
        conn.execute(
            "INSERT INTO conversations (session_id, user_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, role, content, time.time()),
        )
        conn.commit()

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
        if not session_id or user_id is None:
            return []

        if n_pairs is None:
            n_pairs = self.max_history

        limit = n_pairs * 2
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT role, content FROM conversations
               WHERE session_id = ? AND user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (session_id, user_id, limit),
        ).fetchall()

        # Reverse to chronological order
        messages = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
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
        if not session_id or user_id is None:
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
        if not session_id or user_id is None:
            return 0
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM conversations WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
        return row["cnt"]

    def close(self):
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
