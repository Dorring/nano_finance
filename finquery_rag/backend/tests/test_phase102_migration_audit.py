import json
import sqlite3

from src.eval_cli import main as eval_cli_main
from src.services.migration_audit import audit_migration_readiness


def _create_bm25(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE chunk_store (
                doc_id TEXT PRIMARY KEY,
                content TEXT,
                metadata_json TEXT,
                user_id INTEGER,
                doc_name TEXT
            )
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE fts_index USING fts5(
                content,
                doc_id UNINDEXED,
                tokenize='unicode61'
            )
        """)
        for doc_id, user_id, doc_name in rows:
            conn.execute(
                "INSERT INTO chunk_store(doc_id, content, metadata_json, user_id, doc_name) VALUES (?, ?, ?, ?, ?)",
                (doc_id, "redacted", "{}", user_id, doc_name),
            )
            conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("redacted", doc_id))
        conn.commit()


def _create_registry(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE document_registry (
                document_id TEXT PRIMARY KEY,
                tenant_id INTEGER,
                filename TEXT,
                status TEXT,
                chunk_count INTEGER
            )
        """)
        conn.executemany(
            "INSERT INTO document_registry(document_id, tenant_id, filename, status, chunk_count) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()


def _create_chroma(path, ids):
    path.mkdir()
    with sqlite3.connect(path / "chroma.sqlite3") as conn:
        conn.execute("CREATE TABLE embeddings (id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO embeddings(id) VALUES (?)", [(item,) for item in ids])
        conn.commit()


def test_migration_audit_passes_clean_scoped_stores(tmp_path):
    bm25 = tmp_path / "bm25.db"
    registry = tmp_path / "registry.db"
    chroma = tmp_path / "chroma"
    _create_bm25(bm25, [("user_7_report.pdf::1", 7, "report.pdf")])
    _create_registry(registry, [("doc1", 7, "report.pdf", "ready", 1)])
    _create_chroma(chroma, ["user_7_report.pdf::1"])

    report = audit_migration_readiness(
        bm25_db_path=str(bm25),
        registry_db_path=str(registry),
        chroma_path=str(chroma),
    )

    assert report["passed"] is True
    assert report["summary"]["high_risk_count"] == 0
    assert report["stores"]["bm25"]["counts"]["legacy_unscoped_doc_ids"] == 0
    assert report["stores"]["chroma"]["counts"]["legacy_unscoped_embedding_ids"] == 0


def test_migration_audit_flags_legacy_unscoped_indexes(tmp_path):
    bm25 = tmp_path / "bm25.db"
    registry = tmp_path / "registry.db"
    chroma = tmp_path / "chroma"
    _create_bm25(bm25, [("legacy::1", None, ""), ("user_2_doc.pdf::1", 3, "doc.pdf")])
    _create_registry(registry, [("doc1", 7, "report.pdf", "ready", 0)])
    _create_chroma(chroma, ["legacy::1", "user_7_report.pdf::1"])

    report = audit_migration_readiness(
        bm25_db_path=str(bm25),
        registry_db_path=str(registry),
        chroma_path=str(chroma),
    )

    assert report["passed"] is False
    codes = {risk["code"] for risk in report["risks"]}
    assert "bm25_legacy_doc_ids" in codes
    assert "bm25_missing_user_id" in codes
    assert "bm25_scope_user_mismatch" in codes
    assert "registry_ready_zero_chunks" in codes
    assert "chroma_legacy_embedding_ids" in codes
    assert "Rebuild BM25" in report["recommendations"][0]


def test_eval_cli_migration_audit_returns_one_on_high_risk(tmp_path, capsys):
    bm25 = tmp_path / "bm25.db"
    registry = tmp_path / "registry.db"
    _create_bm25(bm25, [("legacy::1", None, "")])
    _create_registry(registry, [("doc1", 7, "report.pdf", "ready", 1)])

    code = eval_cli_main([
        "migration-audit",
        "--bm25-db",
        str(bm25),
        "--registry-db",
        str(registry),
        "--chroma-path",
        str(tmp_path / "missing_chroma"),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 1
    assert payload["passed"] is False
    assert "FinQuery migration audit detected high-risk legacy data:" in captured.err


def test_eval_cli_migration_audit_warn_only_writes_report(tmp_path, capsys):
    bm25 = tmp_path / "bm25.db"
    registry = tmp_path / "registry.db"
    out = tmp_path / "migration.json"
    _create_bm25(bm25, [("legacy::1", None, "")])
    _create_registry(registry, [("doc1", 7, "report.pdf", "ready", 1)])

    code = eval_cli_main([
        "migration-audit",
        "--bm25-db",
        str(bm25),
        "--registry-db",
        str(registry),
        "--out",
        str(out),
        "--warn-only",
    ])

    captured = capsys.readouterr()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert written["summary"]["high_risk_count"] > 0
    assert json.loads(captured.out)["passed"] is False
