"""Phase 1 tests: Document Registry and Trace Logger."""
import os
import sys
import time
import sqlite3
import json
import tempfile
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.document_registry import DocumentRegistry, validate_transition, VALID_TRANSITIONS, SCHEMA_VERSION
from services.trace import TraceLogger


class TestStateMachine:
    def test_valid_transitions(self):
        validate_transition("pending", "parsing")
        validate_transition("pending", "failed")
        validate_transition("parsing", "indexing")
        validate_transition("parsing", "failed")
        validate_transition("indexing", "ready")
        validate_transition("indexing", "failed")
        validate_transition("failed", "pending")

    def test_invalid_transition_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition("ready", "pending")
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition("pending", "ready")

    def test_unknown_state_raises(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown state"):
            validate_transition("nonexistent", "pending")

    def test_all_states_covered(self):
        expected = {"pending", "parsing", "indexing", "ready", "failed"}
        assert set(VALID_TRANSITIONS.keys()) == expected


class TestRegistrySchema:
    def test_schema_version(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            conn = sqlite3.connect(path)
            ver = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            conn.close()
            assert ver[0] == SCHEMA_VERSION
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_idempotent_init(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg1 = DocumentRegistry(db_path=path)
            reg2 = DocumentRegistry(db_path=path)
            conn = sqlite3.connect(path)
            count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            conn.close()
            assert count == 1
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryHashing:
    def test_file_hash_deterministic(self):
        data = b"test pdf content"
        h1 = DocumentRegistry.file_hash(data)
        h2 = DocumentRegistry.file_hash(data)
        assert h1 == h2
        assert len(h1) == 64

    def test_file_hash_different_for_different_data(self):
        h1 = DocumentRegistry.file_hash(b"content A")
        h2 = DocumentRegistry.file_hash(b"content B")
        assert h1 != h2

    def test_content_hash_deterministic(self):
        chunks = [
            {"content": "hello", "metadata": {"doc_id": "a"}},
            {"content": "world", "metadata": {"doc_id": "b"}},
        ]
        h1 = DocumentRegistry.content_hash(chunks)
        h2 = DocumentRegistry.content_hash(chunks)
        assert h1 == h2

    def test_content_hash_order_independent(self):
        chunks_a = [
            {"content": "hello", "metadata": {"doc_id": "a"}},
            {"content": "world", "metadata": {"doc_id": "b"}},
        ]
        chunks_b = [
            {"content": "world", "metadata": {"doc_id": "b"}},
            {"content": "hello", "metadata": {"doc_id": "a"}},
        ]
        assert DocumentRegistry.content_hash(chunks_a) == DocumentRegistry.content_hash(chunks_b)


class TestRegistryDedup:
    def test_find_by_file_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            fh = DocumentRegistry.file_hash(b"test")
            reg.register("doc1", 1, "test.pdf", fh, status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_indexing("doc1")
            reg.mark_ready("doc1", 10, "abc123")
            found = reg.find_by_file_hash(1, fh)
            assert found is not None
            assert found["document_id"] == "doc1"
            assert reg.find_by_file_hash(2, fh) is None
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_find_by_content_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "a.pdf", "fhash1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_indexing("doc1")
            ch = DocumentRegistry.content_hash([{"content": "x", "metadata": {"doc_id": "1"}}])
            reg.mark_ready("doc1", 5, ch)
            found = reg.find_by_content_hash(1, ch)
            assert found is not None
            assert found["document_id"] == "doc1"
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_pending_not_found_by_hash(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            fh = DocumentRegistry.file_hash(b"test")
            reg.register("doc1", 1, "test.pdf", fh, status="pending")
            assert reg.find_by_file_hash(1, fh) is None
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryVersioning:
    def test_version_increments(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            v1 = reg.register("doc1", 1, "test.pdf", "hash1")
            v2 = reg.register("doc2", 1, "test.pdf", "hash2")
            assert v1 == 1
            assert v2 == 2
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_different_files_independent_versions(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            v1 = reg.register("doc1", 1, "a.pdf", "hash1")
            v2 = reg.register("doc2", 1, "b.pdf", "hash2")
            assert v1 == 1
            assert v2 == 1
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryDelete:
    def test_delete_by_tenant_and_filename(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "test.pdf", "h1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_indexing("doc1")
            reg.mark_ready("doc1", 5, "ch1")
            count = reg.delete(1, "test.pdf")
            assert count == 1
            assert reg.find_by_file_hash(1, "h1") is None
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_delete_all_for_tenant(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "a.pdf", "h1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_indexing("doc1")
            reg.mark_ready("doc1", 5, "ch1")
            reg.register("doc2", 1, "b.pdf", "h2", status="pending")
            reg.transition("doc2", "parsing")
            reg.mark_indexing("doc2")
            reg.mark_ready("doc2", 3, "ch2")
            count = reg.delete_all_for_tenant(1)
            assert count == 2
            assert len(reg.list_documents(1)) == 0
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryLifecycle:
    def test_full_lifecycle(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            doc_id = "lifecycle_test"
            fh = DocumentRegistry.file_hash(b"pdf bytes")
            reg.register(doc_id, 1, "report.pdf", fh)
            assert reg.get_latest_version(1, "report.pdf")["status"] == "pending"
            reg.transition(doc_id, "parsing")
            assert reg.get_latest_version(1, "report.pdf")["status"] == "parsing"
            reg.mark_indexing(doc_id)
            assert reg.get_latest_version(1, "report.pdf")["status"] == "indexing"
            ch = DocumentRegistry.content_hash([{"content": "x", "metadata": {"doc_id": "1"}}])
            reg.mark_ready(doc_id, 10, ch)
            entry = reg.get_latest_version(1, "report.pdf")
            assert entry["status"] == "ready"
            assert entry["chunk_count"] == 10
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_failed_and_retry(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "test.pdf", "h1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_failed("doc1", "Parse error")
            entry = reg.get_latest_version(1, "test.pdf")
            assert entry["status"] == "failed"
            assert entry["error_message"] == "Parse error"
            reg.transition("doc1", "pending")
            entry = reg.get_latest_version(1, "test.pdf")
            assert entry["status"] == "pending"
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestTraceLoggerSchema:
    def test_schema_creates_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path)
            conn = sqlite3.connect(path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn.close()
            assert "trace_log" in tables
            assert "schema_version" in tables
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestTraceLogging:
    def test_log_and_retrieve(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, redact_content=False)
            trace_id = tl.log(
                tenant_id=1,
                query_original="What is revenue?",
                query_rewritten="What was the total revenue for Q3?",
                intent="fact_query",
                latency_ms=150.5,
                model_name="nanochat",
            )
            assert trace_id is not None
            trace = tl.get_trace(trace_id)
            assert trace is not None
            assert trace["tenant_id"] == 1
            assert trace["query_original"] == "What is revenue?"
            assert trace["latency_ms"] == 150.5
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_log_with_candidates(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, redact_content=False)
            candidates = [{"doc_id": "c1", "score": 0.95}, {"doc_id": "c2", "score": 0.80}]
            trace_id = tl.log(tenant_id=1, query_original="test", candidates=candidates)
            trace = tl.get_trace(trace_id)
            parsed = json.loads(trace["candidates_json"])
            assert len(parsed) == 2
            assert parsed[0]["score"] == 0.95
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_get_recent(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, redact_content=False)
            for i in range(5):
                tl.log(tenant_id=1, query_original="query %d" % i)
            tl.log(tenant_id=2, query_original="other tenant")
            recent = tl.get_recent(1, limit=3)
            assert len(recent) == 3
            for t in recent:
                assert t["tenant_id"] == 1
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestTraceSanitization:
    def test_redacts_phone_numbers(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, redact_content=True)
            trace_id = tl.log(tenant_id=1, query_original="Call 123-456-7890 for info")
            trace = tl.get_trace(trace_id)
            assert "123-456-7890" not in trace["query_original"]
            assert "[REDACTED]" in trace["query_original"]
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_no_redact_when_disabled(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, redact_content=False)
            trace_id = tl.log(tenant_id=1, query_original="Call 123-456-7890 for info")
            trace = tl.get_trace(trace_id)
            assert "123-456-7890" in trace["query_original"]
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_sanitize_static(self):
        result = TraceLogger.sanitize("My phone is 555-123-4567")
        assert "[REDACTED]" in result
        assert "555-123-4567" not in result


class TestTraceSampling:
    def test_sample_rate_zero_skips(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, sample_rate=0.0)
            tid = tl.log(tenant_id=1, query_original="test")
            assert tid is None
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_sample_rate_one_logs(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            tl = TraceLogger(db_path=path, sample_rate=1.0, redact_content=False)
            tid = tl.log(tenant_id=1, query_original="test")
            assert tid is not None
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryListDocuments:
    def test_list_only_ready(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "a.pdf", "h1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_indexing("doc1")
            ch = DocumentRegistry.content_hash([{"content": "x", "metadata": {"doc_id": "1"}}])
            reg.mark_ready("doc1", 5, ch)
            reg.register("doc2", 1, "b.pdf", "h2", status="pending")
            docs = reg.list_documents(1)
            assert len(docs) == 1
            assert docs[0]["filename"] == "a.pdf"
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass

    def test_empty_for_no_tenant(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            assert len(reg.list_documents(999)) == 0
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass


class TestRegistryRetry:
    def test_get_pending_for_retry(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = f.name
        try:
            reg = DocumentRegistry(db_path=path)
            reg.register("doc1", 1, "a.pdf", "h1", status="pending")
            reg.transition("doc1", "parsing")
            reg.mark_failed("doc1", "error")
            failed = reg.get_pending_for_retry(1)
            assert len(failed) == 1
            assert failed[0]["document_id"] == "doc1"
        finally:
            gc.collect()
            try:
                os.unlink(path)
            except PermissionError:
                time.sleep(0.05)
                gc.collect()
                try:
                    os.unlink(path)
                except PermissionError:
                    pass
