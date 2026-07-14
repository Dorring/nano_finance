"""Upload input hardening regression checks."""
import os


ROOT = os.path.join(os.path.dirname(__file__), "..")
MAIN = os.path.join(ROOT, "src", "main.py")


def _main_content():
    return open(MAIN, encoding="utf-8").read()


def _upload_block():
    content = _main_content()
    return content[content.index("@app.post(\"/upload\""):content.index("@app.post(\"/query\"")]


def test_safe_upload_filename_static_contract():
    content = _main_content()
    helper = content[content.index("def _safe_document_filename"):content.index("def _safe_upload_filename")]
    upload_helper = content[content.index("def _safe_upload_filename"):content.index("def _assistant_session_metadata")]

    assert "os.path.basename(filename).strip()" in helper
    assert "not isinstance(filename, str)" in helper
    assert "len(safe_filename) > 180" in helper
    assert "any(ord(ch) < 32 for ch in safe_filename)" in helper
    assert 'require_pdf and not safe_filename.lower().endswith(".pdf")' in helper
    assert 'raise api_error(400, "invalid_filename", "Invalid filename")' in helper
    assert 'raise api_error(400, "invalid_file_type", "Only PDF files are supported")' in helper
    assert "return _safe_document_filename(filename, require_pdf=True)" in upload_helper


def test_upload_endpoint_uses_sanitized_filename_static():
    upload_block = _upload_block()

    assert "safe_filename = _safe_upload_filename(file.filename)" in upload_block
    assert "temp_path = os.path.join(temp_dir, safe_filename)" in upload_block
    assert "document_registry.register(" in upload_block
    assert "safe_filename, fh, status=\"parsing\"" in upload_block
    assert "add_documents(chunks, safe_filename, current_user.id, no_of_pages)" in upload_block
    assert "filename=safe_filename" in upload_block
    assert "Successfully processed %s\" % safe_filename" in upload_block
    assert "file.filename.endswith" not in upload_block
    assert "os.path.basename(file.filename)" not in upload_block


def test_upload_duplicate_responses_use_sanitized_filename_static():
    upload_block = _upload_block()

    assert upload_block.count("filename=safe_filename") >= 3
    assert "filename=file.filename" not in upload_block
