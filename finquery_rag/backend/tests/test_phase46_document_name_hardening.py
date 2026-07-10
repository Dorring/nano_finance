"""Document filename hardening regression checks."""
import os


ROOT = os.path.join(os.path.dirname(__file__), "..")
MAIN = os.path.join(ROOT, "src", "main.py")
API = os.path.join(ROOT, "..", "frontend", "src", "api.js")


def _main_content():
    return open(MAIN, encoding="utf-8").read()


def test_shared_document_filename_validator_static_contract():
    content = _main_content()
    helper = content[content.index("def _safe_document_filename"):content.index("def _safe_upload_filename")]

    assert "os.path.basename(filename).strip()" in helper
    assert "not isinstance(filename, str)" in helper
    assert "len(safe_filename) > 180" in helper
    assert "any(ord(ch) < 32 for ch in safe_filename)" in helper
    assert "require_pdf and not safe_filename.lower().endswith(\".pdf\")" in helper
    assert 'raise api_error(400, "invalid_filename", "Invalid filename")' in helper
    assert 'raise api_error(400, "invalid_file_type", "Only PDF files are supported")' in helper


def test_upload_and_delete_reuse_document_filename_validator_static():
    content = _main_content()
    upload_block = content[content.index("@app.post(\"/upload\""):content.index("@app.post(\"/query\"")]
    delete_block = content[content.index("@app.delete(\"/documents/{doc_name}\""):content.index("@app.delete(\"/documents\")")]

    assert "return _safe_document_filename(filename, require_pdf=True)" in content
    assert "safe_filename = _safe_upload_filename(file.filename)" in upload_block
    assert "doc_name = _safe_document_filename(doc_name, require_pdf=False)" in delete_block
    assert "delete_document_collection(doc_name, current_user.id)" in delete_block
    assert "document_registry.delete(current_user.id, doc_name)" in delete_block


def test_frontend_delete_document_url_encodes_filename():
    content = open(API, encoding="utf-8").read()

    assert "encodeURIComponent(docName)" in content
    assert "api.delete(`/documents/${docName}`)" not in content
