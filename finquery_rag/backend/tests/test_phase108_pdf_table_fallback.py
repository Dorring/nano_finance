import importlib
from types import SimpleNamespace


class _FakeDf:
    empty = False

    def __len__(self):
        return 1


def _load_process_tables(monkeypatch, read_pdf):
    monkeypatch.setitem(__import__("sys").modules, "camelot", SimpleNamespace(read_pdf=read_pdf))
    module = importlib.import_module("src.services.process_tables")
    return importlib.reload(module)


def test_camelot_failures_do_not_block_pdf_ingest(monkeypatch):
    def read_pdf(*args, **kwargs):
        raise RuntimeError("no table bbox")

    process_tables = _load_process_tables(monkeypatch, read_pdf)

    assert process_tables.extract_tables_with_camelot("sample.pdf") == {}


def test_table_without_bbox_is_kept_as_usable_table(monkeypatch):
    class TableWithoutBbox:
        page = "1"
        df = _FakeDf()

        @property
        def bbox(self):
            raise RuntimeError("no table bbox")

    def read_pdf(*args, **kwargs):
        if kwargs.get("flavor") == "stream":
            return [TableWithoutBbox()]
        return []

    process_tables = _load_process_tables(monkeypatch, read_pdf)
    monkeypatch.setattr(process_tables, "format_table", lambda table: "| a |\n|---|\n| 1 |")

    tables = process_tables.extract_tables_with_camelot("sample.pdf")

    assert tables == {1: [{"md": "| a |\n|---|\n| 1 |", "bbox": None}]}


def test_pymupdf_table_detection_failure_returns_empty_bboxes():
    from src.services.ingest import _safe_find_table_bboxes

    class Page:
        number = 0

        def find_tables(self):
            raise RuntimeError("no table bbox")

    assert _safe_find_table_bboxes(Page()) == []