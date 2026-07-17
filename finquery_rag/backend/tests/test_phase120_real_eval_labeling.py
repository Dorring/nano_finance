import json

from scripts.real_eval_csv_to_jsonl import main as csv_to_jsonl_main


def test_real_eval_csv_converter_skips_placeholders_by_default(tmp_path, capsys):
    csv_path = tmp_path / "labels.csv"
    out_path = tmp_path / "cases.jsonl"
    csv_path.write_text(
        "id,question,document_names,expected_answer_contains,expected_numbers,expected_sources,expected_no_answer,expected_intent,tags\n"
        "placeholder,What is revenue?,report.pdf,REPLACE_WITH_REVENUE,,report.pdf:2,false,document_qa,\"real,number\"\n"
        "ready,What is the title?,report.pdf,Annual Report,,report.pdf:1,false,document_qa,\"real,front_matter\"\n",
        encoding="utf-8",
    )

    code = csv_to_jsonl_main(["--csv", str(csv_path), "--out", str(out_path)])

    payload = json.loads(capsys.readouterr().out)
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert code == 0
    assert payload["written_cases"] == 1
    assert payload["skipped"][0]["id"] == "placeholder"
    case = json.loads(lines[0])
    assert case["id"] == "ready"
    assert case["expected_sources"] == [{"filename": "report.pdf", "page": 1}]


def test_real_eval_csv_converter_handles_no_answer_rows(tmp_path):
    csv_path = tmp_path / "labels.csv"
    out_path = tmp_path / "cases.jsonl"
    csv_path.write_text(
        "id,question,document_names,expected_answer_contains,expected_numbers,expected_sources,expected_no_answer,expected_intent,tags\n"
        "no_answer,What is CEO salary?,report.pdf,,,,true,document_qa,\"real,no_answer\"\n",
        encoding="utf-8",
    )

    csv_to_jsonl_main(["--csv", str(csv_path), "--out", str(out_path)])

    case = json.loads(out_path.read_text(encoding="utf-8"))
    assert case["expected_no_answer"] is True
    assert "expected_sources" not in case
    assert "expected_answer_contains" not in case
