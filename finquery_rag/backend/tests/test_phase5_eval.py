"""Phase 5 tests: offline evaluation and replay layer."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.evaluation import (
    EvaluationCase,
    Prediction,
    evaluate_predictions,
    export_replay_cases_from_traces,
    load_jsonl_cases,
    load_jsonl_predictions,
    score_prediction,
    trace_to_replay_case,
)


def test_score_prediction_requires_expected_phrase_and_source():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "expected_sources": [{"filename": "q3.pdf", "page": 2}],
        "expected_answer_contains": ["$10M"],
    })
    pred = Prediction.from_dict({
        "id": "c1",
        "answer": "Revenue was $10M.",
        "sources": [{"filename": "q3.pdf", "page": 2}],
    })

    score = score_prediction(case, pred)

    assert score["passed"] is True
    assert score["citation_recall"] == 1.0
    assert score["answer_contains"] == 1.0


def test_score_prediction_fails_missing_source():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "expected_sources": [{"filename": "q3.pdf", "page": 2}],
    })
    pred = Prediction.from_dict({
        "id": "c1",
        "answer": "Revenue was $10M.",
        "sources": [{"filename": "q3.pdf", "page": 3}],
    })

    score = score_prediction(case, pred)

    assert score["passed"] is False
    assert score["citation_recall"] == 0.0


def test_score_prediction_matches_scoped_doc_id_filename():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "expected_sources": [{"filename": "report.pdf", "page": 1}],
    })
    pred = Prediction.from_dict({
        "id": "c1",
        "answer": "Revenue was $10M.",
        "retrieved_chunks": [{"doc_id": "user_7_report.pdf::0001", "page": 1}],
    })

    score = score_prediction(case, pred)

    assert score["retrieval_recall"] == 1.0


def test_number_accuracy_normalizes_commas():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was revenue?",
        "expected_numbers": ["1000000", "12.5%"],
    })
    pred = Prediction.from_dict({
        "id": "c1",
        "answer": "Revenue was 1,000,000 and margin was 12.5%.",
    })

    score = score_prediction(case, pred)

    assert score["number_accuracy"] == 1.0
    assert score["passed"] is True


def test_no_answer_case_requires_refusal_marker():
    case = EvaluationCase.from_dict({
        "id": "c1",
        "question": "What was 2099 revenue?",
        "expected_no_answer": True,
    })
    bad = Prediction.from_dict({"id": "c1", "answer": "Revenue was $10M."})
    good = Prediction.from_dict({"id": "c1", "answer": "I couldn't find relevant information."})

    assert score_prediction(case, bad)["passed"] is False
    assert score_prediction(case, good)["passed"] is True


def test_evaluate_predictions_aggregates_and_reports_missing():
    cases = [
        EvaluationCase.from_dict({"id": "c1", "question": "Q1", "expected_answer_contains": ["A"]}),
        EvaluationCase.from_dict({"id": "c2", "question": "Q2", "expected_answer_contains": ["B"]}),
    ]
    predictions = {
        "c1": Prediction.from_dict({"id": "c1", "answer": "A", "latency_ms": 100}),
    }

    report = evaluate_predictions(cases, predictions)

    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["scored_cases"] == 1
    assert report["summary"]["missing_predictions"] == 1
    assert report["missing_case_ids"] == ["c2"]
    assert report["summary"]["p95_latency_ms"] == 100


def test_load_jsonl_cases_and_predictions(tmp_path):
    case_path = tmp_path / "cases.jsonl"
    pred_path = tmp_path / "preds.jsonl"
    case_path.write_text(json.dumps({"id": "c1", "question": "Q"}) + "\n", encoding="utf-8")
    pred_path.write_text(json.dumps({"id": "c1", "answer": "A"}) + "\n", encoding="utf-8")

    cases = load_jsonl_cases(case_path)
    predictions = load_jsonl_predictions(pred_path)

    assert cases[0].case_id == "c1"
    assert predictions["c1"].answer == "A"


def test_trace_to_replay_case_excludes_context_and_keeps_sources():
    trace = {
        "trace_id": "t1",
        "tenant_id": 9,
        "query_original": "What was revenue?",
        "filter_conditions": json.dumps({"doc_names": ["q3.pdf"]}),
        "sources_json": json.dumps([{"filename": "q3.pdf", "page": 2}]),
        "final_context": "sensitive document body",
        "answer": "Revenue was $10M.",
        "model_name": "nanochat",
        "created_at": 123.0,
    }

    case = trace_to_replay_case(trace)
    payload = case.to_dict()

    assert case.case_id == "t1"
    assert case.document_names == ("q3.pdf",)
    assert payload["expected_sources"][0]["filename"] == "q3.pdf"
    assert "final_context" not in json.dumps(payload)
    assert case.metadata["tenant_id"] == 9


def test_export_replay_cases_from_traces(tmp_path):
    output = tmp_path / "replay.jsonl"
    traces = [{
        "trace_id": "t1",
        "tenant_id": 1,
        "query_original": "Q",
        "filter_conditions": "{}",
        "sources_json": "[]",
        "answer": "I couldn't find relevant information.",
    }]

    cases = export_replay_cases_from_traces(traces, output)

    assert len(cases) == 1
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["id"] == "t1"


def test_invalid_case_requires_question():
    try:
        EvaluationCase.from_dict({"id": "bad"})
    except ValueError as exc:
        assert "missing question" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_score_prediction_validates_expected_calculation():
    case = EvaluationCase.from_dict({
        "id": "calc1",
        "question": "What was revenue growth?",
        "expected_calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "args": {"current": "120", "previous": "100"},
            "expected_value": "0.2000",
            "tolerance": "0.0001",
            "unit": "ratio",
        }],
    })
    pred = Prediction.from_dict({
        "id": "calc1",
        "answer": "Revenue grew 20%.",
        "calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "value": "0.2000",
            "unit": "ratio",
        }],
    })

    score = score_prediction(case, pred)

    assert score["calculation_accuracy"] == 1.0
    assert score["passed"] is True


def test_score_prediction_fails_missing_calculation():
    case = EvaluationCase.from_dict({
        "id": "calc1",
        "question": "What was revenue growth?",
        "expected_calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "args": {"current": "120", "previous": "100"},
            "expected_value": "0.2000",
        }],
    })
    pred = Prediction.from_dict({"id": "calc1", "answer": "Revenue grew 20%."})

    score = score_prediction(case, pred)

    assert score["calculation_accuracy"] == 0.0
    assert score["passed"] is False


def test_evaluate_predictions_includes_calculation_accuracy():
    cases = [EvaluationCase.from_dict({
        "id": "calc1",
        "question": "What is share?",
        "expected_calculations": [{
            "id": "share",
            "operation": "percentage_share",
            "args": {"part": "25", "total": "100"},
            "expected_value": "0.2500",
        }],
    })]
    predictions = {
        "calc1": Prediction.from_dict({
            "id": "calc1",
            "answer": "Share was 25%.",
            "calculations": [{"id": "share", "operation": "percentage_share", "value": "0.2500"}],
        })
    }

    report = evaluate_predictions(cases, predictions)

    assert report["summary"]["calculation_accuracy"] == 1.0


def test_score_prediction_checks_answer_calculation_consistency():
    case = EvaluationCase.from_dict({
        "id": "calc1",
        "question": "What was revenue growth?",
        "expected_calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "args": {"current": "120", "previous": "100"},
            "expected_value": "0.2000",
            "tolerance": "0.0001",
            "unit": "ratio",
        }],
    })
    pred = Prediction.from_dict({
        "id": "calc1",
        "answer": "Revenue grew 30%.",
        "calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "value": "0.2000",
            "unit": "ratio",
        }],
    })

    score = score_prediction(case, pred)

    assert score["calculation_accuracy"] == 1.0
    assert score["answer_calculation_consistency"] == 0.0
    assert score["passed"] is False


def test_score_prediction_no_calculations_keeps_consistency_neutral():
    case = EvaluationCase.from_dict({"id": "c1", "question": "Q"})
    pred = Prediction.from_dict({"id": "c1", "answer": "No calculation."})

    score = score_prediction(case, pred)

    assert score["answer_calculation_consistency"] == 1.0
    assert score["passed"] is True


def test_evaluate_predictions_includes_answer_calculation_consistency():
    cases = [EvaluationCase.from_dict({
        "id": "calc1",
        "question": "What was growth?",
        "expected_calculations": [{
            "id": "growth",
            "operation": "growth_rate",
            "args": {"current": "120", "previous": "100"},
            "expected_value": "0.2000",
            "unit": "ratio",
        }],
    })]
    predictions = {
        "calc1": Prediction.from_dict({
            "id": "calc1",
            "answer": "Growth was 20%.",
            "calculations": [{"id": "growth", "operation": "growth_rate", "value": "0.2000", "unit": "ratio"}],
        })
    }

    report = evaluate_predictions(cases, predictions)

    assert report["summary"]["answer_calculation_consistency"] == 1.0



def test_score_prediction_validates_expected_intent():
    case = EvaluationCase.from_dict({
        "id": "intent1",
        "question": "What was revenue?",
        "expected_intent": "document_qa",
    })
    good = Prediction.from_dict({
        "id": "intent1",
        "answer": "Revenue was $10M.",
        "intent": "document_qa",
        "intent_confidence": 0.82,
    })
    bad = Prediction.from_dict({
        "id": "intent1",
        "answer": "Revenue was $10M.",
        "intent": "conversation",
    })

    assert score_prediction(case, good)["passed"] is True
    bad_score = score_prediction(case, bad)
    assert bad_score["passed"] is False
    assert bad_score["intent_accuracy"] == 0.0
    assert bad_score["expected_intent"] == "document_qa"
    assert bad_score["predicted_intent"] == "conversation"


def test_evaluate_predictions_includes_intent_accuracy():
    cases = [
        EvaluationCase.from_dict({
            "id": "intent1",
            "question": "Calculate growth",
            "expected_intent": "financial_calculation",
        })
    ]
    predictions = {
        "intent1": Prediction.from_dict({
            "id": "intent1",
            "answer": "Growth was 20%.",
            "intent": "financial_calculation",
        })
    }

    report = evaluate_predictions(cases, predictions)

    assert report["summary"]["intent_accuracy"] == 1.0


def test_trace_to_replay_case_keeps_intent_when_available():
    case = trace_to_replay_case({
        "trace_id": "t-intent",
        "tenant_id": 1,
        "query_original": "What was revenue?",
        "filter_conditions": "{}",
        "sources_json": "[]",
        "answer": "Revenue was $10M.",
        "intent": "document_qa",
    })

    assert case.expected_intent == "document_qa"
    assert case.to_dict()["expected_intent"] == "document_qa"



def test_load_jsonl_cases_reports_line_number_for_schema_errors(tmp_path):
    case_path = tmp_path / "bad_cases.jsonl"
    case_path.write_text(
        json.dumps({"id": "ok", "question": "Q"}) + "\n"
        + json.dumps({"id": "bad", "expected_sources": []}) + "\n",
        encoding="utf-8",
    )

    try:
        load_jsonl_cases(case_path)
    except ValueError as exc:
        message = str(exc)
        assert "bad_cases.jsonl:2" in message
        assert "missing question" in message
    else:
        raise AssertionError("expected schema error")


def test_load_jsonl_cases_rejects_duplicate_ids(tmp_path):
    case_path = tmp_path / "cases.jsonl"
    case_path.write_text(
        json.dumps({"id": "dup", "question": "Q1"}) + "\n"
        + json.dumps({"id": "dup", "question": "Q2"}) + "\n",
        encoding="utf-8",
    )

    try:
        load_jsonl_cases(case_path)
    except ValueError as exc:
        assert "duplicate evaluation case id 'dup'" in str(exc)
        assert "cases.jsonl:2" in str(exc)
    else:
        raise AssertionError("expected duplicate id error")


def test_load_jsonl_predictions_rejects_duplicate_ids(tmp_path):
    pred_path = tmp_path / "preds.jsonl"
    pred_path.write_text(
        json.dumps({"id": "dup", "answer": "A1"}) + "\n"
        + json.dumps({"id": "dup", "answer": "A2"}) + "\n",
        encoding="utf-8",
    )

    try:
        load_jsonl_predictions(pred_path)
    except ValueError as exc:
        assert "duplicate prediction id 'dup'" in str(exc)
        assert "preds.jsonl:2" in str(exc)
    else:
        raise AssertionError("expected duplicate id error")


def test_case_schema_rejects_non_list_fields():
    try:
        EvaluationCase.from_dict({
            "id": "bad",
            "question": "Q",
            "expected_sources": {"filename": "x.pdf"},
        })
    except ValueError as exc:
        assert "field expected_sources must be a list" in str(exc)
    else:
        raise AssertionError("expected field type error")


def test_prediction_schema_rejects_non_object_sources():
    try:
        Prediction.from_dict({"id": "bad", "answer": "A", "sources": ["not-object"]})
    except ValueError as exc:
        assert "field sources[0] must be an object" in str(exc)
    else:
        raise AssertionError("expected source item type error")
