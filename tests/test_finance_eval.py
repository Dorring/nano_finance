import json

from nanochat.finance_eval import (
    evaluate_records,
    extraction_items,
    numeric_match,
    rouge_l,
    sentiment_label,
)
from nanochat.chat_format import encode_chat_prompt
from nanochat.dialogue_eval import evaluate_records as evaluate_dialogue_records
from nanochat.dialogue_eval import validate_examples as validate_dialogue_examples
from scripts.finance_generate import encode_prompt
from scripts.finance_compare import markdown_report, score_model
from scripts.dialogue_compare import (
    markdown_report as dialogue_markdown_report,
    score_model as score_dialogue_model,
)
from nanochat.training_metadata import (
    save_best_pointer,
    scale_initial_learning_rates,
)


class FakeTokenizer:
    def get_bos_token_id(self):
        return 1

    def encode_special(self, token):
        return {
            "<|user_start|>": 2,
            "<|user_end|>": 3,
            "<|assistant_start|>": 4,
            "<|assistant_end|>": 5,
        }[token]

    def encode(self, text):
        return list(range(10, 10 + len(text)))


def test_numeric_match_handles_percent_and_tolerance():
    assert numeric_match("53%", "0.53")
    assert numeric_match("53%", "53")
    assert numeric_match("100", "100.05", tolerance=1e-3)
    assert not numeric_match("100", "101", tolerance=1e-3)


def test_extraction_items_are_order_independent():
    left = '[{"entity":"Acme","type":"ORG"}]'
    right = json.dumps([{"type": "org", "entity": "acme"}])
    assert extraction_items(left) == extraction_items(right)


def test_rouge_l_bounds_and_identity():
    assert rouge_l("a b c", "a b c") == 1.0
    assert 0 < rouge_l("a b c", "a c") < 1
    assert rouge_l("a b", "x y") == 0.0


def test_sentiment_label_handles_structured_fiqa_answer():
    assert sentiment_label("情感极性：negative (分值: -0.374)") == "negative"


def test_evaluate_records_reports_task_specific_metrics():
    rows = [
        {"source": "finqa", "reference": "53%", "prediction": "0.53"},
        {
            "source": "finer",
            "reference": '[{"entity":"Acme","type":"ORG"}]',
            "prediction": '[{"type":"org","entity":"Acme"}]',
        },
        {"source": "finsen", "reference": "positive", "prediction": "positive"},
    ]
    report = evaluate_records(rows)
    assert report["count"] == 3
    assert report["tasks"]["numeric_qa"]["numeric_accuracy"] == 1.0
    assert report["tasks"]["entity_extraction"]["json_valid_rate"] == 1.0
    assert report["tasks"]["entity_extraction"]["micro_f1"] == 1.0
    assert report["tasks"]["sentiment"]["macro_f1"] == 1.0
    assert report["sources"]["finqa"]["count"] == 1
    assert report["macro_primary_score"] == 1.0


def test_encode_prompt_preserves_boundaries_and_question_tail():
    tokens = encode_prompt(
        FakeTokenizer(),
        [{"role": "user", "content": "abcdefghij"}],
        token_budget=10,
    )
    assert tokens[:2] == [1, 2]
    assert tokens[-2:] == [3, 4]
    assert len(tokens) == 10
    assert tokens[2:6] == [10, 11, 12, 13]
    assert tokens[6:8] == [18, 19]


def test_comparison_joins_predictions_by_id(tmp_path):
    examples = [
        {
            "id": "a",
            "split": "val",
            "source": "finsen",
            "task_type": "sentiment",
            "reference": "positive",
        },
        {
            "id": "b",
            "split": "val",
            "source": "finsen",
            "task_type": "sentiment",
            "reference": "negative",
        },
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "b", "prediction": "negative"}),
            json.dumps({"id": "a", "prediction": "positive"}),
        ]),
        encoding="utf-8",
    )
    report = score_model(examples, path)
    assert report["tasks"]["sentiment"]["accuracy"] == 1.0
    rendered = markdown_report({"model": report})
    assert "| Model | Empty outputs | Contentless outputs | Macro primary |" in rendered
    assert "| model | 0/2 (0.00%) | 0/2 (0.00%) |" in rendered


def test_evaluate_records_counts_empty_predictions():
    report = evaluate_records([
        {
            "source": "ectsum",
            "task_type": "summarization",
            "reference": "expected summary",
            "prediction": "",
        },
        {
            "source": "ectsum",
            "task_type": "summarization",
            "reference": "other summary",
            "prediction": "unrelated words",
        },
    ])
    assert report["tasks"]["summarization"]["rouge_l"] == 0.0
    assert report["empty_prediction_count"] == 1
    assert report["empty_prediction_rate"] == 0.5
    assert report["contentless_prediction_count"] == 1


def test_contentless_punctuation_is_not_reported_as_raw_empty():
    report = evaluate_records([
        {
            "source": "ectsum",
            "task_type": "summarization",
            "reference": "expected summary",
            "prediction": ".",
        },
    ])
    assert report["empty_prediction_count"] == 0
    assert report["contentless_prediction_count"] == 1


def test_save_best_pointer_is_complete_and_replaces_empty_file(tmp_path):
    pointer = tmp_path / "best.json"
    pointer.write_text("", encoding="utf-8")
    result = save_best_pointer(tmp_path, step=25, val_bpb=0.42)
    assert result == pointer
    assert json.loads(pointer.read_text(encoding="utf-8")) == {
        "step": 25,
        "val_bpb": 0.42,
        "checkpoint": "model_000025.pt",
    }
    assert not (tmp_path / "best.json.tmp").exists()


def test_scale_initial_learning_rates_updates_schedule_base_and_current_lr():
    groups = [
        {"initial_lr": 0.1, "lr": 0.0},
        {"initial_lr": 0.01, "lr": 0.0},
    ]
    scale_initial_learning_rates(groups, 0.25)
    assert groups == [
        {"initial_lr": 0.025, "lr": 0.025},
        {"initial_lr": 0.0025, "lr": 0.0025},
    ]


def test_encode_chat_prompt_merges_system_and_preserves_generation_boundary():
    tokens = encode_chat_prompt(
        FakeTokenizer(),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        token_budget=20,
    )
    assert tokens[:2] == [1, 2]
    assert tokens[-1] == 4
    assert tokens.count(4) == 1
    assert tokens[-2] == 3


def test_encode_chat_prompt_drops_leading_assistant_turn_after_truncation():
    tokens = encode_chat_prompt(
        FakeTokenizer(),
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ],
        token_budget=8,
    )
    assert tokens == [1, 2, 10, 3, 4]


def test_dialogue_eval_scores_rules_and_refusal():
    report = evaluate_dialogue_records([
        {
            "id": "a",
            "task_type": "finance_chat",
            "prediction": "无法确定，因为未提供足够上下文。",
            "expect_refusal": True,
            "expected_language": "zh",
            "forbidden_substrings": ["编造"],
        },
        {
            "id": "b",
            "task_type": "normal_chat",
            "prediction": '{"answer": "ok"}',
            "format": "json",
            "required_substrings": ["answer"],
        },
    ])
    assert report["count"] == 2
    assert report["empty_prediction_count"] == 0
    assert report["tasks"]["finance_chat"]["score"] == 1.0
    assert report["tasks"]["normal_chat"]["json_valid"] == 1.0


def test_dialogue_compare_joins_predictions_and_renders(tmp_path):
    examples = [
        {
            "id": "a",
            "split": "val",
            "task_type": "normal_chat",
            "messages": [{"role": "user", "content": "Say hi"}],
            "required_substrings": ["hi"],
        },
        {
            "id": "b",
            "split": "val",
            "task_type": "finance_chat",
            "messages": [{"role": "user", "content": "Need context"}],
            "expect_refusal": True,
        },
    ]
    path = tmp_path / "dialogue_predictions.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"id": "b", "prediction": "cannot answer without context"}),
            json.dumps({"id": "a", "prediction": "hi there"}),
        ]),
        encoding="utf-8",
    )
    report = score_dialogue_model(examples, path)
    assert report["macro_score"] == 1.0
    rendered = dialogue_markdown_report({"model": report})
    assert "| Model | Empty outputs | Contentless outputs | Macro score |" in rendered
    assert "finance_chat" in rendered


def test_dialogue_validation_rejects_mojibake_question_runs():
    rows = [
        {
            "id": "bad",
            "split": "val",
            "task_type": "normal_chat",
            "messages": [{"role": "user", "content": "????????"}],
            "required_substrings": ["??"],
            "reference": "????????",
        }
    ]
    try:
        validate_dialogue_examples(rows)
    except ValueError as error:
        assert (
            "suspicious question-mark run" in str(error)
            or "reference is contentless" in str(error)
        )
    else:
        raise AssertionError("expected dialogue validation to reject mojibake")


def test_dialogue_validation_rejects_contentless_reference():
    rows = [
        {
            "id": "bad",
            "split": "val",
            "task_type": "normal_chat",
            "messages": [{"role": "user", "content": "Say hello"}],
            "reference": "...",
        }
    ]
    try:
        validate_dialogue_examples(rows)
    except ValueError as error:
        assert "reference is contentless" in str(error)
    else:
        raise AssertionError("expected dialogue validation to reject empty reference")
