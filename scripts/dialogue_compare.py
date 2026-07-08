"""Create JSON and Markdown comparisons from dialogue prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanochat.dialogue_eval import evaluate_records, validate_examples


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_model(examples: list[dict], prediction_path: Path) -> dict:
    predictions = read_jsonl(prediction_path)
    prediction_by_id = {row["id"]: row["prediction"] for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError(f"duplicate IDs in {prediction_path}")
    expected_ids = {row["id"] for row in examples}
    if prediction_by_id.keys() != expected_ids:
        missing = len(expected_ids - prediction_by_id.keys())
        extra = len(prediction_by_id.keys() - expected_ids)
        raise ValueError(f"{prediction_path}: missing={missing}, extra={extra}")
    rows = [
        {**example, "prediction": prediction_by_id[example["id"]]}
        for example in examples
    ]
    return evaluate_records(rows)


def markdown_report(reports: dict[str, dict]) -> str:
    task_names = sorted({
        task
        for report in reports.values()
        for task in report["tasks"]
    })
    lines = [
        "# Dialogue validation comparison",
        "",
        "All results use the same validation IDs and deterministic decoding.",
        "",
        "| Model | Empty outputs | Contentless outputs | Macro score | "
        + " | ".join(task_names) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(task_names),
    ]
    for name, report in reports.items():
        values = []
        for task in task_names:
            metrics = report["tasks"].get(task)
            values.append(f"{metrics['score']:.4f}" if metrics else "—")
        lines.append(
            f"| {name} | {report['empty_prediction_count']}/{report['count']} "
            f"({report['empty_prediction_rate']:.2%}) | "
            f"{report['contentless_prediction_count']}/{report['count']} "
            f"({report['contentless_prediction_rate']:.2%}) | "
            f"{report['macro_score']:.4f} | "
            + " | ".join(values)
            + " |"
        )

    lines.extend(["", "## Per-task checks", ""])
    for task in task_names:
        check_names = sorted({
            check
            for report in reports.values()
            for check in report["tasks"].get(task, {})
            if check not in {"count", "score"}
        })
        if not check_names:
            continue
        lines.extend([
            f"### {task}",
            "",
            "| Model | Count | Score | " + " | ".join(check_names) + " |",
            "|---|---:|---:|" + "---:|" * len(check_names),
        ])
        for name, report in reports.items():
            metrics = report["tasks"].get(task)
            if not metrics:
                continue
            values = [f"{metrics[check]:.4f}" for check in check_names]
            lines.append(
                f"| {name} | {metrics['count']} | {metrics['score']:.4f} | "
                + " | ".join(values)
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=PATH; repeat once per model",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    examples = read_jsonl(args.eval_set)
    try:
        validate_examples(examples)
    except ValueError as error:
        parser.error(str(error))
    reports = {}
    for specification in args.prediction:
        name, separator, path = specification.partition("=")
        if not separator or not name or not path:
            parser.error(f"invalid --prediction {specification!r}")
        if name in reports:
            parser.error(f"duplicate model name {name!r}")
        reports[name] = score_model(examples, Path(path))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(markdown_report(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
