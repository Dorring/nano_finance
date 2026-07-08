"""Create JSON and Markdown comparisons from finance prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanochat.finance_eval import evaluate_records, primary_metric


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
        "# Finance validation comparison",
        "",
        "All results use the same validation IDs and deterministic decoding.",
        "",
        "| Model | Empty outputs | Contentless outputs | Macro primary | "
        + " | ".join(task_names) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(task_names),
    ]
    for name, report in reports.items():
        values = []
        for task in task_names:
            metrics = report["tasks"].get(task)
            values.append(
                f"{metrics[primary_metric(task)]:.4f}" if metrics else "—"
            )
        lines.append(
            f"| {name} | {report['empty_prediction_count']}/{report['count']} "
            f"({report['empty_prediction_rate']:.2%}) | "
            f"{report['contentless_prediction_count']}/{report['count']} "
            f"({report['contentless_prediction_rate']:.2%}) | "
            f"{report['macro_primary_score']:.4f} | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(["", "## Per-source primary metrics", ""])
    sources = sorted({
        source
        for report in reports.values()
        for source in report["sources"]
    })
    lines.extend([
        "| Source | " + " | ".join(reports) + " |",
        "|---|" + "---:|" * len(reports),
    ])
    for source in sources:
        values = []
        for report in reports.values():
            source_report = report["sources"].get(source)
            if not source_report:
                values.append("—")
                continue
            task_scores = [
                metrics[primary_metric(task)]
                for task, metrics in source_report["tasks"].items()
            ]
            values.append(f"{sum(task_scores) / len(task_scores):.4f}")
        lines.append(f"| {source} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


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
    if {row.get("split") for row in examples} != {"val"}:
        parser.error("comparison only accepts validation data")
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
