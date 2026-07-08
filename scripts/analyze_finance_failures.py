"""Compare per-example failures across finance validation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanochat.finance_eval import (
    extraction_items,
    normalize_text,
    numeric_match,
    rouge_l,
    sentiment_label,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def example_score(example: dict, prediction: str) -> float:
    task = example["task_type"]
    reference = example["reference"]
    if task == "numeric_qa":
        return float(numeric_match(reference, prediction))
    if task in {"entity_extraction", "relation_extraction"}:
        expected = extraction_items(reference)
        actual = extraction_items(prediction)
        denominator = len(expected) + len(actual)
        return 2 * len(expected & actual) / denominator if denominator else 0.0
    if task == "sentiment":
        return float(sentiment_label(reference) == sentiment_label(prediction))
    if task == "summarization":
        return rouge_l(reference, prediction)
    if task == "instruction_following":
        return float(bool(normalize_text(prediction)))
    return float(normalize_text(reference) == normalize_text(prediction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="NAME=PATH; repeat once per model",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-task", type=int, default=5)
    args = parser.parse_args()
    examples = read_jsonl(args.eval_set)
    models = {}
    metadata = {}
    for specification in args.prediction:
        name, separator, path = specification.partition("=")
        if not separator:
            parser.error(f"invalid prediction specification: {specification}")
        rows = read_jsonl(Path(path))
        models[name] = {row["id"]: row["prediction"] for row in rows}
        metadata[name] = {row["id"]: row for row in rows}
    expected_ids = {row["id"] for row in examples}
    for name, predictions in models.items():
        if predictions.keys() != expected_ids:
            raise ValueError(f"{name} prediction IDs do not match evaluation IDs")

    lines = [
        "# Finance validation failure analysis",
        "",
        "## Completion behavior",
        "",
        "| Model | Empty | Hit token limit | Average completion tokens |",
        "|---|---:|---:|---:|",
    ]
    for name, rows in metadata.items():
        values = list(rows.values())
        empty = sum(not str(row["prediction"]).strip() for row in values)
        hit_limit = sum(row.get("completion_tokens") == 256 for row in values)
        average = sum(row.get("completion_tokens", 0) for row in values) / len(values)
        lines.append(f"| {name} | {empty} | {hit_limit} | {average:.1f} |")

    tasks = sorted({row["task_type"] for row in examples})
    for task in tasks:
        task_rows = [row for row in examples if row["task_type"] == task]
        lines.extend(["", f"## {task}", ""])
        wins = {name: 0 for name in models}
        scored = []
        for example in task_rows:
            scores = {
                name: example_score(example, predictions[example["id"]])
                for name, predictions in models.items()
            }
            best = max(scores.values())
            for name, score in scores.items():
                wins[name] += score == best and best > 0
            scored.append((max(scores.values()), example, scores))
        lines.append(
            "Best-score counts: "
            + ", ".join(f"{name}={count}" for name, count in wins.items())
        )
        lines.extend(["", "Representative hard failures:", ""])
        for _, example, scores in sorted(scored, key=lambda item: item[0])[
            : args.examples_per_task
        ]:
            question = example["messages"][0]["content"].splitlines()[-1]
            lines.append(
                f"- `{example['id']}` ({example['source']}): "
                f"{question[:160]} | "
                + ", ".join(f"{name}={score:.3f}" for name, score in scores.items())
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
