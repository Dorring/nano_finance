"""Score ID-keyed predictions for finance_eval_small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanochat.finance_eval import evaluate_records


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--numeric-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    examples = read_jsonl(args.eval_set)
    if not examples:
        parser.error("evaluation set is empty")
    splits = {row.get("split") for row in examples}
    if splits != {"val"}:
        parser.error(f"only validation evaluation is allowed here, got splits={splits}")
    predictions = read_jsonl(args.predictions)
    prediction_by_id = {row["id"]: row["prediction"] for row in predictions}
    if len(prediction_by_id) != len(predictions):
        parser.error("prediction IDs must be unique")
    expected_ids = {row["id"] for row in examples}
    missing = expected_ids - prediction_by_id.keys()
    extra = prediction_by_id.keys() - expected_ids
    if missing or extra:
        parser.error(f"prediction ID mismatch: missing={len(missing)}, extra={len(extra)}")

    scored = [
        {**example, "prediction": prediction_by_id[example["id"]]}
        for example in examples
    ]
    report = evaluate_records(scored, tolerance=args.numeric_tolerance)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
