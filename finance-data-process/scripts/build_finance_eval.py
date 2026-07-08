"""Build a deterministic, source-balanced financial evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import PROCESSED_SFT_DIR, read_jsonl, write_jsonl


# Eight known sources produce 200 rows by default, within the plan's 100-300
# sample target while retaining equal source representation.
DEFAULT_PER_SOURCE = 25
DEFAULT_SEED = 42
NUMERIC_ANSWER_RE = re.compile(
    r"^\s*[$(+-]?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*"
    r"(?:thousand|million|billion|trillion)?\s*\)?\s*$",
    re.IGNORECASE,
)


def message_key(messages: list[dict]) -> tuple[str, str]:
    by_role = {message["role"]: message["content"] for message in messages}
    return by_role["user"], by_role["assistant"]


def task_for_source(source: str) -> str:
    mapping = {
        "finqa": "numeric_qa",
        "tatqa": "table_qa",
        "finer": "entity_extraction",
        "finred": "relation_extraction",
        "finsen": "sentiment",
        "fiqa": "sentiment",
        "ectsum": "summarization",
        "finance_r1": "instruction_following",
    }
    if source not in mapping:
        raise ValueError(f"Add a task mapping for source {source!r}")
    return mapping[source]


def task_for_record(record: dict) -> str:
    task = task_for_source(record["_source"])
    if (
        record["_source"] == "tatqa"
        and NUMERIC_ANSWER_RE.fullmatch(record["assistant"])
    ):
        return "numeric_qa"
    return task


def build_subset(
    source_records: list[dict],
    split_records: list[dict],
    per_source: int,
    seed: int,
    split: str = "val",
) -> list[dict]:
    split_keys = {message_key(record) for record in split_records}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for record in source_records:
        if (record["user"], record["assistant"]) in split_keys:
            candidates[record["_source"]].append(record)

    output = []
    for source in sorted(candidates):
        records = candidates[source]
        random.Random(f"{seed}:{source}").shuffle(records)
        for record in records[:per_source]:
            stable_id = hashlib.sha256(
                f"{source}:{record['_record_hash']}".encode()
            ).hexdigest()[:16]
            output.append(
                {
                    "id": stable_id,
                    "split": split,
                    "source": source,
                    "task_type": task_for_record(record),
                    "messages": [{"role": "user", "content": record["user"]}],
                    "reference": record["assistant"],
                    "record_hash": record["_record_hash"],
                    "group_hash": record["_group_hash"],
                }
            )
    random.Random(seed).shuffle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--per-source", type=int, default=DEFAULT_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    if args.split == "test" and not args.allow_test:
        parser.error("sealed test access requires --allow-test")

    source_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    split_path = PROCESSED_SFT_DIR / f"{args.split}.jsonl"
    output_path = args.output or PROCESSED_SFT_DIR / f"finance_eval_small_{args.split}.jsonl"
    rows = build_subset(
        read_jsonl(source_path),
        read_jsonl(split_path),
        args.per_source,
        args.seed,
        args.split,
    )
    count = write_jsonl(rows, output_path)
    source_counts = Counter(row["source"] for row in rows)
    print(json.dumps({"output": str(output_path), "count": count, "sources": source_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
