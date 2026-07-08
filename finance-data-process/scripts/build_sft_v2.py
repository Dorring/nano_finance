"""Build a deterministic SFT v2 train set balanced by assistant tokens."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import PROCESSED_SFT_DIR, ROOT, count_tokens, read_jsonl, write_jsonl


DEFAULT_CONFIG = ROOT / "configs" / "sft_v2_balanced.json"


def message_key(messages: list[dict]) -> tuple[str, str]:
    by_role = {message["role"]: message["content"] for message in messages}
    return by_role["user"], by_role["assistant"]


def load_train_records(master_path: Path) -> dict[str, list[dict]]:
    train_keys = set()
    for filename in ("train.jsonl", "cot_train.jsonl"):
        train_keys.update(
            message_key(row)
            for row in read_jsonl(PROCESSED_SFT_DIR / filename)
        )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in read_jsonl(master_path):
        if (record["user"], record["assistant"]) in train_keys:
            grouped[record["_source"]].append(record)
    return grouped


def sample_to_token_budget(
    records: list[dict],
    token_budget: int,
    seed: int,
) -> tuple[list[dict], int]:
    if not records:
        raise ValueError("cannot sample from an empty source")
    lengths = [count_tokens(record["assistant"]) for record in records]
    if any(length < 1 for length in lengths):
        raise ValueError("assistant responses must contain at least one token")
    output = []
    assistant_tokens = 0
    cycle = 0
    while assistant_tokens < token_budget:
        indices = list(range(len(records)))
        random.Random(f"{seed}:{cycle}").shuffle(indices)
        for index in indices:
            record = records[index]
            output.append([
                {"role": "user", "content": record["user"]},
                {"role": "assistant", "content": record["assistant"]},
            ])
            assistant_tokens += lengths[index]
            if assistant_tokens >= token_budget:
                break
        cycle += 1
    return output, assistant_tokens


def build_dataset(
    grouped: dict[str, list[dict]],
    config: dict,
    seed: int,
) -> tuple[list[list[dict]], dict]:
    shares = config["source_shares"]
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError("source_shares must sum to 1.0")
    unknown = set(shares) - grouped.keys()
    if unknown:
        raise ValueError(f"configured sources missing from train data: {sorted(unknown)}")
    total_budget = int(config["total_assistant_tokens"])
    if total_budget < 1:
        raise ValueError("total_assistant_tokens must be positive")

    output = []
    source_stats = {}
    for source, share in shares.items():
        target = round(total_budget * share)
        sampled, actual = sample_to_token_budget(
            grouped[source],
            target,
            seed + sum(map(ord, source)),
        )
        output.extend(sampled)
        source_stats[source] = {
            "unique_train_rows": len(grouped[source]),
            "output_rows": len(sampled),
            "average_repeats": round(len(sampled) / len(grouped[source]), 3),
            "target_assistant_tokens": target,
            "actual_assistant_tokens": actual,
            "actual_share": 0.0,
        }
    random.Random(seed).shuffle(output)
    actual_total = sum(
        stats["actual_assistant_tokens"] for stats in source_stats.values()
    )
    for stats in source_stats.values():
        stats["actual_share"] = stats["actual_assistant_tokens"] / actual_total
    manifest = {
        "version": "sft_v2",
        "seed": seed,
        "config": config,
        "output_rows": len(output),
        "actual_assistant_tokens": actual_total,
        "sources": source_stats,
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--master",
        type=Path,
        default=PROCESSED_SFT_DIR / "finance_sft_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_SFT_DIR / "train_v2_balanced.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROCESSED_SFT_DIR / "train_v2_balanced_manifest.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output, manifest = build_dataset(
        load_train_records(args.master),
        config,
        args.seed,
    )
    write_jsonl(output, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
