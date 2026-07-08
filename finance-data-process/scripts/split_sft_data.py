import json
import random
from collections import defaultdict

from common import PROCESSED_SFT_DIR, write_jsonl


SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.1
COT_SOURCE = "finance_r1"


def to_messages(record):
    return [
        {"role": "user", "content": record["user"]},
        {"role": "assistant", "content": record["assistant"]},
    ]


def split_grouped(records, source):
    """Split whole context groups while keeping source ratios near 80/10/10."""
    groups = defaultdict(list)
    for record in records:
        groups[record["_group_hash"]].append(record)

    grouped_records = list(groups.values())
    random.Random(f"{SEED}:{source}").shuffle(grouped_records)
    # Place larger groups first; the previous shuffle provides deterministic ties.
    grouped_records.sort(key=len, reverse=True)

    total = len(records)
    targets = {
        "train": total * (1 - VAL_RATIO - TEST_RATIO),
        "val": total * VAL_RATIO,
        "test": total * TEST_RATIO,
    }
    splits = {"train": [], "val": [], "test": []}
    counts = {name: 0 for name in splits}

    for group in grouped_records:
        destination = min(
            splits,
            key=lambda name: counts[name] / max(targets[name], 1),
        )
        splits[destination].extend(group)
        counts[destination] += len(group)
    return splits, len(grouped_records)


def main():
    input_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found; run merge_sft_data.py")

    records_by_source = {}
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            source = record.get("_source")
            if not source:
                raise ValueError("Missing _source; rerun merge_sft_data.py")
            records_by_source.setdefault(source, []).append(record)

    train_data, cot_train_data, val_data, test_data = [], [], [], []
    source_stats = {}

    for source, records in sorted(records_by_source.items()):
        source_splits, num_groups = split_grouped(records, source)
        source_train = source_splits["train"]
        source_val = source_splits["val"]
        source_test = source_splits["test"]

        destination = cot_train_data if source == COT_SOURCE else train_data
        destination.extend(map(to_messages, source_train))
        val_data.extend(map(to_messages, source_val))
        test_data.extend(map(to_messages, source_test))
        source_stats[source] = {
            "train": len(source_train),
            "val": len(source_val),
            "test": len(source_test),
            "groups": num_groups,
        }

    random.Random(SEED).shuffle(train_data)
    random.Random(SEED + 1).shuffle(cot_train_data)
    random.Random(SEED + 2).shuffle(val_data)
    random.Random(SEED + 3).shuffle(test_data)

    write_jsonl(train_data, PROCESSED_SFT_DIR / "train.jsonl")
    write_jsonl(cot_train_data, PROCESSED_SFT_DIR / "cot_train.jsonl")
    write_jsonl(val_data, PROCESSED_SFT_DIR / "val.jsonl")
    write_jsonl(test_data, PROCESSED_SFT_DIR / "test.jsonl")

    manifest = {
        "seed": SEED,
        "ratios": {"train": 0.8, "val": VAL_RATIO, "test": TEST_RATIO},
        "cot_source": COT_SOURCE,
        "sources": source_stats,
        "counts": {
            "train": len(train_data),
            "cot_train": len(cot_train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
    }
    with (PROCESSED_SFT_DIR / "split_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(
        f"[split] train={len(train_data)} cot_train={len(cot_train_data)} "
        f"val={len(val_data)} test={len(test_data)}"
    )


if __name__ == "__main__":
    main()
