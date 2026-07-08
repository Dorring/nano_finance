import argparse
import hashlib
import json

from common import (
    INTERIM_CONVERTED_DIR,
    PROCESSED_SFT_DIR,
    count_tokens,
    read_jsonl,
)


DATASETS = [
    "finqa",
    "tatqa",
    "ectsum",
    "finer",
    "finred",
    "fiqa",
    "finsen",
    "finance_r1",
]
CONTEXT_GROUPED_DATASETS = {"finqa", "tatqa"}


def get_pair(record):
    if isinstance(record, list):
        if len(record) != 2:
            raise ValueError("Message-array records must contain one user/assistant pair")
        return record[0]["content"], record[1]["content"]
    return record["user"], record["assistant"]


def get_group_hash(dataset, user, record_hash):
    """Keep questions about the same financial context in one split."""
    if dataset in CONTEXT_GROUPED_DATASETS:
        context = user.split("\n\n【问题】\n", 1)[0]
        group_key = f"{dataset}:{context}"
    else:
        group_key = f"{dataset}:{record_hash}"
    return hashlib.sha256(group_key.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-limit", type=int, default=2020)
    args = parser.parse_args()

    all_records = []
    seen = set()
    stats = {}

    for dataset in DATASETS:
        path = INTERIM_CONVERTED_DIR / f"{dataset}_sft.jsonl"
        records = read_jsonl(path)
        valid = []
        skipped = 0
        duplicates = 0

        for record in records:
            user, assistant = get_pair(record)
            total_len = count_tokens(user) + count_tokens(assistant)
            if total_len > args.safe_limit:
                skipped += 1
                continue

            record_hash = hashlib.sha256(
                json.dumps([user, assistant], ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if record_hash in seen:
                duplicates += 1
                continue

            seen.add(record_hash)
            valid.append({
                "user": user,
                "assistant": assistant,
                "_source": dataset,
                "_record_hash": record_hash,
                "_group_hash": get_group_hash(dataset, user, record_hash),
            })

        all_records.extend(valid)
        stats[dataset] = {
            "total": len(records),
            "valid": len(valid),
            "skipped": skipped,
            "duplicates": duplicates,
        }
        print(
            f"{dataset}: total={len(records)} valid={len(valid)} "
            f"skipped={skipped} duplicates={duplicates}"
        )

    output_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "version": "v2",
        "total_samples": len(all_records),
        "dataset_stats": stats,
        "safe_limit": args.safe_limit,
    }
    with (PROCESSED_SFT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(all_records)} unique samples -> {output_path}")


if __name__ == "__main__":
    main()
