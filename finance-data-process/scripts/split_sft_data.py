import json
import random
from pathlib import Path
from common import PROCESSED_SFT_DIR, write_jsonl

def main():
    input_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)

    total = len(records)
    n_val = int(total * 0.1)
    n_test = int(total * 0.1)
    n_train = total - n_val - n_test

    train_data = records[:n_train]
    val_data = records[n_train:n_train + n_val]
    test_data = records[n_train + n_val:]

    write_jsonl(train_data, PROCESSED_SFT_DIR / "train.jsonl")
    write_jsonl(val_data, PROCESSED_SFT_DIR / "val.jsonl")
    write_jsonl(test_data, PROCESSED_SFT_DIR / "test.jsonl")

    print(f"[split] Total: {total}")
    print(f"[split] train={len(train_data)} val={len(val_data)} test={len(test_data)}")

if __name__ == "__main__":
    main()
