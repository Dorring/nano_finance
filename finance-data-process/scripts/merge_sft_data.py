import json
import argparse
from pathlib import Path
from common import PROCESSED_SFT_DIR, INTERIM_CONVERTED_DIR, read_jsonl, count_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-limit", type=int, default=2020)
    args = parser.parse_args()
    
    datasets = ["finqa", "tatqa", "ectsum", "finer", "finred", "fiqa", "finsen"]
    all_records = []
    stats = {}
    
    for ds in datasets:
        path = INTERIM_CONVERTED_DIR / f"{ds}_sft.jsonl"
        records = read_jsonl(path)
        print(f"Loading {len(records)} samples from {ds}...")
        
        valid = []
        skipped = 0
        for r in records:
            # 这里的 len(tokens) 应该包含 user + assistant
            total_len = count_tokens(r["user"]) + count_tokens(r["assistant"])
            if total_len <= args.safe_limit:
                valid.append(r)
            else:
                skipped += 1
        
        all_records.extend(valid)
        stats[ds] = {"total": len(records), "valid": len(valid), "skipped": skipped}
    
    output_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    # 保存 metadata.json 供 analyze 脚本使用
    metadata = {
        "version": "v1",
        "total_samples": len(all_records),
        "dataset_stats": stats,
        "safe_limit": args.safe_limit
    }
    with open(PROCESSED_SFT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Merged {len(all_records)} samples -> {output_path}")

if __name__ == "__main__":
    main()
