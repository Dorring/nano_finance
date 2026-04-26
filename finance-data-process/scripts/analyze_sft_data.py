import json
from collections import defaultdict
from pathlib import Path
from common import PROCESSED_SFT_DIR, count_tokens

def calc_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0}
    lengths.sort()
    return {
        "avg": round(sum(lengths) / len(lengths), 1),
        "p50": lengths[len(lengths) // 2],
        "p95": lengths[int(len(lengths) * 0.95)],
        "max": lengths[-1],
    }

def main():
    data_path = PROCESSED_SFT_DIR / "finance_sft_v1.jsonl"
    metadata_path = PROCESSED_SFT_DIR / "metadata.json"
    
    if not data_path.exists() or not metadata_path.exists():
        print("Data or metadata not found. Run merge_sft_data.py first.")
        return

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    print("=== 1. Data Retention Stats (Samples) ===")
    for ds, stats in meta["dataset_stats"].items():
        print(f"{ds:10}: Total {stats['total']}, Valid {stats['valid']} ({(stats['valid']/stats['total']*100 if stats['total']>0 else 0):.1f}%)")
    
    user_lens = []
    assistant_lens = []
    total_lens = []
    
    print("\nProcessing tokens for length distribution (this may take a minute)...")
    with open(data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            u_len = count_tokens(item["user"])
            a_len = count_tokens(item["assistant"])
            user_lens.append(u_len)
            assistant_lens.append(a_len)
            total_lens.append(u_len + a_len)

    u_stats = calc_stats(user_lens)
    a_stats = calc_stats(assistant_lens)
    t_stats = calc_stats(total_lens)

    print("\n=== 2. Token Length Distribution Stats ===")
    print(f"User     : Avg: {u_stats['avg']}, P50: {u_stats['p50']}, P95: {u_stats['p95']}, Max: {u_stats['max']}")
    print(f"Assistant: Avg: {a_stats['avg']}, P50: {a_stats['p50']}, P95: {a_stats['p95']}, Max: {a_stats['max']}")
    print(f"Total    : Avg: {t_stats['avg']}, P50: {t_stats['p50']}, P95: {t_stats['p95']}, Max: {t_stats['max']}")
    
    over_limit = sum(1 for t in total_lens if t > 2048)
    print(f"\nSamples exceeding 2048 threshold: {over_limit}")

if __name__ == "__main__":
    main()
