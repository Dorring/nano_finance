from __future__ import annotations
import argparse
from pathlib import Path
from common import (
    INTERIM_CONVERTED_DIR, 
    RAW_DIR, 
    normalize_text, 
    write_jsonl, 
    load_config, 
    truncate_summary_by_budget
)

def convert_dir(base: Path, split: str, config: dict) -> list[dict]:
    ects_dir = base / split / "ects"
    gt_dir = base / split / "gt_summaries"
    if not ects_dir.exists() or not gt_dir.exists():
        return []
        
    records = []
    budget = config.get("budgets", {}).get("summary", {"user_tokens": 1536, "assistant_tokens": 512})
    
    # 遍历 ects 目录下的所有 txt 文件
    for ect_file in ects_dir.glob("*.txt"):
        gt_file = gt_dir / ect_file.name
        if not gt_file.exists():
            continue
            
        text = normalize_text(ect_file.read_text(encoding="utf-8", errors="ignore"))
        summary = normalize_text(gt_file.read_text(encoding="utf-8", errors="ignore"))
        
        if not text or not summary:
            continue
            
        user = f"请为以下金融财报电话会议文本生成摘要：\n\n{text}"
        # 使用 summary 类型的截断逻辑
        final_user, final_assistant = truncate_summary_by_budget(
            user, summary, budget["user_tokens"], budget["assistant_tokens"]
        )
        
        records.append({
            "user": final_user,
            "assistant": final_assistant
        })
    return records

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=INTERIM_CONVERTED_DIR / "ectsum_sft.jsonl")
    args = parser.parse_args()
    
    config = load_config()
    base = RAW_DIR / "ectsum" / "ECTSum_repo" / "data" / "final"
    
    records = []
    for split in ["train", "val", "test"]:
        split_records = convert_dir(base, split, config)
        print(f"[ectsum] found {len(split_records)} samples in {split}")
        records.extend(split_records)
        
    count = write_jsonl(records, args.output)
    print(f"[ectsum] total wrote {count} samples -> {args.output}")

if __name__ == "__main__":
    main()
