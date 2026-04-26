import json
import pandas as pd
from pathlib import Path
from common import (
    INTERIM_CONVERTED_DIR, RAW_DIR, normalize_text, 
    truncate_text_by_tokens
)

def convert_finsen():
    raw_dir = RAW_DIR / "finsen"
    output_path = INTERIM_CONVERTED_DIR / "finsen_sft.jsonl"
    
    files = list(raw_dir.glob("**/*.parquet"))
    if not files:
        print(f"Error: No parquet files found in {raw_dir}")
        return

    user_template = "请对以下金融新闻或推文进行情感倾向分析，将其分类为：positive (积极)、negative (消极) 或 neutral (中性)。\n\n【文本】\n{text}"
    
    BUDGET_USER = 1300
    
    total_success = 0
    total_raw = 0

    with output_path.open("w", encoding="utf-8") as out:
        for fpath in files:
            df = pd.read_parquet(fpath)
            total_raw += len(df)
            for _, row in df.iterrows():
                text = normalize_text(str(row.get("sentence", row.get("text", ""))))
                label = normalize_text(str(row.get("label", row.get("sentiment", ""))))
                
                if not text or not label:
                    continue
                
                label_map = {"0": "negative", "1": "neutral", "2": "positive"}
                label = label_map.get(label, label)
                
                t_text = truncate_text_by_tokens(text, BUDGET_USER)
                prompt = user_template.format(text=t_text)
                
                out.write(json.dumps({
                    "user": prompt,
                    "assistant": label
                }, ensure_ascii=False) + "\n")
                total_success += 1
                
    print(f"\n=== FinSen Conversion Statistics ===")
    print(f"Total: {total_raw}")
    print(f"Success: {total_success}")

if __name__ == "__main__":
    convert_finsen()
