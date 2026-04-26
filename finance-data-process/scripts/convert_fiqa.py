import json
import pandas as pd
from pathlib import Path
from common import (
    INTERIM_CONVERTED_DIR, RAW_DIR, normalize_text, 
    truncate_text_by_tokens
)

def convert_fiqa():
    # FiQA Sentiment (Task 1)
    raw_path = RAW_DIR / "fiqa" / "fiqa_repo" / "data" / "train-00000-of-00001-aeefa1eadf5be10b.parquet"
    output_path = INTERIM_CONVERTED_DIR / "fiqa_sft.jsonl"
    
    if not raw_path.exists():
        print(f"Error: FiQA raw data not found at {raw_path}")
        return

    # SFT 模板 - 情感分析
    user_template = "请对以下金融文本（包含特定的目标主体）进行情感倾向与相关维度的分析。\n\n【文本】\n{text}\n【目标主体】\n{target}"
    
    BUDGET_USER = 1300
    BUDGET_ASSISTANT = 128
    
    df = pd.read_parquet(raw_path)
    stats = {"total": len(df), "success": 0, "dropped": 0}
    
    with output_path.open("w", encoding="utf-8") as out:
        for idx, row in df.iterrows():
            text = normalize_text(str(row.get("sentence", "")))
            target = normalize_text(str(row.get("target", "")))
            aspect = normalize_text(str(row.get("aspect", "")))
            score = str(row.get("score", ""))
            
            if not text or not target:
                stats["dropped"] += 1
                continue
            
            try:
                polarity = "negative" if float(score) < 0 else "positive" if float(score) > 0 else "neutral"
            except:
                polarity = "neutral"
                
            output_content = f"情感极性：{polarity} (分值: {score})\n分析维度：{aspect}"
            
            t_text = truncate_text_by_tokens(text, BUDGET_USER)
            prompt = user_template.format(text=t_text, target=target)
            
            out.write(json.dumps({
                "user": prompt,
                "assistant": output_content
            }, ensure_ascii=False) + "\n")
            
            stats["success"] += 1
                
    print(f"\n=== FiQA (Sentiment) Conversion Statistics ===")
    print(f"Total: {stats['total']}")
    print(f"Success: {stats['success']}")

if __name__ == "__main__":
    convert_fiqa()
