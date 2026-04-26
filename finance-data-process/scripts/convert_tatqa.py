import json
from pathlib import Path
from common import (
    INTERIM_CLEANED_DIR, 
    INTERIM_CONVERTED_DIR, 
    write_jsonl, 
    count_tokens,
    get_tokenizer,
    truncate_text_by_tokens
)

def main():
    input_path = INTERIM_CLEANED_DIR / "tatqa_sft.jsonl"
    output_path = INTERIM_CONVERTED_DIR / "tatqa_sft.jsonl"
    
    if not input_path.exists():
        print(f"Error: {input_path} not found. Please ensure it exists.")
        return

    records = []
    tok = get_tokenizer()
    
    # 预算设定
    SAFE_LIMIT = 2020 
    USER_BUDGET = 1800
    ASST_BUDGET = 200
    
    processed_count = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            user_text = item["user"]
            asst_text = item["assistant"]
            
            u_len = count_tokens(user_text)
            a_len = count_tokens(asst_text)
            
            # 只有总量超标时才进行干预
            if (u_len + a_len) > SAFE_LIMIT:
                # 截断 User (使用 common.py 中的现成函数)
                if u_len > USER_BUDGET:
                    user_text = truncate_text_by_tokens(user_text, USER_BUDGET, head_ratio=0.8)
                
                # 截断 Assistant
                if a_len > ASST_BUDGET:
                    asst_text = truncate_text_by_tokens(asst_text, ASST_BUDGET, head_ratio=1.0)
            
            records.append({"user": user_text, "assistant": asst_text})
            processed_count += 1

    write_jsonl(records, output_path)
    print(f"[tatqa] Processed {processed_count} samples from cleaned to converted -> {output_path}")

if __name__ == "__main__":
    main()
