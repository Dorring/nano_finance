from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from collections import Counter
from common import (
    INTERIM_CONVERTED_DIR, 
    RAW_DIR, 
    normalize_text, 
    write_jsonl, 
    load_config,
    linearize_table,
    truncate_text_by_tokens,
    count_tokens
)

def parse_item(item: dict) -> dict | None:
    """统一解析步骤，从不同层级提取字段"""
    # 1. 提取 Question
    qa = item.get("qa", {})
    question = None
    if isinstance(qa, dict):
        question = qa.get("question")
    if not question:
        question = item.get("question")
    
    # 2. 提取 Answer
    answer = None
    if isinstance(qa, dict):
        answer = qa.get("answer")
    if not answer:
        answer = item.get("answer")
    
    if not question or not answer or str(answer).strip() == "" or answer == "No answer provided":
        return None

    # 3. 提取 Table
    table_raw = item.get("table") or item.get("table_data")
    if not table_raw:
        return None
    
    # 4. 提取 Context (pre_text + post_text)
    pre = item.get("pre_text", [])
    post = item.get("post_text", [])
    if isinstance(pre, str): pre = [pre]
    if isinstance(post, str): post = [post]
    
    context_paras = pre + post
    if not context_paras and not table_raw:
        return None

    return {
        "question": normalize_text(str(question)),
        "answer": normalize_text(str(answer)),
        "table_raw": table_raw,
        "context_paras": [normalize_text(str(p)) for p in context_paras if str(p).strip()]
    }

def build_user_prompt(question: str, table_text: str, context_text: str) -> str:
    """显式使用 f-string 构造最终 User 文本"""
    prompt = (
        "请根据给定的财务表格和文本回答问题。\n\n"
        f"【表格】\n{table_text}\n\n"
        f"【相关文本】\n{context_text}\n\n"
        f"【问题】\n{question}"
    )
    return prompt

def advanced_truncate_qa(parsed: dict, user_budget: int) -> str:
    """段落级裁剪逻辑"""
    question = parsed["question"]
    table_text = linearize_table(parsed["table_raw"])
    
    test_prompt = build_user_prompt(question, "TABLE_PLACEHOLDER", "CONTEXT_PLACEHOLDER")
    fixed_len = count_tokens(test_prompt)
    
    available_budget = user_budget - fixed_len - 20
    if available_budget < 100: available_budget = 100
    
    table_budget = int(available_budget * 0.5)
    context_budget = available_budget - table_budget
    
    final_table = truncate_text_by_tokens(table_text, table_budget)
    
    current_context_list = []
    current_tokens = 0
    for para in parsed["context_paras"]:
        p_tokens = count_tokens(para)
        if current_tokens + p_tokens + 2 < context_budget:
            current_context_list.append(para)
            current_tokens += p_tokens + 2
        else:
            if not current_context_list:
                current_context_list.append(truncate_text_by_tokens(para, context_budget))
            break
    
    final_context = "\n".join(current_context_list)
    return build_user_prompt(question, final_table, final_context)

def convert_split(base: Path, split: str, config: dict, stats: Counter) -> list[dict]:
    path = base / f"{split}.json"
    if not path.exists():
        return []
    
    budget = config.get("budgets", {}).get("qa", {"user_tokens": 1536, "assistant_tokens": 512})
    records = []
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
        
    stats["total"] += len(data)
    for item in data:
        parsed = parse_item(item)
        if not parsed:
            stats["drop_missing_fields"] += 1
            continue
            
        user = advanced_truncate_qa(parsed, budget["user_tokens"])
        placeholders = ["{context}", "{table}", "{question}"]
        if any(p in user for p in placeholders):
            stats["drop_placeholder_detected"] += 1
            continue
            
        records.append({"user": user, "assistant": parsed["answer"]})
        stats["success"] += 1
    return records

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=INTERIM_CONVERTED_DIR / "finqa_sft.jsonl")
    args = parser.parse_args()
    config = load_config()
    base = RAW_DIR / "finqa" / "FinQA_repo" / "dataset"
    stats = Counter()
    all_records = []
    for split in ["train", "dev", "test"]:
        all_records.extend(convert_split(base, split, config, stats))
    if not all_records:
        print("No records converted.")
        return
    count = write_jsonl(all_records, args.output)
    print(f"\n=== FinQA Conversion Statistics ===")
    print(f"Total Raw Samples: {stats['total']}")
    print(f"Successfully Converted: {stats['success']}")
    print(f"Dropped (Missing Q/A/Table/Empty): {stats['drop_missing_fields']}")
    print(f"Dropped (Placeholder Detected): {stats['drop_placeholder_detected']}")
    print(f"Final Saved: {count}\n")
    print("=== Sample Output (First 3) ===")
    for i, s in enumerate(all_records[:3]):
        print(f"--- Sample {i+1} ---\nUser: {s['user'][:250]}...\nAssistant: {s['assistant']}\n")

if __name__ == "__main__":
    main()
