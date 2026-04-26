from __future__ import annotations
import argparse
import csv
import json
import re
import random
from collections import defaultdict, Counter
from pathlib import Path
from common import (
    INTERIM_CONVERTED_DIR, 
    RAW_DIR, 
    normalize_text, 
    write_jsonl,
    load_config,
    truncate_assistant_by_budget,
    count_tokens
)

LABEL_MAP = {"0": "O", "1": "PER_B", "2": "PER_I", "3": "LOC_B", "4": "LOC_I", "5": "ORG_B", "6": "ORG_I"}
TYPE_MAP = {"PER": "人物", "LOC": "地点", "ORG": "机构"}

def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%\)])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return normalize_text(text)

def normalize_entity_text(text: str) -> str:
    # 修正连字符空格: sub - Saharan -> sub-Saharan, Mercedes - Benz -> Mercedes-Benz
    text = re.sub(r"(\w)\s*-\s*(\w)", r"\1-\2", text)
    return normalize_text(text)

def extract_financial_entities(text: str) -> list[dict]:
    financial_entities = []
    
    # 金额: $800 million, Ksh38 billion, £10.5bn, $500,000, 500 million euros
    # 包含了常见的货币符号和扩展的缩写
    money_regex = r"(?:[\$£€¥]|Ksh|Rs\.?|AED)\s?\d+(?:[.,]\d+)?(?:\s?(?:million|billion|trillion|m|bn|tn|thousand))?|\b\d+(?:[.,]\d+)?\s+(?:million|billion|trillion|m|bn|tn)\s+(?:dollars|euros|pounds|shillings|yen)\b"
    # 百分比: 63 per cent, 5.3%, 0.5 percentage points
    percent_regex = r"\b\d+(?:[.,]\d+)?\s?(?:%|per\s?cent|percentage points)\b"
    # 时间: 2007-2014, 10 years, Q3 2023, fiscal 2024
    time_regex = r"\b(?:19|20)\d{2}(?:-(?:19|20)\d{2})?\b|\b\d+\s?years?\b|\b[Qq][1-4]\s\d{4}\b|\bfiscal\s\d{4}\b"
    # 数量: 69,600 jobs, 17,500 shares, 5,000 units
    count_regex = r"\b\d{1,3}(?:,\d{3})+\b\s+(?:jobs|shares|tons|units|people|workers|employees)\b"

    patterns = [
        (money_regex, "金额"),
        (percent_regex, "百分比"),
        (time_regex, "时间"),
        (count_regex, "数量")
    ]

    for pattern, label in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            financial_entities.append({
                "entity": match.group(),
                "type": label,
                "start": match.start()
            })
    return financial_entities

def refine_entities(text: str, raw_entities: list[dict]) -> list[dict]:
    seen = set()
    cleaned = []
    for ent in raw_entities:
        ent_text = normalize_entity_text(ent["entity"])
        if not ent_text or len(ent_text) < 1: continue
        
        # 获取或查找位置用于排序
        start_pos = ent.get("start", text.find(ent["entity"]))
        
        # 按 (text.lower, type) 去重
        key = (ent_text.lower(), ent["type"])
        if key not in seen:
            cleaned.append({"entity": ent_text, "type": ent["type"], "start": start_pos})
            seen.add(key)
    
    # 按原文顺序排列
    cleaned.sort(key=lambda x: x["start"])
    return [{"entity": e["entity"], "type": e["type"]} for e in cleaned]

def extract_original_entities_with_pos(tokens: list[str], labels: list[str]) -> list[dict]:
    entities = []
    cur_tokens, cur_type = [], ""
    # 计算当前 tokens 在 detokenized 字符串中的大概位置是不准确的，
    # 简单起见，我们记录 token 的 index 作为排序依据
    for idx, (tok, raw_label) in enumerate(zip(tokens, labels)):
        label = LABEL_MAP.get(raw_label, "O")
        if label == "O":
            if cur_tokens:
                entities.append({
                    "entity": detokenize([t for t, i in cur_tokens]), 
                    "type": TYPE_MAP.get(cur_type, cur_type),
                    "start": cur_tokens[0][1] * 5 # 粗略线性表示，用于内部相对排序
                })
                cur_tokens, cur_type = [], ""
            continue
        tag_type, tag_pos = label.split("_")
        if tag_pos == "B" or tag_type != cur_type:
            if cur_tokens:
                entities.append({
                    "entity": detokenize([t for t, i in cur_tokens]), 
                    "type": TYPE_MAP.get(cur_type, cur_type),
                    "start": cur_tokens[0][1] * 5
                })
            cur_tokens, cur_type = [(tok, idx)], tag_type
        else:
            cur_tokens.append((tok, idx))
    if cur_tokens:
        entities.append({
            "entity": detokenize([t for t, i in cur_tokens]), 
            "type": TYPE_MAP.get(cur_type, cur_type),
            "start": cur_tokens[0][1] * 5
        })
    return entities

def convert_csv(path: Path, config: dict, stats: Counter) -> list[dict]:
    grouped = defaultdict(lambda: {"tokens": [], "labels": []})
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped[(row["doc_idx"], row["sent_idx"])]["tokens"].append(row["gold_token"])
            grouped[(row["doc_idx"], row["sent_idx"])]["labels"].append(row["gold_label"])
    
    budget = config.get("budgets", {}).get("ner", {"user_tokens": 1400, "assistant_tokens": 512})
    records = []
    for v in grouped.values():
        sentence = detokenize(v["tokens"])
        if not sentence: continue
        
        # 1. 提取原始实体
        orig_entities = extract_original_entities_with_pos(v["tokens"], v["labels"])
        stats["original_entities"] += len(orig_entities)
        
        # 2. 补标金融实体
        fin_entities = extract_financial_entities(sentence)
        stats["augmented_entities"] += len(fin_entities)
        
        # 3. 汇总去重与排序
        final_entities = refine_entities(sentence, orig_entities + fin_entities)
        
        for e in final_entities:
            stats[f"type_{e['type']}"] += 1

        # 4. 空样本与低价值过滤
        if not final_entities:
            stats["empty_samples_before_filter"] += 1
            # 过滤逻辑：如果文本太短且无实体，或者随机 80% 过滤
            if len(sentence) < 40 or random.random() < 0.8:
                stats["filtered_samples"] += 1
                continue
        
        user = f"请识别以下文本中的金融实体，并以 JSON 数组形式输出，每个元素包含 entity 和 type 两个字段。\n\n【文本】\n{sentence}"
        if count_tokens(user) > budget["user_tokens"]: 
            stats["too_long_samples"] += 1
            continue
            
        assistant = json.dumps(final_entities, ensure_ascii=False)
        assistant = truncate_assistant_by_budget(assistant, budget["assistant_tokens"])
        
        records.append({"user": user, "assistant": assistant})
    return records

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=INTERIM_CONVERTED_DIR / "finer_sft.jsonl")
    args = parser.parse_args()
    config = load_config()
    base = RAW_DIR / "finer" / "finer-ord_repo"
    
    stats = Counter()
    records = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        split_path = base / split
        if split_path.exists():
            print(f"Processing {split}...")
            records.extend(convert_csv(split_path, config, stats))
    
    count = write_jsonl(records, args.output)
    
    print("\n=== FiNER Refinement Statistics ===")
    print(f"Original Entities Found: {stats['original_entities']}")
    print(f"Augmented Financial Entities: {stats['augmented_entities']}")
    print(f"Samples Filtered (Empty/Low Info): {stats['filtered_samples']}")
    print(f"Final Samples Saved: {count}")
    print("\nEntity Type Distribution:")
    for k, v in sorted(stats.items()):
        if k.startswith("type_"):
            print(f"  - {k[5:]}: {v}")
    print("===================================\n")

if __name__ == "__main__":
    main()
