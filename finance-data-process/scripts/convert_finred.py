import json
import os
from pathlib import Path
from collections import Counter
from common import (
    INTERIM_CONVERTED_DIR, RAW_DIR, normalize_text, 
    truncate_text_by_tokens, count_tokens, ensure_parent
)

# --- 配置与规则定义 ---

# 1. 关系归并与白名单 (Point 2, 6)
RELATION_MAPPING = {
    "founded_by": "founded by",
    "chief_executive_officer": "ceo",
    "employer": "employer",
    "subsidiary": "subsidiary of",
    "owned_by": "owned by",
    "parent_organization": "parent organization",
    "headquarters_location": "headquarters location",
    "country_of_citizenship": "citizenship",
    "product_or_material_produced": "product or material produced",
    "industry": "industry",
    "developer": "developer",
    "manufacturer": "manufacturer",
    "member_of": "member of",
    "position_held": "position held",
    "place_of_birth": "place of birth",
}

# 2. 关系方向与实体类型启发式约束 (Point 4, 5, 9)
ORG_SUFFIXES = ("inc", "corp", "co", "ltd", "group", "bank", "holdings", "plc", "sa", "ag", "llc", "university", "foundation")
LOCATION_KEYWORDS = ("new york", "london", "paris", "tokyo", "china", "usa", "germany", "france", "california", "street", "city", "county")

def looks_like_org(text: str) -> bool:
    t = text.lower()
    return any(t.endswith(s) or f" {s} " in f" {t} " for s in ORG_SUFFIXES)

def looks_like_person(text: str) -> bool:
    if not text or len(text) < 3: return False
    words = text.split()
    # 简单启发式：首字母大写且通常包含空格
    is_capitalized = all(w[0].isupper() for w in words if w and w[0].isalpha())
    has_space = len(words) >= 2
    return is_capitalized and has_space and not looks_like_org(text)

def check_direction_and_type(head: str, rel: str, tail: str) -> bool:
    """验证关系的方向和实体类型逻辑 (Point 4, 5)"""
    rel_std = RELATION_MAPPING.get(rel, rel)
    
    if rel_std == "employer":
        if looks_like_org(head) and not looks_like_org(tail): 
            return False
            
    if rel_std == "headquarters location":
        if any(kw in head.lower() for kw in LOCATION_KEYWORDS) and not any(kw in tail.lower() for kw in LOCATION_KEYWORDS):
            return False

    return True

# 3. 低质量实体过滤 (Point 1, 8)
BAD_ENTITIES = {"investment", "software", "food", "match", "conglomerate", "retail", "industry", "product", "business", "company"}

def is_low_quality_entity(text: str) -> bool:
    t = text.lower().strip()
    if len(t) <= 2: return True 
    if t in BAD_ENTITIES: return True 
    if t.isdigit(): return True 
    return False

# --- 核心处理逻辑 ---

def parse_and_clean_triples(tup_line: str, stats: Counter, filtered_logs: list):
    if not tup_line.strip():
        return []
        
    raw_triples = tup_line.split(" | ")
    valid_triples = []
    seen_pairs = {} 
    
    for rt in raw_triples:
        parts = [p.strip() for p in rt.split(" ; ")]
        if len(parts) != 3: continue
        
        # 原始格式确认: head ; tail ; relation (根据 head -n 打印结果)
        # 注意：FinRED 的 train.tup 是 head ; tail ; relation 或者是 head ; relation ; tail?
        # 刚才 head 输出为: Apple Inc ; Steve Jobs ; founded_by
        # 这说明是 head ; tail ; relation
        head, tail, rel = parts
        
        if is_low_quality_entity(head) or is_low_quality_entity(tail):
            filtered_logs.append({"triple": rt, "reason": "low_quality_entity"})
            stats["filtered_low_quality_entity"] += 1
            continue
            
        if rel not in RELATION_MAPPING:
            filtered_logs.append({"triple": rt, "reason": "blacklisted_relation"})
            stats["filtered_blacklisted_relation"] += 1
            continue
        rel_std = RELATION_MAPPING[rel]
        
        if not check_direction_and_type(head, rel, tail):
            filtered_logs.append({"triple": rt, "reason": "bad_direction_or_type"})
            stats["filtered_direction"] += 1
            continue
            
        pair = (head, tail)
        if pair not in seen_pairs:
            seen_pairs[pair] = []
        
        if len(seen_pairs[pair]) >= 2:
            stats["filtered_redundant"] += 1
            continue
            
        if rel_std in seen_pairs[pair]:
            continue
            
        seen_pairs[pair].append(rel_std)
        valid_triples.append({
            "head": head,
            "relation": rel_std,
            "tail": tail,
            "_score": 2.0 if rel_std in ["ceo", "employer", "founded by"] else 1.0
        })
        
    return valid_triples

def convert_finred():
    raw_dir = RAW_DIR / "finred"
    output_path = INTERIM_CONVERTED_DIR / "finred_sft.jsonl"
    report_dir = INTERIM_CONVERTED_DIR.parent / "reports"
    log_path = report_dir / "finred_filtered_cases.jsonl"
    ensure_parent(log_path)
    
    sent_path = raw_dir / "train.sent"
    tup_path = raw_dir / "train.tup"
    
    if not sent_path.exists() or not tup_path.exists():
        print(f"Error: FinRED files not found in {raw_dir}")
        return

    user_template = (
        "请从以下商业/金融文本中抽取实体关系，并以 JSON 数组形式输出。"
        "每个关系包含 head、relation、tail 三个字段。\n\n"
        "【文本】\n{text}"
    )
    
    BUDGET_USER = 1400
    BUDGET_ASSISTANT = 512
    
    sent_lines = sent_path.read_text(encoding="utf-8").splitlines()
    tup_lines = tup_path.read_text(encoding="utf-8").splitlines()
    
    stats = Counter()
    records = []
    filtered_logs = []
    
    for sent, tup in zip(sent_lines, tup_lines):
        stats["total_raw"] += 1
        text = normalize_text(sent)
        if not text:
            stats["drop_empty_text"] += 1
            continue
            
        triples = parse_and_clean_triples(tup, stats, filtered_logs)
        
        if not triples:
            stats["drop_no_valid_triples"] += 1
            continue
            
        # 质量排序 (Point 12)
        triples.sort(key=lambda x: x["_score"], reverse=True)
        
        while triples and count_tokens(json.dumps([ {k:v for k,v in t.items() if k!="_score"} for t in triples], ensure_ascii=False)) > BUDGET_ASSISTANT:
            triples.pop()
            stats["truncated_assistant_triples"] += 1
            
        if not triples:
            stats["drop_no_valid_triples"] += 1
            continue

        clean_triples = [ {k:v for k,v in t.items() if k!="_score"} for t in triples ]
        assistant_json = json.dumps(clean_triples, ensure_ascii=False)

        t_text = truncate_text_by_tokens(text, BUDGET_USER)
        user_prompt = user_template.format(text=t_text)
        
        records.append({
            "user": user_prompt,
            "assistant": assistant_json
        })
        stats["success"] += 1
        stats["total_final_relations"] += len(clean_triples)

    with output_path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    with log_path.open("w", encoding="utf-8") as l:
        for entry in filtered_logs[:1000]: 
            l.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\n=== FinRED Enhanced Conversion Statistics ===")
    print(f"Total Raw Samples: {stats['total_raw']}")
    print(f"Successfully Converted: {stats['success']}")
    print(f"Dropped (No Quality Relations): {stats['drop_no_valid_triples']}")
    print(f"\n--- Filtering Breakdown ---")
    print(f"Filtered Low Quality Entities: {stats['filtered_low_quality_entity']}")
    print(f"Filtered Blacklisted Relations: {stats['filtered_blacklisted_relation']}")
    print(f"Filtered Bad Directions: {stats['filtered_direction']}")
    print(f"Filtered Redundant/Stacked: {stats['filtered_redundant']}")
    print(f"Final Total Relations: {stats['total_final_relations']}")
    
    if stats["success"] > 0:
        print(f"Avg Relations per Sample: {stats['total_final_relations'] / stats['success']:.2f}")

    print("\n=== Sample Output (First 2) ===")
    for i, rec in enumerate(records[:2]):
        print(f"--- Sample {i+1} ---")
        print(f"Assistant: {rec['assistant']}")

if __name__ == "__main__":
    convert_finred()
