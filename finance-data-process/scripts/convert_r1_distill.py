"""
Finance R1-Distill CoT 数据转换脚本

将 DeepSeek-R1 蒸馏的金融推理思维链数据转换为标准 SFT 格式:
  - user_input -> user
  - reasoning_content + answer_r1 -> assistant (带 <think> 标签)
  - 超过 SAFE_LIMIT tokens 的样本直接丢弃 (零容忍截断)

用法:
    python convert_r1_distill.py
    python convert_r1_distill.py --safe-limit 2000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
from common import (
    INTERIM_CONVERTED_DIR,
    write_jsonl,
    count_tokens,
    normalize_text,
)

# 默认输入目录 (R1 蒸馏数据)
R1_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "deepseekr1sft"

# 安全线: 2048 窗口 - 48 (special tokens 余量)
DEFAULT_SAFE_LIMIT = 2000


def convert_r1_data(input_dir: Path, safe_limit: int, stats: Counter) -> list[dict]:
    """转换所有 R1 蒸馏 JSONL 文件"""
    all_records = []
    jsonl_files = sorted(input_dir.glob("*.jsonl"))

    if not jsonl_files:
        print(f"错误: 在 {input_dir} 中未找到 JSONL 文件!")
        return []

    for jsonl_file in jsonl_files:
        print(f"  处理: {jsonl_file.name}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    stats["drop_json_error"] += 1
                    continue

                stats["total"] += 1

                # 1. 提取原始字段 (处理 None 值)
                user_text = normalize_text(item.get("user_input") or "")
                reasoning = (item.get("reasoning_content") or "").strip()
                answer = (item.get("answer_r1") or "").strip()

                # 2. 空值检查
                if not user_text:
                    stats["drop_empty_user"] += 1
                    continue
                if not reasoning and not answer:
                    stats["drop_empty_response"] += 1
                    continue

                # 3. 拼装带 <think> 标签的 assistant 回复
                if reasoning and answer:
                    assistant_text = f"<think>\n{reasoning}\n</think>\n\n{answer}"
                elif answer:
                    assistant_text = answer
                else:
                    assistant_text = f"<think>\n{reasoning}\n</think>\n"

                # 4. Token 长度检查 (零容忍: 超限直接丢弃,绝不截断)
                simulated_text = f"<|user_start|>{user_text}<|user_end|><|assistant_start|>{assistant_text}<|assistant_end|>"
                token_count = count_tokens(simulated_text)

                stats["token_sum"] += token_count
                if token_count > stats.get("max_tokens", 0):
                    stats["max_tokens"] = token_count

                if token_count > safe_limit:
                    stats["drop_too_long"] += 1
                    continue

                # 5. 输出标准 SFT 格式
                records_item = {
                    "user": user_text,
                    "assistant": assistant_text,
                }
                all_records.append(records_item)
                stats["kept"] += 1

    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Finance R1-Distill CoT data to SFT format")
    parser.add_argument("--input-dir", type=Path, default=R1_DATA_DIR,
                        help="R1 蒸馏数据目录")
    parser.add_argument("--output", type=Path,
                        default=INTERIM_CONVERTED_DIR / "finance_r1_sft.jsonl",
                        help="输出 SFT JSONL 文件路径")
    parser.add_argument("--safe-limit", type=int, default=DEFAULT_SAFE_LIMIT,
                        help="Token 安全线 (超过则丢弃, 默认 2000)")
    args = parser.parse_args()

    print("=" * 60)
    print("Finance R1-Distill CoT Data Conversion")
    print("=" * 60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出文件: {args.output}")
    print(f"安全线:   {args.safe_limit} tokens")
    print()

    stats = Counter()
    records = convert_r1_data(args.input_dir, args.safe_limit, stats)

    if not records:
        print("\n没有成功转换的记录!")
        return

    count = write_jsonl(records, args.output)

    # 统计报告
    total = stats["total"]
    kept = stats["kept"]

    print(f"\n{'=' * 60}")
    print("Conversion Report")
    print(f"{'=' * 60}")
    print(f"Total Raw Samples:        {total}")
    print(f"Kept Samples:             {kept} ({kept/total*100:.1f}%)")
    print(f"Dropped (Too Long):       {stats['drop_too_long']} ({stats['drop_too_long']/total*100:.1f}%)")
    print(f"Dropped (Empty User):     {stats['drop_empty_user']}")
    print(f"Dropped (Empty Response): {stats['drop_empty_response']}")
    print(f"Dropped (JSON Error):     {stats['drop_json_error']}")
    if kept > 0:
        print(f"Avg Tokens:               {stats['token_sum']/kept:.0f}")
    print(f"Max Tokens:               {stats.get('max_tokens', 0)}")
    print(f"Output saved to:          {args.output}")
    print(f"Final count:              {count}")

    # 样本展示
    print(f"\n{'=' * 60}")
    print("Sample Output (First 2)")
    print(f"{'=' * 60}")
    for i, rec in enumerate(records[:2]):
        print(f"\n--- Sample {i+1} ---")
        print(f"User: {rec['user'][:200]}...")
        asst = rec['assistant']
        if "<think>" in asst:
            think_end = asst.find("</think>")
            if think_end > 0:
                think_part = asst[:think_end]
                answer_part = asst[think_end+len("</think>"):].strip()
                print(f"Think: {think_part[:150]}...")
                print(f"Answer: {answer_part[:100]}...")
            else:
                print(f"Assistant: {asst[:200]}...")
        else:
            print(f"Assistant: {asst[:200]}...")


if __name__ == "__main__":
    main()
