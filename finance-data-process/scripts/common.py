from __future__ import annotations

import json
import random
import re
import os
from pathlib import Path
from typing import Iterable, Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_CONVERTED_DIR = ROOT / "data" / "interim" / "converted"
INTERIM_CLEANED_DIR = ROOT / "data" / "interim" / "cleaned"
PROCESSED_SFT_DIR = ROOT / "data" / "processed" / "sft"
CONFIG_FILE = ROOT / "configs" / "length_control.yaml"

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [to_text(v) for v in value]
        return normalize_text("; ".join([p for p in parts if p]))
    if isinstance(value, dict):
        return normalize_text(json.dumps(value, ensure_ascii=False))
    return normalize_text(str(value))

def write_jsonl(records: Iterable[dict], output_path: Path) -> int:
    ensure_parent(output_path)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count

def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists(): return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# --- Tokenizer and Length Control Tools ---

_TOKENIZER = None

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        # 优先加载自定义训练的 tokenizer (65k vocab)
        try:
            import sys
            project_root = str(ROOT.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from nanochat.tokenizer import RustBPETokenizer
            import os
            tokenizer_dir = os.path.expanduser("~/.cache/nanochat/tokenizer")
            if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")):
                _TOKENIZER = RustBPETokenizer.from_directory(tokenizer_dir)
            else:
                from nanochat.tokenizer import get_tokenizer as get_nano_tokenizer
                _TOKENIZER = get_nano_tokenizer()
        except Exception as e:
            print(f"Warning: Failed to load nanochat tokenizer ({e}), falling back to Qwen")
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    return _TOKENIZER

def count_tokens(text: str) -> int:
    tokenizer = get_tokenizer()
    ids = tokenizer.encode(text)
    return len(ids)

def truncate_text_by_tokens(text: str, budget: int) -> str:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    if len(tokens) <= budget:
        return text
    truncated_tokens = tokens[:budget]
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(truncated_tokens)
    return text[:budget*2]

def truncate_assistant_by_budget(text: str, budget: int) -> str:
    return truncate_text_by_tokens(text, budget)

def truncate_summary_by_budget(text: str, budget_tokens: int, head_ratio: float = 0.7) -> dict:
    tokenizer = get_tokenizer()
    orig_tokens = count_tokens(text)
    if orig_tokens <= budget_tokens:
        return {"text": text, "orig_tokens": orig_tokens, "new_tokens": orig_tokens, "truncated": False}

    tokens = tokenizer.encode(text)
    head_budget = int(budget_tokens * head_ratio)
    tail_budget = budget_tokens - head_budget

    head_tokens = tokens[:head_budget]
    tail_tokens = tokens[-tail_budget:] if tail_budget > 0 else []

    def safe_decode(tks):
        if hasattr(tokenizer, 'decode'): return tokenizer.decode(tks)
        return ""

    new_text = safe_decode(head_tokens) + "\n\n... [TRUNCATED] ...\n\n" + safe_decode(tail_tokens)
    return {"text": new_text, "orig_tokens": orig_tokens, "new_tokens": count_tokens(new_text), "truncated": True}

def linearize_table(table_data: list[list[str]]) -> str:
    if not table_data:
        return ""
    header = table_data[0]
    lines = []
    for row in table_data[1:]:
        row_str = "; ".join([f"{h}: {v}" for h, v in zip(header, row) if v])
        lines.append(row_str)
    return " | ".join(lines)
