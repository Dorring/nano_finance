import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (
INTERIM_CLEANED_DIR,
INTERIM_CONVERTED_DIR,
PROCESSED_SFT_DIR,
clean_records,
ensure_parent,
read_jsonl,
write_jsonl,
)

SOURCE_TO_TASK = {
"finqa": "qa",
"tatqa": "qa",
"ectsum": "summary",
"finer": "ner",
"finred": "re",
"fiqa": "sentiment",
"finsen": "sentiment",
}

TASK_WEIGHTS = {
"qa": 0.30,
"summary": 0.15,
"ner": 0.20,
"re": 0.20,
"sentiment": 0.15,
}

def verify_json_format(task: str, text: str) -> bool:
if task not in ["ner", "re"]:
return True
try:
parsed = json.loads(text)
if not isinstance(parsed, list):
return False
return True
except Exception:
return False

def main() -> None:
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
"--output",
type=Path,
default=PROCESSED_SFT_DIR / "finance_sft_data.jsonl",
)
args = parser.parse_args()

rnd = random.Random(args.seed)
source_files = {k: INTERIM_CONVERTED_DIR / f"{k}_sft.jsonl" for k in SOURCE_TO_TASK}

missing = [k for k, p in source_files.items() if not p.exists()]
if missing:
raise FileNotFoundError(f"Missing converted datasets: {missing}")

cleaned_by_source: dict[str, list[dict]] = {}
raw_counts: dict[str, int] = {}
clean_counts: dict[str, int] = {}
format_drop_counts: dict[str, int] = defaultdict(int)

for source, path in source_files.items():
raw = read_jsonl(path)
cleaned = clean_records(raw)
raw_counts[source] = len(raw)

task = SOURCE_TO_TASK[source]
valid_records = []
for r in cleaned:
if verify_json_format(task, r["assistant"]):
valid_records.append(r)
else:
format_drop_counts[source] += 1

clean_counts[source] = len(valid_records)
cleaned_by_source[source] = valid_records

cleaned_path = INTERIM_CLEANED_DIR / f"{source}_sft.jsonl"
write_jsonl(valid_records, cleaned_path)

grouped: dict[str, list[dict]] = {k: [] for k in TASK_WEIGHTS}
for source, rows in cleaned_by_source.items():
task = SOURCE_TO_TASK[source]
for r in rows:
grouped[task].append({"user": r["user"], "assistant": r["assistant"], "_task": task, "_source": source})

for task, rows in grouped.items():
if not rows:
raise ValueError(f"Task '{task}' has no available samples after cleaning")

max_total = min(int(len(grouped[t]) / TASK_WEIGHTS[t]) for t in TASK_WEIGHTS)
if max_total <= 0:
raise ValueError("No data available to merge after balancing")

target_per_task: dict[str, int] = {}
used = 0
task_names = list(TASK_WEIGHTS.keys())
for t in task_names[:-1]:
n = int(max_total * TASK_WEIGHTS[t])
n = max(1, min(n, len(grouped[t])))
target_per_task[t] = n
used += n
last = task_names[-1]
target_per_task[last] = max(1, min(max_total - used, len(grouped[last])))

merged: list[dict] = []
for task, rows in grouped.items():
sources_in_task = defaultdict(list)
for r in rows:
sources_in_task[r["_source"]].append(r)

num_sources = len(sources_in_task)
target_n = target_per_task[task]

picked = []
per_source_target = target_n // num_sources if num_sources > 0 else 0

remaining_target = target_n
sorted_sources = sorted(sources_in_task.keys(), key=lambda s: len(sources_in_task[s]))
for i, src in enumerate(sorted_sources):
avg_needed = remaining_target // (num_sources - i)
take_n = min(avg_needed, len(sources_in_task[src]))
rnd.shuffle(sources_in_task[src])
picked.extend(sources_in_task[src][:take_n])
remaining_target -= take_n

rnd.shuffle(picked)
merged.extend(picked)

rnd.shuffle(merged)

main_records = [
{
"user": r["user"], 
"assistant": r["assistant"], 
"source_dataset": r["_source"], 
"task_type": r["_task"]
} 
for r in merged
]

task_index = [{"idx": i, "task": r["task_type"], "source": r["source_dataset"]} for i, r in enumerate(main_records)]

ensure_parent(args.output)
write_jsonl(main_records, args.output)
index_path = PROCESSED_SFT_DIR / "finance_sft_index_tasks.jsonl"
write_jsonl(task_index, index_path)

metadata = {
"raw_counts": raw_counts,
"format_drop_counts": dict(format_drop_counts),
"clean_counts": clean_counts,
"group_counts_after_clean": {k: len(v) for k, v in grouped.items()},
"target_per_task": target_per_task,
"merged_total": len(main_records),
"seed": args.seed,
}
metadata_path = PROCESSED_SFT_DIR / "metadata.json"
ensure_parent(metadata_path)
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"[merge] wrote {len(main_records)} samples -> {args.output}")
print(f"[merge] wrote task index -> {index_path}")
print(f"[merge] wrote metadata -> {metadata_path}")

if __name__ == "__main__":
main()
