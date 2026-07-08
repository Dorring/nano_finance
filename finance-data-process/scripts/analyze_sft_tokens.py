"""Report supervised-token composition of the merged finance SFT data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import PROCESSED_SFT_DIR, count_tokens, read_jsonl


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def analyze(records: list[dict]) -> dict:
    grouped: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "user_tokens": 0, "assistant_lengths": []}
    )
    for record in records:
        stats = grouped[record["_source"]]
        stats["count"] += 1
        stats["user_tokens"] += count_tokens(record["user"])
        stats["assistant_lengths"].append(count_tokens(record["assistant"]))

    total_assistant_tokens = sum(
        sum(stats["assistant_lengths"]) for stats in grouped.values()
    )
    sources = {}
    for source, stats in sorted(grouped.items()):
        lengths = stats.pop("assistant_lengths")
        assistant_tokens = sum(lengths)
        sources[source] = {
            **stats,
            "assistant_tokens": assistant_tokens,
            "assistant_token_share": (
                assistant_tokens / total_assistant_tokens
                if total_assistant_tokens
                else 0.0
            ),
            "assistant_avg": round(assistant_tokens / len(lengths), 2),
            "assistant_p50": percentile(lengths, 0.50),
            "assistant_p95": percentile(lengths, 0.95),
            "assistant_max": max(lengths, default=0),
        }
    return {
        "count": len(records),
        "assistant_tokens": total_assistant_tokens,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_SFT_DIR / "finance_sft_v1.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_SFT_DIR / "supervised_token_report.json",
    )
    args = parser.parse_args()
    report = analyze(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
