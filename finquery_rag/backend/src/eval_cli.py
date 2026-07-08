"""CLI for offline FinQuery RAG evaluation fixtures.

Examples:
  python -m src.eval_cli score --cases eval/golden.jsonl --predictions eval/preds.jsonl
  python -m src.eval_cli replay-from-traces --db trace_log.db --tenant-id 1 --out eval/replay.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .services.evaluation import (
    evaluate_predictions,
    export_replay_cases_from_traces,
    load_jsonl_cases,
    load_jsonl_predictions,
)
from .services.trace import TraceLogger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinQuery RAG offline evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score predictions against JSONL cases")
    score.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    score.add_argument("--predictions", required=True, help="Predictions JSONL")
    score.add_argument("--out", help="Optional report JSON output path")

    replay = sub.add_parser("replay-from-traces", help="Export replay cases from trace DB")
    replay.add_argument("--db", default="trace_log.db", help="TraceLogger SQLite DB")
    replay.add_argument("--tenant-id", type=int, required=True)
    replay.add_argument("--limit", type=int, default=100)
    replay.add_argument("--out", required=True, help="Output replay JSONL")

    args = parser.parse_args(argv)

    if args.command == "score":
        cases = load_jsonl_cases(args.cases)
        predictions = load_jsonl_predictions(args.predictions)
        report = evaluate_predictions(cases, predictions)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.command == "replay-from-traces":
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        traces = logger.get_recent(args.tenant_id, limit=args.limit)
        cases = export_replay_cases_from_traces(traces, args.out)
        print(f"exported {len(cases)} replay cases to {args.out}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
