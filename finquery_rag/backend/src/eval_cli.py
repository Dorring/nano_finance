"""CLI for offline FinQuery RAG evaluation fixtures.

Examples:
  python -m src.eval_cli score --cases eval/golden.jsonl --predictions eval/preds.jsonl
  python -m src.eval_cli replay-from-traces --db trace_log.db --tenant-id 1 --out eval/replay.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import os
import sys

from .services.evaluation import (
    compare_reports,
    evaluate_predictions,
    export_replay_cases_from_feedback,
    export_replay_cases_from_traces,
    load_jsonl_cases,
    load_jsonl_predictions,
)
from .services.feedback import FeedbackStore
from .services.trace import TraceLogger
from .services.eval_runner import run_jsonl_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FinQuery RAG offline evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score predictions against JSONL cases")
    score.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    score.add_argument("--predictions", required=True, help="Predictions JSONL")
    score.add_argument("--out", help="Optional report JSON output path")

    run = sub.add_parser("run", help="Run RAGEngine against JSONL cases")
    run.add_argument("--cases", required=True, help="Golden/replay cases JSONL")
    run.add_argument("--out", required=True, help="Predictions JSONL output path")
    run.add_argument("--user-id", type=int, required=True, help="Tenant/user id for scoped retrieval")
    run.add_argument("--n-results", type=int, default=5, help="Top-k chunks per query")

    compare = sub.add_parser("compare", help="Compare baseline and candidate reports")
    compare.add_argument("--baseline", required=True, help="Baseline report JSON")
    compare.add_argument("--candidate", required=True, help="Candidate report JSON")
    compare.add_argument("--tolerance", type=float, default=0.0, help="Allowed negative metric delta")
    compare.add_argument("--out", help="Optional comparison JSON output path")

    traces = sub.add_parser("traces", help="Export tenant-scoped trace rows as JSONL")
    traces.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    traces.add_argument("--tenant-id", type=int, required=True)
    traces.add_argument("--limit", type=int, default=100)
    traces.add_argument("--offset", type=int, default=0)
    traces.add_argument("--created-after", type=float)
    traces.add_argument("--created-before", type=float)
    traces.add_argument("--error-only", action="store_true")
    traces.add_argument("--out", required=True, help="Output traces JSONL")

    traces_cleanup = sub.add_parser("traces-cleanup", help="Delete old trace rows")
    traces_cleanup.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    traces_cleanup.add_argument(
        "--ttl-seconds",
        type=int,
        default=int(os.getenv("TRACE_TTL_SECONDS", "0")),
        help="Delete traces older than this TTL. 0 deletes traces older than now.",
    )
    traces_cleanup.add_argument("--tenant-id", type=int, help="Optional tenant/user scope")
    traces_cleanup.add_argument("--out", help="Optional JSON cleanup report output path")

    replay = sub.add_parser("replay-from-traces", help="Export replay cases from trace DB")
    replay.add_argument(
        "--db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    replay.add_argument("--tenant-id", type=int, required=True)
    replay.add_argument("--limit", type=int, default=100)
    replay.add_argument("--out", required=True, help="Output replay JSONL")

    feedback_replay = sub.add_parser("feedback-to-replay", help="Export feedback-linked trace replay cases")
    feedback_replay.add_argument(
        "--feedback-db",
        default=os.getenv("FEEDBACK_DB_PATH", "feedback.db"),
        help="Feedback SQLite DB",
    )
    feedback_replay.add_argument(
        "--trace-db",
        default=os.getenv("TRACE_DB_PATH", "trace_log.db"),
        help="TraceLogger SQLite DB",
    )
    feedback_replay.add_argument("--tenant-id", type=int, required=True)
    feedback_replay.add_argument("--rating", choices=["up", "down"], default="down")
    feedback_replay.add_argument("--limit", type=int, default=100)
    feedback_replay.add_argument("--offset", type=int, default=0)
    feedback_replay.add_argument("--out", required=True, help="Output replay JSONL")

    bm25_check = sub.add_parser("bm25-check", help="Check BM25/FTS5 index consistency")
    bm25_check.add_argument(
        "--db",
        default=os.getenv("BM25_DB_PATH", "rag_bm25.db"),
        help="BM25 SQLite DB",
    )
    bm25_check.add_argument("--user-id", type=int, help="Optional tenant/user scope")
    bm25_check.add_argument("--out", help="Optional JSON report output path")

    bm25_rebuild = sub.add_parser("bm25-rebuild", help="Rebuild BM25/FTS5 index from chunk_store")
    bm25_rebuild.add_argument(
        "--db",
        default=os.getenv("BM25_DB_PATH", "rag_bm25.db"),
        help="BM25 SQLite DB",
    )
    bm25_rebuild.add_argument("--user-id", type=int, help="Optional tenant/user scope")
    bm25_rebuild.add_argument("--out", help="Optional JSON report output path")

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

    if args.command == "run":
        # Import lazily because main initializes FastAPI globals and the OpenAI client.
        from .main import get_rag_engine

        predictions = asyncio.run(run_jsonl_cases(
            args.cases,
            args.out,
            get_rag_engine(),
            user_id=args.user_id,
            n_results=args.n_results,
        ))
        print(f"wrote {len(predictions)} predictions to {args.out}")
        return 0

    if args.command == "compare":
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
        comparison = compare_reports(baseline, candidate, regression_tolerance=args.tolerance)
        payload = json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        if not comparison["passed"]:
            _print_compare_failure_summary(comparison)
        return 0 if comparison["passed"] else 1

    if args.command == "traces":
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        count = logger.export_traces_jsonl(
            tenant_id=args.tenant_id,
            output_path=args.out,
            limit=args.limit,
            offset=args.offset,
            created_after=args.created_after,
            created_before=args.created_before,
            error_only=args.error_only,
        )
        print(f"exported {count} traces to {args.out}")
        return 0

    if args.command == "traces-cleanup":
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        report = logger.cleanup_by_ttl(args.ttl_seconds, tenant_id=args.tenant_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0

    if args.command == "replay-from-traces":
        logger = TraceLogger(db_path=args.db, sample_rate=1.0, redact_content=True)
        traces = logger.query_traces(tenant_id=args.tenant_id, limit=args.limit)
        cases = export_replay_cases_from_traces(traces, args.out)
        print(f"exported {len(cases)} replay cases to {args.out}")
        return 0

    if args.command == "feedback-to-replay":
        feedback_store = FeedbackStore(db_path=args.feedback_db)
        trace_logger = TraceLogger(db_path=args.trace_db, sample_rate=1.0, redact_content=True)
        feedback_rows = feedback_store.list_for_tenant(
            tenant_id=args.tenant_id,
            limit=args.limit,
            offset=args.offset,
            rating=args.rating,
        )
        cases = export_replay_cases_from_feedback(
            feedback_rows,
            lambda trace_id: trace_logger.get_trace_for_tenant(args.tenant_id, trace_id),
            args.out,
        )
        print(f"exported {len(cases)} feedback replay cases to {args.out}")
        return 0

    if args.command == "bm25-check":
        from .services.retrieval import SqliteBM25Retriever

        retriever = SqliteBM25Retriever(db_path=args.db)
        report = retriever.integrity_report(user_id=args.user_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if report["ok"] else 1

    if args.command == "bm25-rebuild":
        from .services.retrieval import SqliteBM25Retriever

        retriever = SqliteBM25Retriever(db_path=args.db)
        report = retriever.rebuild_fts_index(user_id=args.user_id)
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0 if report["ok"] else 1

    return 2


def _print_compare_failure_summary(comparison: dict) -> None:
    """Print a compact human-readable compare failure summary to stderr."""
    print("FinQuery eval comparison failed:", file=sys.stderr)
    reasons = comparison.get("failure_reasons") or []
    if reasons:
        for reason in reasons[:10]:
            print(f"- {reason}", file=sys.stderr)
        if len(reasons) > 10:
            print(f"- ... {len(reasons) - 10} more failure reasons", file=sys.stderr)
    else:
        print("- no failure reason details available", file=sys.stderr)

    regressions = comparison.get("regression_details") or []
    if regressions:
        print("Metric regressions:", file=sys.stderr)
        for item in regressions[:10]:
            print(
                "- {metric}: {baseline:.6f} -> {candidate:.6f} (delta {delta:.6f}, tolerance {allowed_drop:.6f})".format(
                    metric=item.get("metric", "unknown"),
                    baseline=float(item.get("baseline") or 0.0),
                    candidate=float(item.get("candidate") or 0.0),
                    delta=float(item.get("delta") or 0.0),
                    allowed_drop=float(item.get("allowed_drop") or 0.0),
                ),
                file=sys.stderr,
            )

    case_failures = comparison.get("case_failure_details") or []
    if case_failures:
        print("Newly failed cases:", file=sys.stderr)
        for item in case_failures[:10]:
            tags = item.get("tags") or []
            suffix = f" tags={','.join(tags)}" if tags else ""
            print(f"- {item.get('id')}{suffix}", file=sys.stderr)



if __name__ == "__main__":
    raise SystemExit(main())
