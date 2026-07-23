#!/usr/bin/env python3
"""generate_phase5_questions_labels.py

Generates real evaluation questions and labels for Phase 5 from the
synthetic financial corpus.

This script imports the hardcoded ``COMPANY_DATA`` from
``generate_phase5_eval_corpus.py`` and produces, for each partition:

- ``questions.jsonl``  — EvaluationQuery objects (no expected_* fields)
- ``labels.jsonl``     — EvaluationLabel objects (all expected_* fields)
- ``manifest.json``    — case count, SHA256 hashes, slice counts

Case count requirements:
    dev:          >= 30  (6 docs x 8 questions = 48)
    calibration:  >= 40  (6 docs x 8 questions = 48)
    sealed:       >= 50  (6 docs x 9 questions = 54)

Question types (slices):
    front_matter          — company name, stock code
    document_qa           — revenue, net profit, total assets, cash flow
    financial_calculation — gross margin, net margin, debt ratio, growth
    expected_no_answer    — asking about data not in the document

Sealed v2:
    For the sealed partition, only ``manifest.public.json`` is written
    with case_count, slice counts, and hash placeholders. The actual
    ``questions.jsonl`` and ``labels.jsonl`` are written but must be
    held by an independent custodian until RC freeze.

Run from the ``finquery_rag/backend/`` directory::

    python scripts/generate_phase5_questions_labels.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Import company financial data from the corpus generator
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from generate_phase5_eval_corpus import COMPANY_DATA  # noqa: E402

from src.evaluation.manifests import compute_jsonl_sha256  # noqa: E402

EVAL_DATA_DIR = ROOT_DIR / "eval_data" / "phase5"

EVAL_USER_IDS: dict[str, int] = {
    "dev": 9001,
    "calibration": 9002,
    "sealed": 9003,
}

# Question definitions per partition
# Each entry: (question_template, slice_tag, question_generator)
# The question_generator takes a CompanyFinancials and returns a dict with
# question text and label data.


def _fmt_amount(value: int) -> str:
    """Format an integer amount with comma separators."""
    return f"{value:,}"


def make_front_matter_name(fin) -> dict:
    return {
        "question": f"{fin.name}的公司名称是什么？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 1}],
        "expected_numbers": (),
        "expected_calculations": (),
        "expected_intent": "front_matter",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (fin.name,),
        "forbidden_answer_terms": (),
        "slice_tags": ("front_matter",),
    }


def make_front_matter_stock(fin) -> dict:
    return {
        "question": f"{fin.name}的股票代码是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 1}],
        "expected_numbers": (fin.stock_code,),
        "expected_calculations": (),
        "expected_intent": "front_matter",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (fin.stock_code,),
        "forbidden_answer_terms": (),
        "slice_tags": ("front_matter",),
    }


def make_qa_revenue(fin) -> dict:
    return {
        "question": f"{fin.name}{fin.period}年度的营业收入是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 2}],
        "expected_numbers": (_fmt_amount(fin.revenue),),
        "expected_calculations": (),
        "expected_intent": "document_qa",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("document_qa",),
    }


def make_qa_net_profit(fin) -> dict:
    return {
        "question": f"{fin.name}{fin.period}年度的净利润是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 2}],
        "expected_numbers": (_fmt_amount(fin.net_profit),),
        "expected_calculations": (),
        "expected_intent": "document_qa",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("document_qa",),
    }


def make_qa_total_assets(fin) -> dict:
    return {
        "question": f"{fin.name}{fin.period}年度的资产总计是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 3}],
        "expected_numbers": (_fmt_amount(fin.total_assets),),
        "expected_calculations": (),
        "expected_intent": "document_qa",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("document_qa",),
    }


def make_qa_operating_cash_flow(fin) -> dict:
    return {
        "question": f"{fin.name}{fin.period}年度经营活动产生的现金流量净额是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 4}],
        "expected_numbers": (_fmt_amount(fin.operating_cash_flow),),
        "expected_calculations": (),
        "expected_intent": "document_qa",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("document_qa",),
    }


def make_calc_gross_margin(fin) -> dict:
    gross_margin = round(fin.gross_margin, 2)
    return {
        "question": f"{fin.name}{fin.period}年度的毛利率是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 2}],
        "expected_numbers": (f"{gross_margin:.2f}",),
        "expected_calculations": (
            {
                "id": f"gm_{fin.letter}",
                "operation": "gross_margin",
                "args": {
                    "revenue": fin.revenue,
                    "cost_of_revenue": fin.cost_of_revenue,
                },
                "expected_value": f"{gross_margin:.2f}",
                "tolerance": "0.01",
                "unit": "percent",
                "metric": "gross_margin",
                "period": str(fin.period),
                "currency": "CNY",
                "scale": "元",
                "formula_version": "v1",
            },
        ),
        "expected_intent": "financial_calculation",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("financial_calculation",),
    }


def make_calc_net_margin(fin) -> dict:
    net_margin = round(fin.net_margin, 2)
    return {
        "question": f"{fin.name}{fin.period}年度的净利率是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 2}],
        "expected_numbers": (f"{net_margin:.2f}",),
        "expected_calculations": (
            {
                "id": f"nm_{fin.letter}",
                "operation": "net_margin",
                "args": {
                    "net_profit": fin.net_profit,
                    "revenue": fin.revenue,
                },
                "expected_value": f"{net_margin:.2f}",
                "tolerance": "0.01",
                "unit": "percent",
                "metric": "net_margin",
                "period": str(fin.period),
                "currency": "CNY",
                "scale": "元",
                "formula_version": "v1",
            },
        ),
        "expected_intent": "financial_calculation",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("financial_calculation",),
    }


def make_calc_debt_ratio(fin) -> dict:
    debt_ratio = round(fin.debt_ratio, 2)
    return {
        "question": f"{fin.name}{fin.period}年度的资产负债率是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 3}],
        "expected_numbers": (f"{debt_ratio:.2f}",),
        "expected_calculations": (
            {
                "id": f"dr_{fin.letter}",
                "operation": "debt_ratio",
                "args": {
                    "total_liabilities": fin.total_liabilities,
                    "total_assets": fin.total_assets,
                },
                "expected_value": f"{debt_ratio:.2f}",
                "tolerance": "0.01",
                "unit": "percent",
                "metric": "debt_ratio",
                "period": str(fin.period),
                "currency": "CNY",
                "scale": "元",
                "formula_version": "v1",
            },
        ),
        "expected_intent": "financial_calculation",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("financial_calculation",),
    }


def make_calc_revenue_growth(fin) -> dict:
    growth = round(fin.revenue_growth, 2)
    return {
        "question": f"{fin.name}{fin.period}年度的营业收入增长率是多少？",
        "document_names": [fin.filename],
        "expected_sources": [{"filename": fin.filename, "page": 2}],
        "expected_numbers": (f"{growth:.2f}",),
        "expected_calculations": (
            {
                "id": f"rg_{fin.letter}",
                "operation": "growth_rate",
                "args": {
                    "current": fin.revenue,
                    "previous": fin.prev_revenue,
                },
                "expected_value": f"{growth:.2f}",
                "tolerance": "0.01",
                "unit": "percent",
                "metric": "revenue_growth",
                "period": str(fin.period),
                "currency": "CNY",
                "scale": "元",
                "formula_version": "v1",
            },
        ),
        "expected_intent": "financial_calculation",
        "expected_answerability": "answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": False,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("financial_calculation",),
    }


def make_no_answer_rd(fin) -> dict:
    """No-answer: asking about R&D spending which is not in the document."""
    return {
        "question": f"{fin.name}{fin.period}年度的研发费用是多少？",
        "document_names": [fin.filename],
        "expected_sources": (),
        "expected_numbers": (),
        "expected_calculations": (),
        "expected_intent": "document_qa",
        "expected_answerability": "not_answerable",
        "expected_validation_status": "passed",
        "expected_no_answer": True,
        "required_answer_terms": (),
        "forbidden_answer_terms": (),
        "slice_tags": ("expected_no_answer",),
    }


# Question generators per partition
# Dev: 8 questions per doc (48 total)
# Cal: 8 questions per doc (48 total)
# Sealed: 9 questions per doc (54 total)
QUESTION_GENERATORS: dict[str, list] = {
    "dev": [
        make_front_matter_name,
        make_front_matter_stock,
        make_qa_revenue,
        make_qa_net_profit,
        make_qa_total_assets,
        make_calc_gross_margin,
        make_calc_debt_ratio,
        make_no_answer_rd,
    ],
    "calibration": [
        make_front_matter_name,
        make_front_matter_stock,
        make_qa_revenue,
        make_qa_net_profit,
        make_qa_total_assets,
        make_calc_gross_margin,
        make_calc_net_margin,
        make_no_answer_rd,
    ],
    "sealed": [
        make_front_matter_name,
        make_front_matter_stock,
        make_qa_revenue,
        make_qa_net_profit,
        make_qa_total_assets,
        make_qa_operating_cash_flow,
        make_calc_gross_margin,
        make_calc_debt_ratio,
        make_calc_revenue_growth,
    ],
}


def generate_partition_data(partition: str) -> tuple[list[dict], list[dict]]:
    """Generate questions and labels for one partition.

    Returns (questions_list, labels_list) where each item is a dict
    ready to be written as a JSONL line.
    """
    companies = COMPANY_DATA[partition]
    generators = QUESTION_GENERATORS[partition]
    questions: list[dict] = []
    labels: list[dict] = []

    for fin in companies:
        for gen in generators:
            data = gen(fin)
            case_id = f"{partition}_{fin.letter}_{gen.__name__}"

            # Build EvaluationQuery (no expected_* fields)
            question = {
                "case_id": case_id,
                "question": data["question"],
                "document_names": list(data["document_names"]),
                "tags": list(data["slice_tags"]),
                "metadata": {
                    "partition": partition,
                    "company": fin.name,
                    "period": fin.period,
                    "user_id": EVAL_USER_IDS[partition],
                },
            }
            questions.append(question)

            # Build EvaluationLabel (all expected_* fields)
            label = {
                "case_id": case_id,
                "expected_sources": list(data["expected_sources"]),
                "expected_numbers": list(data["expected_numbers"]),
                "expected_calculations": list(data["expected_calculations"]),
                "expected_intent": data["expected_intent"],
                "expected_answerability": data["expected_answerability"],
                "expected_validation_status": data["expected_validation_status"],
                "expected_no_answer": data["expected_no_answer"],
                "required_answer_terms": list(data["required_answer_terms"]),
                "forbidden_answer_terms": list(data["forbidden_answer_terms"]),
                "slice_tags": list(data["slice_tags"]),
                "annotation_evidence": {
                    "source": "synthetic_corpus",
                    "annotator": "script",
                    "document_sha256": _doc_sha256(fin.filename),
                },
            }
            labels.append(label)

    return questions, labels


def _doc_sha256(filename: str) -> str:
    """Compute SHA256 of a corpus document."""
    doc_path = ROOT_DIR / "eval_corpus" / "phase5" / filename
    if doc_path.exists():
        return hashlib.sha256(doc_path.read_bytes()).hexdigest()
    return ""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL file with sorted keys for deterministic output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def count_slices(labels: list[dict]) -> dict[str, int]:
    """Count cases per slice tag."""
    counts: dict[str, int] = {}
    for label in labels:
        for tag in label.get("slice_tags", []):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def main() -> int:
    print("=" * 60)
    print("Phase 5 Questions & Labels Generator")
    print("=" * 60)

    all_counts: dict[str, dict] = {}

    for partition in ("dev", "calibration", "sealed"):
        print(f"\n--- Generating {partition} partition ---")
        questions, labels = generate_partition_data(partition)

        partition_dir = EVAL_DATA_DIR / partition
        partition_dir.mkdir(parents=True, exist_ok=True)

        questions_path = partition_dir / "questions.jsonl"
        labels_path = partition_dir / "labels.jsonl"

        write_jsonl(questions_path, questions)
        write_jsonl(labels_path, labels)

        questions_sha = compute_jsonl_sha256(questions_path)
        labels_sha = compute_jsonl_sha256(labels_path)
        slice_counts = count_slices(labels)

        # Write manifest
        manifest = {
            "partition": partition,
            "case_count": len(questions),
            "questions_sha256": questions_sha,
            "labels_sha256": labels_sha,
            "slices": sorted(slice_counts.keys()),
            "slice_counts": slice_counts,
            "user_id": EVAL_USER_IDS[partition],
            "created_at": "2026-07-23T00:00:00Z",
        }
        manifest_path = partition_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # For sealed partition, also write public manifest (no hashes)
        if partition == "sealed":
            public_manifest = {
                "partition": "sealed",
                "case_count": len(questions),
                "questions_sha256": None,  # withheld until RC freeze
                "labels_sha256": None,     # withheld until blind run
                "slice_counts": slice_counts,
                "sealed_v2": True,
                "note": (
                    "Questions and labels are held by an independent "
                    "custodian. Hashes are published only after RC freeze "
                    "(questions) and before blind run (labels)."
                ),
            }
            public_path = EVAL_DATA_DIR / "sealed" / "manifest.public.json"
            public_path.write_text(
                json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        all_counts[partition] = {
            "questions": len(questions),
            "labels": len(labels),
            "slices": slice_counts,
        }

        print(f"  Questions: {len(questions)}")
        print(f"  Labels: {len(labels)}")
        print(f"  Slices: {slice_counts}")
        print(f"  Questions SHA256: {questions_sha[:16]}...")
        print(f"  Labels SHA256: {labels_sha[:16]}...")

    # Verify minimum case counts
    print(f"\n{'=' * 60}")
    print("Verification:")
    mins = {"dev": 30, "calibration": 40, "sealed": 50}
    all_ok = True
    for p, min_count in mins.items():
        actual = all_counts[p]["questions"]
        status = "OK" if actual >= min_count else "FAIL"
        if actual < min_count:
            all_ok = False
        print(f"  {p}: {actual} cases (min {min_count}) [{status}]")

    total = sum(all_counts[p]["questions"] for p in all_counts)
    print(f"  Total: {total} cases")

    if not all_ok:
        print("\nERROR: Minimum case count not met!")
        return 1

    print(f"\n{'=' * 60}")
    print("Questions and labels generation complete.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
