"""Deterministic query intent classification for FinQuery RAG.

This module is deliberately dependency-free. It provides a conservative router:
only obvious conversational or out-of-scope queries bypass retrieval; finance,
document QA, summarization, and calculation questions continue through RAG.
"""
from __future__ import annotations

import re
from typing import Any


FINANCIAL_KEYWORDS = (
    "revenue", "sales", "expense", "profit", "loss", "income", "cash",
    "balance", "debt", "equity", "margin", "growth", "quarter", "fiscal",
    "earnings", "dividend", "asset", "liability", "ebitda", "operating",
    "gross", "net", "capex", "opex", "cashflow", "cash flow", "statement",
    "report", "table", "page", "q1", "q2", "q3", "q4", "fy", "yoy",
    "营收", "收入", "利润", "亏损", "现金", "负债", "资产", "权益",
    "增长", "季度", "财报", "股息", "报表", "成本", "费用", "净利",
)

STRONG_CALCULATION_KEYWORDS = (
    "calculate", "compute", "ratio", "percentage", "percent", "growth rate",
    "margin", "variance", "yoy", "qoq", "cagr", "rate",
    "计算", "比例", "百分比", "增长率", "同比", "环比", "毛利率", "净利率",
)

WEAK_CALCULATION_KEYWORDS = (
    "change", "increase", "decrease", "difference", "变化", "增长", "下降", "差异",
)

SUMMARY_KEYWORDS = (
    "summarize", "summary", "overview", "key points", "key metrics",
    "main takeaways", "highlights", "概括", "总结", "摘要", "主要指标",
    "要点", "亮点",
)

DOCUMENT_LOOKUP_PATTERNS = (
    "what was", "what were", "what is", "what are", "how much", "which",
    "what percentage", "what percent",
    "according to", "shown", "given", "reported", "in the report",
    "in the document", "in the illustration", "in the table",
)

EXPLICIT_CALCULATION_PATTERNS = (
    "calculate", "compute", "derive", "work out", "what is the ratio",
    "what is the difference", "difference between", "growth rate from",
    "change from", "increase from", "decrease from", "variance between",
    "cagr",
)

GREETING_RE = re.compile(
    r"^(hi|hello|hey|good morning|good afternoon|good evening|你好|您好|嗨)[!.。！\s]*$",
    re.IGNORECASE,
)
THANKS_RE = re.compile(r"^(thank you|thanks|thx|谢谢|感谢)[!.。！\s]*$", re.IGNORECASE)
GOODBYE_RE = re.compile(r"^(bye|goodbye|see you|再见|拜拜)[!.。！\s]*$", re.IGNORECASE)

CONVERSATIONAL_PATTERNS = (
    "what are you", "who are you", "what is finquery", "tell me about yourself",
    "what do you do", "what can you do", "how do you work", "how does this work",
    "how to use", "what can i ask", "help me", "你是谁", "你能做什么",
    "怎么使用", "如何使用",
)

OUT_OF_SCOPE_PATTERNS = (
    "weather", "recipe", "capital of", "president of", "news today",
    "sports score", "write code", "generate image", "tell me a joke",
    "天气", "菜谱", "首都", "总统", "今日新闻", "体育比分", "写代码", "画图", "笑话",
)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _is_document_lookup_calculation_text(text: str) -> bool:
    """Treat reported percentages/margins as document QA unless calculation is explicit."""
    if _contains_any(text, EXPLICIT_CALCULATION_PATTERNS):
        return False
    return _contains_any(text, DOCUMENT_LOOKUP_PATTERNS)


def classify_query_intent(question: str | None) -> dict[str, Any]:
    """Classify a user query into a conservative RAG routing intent."""
    raw = (question or "").strip()
    text = raw.lower()

    if not raw:
        return {
            "intent": "unsupported",
            "confidence": 1.0,
            "requires_retrieval": False,
            "reason": "empty_query",
        }

    has_finance = _contains_any(text, FINANCIAL_KEYWORDS) or "$" in raw or "%" in raw
    has_strong_calc = _contains_any(text, STRONG_CALCULATION_KEYWORDS)
    has_weak_calc = _contains_any(text, WEAK_CALCULATION_KEYWORDS)
    has_calc = has_strong_calc or (has_weak_calc and has_finance)
    has_summary = _contains_any(text, SUMMARY_KEYWORDS)

    if GREETING_RE.match(raw) or THANKS_RE.match(raw) or GOODBYE_RE.match(raw):
        return {
            "intent": "conversation",
            "confidence": 0.95,
            "requires_retrieval": False,
            "reason": "short_conversation",
        }

    if _contains_any(text, CONVERSATIONAL_PATTERNS) and not has_finance:
        return {
            "intent": "conversation",
            "confidence": 0.85,
            "requires_retrieval": False,
            "reason": "assistant_meta_question",
        }

    if has_calc and _is_document_lookup_calculation_text(text):
        return {
            "intent": "document_qa",
            "confidence": 0.84 if has_finance else 0.68,
            "requires_retrieval": True,
            "reason": "reported_metric_lookup",
        }

    if has_calc:
        return {
            "intent": "financial_calculation",
            "confidence": 0.9 if has_finance else 0.72,
            "requires_retrieval": True,
            "reason": "calculation_keyword",
        }

    if has_summary:
        return {
            "intent": "document_summary",
            "confidence": 0.86,
            "requires_retrieval": True,
            "reason": "summary_keyword",
        }

    if has_finance:
        return {
            "intent": "document_qa",
            "confidence": 0.82,
            "requires_retrieval": True,
            "reason": "financial_or_document_keyword",
        }

    if _contains_any(text, OUT_OF_SCOPE_PATTERNS):
        return {
            "intent": "unsupported",
            "confidence": 0.82,
            "requires_retrieval": False,
            "reason": "out_of_scope_general_query",
        }

    # Conservative default: unknown questions still go through retrieval because
    # uploaded document domains may contain user-specific vocabulary.
    return {
        "intent": "document_qa",
        "confidence": 0.55,
        "requires_retrieval": True,
        "reason": "default_to_retrieval",
    }
