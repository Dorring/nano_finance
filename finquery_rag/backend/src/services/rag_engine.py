from .trace import TraceLogger
import time
import os
import re

import asyncio
from .vector_store import query_collection, list_all_documents, get_front_matter_chunks, get_page_chunks
from .retrieval import SqliteBM25Retriever, rrf
from .reranker import build_reranker
from .intent import classify_query_intent
from .memory_profile import build_memory_profile_context
from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.candidate_fusion import (
    normalize_scores as _normalize_scores,
    dedupe_chunks as _dedupe_chunks,
    chunk_doc_name as _chunk_doc_name,
    ensure_multi_doc_coverage as _ensure_multi_doc_coverage,
    boost_front_matter_chunks as _boost_front_matter_chunks,
    summarize_retrieved_chunks as _summarize_retrieved_chunks,
    source_from_chunk as _source_from_chunk,
)
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator
from src.generation.prompt_builder import get_system_prompt
from src.generation.llm_gateway import LLMGateway
from src.generation.response_renderer import validate_answer
from src.generation.deterministic_answers import DeterministicAnswerExtractor

# 尝试导入 tiktoken，如果未安装则降级为字符估算
try:
    import tiktoken
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class RAGEngine:
    """
    多文档检索增强生成系统（适配 NanoChat 2B 金融垂类模型）。
    支持查询单个文档或跨多个文档进行查询。

    完整的 RAG 流水线：
    1. 混合检索（稠密向量 + 稀疏BM25）
    2. 使用倒数秩融合（RRF）进行重排
    3. 上下文构建（带 Token 动态截断与部分保留）
    4. 大语言模型（LLM）异步生成

    关键约束：
    - NanoChat 2B 模型上下文长度仅 2048 token
    - 不支持独立的 system 角色（适配层会自动合并到 user 消息）
    - System prompt 需精简，为检索上下文和生成留足空间
    """

    # 2048 上下文的紧凑分配：
    # system_prompt(~150) + 检索上下文(~1000) + 用户问题(~100) + 生成回答(~700) + 特殊token(~50) ≈ 2000 < 2048
    DEFAULT_MAX_CONTEXT_TOKENS = 1100
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TOP_K_CHUNKS = 3

    def __init__(self, llm_client, model_name: str = "nanochat",
                 use_hybrid: bool = True,
                 max_context_tokens: int = None,
                 max_new_tokens: int = None,
                 bm25_db_path: str | None = None,
                 trace_db_path: str | None = None,
                 reranker_name: str | None = None,
                 reranker_model: str | None = None,
                 retrieval_candidate_multiplier: int = 2):
        """
        RAGEngine 类的初始化方法。

        Args:
            llm_client: OpenAI API 客户端实例，指向 nanochat OpenAI 兼容适配层。
            model_name (str): 模型名称，对应 chat_openai_compat.py 暴露的模型名，默认 "nanochat"。
            use_hybrid (bool): 是否启用 BM25 + 向量搜索的混合检索模式，默认为 True。
            max_context_tokens (int): 上下文最大 Token 限制，默认 1100（适配 2048 上下文窗口）。
            max_new_tokens (int): 模型单次最大生成 Token 数，默认 512。
            bm25_db_path (str | None): SQLite FTS5 稀疏检索数据库路径。未传入时读取 BM25_DB_PATH。
            trace_db_path (str | None): TraceLogger SQLite 路径。未传入时读取 TRACE_DB_PATH。
            reranker_name (str | None): Optional reranker name. None disables reranking.
            reranker_model (str | None): Optional reranker model name/path for model-backed rerankers.
            retrieval_candidate_multiplier (int): Candidate expansion factor for hybrid retrieval.
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.use_hybrid = use_hybrid
        self.max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS

        if bm25_db_path is None:
            bm25_db_path = os.getenv("BM25_DB_PATH", "rag_bm25.db")
        if trace_db_path is None:
            trace_db_path = os.getenv("TRACE_DB_PATH", "trace_log.db")

        self.bm25_db_path = bm25_db_path
        self.trace_db_path = trace_db_path
        self.bm25_retriever = SqliteBM25Retriever(db_path=bm25_db_path)
        self.trace_logger = TraceLogger(db_path=trace_db_path, sample_rate=1.0, redact_content=True)
        self.min_score_threshold = 0.0  # chunks below this score are discarded
        self.rrf_sufficiency_threshold = float(os.getenv("RAG_RRF_SUFFICIENCY_THRESHOLD", "0.025"))
        self.dense_sufficiency_threshold = float(os.getenv("RAG_DENSE_SUFFICIENCY_THRESHOLD", "0.15"))
        self.numeric_rrf_floor = float(os.getenv("RAG_NUMERIC_RRF_FLOOR", "0.008"))
        self.numeric_dense_floor = float(os.getenv("RAG_NUMERIC_DENSE_FLOOR", "0.08"))
        self.reranker = build_reranker(reranker_name, model_name_or_path=reranker_model)
        self.retrieval_candidate_multiplier = max(1, int(retrieval_candidate_multiplier or 1))
        self._last_retrieval_debug = self._make_retrieval_debug(0, 0)
        self._query_processor = QueryProcessor()

        # 初始化 Token 计算器
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

        self._retrieval_pipeline = RetrievalPipeline(
            dense_query_fn=query_collection,
            bm25_retriever=self.bm25_retriever,
            reranker=self.reranker,
            query_processor=self._query_processor,
            candidate_multiplier=self.retrieval_candidate_multiplier,
            use_hybrid=self.use_hybrid,
        )
        self._context_builder = ContextBuilder(
            max_context_tokens=self.max_context_tokens,
            min_score_threshold=self.min_score_threshold,
            tokenizer=self.tokenizer,
        )
        self._sufficiency_evaluator = EvidenceSufficiencyEvaluator(
            rrf_sufficiency_threshold=self.rrf_sufficiency_threshold,
            dense_sufficiency_threshold=self.dense_sufficiency_threshold,
        )
        self._llm_gateway = LLMGateway(
            llm_client=self.llm_client,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
        )
        self._deterministic_extractor = DeterministicAnswerExtractor(
            query_processor=self._query_processor,
        )

    def _get_bm25_retriever(self, doc_name=str, user_id: int = None):
        """获取 SQLite FTS5 稀疏检索器。如果未启用混合检索则返回 None。"""
        if not self.use_hybrid:
            return None
        return self.bm25_retriever

    def _normalize_scores(self, chunks: list) -> list:
        return _normalize_scores(chunks)

    def _make_retrieval_debug(self, candidate_count: int, returned_count: int) -> dict:
        return {
            "reranker": self.reranker.name if self.reranker else None,
            "reranker_enabled": self.reranker is not None,
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "candidate_multiplier": self.retrieval_candidate_multiplier,
        }

    def _apply_reranker(self, query: str, chunks: list, top_k: int) -> list:
        candidate_count = len(chunks)
        if not self.reranker:
            selected = chunks[:top_k]
        else:
            selected = self.reranker.rerank(query, chunks, top_k=top_k)
        self._last_retrieval_debug = self._make_retrieval_debug(
            candidate_count,
            len(selected),
        )
        return selected

    @staticmethod
    def _dedupe_chunks(chunks: list) -> list:
        return _dedupe_chunks(chunks)

    @staticmethod
    def _chunk_doc_name(chunk: dict) -> str | None:
        return _chunk_doc_name(chunk)

    def _ensure_multi_doc_coverage(self, candidates: list, selected: list, doc_names: list[str], top_k: int | None) -> list:
        return _ensure_multi_doc_coverage(candidates, selected, doc_names, top_k)

    def _has_cjk(self, text: str) -> bool:
        return self._query_processor._has_cjk(text)

    def _is_document_front_matter_query(self, query: str) -> bool:
        return self._query_processor.is_front_matter_query(query)

    def _expand_retrieval_query(self, query: str) -> str:
        return self._query_processor.expand(query)

    def _boost_front_matter_chunks(self, query: str, chunks: list) -> list:
        return _boost_front_matter_chunks(
            query, chunks,
            is_front_matter_query_fn=self._query_processor.is_front_matter_query,
        )

    @staticmethod
    def _summarize_retrieved_chunks(chunks: list) -> list:
        return _summarize_retrieved_chunks(chunks)

    @staticmethod
    def _source_from_chunk(chunk: dict) -> dict:
        return _source_from_chunk(chunk)

    def retrieve_single_document(self, doc_name: str, query: str, user_id: int = None, n_results: int = 3) -> list:
        result = self._retrieval_pipeline.retrieve_single(
            document_name=doc_name, query=query, user_id=user_id, top_k=n_results,
        )
        self._last_retrieval_debug = self._retrieval_pipeline._last_retrieval_debug
        return result

    async def retrieve_multiple_documents(self, doc_names: list[str], query: str, user_id: int = None, n_results: int = 3) -> list:
        result = await self._retrieval_pipeline.retrieve_multiple(
            document_names=doc_names, query=query, user_id=user_id, top_k=n_results,
        )
        self._last_retrieval_debug = self._retrieval_pipeline._last_retrieval_debug
        return result

    def _is_title_query(self, query: str) -> bool:
        return self._query_processor.is_title_query(query)

    def _is_numeric_financial_query(self, query: str) -> bool:
        return self._query_processor.is_numeric_query(query)

    def _should_try_deterministic_numeric_answer(self, query: str, chunks: list) -> bool:
        return self._query_processor.should_try_deterministic_numeric_answer(query, chunks)

    def _should_generate_with_low_confidence(self, query: str, chunks: list) -> bool:
        return self._query_processor.should_generate_with_low_confidence(
            query, chunks,
            numeric_rrf_floor=self.numeric_rrf_floor,
            numeric_dense_floor=self.numeric_dense_floor,
        )

    def answer_front_matter_query(self, query: str, chunks: list) -> dict | None:
        return self._deterministic_extractor.answer_front_matter_query(query, chunks)

    @staticmethod
    def _clean_deterministic_title(title: str) -> str:
        return DeterministicAnswerExtractor._clean_deterministic_title(title)

    @staticmethod
    def _is_valid_deterministic_title(title: str) -> bool:
        return DeterministicAnswerExtractor._is_valid_deterministic_title(title)

    def retrieve_front_matter_chunks(self, doc_names: list[str], query: str, user_id: int | None = None) -> list:
        """Direct metadata lookup for deterministic front-matter questions."""
        if not self._is_title_query(query) or not doc_names or user_id is None:
            return []
        if "reporting period" in (query or "").lower():
            return []
        chunks = []
        for doc_name in doc_names:
            for chunk in get_front_matter_chunks(doc_name=doc_name, user_id=user_id, subtype="title"):
                title = re.sub(r"^title\s*:\s*", "", chunk.get("content", ""), flags=re.IGNORECASE).strip()
                if self._is_valid_deterministic_title(self._clean_deterministic_title(title)):
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _parent_context_key(chunk: dict) -> str | None:
        return ContextBuilder._parent_context_key(chunk)

    def _merge_parent_context_chunks(self, chunks: list) -> list:
        return self._context_builder._merge_parent_context_chunks(chunks)

    @staticmethod
    def _compact_child_snippet(content: str, *, max_chars: int = 500) -> str:
        return ContextBuilder._compact_child_snippet(content, max_chars=max_chars)

    @staticmethod
    def _compose_parent_context(parent_excerpt: str, child_snippets: list[str]) -> str:
        return ContextBuilder._compose_parent_context(parent_excerpt, child_snippets)

    def build_context(self, chunks: list) -> tuple:
        return self._context_builder.build(chunks)

    def _get_system_prompt(self) -> str:
        return get_system_prompt()

    def _validate_answer(self, answer: str, sources: list) -> str:
        return validate_answer(answer, sources, max_new_tokens=self.max_new_tokens)

    def _check_context_sufficiency(self, chunks: list) -> tuple:
        result = self._sufficiency_evaluator.evaluate(chunks)
        return result.is_sufficient, result.best_score, result.average_score

    def _compute_confidence(self, chunks: list) -> float:
        return self._sufficiency_evaluator.confidence(chunks)

    def _looks_like_followup_question(self, question: str) -> bool:
        return self._query_processor.looks_like_followup_question(question)

    def _is_valid_rewritten_query(self, original: str, rewritten: str) -> bool:
        return self._query_processor.is_valid_rewritten_query(original, rewritten)

    async def _rewrite_query_with_context(
        self,
        question: str,
        conversation_history: list,
        memory_profile: dict | None = None,
    ) -> str:
        return await self._query_processor.rewrite(
            question, conversation_history, memory_profile,
            llm_client=self.llm_client, model_name=self.model_name,
        )

    async def generate_answer(self, context: str, query: str) -> str:
        return await self._llm_gateway.generate(context, query)

    def answer_numeric_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_numeric_query_from_context(query, context, sources)

    def answer_factual_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_factual_query_from_context(query, context, sources)

    def answer_deterministic_query_from_context(self, query: str, context: str, sources: list) -> dict | None:
        return self._deterministic_extractor.answer_deterministic_query_from_context(query, context, sources)


    @staticmethod
    def _parse_context_lines(context: str) -> list[dict]:
        return DeterministicAnswerExtractor._parse_context_lines(context)

    def _rank_context_evidence(self, query: str, context: str, *, require_number: bool, window_radius: int = 1) -> list[dict]:
        return self._deterministic_extractor._rank_context_evidence(query, context, require_number=require_number, window_radius=window_radius)

    @staticmethod
    def _evidence_window(parsed: list[dict], index: int, *, radius: int, require_number: bool, query_terms: set[str]) -> str:
        return DeterministicAnswerExtractor._evidence_window(parsed, index, radius=radius, require_number=require_number, query_terms=query_terms)

    @staticmethod
    def _select_distinct_evidence(evidence: list[dict], *, limit: int) -> list[dict]:
        return DeterministicAnswerExtractor._select_distinct_evidence(evidence, limit=limit)

    @staticmethod
    def _normalize_numeric_phrase(value: str, query: str, evidence_text: str) -> str:
        return DeterministicAnswerExtractor._normalize_numeric_phrase(value, query, evidence_text)

    @classmethod
    def _extract_numeric_phrases(cls, query: str, text: str) -> list[str]:
        return DeterministicAnswerExtractor._extract_numeric_phrases(query, text)

    @classmethod
    def _summarize_numeric_values(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._summarize_numeric_values(query, selected, context=context)

    @classmethod
    def _targeted_numeric_summary(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._targeted_numeric_summary(query, selected, context=context)

    @staticmethod
    def _parse_comma_number(value: str) -> int:
        return DeterministicAnswerExtractor._parse_comma_number(value)

    @staticmethod
    def _summarize_factual_evidence(query: str, selected: list[dict], *, context: str | None = None) -> str | None:
        return DeterministicAnswerExtractor._summarize_factual_evidence(query, selected, context=context)

    @staticmethod
    def _best_sentence_with_terms(text: str, terms: tuple[str, ...]) -> str | None:
        return DeterministicAnswerExtractor._best_sentence_with_terms(text, terms)

    @staticmethod
    def _first_evidence_sentence(text: str) -> str | None:
        return DeterministicAnswerExtractor._first_evidence_sentence(text)

    @staticmethod
    def _should_try_deterministic_factual_answer(query: str) -> bool:
        return QueryProcessor().should_try_deterministic_factual_answer(query)

    @staticmethod
    def _important_query_terms(query: str) -> set[str]:
        return DeterministicAnswerExtractor._important_query_terms(query)

    @staticmethod
    def _numeric_evidence_score(line: str, query_terms: set[str]) -> float:
        return DeterministicAnswerExtractor._numeric_evidence_score(line, query_terms)

    @staticmethod
    def _factual_evidence_score(line: str, query_terms: set[str]) -> float:
        return DeterministicAnswerExtractor._factual_evidence_score(line, query_terms)

    @staticmethod
    def _is_numeric_financial_query(query: str) -> bool:
        return DeterministicAnswerExtractor._is_numeric_financial_query(query)

    def generate_answer_stream(self, context: str, query: str):
        return self._llm_gateway.generate_stream(context, query)

    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int = None,
        n_results: int = 3,
        conversation_history: list = None,
        memory_profile: dict | None = None,
    ) -> dict:
        """查询一个或多个文档的统一入口方法（全异步）。默认 top-k=3 适配短上下文。"""
        t0 = time.time()
        trace_data = {
            "tenant_id": user_id,
            "query_original": question,
        }

        # Phase 4: Rewrite follow-up question using conversation context
        original_question = question
        if conversation_history:
            question = await self._rewrite_query_with_context(question, conversation_history, memory_profile)
            trace_data["query_rewritten"] = question
            if build_memory_profile_context(memory_profile):
                trace_data["memory_profile_used"] = True

        intent = classify_query_intent(question)
        trace_data["intent"] = intent["intent"]

        conversational_response = self._handle_conversational_query(question)
        if conversational_response:
            result = {
                "answer": conversational_response,
                "sources": [],
                "context": None,
                "searched_docs": [],
                "context_sufficient": True,
                "intent": "conversation",
                "intent_confidence": intent["confidence"],
            }
            if conversation_history:
                result["rewritten_question"] = question
            return result

        if not intent["requires_retrieval"]:
            result = {
                "answer": "This question appears to be outside the uploaded financial documents. Please ask about your uploaded reports or financial data.",
                "sources": [],
                "context": None,
                "searched_docs": [],
                "context_sufficient": True,
                "intent": intent["intent"],
                "intent_confidence": intent["confidence"],
            }
            if conversation_history:
                result["rewritten_question"] = question
            return result

        if doc_names is None:
            all_docs = list_all_documents(user_id)
            doc_names = [doc["name"] for doc in all_docs]

        if not doc_names:
            result = {
                "answer": "No documents found in database. Please upload documents first.",
                "sources": [],
                "context": None,
                "searched_docs": [],
                "context_sufficient": True
            }
            if conversation_history:
                result["rewritten_question"] = question
            return result

        # 1. Retrieve relevant chunks. Front-matter facts use direct metadata lookup first.
        chunks = self.retrieve_front_matter_chunks(doc_names, question, user_id)
        if not chunks:
            if len(doc_names) == 1:
                chunks = self.retrieve_single_document(doc_names[0], question, user_id, n_results)
            else:
                chunks = await self.retrieve_multiple_documents(doc_names, question, user_id, n_results)

        front_matter_answer = self.answer_front_matter_query(question, chunks)
        deterministic_answer = None
        if front_matter_answer:
            chunks = front_matter_answer["chunks"]
            context, sources = self.build_context(chunks)
            answer = front_matter_answer["answer"]
            is_sufficient = True
            best_score = 1.0
            avg_score = 1.0
            confidence = 1.0
            deterministic_answer = front_matter_answer["diagnostic"]
        else:
            # Phase 3: Check context sufficiency
            is_sufficient, best_score, avg_score = self._check_context_sufficiency(chunks)
            confidence = self._compute_confidence(chunks)

            # 2. Build context (with dedup and score threshold)
            context, sources = self.build_context(chunks)
            # 3. Generate answer (skip LLM if context is insufficient)
            deterministic_context_answer = self.answer_deterministic_query_from_context(question, context, sources)
            if deterministic_context_answer:
                answer = deterministic_context_answer["answer"]
                is_sufficient = True
                deterministic_answer = deterministic_context_answer["diagnostic"]
                low_confidence_numeric_override = False
            else:
                low_confidence_numeric_override = self._should_generate_with_low_confidence(question, chunks)
                if low_confidence_numeric_override:
                    is_sufficient = True

                if not is_sufficient:
                    answer = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
                else:
                    answer = await self.generate_answer(context, question)

        # 4. Log trace
        elapsed_ms = (time.time() - t0) * 1000
        trace_data.update({
            "filter_conditions": {"doc_names": doc_names, "n_results": n_results},
            "candidates": [
                {
                    "doc_id": c.get("doc_id", ""),
                    "score": c.get("score", 0),
                    "rerank_score": c.get("rerank_score"),
                    "reranker": c.get("reranker"),
                }
                for c in chunks
            ],
            "final_context": context,
            "answer": answer,
            "sources": sources,
            "diagnostics": {
                "confidence": confidence,
                "context_sufficient": is_sufficient,
                "intent_confidence": intent["confidence"],
                "deterministic_answer": deterministic_answer,
                "low_confidence_numeric_override": (
                    low_confidence_numeric_override if not front_matter_answer else False
                ),
            },
            "model_name": self.model_name,
            "latency_ms": elapsed_ms,
        })
        trace_id = None
        try:
            trace_id = self.trace_logger.log(**trace_data)
        except Exception:
            pass  # tracing must never break the query path

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "searched_docs": doc_names,
            "confidence": confidence,
            "context_sufficient": is_sufficient,
            "intent": intent["intent"],
            "intent_confidence": intent["confidence"],
            "rewritten_question": question if conversation_history else None,
            "retrieved_chunks": self._summarize_retrieved_chunks(chunks),
            "retrieval_debug": dict(self._last_retrieval_debug),
            "trace_id": trace_id,
        }

    def _handle_conversational_query(self, query: str) -> str | None:
        """
        处理对话性/元问题（无需 RAG 检索）。
        增加财务关键词前置保护，防止合法查询被误判为闲聊。
        """
        query_lower = query.lower().strip()

        # 财务强相关关键词，出现这些词绝不能被判定为闲聊
        financial_indicators = [
            "revenue", "expense", "profit", "loss", "income", "cash",
            "balance", "debt", "equity", "margin", "growth", "quarter",
            "fiscal", "earnings", "dividend", "asset", "liability",
            "$", "%", "million", "billion", "q1", "q2", "q3", "q4",
            "fy", "yoy", "table", "page", "report", "statement", "cost",
            # 中文金融关键词
            "营收", "利润", "亏损", "收入", "现金", "负债", "资产", "权益",
            "增长", "季度", "财报", "股息", "报表", "成本", "费用", "净利"
        ]
        if any(ind in query_lower for ind in financial_indicators):
            return None  # 强制走 RAG 路径

        # Greetings
        greetings = ["hi", "hello", "hi there", "hey", "good morning", "good afternoon", "good evening"]
        if any(query_lower.startswith(g) for g in greetings) and len(query_lower.split()) <= 3:
            return "Hello! I'm FinQuery, your financial document assistant. I can help you find information in your uploaded documents. What would you like to know?"

        # Identity questions
        identity_keywords = [
            "what are you", "who are you", "what is finquery",
            "tell me about yourself", "what do you do", "what can you do",
            "how do you work", "what's your purpose"
        ]
        if any(keyword in query_lower for keyword in identity_keywords):
            return "I'm FinQuery, an AI assistant that helps you analyze financial documents. Upload PDFs of reports, statements, or other financial documents, and I'll answer questions about them using the exact information from those documents."

        # Capability questions
        capability_keywords = ["how does this work", "how to use", "help me", "what can i ask", "how do i use this"]
        if any(keyword in query_lower for keyword in capability_keywords):
            return "Here's how to use FinQuery:\n1. Upload financial documents (PDFs)\n2. Ask questions about the content\n3. I'll provide answers with page citations\n\nTry: 'What was the revenue in Q3?' or 'Summarize key financial metrics'"

        # Thanks/gratitude
        thanks_keywords = ["thank you", "thanks", "thx", "appreciate"]
        if any(keyword in query_lower for keyword in thanks_keywords) and len(query_lower.split()) <= 5:
            return "You're welcome! Let me know if you have any other questions about your documents."

        # Goodbyes
        goodbye_keywords = ["bye", "goodbye", "see you", "exit", "quit"]
        if any(keyword in query_lower for keyword in goodbye_keywords) and len(query_lower.split()) <= 3:
            return "Goodbye! Feel free to come back anytime you need to analyze financial documents."

        return None
