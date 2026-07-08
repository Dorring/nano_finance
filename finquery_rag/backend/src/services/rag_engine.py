from .trace import TraceLogger
import time

import asyncio
from .vector_store import query_collection, list_all_documents
from .retrieval import SqliteBM25Retriever, rrf
from .reranker import build_reranker

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
                 bm25_db_path: str = "rag_bm25.db",
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
            bm25_db_path (str): SQLite FTS5 稀疏检索数据库路径，默认 "rag_bm25.db"。
            reranker_name (str | None): Optional reranker name. None disables reranking.
            reranker_model (str | None): Optional reranker model name/path for model-backed rerankers.
            retrieval_candidate_multiplier (int): Candidate expansion factor for hybrid retrieval.
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.use_hybrid = use_hybrid
        self.max_context_tokens = max_context_tokens or self.DEFAULT_MAX_CONTEXT_TOKENS
        self.max_new_tokens = max_new_tokens or self.DEFAULT_MAX_NEW_TOKENS

        self.bm25_retriever = SqliteBM25Retriever(db_path=bm25_db_path)
        self.trace_logger = TraceLogger(sample_rate=1.0, redact_content=True)
        self.min_score_threshold = 0.0  # chunks below this score are discarded
        self.reranker = build_reranker(reranker_name, model_name_or_path=reranker_model)
        self.retrieval_candidate_multiplier = max(1, int(retrieval_candidate_multiplier or 1))
        self._last_retrieval_debug = self._make_retrieval_debug(0, 0)

        # 初始化 Token 计算器
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.tokenizer = None
        else:
            self.tokenizer = None

    def _get_bm25_retriever(self, doc_name=str, user_id: int = None):
        """获取 SQLite FTS5 稀疏检索器。如果未启用混合检索则返回 None。"""
        if not self.use_hybrid:
            return None
        return self.bm25_retriever

    def _normalize_scores(self, chunks: list) -> list:
        """统一分数字段，将 RRF 融合后的 fused_score 统一写入 score 字段。"""
        for chunk in chunks:
            if "fused_score" in chunk:
                chunk["score"] = chunk["fused_score"]
            elif "score" not in chunk:
                chunk["score"] = 0
        return chunks

    def _make_retrieval_debug(self, candidate_count: int, returned_count: int) -> dict:
        """Small metadata payload used by eval/replay to audit retrieval changes."""
        return {
            "reranker": self.reranker.name if self.reranker else None,
            "reranker_enabled": self.reranker is not None,
            "candidate_count": candidate_count,
            "returned_count": returned_count,
            "candidate_multiplier": self.retrieval_candidate_multiplier,
        }

    def _apply_reranker(self, query: str, chunks: list, top_k: int) -> list:
        """Apply optional reranker while preserving default retrieval behavior."""
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
    def _summarize_retrieved_chunks(chunks: list) -> list:
        """Return eval-safe retrieval metadata without copying chunk content."""
        summary = []
        for chunk in chunks:
            meta = chunk.get("metadata", {}) or {}
            summary.append({
                "doc_id": chunk.get("doc_id", ""),
                "filename": meta.get("doc_name"),
                "page": meta.get("page"),
                "type": meta.get("type"),
                "score": chunk.get("score", 0),
                "rerank_score": chunk.get("rerank_score"),
                "reranker": chunk.get("reranker"),
            })
        return summary

    def retrieve_single_document(self, doc_name: str, query: str, user_id: int = None, n_results: int = 3) -> list:
        """使用混合搜索从单个文档中检索相关文本块。默认 top-k=3 适配短上下文。"""
        if not self.use_hybrid:
            results = query_collection(query_text=query, doc_name=doc_name, n_results=n_results, user_id=user_id)
            results = self._normalize_scores(results)
            return self._apply_reranker(query, results, n_results)

        # Hybrid search
        candidate_k = n_results * self.retrieval_candidate_multiplier
        dense_results = query_collection(query_text=query, doc_name=doc_name, n_results=candidate_k, user_id=user_id)

        bm25_retriever = self._get_bm25_retriever(doc_name, user_id)
        if bm25_retriever:
            print(f"✓ BM25 retrieved for '{doc_name}'")
            sparse_results = bm25_retriever.search(query, k=candidate_k, doc_name=doc_name, user_id=user_id)
            fused = rrf([dense_results, sparse_results])
            results = self._normalize_scores(fused)
            return self._apply_reranker(query, results, n_results)

        results = self._normalize_scores(dense_results)
        return self._apply_reranker(query, results, n_results)

    async def retrieve_multiple_documents(self, doc_names: list[str], query: str, user_id: int = None, n_results: int = 3) -> list:
        """异步并发地从多个文档中检索相关文本块，并按相关性得分降序返回前 N 个结果。"""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(
                None,
                self.retrieve_single_document,
                doc_name, query, user_id, n_results
            )
            for doc_name in doc_names
        ]

        results_list = await asyncio.gather(*tasks)

        all_results = []
        for results in results_list:
            all_results.extend(results)

        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        return self._apply_reranker(query, all_results, n_results)

    def build_context(self, chunks: list) -> tuple:
        """Build context from retrieved chunks with dedup, score threshold, and token budget."""
        if not chunks:
            return "", []

        # Phase 2: deduplicate chunks by content
        seen_content = set()
        deduped = []
        for chunk in chunks:
            content_key = chunk["content"][:100]  # first 100 chars as dedup key
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduped.append(chunk)
        chunks = deduped

        # Phase 2: filter by minimum score threshold
        if self.min_score_threshold > 0:
            chunks = [c for c in chunks if c.get("score", 0) >= self.min_score_threshold]

        if not chunks:
            return "", []

        context_parts = []
        sources = []
        current_tokens = 0
        safe_limit = self.max_context_tokens - 200

        for i, chunk in enumerate(chunks, 1):
            doc_id = chunk["doc_id"]
            content = chunk["content"]
            chunk_type = chunk["metadata"].get("type")
            page = chunk["metadata"].get("page")

            # Parse filename from scoped chunk ID
            if "::" in doc_id:
                parts = doc_id.split("::")[0]
                # Remove user_N_ prefix if present
                if parts.startswith("user_"):
                    parts = "_".join(parts.split("_")[2:])
                filename = parts
            else:
                filename = doc_id

            if chunk_type == "table":
                table_num = chunk["metadata"].get("table_num", "")
                source_ref = "%s, p%s(T%s)" % (filename, page, table_num)
            else:
                source_ref = "%s, p%s" % (filename, page)

            chunk_text = "[%s]\n%s" % (source_ref, content)

            if self.tokenizer:
                chunk_tokens = len(self.tokenizer.encode(chunk_text))
            else:
                chunk_tokens = len(chunk_text) / 3

            if current_tokens + chunk_tokens > safe_limit:
                remaining_tokens = safe_limit - current_tokens
                if remaining_tokens > 80:
                    if self.tokenizer:
                        truncated_tokens = self.tokenizer.encode(content)[:remaining_tokens-20]
                        truncated_content = self.tokenizer.decode(truncated_tokens) + "\n[...]"
                    else:
                        truncated_content = content[:int(remaining_tokens * 3)] + "\n[...]"
                    chunk_text = "[%s]\n%s" % (source_ref, truncated_content)
                    context_parts.append(chunk_text)
                    sources.append({
                        "filename": filename, "page": page,
                        "type": chunk_type, "score": chunk.get("score", 0),
                    })
                break

            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
            sources.append({
                "filename": filename, "page": page,
                "type": chunk_type, "score": chunk.get("score", 0),
            })

        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, sources

    def _get_system_prompt(self) -> str:
        """
        精简版 System Prompt，适配 2B 模型 + 2048 上下文。
        原版约 230 token，精简至约 120 token，为核心检索内容腾出空间。
        """
        return """You are FinQuery, a financial document assistant. Rules:
1. Answer based ONLY on the provided context
2. Cite sources: "Source: <filename>, page <number>"
3. Preserve exact numbers, currencies, dates from tables
4. If no relevant info found, say so clearly
5. Answer in prose, never use markdown table syntax
6. Be concise and precise."""

    def _validate_answer(self, answer: str, sources: list) -> str:
        """
        Phase 3: Post-generation answer validation and cleanup.
        - Strips whitespace and model artifacts
        - Returns refusal message if answer is empty or near-empty
        - Truncates overly long answers to max_new_tokens * 4 chars
        """
        if not answer:
            return "I couldn't generate a valid answer. Please try rephrasing your question."

        # Strip model artifacts and excessive whitespace
        answer = answer.strip()
        for artifact in ["<|end|>", "</s>", "[END]", "[/INST]"]:
            answer = answer.replace(artifact, "")
        answer = answer.strip()

        # Near-empty after cleanup
        if len(answer) < 10:
            return "I couldn't generate a meaningful answer. Please try rephrasing your question."

        # Truncate overly long answers (safety cap)
        max_chars = self.max_new_tokens * 4
        if len(answer) > max_chars:
            answer = answer[:max_chars].rsplit(" ", 1)[0] + "..."

        return answer

    def _check_context_sufficiency(self, chunks: list) -> tuple:
        """
        Phase 3: Check if retrieved context is sufficient for a reliable answer.
        Returns (is_sufficient: bool, best_score: float, avg_score: float).

        Scores are mode-dependent:
        - Dense-only (cosine): 0-1 range, threshold 0.15
        - Hybrid/RRF fused_score: ~0.01-0.05 range, threshold 0.008
        """
        if not chunks:
            return False, 0.0, 0.0

        scores = [c.get("score", 0) for c in chunks]
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Detect score scale: RRF fused_scores are typically < 0.05
        # Dense cosine scores are typically 0-1
        max_possible_rrf = 0.05
        if best_score < max_possible_rrf:
            # RRF mode — use calibrated threshold
            SUFFICIENCY_THRESHOLD = 0.008
        else:
            # Dense mode — use original threshold
            SUFFICIENCY_THRESHOLD = 0.15

        is_sufficient = best_score >= SUFFICIENCY_THRESHOLD
        return is_sufficient, best_score, avg_score

    def _compute_confidence(self, chunks: list) -> float:
        """
        Phase 3: Compute answer confidence based on retrieval quality.
        Returns a float between 0.0 and 1.0.
        """
        if not chunks:
            return 0.0

        scores = [c.get("score", 0) for c in chunks]
        best = max(scores)
        avg = sum(scores) / len(scores)

        # Confidence = weighted blend of best and average score
        confidence = 0.7 * best + 0.3 * avg
        return min(1.0, max(0.0, confidence))

    async def _rewrite_query_with_context(self, question: str, conversation_history: list) -> str:
        """
        Phase 4: Rewrite a follow-up question into a standalone query using conversation context.

        Uses a lightweight prompt to the LLM to resolve pronouns and implicit references.
        Returns the original question if rewriting fails or history is empty.

        Args:
            question: The current user question (may be a follow-up)
            conversation_history: List of {"role": ..., "content": ...} dicts

        Returns:
            Standalone version of the question, or original if rewriting fails
        """
        if not conversation_history or len(conversation_history) < 2:
            return question  # no context to rewrite with

        # Build a compact conversation summary from recent messages
        # Limit to last 4 messages (2 pairs) to stay within token budget
        recent = conversation_history[-4:]
        history_parts = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages to 200 chars to save tokens
            content = msg["content"][:200]
            history_parts.append(f"{role}: {content}")
        history_text = "\n".join(history_parts)

        rewrite_prompt = (
            f"Given this conversation:\n{history_text}\n\n"
            f"Rewrite the following question as a standalone question "
            f"that can be understood without the conversation context. "
            f"Only output the rewritten question, nothing else.\n\n"
            f"Question: {question}"
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    temperature=0,
                    max_tokens=100,
                )
            )
            rewritten = response.choices[0].message.content
            if rewritten and len(rewritten.strip()) > 5:
                return rewritten.strip()
            return question
        except Exception:
            return question  # graceful fallback: use original question

    async def generate_answer(self, context: str, query: str) -> str:
        """使用大语言模型生成回答（非流式输出，异步不阻塞）。"""
        if not context:
            return "I couldn't find relevant information in the documents to answer your question."

        system_prompt = self._get_system_prompt()
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=self.max_new_tokens
                )
            )
            raw_answer = response.choices[0].message.content
            return self._validate_answer(raw_answer, [])
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def generate_answer_stream(self, context: str, query: str):
        """
        使用大语言模型生成回答（流式输出）。
        通过 openai SDK 对接 nanochat OpenAI 兼容适配层。
        """
        if not context:
            yield "I couldn't find relevant information in the documents to answer your question."
            return

        system_prompt = self._get_system_prompt()
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=self.max_new_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error generating answer: {str(e)}"

    async def query(self, question: str, doc_names: list[str] | None = None, user_id: int = None, n_results: int = 3, conversation_history: list = None) -> dict:
        """查询一个或多个文档的统一入口方法（全异步）。默认 top-k=3 适配短上下文。"""
        t0 = time.time()
        trace_data = {
            "tenant_id": user_id,
            "query_original": question,
        }

        # Phase 4: Rewrite follow-up question using conversation context
        original_question = question
        if conversation_history:
            question = await self._rewrite_query_with_context(question, conversation_history)
            trace_data["query_rewritten"] = question

        conversational_response = self._handle_conversational_query(question)
        if conversational_response:
            result = {
                "answer": conversational_response,
                "sources": [],
                "context": None,
                "searched_docs": [],
                "context_sufficient": True
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

        # 1. Retrieve relevant chunks
        if len(doc_names) == 1:
            chunks = self.retrieve_single_document(doc_names[0], question, user_id, n_results)
        else:
            chunks = await self.retrieve_multiple_documents(doc_names, question, user_id, n_results)

        # Phase 3: Check context sufficiency
        is_sufficient, best_score, avg_score = self._check_context_sufficiency(chunks)
        confidence = self._compute_confidence(chunks)

        # 2. Build context (with dedup and score threshold)
        context, sources = self.build_context(chunks)

        # 3. Generate answer (skip LLM if context is insufficient)
        if not is_sufficient:
            answer = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
        else:
            answer = await self.generate_answer(context, question)

        # 4. Log trace
        elapsed_ms = (time.time() - t0) * 1000
        trace_data.update({
            "filter_conditions": {"doc_names": doc_names},
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
            "model_name": self.model_name,
            "latency_ms": elapsed_ms,
        })
        try:
            self.trace_logger.log(**trace_data)
        except Exception:
            pass  # tracing must never break the query path

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "searched_docs": doc_names,
            "confidence": confidence,
            "context_sufficient": is_sufficient,
            "rewritten_question": question if conversation_history else None,
            "retrieved_chunks": self._summarize_retrieved_chunks(chunks),
            "retrieval_debug": dict(self._last_retrieval_debug),
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
