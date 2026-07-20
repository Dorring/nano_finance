"""RAG Orchestrator: coordinates retrieval, context building, and generation.

This is the top-level orchestration layer that ties together the retrieval
pipeline, context builder, sufficiency evaluator, deterministic answer
extractor, and LLM gateway into a single query flow.
"""
import re
import time

from src.retrieval.query_processor import QueryProcessor
from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.retrieval.context_builder import ContextBuilder, EvidenceSufficiencyEvaluator
from src.generation.llm_gateway import LLMGateway
from src.generation.deterministic_answers import DeterministicAnswerExtractor
from src.retrieval.candidate_fusion import summarize_retrieved_chunks


class RAGOrchestrator:
    """Orchestrates the full RAG query pipeline.

    Coordinates:
    1. Query rewriting (follow-up resolution)
    2. Intent classification
    3. Conversational query handling
    4. Document retrieval (single/multi)
    5. Front-matter deterministic answering
    6. Context sufficiency evaluation
    7. Context building
    8. Deterministic context answering
    9. LLM generation (when needed)
    10. Trace logging
    """

    def __init__(
        self,
        *,
        query_processor: QueryProcessor,
        retrieval_pipeline: RetrievalPipeline,
        context_builder: ContextBuilder,
        sufficiency_evaluator: EvidenceSufficiencyEvaluator,
        llm_gateway: LLMGateway,
        deterministic_extractor: DeterministicAnswerExtractor,
        trace_logger,
        intent_classifier,
        list_all_documents_fn,
        get_front_matter_chunks_fn,
        numeric_rrf_floor: float = 0.008,
        numeric_dense_floor: float = 0.08,
        model_name: str = "nanochat",
    ):
        self._query_processor = query_processor
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._sufficiency_evaluator = sufficiency_evaluator
        self._llm_gateway = llm_gateway
        self._deterministic_extractor = deterministic_extractor
        self._trace_logger = trace_logger
        self._classify_intent = intent_classifier
        self._list_all_documents = list_all_documents_fn
        self._get_front_matter_chunks = get_front_matter_chunks_fn
        self._numeric_rrf_floor = numeric_rrf_floor
        self._numeric_dense_floor = numeric_dense_floor
        self._model_name = model_name

    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int = None,
        n_results: int = 3,
        conversation_history: list = None,
        memory_profile: dict | None = None,
    ) -> dict:
        """Execute full RAG query pipeline."""
        t0 = time.time()
        trace_data = {
            "tenant_id": user_id,
            "query_original": question,
        }

        # Phase 4: Rewrite follow-up question using conversation context
        original_question = question
        if conversation_history:
            question = await self._query_processor.rewrite(
                question, conversation_history, memory_profile,
                llm_client=self._llm_gateway._llm_client,
                model_name=self._llm_gateway._model_name,
            )
            trace_data["query_rewritten"] = question

        intent = self._classify_intent(question)
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
            all_docs = self._list_all_documents(user_id)
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
        chunks = self._retrieve_front_matter_chunks(doc_names, question, user_id)
        if not chunks:
            if len(doc_names) == 1:
                chunks = self._retrieval_pipeline.retrieve_single(
                    document_name=doc_names[0], query=question, user_id=user_id, top_k=n_results,
                )
            else:
                chunks = await self._retrieval_pipeline.retrieve_multiple(
                    document_names=doc_names, query=question, user_id=user_id, top_k=n_results,
                )

        front_matter_answer = self._deterministic_extractor.answer_front_matter_query(question, chunks)
        deterministic_answer = None
        low_confidence_numeric_override = False
        if front_matter_answer:
            chunks = front_matter_answer["chunks"]
            context, sources = self._context_builder.build(chunks)
            answer = front_matter_answer["answer"]
            is_sufficient = True
            best_score = 1.0
            avg_score = 1.0
            confidence = 1.0
            deterministic_answer = front_matter_answer["diagnostic"]
        else:
            # Phase 3: Check context sufficiency
            sufficiency = self._sufficiency_evaluator.evaluate(chunks)
            is_sufficient = sufficiency.is_sufficient
            best_score = sufficiency.best_score
            avg_score = sufficiency.average_score
            confidence = self._sufficiency_evaluator.confidence(chunks)

            # 2. Build context (with dedup and score threshold)
            context, sources = self._context_builder.build(chunks)
            # 3. Generate answer (skip LLM if context is insufficient)
            deterministic_context_answer = self._deterministic_extractor.answer_deterministic_query_from_context(question, context, sources)
            if deterministic_context_answer:
                answer = deterministic_context_answer["answer"]
                is_sufficient = True
                deterministic_answer = deterministic_context_answer["diagnostic"]
                low_confidence_numeric_override = False
            else:
                low_confidence_numeric_override = self._query_processor.should_generate_with_low_confidence(
                    question, chunks,
                    numeric_rrf_floor=self._numeric_rrf_floor,
                    numeric_dense_floor=self._numeric_dense_floor,
                )
                if low_confidence_numeric_override:
                    is_sufficient = True

                if not is_sufficient:
                    answer = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
                else:
                    answer = await self._llm_gateway.generate(context, question)

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
            "model_name": self._model_name,
            "latency_ms": elapsed_ms,
        })
        trace_id = None
        try:
            trace_id = self._trace_logger.log(**trace_data)
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
            "retrieved_chunks": summarize_retrieved_chunks(chunks),
            "retrieval_debug": dict(self._retrieval_pipeline._last_retrieval_debug),
            "trace_id": trace_id,
        }

    def _retrieve_front_matter_chunks(self, doc_names: list[str], query: str, user_id: int | None = None) -> list:
        """Direct metadata lookup for deterministic front-matter questions."""
        if not self._query_processor.is_title_query(query) or not doc_names or user_id is None:
            return []
        if "reporting period" in (query or "").lower():
            return []
        chunks = []
        for doc_name in doc_names:
            for chunk in self._get_front_matter_chunks(doc_name=doc_name, user_id=user_id, subtype="title"):
                title = re.sub(r"^title\s*:\s*", "", chunk.get("content", ""), flags=re.IGNORECASE).strip()
                if DeterministicAnswerExtractor._is_valid_deterministic_title(
                    DeterministicAnswerExtractor._clean_deterministic_title(title)
                ):
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _handle_conversational_query(query: str) -> str | None:
        """Handle conversational/meta questions without RAG retrieval."""
        query_lower = query.lower().strip()

        # Financial keyword guard - never classify as chitchat
        financial_indicators = [
            "revenue", "expense", "profit", "loss", "income", "cash",
            "balance", "debt", "equity", "margin", "growth", "quarter",
            "fiscal", "earnings", "dividend", "asset", "liability",
            "$", "%", "million", "billion", "q1", "q2", "q3", "q4",
            "fy", "yoy", "table", "page", "report", "statement", "cost",
            "营收", "利润", "亏损", "收入", "现金", "负债", "资产", "权益",
            "增长", "季度", "财报", "股息", "报表", "成本", "费用", "净利"
        ]
        if any(ind in query_lower for ind in financial_indicators):
            return None

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
