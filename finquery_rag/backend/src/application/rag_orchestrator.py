"""RAG Orchestrator: coordinates retrieval, context building, and generation.

This is the top-level orchestration layer that ties together the retrieval
pipeline, context builder, sufficiency evaluator, deterministic answer
extractor, and LLM gateway into a single query flow.

The orchestrator exposes a typed boundary:
- Input:  ``QueryRequest`` (domain object)
- Output: ``AnswerResult`` (domain object)

The legacy ``query`` method is retained as a thin compatibility shim that
forwards to ``answer`` and unwraps the legacy dict. Production code
(``RAGEngine``) must use ``answer`` directly so the typed boundary is
exercised on every request.

Phase 3 Commit 8 adds an optional ``calculation_pipeline`` constructor
param. When set, the orchestrator invokes the deterministic calculation
pipeline after context build and before the LLM. If the pipeline returns
``EXECUTED`` or ``BLOCKED``, the LLM is bypassed and the rendered
calculation result is returned directly. When ``calculation_pipeline`` is
None (the default), the orchestrator behaves exactly as before.
"""

import re
import time

from src.domain.answer import AnswerPath, AnswerResult
from src.domain.calculation import CalculationStatus
from src.domain.evidence import EvidenceItem
from src.domain.query import QueryRequest
from src.finance.calculation_pipeline import CalculationPipeline
from src.finance.calculation_renderer import render_calculation_result
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
    8. Deterministic calculation pipeline (Phase 3, when enabled)
    9. Deterministic context answering
    10. LLM generation (when needed)
    11. Trace logging
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
        calculation_pipeline: CalculationPipeline | None = None,
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
        self._calculation_pipeline = calculation_pipeline

    # ------------------------------------------------------------------
    # Typed boundary (production entry point)
    # ------------------------------------------------------------------
    async def answer(
        self,
        request: QueryRequest,
        *,
        n_results: int = 3,
    ) -> AnswerResult:
        """Execute the full RAG pipeline and return a typed ``AnswerResult``.

        ``RAGEngine.query`` must call this method, not the legacy ``query``
        shim, so that ``QueryRequest`` and ``AnswerResult`` are exercised on
        every production request.
        """
        t0 = time.time()
        trace_data: dict = {
            "tenant_id": request.user_id,
            "query_original": request.question,
        }

        question = request.question
        if request.conversation_history:
            question = await self._llm_gateway.rewrite_query(
                question,
                list(request.conversation_history),
                request.memory_profile,
            )
            trace_data["query_rewritten"] = question

        intent = self._classify_intent(question)
        trace_data["intent"] = intent["intent"]

        had_history = bool(request.conversation_history)
        conversational_response = self._handle_conversational_query(question)
        if conversational_response:
            return self._build_conversational_result(
                answer=conversational_response,
                intent=intent,
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        if not intent["requires_retrieval"]:
            return self._build_no_retrieval_result(
                intent=intent,
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        doc_names: list[str]
        if request.document_names:
            doc_names = list(request.document_names)
        else:
            all_docs = self._list_all_documents(request.user_id)
            doc_names = [doc["name"] for doc in all_docs]

        if not doc_names:
            return self._build_no_documents_result(
                rewritten_question=question if had_history else None,
                had_conversation_history=had_history,
            )

        # 1. Retrieve relevant chunks. Front-matter facts use direct metadata lookup first.
        chunks = self._retrieve_front_matter_chunks(
            doc_names, question, request.user_id
        )
        if not chunks:
            if len(doc_names) == 1:
                chunks = self._retrieval_pipeline.retrieve_single(
                    document_name=doc_names[0],
                    query=question,
                    user_id=request.user_id,
                    top_k=n_results,
                )
            else:
                chunks = await self._retrieval_pipeline.retrieve_multiple(
                    document_names=doc_names,
                    query=question,
                    user_id=request.user_id,
                    top_k=n_results,
                )

        front_matter_answer = self._deterministic_extractor.answer_front_matter_query(
            question, chunks
        )
        deterministic_answer = None
        low_confidence_numeric_override = False
        calculation_result = None
        calculation_answer = None

        if front_matter_answer:
            chunks = front_matter_answer["chunks"]
            context, sources = self._context_builder.build(chunks)
            answer = front_matter_answer["answer"]
            is_sufficient = True
            confidence = 1.0
            deterministic_answer = front_matter_answer["diagnostic"]
        else:
            # Phase 3: Check context sufficiency
            sufficiency = self._sufficiency_evaluator.evaluate(chunks)
            is_sufficient = sufficiency.is_sufficient
            confidence = self._sufficiency_evaluator.confidence(chunks)

            # 2. Build context (with dedup and score threshold)
            context, sources = self._context_builder.build(chunks)

            # 3. Deterministic calculation pipeline (Phase 3 Commit 8).
            #    Runs after context build, before the deterministic context
            #    answer extractor / LLM. Bypasses the LLM on EXECUTED/BLOCKED.
            if self._calculation_pipeline is not None:
                evidence = tuple(EvidenceItem.from_chunk(c) for c in chunks)
                calculation_result = self._calculation_pipeline.try_calculate(
                    question,
                    intent,
                    evidence,
                )
                if calculation_result.status is not CalculationStatus.NOT_APPLICABLE:
                    calculation_answer = render_calculation_result(calculation_result)

            # If the calculation pipeline produced a bypass answer, use it
            # and skip the deterministic context extractor + LLM.
            if calculation_result is not None and calculation_result.status in (
                CalculationStatus.EXECUTED,
                CalculationStatus.BLOCKED,
                CalculationStatus.FAILED,
            ):
                answer = calculation_answer
                is_sufficient = True
                confidence = (
                    1.0
                    if calculation_result.status is CalculationStatus.EXECUTED
                    else 0.0
                    if calculation_result.status is CalculationStatus.FAILED
                    else confidence
                )
                deterministic_answer = {
                    "path": "calculation_pipeline",
                    "status": calculation_result.status.value,
                    "operation": (
                        calculation_result.operation.value
                        if calculation_result.operation
                        else None
                    ),
                    "formula_version": calculation_result.formula_version,
                    "error_code": calculation_result.error_code,
                }
            else:
                # 4. Deterministic context answering
                deterministic_context_answer = self._deterministic_extractor.answer_deterministic_query_from_context(
                    question,
                    context,
                    sources,
                )
                if deterministic_context_answer:
                    answer = deterministic_context_answer["answer"]
                    is_sufficient = True
                    deterministic_answer = deterministic_context_answer["diagnostic"]
                    low_confidence_numeric_override = False
                else:
                    low_confidence_numeric_override = (
                        self._query_processor.should_generate_with_low_confidence(
                            question,
                            chunks,
                            numeric_rrf_floor=self._numeric_rrf_floor,
                            numeric_dense_floor=self._numeric_dense_floor,
                        )
                    )
                    if low_confidence_numeric_override:
                        is_sufficient = True

                    if not is_sufficient:
                        answer = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
                    else:
                        answer = await self._llm_gateway.generate(context, question)

        # 5. Log trace
        elapsed_ms = (time.time() - t0) * 1000
        trace_data.update(
            {
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
                        low_confidence_numeric_override
                        if not front_matter_answer
                        else False
                    ),
                    "calculation": (
                        {
                            "status": calculation_result.status.value,
                            "operation": (
                                calculation_result.operation.value
                                if calculation_result.operation
                                else None
                            ),
                            "formula_version": calculation_result.formula_version,
                            "operand_count": len(calculation_result.operands),
                            "error_code": calculation_result.error_code,
                        }
                        if calculation_result is not None
                        and calculation_result.status
                        is not CalculationStatus.NOT_APPLICABLE
                        else None
                    ),
                },
                "model_name": self._model_name,
                "latency_ms": elapsed_ms,
            }
        )
        trace_id = None
        try:
            trace_id = self._trace_logger.log(**trace_data)
        except Exception:
            pass  # tracing must never break the query path

        # Build calculations tuple for AnswerResult (additive field).
        calculations: tuple[dict, ...] = ()
        if (
            calculation_result is not None
            and calculation_result.status is not CalculationStatus.NOT_APPLICABLE
        ):
            calculations = (calculation_result.to_dict(),)

        return AnswerResult(
            answer=answer,
            sources=tuple(sources),
            context=context,
            searched_docs=tuple(doc_names),
            confidence=confidence,
            context_sufficient=is_sufficient,
            intent=intent["intent"],
            intent_confidence=intent["confidence"],
            rewritten_question=question if had_history else None,
            retrieved_chunks=tuple(summarize_retrieved_chunks(chunks)),
            retrieval_debug=dict(self._retrieval_pipeline.last_retrieval_debug),
            trace_id=trace_id,
            path=AnswerPath.FULL,
            had_conversation_history=had_history,
            calculations=calculations,
        )

    # ------------------------------------------------------------------
    # Legacy compatibility shim (not for production use)
    # ------------------------------------------------------------------
    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int = None,
        n_results: int = 3,
        conversation_history: list = None,
        memory_profile: dict | None = None,
    ) -> dict:
        """Legacy entry point retained for tests that mock the orchestrator.

        Production code must call ``answer`` with a ``QueryRequest``.
        """
        request = QueryRequest(
            question=question,
            document_names=tuple(doc_names or ()),
            user_id=user_id,
            conversation_history=tuple(conversation_history or ()),
            memory_profile=memory_profile,
        )
        result = await self.answer(request, n_results=n_results)
        return result.to_legacy_dict()

    # ------------------------------------------------------------------
    # Branch constructors for early-return paths
    # ------------------------------------------------------------------
    @staticmethod
    def _build_conversational_result(
        *,
        answer: str,
        intent: dict,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer=answer,
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent="conversation",
            intent_confidence=intent["confidence"],
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=None,
            path=AnswerPath.CONVERSATIONAL,
            had_conversation_history=had_conversation_history,
        )

    @staticmethod
    def _build_no_retrieval_result(
        *,
        intent: dict,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer=(
                "This question appears to be outside the uploaded financial documents. "
                "Please ask about your uploaded reports or financial data."
            ),
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent=intent["intent"],
            intent_confidence=intent["confidence"],
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=None,
            path=AnswerPath.NO_RETRIEVAL,
            had_conversation_history=had_conversation_history,
        )

    @staticmethod
    def _build_no_documents_result(
        *,
        rewritten_question: str | None,
        had_conversation_history: bool = False,
    ) -> AnswerResult:
        return AnswerResult(
            answer="No documents found in database. Please upload documents first.",
            sources=(),
            context=None,
            searched_docs=(),
            confidence=None,
            context_sufficient=True,
            intent=None,
            intent_confidence=None,
            rewritten_question=rewritten_question,
            retrieved_chunks=(),
            retrieval_debug={},
            trace_id=None,
            path=AnswerPath.NO_DOCUMENTS,
            had_conversation_history=had_conversation_history,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _retrieve_front_matter_chunks(
        self,
        doc_names: list[str],
        query: str,
        user_id: int | None = None,
    ) -> list:
        """Direct metadata lookup for deterministic front-matter questions."""
        if (
            not self._query_processor.is_title_query(query)
            or not doc_names
            or user_id is None
        ):
            return []
        if "reporting period" in (query or "").lower():
            return []
        chunks = []
        for doc_name in doc_names:
            for chunk in self._get_front_matter_chunks(
                doc_name=doc_name, user_id=user_id, subtype="title"
            ):
                title = re.sub(
                    r"^title\s*:\s*", "", chunk.get("content", ""), flags=re.IGNORECASE
                ).strip()
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
            "revenue",
            "expense",
            "profit",
            "loss",
            "income",
            "cash",
            "balance",
            "debt",
            "equity",
            "margin",
            "growth",
            "quarter",
            "fiscal",
            "earnings",
            "dividend",
            "asset",
            "liability",
            "$",
            "%",
            "million",
            "billion",
            "q1",
            "q2",
            "q3",
            "q4",
            "fy",
            "yoy",
            "table",
            "page",
            "report",
            "statement",
            "cost",
            "营收",
            "利润",
            "亏损",
            "收入",
            "现金",
            "负债",
            "资产",
            "权益",
            "增长",
            "季度",
            "财报",
            "股息",
            "报表",
            "成本",
            "费用",
            "净利",
        ]
        if any(ind in query_lower for ind in financial_indicators):
            return None

        # Greetings
        greetings = [
            "hi",
            "hello",
            "hi there",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if (
            any(query_lower.startswith(g) for g in greetings)
            and len(query_lower.split()) <= 3
        ):
            return "Hello! I'm FinQuery, your financial document assistant. I can help you find information in your uploaded documents. What would you like to know?"

        # Identity questions
        identity_keywords = [
            "what are you",
            "who are you",
            "what is finquery",
            "tell me about yourself",
            "what do you do",
            "what can you do",
            "how do you work",
            "what's your purpose",
        ]
        if any(keyword in query_lower for keyword in identity_keywords):
            return "I'm FinQuery, an AI assistant that helps you analyze financial documents. Upload PDFs of reports, statements, or other financial documents, and I'll answer questions about them using the exact information from those documents."

        # Capability questions
        capability_keywords = [
            "how does this work",
            "how to use",
            "help me",
            "what can i ask",
            "how do i use this",
        ]
        if any(keyword in query_lower for keyword in capability_keywords):
            return "Here's how to use FinQuery:\n1. Upload financial documents (PDFs)\n2. Ask questions about the content\n3. I'll provide answers with page citations\n\nTry: 'What was the revenue in Q3?' or 'Summarize key financial metrics'"

        # Thanks/gratitude
        thanks_keywords = ["thank you", "thanks", "thx", "appreciate"]
        if (
            any(keyword in query_lower for keyword in thanks_keywords)
            and len(query_lower.split()) <= 5
        ):
            return "You're welcome! Let me know if you have any other questions about your documents."

        # Goodbyes
        goodbye_keywords = ["bye", "goodbye", "see you", "exit", "quit"]
        if (
            any(keyword in query_lower for keyword in goodbye_keywords)
            and len(query_lower.split()) <= 3
        ):
            return "Goodbye! Feel free to come back anytime you need to analyze financial documents."

        return None
