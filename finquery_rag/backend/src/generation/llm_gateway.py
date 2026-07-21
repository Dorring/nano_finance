"""LLM gateway for answer generation (streaming and non-streaming).

The gateway is the only module that talks to the underlying LLM client.
Query rewriting is exposed here so the orchestrator does not have to reach
into private attributes (``_llm_client`` / ``_model_name``) to perform a
rewrite.
"""
import asyncio

from src.generation.prompt_builder import get_system_prompt
from src.generation.response_renderer import validate_answer
from src.retrieval.query_processor import QueryProcessor


class LLMGateway:
    """Wraps LLM client calls for answer generation and query rewriting."""

    def __init__(
        self,
        *,
        llm_client,
        model_name: str,
        max_new_tokens: int = 512,
        query_processor: QueryProcessor | None = None,
    ):
        self._llm_client = llm_client
        self._model_name = model_name
        self._max_new_tokens = max_new_tokens
        self._query_processor = query_processor or QueryProcessor()

    @property
    def model_name(self) -> str:
        """Public read-only accessor for the configured model name."""
        return self._model_name

    @property
    def max_new_tokens(self) -> int:
        """Public read-only accessor for the max generation token budget."""
        return self._max_new_tokens

    async def rewrite_query(
        self,
        question: str,
        conversation_history: list,
        memory_profile: dict | None = None,
    ) -> str:
        """Rewrite a follow-up question into a standalone search query.

        Thin wrapper around ``QueryProcessor.rewrite`` so the orchestrator
        does not access the gateway's private LLM client and model name.
        """
        return await self._query_processor.rewrite(
            question,
            conversation_history,
            memory_profile,
            llm_client=self._llm_client,
            model_name=self._model_name,
        )

    async def generate(self, context: str, query: str) -> str:
        """Generate answer using LLM (non-streaming, async)."""
        if not context:
            return "I couldn't find relevant information in the documents to answer your question."

        system_prompt = get_system_prompt()
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: self._llm_client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,
                    max_tokens=self._max_new_tokens
                )
            )
            raw_answer = response.choices[0].message.content
            return validate_answer(raw_answer, [], max_new_tokens=self._max_new_tokens)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def generate_stream(self, context: str, query: str):
        """Generate answer using LLM (streaming)."""
        if not context:
            yield "I couldn't find relevant information in the documents to answer your question."
            return

        system_prompt = get_system_prompt()
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,
                max_tokens=self._max_new_tokens,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error generating answer: {str(e)}"
