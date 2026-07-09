#!/usr/bin/env python3
"""
OpenAI-compatible API adapter for NanoChat.

Wraps the native NanoChat engine and exposes a standard OpenAI /v1/chat/completions
endpoint, making it directly usable with the official `openai` Python client,
LangChain, LlamaIndex, and any framework that speaks the OpenAI protocol.

Key features:
  - Full OpenAI /v1/chat/completions compatibility (streaming + non-streaming)
  - System prompt support (merged into first user message, matching training convention)
  - Safe multi-byte UTF-8 streaming (no garbled Chinese/emoji output)
  - Multi-GPU worker pool with async request distribution
  - /v1/models endpoint for model discovery

Architecture:
  [Client (openai SDK / LangChain)]
      |  (standard OpenAI format)
  [This server: /v1/chat/completions]
      |  (internal token state machine)
  [NanoChat Engine + GPU Worker Pool]

Launch examples:
  # Single GPU
  python -m scripts.chat_openai_compat

  # 4 GPUs
  python -m scripts.chat_openai_compat --num-gpus 4

  # Custom model name
  python -m scripts.chat_openai_compat --model-name financial-llm-7b

Client usage:
  from openai import OpenAI
  client = OpenAI(base_url="http://your-server:8998/v1", api_key="not-needed")
  response = client.chat.completions.create(
      model="nanochat",
      messages=[{"role": "user", "content": "Hello!"}],
      stream=True
  )
"""

import argparse
import json
import time
import uuid
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.chat_format import encode_chat_prompt
from nanochat.engine import Engine

# ---------------------------------------------------------------------------
# Abuse prevention limits
# ---------------------------------------------------------------------------
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='NanoChat OpenAI-Compatible API Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8998, help='Port to run the server on (default: 8998, FRP maps to cloud 8500)')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
parser.add_argument('--model-name', type=str, default='nanochat', help='Model name exposed in OpenAI API')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type: cuda|cpu|mps. empty => autodetect')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device init
# ---------------------------------------------------------------------------
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)


# ---------------------------------------------------------------------------
# Worker Pool (same pattern as chat_web.py)
# ---------------------------------------------------------------------------
@dataclass
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object


class WorkerPool:
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        print(f"Initializing worker pool with {self.num_gpus} GPU(s)...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs."

        for gpu_id in range(self.num_gpus):
            if device_type == "cuda":
                dev = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                dev = torch.device(device_type)
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, dev, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(gpu_id=gpu_id, device=dev, engine=engine, tokenizer=tokenizer)
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} worker(s) initialized!")

    async def acquire_worker(self) -> Worker:
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        await self.available_workers.put(worker)


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="nanochat")
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = Field(default=None, description="NanoChat-specific: top-k sampling")
    stream: Optional[bool] = False
    # Fields accepted but ignored for compatibility
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_chat_request(request: ChatCompletionRequest):
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} allowed")

    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(status_code=400, detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters")
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Total conversation too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters")

    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant", "system"]:
            raise HTTPException(status_code=400, detail=f"Message {i} has invalid role '{message.role}'. Must be 'user', 'assistant', or 'system'")

    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(status_code=400, detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")

    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(status_code=400, detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}")

    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(status_code=400, detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}")


# ---------------------------------------------------------------------------
# Token state machine: build conversation tokens from messages
# ---------------------------------------------------------------------------
def build_conversation_tokens(worker: Worker, messages: List[ChatMessage], token_budget: int) -> List[int]:
    """
    Build the NanoChat token state machine from OpenAI-style messages.

    Handles system prompts by merging them into the first user message,
    matching the convention used in tokenizer.render_conversation() during training.

    The shared helper also handles prompt truncation for long RAG contexts.
    """
    return encode_chat_prompt(
        worker.tokenizer,
        [{"role": message.role, "content": message.content} for message in messages],
        token_budget=token_budget,
    )


def next_generation_step(sync_gen):
    """Return the next generation item, or None when the generator is exhausted."""
    try:
        return next(sync_gen)
    except StopIteration:
        return None


# ---------------------------------------------------------------------------
# Streaming generation with OpenAI SSE format
# ---------------------------------------------------------------------------
async def generate_openai_stream(
    worker: Worker,
    conversation_tokens: List[int],
    request_id: str,
    model_name: str,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate assistant response and yield OpenAI-compatible SSE chunks.

    Handles multi-byte UTF-8 safely (same technique as chat_web.py):
    accumulate tokens, decode the full sequence, only emit when we have
    a clean UTF-8 boundary (no trailing replacement character).
    """
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    created = int(time.time())

    # Run the synchronous generator in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()

    accumulated_tokens: List[int] = []
    last_clean_text = ""
    finish_reason = "stop"
    completion_tokens = 0

    # Create the synchronous generator
    sync_gen = worker.engine.generate(
        conversation_tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    )

    try:
        while True:
            # Run one step of the synchronous generator in a thread
            item = await loop.run_in_executor(None, next_generation_step, sync_gen)
            if item is None:
                break
            token_column, token_masks = item

            token = token_column[0]

            # Stopping criteria
            if token == assistant_end or token == bos:
                break

            # Accumulate tokens for safe UTF-8 decoding
            accumulated_tokens.append(token)
            completion_tokens += 1
            current_text = worker.tokenizer.decode(accumulated_tokens)

            # Only emit if we don't end with a replacement character (incomplete UTF-8)
            if not current_text.endswith('\ufffd'):
                new_text = current_text[len(last_clean_text):]
                if new_text:
                    # Yield OpenAI-format SSE chunk
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": new_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

        # Check if we hit max_tokens limit
        if completion_tokens >= max_new_tokens:
            finish_reason = "length"

    except Exception as e:
        logger.error(f"Generation error: {e}")
        finish_reason = "stop"

    # Send final chunk with finish_reason
    final_chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

    # Log the full response
    full_response = last_clean_text
    logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response} [{finish_reason}]")


# ---------------------------------------------------------------------------
# Non-streaming generation
# ---------------------------------------------------------------------------
async def generate_openai_non_stream(
    worker: Worker,
    conversation_tokens: List[int],
    request_id: str,
    model_name: str,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
) -> ChatCompletionResponse:
    """
    Generate a complete assistant response (non-streaming).
    """
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    loop = asyncio.get_event_loop()

    accumulated_tokens: List[int] = []
    finish_reason = "stop"

    sync_gen = worker.engine.generate(
        conversation_tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1),
    )

    try:
        while True:
            item = await loop.run_in_executor(None, next_generation_step, sync_gen)
            if item is None:
                break
            token_column, token_masks = item

            token = token_column[0]
            if token == assistant_end or token == bos:
                break
            accumulated_tokens.append(token)

            if len(accumulated_tokens) >= max_new_tokens:
                finish_reason = "length"
                break

    except Exception as e:
        logger.error(f"Generation error: {e}")
        finish_reason = "stop"

    full_text = worker.tokenizer.decode(accumulated_tokens) if accumulated_tokens else ""

    logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_text} [{finish_reason}]")

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=full_text),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=len(conversation_tokens),
            completion_tokens=len(accumulated_tokens),
            total_tokens=len(conversation_tokens) + len(accumulated_tokens),
        ),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading NanoChat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"OpenAI-compatible API server ready at http://localhost:{args.port}")
    print(f"Model name: {args.model_name}")
    print(f"Endpoint: /v1/chat/completions")
    yield


app = FastAPI(title="NanoChat OpenAI-Compatible API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0,
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": args.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nanochat",
            }
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """OpenAI-compatible model detail endpoint."""
    if model_id != args.model_name:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "id": args.model_name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "nanochat",
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.

    Supports both streaming (stream=True) and non-streaming (stream=False) modes.
    System prompts are automatically merged into the first user message.
    """
    validate_chat_request(request)

    # Log incoming conversation
    logger.info("=" * 20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content[:200]}{'...' if len(message.content) > 200 else ''}")
    logger.info(f"[STREAM]: {request.stream}")
    logger.info("-" * 20)

    # Acquire a worker
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        max_new_tokens = request.max_tokens if request.max_tokens is not None else args.max_tokens
        if max_new_tokens >= worker.engine.model.config.sequence_len:
            raise HTTPException(
                status_code=400,
                detail="max_tokens must be smaller than the model context length",
            )
        prompt_budget = worker.engine.model.config.sequence_len - max_new_tokens
        try:
            conversation_tokens = build_conversation_tokens(
                worker,
                request.messages,
                token_budget=prompt_budget,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        model_name = args.model_name

        if request.stream:
            # Streaming response
            async def stream_and_release():
                try:
                    async for chunk in generate_openai_stream(
                        worker,
                        conversation_tokens,
                        request_id,
                        model_name,
                        temperature=request.temperature,
                        max_new_tokens=request.max_tokens,
                        top_k=request.top_k,
                    ):
                        yield chunk
                finally:
                    await worker_pool.release_worker(worker)

            return StreamingResponse(
                stream_and_release(),
                media_type="text/event-stream",
            )
        else:
            # Non-streaming response
            try:
                response = await generate_openai_non_stream(
                    worker,
                    conversation_tokens,
                    request_id,
                    model_name,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k,
                )
                return response
            finally:
                await worker_pool.release_worker(worker)

    except HTTPException:
        await worker_pool.release_worker(worker)
        raise
    except Exception as e:
        await worker_pool.release_worker(worker)
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Also register at /chat/completions for backward compatibility with chat_web.py clients
@app.post("/chat/completions")
async def chat_completions_legacy(request: ChatCompletionRequest):
    """Legacy endpoint (same as /v1/chat/completions)."""
    return await chat_completions(request)


@app.get("/stats")
async def stats():
    """Worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {"gpu_id": w.gpu_id, "device": str(w.device)}
            for w in worker_pool.workers
        ],
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat OpenAI-Compatible API Server")
    print(f"Model name: {args.model_name}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
