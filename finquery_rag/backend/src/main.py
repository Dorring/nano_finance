from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
import json
import time

from .services.auth import create_access_token, get_current_user, get_current_user_optional, get_password_hash, verify_password
from .services.ingest import process_pdf
from .services.vector_store import add_documents, list_all_documents, delete_document_collection, get_collection_stats, clear_all_for_user
from .services.rag_engine import RAGEngine
from .services.document_registry import DocumentRegistry, VALID_TRANSITIONS
from .services.session_manager import SessionManager
from .services.health import collect_health_snapshot
from .services.intent import classify_query_intent
from .services.feedback import FeedbackStore
from .services.query_scope import resolve_query_document_names
from .services.streaming import make_stream_done_event, make_stream_error_event, safe_log_query_trace
from .models.schemas import *  #全部 Pydantic 模型
from .models.user import User #User ORM 模型
from .database import get_db, engine, Base #SQLAlchemy 数据库连接和基础模型
from sqlalchemy.orm import Session #SQLAlchemy 会话管理

from datetime import timedelta, datetime
import os
import uuid
import shutil
import tempfile
from dotenv import load_dotenv

# Create database tables (if relying on this instead of alembic for initial dev)
#创建数据库表（如果不使用 alembic 进行迁移的话）
Base.metadata.create_all(bind=engine)

# Disable tokenizer parallelism to avoid fork warnings
# 禁用分词器并行化以避免 fork 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize FastAPI
# 初始化 FastAPI 应用
app = FastAPI(
    title="FinQuery API",
    description="Multi-Document Financial Q&A System with User Management",
    version="3.0.0"
)

# Get allowed origins from environment variable
# 从环境变量获取允许的跨域请求来源
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

# Add CORS middleware
# 添加跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 加载环境变量（必须在读取环境变量之前）
load_dotenv()

# Initialize LLM clients
# 在线对话：通过 FRP 穿透访问学校 GPU 服务器上的 nanochat OpenAI 兼容适配层
llm_client = OpenAI(
    base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
    api_key=os.getenv("LLM_API_KEY", "not-needed-for-local")
)
llm_model_name = os.getenv("LLM_MODEL_NAME", "nanochat")

rag_engine: RAGEngine | None = None
document_registry = DocumentRegistry()
session_manager = SessionManager()
feedback_store = FeedbackStore()

def get_rag_engine():
    """
    延迟初始化 RAG 引擎。
    当需要使用时才进行实例化，以节省系统资源。

    Returns:
        RAGEngine: 初始化后的 RAGEngine 实例。
    """
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine(
            llm_client,
            model_name=llm_model_name,
            use_hybrid=True,
            reranker_name=os.getenv("RAG_RERANKER"),
            reranker_model=os.getenv("RAG_RERANKER_MODEL"),
            retrieval_candidate_multiplier=int(os.getenv("RAG_CANDIDATE_MULTIPLIER", "2")),
        )
    return rag_engine

def api_error(status_code: int, error_code: str, message: str) -> HTTPException:
    """Build a stable business-error response envelope."""
    return HTTPException(
        status_code=status_code,
        detail={"error_code": error_code, "message": message},
    )


def _normalize_api_pagination(limit, offset, default_limit=50, max_limit=1000):
    """Normalize API pagination and reject ambiguous negative values."""
    try:
        normalized_limit = int(limit if limit is not None else default_limit)
        normalized_offset = int(offset or 0)
    except (TypeError, ValueError):
        raise api_error(400, "invalid_pagination", "limit and offset must be integers")
    if normalized_limit < 1:
        raise api_error(400, "invalid_pagination", "limit must be >= 1")
    if normalized_offset < 0:
        raise api_error(400, "invalid_pagination", "offset must be >= 0")
    return min(normalized_limit, max_limit), normalized_offset


def _public_registry_document(row: dict) -> dict:
    """Return document lifecycle fields without file/content hashes."""
    keys = [
        "document_id",
        "filename",
        "chunk_count",
        "page_count",
        "version",
        "status",
        "parser_version",
        "splitter_version",
        "embedding_version",
        "created_at",
        "updated_at",
        "error_message",
    ]
    return {key: row.get(key) for key in keys}


def _json_field(value):
    """Decode trace JSON columns while tolerating legacy/plain values."""
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def _public_feedback(row: dict) -> dict:
    """Return feedback data without exposing tenant_id."""
    keys = ["feedback_id", "trace_id", "rating", "comment", "created_at"]
    return {key: row.get(key) for key in keys}



def _validate_session_id(session_id: str) -> str:
    """Validate path/body session IDs before touching session storage."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise api_error(400, "invalid_session_id", "session_id must be 1-128 characters")
    return session_id


def _assistant_session_metadata(result=None, sources=None, trace_id=None, context_sufficient=None,
                                confidence=None, intent=None, intent_confidence=None):
    """Build UI-facing metadata for persisted assistant session messages."""
    result = result or {}
    return {
        "sources": sources if sources is not None else result.get("sources", []),
        "diagnostics": {
            "traceId": trace_id if trace_id is not None else result.get("trace_id"),
            "contextSufficient": (
                context_sufficient if context_sufficient is not None else result.get("context_sufficient")
            ),
            "retrievalConfidence": confidence if confidence is not None else result.get("confidence"),
            "intent": intent if intent is not None else result.get("intent"),
            "intentConfidence": intent_confidence if intent_confidence is not None else result.get("intent_confidence"),
        },
    }


def _resolve_query_document_names_for_user(user_id, requested_doc_names):
    """Return ready document names for query, rejecting stale/unready filters."""
    ready_names = [
        row.get("filename")
        for row in document_registry.list_documents(user_id)
        if row.get("filename")
    ]
    fallback_names = []
    if not ready_names:
        fallback_names = [
            row.get("name")
            for row in list_all_documents(user_id)
            if row.get("name")
        ]

    resolved, invalid = resolve_query_document_names(
        requested_doc_names,
        ready_names,
        fallback_names,
    )
    if invalid:
        raise api_error(
            400,
            "documents_not_ready",
            "Documents are not ready or not found: %s" % ", ".join(invalid),
        )
    return resolved


def _public_trace(row: dict) -> dict:
    """Return tenant-scoped trace data without exposing tenant_id."""
    keys = [
        "trace_id",
        "query_original",
        "query_rewritten",
        "intent",
        "final_context",
        "answer",
        "model_name",
        "prompt_version",
        "index_version",
        "latency_ms",
        "error_message",
        "created_at",
    ]
    trace = {key: row.get(key) for key in keys}
    trace["filter_conditions"] = _json_field(row.get("filter_conditions"))
    trace["candidates"] = _json_field(row.get("candidates_json"))
    trace["sources"] = _json_field(row.get("sources_json")) or []
    trace["diagnostics"] = _json_field(row.get("diagnostics_json")) or {}
    return trace


######################### API Endpoints #########################

# <---------------------- GET requests ---------------------->
@app.get("/")
async def root():
    """
    健康检查端点。
    用于验证服务是否正常运行。

    Returns:
        dict: 包含服务状态、名称和版本号的字典。
    """
    return {
        "status": "healthy",
        "service": "FinQuery Multi-Document API",
        "version": "2.0.0"
    }

@app.get("/healthz")
async def healthz():
    """Lightweight liveness probe."""
    return {
        "status": "healthy",
        "service": "FinQuery Multi-Document API",
        "version": app.version,
    }

@app.get("/readyz")
async def readyz():
    """Readiness probe with non-secret RAG dependency diagnostics."""
    snapshot = collect_health_snapshot(
        document_registry=document_registry,
        session_manager=session_manager,
    )
    status_code = 200 if snapshot["ready"] else 503
    return JSONResponse(status_code=status_code, content=snapshot)

@app.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    获取当前登录用户的信息。
    通过 JWT 依赖注入直接获取 User 对象，无需额外查询数据库。

    Args:
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含用户邮箱和账户创建时间的字典。
    """
    return {
        "email": current_user.email,
        "created_at": current_user.created_at
    }

@app.get("/documents", response_model=DocumentsListResponse)
async def list_documents(current_user: User = Depends(get_current_user)):
    """
    列出当前用户上传的所有文档。
    如果向量数据库目录不存在，则返回空列表。

    Args:
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        DocumentsListResponse: 包含文档信息列表和文档总数的响应模型。
    """
    if not os.path.exists("./chroma_db"):
        return DocumentsListResponse(documents=[], total_documents=0)

    docs = list_all_documents(current_user.id)

    return DocumentsListResponse(
        documents=[DocumentInfo(**doc) for doc in docs],
        total_documents=len(docs)
    )

@app.get("/document-registry")
async def list_document_registry(status: str | None = None, current_user: User = Depends(get_current_user)):
    """List current user's document lifecycle registry rows."""
    if status is not None and status not in VALID_TRANSITIONS:
        raise api_error(400, "invalid_document_status", f"Invalid document status: {status}")

    rows = document_registry.list_all(current_user.id, status=status)
    return {
        "documents": [_public_registry_document(row) for row in rows],
        "total_documents": len(rows),
        "status_counts": document_registry.status_counts(current_user.id),
    }

@app.get("/traces")
async def list_query_traces(
    limit: int = 20,
    offset: int = 0,
    error_only: bool = False,
    created_after: float | None = None,
    created_before: float | None = None,
    current_user: User = Depends(get_current_user),
):
    """List current user's recent query traces for troubleshooting/replay."""
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=20)
    if created_after is not None and created_before is not None and created_after > created_before:
        raise api_error(400, "invalid_time_range", "created_after must be <= created_before")
    rows = get_rag_engine().trace_logger.query_traces(
        tenant_id=current_user.id,
        limit=normalized_limit,
        offset=normalized_offset,
        created_after=created_after,
        created_before=created_before,
        error_only=error_only,
    )
    return {
        "traces": [_public_trace(row) for row in rows],
        "total_returned": len(rows),
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@app.get("/traces/{trace_id}")
async def get_query_trace(trace_id: str, current_user: User = Depends(get_current_user)):
    """Return one query trace when it belongs to the current user."""
    if not trace_id or len(trace_id) > 128:
        raise api_error(400, "invalid_trace_id", "Invalid trace_id")

    row = get_rag_engine().trace_logger.get_trace_for_tenant(current_user.id, trace_id)
    if row is None:
        raise api_error(404, "trace_not_found", "Trace not found")
    return {"trace": _public_trace(row)}


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_answer_feedback(request: FeedbackRequest, current_user: User = Depends(get_current_user)):
    """Store current user's answer feedback after validating trace ownership."""
    trace = get_rag_engine().trace_logger.get_trace_for_tenant(current_user.id, request.trace_id)
    if trace is None:
        raise api_error(404, "trace_not_found", "Trace not found")

    row = feedback_store.submit(
        tenant_id=current_user.id,
        trace_id=request.trace_id,
        rating=request.rating,
        comment=request.comment,
    )
    if row is None:
        raise api_error(400, "invalid_feedback", "Invalid feedback")
    return _public_feedback(row)


@app.get("/feedback")
async def list_answer_feedback(
    limit: int = 50,
    offset: int = 0,
    rating: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """List current user's answer feedback for review/export."""
    if rating is not None and rating not in FeedbackStore.VALID_RATINGS:
        raise api_error(400, "invalid_rating", "Invalid rating")
    normalized_limit, normalized_offset = _normalize_api_pagination(limit, offset, default_limit=50)
    rows = feedback_store.list_for_tenant(current_user.id, limit=normalized_limit, offset=normalized_offset, rating=rating)
    return {
        "feedback": [_public_feedback(row) for row in rows],
        "total_returned": len(rows),
        "limit": normalized_limit,
        "offset": normalized_offset,
    }


@app.get("/documents/{doc_name}")
async def get_document_stats(doc_name: str, current_user: User = Depends(get_current_user)):
    """
    获取特定文档的统计信息。
    根据文档名称和用户 ID 查询对应向量库集合的统计信息。

    Args:
        doc_name (str): 文档名称。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含文档统计信息的字典（如块数量等）。

    Raises:
        HTTPException: 如果指定文档不存在，抛出 404 异常。
    """
    stats = get_collection_stats(doc_name, current_user.id)

    if not stats["exists"]:
        raise api_error(404, "document_not_found", f"Document '{doc_name}' not found")

    return stats


# <---------------------- POST requests ---------------------->
@app.post("/register", response_model=Token)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """
    注册新用户。
    检查邮箱是否已被注册，若未注册则对密码进行哈希处理并存入数据库，随后生成并返回 JWT。

    Args:
        user (UserRegister): 用户注册请求体，包含邮箱和密码。
        db (Session): SQLAlchemy 数据库会话，通过依赖注入获取。

    Returns:
        dict: 包含访问令牌、令牌类型和用户邮箱的字典。

    Raises:
        HTTPException: 如果邮箱已被注册，抛出 400 异常。
    """
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(400, "Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        email=user.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(
        data={"sub": str(new_user.id)},
        expires_delta=timedelta(minutes=30)
    )

    print(f"✓ New user registered: {new_user.email} (ID: {new_user.id})")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": new_user.email
    }

@app.post("/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """
    用户登录。
    验证邮箱和密码，如果验证成功则生成并返回 JWT。

    Args:
        user (UserLogin): 用户登录请求体，包含邮箱和密码。
        db (Session): SQLAlchemy 数据库会话，通过依赖注入获取。

    Returns:
        dict: 包含访问令牌、令牌类型和用户邮箱的字典。

    Raises:
        HTTPException: 如果邮箱不存在或密码错误，抛出 401 异常。
    """
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user:
        raise HTTPException(401, "Invalid email or password")

    if not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(401, "Invalid email or password")

    access_token = create_access_token(
        data={"sub": str(db_user.id)},
        expires_delta=timedelta(minutes=30)
    )

    print(f"✓ User logged in: {db_user.email} (ID: {db_user.id})")

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": db_user.email
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """
    上传并处理 PDF 文档。
    仅支持 PDF 格式。文件会被临时保存，提取文本块后存入当前用户专属的向量集合中，最后删除临时文件并同步稀疏索引。

    Args:
        file (UploadFile): 上传的文件对象，默认通过表单数据获取。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        UploadResponse: 包含文件名、处理消息、页数及向量化结果的响应模型。

    Raises:
        HTTPException: 如果文件非 PDF 格式抛出 400 异常；如果 PDF 未提取到内容抛出 400 异常；处理过程中发生其他错误抛出 500 异常。
    """
    # Validate file type
    # 验证文件类型
    if not file.filename.endswith(".pdf"):
        raise api_error(400, "invalid_file_type", "Only PDF files are supported")

    safe_filename = os.path.basename(file.filename)
    if not safe_filename:
        raise api_error(400, "invalid_filename", "Invalid filename")
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, safe_filename)

    # Save file temporarily and process it
    # 临时保存文件并进行处理
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Phase 1: compute file hash and check for duplicate
        with open(temp_path, "rb") as f:
            file_bytes = f.read()
        fh = DocumentRegistry.file_hash(file_bytes)
        existing = document_registry.find_by_file_hash(current_user.id, fh)
        if existing:
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return UploadResponse(
                filename=file.filename,
                message="Duplicate file skipped (already indexed)",
                pages=existing["page_count"],
                collection_name="",
                total_docs=existing["chunk_count"],
            )

        # Register document in lifecycle registry
        doc_id = uuid.uuid4().hex
        document_registry.register(
            doc_id, current_user.id, safe_filename, fh, status="parsing"
        )

        # process pdf
        chunks, no_of_pages = process_pdf(temp_path, user_id=current_user.id)

        if not chunks:
            document_registry.mark_failed(doc_id, "No content extracted from PDF")
            raise api_error(400, "empty_document", "No content extracted from PDF")

        # Phase 1: content hash dedup
        ch = DocumentRegistry.content_hash(chunks)
        content_existing = document_registry.find_by_content_hash(current_user.id, ch)
        if content_existing and content_existing["document_id"] != doc_id:
            document_registry.mark_failed(doc_id, "Duplicate content (same as %s)" % content_existing["filename"])
            os.remove(temp_path)
            os.rmdir(temp_dir)
            return UploadResponse(
                filename=file.filename,
                message="Duplicate content skipped (matches %s)" % content_existing["filename"],
                pages=content_existing["page_count"],
                collection_name="",
                total_docs=content_existing["chunk_count"],
            )

        document_registry.mark_indexing(doc_id)

        # add to vector store
        result = add_documents(chunks, safe_filename, current_user.id, no_of_pages)

        # sync to BM25 (rollback dense on failure)
        try:
            engine = get_rag_engine()
            engine.bm25_retriever.add_chunks(chunks, current_user.id)
        except Exception as bm25_err:
            # Rollback ChromaDB dense data for this doc
            delete_document_collection(safe_filename, current_user.id)
            document_registry.mark_failed(doc_id, f"BM25 write failed, dense rolled back: {bm25_err}")
            raise api_error(500, "indexing_error", f"Indexing error: {bm25_err}")

        # Mark ready in registry
        document_registry.mark_ready(doc_id, len(chunks), ch)

        os.remove(temp_path)
        os.rmdir(temp_dir)

        print("\u2713 Document uploaded: %s (user: %s, doc_id: %s)" % (safe_filename, current_user.id, doc_id))

        return UploadResponse(
            filename=file.filename,
            message="Successfully processed %s" % file.filename,
            pages=no_of_pages,
            **result,
        )

    except HTTPException:
        # Mark failed in registry if doc_id exists
        try:
            document_registry.mark_failed(doc_id, "Client error")
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        try:
            document_registry.mark_failed(doc_id, str(e))
        except Exception:
            pass
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise api_error(500, "processing_error", "Processing error: %s" % str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    对一个或多个文档进行提问。
    支持会话记忆：传入 session_id 将自动加载历史消息用于问题改写。
    历史回答不会作为金融事实进入检索上下文。
    """
    try:
        engine = get_rag_engine()

        # Phase 4: Load conversation history for query rewriting
        conversation_history = None
        if request.session_id:
            conversation_history = session_manager.get_recent_messages(
                request.session_id, current_user.id
            )

        resolved_doc_names = _resolve_query_document_names_for_user(
            current_user.id,
            request.document_names,
        )

        # Run RAG pipeline
        result = await engine.query(
            question=request.question,
            doc_names=resolved_doc_names,
            n_results=request.n_results,
            user_id=current_user.id,
            conversation_history=conversation_history,
        )

        # Phase 4: Save messages to session
        rewritten = result.get("rewritten_question")
        if request.session_id:
            session_manager.add_message(request.session_id, current_user.id, "user", request.question)
            session_manager.add_message(
                request.session_id,
                current_user.id,
                "assistant",
                result["answer"],
                metadata=_assistant_session_metadata(result=result),
            )

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            question=request.question,
            searched_docs=result["searched_docs"],
            session_id=request.session_id,
            rewritten_question=rewritten,
            confidence=result.get("confidence"),
            context_sufficient=result.get("context_sufficient"),
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence"),
            trace_id=result.get("trace_id"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise api_error(500, "query_error", f"Query error: {str(e)}")



@app.post("/query/stream")
async def query_documents_stream(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    使用服务器发送事件（SSE）流式传输查询响应。
    统一共享 /query 的改写、充分性判断和 session 行为。

    Args:
        request (QueryRequest): 查询请求体。
        current_user (User): 当前登录用户。
    """
    resolved_doc_names = _resolve_query_document_names_for_user(
        current_user.id,
        request.document_names,
    )

    async def generate():
        trace_id = None
        try:
            engine = get_rag_engine()
            started_at = time.time()
            trace_data = {
                "tenant_id": current_user.id,
                "query_original": request.question,
                "model_name": llm_model_name,
            }

            def finish_trace(answer, sources=None, doc_names=None, chunks=None, context=None, diagnostics=None):
                elapsed_ms = (time.time() - started_at) * 1000
                trace_data.update({
                    "filter_conditions": {"doc_names": doc_names or [], "n_results": request.n_results},
                    "candidates": [
                        {
                            "doc_id": c.get("doc_id", ""),
                            "score": c.get("score", 0),
                            "rerank_score": c.get("rerank_score"),
                            "reranker": c.get("reranker"),
                        }
                        for c in (chunks or [])
                    ],
                    "final_context": context,
                    "answer": answer,
                    "sources": sources or [],
                    "diagnostics": diagnostics,
                    "latency_ms": elapsed_ms,
                })
                return safe_log_query_trace(engine, trace_data)

            # Phase 4: Load conversation history and rewrite query
            question = request.question
            conversation_history = None
            if request.session_id:
                conversation_history = session_manager.get_recent_messages(
                    request.session_id, current_user.id
                )
            if conversation_history:
                question = await engine._rewrite_query_with_context(question, conversation_history)
                trace_data["query_rewritten"] = question

            intent = classify_query_intent(question)
            trace_data["intent"] = intent["intent"]

            # Phase 3: Check if conversational (no RAG needed)
            conversational = engine._handle_conversational_query(question)
            if conversational:
                trace_id = finish_trace(conversational)
                yield f"data: {json.dumps({'type': 'token', 'content': conversational})}\n\n"
                yield make_stream_done_event(sources=[], context_sufficient=True, intent='conversation', intent_confidence=intent['confidence'], trace_id=trace_id)
                return

            if not intent["requires_retrieval"]:
                refusal = "This question appears to be outside the uploaded financial documents. Please ask about your uploaded reports or financial data."
                trace_id = finish_trace(refusal)
                yield f"data: {json.dumps({'type': 'token', 'content': refusal})}\n\n"
                yield make_stream_done_event(sources=[], context_sufficient=True, intent=intent['intent'], intent_confidence=intent['confidence'], trace_id=trace_id)
                return

            # Get document names resolved from ready lifecycle state.
            doc_names = resolved_doc_names

            if not doc_names:
                answer = 'No documents found. Please upload documents first.'
                trace_id = finish_trace(answer, doc_names=[])
                yield f"data: {json.dumps({'type': 'token', 'content': answer})}\n\n"
                yield make_stream_done_event(sources=[], context_sufficient=True, intent=intent['intent'], intent_confidence=intent['confidence'], trace_id=trace_id)
                return

            # Phase 3: Retrieve chunks and check sufficiency
            if len(doc_names) == 1:
                chunks = engine.retrieve_single_document(doc_names[0], question, current_user.id, request.n_results)
            else:
                chunks = await engine.retrieve_multiple_documents(doc_names, question, current_user.id, request.n_results)

            is_sufficient, best_score, avg_score = engine._check_context_sufficiency(chunks)
            confidence = engine._compute_confidence(chunks)

            # Phase 3: Build context
            context, sources = engine.build_context(chunks)

            # Phase 3: If context is insufficient, return refusal without calling LLM
            if not is_sufficient:
                refusal = "I couldn't find sufficiently relevant information in the documents to answer this question reliably."
                if request.session_id:
                    session_manager.add_message(request.session_id, current_user.id, "user", request.question)
                diagnostics = {
                    "confidence": confidence,
                    "context_sufficient": False,
                    "intent_confidence": intent["confidence"],
                }
                trace_id = finish_trace(refusal, sources=sources, doc_names=doc_names, chunks=chunks, context=context, diagnostics=diagnostics)
                if request.session_id:
                    session_manager.add_message(
                        request.session_id,
                        current_user.id,
                        "assistant",
                        refusal,
                        metadata=_assistant_session_metadata(
                            sources=sources,
                            trace_id=trace_id,
                            context_sufficient=False,
                            confidence=confidence,
                            intent=intent["intent"],
                            intent_confidence=intent["confidence"],
                        ),
                    )
                yield f"data: {json.dumps({'type': 'token', 'content': refusal})}\n\n"
                yield make_stream_done_event(sources=sources, context_sufficient=False, confidence=confidence, intent=intent['intent'], intent_confidence=intent['confidence'], trace_id=trace_id)
                return

            # Phase 4: Save user question to session
            if request.session_id:
                session_manager.add_message(request.session_id, current_user.id, "user", request.question)

            # Stream LLM response
            full_answer = ""
            for token in engine.generate_answer_stream(context, question):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            diagnostics = {
                "confidence": confidence,
                "context_sufficient": True,
                "intent_confidence": intent["confidence"],
            }
            trace_id = finish_trace(full_answer, sources=sources, doc_names=doc_names, chunks=chunks, context=context, diagnostics=diagnostics)

            # Phase 4: Save assistant answer to session with trace/source metadata.
            if request.session_id:
                session_manager.add_message(
                    request.session_id,
                    current_user.id,
                    "assistant",
                    full_answer,
                    metadata=_assistant_session_metadata(
                        sources=sources,
                        trace_id=trace_id,
                        context_sufficient=True,
                        confidence=confidence,
                        intent=intent["intent"],
                        intent_confidence=intent["confidence"],
                    ),
                )
            yield make_stream_done_event(sources=sources, context_sufficient=True, confidence=confidence, intent=intent['intent'], intent_confidence=intent['confidence'], trace_id=trace_id)

        except Exception as exc:
            error_message = "Streaming query failed. Please retry."
            try:
                engine = get_rag_engine()
                trace_payload = {
                    "tenant_id": current_user.id,
                    "query_original": request.question,
                    "model_name": llm_model_name,
                    "answer": error_message,
                    "error_message": str(exc),
                    "diagnostics": {"stream_error": True},
                    "latency_ms": 0,
                }
                trace_id = safe_log_query_trace(engine, trace_payload)
            except Exception:
                trace_id = None
            yield make_stream_error_event("stream_error", error_message, retryable=True, trace_id=trace_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# <---------------------- Session endpoints (Phase 4) ---------------------->

@app.post("/sessions/clear")
async def clear_session(request: QueryRequest, current_user: User = Depends(get_current_user)):
    """
    清除指定会话的历史消息。
    """
    if not request.session_id:
        raise api_error(400, "session_id_required", "session_id is required")
    session_id = _validate_session_id(request.session_id)
    cleared = session_manager.clear_session(session_id, current_user.id)
    return {"message": "Session cleared", "session_id": session_id, "cleared": cleared}

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str, current_user: User = Depends(get_current_user)):
    """
    获取指定会话的历史消息（用于前端恢复对话）。
    """
    session_id = _validate_session_id(session_id)
    messages = session_manager.get_recent_messages(session_id, current_user.id)
    count = session_manager.get_session_count(session_id, current_user.id)
    return {"session_id": session_id, "messages": messages, "total_messages": count}


# <---------------------- DELETE requests ---------------------->
@app.delete("/documents/{doc_name}")
async def delete_document(doc_name: str, current_user: User = Depends(get_current_user)):
    """
    删除特定文档及其对应的向量集合。
    同时会清理该文档在 BM25 索引中的数据。

    Args:
        doc_name (str): 要删除的文档名称。
        current_user (User): 当前登录用户，通过 JWT 依赖注入获取。

    Returns:
        dict: 包含成功删除消息的字典。

    Raises:
        HTTPException: 如果指定文档不存在，抛出 404 异常。
    """
    success = delete_document_collection(doc_name, current_user.id)

    if not success:
        raise api_error(404, "document_not_found", f"Document '{doc_name}' not found")

    # Clear from SQLite FTS5 index
    # 从 SQLite FTS5 索引中清除
    engine = get_rag_engine()
    engine.bm25_retriever.delete_doc(doc_name, current_user.id)
    print(f"✓ Deleted {doc_name} from BM25 index (user: {current_user.id})")

    # Sync document registry (remove orphaned entry)
    document_registry.delete(current_user.id, doc_name)
    print(f"✓ Deleted {doc_name} from document registry (user: {current_user.id})")

    return {"message": f"Document '{doc_name}' deleted successfully"}

@app.delete("/documents")
async def clear_all_documents(current_user: User = Depends(get_current_user)):
    """
    清除当前登录用户的所有文档（不再清除全局数据）。
    分别从向量库和 BM25 索引中删除该用户的全部数据。

    Returns:
        dict: 包含成功清除消息的字典。

    Raises:
        HTTPException: 如果清除过程中发生错误，抛出 500 异常。
    """
    errors = []
    # Delete current user's vectors from ChromaDB
    # Note: delete_document_collection returns False when no data exists (by design).
    # That is NOT an error for clear-all — clearing nothing is idempotent.
    try:
        delete_document_collection(None, current_user.id)
    except ValueError:
        pass  # user_id is always provided here via Depends(get_current_user)

    # Delete current user's BM25 index entries
    try:
        engine = get_rag_engine()
        engine.bm25_retriever.delete_all_for_user(current_user.id)
    except Exception as e:
        errors.append(f"BM25 deletion failed: {e}")

    # Sync document registry (clear orphaned entries)
    try:
        document_registry.delete_all_for_tenant(current_user.id)
    except Exception as e:
        errors.append(f"Registry sync failed: {e}")

    if errors:
        raise api_error(500, "partial_failure", f"Partial failure: {'; '.join(errors)}")

    return {"message": f"All documents cleared for user {current_user.id}"}
