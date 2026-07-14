from pydantic import BaseModel, Field, field_validator
from datetime import datetime

QUERY_QUESTION_MAX_CHARS = 4000
QUERY_DOCUMENT_NAMES_MAX_ITEMS = 20
QUERY_DOCUMENT_NAME_MAX_CHARS = 180

class QueryRequest(BaseModel):
    """
    查询请求模型，用于封装用户发起的查询请求参数。
    """
    question: str = Field(..., min_length=2, max_length=QUERY_QUESTION_MAX_CHARS)
    # 查询的问题内容，最小长度为2
    document_names: list[str] | None = Field(None, max_length=QUERY_DOCUMENT_NAMES_MAX_ITEMS, description="List of docs to search. If null, searches all docs.")
    # 指定要搜索的文档名称列表。如果为None，则搜索所有文档
    n_results: int = Field(default=5, ge=1, le=20)
    # 返回的结果数量，默认为5，取值范围在1到20之间
    session_id: str | None = Field(None, min_length=1, max_length=128, description="Session ID for conversation memory. If null, no history is used.")

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("question must contain at least 2 non-whitespace characters")
        return value

    @field_validator("document_names")
    @classmethod
    def normalize_document_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = []
        seen = set()
        for raw_name in value:
            if not isinstance(raw_name, str):
                raise ValueError("document_names must contain strings")
            name = raw_name.strip()
            if (
                not name
                or len(name) > QUERY_DOCUMENT_NAME_MAX_CHARS
                or any(ord(ch) < 32 for ch in name)
                or "/" in name
                or "\\" in name
            ):
                raise ValueError("invalid document name")
            if name not in seen:
                seen.add(name)
                normalized.append(name)
        return normalized or None

    # 会话ID，用于多轮对话记忆。如果为None则不使用历史上下文


class SourceInfo(BaseModel):
    """A retrieved source chunk used to build a RAG answer."""
    filename: str | None = None
    page: int | str | None = None
    type: str | None = None
    score: float | None = None
    chunk_id: str | None = None


class QueryResponse(BaseModel):
    """
    查询响应模型，用于封装查询操作返回的结果数据。
    """
    answer: str
    # 查询得到的答案
    sources: list[SourceInfo]
    # 答案的来源信息列表
    question: str
    # 原始查询的问题
    searched_docs: list[str]
    # 实际参与搜索的文档名称列表
    session_id: str | None = None
    # 会话ID（Phase 4）
    rewritten_question: str | None = None
    # 改写后的独立问题（Phase 4，仅当使用了历史上下文时有值）
    confidence: float | None = None
    # 检索置信度 0.0-1.0（Phase 3）
    context_sufficient: bool | None = None
    # 检索上下文是否充分（Phase 3）
    intent: str | None = None
    # 查询意图（Phase 10）
    intent_confidence: float | None = None
    # 意图识别置信度（Phase 10）
    trace_id: str | None = None
    # 查询追踪ID（Phase 12）

class FeedbackRequest(BaseModel):
    """User answer feedback tied to a persisted query trace."""
    trace_id: str = Field(..., min_length=1, max_length=128)
    rating: str = Field(..., pattern=r'^(up|down)$')
    comment: str | None = Field(None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Stored answer feedback response."""
    feedback_id: str
    trace_id: str
    rating: str
    comment: str | None = None
    created_at: float


class UploadResponse(BaseModel):
    """
    文件上传响应模型，用于封装文件上传成功后返回的信息。
    """
    filename: str
    # 上传的文件名称
    collection_name: str
    # 文档存入的集合/类别名称
    pages: int
    # 文档解析出的页数
    total_docs: int
    # 该集合下的文档总数
    message: str
    # 上传操作的提示信息

class DocumentInfo(BaseModel):
    """
    文档信息模型，用于描述单个文档的基本属性。
    """
    name: str
    # 文档名称
    count: int
    # 文档相关的数据块或条目数量
    pages: int | None
    # 文档的页数，可能为空

class DocumentsListResponse(BaseModel):
    """
    文档列表响应模型，用于封装获取文档列表的返回结果。
    """
    documents: list[DocumentInfo]
    # 文档信息对象的列表
    total_documents: int
    # 文档总数

class UserRegister(BaseModel):
    """
    用户注册请求模型，用于封装新用户注册时提交的数据。
    """
    email: str = Field(..., min_length=3, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    # 用户邮箱，最小长度为3，且必须符合邮箱的正则表达式格式
    password: str = Field(..., min_length=6)
    # 用户密码，最小长度为6

class UserLogin(BaseModel):
    """
    用户登录请求模型，用于封装用户登录时提交的凭据。
    """
    email: str
    # 用户邮箱
    password: str
    # 用户密码

class Token(BaseModel):
    """
    令牌模型，用于封装认证成功后下发的访问令牌信息。
    """
    access_token: str
    # 访问令牌字符串
    token_type: str = "bearer"
    # 令牌类型，默认为"bearer"
    email: str
    # 关联的用户邮箱

class UserResponse(BaseModel):
    """
    用户信息响应模型，用于封装返回给前端的用户公开信息。
    """
    email: str
    # 用户邮箱
    created_at: datetime
    # 用户账号的创建时间
