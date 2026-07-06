# FinQuery 项目深度分析文档

> 本文档面向源码学习与面试准备，按文件逐个详解，明确文件间调用关系与前后端交互流程。

---

## 1. 项目概述

**FinQuery** 是一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 架构的**金融文档智能问答系统**。

**核心能力：**
- 用户注册/登录，JWT 认证隔离
- 上传 PDF 金融文档（年报、财报、银行流水等）
- 对单文档或多文档（最多 2 个）进行自然语言提问
- 系统自动执行**混合检索**（向量 + BM25）→ **RRF 融合** → **LLM 生成**，返回带来源引用的答案
- 支持 SSE 流式响应，逐 token 实时渲染

**面试亮点关键词：** RAG、混合检索、Hybrid Search、RRF 倒数排名融合、向量数据库 ChromaDB、嵌入模型 Sentence-Transformers、BM25 稀疏检索、LLM 流式生成 SSE、JWT 认证、多集合隔离、表格感知摄取

---

## 2. 系统架构总览

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────┐
│                  React 前端 SPA                      │
│  (Vite + React 19 + react-router-dom + Axios)       │
│  页面: Login / Register / Dashboard                  │
│  组件: Sidebar + ChatArea + InputBar + Message       │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │  HTTP REST + SSE 流式    │
          │  (前端 :5173 ↔ 后端 :8000)│
          │  开发: Vite Proxy        │
          │  生产: Vercel → Railway  │
          └────────────┬────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│              FastAPI 后端 (Python 3.13)              │
│                                                       │
│  ┌─────────┐  ┌──────────┐  ┌─────────────────────┐ │
│  │  Auth   │  │  Ingest  │  │     RAG Engine      │ │
│  │ 认证模块 │  │ 摄取模块  │  │     RAG 引擎       │ │
│  └────┬────┘  └────┬─────┘  └──────┬──────────────┘ │
│       │            │               │                  │
│  ┌────┴────┐  ┌────┴─────┐  ┌─────┴──────────────┐ │
│  │PostgreSQL│  │PDF处理管线│  │  检索 + 生成模块    │ │
│  │ (用户表) │  │          │  │                     │ │
│  └─────────┘  │ ┌──────┐ │  │ ┌────────┐         │ │
│               │ │PyMuPDF│ │  │ │ChromaDB│ 向量检索 │ │
│               │ └──────┘ │  │ └────────┘         │ │
│               │ ┌──────┐ │  │ ┌────────┐         │ │
│               │ │Camelot│ │  │ │  BM25  │ 稀疏检索 │ │
│               │ └──────┘ │  │ └────────┘         │ │
│               │ ┌──────┐ │  │ ┌────────┐         │ │
│               │ │LLM 8B │ │  │ │  RRF   │ 融合    │ │
│               │ │表格增强│ │  │ └────────┘         │ │
│               │ └──────┘ │  │ ┌────────┐         │ │
│               │ ┌──────┐ │  │ │LLM 70B │ 答案生成 │ │
│               │ │分块器  │ │  │ └────────┘         │ │
│               │ └──────┘ │  └─────────────────────┘ │
│               └──────────┘                           │
│  ┌─────────────────────────────────────────────────┐ │
│  │           ChromaDB (向量持久化存储)               │ │
│  │  嵌入模型: all-MiniLM-L6-v2 (384维)             │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 2.2 核心架构设计决策（面试重点）

| 设计决策 | 实现方式 | 为什么这样做 |
|----------|----------|-------------|
| **多集合隔离** | 每个文档独立 ChromaDB collection，用户 MD5 哈希前缀 | 数据隔离安全、单文档精确检索、跨文档灵活组合 |
| **混合检索** | ChromaDB 向量检索 + BM25Okapi 稀疏检索 | 向量检索捕获语义相似性，BM25 捕获精确关键词/数字匹配，互补提升召回率 |
| **RRF 融合** | 倒数排名融合 `1/(k+rank+1)`，k=60 | 无需归一化分数即可融合异构排序，对异常值鲁棒，k=60 是经验最优值 |
| **表格感知摄取** | Camelot 提取表格 + LLM(8B) 增强语义摘要 | 金融文档表格含关键数值，纯文本提取丢失结构，增强后提升表格检索命中率 |
| **SSE 流式响应** | FastAPI StreamingResponse + SSE 协议 | 降低首 token 延迟感知，提升用户体验，避免长答案时长时间等待 |
| **懒加载 RAGEngine** | 全局单例，首次查询时初始化 | 避免启动时加载嵌入模型耗时，按需初始化节省资源 |
| **BM25 缓存** | `bm25_cache` 字典按文档缓存索引 | 避免每次查询重建 BM25 索引，相同文档多次查询时直接命中缓存 |
| **JWT + bcrypt** | passlib/bcrypt 哈希 + python-jose JWT | bcrypt 抗彩虹表，JWT 无状态认证减少数据库查询 |

---

## 3. 技术栈详解

### 3.1 后端技术栈（Python 3.13+）

| 类别 | 技术 | 版本 | 作用 | 面试知识点 |
|------|------|------|------|-----------|
| **Web 框架** | FastAPI | >=0.124.2 | 异步 API 服务，自动 OpenAPI 文档 | ASGI、依赖注入、Pydantic 验证 |
| **ASGI 服务器** | Uvicorn | >=0.38.0 | 运行 FastAPI 应用 | ASGI vs WSGI 区别 |
| **ORM** | SQLAlchemy | >=2.0.25 | 用户表 CRUD | Unit of Work、会话管理 |
| **数据库驱动** | psycopg2-binary | >=2.9.9 | 连接 PostgreSQL | 连接池、二进制包 vs 源码编译 |
| **JWT** | python-jose[cryptography] | >=3.5.0 | JWT 编解码 | HS256 对称签名、exp 过期、sub 主题 |
| **密码哈希** | passlib[bcrypt] + bcrypt | >=1.7.4 / ==4.0.1 | 密码哈希与验证 | bcrypt salt+cost factor、抗彩虹表 |
| **LLM SDK** | Together AI | >=1.5.32 | 调用 LLM API | OpenAI 兼容接口、流式 vs 非流式 |
| **生成模型** | Meta-Llama-3.1-70B-Instruct-Turbo | — | 答案生成 | 大模型推理、temperature=0 确定性输出 |
| **表格增强模型** | Meta-Llama-3.1-8B-Instruct-Turbo | — | 表格语义增强 | 小模型做预处理、cost-performance 平衡 |
| **嵌入模型** | Sentence-Transformers (all-MiniLM-L6-v2) | >=5.2.0 | 文本→384维向量 | 对比学习、sentence embedding、余弦相似度 |
| **向量数据库** | ChromaDB | >=1.3.5 | 向量存储与近邻检索 | PersistentClient、HNSW 索引、collection 隔离 |
| **稀疏检索** | rank-bm25 (BM25Okapi) | >=0.2.2 | 关键词匹配检索 | BM25 公式、IDF、文档长度归一化 |
| **PDF 文本** | PyMuPDF | >=1.24.0 | PDF 文本抽取 | 流式解析、页面级文本提取 |
| **PDF 表格** | camelot-py[cv] | >=1.0.9 | PDF 表格提取 | stream 模式(无边框) vs lattice 模式(有边框) |
| **文本分割** | langchain-text-splitters | >=1.0.0 | 文本分块 | RecursiveCharacterTextSplitter、chunk_size/overlap 权衡 |
| **数据处理** | pandas | >=2.3.3 | DataFrame→Markdown | Camelot 表格格式化中间件 |
| **环境变量** | python-dotenv | >=1.2.1 | .env 加载 | 12-factor app 配置管理 |
| **文件上传** | python-multipart | >=0.0.21 | multipart/form-data | FastAPI UploadFile 依赖 |
| **RAG 评估** | ragas | >=0.4.2 | RAG 管线质量评估 | Faithfulness、Relevancy 等指标 |
| **包管理** | uv | — | 依赖安装 + 锁文件 | 比 pip 快 10-100x、确定性构建 |

### 3.2 前端技术栈

| 类别 | 技术 | 版本 | 作用 | 面试知识点 |
|------|------|------|------|-----------|
| **UI 库** | React | ^19.2.0 | 声明式 UI 渲染 | Hooks、函数组件、状态管理 |
| **DOM** | react-dom | ^19.2.0 | React DOM 挂载 | createRoot、StrictMode |
| **路由** | react-router-dom | ^7.11.0 | SPA 客户端路由 | BrowserRouter、ProtectedRoute 守卫模式 |
| **HTTP** | Axios | ^1.13.2 | API 请求 + 拦截器 | 请求拦截(JWT注入)、响应拦截(401处理) |
| **通知** | react-hot-toast | ^2.6.0 | 操作反馈 | toast.loading/success/error 模式 |
| **分析** | @vercel/analytics | ^1.6.1 | Web 分析 | 生产环境性能监控 |
| **构建** | Vite | ^7.2.4 | 开发服务器 + 打包 | HMR、开发代理 proxy、ESBuild 预构建 |
| **代码检查** | ESLint | ^9.39.1 | 代码规范 | Flat config、React Hooks 规则 |

### 3.3 基础设施

| 类别 | 技术 | 作用 | 面试知识点 |
|------|------|------|-----------|
| **后端部署** | Railway | Docker 容器化部署 | Dockerfile 构建、健康检查、重启策略 |
| **前端部署** | Vercel | 静态 SPA 部署 | vercel.json SPA rewrite、自动 HTTPS |
| **容器化** | Docker (python:3.13-slim) | 后端镜像 | 多阶段构建、uv 缓存层、系统依赖安装 |
| **关系数据库** | PostgreSQL (Railway) | 用户认证数据 | SQLAlchemy 引擎、连接字符串、迁移 |
| **向量数据库** | ChromaDB (PersistentClient) | 文档向量存储 | 本地持久化路径、Volume 挂载 |
| **开发数据库** | SQLite | 本地开发回退 | DATABASE_URL 未设置时默认 |

---

## 4. 项目结构与文件组织

```
E:/financial/finquery-main/
│
├── .gitignore                          # Python 通用忽略 + chroma_db/ + rag.ipynb
├── .python-version                     # Python 3.13 版本声明
├── railway.toml                        # ★ Railway 后端部署配置
├── assets/                             # 静态资源（架构图、测试 PDF、缩略图）
│
├── backend/                            # ★★★ 后端 Python/FastAPI
│   ├── Dockerfile                      # Docker 构建配置
│   ├── pyproject.toml                  # 项目依赖声明（16个）
│   ├── uv.lock                         # uv 锁文件（确定性构建）
│   ├── .dockerignore                   # Docker 构建忽略
│   ├── .gitignore                      # Python 专用忽略
│   └── src/                            # 主应用包
│       ├── __init__.py                 # 包标记（空）
│       ├── main.py                     # ★★★ 应用入口 + 全部 API 端点
│       ├── database.py                 # ★ 数据库引擎/会话配置
│       ├── models/                     # 数据模型层
│       │   ├── __init__.py             # 包标记（空）
│       │   ├── schemas.py              # ★ Pydantic 请求/响应模型（9个）
│       │   └── user.py                 # ★ SQLAlchemy 用户 ORM 模型
│       └── services/                   # 业务逻辑层
│           ├── __init__.py             # 包标记（空）
│           ├── auth.py                 # ★★ JWT 认证服务
│           ├── ingest.py               # ★★★ PDF 摄取管线
│           ├── process_tables.py       # ★★ 表格提取与 LLM 增强
│           ├── rag_engine.py           # ★★★★ 核心 RAG 引擎
│           ├── retrieval.py            # ★★ BM25 检索 + RRF 融合
│           └── vector_store.py         # ★★★ ChromaDB 向量库管理
│
└── frontend/                           # ★★★ 前端 React/Vite
    ├── index.html                      # HTML 入口
    ├── package.json                    # NPM 依赖
    ├── package-lock.json               # 锁文件
    ├── vite.config.js                  # ★ Vite 开发/构建/代理配置
    ├── eslint.config.js                # ESLint 规则
    ├── vercel.json                     # ★ Vercel SPA 部署配置
    ├── .env.production                 # ★ 生产环境 API 地址
    ├── .gitignore                      # 前端忽略规则
    ├── public/
    │   └── finquery-favicon.png        # 站点图标
    └── src/
        ├── main.jsx                    # ★ React 入口
        ├── App.jsx                     # ★★ 根组件 + 路由 + 认证守卫
        ├── App.css                     # ★ 仪表盘完整布局样式
        ├── index.css                   # 全局 CSS 重置
        ├── api.js                      # ★★★ Axios API 客户端 + SSE
        ├── context/
        │   └── AuthContext.jsx         # ★★ 认证上下文（JWT 状态管理）
        ├── components/
        │   ├── ChatArea.jsx            # ★ 聊天消息区
        │   ├── InputBar.jsx            # ★ 输入栏
        │   ├── Message.jsx             # 消息气泡
        │   └── Sidebar.jsx             # ★★ 侧边栏（文档列表+上传）
        └── pages/
            ├── Dashboard.jsx           # ★★★ 主仪表盘（状态管理中心）
            ├── Login.jsx               # ★ 登录页
            ├── Register.jsx            # ★ 注册页
            └── Auth.css                # 认证页共享样式
```

> ★ 数量表示文件重要程度，★ 越多越核心

---

## 5. 文件逐个详解

---

### 5.1 后端文件详解

---

#### 5.1.1 `backend/src/main.py` — 应用入口与 API 端点定义

**文件位置：** `backend/src/main.py`  
**重要程度：** ★★★  
**被调用关系：** 这是整个后端的入口点，所有其他模块都被它导入和调用。

**导入依赖关系：**
```
main.py 导入:
  ├── services/auth.py      → create_access_token, get_current_user, get_current_user_optional, get_password_hash, verify_password
  ├── services/ingest.py    → process_pdf
  ├── services/vector_store.py → add_documents, list_all_documents, delete_document_collection, get_collection_stats
  ├── services/rag_engine.py → RAGEngine
  ├── models/schemas.py     → 全部 Pydantic 模型 (from .models.schemas import *)
  ├── models/user.py        → User ORM 模型
  └── database.py           → get_db, engine, Base
```

**启动初始化流程（按代码执行顺序）：**

1. `Base.metadata.create_all(bind=engine)` — 自动创建数据库表（开发模式，非 Alembic 迁移）
2. `os.environ["TOKENIZERS_PARALLELISM"] = "false"` — 禁用 HuggingFace tokenizer 并行，避免 fork 警告
3. `app = FastAPI(title="FinQuery API", version="3.0.0")` — 创建 FastAPI 实例
4. CORS 中间件配置 — `ALLOWED_ORIGINS` 从环境变量读取，默认 `localhost:5173`
5. `load_dotenv()` — 加载 .env 环境变量
6. `together_client = Together()` — 初始化 Together AI 客户端（API Key 从 `TOGETHER_API_KEY` 环境变量读取）
7. `rag_engine = None` — RAGEngine 全局变量，懒加载

**`get_rag_engine()` 函数：**
- 懒加载 RAGEngine 单例
- 首次调用时创建 `RAGEngine(together_client, use_hybrid=True)`
- 后续调用直接返回已有实例

**全部 API 端点详解：**

| 方法 | 路径 | 认证 | 处理逻辑 | 调用的服务 |
|------|------|------|----------|-----------|
| GET | `/` | 否 | 返回健康状态 JSON | 无 |
| GET | `/me` | 是 | 从 JWT 解析 email → 查询 User 表 → 返回用户信息 | `auth.get_current_user`, `User` ORM |
| GET | `/documents` | 是 | 列出当前用户所有文档 | `auth.get_current_user`, `vector_store.list_all_documents` |
| GET | `/documents/{doc_name}` | 是 | 获取指定文档统计信息 | `auth.get_current_user`, `vector_store.get_collection_stats` |
| POST | `/register` | 否 | 检查邮箱重复 → 哈希密码 → 创建用户 → 生成 JWT | `auth.get_password_hash`, `auth.create_access_token` |
| POST | `/login` | 否 | 查找用户 → 验证密码 → 生成 JWT | `auth.verify_password`, `auth.create_access_token` |
| POST | `/upload` | 是 | 验证 PDF → 临时保存 → 调用摄取管线 → 存入向量库 → 清除 BM25 缓存 | `auth.get_current_user`, `ingest.process_pdf`, `vector_store.add_documents`, `rag_engine.bm25_cache` |
| POST | `/query` | 是 | 完整 RAG 管线（非流式） | `auth.get_current_user`, `rag_engine.query` |
| POST | `/query/stream` | 是 | 完整 RAG 管线（SSE 流式） | `auth.get_current_user`, `rag_engine` 各步骤分解调用 |
| DELETE | `/documents/{doc_name}` | 是 | 删除文档集合 + 清除 BM25 缓存 | `auth.get_current_user`, `vector_store.delete_document_collection`, `rag_engine.bm25_cache` |
| DELETE | `/documents` | 否 | 删除整个 chroma_db 目录 + 重置 RAGEngine | `shutil.rmtree`, `rag_engine = None` |

**`/upload` 端点详细流程：**
```
1. 验证文件类型 (.pdf)
2. 保存上传文件到临时路径
3. 调用 process_pdf(together_client, temp_path) → 返回 (chunks, pages)
4. 调用 add_documents(chunks, filename, user_id, pages) → 返回 {collection_name, total_docs}
5. 清除该文档的 BM25 缓存（key = "{user_id}_{filename}"）
6. 删除临时文件
7. 返回 UploadResponse
```

**`/query/stream` 端点详细流程：**
```
1. 获取 RAGEngine 实例
2. 检查是否为对话式查询（问候/身份/能力/感谢/告别）→ 直接返回固定回复
3. 确定要搜索的文档列表（用户指定 or 全部）
4. 检索阶段：
   - 单文档：retrieve_single_document()
   - 多文档：retrieve_multiple_documents()
5. 构建上下文：build_context(chunks) → (context_string, sources)
6. 流式生成：generate_answer_stream() → 逐 token yield
7. SSE 格式输出：
   - token:  data: {"type": "token", "content": "..."}\n\n
   - done:   data: {"type": "done", "sources": [...]}\n\n
```

---

#### 5.1.2 `backend/src/database.py` — 数据库配置

**文件位置：** `backend/src/database.py`  
**重要程度：** ★  
**调用者：** `main.py`（导入 get_db, engine, Base）、`models/user.py`（导入 Base）、`services/auth.py`（导入 get_db）

**代码逻辑：**
```python
DATABASE_URL = os.getenv("DATABASE_URL")  # 从环境变量读取
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./temp.db"  # 本地开发回退 SQLite

engine = create_engine(DATABASE_URL)       # SQLAlchemy 引擎
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # 会话工厂
Base = declarative_base()                  # ORM 声明基类

def get_db():                              # FastAPI 依赖注入生成器
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**关键设计：**
- `get_db()` 使用 Python 生成器模式实现 FastAPI 依赖注入，确保每个请求获取独立会话并在请求结束后关闭
- 生产环境使用 PostgreSQL（`DATABASE_URL` 由 Railway 注入），开发环境使用 SQLite

---

#### 5.1.3 `backend/src/models/schemas.py` — Pydantic 数据验证模型

**文件位置：** `backend/src/models/schemas.py`  
**重要程度：** ★  
**调用者：** `main.py`（`from .models.schemas import *`，用于端点参数和响应类型注解）

**9 个模型详解：**

| 模型 | 字段 | 用途 | 对应端点 |
|------|------|------|----------|
| `QueryRequest` | question(str, min=2), document_names(list\[str]\|None), n_results(int, 1-20, default=5) | 查询请求体 | POST /query, /query/stream |
| `QueryResponse` | answer(str), sources(list\[dict]), question(str), searched_docs(list\[str]) | 查询响应体 | POST /query |
| `UploadResponse` | filename(str), collection_name(str), pages(int), total_docs(int), message(str) | 上传响应体 | POST /upload |
| `DocumentInfo` | name(str), count(int), pages(int\|None) | 单个文档信息 | GET /documents 嵌套 |
| `DocumentsListResponse` | documents(list\[DocumentInfo]), total_documents(int) | 文档列表响应 | GET /documents |
| `UserRegister` | email(str, 正则验证), password(str, min=6) | 注册请求体 | POST /register |
| `UserLogin` | email(str), password(str) | 登录请求体 | POST /login |
| `Token` | access_token(str), token_type(str="bearer"), email(str) | JWT 响应体 | POST /register, /login |
| `UserResponse` | email(str), created_at(datetime) | 用户信息响应 | GET /me |

**面试知识点：** Pydantic 的 `Field` 验证器（min_length、pattern 正则）、Python 3.10+ 联合类型语法 `list[str] | None`

---

#### 5.1.4 `backend/src/models/user.py` — 用户 ORM 模型

**文件位置：** `backend/src/models/user.py`  
**重要程度：** ★  
**调用者：** `main.py`（注册/登录时查询和创建用户）、`services/auth.py`（验证 JWT 时查询用户是否存在）

```python
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)      # 唯一索引，用于快速查找
    hashed_password = Column(String)                      # bcrypt 哈希，不存明文
    created_at = Column(DateTime, default=datetime.utcnow)
```

**与数据库的关系：** 继承 `database.py` 的 `Base`，由 `main.py` 启动时 `Base.metadata.create_all()` 自动建表

---

#### 5.1.5 `backend/src/services/auth.py` — JWT 认证服务

**文件位置：** `backend/src/services/auth.py`  
**重要程度：** ★★  
**调用者：** `main.py`（所有需要认证的端点通过 `Depends(get_current_user)` 调用）

**导入依赖：**
```
auth.py 导入:
  ├── jose (JWT 库)          → jwt.encode, jwt.decode, JWTError
  ├── passlib.context        → CryptContext(schemes=["bcrypt"])
  ├── fastapi.security       → HTTPBearer, HTTPAuthorizationCredentials
  ├── database.py            → get_db (依赖注入)
  └── models/user.py         → User (验证用户存在)
```

**函数详解：**

| 函数 | 输入 | 输出 | 逻辑 |
|------|------|------|------|
| `verify_password(plain, hashed)` | 明文密码, 哈希 | bool | passlib bcrypt 验证 |
| `get_password_hash(password)` | 明文密码 | 哈希字符串 | passlib bcrypt 哈希（自动加盐） |
| `create_access_token(data, expires_delta)` | {"sub": email}, 过期时间 | JWT 字符串 | jose.jwt.encode，HS256 签名，默认 30 分钟过期 |
| `get_current_user(credentials, db)` | Bearer token + DB 会话 | email 字符串 | 解码 JWT → 取 sub → 查 User 表验证存在 → 返回 email |
| `get_current_user_optional(credentials)` | 可选 Bearer token | email 或 None | 无 token 返回 None，有则解码返回 |

**JWT 认证完整流程：**
```
注册/登录:
  email + password → get_password_hash(password) → 存入 User 表
                  → create_access_token({"sub": email}) → 返回 JWT

请求认证:
  Authorization: Bearer <token>
  → get_current_user() 解码 JWT
  → 验证 sub 对应用户在数据库中存在
  → 返回 user_id (email) 供端点使用
```

**面试知识点：** 
- HS256 对称签名（同一密钥签发和验证）vs RS256 非对称签名
- `HTTPBearer()` 自动从请求头提取 Bearer token
- `Depends()` 实现声明式依赖注入，FastAPI 自动执行

---

#### 5.1.6 `backend/src/services/ingest.py` — PDF 摄取管线

**文件位置：** `backend/src/services/ingest.py`  
**重要程度：** ★★★  
**调用者：** `main.py` 的 `/upload` 端点调用 `process_pdf(together_client, temp_path)`

**导入依赖：**
```
ingest.py 导入:
  ├── pymupdf                              → PDF 文本提取
  ├── langchain_text_splitters             → RecursiveCharacterTextSplitter
  └── services/process_tables.py           → enhance_table_with_context, extract_tables_with_camelot
```

**`process_pdf(llm_client, pdf_path)` 完整流程：**

```
输入: Together AI 客户端 + PDF 文件路径
输出: (chunks: list[dict], pages: int)

Step 1: 初始化分块器
  RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

Step 2: 打开 PDF (pymupdf.open)

Step 3: Camelot 提取表格
  → extract_tables_with_camelot(pdf_path)
  → 返回 tables_by_page: {page_num: [table1_md, table2_md, ...]}

Step 4: 逐页处理
  for page_num in range(pages):
    page_text = doc[page_num].get_text("text")    # PyMuPDF 提取文本

    # Step 4a: 处理该页表格（如果存在）
    if actual_page_num in tables_by_page:
      for table_md in tables_by_page[actual_page_num]:
        enhanced_table = enhance_table_with_context(     # ★ 调用 LLM 8B 增强表格
          llm_client, table_md, page_text, actual_page_num
        )
        chunks.append({
          "content": enhanced_table,
          "metadata": {
            "type": "table",
            "page": actual_page_num,
            "source": pdf_path,
            "doc_id": "{doc_name}::page_{n}::table_{idx}",
            "table_num": table_idx + 1
          }
        })

    # Step 4b: 分块文本
    page_chunks = TEXT_SPLITTER.split_text(page_text)
    for chunk_text in page_chunks:
      chunks.append({
        "content": chunk_text.strip(),
        "metadata": {
          "type": "text",
          "page": actual_page_num,
          "source": pdf_path,
          "doc_id": "{doc_name}::page_{n}::chunk_{idx}"
        }
      })

Step 5: 关闭 PDF，返回 (chunks, pages)
```

**chunk 元数据设计（面试重点）：**
- `type`: "text" 或 "table"，检索时可按类型过滤
- `page`: 来源页码，用于生成引用
- `doc_id`: 全局唯一标识（格式：`文件名::page_X::chunk_Y` 或 `文件名::page_X::table_Y`），作为 ChromaDB 文档 ID
- `table_num`: 表格序号（仅表格块有）

---

#### 5.1.7 `backend/src/services/process_tables.py` — 表格提取与 LLM 增强

**文件位置：** `backend/src/services/process_tables.py`  
**重要程度：** ★★  
**调用者：** `ingest.py` 调用 `extract_tables_with_camelot()` 和 `enhance_table_with_context()`

**函数 1: `extract_tables_with_camelot(pdf_path, pages="all")`**

```
输入: PDF 路径
输出: {page_num: [table1_markdown, table2_markdown, ...]}

逻辑:
  1. 先尝试 stream 模式 (camelot.read_pdf(flavor="stream", edge_tol=50, row_tol=10))
     - stream 模式：适用于无边框表格（银行流水等）
     - edge_tol=50: 边缘容差，row_tol=10: 行容差
  2. stream 失败 → 回退 lattice 模式 (flavor="lattice")
     - lattice 模式：适用于有边框表格（财务报表等）
  3. 对每个 table → format_table() → 转为 Markdown 字符串
```

**`format_table(table)` 内部逻辑：**
```python
# 清洗 DataFrame：去除换行符和制表符
formatted_table = table.df.apply(lambda x: x.str.replace('\n','').str.replace('\t', ' '))
# 第一行作为列名，删除原第一行
final_table = formatted_table.rename(columns=formatted_table.iloc[0]).drop(formatted_table.index[0])
# 转为 Markdown
return final_table.to_markdown(index=False)
```

**函数 2: `enhance_table_with_context(llm_client, table_md, page_text, page_num)`**

```
输入: LLM 客户端 + 表格 Markdown + 同页文本 + 页码
输出: 增强后的字符串

逻辑:
  1. 构建 system_prompt:
     - 角色：数据预处理助手
     - 规则：生成检索友好摘要(2-3句)、清洗结构、保持数值精度、确定性输出
  2. 构建 user_prompt:
     - 包含页码、表格 Markdown、同页上下文文本
     - 要求输出格式：TABLE SUMMARY + CLEANED TABLE
  3. 调用 Together AI (meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo):
     - temperature=0.3 (较低，偏向确定性)
     - max_tokens=1000
  4. 失败时返回原始 table_md（优雅降级）
```

**为什么用 8B 而不是 70B 做表格增强？**（面试重点）
- 表格增强是预处理步骤，调用频率高（每个表格一次），用小模型节省成本
- 8B 模型对结构化数据处理能力足够，不需要 70B 的复杂推理
- temperature=0.3 既保持结构稳定性又允许适度表达

---

#### 5.1.8 `backend/src/services/rag_engine.py` — 核心 RAG 引擎

**文件位置：** `backend/src/services/rag_engine.py`  
**重要程度：** ★★★★ （整个项目最核心的文件）  
**调用者：** `main.py` 的 `/query` 和 `/query/stream` 端点

**导入依赖：**
```
rag_engine.py 导入:
  ├── services/vector_store.py → query_collection, get_or_create_collection, list_all_documents
  └── services/retrieval.py    → BM25Retriever, rrf
```

**RAGEngine 类结构：**

```python
class RAGEngine:
    def __init__(self, llm_client, use_hybrid=True):
        self.llm_client = llm_client         # Together AI 客户端
        self.use_hybrid = use_hybrid         # 是否启用混合检索
        self.bm25_cache = {}                # BM25 索引缓存 dict[doc_name, BM25Retriever]
```

**核心方法详解：**

**① `_get_bm25_retriever(doc_name, user_id)` — BM25 索引懒加载与缓存**

```
逻辑:
  1. 如果 use_hybrid=False → 返回 None
  2. cache_key = "{user_id}_{doc_name}"
  3. 如果 cache_key 在 bm25_cache 中 → 命中缓存，直接返回
  4. 未命中 → 从 ChromaDB 加载该文档所有 chunks
  5. 创建 BM25Retriever(chunks) → 存入 bm25_cache
  6. 返回 retriever

缓存失效时机:
  - 文档上传时: main.py /upload 端点中删除 bm25_cache[key]
  - 文档删除时: main.py /delete 端点中删除 bm25_cache[key]
  - 清空全部时: main.py /clear 端点中 rag_engine = None（重建整个引擎）
```

**② `retrieve_single_document(doc_name, query, user_id, n_results)` — 单文档混合检索**

```
逻辑:
  1. if not use_hybrid:
       → 直接向量检索: query_collection(doc_name, query, n_results)
  2. else (混合检索):
       a. 稠密检索: dense_results = query_collection(doc_name, query, n_results * 2)
          # 请求 2 倍结果，留出 RRF 融合空间
       b. 稀疏检索: bm25_retriever = _get_bm25_retriever(doc_name, user_id)
                    sparse_results = bm25_retriever.search(query, k=n_results * 2)
       c. 融合: fused = rrf([dense_results, sparse_results])
       d. 截断: return fused[:n_results]
```

**③ `retrieve_multiple_documents(doc_names, query, user_id, n_results)` — 多文档检索**

```
逻辑:
  1. 对每个文档调用 retrieve_single_document() → 合并结果
  2. 按 score/fused_score 降序排序
  3. 返回前 n_results 个
```

**④ `build_context(chunks)` — 构建带引用的上下文字符串**

```
逻辑:
  对每个 chunk:
    1. 从 doc_id 提取文件名 (split("::")[0])
    2. 生成来源引用: "filename, page X" 或 "filename, page X (Table Y)"
    3. 拼接: "[Source: {source_ref}]\n{content}"
  所有 chunk 间用 "---" 分隔
  返回 (context_string, sources_list)
```

**⑤ `generate_answer(context, query)` — 非流式生成**

```
逻辑:
  1. 如果 context 为空 → 返回固定提示
  2. 调用 Together AI:
     - model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
     - temperature=0 (完全确定性)
     - max_tokens=1000
  3. 返回完整答案
```

**⑥ `generate_answer_stream(context, query)` — 流式生成**

```
逻辑:
  与 generate_answer 相同，但 stream=True
  逐 chunk yield: chunk.choices[0].delta.content
```

**⑦ `query(question, doc_names, user_id, n_results)` — 完整 RAG 管线**

```
逻辑:
  1. 检查对话式查询 → _handle_conversational_query()
     如果是问候/身份/能力/感谢/告别 → 直接返回固定回复
  2. 如果 doc_names 为 None → 搜索用户所有文档
  3. 检索:
     - 单文档 → retrieve_single_document()
     - 多文档 → retrieve_multiple_documents()
  4. 构建上下文 → build_context()
  5. 生成答案 → generate_answer()
  6. 返回 {answer, sources, context, searched_docs}
```

**⑧ `_handle_conversational_query(query)` — 对话式查询处理**

```
匹配模式:
  - 问候: hi, hello, hey, good morning/afternoon/evening (≤3 词)
  - 身份: what are you, who are you, what is finquery...
  - 能力: how does this work, how to use, help me...
  - 感谢: thank you, thanks, thx, appreciate (≤5 词)
  - 告别: bye, goodbye, see you, exit, quit (≤3 词)
匹配到 → 返回预设回复
未匹配 → 返回 None（需要走 RAG 管线）
```

**System Prompt 设计要点（面试重点）：**
```
核心原则:
  - 金融数据精确性：不修改/舍入数字，保留货币/日期/单位
  - 来源引用：必须标注 "Source: filename, page X"
  - 表格优先：表格是数值数据的权威来源
  - 避免幻觉：只在上下文中找到信息时才回答
  - 禁止 markdown 表格语法：答案用 prose 叙述，不用 | 语法
```

---

#### 5.1.9 `backend/src/services/retrieval.py` — BM25 检索与 RRF 融合

**文件位置：** `backend/src/services/retrieval.py`  
**重要程度：** ★★  
**调用者：** `rag_engine.py` 导入 `BM25Retriever` 和 `rrf`

**BM25Retriever 类：**

```python
class BM25Retriever:
    def __init__(self, chunks):
        self.documents = [c["content"] for c in chunks]      # 文档内容列表
        self.ids = [c["metadata"]["doc_id"] for c in chunks]  # doc_id 列表
        self.metadatas = [c["metadata"] for c in chunks]      # 元数据列表
        # 分词 + 构建 BM25 索引
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(query.lower().split())  # 计算 BM25 分数
        # 按 score 降序排序，取 top-k
        ranked = sorted(zip(self.ids, self.documents, self.metadatas, scores), 
                        key=lambda x: x[3], reverse=True)[:k]
        return [{"doc_id", "content", "metadata", "score"}, ...]
```

**面试知识点 — BM25 公式：**
```
score(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

其中:
  f(qi, D): 查询词 qi 在文档 D 中的词频
  |D|: 文档长度
  avgdl: 平均文档长度
  k1: 词频饱和参数 (通常 1.2-2.0)
  b: 文档长度归一化参数 (通常 0.75)
```

**`rrf(ranked_lists, k=60)` — 倒数排名融合算法：**

```python
def rrf(ranked_lists, k=60):
    fused_scores = defaultdict(float)  # 融合分数累加器
    doc_map = {}                       # 文档内容映射

    for ranked_list in ranked_lists:       # 遍历每个排序列表
        for rank, item in enumerate(ranked_list):  # 遍历排名
            doc_id = item["doc_id"]
            fused_scores[doc_id] += 1 / (k + rank + 1)  # ★ RRF 核心公式
            if doc_id not in doc_map:
                doc_map[doc_id] = item    # 保留文档内容

    # 按融合分数降序排序
    sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_map[doc_id], "fused_score": score} for doc_id, score in sorted_ids]
```

**面试知识点 — RRF 原理：**
```
RRF 公式: score(d) = Σ 1/(k + rank_i(d))

为什么用 RRF 而不是分数归一化？
  1. 不同检索方法的分数尺度不同（向量余弦 vs BM25 分数），归一化困难
  2. RRF 只依赖排名，不依赖绝对分数，天然消除尺度差异
  3. 对异常值鲁棒（一个检索方法的极端分数不会主导融合结果）
  4. k=60 是原论文推荐的经验值，控制排名的衰减速度
```

---

#### 5.1.10 `backend/src/services/vector_store.py` — ChromaDB 向量库管理

**文件位置：** `backend/src/services/vector_store.py`  
**重要程度：** ★★★  
**调用者：** `main.py`（add_documents, list_all_documents, delete_document_collection, get_collection_stats）、`rag_engine.py`（query_collection, get_or_create_collection, list_all_documents）

**全局初始化：**
```python
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")  # 持久化路径
embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")  # 嵌入函数
```

**函数详解：**

| 函数 | 输入 | 输出 | 逻辑 |
|------|------|------|------|
| `get_chroma_client()` | 无 | PersistentClient | 返回持久化客户端实例 |
| `create_collection_name(doc_name, user_id)` | 文件名, 用户ID | 集合名字符串 | 去扩展名 → MD5(user_id)前8位前缀 → 替换非法字符 → 截断63字符 |
| `get_or_create_collection(doc_name, user_id, pages, creating)` | 同上 | ChromaDB Collection | creating=True 时创建（含metadata），否则获取 |
| `add_documents(chunks, doc_name, user_id, pages)` | 块列表+文档信息 | {collection_name, total_docs} | 创建集合 → collection.add(ids, documents, metadatas) |
| `query_collection(doc_name, query_text, n_results, filters, user_id)` | 文档名+查询文本 | [{doc_id, content, metadata, score}, ...] | collection.query() → 距离转相似度 (1-distance) |
| `query_multiple_collections(doc_names, query_text, n_results, user_id)` | 多文档名+查询 | 同上 | 逐文档 query_collection → 合并排序 → 截断 |
| `list_all_documents(user_id)` | 用户ID | [{name, count, pages}, ...] | list_collections() → 按 user_id 过滤 |
| `delete_document_collection(doc_name, user_id)` | 文档名+用户ID | bool | 验证归属 → delete_collection() |
| `get_collection_stats(doc_name, user_id)` | 文档名+用户ID | {name, count, exists} | 获取集合 → 验证归属 → 返回统计 |

**集合命名规则（面试重点）：**
```
原始: "annual_report_2024.pdf", user_id="alice@example.com"
处理: 
  1. 去扩展名: "annual_report_2024"
  2. MD5("alice@example.com")[:8] = "a1b2c3d4"
  3. 添加前缀: "ua1b2c3d4_annual_report_2024"
  4. 替换非法字符、转小写、截断63字符

为什么用 MD5 哈希？
  - 邮箱包含 @ 等非法字符，直接用会违反 ChromaDB 命名规则
  - 哈希固定8位，保证集合名不超过63字符限制
  - 相同用户总是映射到相同前缀，不同用户映射到不同前缀
```

---

#### 5.1.11 `backend/Dockerfile` — Docker 构建配置

**构建流程：**
```dockerfile
FROM python:3.13-slim          # 精简基础镜像

# 系统依赖
RUN apt-get install -y libgl1 ghostscript
  # libgl1: OpenCV 依赖（Camelot 表格提取需要）
  # ghostscript: PDF 后处理依赖

# 安装 uv 包管理器
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

# 依赖缓存层（只复制 pyproject.toml + uv.lock）
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project  # 安装依赖但不安装项目本身

# 复制应用代码
COPY src/ ./src/
RUN uv sync --frozen --no-dev  # 安装项目（依赖已缓存）

# 创建 ChromaDB 持久化目录
RUN mkdir -p /app/chroma_db

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**面试知识点：** 依赖缓存层分离 — 先复制 pyproject.toml + uv.lock 安装依赖，再复制源码。这样代码变更不会重建依赖层，加速构建。

---

### 5.2 前端文件详解

---

#### 5.2.1 `frontend/src/main.jsx` — React 入口

```jsx
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
    <Analytics />    // Vercel Web 分析组件
  </StrictMode>
);
```

**作用：** 挂载 React 应用到 DOM，启用 StrictMode（开发时双重渲染检测副作用），集成 Vercel 分析。

---

#### 5.2.2 `frontend/src/App.jsx` — 根组件 + 路由 + 认证守卫

**文件位置：** `frontend/src/App.jsx`  
**重要程度：** ★★  
**导入依赖：** `AuthContext`, `Login`, `Register`, `Dashboard`, `react-router-dom`, `react-hot-toast`

**组件结构：**
```jsx
<BrowserRouter>
  <AuthProvider>              {/* 认证上下文包裹整个应用 */}
    <Toaster position="top-right" />  {/* 全局 toast 通知 */}
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/" element={
        <ProtectedRoute>      {/* ★ 认证守卫 */}
          <Dashboard />
        </ProtectedRoute>
      } />
    </Routes>
  </AuthProvider>
</BrowserRouter>
```

**ProtectedRoute 认证守卫逻辑：**
```
1. loading=true → 显示 "Loading..."
2. user=null → <Navigate to="/login" replace />  (重定向到登录页)
3. user 存在 → 渲染 children (Dashboard)
```

**面试知识点：** 认证守卫模式（ProtectedRoute 组件包裹受保护路由）、Context 向下传递认证状态、`Navigate` 组件式重定向 vs `useNavigate` 命令式导航

---

#### 5.2.3 `frontend/src/api.js` — Axios API 客户端 + SSE 流式查询

**文件位置：** `frontend/src/api.js`  
**重要程度：** ★★★  
**调用者：** `AuthContext.jsx`（getCurrentUser）、`Dashboard.jsx`（uploadDocument, listDocuments, queryDocumentsStream, deleteDocument）、`Login.jsx`（login）、`Register.jsx`（register）

**核心设计：**

```javascript
// 1. Axios 实例创建
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
});

// 2. 请求拦截器 — 自动注入 JWT
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// 3. 响应拦截器 — 401 自动登出
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';  // 强制跳转登录页
    }
    return Promise.reject(error);
  }
);
```

**封装的 API 函数：**

| 函数 | HTTP 方法 | 后端端点 | 请求格式 | 返回格式 |
|------|-----------|----------|----------|----------|
| `register(email, password)` | POST | `/register` | {email, password} | {access_token, token_type, email} |
| `login(email, password)` | POST | `/login` | {email, password} | 同上 |
| `getCurrentUser()` | GET | `/me` | — (Bearer) | {email, created_at} |
| `uploadDocument(file)` | POST | `/upload` | FormData (multipart) | {filename, collection_name, pages, total_docs, message} |
| `listDocuments()` | GET | `/documents` | — (Bearer) | {documents, total_documents} |
| `queryDocuments(question, documentNames)` | POST | `/query` | {question, document_names, n_results:5} | {answer, sources, question, searched_docs} |
| `queryDocumentsStream(question, documentNames, onToken, onDone, onError)` | POST | `/query/stream` | 同上 (SSE) | 逐 token 回调 |
| `deleteDocument(docName)` | DELETE | `/documents/{docName}` | — (Bearer) | {message} |

**★ `queryDocumentsStream` — SSE 流式查询核心实现（面试重点）：**

```javascript
export const queryDocumentsStream = async (question, documentNames, onToken, onDone, onError) => {
  // 使用原生 fetch（非 Axios），因为 Axios 不原生支持 SSE
  const response = await fetch(`${API_BASE_URL}/query/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({ question, document_names: documentNames, n_results: 5 }),
  });

  // 使用 ReadableStream API 逐块读取
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();  // 读取二进制块
    if (done) break;

    const text = decoder.decode(value);           // 解码为字符串
    const lines = text.split('\n').filter(line => line.startsWith('data: '));

    for (const line of lines) {
      const data = JSON.parse(line.slice(6));     // 解析 SSE data
      if (data.type === 'token') onToken(data.content);    // token 回调
      else if (data.type === 'done') onDone(data.sources); // 完成回调
    }
  }
};
```

**为什么用 fetch 而不是 Axios？** Axios 不原生支持 ReadableStream/SSE，需要额外库。原生 fetch + ReadableStream 是浏览器标准 API，可直接逐块读取响应体。

---

#### 5.2.4 `frontend/src/context/AuthContext.jsx` — 认证上下文

**文件位置：** `frontend/src/context/AuthContext.jsx`  
**重要程度：** ★★  
**调用者：** `App.jsx`（AuthProvider 包裹）、`Dashboard.jsx`（useAuth 获取 user/logout）、`Login.jsx`（useAuth 获取 loginUser）、`Register.jsx`（useAuth 获取 loginUser）

**提供的 Context 值：**
```javascript
{ user, loading, loginUser, logout, loadUser }
```

| 值 | 类型 | 说明 |
|----|------|------|
| `user` | {email} \| null | 当前登录用户，null 表示未登录 |
| `loading` | boolean | 初始化加载状态（验证 token 时为 true） |
| `loginUser(token, email)` | function | 登录：存 token 到 localStorage，设置 user 状态 |
| `logout()` | function | 登出：清除 localStorage token，设置 user=null |
| `loadUser()` | async function | 调用 /me 验证 token 有效性，失败则 logout |

**初始化流程：**
```
组件挂载 → useEffect
  → localStorage 有 token? → loadUser() → GET /me → 设置 user
  → localStorage 无 token? → loading=false
```

**面试知识点：** React Context 全局状态管理模式、自定义 Hook (`useAuth`) 封装 Context 访问、`useEffect` 初始化副作用

---

#### 5.2.5 `frontend/src/pages/Dashboard.jsx` — 主仪表盘（状态管理中心）

**文件位置：** `frontend/src/pages/Dashboard.jsx`  
**重要程度：** ★★★  
**导入依赖：** `Sidebar`, `ChatArea`, `InputBar`, `api.js`（4个API函数）, `AuthContext`（useAuth）, `App.css`

**状态管理：**
```javascript
const [documents, setDocuments] = useState([]);      // 文档列表
const [selectedDocs, setSelectedDocs] = useState([]); // 选中文档（最多2个）
const [messages, setMessages] = useState([]);         // 聊天消息 [{role, content, sources}]
const [isLoading, setIsLoading] = useState(false);    // 查询加载状态
const [isUploading, setIsUploading] = useState(false); // 上传加载状态
const { user, logout } = useAuth();                   // 认证上下文
```

**事件处理函数详解：**

| 函数 | 触发时机 | 逻辑 | 调用的 API |
|------|----------|------|-----------|
| `fetchDocuments()` | 组件挂载 + 上传/删除后 | `listDocuments()` → 更新 documents | GET /documents |
| `handleUpload(file)` | 拖拽/选择文件 | 验证PDF → `uploadDocument(file)` → `fetchDocuments()` | POST /upload |
| `handleDelete(docName)` | 点击删除按钮 | `deleteDocument(docName)` → 移除选中 → `fetchDocuments()` | DELETE /documents/{name} |
| `handleSelectDoc(docName)` | 点击文档项 | 切换选中/取消，最多2个 | 无 |
| `handleSendMessage(question)` | 发送消息 | ★ 流式查询核心逻辑 | POST /query/stream |
| `handleLogout()` | 点击登出 | `logout()` | 无 |

**★ `handleSendMessage` 流式查询完整流程：**
```
1. 创建 user 消息: {role: "user", content: question}
2. 创建空 assistant 消息: {role: "assistant", content: "", sources: []}
3. setIsLoading(true)
4. 确定搜索文档: selectedDocs.length > 0 ? selectedDocs : null
5. 调用 queryDocumentsStream(question, documentNames,
     onToken: (token) => {
       // 逐 token 追加到最后一条 assistant 消息
       setMessages(prev => [...prev.slice(0,-1), {...lastMsg, content: lastMsg.content + token}])
     },
     onDone: (sources) => {
       // 完成时添加 sources
       setMessages(prev => [...prev.slice(0,-1), {...lastMsg, sources}])
     }
   )
6. finally: setIsLoading(false)
```

**组件传递的 Props：**
```
<Sidebar
  documents={documents}          // 文档列表
  selectedDocs={selectedDocs}    // 选中文档
  onSelectDoc={handleSelectDoc}  // 选中/取消回调
  onUpload={handleUpload}        // 上传回调
  onDelete={handleDelete}        // 删除回调
  isUploading={isUploading}      // 上传状态
  user={user}                    // 用户信息
  onLogout={handleLogout}        // 登出回调
/>
<ChatArea
  messages={messages}            // 消息列表
  isLoading={isLoading}          // 加载状态
  onExampleClick={handleSendMessage}  // 示例问题点击回调
/>
<InputBar
  selectedDocs={selectedDocs}    // 选中文档
  onRemoveDoc={handleRemoveDoc}  // 移除文档回调
  onSendMessage={handleSendMessage}  // 发送消息回调
  disabled={isLoading}           // 禁用状态
/>
```

---

#### 5.2.6 `frontend/src/components/Sidebar.jsx` — 侧边栏

**文件位置：** `frontend/src/components/Sidebar.jsx`  
**重要程度：** ★★  
**调用者：** `Dashboard.jsx` 传入 8 个 props

**功能区域：**

| 区域 | 功能 | 实现细节 |
|------|------|----------|
| Header | Logo + 用户邮箱 + 登出按钮 | `user.email` 显示，`onLogout` 回调 |
| Document List | 文档列表，可选中/删除 | `documents.map()` 渲染，点击选中，× 按钮删除 |
| Upload Area | 拖拽上传 + 点击上传 | `onDragOver/onDrop` + 隐藏 `<input type="file">` |

**删除确认交互：** 使用 `react-hot-toast` 的自定义 toast，显示确认按钮（Delete/Cancel），`duration: Infinity` 永不自动消失。

**拖拽上传逻辑：**
```javascript
handleDrop: 
  e.preventDefault() → 获取 e.dataTransfer.files[0]
  → 验证 .pdf 后缀 → onUpload(file)
```

---

#### 5.2.7 `frontend/src/components/ChatArea.jsx` — 聊天消息区

**调用者：** `Dashboard.jsx` 传入 messages, isLoading, onExampleClick

**两种状态渲染：**

| 状态 | 渲染内容 |
|------|----------|
| 无消息 | 空状态：图标 + "Ready when you are." + 4 个示例问题按钮 |
| 有消息 | 消息列表 `<Message>` + 加载指示器 "Thinking..." |

**示例问题：**
```javascript
["Hi, what's up?", "What do you do?", "What was my highest expense?", "How much did I spend at bokku?"]
```

**自动滚动：** `useEffect` 监听 messages 变化 → `scrollIntoView({behavior: 'smooth'})`

---

#### 5.2.8 `frontend/src/components/InputBar.jsx` — 输入栏

**调用者：** `Dashboard.jsx` 传入 selectedDocs, onRemoveDoc, onSendMessage, disabled

**交互逻辑：**
- **Enter** → 发送消息（`handleSubmit`）
- **Shift+Enter** → 换行（不触发 handleSubmit）
- **动态占位文本：** 无选中文档 → "Ask a question (will search all documents)..." / 有选中文档 → "Ask about doc1, doc2..."
- **文档标签（Pills）：** 已选文档显示为可移除标签，点击 × 移除

---

#### 5.2.9 `frontend/src/components/Message.jsx` — 消息气泡

**渲染逻辑：**
```jsx
<div className={`message ${isUser ? 'user' : 'assistant'}`}>
  {!isUser && <div className="message-sources">FinQuery</div>}  // 助手消息显示品牌标签
  <div style={{whiteSpace: 'pre-wrap'}}>{message.content}</div>  // pre-wrap 保留换行和空格
</div>
```

---

#### 5.2.10 `frontend/src/pages/Login.jsx` — 登录页

**流程：**
```
提交表单 → 验证非空 → login(email, password) 
→ loginUser(data.access_token, data.email)  // 存 token + 设置用户状态
→ navigate('/')  // 跳转仪表盘
```

**错误处理：** `error.response?.data?.detail || 'Login failed'` — 优先显示后端错误详情

---

#### 5.2.11 `frontend/src/pages/Register.jsx` — 注册页

**客户端验证（在调用 API 前）：**
1. 所有字段非空
2. `password === confirmPassword`
3. `password.length >= 6`

**成功后：** 与 Login 相同 — `loginUser(token, email)` → `navigate('/')`

---

### 5.3 配置文件详解

#### 5.3.1 `frontend/vite.config.js`

```javascript
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,               // 开发服务器端口
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // 代理到后端
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),  // 去除 /api 前缀
      },
    },
  },
})
```

**开发代理的作用：** 前端 `http://localhost:5173` 请求 `/api/xxx` → Vite 代理转发到 `http://localhost:8000/xxx`，解决开发环境跨域问题。

**注意：** 生产环境不使用代理，前端通过 `VITE_API_URL` 环境变量直连后端地址。

#### 5.3.2 `frontend/.env.production`

```
VITE_API_URL=http://43.139.216.41:8000
```

生产环境后端 API 地址，Vite 构建时注入。

#### 5.3.3 `frontend/vercel.json`

```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
```

SPA 路由重写：所有路径请求都返回 index.html，由 React Router 在客户端处理路由。

#### 5.3.4 `railway.toml`

```toml
[build]
path = "backend"
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
startCommand = "uvicorn src.main:app --host 0.0.0.0 --port 8000"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

---

## 6. 前后端交互完整流程

### 6.1 用户注册流程

```
[Register.jsx]
  用户填写 email + password + confirmPassword
  → 客户端验证（非空、匹配、≥6位）
  → api.register(email, password)
    → POST /register {email, password}
      → main.py: 检查邮箱唯一性
        → auth.get_password_hash(password)  [bcrypt 哈希]
        → 创建 User 记录存入 PostgreSQL
        → auth.create_access_token({"sub": email})  [JWT 签发]
      → 返回 {access_token, token_type: "bearer", email}
  → AuthContext.loginUser(token, email)
    → localStorage.setItem('token', token)
    → setUser({email})
  → navigate('/') → Dashboard
```

### 6.2 用户登录流程

```
[Login.jsx]
  → api.login(email, password)
    → POST /login {email, password}
      → main.py: 查找 User → auth.verify_password() → auth.create_access_token()
      → 返回 {access_token, token_type, email}
  → AuthContext.loginUser(token, email) → navigate('/')
```

### 6.3 文档上传流程

```
[Sidebar.jsx] 拖拽或点击上传
  → Dashboard.handleUpload(file)
    → api.uploadDocument(file)
      → POST /upload (multipart/form-data, Bearer token)
        → main.py /upload:
          1. 验证 .pdf 后缀
          2. 保存临时文件
          3. process_pdf(together_client, temp_path)  [ingest.py]
             a. PyMuPDF 提取每页文本
             b. extract_tables_with_camelot()  [process_tables.py]
                → Camelot stream/lattice 提取表格 → Markdown
             c. enhance_table_with_context()  [process_tables.py]
                → Together AI 8B 生成表格摘要 + 清洗
             d. RecursiveCharacterTextSplitter 分块 (1000/200)
             → 返回 (chunks, pages)
          4. add_documents(chunks, filename, user_id, pages)  [vector_store.py]
             → 创建 ChromaDB 集合 (用户MD5前缀)
             → Sentence-Transformers 嵌入 (all-MiniLM-L6-v2)
             → collection.add(ids, documents, metadatas)
          5. 清除 BM25 缓存
          6. 删除临时文件
          → 返回 {filename, collection_name, pages, total_docs, message}
    → fetchDocuments() → 刷新文档列表
```

### 6.4 流式查询流程（核心！）

```
[InputBar.jsx] Enter 发送
  → Dashboard.handleSendMessage(question)
    1. 创建 user 消息 + 空 assistant 消息
    2. api.queryDocumentsStream(question, documentNames, onToken, onDone)
      → POST /query/stream {question, document_names, n_results:5} (Bearer token)
        → main.py /query/stream:
          a. get_rag_engine()  [懒加载 RAGEngine]
          b. _handle_conversational_query(question)  [检查对话式查询]
             → 如果匹配 → 直接 yield 固定回复
          c. 确定文档列表 (指定 or 全部)
          d. 检索阶段:
             单文档:
               - dense = query_collection()  [ChromaDB 向量检索, n*2]
               - sparse = bm25_retriever.search()  [BM25 稀疏检索, n*2]
               - fused = rrf([dense, sparse])  [RRF 融合]
             多文档:
               - 逐文档 retrieve_single_document() → 合并排序
          e. build_context(chunks)  [构建带引用上下文]
          f. generate_answer_stream(context, question)  [Together AI 70B 流式]
             → 逐 token yield SSE: data: {"type": "token", "content": "..."}
          g. yield SSE: data: {"type": "done", "sources": [...]}
      ← 前端 ReadableStream 逐块读取:
        - type="token" → onToken(content) → 追加到 assistant 消息
        - type="done" → onDone(sources) → 设置消息来源
    3. setIsLoading(false)
```

### 6.5 文档删除流程

```
[Sidebar.jsx] 点击 × → 确认 toast → 点击 Delete
  → Dashboard.handleDelete(docName)
    → api.deleteDocument(docName)
      → DELETE /documents/{docName} (Bearer token)
        → main.py: delete_document_collection() [vector_store.py]
        → 清除 BM25 缓存
    → 移除选中 → fetchDocuments()
```

---

## 7. 后端模块调用关系图

```
                    ┌─────────────────────────────────────────┐
                    │            main.py (API 层)              │
                    │  端点定义 + 依赖注入 + SSE 流式生成       │
                    └──┬──────┬──────┬──────┬──────┬──────────┘
                       │      │      │      │      │
            ┌──────────┘      │      │      │      └──────────┐
            │                 │      │      │                 │
            ▼                 ▼      │      ▼                 ▼
     ┌────────────┐  ┌────────────┐ │ ┌────────────┐  ┌────────────┐
     │  auth.py   │  │  ingest.py │ │ │vector_store│  │ rag_engine │
     │  认证服务   │  │  摄取管线   │ │ │  向量库管理 │  │  RAG 引擎  │
     └─────┬──────┘  └─────┬──────┘ │ └──────┬─────┘  └─────┬──────┘
           │               │        │        │               │
           │               ▼        │        │        ┌──────┴──────┐
           │     ┌────────────────┐│        │        │             │
           │     │process_tables  ││        │        ▼             ▼
           │     │  表格处理       ││        │  ┌──────────┐ ┌──────────┐
           │     └───────┬────────┘│        │  │retrieval │ │vector_   │
           │             │         │        │  │BM25+RRF  │ │store     │
           │             │ Camelot │        │  └──────────┘ └──────────┘
           │             │ + LLM   │        │
           ▼             ▼         ▼        ▼
     ┌──────────────────────────────────────────────┐
     │              database.py + models/             │
     │  SQLAlchemy 引擎/会话 + User ORM + Schemas   │
     └──────────────────────────────────────────────┘
           │
           ▼
     ┌──────────────┐    ┌──────────────┐
     │  PostgreSQL   │    │   ChromaDB   │
     │  (用户数据)   │    │  (向量数据)  │
     └──────────────┘    └──────────────┘
```

**依赖方向规则：** services/ 内部模块可互相调用（rag_engine → vector_store, retrieval），但 main.py 是唯一入口，不形成循环依赖。

---

## 8. 前端组件层级与数据流

```
App
├── AuthProvider (Context: user, loading, loginUser, logout)
│   └── Toaster
└── Routes
    ├── /login → Login
    │     └── useAuth() → loginUser()
    │     └── api.login()
    │
    ├── /register → Register
    │     └── useAuth() → loginUser()
    │     └── api.register()
    │
    └── / → ProtectedRoute → Dashboard
          ├── useAuth() → user, logout
          ├── State: documents, selectedDocs, messages, isLoading, isUploading
          │
          ├── Sidebar
          │     Props: documents, selectedDocs, onSelectDoc, onUpload, onDelete, isUploading, user, onLogout
          │     内部: fileInputRef, isDragging
          │
          ├── ChatArea
          │     Props: messages, isLoading, onExampleClick
          │     内部: messagesEndRef (自动滚动)
          │     └── Message (per message)
          │
          └── InputBar
                Props: selectedDocs, onRemoveDoc, onSendMessage, disabled
                内部: input state
```

**数据流向（单向数据流）：**
```
Dashboard (状态中心)
  ↓ props    ↓ props     ↓ props
Sidebar    ChatArea    InputBar
  ↑ events   ↑ events    ↑ events
  (onSelectDoc, onUpload, onDelete)  (onExampleClick)  (onSendMessage, onRemoveDoc)
```

---

## 9. API 端点与前后端对应关系

| API 端点 | 后端处理文件 | 前端调用位置 | 前端函数 | 触发场景 |
|----------|-------------|-------------|----------|----------|
| POST /register | main.py:117 | Register.jsx:36 | `api.register()` | 注册表单提交 |
| POST /login | main.py:149 | Login.jsx:25 | `api.login()` | 登录表单提交 |
| GET /me | main.py:74 | AuthContext.jsx:22 | `api.getCurrentUser()` | 应用挂载时验证 token |
| GET /documents | main.py:88 | Dashboard.jsx:26 | `api.listDocuments()` | 挂载/上传后/删除后 |
| GET /documents/{name} | main.py:103 | — (未使用) | — | 预留接口 |
| POST /upload | main.py:174 | Dashboard.jsx:44 | `api.uploadDocument()` | 拖拽/点击上传 PDF |
| POST /query | main.py:225 | — (未使用) | `api.queryDocuments()` | 预留非流式查询 |
| POST /query/stream | main.py:253 | Dashboard.jsx:103 | `api.queryDocumentsStream()` | 发送消息 |
| DELETE /documents/{name} | main.py:308 | Dashboard.jsx:56 | `api.deleteDocument()` | 确认删除文档 |
| DELETE /documents | main.py:327 | — (未使用) | — | 预留清空接口 |

---

## 10. 部署架构

```
┌─────────────────────────────────────────────────┐
│                   Vercel (前端)                   │
│  静态 SPA 部署                                    │
│  VITE_API_URL = http://43.139.216.41:8000        │
│  vercel.json: SPA rewrite                        │
└──────────────────────┬──────────────────────────┘
                       │ HTTPS
                       ▼
┌──────────────────────────────────────────────────┐
│                 Railway (后端)                     │
│  Docker 容器部署 (python:3.13-slim)               │
│  railway.toml: DOCKERFILE builder                │
│  启动: uvicorn src.main:app --host 0.0.0.0       │
│                                                    │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  PostgreSQL   │  │    ChromaDB (Volume)     │  │
│  │  (Railway 托管)│  │    ./chroma_db 持久化    │  │
│  │  用户认证数据  │  │    文档向量 + 元数据     │  │
│  └──────────────┘  └──────────────────────────┘  │
└──────────────────────────────────────────────────┘

开发环境:
  前端: http://localhost:5173 (Vite dev server)
  后端: http://localhost:8000 (uvicorn)
  代理: Vite proxy /api → localhost:8000
  数据库: SQLite (./temp.db)
  向量库: ChromaDB (./chroma_db)
```

---

## 11. 环境变量汇总

| 变量名 | 使用位置 | 用途 | 默认值 |
|--------|----------|------|--------|
| `DATABASE_URL` | database.py | PostgreSQL 连接字符串 | `sqlite:///./temp.db` |
| `SECRET_KEY` | auth.py | JWT 签名密钥 | 硬编码默认值 |
| `TOGETHER_API_KEY` | main.py (Together()) | Together AI API 密钥 | 无（必须设置） |
| `ALLOWED_ORIGINS` | main.py | CORS 允许源（逗号分隔） | `localhost:5173` |
| `CHROMA_PATH` | vector_store.py | ChromaDB 持久化路径 | `./chroma_db` |
| `VITE_API_URL` | api.js (前端) | 后端 API 基地址 | `http://localhost:8000` |

---

## 12. 面试高频问题预备

### 12.1 RAG 相关

**Q: 什么是 RAG？为什么要用 RAG 而不是直接让 LLM 回答？**
> RAG (Retrieval-Augmented Generation) 先从外部知识库检索相关文档，再将检索结果作为上下文喂给 LLM 生成答案。优势：(1) 避免幻觉（基于真实文档回答）；(2) 知识可更新（新文档入库即可查询）；(3) 可溯源（引用来源页码）；(4) 节省成本（不需要 fine-tune 大模型）。

**Q: 为什么用混合检索？纯向量检索不够吗？**
> 纯向量检索擅长语义相似性，但对精确数字/名称匹配较弱。例如查询"2024年Q3收入"，BM25 能精确匹配"2024"、"Q3"这些关键词，而向量检索可能返回语义相关但年份不对的内容。混合检索互补提升召回率。

**Q: RRF 融合的原理和优势？**
> RRF 只使用排名信息 `1/(k+rank+1)`，不依赖绝对分数，因此天然消除不同检索方法的分数尺度差异。对比加权融合，RRF 无需调权重、对异常值鲁棒、实现简单。

### 12.2 系统设计相关

**Q: 为什么每个文档用独立 ChromaDB 集合？**
> (1) 数据隔离：不同文档互不干扰；(2) 精确检索：单文档查询只搜一个集合，减少噪音；(3) 灵活组合：多文档查询时逐集合检索再合并；(4) 生命周期管理：删除文档直接删集合。

**Q: 为什么用 MD5 哈希用户 ID 做集合名前缀？**
> ChromaDB 集合名限制 3-63 字符、仅允许字母数字+下划线+连字符。用户邮箱含 @ 等非法字符，MD5 哈希转为合法字符且固定长度，同时保证同用户映射到相同前缀。

**Q: 为什么用 SSE 而不是 WebSocket？**
> SSE 是单向推送（服务器→客户端），适合 LLM 流式输出场景。WebSocket 支持双向通信但更复杂。SSE 基于 HTTP，天然兼容代理/负载均衡/CORS，实现更简单。

### 12.3 技术选型相关

**Q: 为什么选 ChromaDB 而不是 Pinecone/Milvus？**
> ChromaDB 是嵌入式向量数据库，无需额外服务部署，PersistentClient 直接本地持久化，适合中小规模应用。Pinecone 是托管服务需付费，Milvus 适合大规模生产但部署复杂。本项目是 MVP/演示级别，ChromaDB 开发运维成本最低。

**Q: 为什么用 Together AI 而不是 OpenAI？**
> Together AI 提供 Meta-Llama-3.1 系列模型的开源推理服务，API 兼容 OpenAI 格式，成本更低。70B 模型做答案生成（需要强推理），8B 模型做表格预处理（高频低成本）。

**Q: 为什么嵌入模型选 all-MiniLM-L6-v2？**
> 轻Google 系模型，384维嵌入，速度快（~0.01s/句），模型小（~80MB），语义质量对金融文档足够。对比 BGE-large 虽然质量更高但维度更高、推理更慢。
