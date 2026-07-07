import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from typing import List, Dict, Optional
from .chunk_id import ensure_scoped_chunk_id

# 使用环境变量配置 ChromaDB 路径
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

# 全局统一集合名称，彻底摒弃“一个文档一个集合”的设计
GLOBAL_COLLECTION_NAME = "rag_global_knowledge_base"

# 初始化 Embedding 模型
# 注意：all-MiniLM-L6-v2 对中文支持较弱，生产环境建议替换为中文友好模型或 OpenAI 接口
embed_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# <--------------- 重构：单集合 + 元数据过滤架构 ----------------->

# 模块级全局客户端缓存，避免频繁实例化导致资源浪费和 SQLite 锁冲突
_chroma_client = None

def get_chroma_client() -> chromadb.PersistentClient:
    """
    获取单例的 ChromaDB 持久化客户端。
    """
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _chroma_client

def get_or_create_collection() -> chromadb.Collection:
    """
    获取全局唯一的向量集合。
    核心改变：不再按文档创建集合，所有数据存入一个集合，通过 metadata 隔离。
    """
    client = get_chroma_client()
    # 显式指定 hnsw:space 为 cosine，确保 1 - distance 转换相似度在数学上成立
    return client.get_or_create_collection(
        name=GLOBAL_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

def add_documents(chunks: list, doc_name: str, user_id: int = None, pages: int = None) -> dict:
    """
    将处理后的文本块添加到全局集合中。

    Args:
        chunks (list): 已处理的文本块列表。
        doc_name (str): 原始文档的文件名。
        user_id (str, optional): 用户ID。默认为 None。
        pages (int, optional): 文档的页数。默认为 None。

    Returns:
        dict: 包含集合名称和文档总数的字典。
    """
    collection = get_or_create_collection()

    ids = []
    documents = []
    metadatas = []

    for c in chunks:
        raw_id = c.get("metadata", {}).get("doc_id") or c.get("id")
        content = c["content"]
        metadata = c.get("metadata", {})

        # Enforce tenant-scoped ID at storage boundary
        doc_id = ensure_scoped_chunk_id(raw_id, user_id, doc_name)

        # 强制注入 doc_name 和 user_id 到元数据，这是多租户隔离的生命线
        metadata["doc_name"] = doc_name
        if user_id is not None:
            metadata["user_id"] = user_id
        if pages:
            metadata["pages"] = pages

        ids.append(doc_id)
        documents.append(content)
        metadatas.append(metadata)

    # 分批写入，防止大文档导致 OOM
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]

        # 使用 upsert 防止重复插入报错
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas
        )

    print(f"✓ Added/Updated {len(chunks)} chunks for doc '{doc_name}' in global collection.")
    return {
        "collection_name": collection.name,
        "total_docs": collection.count()
    }

def query_collection(
    query_text: str,
    doc_name: str = None,
    n_results: int = 5,
    user_id: int = None
) -> List[Dict]:
    """
    在全局集合中查询特定文档/用户的相似文本。

    Args:
        query_text (str): 查询文本内容。
        doc_name (str, optional): 限定查询的文档名。默认为 None (搜索该用户所有文档)。
        n_results (int, optional): 返回的结果数量。默认为 5。
        user_id (str, optional): 限定查询的用户ID。默认为 None。

    Returns:
        list: 匹配结果列表，格式与 BM25 模块完全一致，包含 doc_id, content, metadata, score。
    """
    collection = get_or_create_collection()

    # 构建多租户过滤条件 - 无 user_id 时拒绝查询（fail closed）
    if user_id is None:
        return []
    where_filter = {"user_id": user_id}
    if doc_name:
        where_filter["doc_name"] = doc_name

    # 执行查询
    query_results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )

    # 整理结果格式，确保与 retrieval.py (BM25) 输出结构一致
    results = []
    if query_results and query_results["ids"][0]:
        for doc_id, doc, meta, distance in zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            results.append({
                "doc_id": doc_id,
                "content": doc,
                "metadata": meta,
                # 因为我们强制使用了 cosine 距离，1 - distance 就是余弦相似度 (0~1)
                "score": 1 - distance
            })

    return results

def query_multiple_collections(
    doc_names: List[str],
    query_text: str,
    n_results: int = 5,
    user_id: int = None
) -> List[Dict]:
    """
    跨多个文档进行查询。
    核心优化：不再使用 for 循环遍历集合，而是利用 ChromaDB 的 $in 操作符一次查完。

    Args:
        doc_names (list[str]): 要查询的文档文件名列表。
        query_text (str): 查询文本内容。
        n_results (int, optional): 返回的最终结果数量。默认为 5。
        user_id (str, optional): 用户ID。默认为 None。

    Returns:
        list: 按相似度得分降序排列的前 n_results 个结果列表。
    """
    collection = get_or_create_collection()

    # 构建高级过滤条件 - 无 user_id 时拒绝查询（fail closed）
    if user_id is None:
        return []
    where_filter = {"user_id": user_id}

    # 核心优化点：单次 IO 解决跨文档查询
    if doc_names:
        where_filter["doc_name"] = {"$in": doc_names}

    query_results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where_filter
    )

    # 复用格式化逻辑
    results = []
    if query_results and query_results["ids"][0]:
        for doc_id, doc, meta, distance in zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0]
        ):
            results.append({
                "doc_id": doc_id,
                "content": doc,
                "metadata": meta,
                "score": 1 - distance
            })

    return results

def list_all_documents(user_id: int = None) -> List[Dict]:
    """
    列出数据库中的所有文档信息。
    由于采用单集合架构，需要通过提取元数据来统计文档列表。

    Args:
        user_id (str, optional): 用户ID。默认为 None。

    Returns:
        list[dict]: 包含文档信息的字典列表。
    """
    collection = get_or_create_collection()

    # 获取集合中所有数据的元数据 (注意：如果数据量达到千万级，此操作较重)
    if user_id is None:
        return []
    where_filter = {"user_id": user_id}
    all_metas = collection.get(include=["metadatas"], where=where_filter)["metadatas"]

    # 聚合统计文档信息
    doc_stats = {}
    for meta in all_metas:
        d_name = meta.get("doc_name", "unknown")
        if d_name not in doc_stats:
            doc_stats[d_name] = {
                "name": d_name,
                "count": 0,
                "pages": meta.get("pages", None)
            }
        doc_stats[d_name]["count"] += 1

    return list(doc_stats.values())

def delete_document_collection(doc_name: str, user_id: int) -> bool:
    """
    删除特定文档的所有向量数据。user_id 必须提供，不允许跨租户删除。
    修复：删除前先检查是否存在匹配数据，0 条匹配返回 False。

    Args:
        doc_name (str): 要删除的文档文件名。为 None 时删除该用户全部数据。
        user_id (int): 用户ID，必填。

    Returns:
        bool: 删除成功且有数据被删除返回 True，无匹配数据返回 False。

    Raises:
        ValueError: user_id 为 None 时抛出。
    """
    if user_id is None:
        raise ValueError("user_id is required for delete_document_collection")
    collection = get_or_create_collection()

    where_filter = {"user_id": user_id}
    if doc_name is not None:
        where_filter["doc_name"] = doc_name

    try:
        # Check existence before delete (fixes 404 semantics)
        existing = collection.get(where=where_filter, limit=1)
        if not existing or not existing.get("ids"):
            return False
        collection.delete(where=where_filter)
        return True
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        return False


def clear_all_for_user(user_id: int) -> bool:
    """
    清除指定用户的全部向量数据。强制要求 user_id。

    Args:
        user_id (int): 用户ID，必填。

    Returns:
        bool: 删除成功返回 True。
    """
    return delete_document_collection(doc_name=None, user_id=user_id)

def get_collection_stats(doc_name: str = None, user_id: int = None) -> dict:
    """
    获取集合的统计信息。

    Args:
        doc_name (str, optional): 文档名。默认为 None。
        user_id (str, optional): 用户ID。默认为 None。

    Returns:
        dict: 统计信息字典。
    """
    collection = get_or_create_collection()

    where_filter = {}
    if doc_name:
        where_filter["doc_name"] = doc_name
    if user_id is None:
        return {"name": "", "count": 0, "exists": False}
    where_filter["user_id"] = user_id

    count = collection.count() if not where_filter else len(collection.get(where=where_filter)["ids"])

    return {
        "name": collection.name,
        "count": count,
        "exists": count > 0
    }
