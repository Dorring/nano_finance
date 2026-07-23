#!/usr/bin/env python3
"""build_phase5_eval_index.py

Builds per-partition ChromaDB (dense) and SQLite FTS5 (BM25) indexes for
the Phase 5 evaluation corpus.

The corpus lives under ``eval_corpus/phase5/{dev,calibration,sealed}/``
as Markdown files with ``--- Page N ---`` delimiters. This script parses
each document into page-level chunks and indexes them using the production
ingestion APIs so the evaluation retriever exercises the same code paths
as production.

Indexes are written to ``indexes/phase5/{partition}/`` with separate
ChromaDB persistent directories and BM25 SQLite databases per partition.
A chunk manifest JSON is written alongside the indexes recording every
chunk ID, document name, page, and content hash.

Evaluation user IDs:
    dev:          9001
    calibration:  9002
    sealed:       9003

Run from the ``finquery_rag/backend/`` directory::

    python scripts/build_phase5_eval_index.py

Exit codes:
    0 - all indexes built successfully
    1 - an error occurred
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CORPUS_DIR = ROOT_DIR / "eval_corpus" / "phase5"
INDEX_DIR = ROOT_DIR / "indexes" / "phase5"

# Ensure backend root is on sys.path for production imports
sys.path.insert(0, str(ROOT_DIR))

PARTITIONS: tuple[str, ...] = ("dev", "calibration", "sealed")
EVAL_USER_IDS: dict[str, int] = {
    "dev": 9001,
    "calibration": 9002,
    "sealed": 9003,
}

PAGE_DELIMITER = re.compile(r"^---\s*Page\s+(\d+)\s*---\s*$", re.MULTILINE)


def parse_document_pages(md_path: Path) -> list[tuple[int, str]]:
    """Parse a markdown document into (page_number, content) tuples.

    The document uses ``--- Page N ---`` delimiters. Content before the
    first delimiter (if any) is assigned to page 1.
    """
    text = md_path.read_text(encoding="utf-8")
    parts = PAGE_DELIMITER.split(text)
    pages: list[tuple[int, str]] = []
    # split produces: [pre, page_num_1, content_1, page_num_2, content_2, ...]
    if len(parts) == 1:
        # No delimiters found — entire file is page 1
        content = parts[0].strip()
        if content:
            pages.append((1, content))
        return pages

    # parts[0] is content before first delimiter
    pre = parts[0].strip()
    if pre:
        pages.append((1, pre))

    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            pages.append((page_num, content))

    return pages


def build_chunks_for_document(
    md_path: Path,
    doc_name: str,
    user_id: int,
) -> list[dict]:
    """Create chunk dicts from a markdown document.

    Each page becomes one content chunk. The first page also gets a
    front_matter_title chunk extracted from the first heading.
    """
    pages = parse_document_pages(md_path)
    chunks: list[dict] = []

    for page_num, content in pages:
        # Extract title from first heading on page 1
        if page_num == 1:
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
                chunks.append({
                    "content": f"Title: {title}",
                    "metadata": {
                        "type": "front_matter",
                        "subtype": "title",
                        "doc_id": f"page_{page_num}::front_matter_title",
                        "doc_name": doc_name,
                        "page": page_num,
                        "user_id": user_id,
                    },
                })

        chunk_id_suffix = f"page_{page_num}::content"
        chunks.append({
            "content": content,
            "metadata": {
                "type": "content",
                "doc_id": chunk_id_suffix,
                "doc_name": doc_name,
                "page": page_num,
                "user_id": user_id,
            },
        })

    return chunks


def build_partition_index(partition: str) -> dict:
    """Build ChromaDB and BM25 indexes for one partition.

    Returns a dict with partition metadata and chunk manifest.
    """
    user_id = EVAL_USER_IDS[partition]
    corpus_partition_dir = CORPUS_DIR / partition
    index_partition_dir = INDEX_DIR / partition
    chroma_path = index_partition_dir / "chroma"
    bm25_path = str(index_partition_dir / "rag_bm25.db")

    # Clean previous indexes
    import shutil
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    if Path(bm25_path).exists():
        Path(bm25_path).unlink()
    # Also remove WAL/SHM files
    for suffix in ("-wal", "-shm"):
        wal = Path(bm25_path + suffix)
        if wal.exists():
            wal.unlink()

    index_partition_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for production code
    os.environ["CHROMA_PATH"] = str(chroma_path)
    os.environ["BM25_DB_PATH"] = bm25_path

    # Reset ChromaDB client cache (module-level singleton)
    import src.services.vector_store as vs
    vs._chroma_client = None

    # Import production APIs
    from src.services.vector_store import add_documents
    from src.services.retrieval import SqliteBM25Retriever

    bm25_retriever = SqliteBM25Retriever(db_path=bm25_path)

    # Collect all chunks for this partition
    md_files = sorted(corpus_partition_dir.glob("*.md"))
    if not md_files:
        raise RuntimeError(f"No markdown files found in {corpus_partition_dir}")

    chunk_manifest: list[dict] = []
    total_chunks = 0

    for md_path in md_files:
        doc_name = md_path.name
        chunks = build_chunks_for_document(md_path, doc_name, user_id)

        # Add to ChromaDB
        add_documents(chunks, doc_name=doc_name, user_id=user_id, pages=4)

        # Add to BM25
        bm25_retriever.add_chunks(chunks, user_id=user_id)

        # Record in manifest
        for chunk in chunks:
            content_hash = hashlib.sha256(
                chunk["content"].encode("utf-8")
            ).hexdigest()
            chunk_manifest.append({
                "chunk_id": f"user_{user_id}_{doc_name}::{chunk['metadata']['doc_id']}",
                "doc_name": doc_name,
                "page": chunk["metadata"]["page"],
                "type": chunk["metadata"]["type"],
                "content_sha256": content_hash,
                "content_length": len(chunk["content"]),
            })
            total_chunks += 1

        print(f"  [{partition}] {doc_name}: {len(chunks)} chunks indexed")

    # Write chunk manifest
    manifest_path = index_partition_dir / "chunk-manifest.json"
    manifest_data = {
        "partition": partition,
        "user_id": user_id,
        "document_count": len(md_files),
        "chunk_count": total_chunks,
        "chunks": chunk_manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"  [{partition}] Total: {total_chunks} chunks, {len(md_files)} documents")
    return manifest_data


def verify_index(partition: str) -> None:
    """Verify that the index can retrieve documents for each partition."""
    user_id = EVAL_USER_IDS[partition]
    index_partition_dir = INDEX_DIR / partition

    os.environ["CHROMA_PATH"] = str(index_partition_dir / "chroma")
    os.environ["BM25_DB_PATH"] = str(index_partition_dir / "rag_bm25.db")

    import src.services.vector_store as vs
    vs._chroma_client = None

    from src.services.vector_store import query_collection
    from src.services.retrieval import SqliteBM25Retriever

    bm25 = SqliteBM25Retriever(db_path=str(index_partition_dir / "rag_bm25.db"))

    # Test query
    test_query = "营业收入"
    dense_results = query_collection(
        query_text=test_query,
        user_id=user_id,
        n_results=3,
    )
    bm25_results = bm25.search(query=test_query, k=3, user_id=user_id)

    dense_count = len(dense_results) if isinstance(dense_results, list) else 0
    bm25_count = len(bm25_results) if isinstance(bm25_results, list) else 0

    if dense_count == 0:
        print(f"  WARNING [{partition}] Dense retrieval returned 0 results")
    if bm25_count == 0:
        print(f"  WARNING [{partition}] BM25 retrieval returned 0 results")

    print(
        f"  [{partition}] Verification: dense={dense_count}, bm25={bm25_count}"
    )


def main() -> int:
    print("=" * 60)
    print("Phase 5 Evaluation Index Builder")
    print("=" * 60)

    if not CORPUS_DIR.exists():
        print(f"ERROR: Corpus directory not found: {CORPUS_DIR}")
        return 1

    all_manifests: dict[str, dict] = {}
    for partition in PARTITIONS:
        print(f"\n--- Building {partition} partition ---")
        manifest = build_partition_index(partition)
        all_manifests[partition] = manifest
        verify_index(partition)

    # Write top-level index manifest
    top_manifest = {
        "phase": "phase5",
        "partitions": {
            p: {
                "user_id": EVAL_USER_IDS[p],
                "document_count": all_manifests[p]["document_count"],
                "chunk_count": all_manifests[p]["chunk_count"],
            }
            for p in PARTITIONS
        },
        "total_documents": sum(
            all_manifests[p]["document_count"] for p in PARTITIONS
        ),
        "total_chunks": sum(
            all_manifests[p]["chunk_count"] for p in PARTITIONS
        ),
    }
    top_path = INDEX_DIR / "index-manifest.json"
    top_path.write_text(
        json.dumps(top_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print("Index building complete.")
    print(f"  Total documents: {top_manifest['total_documents']}")
    print(f"  Total chunks: {top_manifest['total_chunks']}")
    print(f"  Index directory: {INDEX_DIR}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
