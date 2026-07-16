import pymupdf
import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from .process_tables import enhance_table_with_context, extract_tables_with_camelot
from .chunk_id import make_chunk_id

# 1. 用于长章节内部二次切分的备选方案
# 适配 2048 上下文：chunk_size 从 1000 降至 350，overlap 从 200 降至 50
RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

# 2. 核心：Markdown 逻辑结构切分器配置
MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ],
    strip_headers=False
)

# 长章节二次切分阈值：从 1500 降至 500，适配短上下文
LONG_CHUNK_THRESHOLD = 500


def _analyze_font_hierarchy(page: pymupdf.Page) -> dict:
    """
    动态分析当前页的字体层级。
    返回一个 {字体大小: markdown层级} 的映射字典。
    """
    spans_info = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    spans_info.append({"size": span["size"], "text": span["text"]})

    if not spans_info:
        return {}

    unique_sizes = sorted(list(set([s["size"] for s in spans_info])), reverse=True)
    sizes_list = [s["size"] for s in spans_info]
    most_common_size = max(set(sizes_list), key=sizes_list.count)

    hierarchy_map = {}
    level = 1
    for size in unique_sizes:
        if size > most_common_size and level <= 3:
            hierarchy_map[size] = level
            level += 1

    return hierarchy_map


def _reconstruct_page_to_markdown(page: pymupdf.Page, tab_bboxes: list, hierarchy_map: dict) -> str:
    """
    将 PyMuPDF 提取的纯文本，逆向重构为 Markdown 格式。
    剔除表格区域，并根据字体层级添加 # 符号。
    """
    md_lines = []
    blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        if block["type"] != 0:
            continue

        block_rect = pymupdf.Rect(block["bbox"])
        is_in_table = any(block_rect.intersects(pymupdf.Rect(tab)) for tab in tab_bboxes)
        if is_in_table:
            continue

        block_text_parts = []
        block_main_size = 0
        for line in block["lines"]:
            for span in line["spans"]:
                block_text_parts.append(span["text"])
                if span["size"] > block_main_size:
                    block_main_size = span["size"]

        full_text = "".join(block_text_parts).strip()
        if not full_text:
            continue

        if block_main_size in hierarchy_map:
            level = hierarchy_map[block_main_size]
            md_lines.append(f"{'#' * level} {full_text}")
        else:
            md_lines.append(full_text)

    return "\n\n".join(md_lines)




def _safe_find_table_bboxes(page: pymupdf.Page) -> list:
    """Return PyMuPDF table bboxes, or an empty list when detection fails."""
    try:
        finder = page.find_tables()
        tables = getattr(finder, "tables", []) or []
    except Exception as exc:
        print(f"PyMuPDF table detection failed on page {page.number + 1}: {exc}; continuing without table bboxes.")
        return []

    bboxes = []
    for table in tables:
        try:
            bbox = getattr(table, "bbox", None)
            if bbox:
                bboxes.append(tuple(bbox))
        except Exception as exc:
            print(f"Skipping PyMuPDF table bbox on page {page.number + 1}: {exc}")
    return bboxes

def _clean_front_matter_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^\d{1,4}\s+", "", text)
    text = re.sub(r"\s+\d{1,4}$", "", text)
    return text.strip()


def _extract_title_from_first_page(page: pymupdf.Page) -> str | None:
    """Extract a likely title from page 1 using font-size/layout signals."""
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return None

    lines = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            raw_text = " ".join(span.get("text", "").strip() for span in spans)
            clean_text = _clean_front_matter_line(raw_text)
            if not clean_text or clean_text.isdigit():
                continue
            max_size = max(float(span.get("size", 0) or 0) for span in spans)
            y0 = min(float(span.get("bbox", [0, 0, 0, 0])[1]) for span in spans)
            lines.append({"text": clean_text, "size": max_size, "y0": y0})

    if not lines:
        return None

    max_size = max(line["size"] for line in lines)
    min_title_size = max(12.0, max_size * 0.75)
    page_height = float(getattr(page.rect, "height", 1000) or 1000)
    title_lines = []

    for line in sorted(lines, key=lambda item: item["y0"]):
        lower = line["text"].lower()
        if lower.startswith(("abstract", "keywords", "paper id", "anonymous")):
            if title_lines:
                break
            continue
        if line["y0"] > page_height * 0.45 and title_lines:
            break
        if line["size"] >= min_title_size:
            title_lines.append(line["text"])
        elif title_lines:
            break

    if not title_lines:
        return None
    title = " ".join(title_lines[:5])
    title = re.sub(r"\s+", " ", title).strip(" -")
    return title or None

def process_pdf(pdf_path: str, user_id: int = None) -> tuple[list[dict], int]:
    """
    经济型结构化切分管线：基于规则重构Markdown，实现树状逻辑切分。
    适配 2048 上下文：更小的 chunk_size，更紧凑的切分策略。

    :param pdf_path: 待处理的PDF文件路径
    :return: (切分后的文本块列表, PDF总页数)
    """
    chunks = []
    doc_name = os.path.basename(pdf_path)

    doc = pymupdf.open(pdf_path)
    pages = len(doc)

    print(f"\n{'=' * 60}")
    print(f"[Economic Structured Mode] Processing: {doc_name}")
    print(f"{'=' * 60}")

    tables_by_page = extract_tables_with_camelot(pdf_path)

    title = _extract_title_from_first_page(doc[0]) if pages else None
    if title:
        title_doc_id = make_chunk_id(user_id, doc_name, "page_1::front_matter_title")
        chunks.append({
            "content": f"Title: {title}",
            "metadata": {
                "type": "front_matter",
                "subtype": "title",
                "page": 1,
                "source": pdf_path,
                "doc_id": title_doc_id,
            },
        })

    global_markdown_text = ""
    current_page_metadata = {"source": pdf_path}

    for page_num in range(pages):
        page = doc[page_num]
        actual_page_num = page_num + 1

        tab_bboxes = _safe_find_table_bboxes(page)
        hierarchy_map = _analyze_font_hierarchy(page)
        page_md = _reconstruct_page_to_markdown(page, tab_bboxes, hierarchy_map)

        if not page_md.strip():
            continue

        global_markdown_text += f"\n\n#### 🈳PAGE_MARKER_{actual_page_num}\n\n{page_md}"

    # Markdown 逻辑切分
    try:
        md_splits = MARKDOWN_SPLITTER.split_text(global_markdown_text)
    except Exception as e:
        print(f"Markdown split failed, fallback to recursive. Error: {e}")
        fallback_docs = [Document(page_content=global_markdown_text, metadata=current_page_metadata)]
        md_splits = RECURSIVE_SPLITTER.split_documents(fallback_docs)

    # 组装文本块
    for chunk_idx, split_doc in enumerate(md_splits):
        content = split_doc.page_content
        metadata = split_doc.metadata.copy()

        actual_page = 1

        if "Header 4" in metadata and "🈳PAGE_MARKER_" in metadata.get("Header 4", ""):
            try:
                actual_page = int(metadata["Header 4"].split("_")[-1])
            except ValueError:
                pass

            content = content.replace(f"#### {metadata['Header 4']}", "").strip()
            del metadata["Header 4"]

        if not content:
            continue

        # 长章节二次切分（阈值降低适配短上下文）
        if len(content) > LONG_CHUNK_THRESHOLD:
            sub_docs = RECURSIVE_SPLITTER.split_documents([Document(page_content=content, metadata=metadata)])
            for sub_idx, sub_doc in enumerate(sub_docs):
                doc_id = make_chunk_id(user_id, doc_name, f"page_{actual_page}::chunk_{chunk_idx}_{sub_idx}")
                chunks.append({
                    "content": sub_doc.page_content.strip(),
                    "metadata": {
                        **metadata,
                        "type": "text",
                        "page": actual_page,
                        "source": pdf_path,
                        "doc_id": doc_id
                    }
                })
        else:
            doc_id = make_chunk_id(user_id, doc_name, f"page_{actual_page}::chunk_{chunk_idx}")
            chunks.append({
                "content": content,
                "metadata": {
                    **metadata,
                    "type": "text",
                    "page": actual_page,
                    "source": pdf_path,
                    "doc_id": doc_id
                }
            })

    # 处理表格块（使用 NVIDIA API 增强表格上下文）
    for actual_page_num, table_list in tables_by_page.items():
        page = doc[actual_page_num - 1]
        page_text = page.get_text("text")

        for table_idx, table_md_dict in enumerate(table_list):
            # 使用 NVIDIA API 增强表格（不再需要 llm_client 参数）
            enhanced_table = enhance_table_with_context(table_md_dict, page_text, actual_page_num)
            doc_id = make_chunk_id(user_id, doc_name, f"page_{actual_page_num}::table_{table_idx + 1}")

            # 将增强结果拼接为字符串，与文本块的 content 类型保持一致
            table_content = enhanced_table["content"]
            if enhanced_table["summary"]:
                table_content = f"Summary: {enhanced_table['summary']}\n\n{table_content}"

            chunks.append({
                "content": table_content,
                "metadata": {
                    "type": "table",
                    "page": actual_page_num,
                    "source": pdf_path,
                    "doc_id": doc_id,
                    "table_num": table_idx + 1
                }
            })

    doc.close()

    table_count = sum(1 for c in chunks if c["metadata"]["type"] == "table")
    text_count = len(chunks) - table_count

    print(f"✓ Extracted {len(chunks)} structured chunks: ({text_count} text, {table_count} tables)")
    print(f"{'=' * 60}\n")

    return chunks, pages
