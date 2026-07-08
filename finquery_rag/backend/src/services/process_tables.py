import camelot
import re
import os
import requests


def format_table(table):
    """格式化表格并转换为 Markdown"""
    if table.df.empty or len(table.df) < 1:
        return ""

    # 清除单元格内的换行和制表符
    formatted_table = table.df.apply(lambda x: x.str.replace('\n', '').str.replace('\t', ' '))

    # 鲁棒性增强：尝试将第一行提升为表头，若失败则保留原状
    try:
        final_table = formatted_table.rename(columns=formatted_table.iloc[0]).drop(formatted_table.index[0]).reset_index(drop=True)
    except Exception:
        final_table = formatted_table

    return final_table.to_markdown(index=False)


def extract_tables_with_camelot(pdf_path: str, pages: str = "1-end") -> dict[int, list[dict]]:
    """
    使用 Camelot 从 PDF 提取表格，转为 Markdown，并保留边界框 (bbox)。
    Returns: {page_num: [{"md": table1_md, "bbox": table1_bbox}, ...]}
    """
    tables_by_page = {}
    tables = []

    # 优先使用 stream 模式 (适合无边框表格，如银行流水)
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream", edge_tol=50, row_tol=10)
        if len(tables) == 0:
            print("Stream mode found 0 tables, falling back to lattice.")
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice')
    except Exception as e:
        print(f"Stream mode failed: {e}, falling back to lattice.")
        try:
            tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice')
        except Exception as e2:
            print(f"Lattice mode also failed: {e2}")

    for table in tables:
        page_num = int(table.page)
        if page_num not in tables_by_page:
            tables_by_page[page_num] = []

        table_markdown = format_table(table=table)
        if not table_markdown:
            continue

        tables_by_page[page_num].append({
            "md": table_markdown,
            "bbox": tuple(table.bbox)
        })

    print(f"✓ Extracted {len(tables)} tables in total")
    return tables_by_page


def enhance_table_with_context(table_md: dict, page_text: str, page_num: int) -> dict:
    """
    使用 NVIDIA 开放平台的 Llama-3.1-8B 模型对 Markdown 表格进行清洗和摘要。
    离线数据清洗任务，不占用本地 GPU 资源。

    Args:
        table_md: 包含 "md" (markdown表格) 和 "bbox" 的字典
        page_text: 同页面的上下文文本
        page_num: 页码

    Returns:
        dict: {"summary": "表格摘要", "content": "清洗后的表格"}
    """
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_model = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

    # 如果没有配置 NVIDIA API Key，跳过增强，直接返回原始表格
    if not nvidia_api_key:
        print("⚠️ NVIDIA_API_KEY not set, skipping table enhancement")
        return {"summary": "", "content": table_md["md"]}

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    prompt = f"""You are a data preprocessing assistant for a retrieval system operating on financial documents.

Rules:
1. Write a concise 2-3 sentence description of what the table represents
2. Clean the table: remove stray text, preserve headers and alignment
3. Do NOT change numeric values, dates, currencies, or units
4. Do NOT infer missing values or recompute totals

Page number: {page_num}

Table (markdown):
{table_md["md"]}

Surrounding text from the same page:
{page_text}

Return the result in this format:
TABLE SUMMARY:
<summary text>

CLEANED TABLE:
<markdown table>"""

    payload = {
        "model": nvidia_model,
        "temperature": 0.2,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1024,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {nvidia_api_key}"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response_json = response.json()

        enhanced = response_json["choices"][0]["message"]["content"].strip()

        # 解析输出，分离摘要和表格
        summary = ""
        cleaned_table = table_md["md"]  # 默认降级为原始表格

        summary_match = re.search(r"TABLE SUMMARY:\s*(.*?)(?=\nCLEANED TABLE:)", enhanced, re.DOTALL)
        table_match = re.search(r"CLEANED TABLE:\s*(.*)", enhanced, re.DOTALL)

        if summary_match:
            summary = summary_match.group(1).strip()
        if table_match:
            parsed_table = table_match.group(1).strip()
            if parsed_table:
                cleaned_table = parsed_table

        return {"summary": summary, "content": cleaned_table}

    except Exception as e:
        print(f"NVIDIA API table enhancement failed: {e}")
        return {"summary": "", "content": table_md["md"]}
