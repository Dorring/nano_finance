# Reproducibility Guide — NanoFinance 复现指南

## 概述

本文档提供 NanoFinance 项目各阶段的复现命令，包括 Tokenizer 训练、预训练、SFT、推理、RAG 部署。**重要声明**：预训练数据相关的部分标记为 `historical artifact unavailable`——原始训练日志不可重新获取，17.68B 总 Token 池等数字为 **[历史自报]**，无法完整复现。本指南提供的命令为项目实际使用的命令模板，具体参数应以仓库内的 manifest/配置文件为准。训练硬件存在历史记录冲突（A6000×2 vs RTX 4090×3），未独立验证。

---

## 1. 环境准备

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 硬件 | 历史记录冲突（A6000×2 vs RTX 4090×3），未独立验证 | [历史记录冲突] |
| 基础项目 | karpathy/nanochat (MIT License) | [已验证] |
| 操作系统 | Linux（训练）/ Windows（开发） | [已验证] |

> 具体依赖安装请参考 nanochat 仓库的 `requirements.txt` / `pyproject.toml`。

---

## 2. Tokenizer 训练

### 2.1 训练命令

```bash
# Tokenizer 训练（Byte-Level BPE, RustBPE, GPT-4 style）
# vocab_size=65000, max_chars=2B, doc_cap=10000
# 训练数据: 40% 英文通用(ClimbMix) + 40% 中文通用(SkyPile/Wiki) + 20% 中文金融
python train_tokenizer.py \
    --vocab_size 65000 \
    --max_chars 2000000000 \
    --doc_cap 10000 \
    --output_dir <tokenizer_output_dir> \
    --data_config <data_config_path>
```

### 2.2 关键参数

| 参数 | 值 | 来源 |
| --- | --- | --- |
| `--vocab_size` | 65000 | [已验证] |
| `--max_chars` | 2,000,000,000（2B） | [已验证] |
| `--doc_cap` | 10000 | [已验证] |
| Split pattern | GPT-4 style，含 `\p{N}{1,2}` | [已验证] |
| Special Tokens | 9 个（见 `tokenizer-card.md`） | [已验证] |

> 具体脚本路径与参数名称以仓库内实际实现为准。`<tokenizer_output_dir>` 与 `<data_config_path>` 需替换为实际路径。

---

## 3. 预训练

### 3.1 训练命令（从 step 24000 恢复）

```bash
# 预训练（从 d24 step 24000 恢复训练至 step 28000）
# total_batch_size=1,048,576 (1M tokens), target_param_data_ratio=50.0
# 数据: 171 parquet shards (ClimbMix + 中文数据混合)
python train.py \
    --config <pretrain_config_path> \
    --resume_from <checkpoint_step_24000_path> \
    --total_batch_size 1048576 \
    --target_param_data_ratio 50.0 \
    --data_shards <shards_dir> \
    --output_dir <pretrain_output_dir>
```

### 3.2 关键参数

| 参数 | 值 | 来源 |
| --- | --- | --- |
| `--total_batch_size` | 1,048,576（1M tokens） | [已验证] |
| `--target_param_data_ratio` | 50.0 | [已验证] |
| 数据 shards | 171 parquet shards | [已验证] |
| 恢复点 | step 24000 | [已验证] |
| 最终 checkpoint | `d24_final_mixdata` step 28000 | [已验证] |
| val_bpb @ step 28000 | 0.7626 | [已验证] |
| smooth_train_loss | 2.5539 | [已验证] |
| total_training_time | ~2,428,705 秒（~28 天） | [已验证] |

### 3.3 不可完整复现的部分

| 不可复现项 | 标记 | 说明 |
| --- | --- | --- |
| 原始训练日志 | `historical artifact unavailable` | 不可重新获取 |
| 17.68B 总 Token 池计数 | `historical_self_reported` | 原始日志不可重新获取 |
| 各组成部分 Token 数 | `historical_self_reported` | 原始日志不可重新获取 |
| 预训练数据再分发 | `待确认` | 数据源许可待确认 |

> **重要声明**：预训练的完整复现受限于：①原始训练日志不可重新获取；②预训练数据源许可待确认。因此预训练结果**无法完整独立复现**，相关数字标记为 `historical artifact unavailable`。详见 `pretraining-data-card.md`。

---

## 4. SFT 训练

### 4.1 训练命令

```bash
# SFT 训练（d24_finance_v2_lr010）
# 基于 d24_final_mixdata step 28000 预训练 checkpoint
python train_sft.py \
    --config <sft_config_path> \
    --pretrained_checkpoint <d24_final_mixdata_step_28000_path> \
    --tokenizer <tokenizer_path> \
    --max_seq_len 2048 \
    --device_batch_size 4 \
    --total_batch_size 1048576 \
    --embedding_lr 0.3 \
    --unembedding_lr 0.008 \
    --matrix_lr 0.02 \
    --init_lr_frac 0.1 \
    --warmup_ratio 0.05 \
    --warmdown_ratio 0.5 \
    --finance_epochs 1 \
    --finance_cot_epochs 0 \
    --smoltalk_size 30000 \
    --output_dir <sft_output_dir>
```

### 4.2 关键参数

| 参数 | 值 | 来源 |
| --- | --- | --- |
| `--max_seq_len` | 2048 | [已验证] |
| `--device_batch_size` | 4 | [已验证] |
| `--total_batch_size` | 1,048,576 | [已验证] |
| `--embedding_lr` | 0.3 | [已验证] |
| `--unembedding_lr` | 0.008 | [已验证] |
| `--matrix_lr` | 0.02 | [已验证] |
| `--init_lr_frac` | 0.1 | [已验证] |
| `--warmup_ratio` | 0.05 | [已验证] |
| `--warmdown_ratio` | 0.5 | [已验证] |
| `--finance_epochs` | 1 | [已验证] |
| `--finance_cot_epochs` | 0 | [已验证] |
| `--smoltalk_size` | 30000 | [已验证] |
| Loss mask | 仅 assistant 部分（user labels=-100） | [已验证] |
| SFT 数据总量 | 39,534 samples | [已验证] |

### 4.3 SFT Run 说明

| Run | 步数范围 | best checkpoint | 来源 |
| --- | --- | --- | --- |
| `d24_finance_v2_lr005` | 0–150 | — | [已验证] |
| `d24_finance_v2_lr010` | 125–375 | step 150（val_bpb=0.5558） | [已验证] |

> 具体脚本路径与参数名称以仓库内实际实现为准。

---

## 5. 推理

### 5.1 加载 Checkpoint

```bash
# 加载 SFT checkpoint 进行推理
# checkpoint 为 nanochat 原生格式（fp32）
python inference.py \
    --checkpoint <sft_checkpoint_path> \
    --tokenizer <tokenizer_path> \
    --prompt "<|bos|><|user_start|>{user_input}<|user_end|><|assistant_start|>"
```

### 5.2 加载 Tokenizer

```bash
# Tokenizer 为 RustBPE Byte-Level BPE, vocab_size=65000
# 包含 9 个 Special Tokens
# <think>/<think> 未注册为 Special Token，保留为普通文本序列
```

> Tokenizer 加载逻辑由 nanochat 仓库内的实现提供，具体 API 以仓库代码为准。

### 5.3 CLI 推理

```bash
# CLI 交互式推理
python cli_chat.py \
    --checkpoint <sft_checkpoint_path> \
    --tokenizer <tokenizer_path> \
    --max_seq_len 2048
```

### 5.4 HTTP 服务（OpenAI 兼容）

```bash
# 启动 OpenAI 兼容的 HTTP 服务
# 服务地址: http://127.0.0.1:8500/v1
python serve.py \
    --checkpoint <sft_checkpoint_path> \
    --tokenizer <tokenizer_path> \
    --host 127.0.0.1 \
    --port 8500 \
    --max_seq_len 2048
```

| 字段 | 值 | 来源 |
| --- | --- | --- |
| API 协议 | OpenAI 兼容 | [已验证] |
| 服务地址 | http://127.0.0.1:8500/v1 | [已验证] |
| max_seq_len | 2048 | [已验证] |

> 具体脚本路径与服务启动方式以仓库内实际实现为准。

---

## 6. RAG 部署

### 6.1 建立 ChromaDB（Dense 索引）

```bash
# 建立 ChromaDB Dense 索引
python build_chroma.py \
    --data_dir <documents_dir> \
    --collection <collection_name> \
    --persist_dir <chroma_persist_dir>
```

### 6.2 建立 BM25（Sparse 索引）

```bash
# 建立 BM25 Sparse 索引（jieba-fast 中文分词）
python build_bm25.py \
    --data_dir <documents_dir> \
    --index_path <bm25_index_path>
```

### 6.3 启动 RAG 后端

```bash
# 启动 FastAPI RAG 后端
# 模型服务: http://127.0.0.1:8500/v1 (OpenAI 兼容)
python run_rag_backend.py \
    --chroma_persist_dir <chroma_persist_dir> \
    --bm25_index_path <bm25_index_path> \
    --model_endpoint http://127.0.0.1:8500/v1 \
    --host 0.0.0.0 \
    --port <rag_backend_port>
```

### 6.4 运行 Query

```bash
# 同步查询
curl -X POST http://<rag_backend_host>:<rag_backend_port>/query \
    -H "Content-Type: application/json" \
    -d '{"question": "<你的金融问题>"}'

# SSE 流式查询
curl -X POST http://<rag_backend_host>:<rag_backend_port>/query/stream \
    -H "Content-Type: application/json" \
    -d '{"question": "<你的金融问题>"}'
```

### 6.5 RAG 系统组件

| 组件 | 实现 | 来源 |
| --- | --- | --- |
| 后端框架 | FastAPI | [已验证] |
| Dense 检索 | ChromaDB | [已验证] |
| Sparse 检索 | BM25（jieba-fast 中文分词） | [已验证] |
| 融合 | RRF（Reciprocal Rank Fusion） | [已验证] |
| 重排序 | Reranker | [已验证] |
| 确定性计算 | Calculator（9 种操作） | [已验证] |
| 校验 | Answerability + Grounding + Validation（fail-closed） | [已验证] |
| 安全降级 | Safe Fallback | [已验证] |
| SSE 安全 | 不泄露 blocked tokens | [已验证] |
| 前端 | Vite + React | [已验证] |

### 6.6 RAG HTTP 端点

| 端点 | 方法 | 功能 | 来源 |
| --- | --- | --- | --- |
| `/query` | POST | 同步查询 | [已验证] |
| `/query/stream` | POST | SSE 流式查询 | [已验证] |

> 具体脚本路径、端点路径与参数名称以仓库内实际实现为准。

---

## 7. 无法完整复现的部分

以下部分标记为 `historical artifact unavailable`，**无法完整独立复现**：

| 不可复现项 | 标记 | 说明 | 相关文档 |
| --- | --- | --- | --- |
| 原始预训练日志 | `historical artifact unavailable` | 不可重新获取 | `pretraining-data-card.md` |
| 17.68B 总 Token 池计数 | `historical_self_reported` | 原始日志不可重新获取 | `pretraining-data-card.md` |
| 各组成部分 Token 数 | `historical_self_reported` | 原始日志不可重新获取 | `pretraining-data-card.md` |
| SFT800 / SFT1147 checkpoint | `不可验证` | 不在当前服务器 | `model-card.md` |
| 预训练数据再分发 | `待确认` | 数据源许可待确认 | `pretraining-data-card.md` |
| Tokenizer 压缩率数字 | `待验证` | 需脚本重现后填入 | `tokenizer-card.md` |

> **重要声明**：预训练阶段因原始日志不可重新获取、数据源许可待确认，**无法完整独立复现**。SFT800/SFT1147 历史指标因 checkpoint 不在当前服务器，**不可验证**。Tokenizer 压缩率数字**待脚本重现后填入**。

---

## 8. 相关文档

- `model-card.md` — 模型卡片
- `tokenizer-card.md` — Tokenizer 卡片
- `pretraining-data-card.md` — 预训练数据卡片
- `sft-data-card.md` — SFT 数据卡片
- `rag-system-card.md` — RAG 系统卡片
