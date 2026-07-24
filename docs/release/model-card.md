# Model Card — NanoFinance 金融领域大模型

## 概述

本文档为 NanoFinance 项目的模型卡片（Model Card），描述基于 karpathy/nanochat（MIT License）二次开发的中文金融领域大模型。Release ID 为 `nano-finance-d24-sft-v1`，Release Type 为 `documentation_and_evidence`（文档与证据框架）。本发布**不包含可下载、可验证的正式模型权重版本**；`release_model_checkpoint` 为 `null`。历史生产 checkpoint（`sft1147`）当前不可访问，标记为 `unavailable_unverified`。step 150 checkpoint 为失败实验的烟雾测试 checkpoint（`evaluation_smoke_checkpoint`），不作为正式发布模型。

模型经过预训练（Pretraining）与监督微调（SFT）两个阶段，并配套 RAG 检索增强系统用于金融报告问答。本卡片仅描述模型本身的能力边界、训练依据与已验证的评测事实，不包含任何未被验证或不可复现的指标声明。所有数字均标注来源类别：**[已验证]**（当前服务器可查的 manifest/checkpoint/log）、**[历史自报]**（项目历史记录中存在但原始日志不可重新获取）、**[不可验证]**（无原始数据支撑，仅作背景说明）。

> **Checkpoint 哈希口径说明**：本发布中 Checkpoint 的 `identity_digest` 是基于 `run_name/step` 字符串计算的 SHA256，仅用于稳定标识，**不是模型文件内容哈希**。`checkpoint_content_sha256` 始终为 `null`，因为模型权重文件过大且当前服务器不可访问。涉及 Checkpoint 内容的声明标记为 `historical_unavailable`，不作为已验证证据。

---

## 1. Model Details（模型详情）

| 字段 | 值 | 来源 |
| --- | --- | --- |
| Release ID | `nano-finance-d24-sft-v1` | [已验证] |
| 基础项目 | karpathy/nanochat | [已验证] |
| 基础项目许可证 | MIT (Copyright 2025 Andrej Karpathy) | [已验证] |
| 模型架构 | GPT-2 style Transformer (nanochat) | [已验证] |
| n_layer | 24 | [已验证] |
| n_head | 12 | [已验证] |
| n_kv_head | 12 | [已验证] |
| n_embd | 1536 | [已验证] |
| head_dim | 128 | [已验证] |
| aspect_ratio | 64 | [已验证] |
| depth | 24 | [已验证] |
| vocab_size | 65000 | [已验证] |
| sequence_len / max_seq_len | 2048 | [已验证] |
| window_pattern | "L"（full context，非滑动窗口） | [已验证] |
| 是否使用 bias | 否 | [已验证] |
| 优化器 | Muon | [已验证] |
| 参数量（约） | 1.4B | [已验证]（checkpoint 文件 5.7GB，fp32） |
| Tokenizer | Byte-Level BPE (RustBPE, GPT-4 style) | [已验证] |
| 支持语言 | 中文（主要）、英文 | [已验证] |
| 训练阶段 | 预训练 → SFT | [已验证] |
| 模型格式 | nanochat 原生 checkpoint（fp32） | [已验证] |
| 模型权重许可证 | 未明确授权 | [已验证] |
| 预训练数据许可证 | ClimbMix (NVIDIA)、SkyPile、中文金融数据（来源待确认） | [待确认] |

### 训练阶段说明

- **预训练 Checkpoint**：`d24_final_mixdata`，step 28000 — `verification_status: historical_unavailable`（checkpoint 内容哈希不可验证）
- **SFT 烟雾测试 Checkpoint**：`d24_finance_v2_lr010` step 150 — 失败实验，不作为发布模型（`evaluation_smoke_checkpoint`）
- **历史生产 Checkpoint**：`sft1147` — `unavailable_unverified`（checkpoint 不在当前服务器，未验证）
- **正式发布模型 Checkpoint**：`null`（本发布为文档与证据框架，无可验证的模型权重）

详细训练与数据信息请引用对应 manifest 路径，参见 `pretraining-data-card.md`、`sft-data-card.md`、`reproducibility.md`。

---

## 2. Intended Use（预期用途）

本模型预期用于以下**研究与原型开发**场景，且应在 RAG 系统或人工审核流程中使用：

| 用途 | 说明 |
| --- | --- |
| 中文金融报告问答 | 在 RAG 系统提供的证据上下文中，回答基于中文金融报告（如年报、季报、研究报告）的问题 |
| 金融文本摘要 | 对金融报告段落、会议纪要进行结构化摘要（如 ECTSum 风格） |
| 信息抽取 | 从金融文本中抽取关系（如 FinRed）、情感（如 FinSen）、实体等结构化信息 |
| RAG 有证据回答 | 在 Answerability + Grounding 校验通过的前提下，基于检索证据生成带来源的回答 |
| 确定性计算辅助 | 配合 RAG 系统中的确定性 Calculator（9 种操作）完成金融数值计算，由 Calculator 而非模型本身保证数值正确性 |

> 注：模型的"计算能力"指 RAG 系统中的确定性 Calculator 模块，**不是**模型本身的数学推理能力。详见 `rag-system-card.md`。

---

## 3. Out-of-scope（不适用场景）

以下用途**明确不在**本模型的设计与验证范围内，禁止或强烈不建议使用：

| 不适用场景 | 说明 |
| --- | --- |
| 投资建议 | 模型不提供任何投资、买卖、持仓建议 |
| 证券买卖决策 | 不可作为证券交易决策依据 |
| 自动交易 | 不可接入自动交易系统执行真实交易 |
| 税务法律意见 | 不提供税务、法律专业意见 |
| 实时市场预测 | 模型无实时数据接入，不具备市场预测能力 |
| 无来源计算 | 在没有 RAG 证据或 Calculator 的情况下，模型自行进行的数值计算不可信 |
| 医疗信贷决策 | 不可用于医疗、信贷、保险等高风险个人决策 |
| 原生 Function Calling | 模型**未训练**原生 Function Calling / Tool Calling 能力，不声称支持 |
| 任意 Python 执行 | 模型不执行 Python 代码；`<python_start|>`/`<python_end|>` 为预定义 Special Token，但当前发布版本不包含可用的代码执行回路 |

---

## 4. Training（训练）

本节引用已验证的 manifest 与 checkpoint，**不硬编码训练数字**。具体数值以 manifest 文件为准。

### 4.1 预训练

| 字段 | 值 | 来源 |
| --- | --- | --- |
| Checkpoint | `d24_final_mixdata`（checkpoint 内容哈希未验证，`historical_unavailable`） | [历史自报] |
| 训练步数 | 28000（从 step 24000 恢复训练） | [历史自报] |
| val_bpb @ step 28000 | 0.7626 | [历史自报] |
| smooth_train_loss | 2.5539 | [历史自报] |
| total_training_time | ~2,428,705 秒（~28 天） | [历史自报] |
| 数据 | 171 parquet shards（ClimbMix + 中文数据混合） | [历史自报]（原始目录不可访问，未重新枚举验证） |
| total_batch_size | 1,048,576（1M tokens） | [已验证] |
| target_param_data_ratio | 50.0 | [已验证] |
| 硬件 | 双 A6000 GPU（48GB each） | [已验证] |

> 训练数据组成详见 `pretraining-data-card.md`。预训练命令详见 `reproducibility.md`。

### 4.2 SFT

| 字段 | 值 | 来源 |
| --- | --- | --- |
| SFT Run | `d24_finance_v2_lr005`、`d24_finance_v2_lr010` | [已验证] |
| lr005 步数范围 | 0–150 | [已验证] |
| lr010 步数范围 | 125–375 | [已验证] |
| lr010 best @ step 150 | val_bpb = 0.5558 | [历史自报]（烟雾测试 checkpoint，不作为发布模型） |
| SFT 数据总量 | 39,534 samples | [已验证] |
| max_seq_len | 2048 | [已验证] |
| device_batch_size | 4 | [已验证] |
| total_batch_size | 1,048,576 | [已验证] |
| embedding_lr | 0.3 | [已验证] |
| unembedding_lr | 0.008 | [已验证] |
| matrix_lr | 0.02 | [已验证] |
| init_lr_frac | 0.1 | [已验证] |
| warmup_ratio | 0.05 | [已验证] |
| warmdown_ratio | 0.5 | [已验证] |
| finance_epochs | 1 | [已验证] |
| finance_cot_epochs | 0 | [已验证] |
| smoltalk_size | 30000 | [已验证] |
| 对话模板 | `<|bos|> <|assistant_start|> ...`（user/assistant 交替） | [已验证] |
| Loss mask | 仅对 assistant 部分计算 loss（user 部分 labels = -100） | [已验证] |

> SFT 数据组成详见 `sft-data-card.md`。

### 4.3 SFT V2 历史结果（部分不可验证）

| Run / Checkpoint | val_bpb | finance macro | 状态 | 来源 |
| --- | --- | --- | --- | --- |
| SFT800 | 0.4783 | 0.3736 | 历史，checkpoint 不在当前服务器 | [不可验证] |
| SFT1147 | 0.4842 | 0.4432 | 生产基线声明，checkpoint 不在当前服务器 | [不可验证] |
| V2 lr010 step 150 | 0.5558 | 0.2297 | 失败实验（烟雾测试 checkpoint，非发布模型）；checkpoint 文件存在但内容哈希未验证 | [历史自报] |
| V2 lr010 step 275 | 0.5527 | 0.2077 | 失败实验；checkpoint 文件存在但内容哈希未验证 | [历史自报] |

> SFT800 与 SFT1147 的指标因 checkpoint 不在当前服务器、原始日志不可重新获取，标记为 **[不可验证]**，**不作为本发布的质量基线引用**。V2 lr010 系列的 checkpoint 文件虽存在，但 `checkpoint_content_sha256` 为 `null`（内容哈希未验证），相关指标来自历史训练日志，标记为 **[历史自报]**。

---

## 5. Evaluation（评测）

### 5.1 可引用的评测事实

| 评测项 | 结果 | 来源 |
| --- | --- | --- |
| Phase 5 基础设施功能测试 | 通过（Blind/Scoring 隔离、RC Freeze 8 项资源 Hash 验证、EvaluationQuery/Label 隔离） | [已验证] |
| 被评测模型身份 | `finquery-finance-v2-lr010-150`（step 150） | [已验证] |
| 样本数 | Dev 48 / Cal 48 / Sealed 54 | [已验证] |
| Calibration | 两阶段，11664 候选 | [已验证] |
| A0–A9 消融 | 已执行 | [已验证] |
| 数据分类 | `synthetic_held_out` | [已验证] |

### 5.2 禁止引用的指标

| 评测项 | 结果 | 处理方式 |
| --- | --- | --- |
| Sealed Evaluation strict pass | 0/54 | **不作为质量估计、模型比较或简历指标引用** |

> **重要声明**：Phase 5 的 0/54 strict pass 结果基于 `synthetic_held_out` 数据分类，**不是真正独立的 Sealed Evaluation**。该结果**仅用于基础设施功能测试**，**不可用于**：
> - 模型质量估计
> - 模型间比较
> - 简历或对外宣传指标
>
> 详见 `evaluation-card.md`。

---

## 6. Limitations（局限性）

| 局限性类别 | 说明 |
| --- | --- |
| 模型规模 | 约 1.4B 参数，小于主流金融大模型（7B+），复杂推理能力有限 |
| 金融知识时效性 | 训练数据有截止日期，不包含最新市场动态、法规变更 |
| 训练语料偏差 | 中文金融语料占比有限（详见 `pretraining-data-card.md`），可能存在领域与来源偏差 |
| 中英文能力差异 | 中文为主要目标语言，英文能力依赖 ClimbMix，金融专业英文表现可能弱于中文 |
| 跨表计算限制 | 复杂跨表、多步数值计算由 RAG Calculator 处理，模型自身跨表计算能力有限 |
| 无原生 Tool Calling | 模型未训练原生 Function Calling，不声称支持 |
| RAG 依赖索引质量 | 检索质量受 ChromaDB/BM25 索引质量影响，索引缺失或错误会导致回答失败 |
| Validation 不能验证所有事实 | RAG 的 Answerability/Grounding/Validation 为 fail-closed 校验，但不能验证所有自然语言事实 |
| 不能保证消除幻觉 | 即使通过 Validation，模型仍可能产生幻觉，**不声称完全消除幻觉** |
| 无实时数据 | 模型与 RAG 系统均无实时市场数据接入 |
| 不承担金融决策责任 | 模型输出仅供参考研究，不承担任何金融决策后果 |

> 更完整的局限性讨论见 `limitations-and-risks.md`。

---

## 7. 引用与许可

- 基础代码：MIT License (Copyright 2025 Andrej Karpathy)，见 karpathy/nanochat 仓库
- RAG 后端依赖：chromadb、sentence-transformers、jieba-fast、openai、fastapi
- 前端依赖：Vite、React
- 模型权重：未明确授权
- 预训练数据：ClimbMix (NVIDIA)、SkyPile、中文金融数据（来源待确认）

## 8. 相关文档

- `tokenizer-card.md` — Tokenizer 卡片
- `pretraining-data-card.md` — 预训练数据卡片
- `sft-data-card.md` — SFT 数据卡片
- `rag-system-card.md` — RAG 系统卡片
- `evaluation-card.md` — 评测卡片
- `limitations-and-risks.md` — 局限性与风险
- `responsible-use.md` — 负责任使用
- `reproducibility.md` — 复现指南
