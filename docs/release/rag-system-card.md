# RAG System Card — NanoFinance 检索增强生成系统

## 概述

本文档描述 NanoFinance 项目配套的检索增强生成（RAG）系统。该系统由 FastAPI 后端、ChromaDB（Dense）+ BM25（Sparse）混合检索、RRF 融合、Reranker、确定性 Calculator、Answerability/Grounding/Validation 三层校验、以及 Safe Fallback 组成。**重要声明**：本卡片严格区分四种能力——**模型生成能力**、**检索能力**、**确定性计算能力**、**校验能力**，以及**系统级安全门禁**。Calculator 的数值正确性**不是**模型本身的数学推理能力；Validation 的阻断**不是**模型完全不会幻觉。RAG 系统是 fail-closed 设计，但不能保证消除所有幻觉。

---

## 1. 系统架构流程图

```
┌─────────────┐
│   Query     │  用户自然语言问题
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  Dense (ChromaDB) + BM25 (Sparse)│  双路检索
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────┐
│     RRF      │  Reciprocal Rank Fusion 融合
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Reranker    │  重排序
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Context Builder     │  上下文构建
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Calculator (确定性) │  9 种金融计算操作
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Answerability       │  可回答性校验
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Generation (模型)   │  基于证据生成回答
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Validation          │  Grounding + 事实校验（fail-closed）
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Safe Fallback       │  校验失败时安全降级
└─────────────────────┘
```

---

## 2. 能力边界区分

本系统包含**四种独立能力**与**一层系统级安全门禁**，必须严格区分：

| 能力类别 | 实现模块 | 性质 | 重要声明 |
| --- | --- | --- | --- |
| 模型生成能力 | SFT 模型（nano-finance-d24-sft-v1） | 神经网络，概率性 | 模型负责基于证据生成自然语言回答 |
| 检索能力 | ChromaDB (Dense) + BM25 (Sparse) + RRF + Reranker | 检索系统，确定性+排序 | 检索质量受索引质量影响 |
| 确定性计算能力 | Calculator 模块 | 确定性算法 | **Calculator 的数值正确性不是模型本身的数学推理能力** |
| 校验能力 | Answerability + Grounding + Validation | 规则/校验系统 | **Validation 阻断不是模型完全不会幻觉** |
| 系统级安全门禁 | Safe Fallback | fail-closed 设计 | 校验失败时安全降级，不返回未经验证的回答 |

### 2.1 Calculator 的正确性不是模型的数学推理能力

| 字段 | 说明 |
| --- | --- |
| Calculator 性质 | 确定性算法模块，非神经网络 |
| 数值正确性来源 | 算法实现，**不是**模型推理 |
| 模型角色 | 模型不直接执行数值计算，Calculator 的结果由确定性算法保证 |
| 重要声明 | **不可声称**模型本身具备 Calculator 的数值计算精度 |

### 2.2 Validation 阻断不是模型完全不会幻觉

| 字段 | 说明 |
| --- | --- |
| Validation 性质 | 规则/校验系统，fail-closed |
| 阻断行为 | 校验失败时阻断回答，触发 Safe Fallback |
| 重要声明 | Validation **不能验证所有自然语言事实**，即使通过 Validation，模型仍可能产生幻觉 |
| 结论 | **不可声称**完全消除幻觉 |

---

## 3. 检索系统

| 字段 | 值 | 来源 |
| --- | --- | --- |
| Dense 检索 | ChromaDB | [已验证] |
| Sparse 检索 | BM25 | [已验证] |
| 融合方式 | RRF（Reciprocal Rank Fusion） | [已验证] |
| 重排序 | Reranker | [已验证] |
| 中文分词 | jieba-fast | [已验证] |

### 检索流程

1. **Dense 检索**：ChromaDB 基于向量相似度检索候选文档
2. **Sparse 检索**：BM25 基于词频检索候选文档（jieba-fast 中文分词）
3. **RRF 融合**：Reciprocal Rank Fusion 融合 Dense 与 BM25 结果
4. **Reranker**：对融合后的候选文档进行重排序

---

## 4. 确定性 Calculator

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 性质 | 确定性算法模块 | [已验证] |
| 支持操作数 | 9 种 | [已验证] |

### 支持的 9 种金融计算操作

| 序号 | 操作 | 说明 | 来源 |
| --- | --- | --- | --- |
| 1 | difference | 差值计算 | [已验证] |
| 2 | growth_rate | 增长率 | [已验证] |
| 3 | percentage_share | 百分比占比 | [已验证] |
| 4 | sum | 求和 | [已验证] |
| 5 | average | 平均值 | [已验证] |
| 6 | gross_margin | 毛利率 | [已验证] |
| 7 | net_margin | 净利率 | [已验证] |
| 8 | debt_ratio | 负债比率 | [已验证] |
| 9 | scale_conversion | 量级转换 | [已验证] |

> **重要声明**：Calculator 的数值正确性由确定性算法保证，**不是**模型本身的数学推理能力。模型不直接执行这些计算。

---

## 5. 校验系统（fail-closed）

| 校验阶段 | 功能 | 来源 |
| --- | --- | --- |
| Answerability | 可回答性校验：判断问题是否可基于检索证据回答 | [已验证] |
| Grounding | 接地校验：判断生成回答是否基于检索证据 | [已验证] |
| Validation | 事实校验：对生成回答进行事实层面校验 | [已验证] |
| Safe Fallback | 安全降级：校验失败时不返回未经验证的回答 | [已验证] |
| SSE 安全 | SSE 流不泄露 blocked tokens | [已验证] |

### fail-closed 设计

- 校验失败时，系统**不返回**未经验证的回答，而是触发 Safe Fallback
- SSE 流式输出**不泄露**被阻断的 token，避免暴露内部校验状态
- 这是系统级安全门禁，**不是**模型本身的能力

> **重要声明**：Validation **不能验证所有自然语言事实**。即使通过 Validation，模型仍可能产生幻觉。**不可声称**完全消除幻觉。

---

## 6. 模型服务

| 字段 | 值 | 来源 |
| --- | --- | --- |
| API 协议 | OpenAI 兼容 | [已验证] |
| 服务地址 | http://127.0.0.1:8500/v1 | [已验证] |

---

## 7. HTTP 端点

| 端点 | 方法 | 功能 | 来源 |
| --- | --- | --- | --- |
| `/query` | POST | 同步查询：返回完整回答 | [已验证] |
| `/query/stream` | POST | SSE 流式查询：逐 token 返回 | [已验证] |

> SSE 流式输出不泄露 blocked tokens，保证校验状态不外泄。

---

## 8. 前端

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 技术栈 | Vite + React | [已验证] |

---

## 9. Phase 3-4 功能

RAG 系统在 Phase 3-4 阶段实现了以下功能：

| 功能 | 说明 | 来源 |
| --- | --- | --- |
| 混合检索 | Dense (ChromaDB) + BM25 + RRF + Reranker | [已验证] |
| 确定性 Calculator | 9 种金融计算操作 | [已验证] |
| 三层校验 | Answerability + Grounding + Validation | [已验证] |
| Safe Fallback | fail-closed 安全降级 | [已验证] |
| SSE 流式 | 不泄露 blocked tokens | [已验证] |
| OpenAI 兼容 API | http://127.0.0.1:8500/v1 | [已验证] |
| React 前端 | Vite + React | [已验证] |

---

## 10. RAG 依赖与局限

| 局限性 | 说明 |
| --- | --- |
| 依赖索引质量 | 检索质量受 ChromaDB/BM25 索引质量影响，索引缺失或错误会导致回答失败 |
| Validation 不能验证所有事实 | Validation 为规则/校验系统，不能验证所有自然语言事实 |
| 不能保证消除幻觉 | 即使通过 Validation，模型仍可能产生幻觉 |
| 无实时数据 | RAG 系统无实时市场数据接入，依赖预先建立的索引 |
| Calculator 非模型能力 | Calculator 数值正确性由算法保证，不是模型数学推理能力 |

---

## 11. 相关文档

- `model-card.md` — 模型卡片（RAG 使用的 SFT 模型）
- `limitations-and-risks.md` — 局限性与风险（RAG 相关局限）
- `reproducibility.md` — 复现指南（RAG 部署命令）
