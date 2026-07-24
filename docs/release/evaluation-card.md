# Evaluation Card — NanoFinance Phase 5 评测

## 概述

本文档描述 NanoFinance 项目 Phase 5 评测的目的、方法论、隔离机制、Calibration 流程、RC Freeze、数据分类，以及哪些指标可以引用、哪些指标禁止引用。**核心声明**：Phase 5 评测的 54 个 sealed cases 数据分类为 `synthetic_held_out`，**不是真正独立的 Sealed Evaluation**。0/54 strict pass 结果**仅用于基础设施功能测试**，**不作为质量估计、模型比较或简历指标引用**。本卡片严格区分"可引用的评测事实"与"禁止引用的指标"。

---

## 1. Phase 5 评测目的

| 字段 | 说明 | 来源 |
| --- | --- | --- |
| 评测目的 | 基础设施功能测试（Blind/Scoring 隔离、RC Freeze、Calibration、A0–A9 消融等机制的功能性验证） | [已验证] |
| 数据分类 | `synthetic_held_out` | [已验证] |
| 是否为真正独立 Sealed Evaluation | **否** | [已验证] |

> **重要声明**：Phase 5 的 54 个 sealed cases 数据分类为 `synthetic_held_out`，**不是真正独立的 Sealed Evaluation**。评测目的为基础设施功能测试，**不是**模型质量评估。

---

## 2. 评测模型身份

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 被评测模型身份 | `finquery-finance-v2-lr010-150` | [已验证] |
| 对应 checkpoint | `d24_finance_v2_lr010` step 150 | [已验证] |
| 模型性质 | SFT V2 lr010 失败实验 checkpoint（val_bpb=0.5558, finance macro=0.2297） | [已验证] |

> **重要声明**：被评测的 `finquery-finance-v2-lr010-150` 是 SFT V2 lr010 step 150 checkpoint，属于失败实验（finance macro=0.2297）。该 checkpoint **不是**生产基线。生产基线 SFT1147 的 checkpoint 不在当前服务器，标记为 **[不可验证]**。

---

## 3. 样本数

| 划分 | 样本数 | 来源 |
| --- | --- | --- |
| Dev | 48 | [已验证] |
| Cal | 48 | [已验证] |
| Sealed | 54 | [已验证] |

---

## 4. 隔离机制

### 4.1 EvaluationQuery / Label 隔离

| 字段 | 说明 | 来源 |
| --- | --- | --- |
| 隔离机制 | EvaluationQuery 与 Label 分离 | [已验证] |
| 目的 | 防止评测过程中标签泄露 | [已验证] |

### 4.2 Blind / Scoring 隔离

| 字段 | 说明 | 来源 |
| --- | --- | --- |
| Blind 阶段 | 盲评阶段，评测方不可见标准答案 | [已验证] |
| Scoring 阶段 | 评分阶段，独立于 Blind | [已验证] |
| 隔离目的 | 防止评分方在盲评阶段获取答案 | [已验证] |

### 4.3 Raw / Canonical Hash

| 字段 | 说明 | 来源 |
| --- | --- | --- |
| Raw Hash | 原始数据 Hash | [已验证] |
| Canonical Hash | 规范化数据 Hash | [已验证] |
| 目的 | 数据完整性校验，防止数据被篡改 | [已验证] |

---

## 5. Calibration（校准）

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 校准阶段 | 两阶段 | [已验证] |
| 候选数 | 11664 | [已验证] |

> Calibration 通过两阶段从 11664 候选中筛选评测配置，确保评测阈值与判定规则的合理性。

---

## 6. A0–A9 消融

| 字段 | 说明 | 来源 |
| --- | --- | --- |
| 消融实验 | A0–A9 共 10 组消融 | [已验证] |
| 目的 | 验证评测系统各组件的贡献与稳定性 | [已验证] |

---

## 7. RC Freeze（资源冻结）

| 字段 | 值 | 来源 |
| --- | --- | --- |
| RC Freeze | 已执行 | [已验证] |
| 资源 Hash 验证项数 | 8 项 | [已验证] |

> RC Freeze 对 8 项资源进行 Hash 验证，确保评测过程中使用的资源（模型、数据、配置等）被冻结，不可篡改。

---

## 8. synthetic-held-out 数据分类

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 数据分类 | `synthetic_held_out` | [已验证] |
| 是否真正独立 Sealed Evaluation | **否** | [已验证] |
| 含义 | 数据为合成 held-out，**不是**真正从未见过的独立评测集 | [已验证] |

> **重要声明**：`synthetic_held_out` 意味着评测数据并非真正独立的 Sealed Evaluation。该分类下的结果**仅用于基础设施功能测试**。

---

## 9. Sealed Evaluation 结果

| 字段 | 值 | 来源 |
| --- | --- | --- |
| Sealed cases 数 | 54 | [已验证] |
| strict pass 数 | 0 | [已验证] |
| strict pass 比例 | 0/54 | [已验证] |

---

## 10. 哪些指标可以引用

以下指标为**基础设施功能测试**结果，**可以引用**：

| 可引用指标 | 结果 | 来源 |
| --- | --- | --- |
| Blind/Scoring 隔离机制 | 功能正常 | [已验证] |
| EvaluationQuery/Label 隔离 | 功能正常 | [已验证] |
| RC Freeze 8 项资源 Hash 验证 | 全部通过 | [已验证] |
| Calibration 两阶段（11664 候选） | 功能正常 | [已验证] |
| A0–A9 消融 | 已执行 | [已验证] |
| Raw/Canonical Hash | 功能正常 | [已验证] |
| 评测模型身份 | `finquery-finance-v2-lr010-150`（step 150） | [已验证] |
| 样本数 | Dev 48 / Cal 48 / Sealed 54 | [已验证] |
| 数据分类 | `synthetic_held_out` | [已验证] |

> 引用上述指标时，应明确说明为**基础设施功能测试**结果。

---

## 11. 哪些指标禁止引用

以下指标**禁止作为质量估计、模型比较或简历指标引用**：

| 禁止引用指标 | 值 | 禁止原因 |
| --- | --- | --- |
| Sealed Evaluation strict pass | 0/54 | 数据分类为 `synthetic_held_out`，**不是真正独立 Sealed Evaluation** |

### 禁止引用的用途

0/54 strict pass 结果**不可用于**：

| 禁止用途 | 说明 |
| --- | --- |
| 模型质量估计 | 不可作为模型质量的定量估计 |
| 模型间比较 | 不可用于与其他模型的横向比较 |
| 简历或对外宣传指标 | 不可写入简历、对外宣传材料作为性能指标 |
| 质量基线 | 不可作为发布版本的质量基线 |

> **核心声明**：0/54 strict pass 是基础设施功能测试的结果，**不代表**模型质量。该结果仅证明评测基础设施能够正确执行 strict 判定流程，**不证明**模型在金融问答任务上的实际能力。

---

## 12. 不可验证的历史评测结果

以下历史评测结果因 checkpoint 不在当前服务器、原始日志不可重新获取，标记为 **[不可验证]**，**不作为本发布的质量基线引用**：

| 历史 Run | val_bpb | finance macro | 状态 | 来源 |
| --- | --- | --- | --- | --- |
| SFT800 | 0.4783 | 0.3736 | checkpoint 不在当前服务器 | [不可验证] |
| SFT1147 | 0.4842 | 0.4432 | 生产基线声明，checkpoint 不在当前服务器 | [不可验证] |

> SFT800 与 SFT1147 的指标**不可验证**，**不作为本发布的质量基线引用**。

---

## 13. 相关文档

- `model-card.md` — 模型卡片（SFT 历史结果）
- `limitations-and-risks.md` — 局限性与风险（synthetic-held-out 0/54 不代表质量）
- `responsible-use.md` — 负责任使用（评测局限的对外表述）
