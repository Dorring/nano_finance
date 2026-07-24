# SFT Data Card — NanoFinance 监督微调数据

## 概述

本文档描述 NanoFinance 项目监督微调（SFT）阶段使用的数据组成、对话模板、Loss Mask 策略、序列长度与划分规则。SFT 数据总量为 39,534 samples **[已验证]**，由金融多任务（finqa、tatqa、ectsum、finer、finred、fiqa、finsen）、Finance R1 风格显式推理数据（finance_r1）、以及通用能力正则数据（SmolTalk）混合而成。**重要表述声明**：finance_r1 数据的正确表述为 **"引入 R1 风格的显式推理格式对齐数据"**，**不是** "改造成 R1 架构"——模型架构未改变，仅引入了 R1 风格的推理格式训练样本。

---

## 1. 总样本数

| 字段 | 值 | 来源 |
| --- | --- | --- |
| SFT 数据总量 | 39,534 samples | [已验证] |

---

## 2. 数据组成（按类别）

| 类别 | 占比 | 说明 | 来源 |
| --- | --- | --- | --- |
| 金融多任务 | 91.4% | finqa, tatqa, ectsum, finer, finred, fiqa, finsen | [已验证] |
| Finance R1 风格显式推理 | 3.1% | 1225 条，**引入 R1 风格的显式推理格式对齐数据**（非 R1 架构改造） | [已验证] |
| 通用能力正则（SmolTalk） | 5.5% | 30000 条（smoltalk_size=30000） | [已验证] |

> **重要表述声明**：finance_r1 的正确表述为 **"引入 R1 风格的显式推理格式对齐数据"**。这表示训练数据中引入了 R1 风格的显式推理格式样本，用于格式对齐。**不是** "改造成 R1 架构"——模型架构（GPT-2 style Transformer）未发生任何改变。

---

## 3. 每个数据源详细统计

| 数据源 | 样本数 | 类别 | 来源 |
| --- | --- | --- | --- |
| finqa | 8,144 | 金融多任务（问答） | [已验证] |
| tatqa | 16,543 | 金融多任务（表格问答） | [已验证] |
| ectsum | 2,425 | 金融多任务（摘要） | [已验证] |
| finer | 3,034 | 金融多任务（关系抽取） | [已验证] |
| finred | 4,359 | 金融多任务（关系抽取） | [已验证] |
| fiqa | 822 | 金融多任务（问答/意见） | [已验证] |
| finsen | 2,982 | 金融多任务（情感分析） | [已验证] |
| finance_r1 | 1,225 | R1 风格显式推理格式对齐 | [已验证] |
| smoltalk | 30,000 | 通用能力正则 | [已验证] |

> 注：smoltalk_size=30000 为配置参数，实际参与训练的样本数以 manifest 为准。

---

## 4. Train/Validation/Test 划分

| 划分 | 样本数 | 来源 |
| --- | --- | --- |
| train | 30,641 | [已验证] |
| cot_train | 979 | [已验证] |
| effective_train | 31,620（train + cot_train） | [已验证] |
| val | 3,958 | [已验证] |
| test | 3,956 | [已验证] |
| 总样本数 | 39,534 | [已验证] |
| 划分比例 | 约 80 / 10 / 10（实际 80.0% / 10.0% / 10.0%，允许合理舍入误差） | [已验证] |
| random seed | 42 | [已验证] |

---

## 5. Conversation Template（对话模板）

SFT 数据使用以下对话模板（基于 9 个 Special Tokens）：

```
<|bos|><|user_start|>{user_content}<|user_end|><|assistant_start|>{assistant_content}<|assistant_end|>
```

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 起始 token | `<|bos|>` | [已验证] |
| 用户段 | `<|user_start|> ... <|user_end|>` | [已验证] |
| 助手段 | `<|assistant_start|> ... <|assistant_end|>` | [已验证] |
| 多轮对话 | user/assistant 交替 | [已验证] |
| 模板示例 | `<|bos|> <|assistant_start|> ...`（user/assistant 交替） | [已验证] |

---

## 6. Assistant-only Loss Mask

| 字段 | 值 | 来源 |
| --- | --- | --- |
| Loss mask 策略 | 仅对 assistant 部分计算 loss | [已验证] |
| user 部分 labels | -100（不参与 loss 计算） | [已验证] |
| 训练目标 | 模型仅学习生成 assistant 回复 | [已验证] |

> 说明：user 部分的 token labels 被设为 -100，在 loss 计算中被忽略。模型仅在 assistant 部分进行 next-token prediction 训练。

---

## 7. 序列长度与截断策略

| 字段 | 值 | 来源 |
| --- | --- | --- |
| max_seq_len | 2048 | [已验证] |
| 截断策略 | 超出 2048 的样本按既定规则截断 | [已验证]（参数）；[待验证]（具体截断规则需引用脚本） |

> 具体截断规则（左侧/右侧截断、是否保留 Special Tokens 等）需引用 SFT 数据处理脚本源码确认。

---

## 8. 训练超参数（SFT）

| 字段 | 值 | 来源 |
| --- | --- | --- |
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

---

## 9. SFT Run 说明

| Run | 步数范围 | best checkpoint | 来源 |
| --- | --- | --- | --- |
| `d24_finance_v2_lr005` | 0–150 | — | [历史自报] |
| `d24_finance_v2_lr010` | 125–375 | step 150（val_bpb=0.5558，烟雾测试 checkpoint，非发布模型） | [历史自报] |

> 详细的 SFT 历史结果与不可验证声明见 `model-card.md` 第 4.3 节。

---

## 10. 数据许可

| 字段 | 值 | 来源 |
| --- | --- | --- |
| finqa 许可 | 待确认 | [待确认] |
| tatqa 许可 | 待确认 | [待确认] |
| ectsum 许可 | 待确认 | [待确认] |
| finer 许可 | 待确认 | [待确认] |
| finred 许可 | 待确认 | [待确认] |
| fiqa 许可 | 待确认 | [待确认] |
| finsen 许可 | 待确认 | [待确认] |
| finance_r1 许可 | 待确认 | [待确认] |
| smoltalk 许可 | 待确认 | [待确认] |

> 各 SFT 数据源的许可条款待确认，在确认前不声明可自由再分发。

---

## 11. 敏感信息处理

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 敏感信息审计 | 未明确验证 | [待验证] |
| PII 处理 | 未经过独立 PII 审计 | [待验证] |

> SFT 数据是否包含敏感个人信息未经过独立审计，引用时必须标注 **[待验证]**。

---

## 12. 人工/合成比例

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 人工标注比例 | 待确认 | [待确认] |
| 合成数据比例 | 待确认（finance_r1 含合成推理样本，smoltalk 含合成对话） | [待确认] |

> 各数据源的人工/合成比例需引用数据集原始说明，本卡片不硬编码比例。

---

## 13. 重要表述声明

为避免歧义，本卡片对以下表述作出明确声明：

| 表述 | 是否正确 | 说明 |
| --- | --- | --- |
| "引入 R1 风格的显式推理格式对齐数据" | ✅ 正确 | finance_r1 数据引入了 R1 风格的显式推理格式样本，用于格式对齐 |
| "改造成 R1 架构" | ❌ 错误 | 模型架构未改变，仍为 GPT-2 style Transformer |
| "引入 R1 风格推理" | ✅ 正确（需注意是"格式对齐"） | 强调是格式对齐，非架构改造 |

---

## 14. 相关文档

- `model-card.md` — 模型卡片（SFT checkpoint 信息）
- `evaluation-card.md` — 评测卡片（SFT 模型评测）
- `reproducibility.md` — 复现指南（SFT 训练命令）
