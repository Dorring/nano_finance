# Tokenizer Card — NanoFinance Byte-Level BPE Tokenizer

## 概述

本文档描述 NanoFinance 项目为中文金融领域重新训练的 Byte-Level BPE Tokenizer。该 Tokenizer 基于 RustBPE（GPT-4 style），vocab_size 为 65000，旨在解决 nanochat 默认 32K vocab 对中文金融文本覆盖不足的问题。本卡片记录重新训练的动机、算法、训练数据、Special Tokens 设计、`<think>` 处理方式、Byte Fallback 机制、中英文与金融文本覆盖情况，以及压缩率评测现状。所有数字均标注来源类别。**不声称 Tokenizer 重新训练直接带来推理倍速提升**——Tokenizer 的影响主要体现在压缩率与多语言覆盖上，对推理速度的影响需独立 benchmark 验证。

---

## 1. 原 Tokenizer 问题

| 问题 | 说明 |
| --- | --- |
| nanochat 默认 vocab | 32K，主要为英文优化 |
| 中文覆盖不足 | 默认 32K vocab 对中文字符的 BPE 合并不充分，中文文本被切分为大量单字节或单字符 token |
| 金融术语缺失 | 默认 vocab 不包含金融领域高频术语（如财务报表科目、金融指标缩写） |
| 压缩率低 | 中文金融文本在默认 vocab 下压缩率低，单 token 携带信息量少 |

---

## 2. 为什么重新训练

| 动机 | 说明 |
| --- | --- |
| 中文金融文本需要更大 vocab | 中文金融报告、研报、新闻包含大量中文字符与金融术语，需要更大的 vocab 以提升压缩率与语义覆盖 |
| 多语言平衡 | 训练数据包含英文通用（ClimbMix）与中文通用（SkyPile/Wiki），需 Tokenizer 同时覆盖中英文 |
| 领域适配 | 金融领域术语（如"营业收入"、"净利润"、"资产负债率"）需要作为完整 token 或更优子词序列被编码 |

> **重要声明**：重新训练 Tokenizer 的目标是改善压缩率与多语言覆盖，**不声称** Tokenizer 重新训练直接带来推理速度的倍数提升。推理速度受模型架构、batch size、KV cache、硬件等多因素影响，需独立 benchmark 验证。

---

## 3. 训练算法

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 算法 | Byte-Level BPE | [已验证] |
| 实现 | RustBPE（GPT-4 style） | [已验证] |
| Split pattern | GPT-4 style，包含 `\p{N}{1,2}` | [已验证] |
| Vocab Size | 65000 | [已验证] |

### Byte-Level BPE 说明

- Byte-Level BPE 在字节层面进行 BPE 合并，天然支持任意 UTF-8 字符
- 通过 Byte-Level 实现 Byte Fallback：任何无法被 BPE 合并的字符最终可回退到字节表示，避免 `<unk>`

---

## 4. 训练数据组成

| 数据类别 | 占比 | 来源 |
| --- | --- | --- |
| 英文通用（ClimbMix） | 40% | [已验证] |
| 中文通用（SkyPile/Wiki） | 40% | [已验证] |
| 中文金融 | 20% | [已验证] |
| max_chars | 2B | [已验证] |
| doc_cap | 10000 | [已验证] |

> 数据来源与许可详见 `pretraining-data-card.md`。

---

## 5. Special Tokens

共 9 个 Special Tokens：

| 序号 | Special Token | 用途 | 来源 |
| --- | --- | --- | --- |
| 1 | `<|bos|>` | 序列起始 | [已验证] |
| 2 | `<|user_start|>` | 用户输入起始 | [已验证] |
| 3 | `<|user_end|>` | 用户输入结束 | [已验证] |
| 4 | `<|assistant_start|>` | 助手回复起始 | [已验证] |
| 5 | `<|assistant_end|>` | 助手回复结束 | [已验证] |
| 6 | `<|python_start|>` | Python 代码段起始（预定义，当前发布版本不包含可用代码执行回路） | [已验证] |
| 7 | `<|python_end|>` | Python 代码段结束 | [已验证] |
| 8 | `<|output_start|>` | 输出段起始 | [已验证] |
| 9 | `<|output_end|>` | 输出段结束 | [已验证] |

---

## 6. `<think>` 处理方式

| 字段 | 值 | 来源 |
| --- | --- | --- |
| `<think>` 是否注册为 Special Token | 否 | [已验证] |
| `</think>` 是否注册为 Special Token | 否 | [已验证] |
| 处理方式 | 保留为**普通文本序列**，由 BPE 按常规子词规则切分 | [已验证] |

> 说明：`<think>` 与 `</think>` 未作为 Special Token 注册，因此不会作为单一不可分割的 token 出现，而是被 BPE 切分为多个普通子词 token。这意味着模型识别 `<think>` 依赖其在训练数据中的出现模式，而非 token 边界。

---

## 7. Byte Fallback

| 机制 | 说明 |
| --- | --- |
| 实现方式 | 通过 Byte-Level BPE 实现 |
| 行为 | 任何 UTF-8 字符最终可表示为字节序列，避免 `<unk>` token |
| 覆盖范围 | 任意 Unicode 字符（包括训练中未见过的字符） |

> 注：Byte-Level BPE 本身即隐含 Byte Fallback 能力，本项目未额外实现独立的 Byte Fallback 模块。

---

## 8. 中英文与金融文本覆盖

| 维度 | 覆盖情况 |
| --- | --- |
| 中文通用 | 通过 SkyPile/Wiki 训练数据覆盖（40%） |
| 英文通用 | 通过 ClimbMix 训练数据覆盖（40%） |
| 中文金融 | 通过中文金融训练数据覆盖（20%） |
| 金融术语 | 训练数据包含金融报告、研报等，金融高频术语在 vocab 中应有较好覆盖 |

> 具体 vocab 中金融术语的覆盖统计需通过脚本分析 vocab 文件得出，当前未提供硬编码数字。

---

## 9. 压缩率评测

| 字段 | 值 | 来源 |
| --- | --- | --- |
| 压缩率（中文金融文本） | **待验证** | 待脚本重现后填入正式数字 |
| 压缩率（中文通用文本） | **待验证** | 待脚本重现后填入正式数字 |
| 压缩率（英文通用文本） | **待验证** | 待脚本重现后填入正式数字 |

> **重要声明**：压缩率数字当前**标记为待验证**。需通过重现脚本对 held-out 文本计算 bytes/token 或 chars/token 指标后，方可填入正式数字。在此之前**不引用任何具体压缩率数值**。

---

## 10. 限制

| 局限性 | 说明 |
| --- | --- |
| 压缩率待验证 | 正式压缩率数字需脚本重现后填入，当前不引用 |
| 不保证推理倍速 | Tokenizer 重新训练不直接带来推理速度倍数提升，需独立 benchmark |
| 金融术语覆盖未量化 | vocab 中金融术语的具体覆盖比例未提供硬编码统计 |
| 训练数据规模有限 | max_chars=2B，相对于生产级 Tokenizer 训练数据规模较小 |
| `<think>` 非 Special Token | `<think>`/`</think>` 作为普通文本序列，模型识别依赖训练模式而非 token 边界 |

---

## 11. 相关文档

- `model-card.md` — 模型卡片
- `pretraining-data-card.md` — 预训练数据卡片（含 Tokenizer 训练数据来源）
- `reproducibility.md` — 复现指南（含 Tokenizer 训练命令）
