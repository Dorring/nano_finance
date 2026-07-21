# Phase 3 Formula Registry

This document records every deterministic financial calculation formula
implemented in Phase 3. Each formula is versioned, bound to fixed operand
roles, and executed with `Decimal` arithmetic only. No `eval`, `exec`, or
model-generated code is used.

## Registry Overview

| Operation           | Formula Version         | Supported |
|---------------------|-------------------------|-----------|
| difference          | difference.v1           | true      |
| growth_rate         | growth_rate.v1          | true      |
| percentage_share    | percentage_share.v1     | true      |
| sum                 | sum.v1                  | true      |
| average             | average.v1              | true      |
| gross_margin        | gross_margin.v1         | true      |
| net_margin          | net_margin.v1           | true      |
| debt_ratio          | debt_ratio.v1           | true      |
| scale_conversion    | scale_conversion.v1     | true      |

All 9 operations are executable. `SCALE_CONVERSION` was fully implemented
in the Phase 3 closeout (Option A).

---

## Per-Operation Specification

### 1. difference

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `current - previous`           |
| Formula Version  | `difference.v1`                |
| Required Roles   | `current`, `previous`          |
| Output Unit      | `base`                         |
| Precision        | 2 decimal places               |
| Zero Division    | N/A (no division)              |
| Supported        | true                           |

### 2. growth_rate

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `(current - previous) / previous` |
| Formula Version  | `growth_rate.v1`               |
| Required Roles   | `current`, `previous`          |
| Output Unit      | `ratio` (rendered as percentage) |
| Precision        | 2 decimal places (percentage)  |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `previous = 0` |
| Supported        | true                           |

### 3. percentage_share

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `part / total`                 |
| Formula Version  | `percentage_share.v1`          |
| Required Roles   | `part`, `total`                |
| Output Unit      | `ratio` (rendered as percentage) |
| Precision        | 2 decimal places (percentage)  |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `total = 0` |
| Supported        | true                           |

### 4. sum

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `sum(operands)`                |
| Formula Version  | `sum.v1`                       |
| Required Roles   | (none — generic, variable operands) |
| Output Unit      | `base`                         |
| Precision        | 2 decimal places               |
| Zero Division    | N/A (no division)              |
| Supported        | true                           |

### 5. average

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `sum(operands) / count`        |
| Formula Version  | `average.v1`                   |
| Required Roles   | (none — generic, variable operands) |
| Output Unit      | `base`                         |
| Precision        | 2 decimal places               |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `count = 0` (prevented by `min_operands = 1`) |
| Supported        | true                           |

### 6. gross_margin

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `(revenue - cogs) / revenue`   |
| Formula Version  | `gross_margin.v1`              |
| Required Roles   | `revenue`, `cogs`              |
| Output Unit      | `ratio` (rendered as percentage) |
| Precision        | 2 decimal places (percentage)  |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `revenue = 0` |
| Supported        | true                           |

### 7. net_margin

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `net_income / revenue`         |
| Formula Version  | `net_margin.v1`                |
| Required Roles   | `revenue`, `net_income`        |
| Output Unit      | `ratio` (rendered as percentage) |
| Precision        | 2 decimal places (percentage)  |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `revenue = 0` |
| Supported        | true                           |

### 8. debt_ratio

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `total_liabilities / total_assets` |
| Formula Version  | `debt_ratio.v1`                |
| Required Roles   | `total_liabilities`, `total_assets` |
| Output Unit      | `ratio` (rendered as percentage) |
| Precision        | 2 decimal places (percentage)  |
| Zero Division    | Returns `BLOCKED` with `ZERO_DIVISION` when `total_assets = 0` |
| Supported        | true                           |

### 9. scale_conversion

| Field            | Value                          |
|------------------|--------------------------------|
| Formula          | `value * from_factor / to_factor` |
| Formula Version  | `scale_conversion.v1`          |
| Required Roles   | `value` (single operand)       |
| Output Unit      | `base`                         |
| Precision        | 2 decimal places               |
| Zero Division    | N/A (factors are non-zero constants) |
| Supported        | true                           |

**Scale factors** (defined in `primitive_tools._SCALE_FACTORS`):

| Scale Name      | Factor (Decimal)   |
|-----------------|--------------------|
| `""`, `"ones"`, `"unit"` | 1               |
| `"thousand"`, `"k"`      | 1,000           |
| `"million"`, `"m"`       | 1,000,000       |
| `"billion"`, `"bn"`      | 1,000,000,000   |
| `"万"`, `"万元"`          | 10,000          |
| `"千万"`, `"千万元"`       | 10,000,000      |
| `"百万"`, `"百万元"`       | 1,000,000       |
| `"亿"`, `"亿元"`          | 100,000,000     |

**BLOCKED conditions for scale_conversion:**
- Missing source scale → `UNIT_AMBIGUOUS`
- Missing target scale → `UNIT_AMBIGUOUS`
- Currency keyword detected (e.g., "美元", "rmb", "usd") → `CURRENCY_NOT_SUPPORTED`
- Percentage value → rejected

---

## Conventions

1. **Decimal-only arithmetic**: All operands and results use `decimal.Decimal`.
   No `float`, no `eval`, no `exec`, no `compile`.
2. **Fixed formulas**: Each operation has exactly one formula version. No
   model-generated code is executed.
3. **Evidence binding**: Every operand carries `source_text`,
   `evidence_chunk_id`, `document_name`, and `page` from the retrieved
   evidence.
4. **Layer dependency**: `domain → finance → application → services`.
   Formula primitives live in `src/finance/primitive_tools.py` and
   `src/finance/calculation_registry.py`.
5. **LLM bypass**: `EXECUTED`, `BLOCKED`, and `FAILED` all bypass the LLM.
   Only `NOT_APPLICABLE` continues to the normal RAG/LLM flow.
