# FinQuery deterministic financial tools

Phase 7 adds pure calculation helpers under `src/services/financial_tools.py`.
They are deliberately not wired into autonomous LLM tool-calling yet.

Supported helpers:

- `parse_financial_number`
- `growth_rate`
- `percentage_share`
- `sum_values`
- `verify_sum`
- `convert_scale`
- `format_ratio_percent`

The intended production flow is:

1. Retrieve cited context.
2. Extract candidate values with their source metadata.
3. Run deterministic calculations.
4. Include the calculated result and source values in the final answer.
5. Keep the original table value, unit, currency, period, and citation.

Do not let the model perform arithmetic when exact financial numbers matter.
