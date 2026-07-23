# Invalidated Placeholder Run (Phase 5 v0)

> 该运行只验证了评测框架能够执行，不代表系统质量。

This directory preserves the original Phase 5 v0 artifacts that were produced
before the evaluation corpus, indexes, labels, calibration, ablation, and
sealed-run discipline were rebuilt. They are kept for traceability only and
**MUST NOT** be cited in any of the following contexts:

- Resume / CV metrics
- Model or config selection decisions
- Public PR descriptions of system quality
- Component-contribution (ablation) claims
- Threshold calibration winners

## Reason Codes

| Code | Meaning |
| --- | --- |
| `EVALUATION_DOCUMENTS_NOT_INDEXED` | Expected source documents referenced by labels were never indexed in the production ChromaDB / BM25 stores, so retrieval-grounded metrics were driven by "document not found" failures rather than real system behavior. |
| `CALIBRATION_NOT_RERUN_PER_CONFIG` | The calibration "winner" was produced by `apply_params_to_prediction()` post-hoc simulation on existing predictions, not by re-running `Retrieval → Context → Calculator → LLM → Validation` per candidate config. |
| `ABLATION_RUNTIME_CONFIG_NOT_APPLIED` | A0-A9 ablation variants only mutated a config-dict key without disabling the actual component at runtime, so observed metrics were identical across variants and cannot support component-contribution claims. |

## Original Provenance

- Original HEAD: `ded8a6f21606e0f0ede6b4597e4e5a809ebc4b3d`
- Original RC commit: `82a92d8`
- Original selected-config commit: `c29f4ca`

## Original Results (Invalidated)

| Partition | Result | Note |
| --- | --- | --- |
| Dev baseline | 0/10 strict pass | All expected sources missing from index |
| Calibration | 11,664 combinations, winner = default | Post-hoc simulation, not end-to-end |
| Ablation A0-A9 | All 0.0 macro_strict_pass_rate | Identical → config not applied at runtime |
| Sealed v1 | 0/5 strict pass | Same "document not found" pattern |

The original Phase 5 v1 selected-config (commit `c29f4ca`) is also marked
`invalidated` and must not be loaded by the sealed runner. Phase 5 v2 will
rebuild a real selected-config (or keep Phase 4 baseline) under the new
discipline.
