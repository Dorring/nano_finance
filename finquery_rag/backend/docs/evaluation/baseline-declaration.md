# Phase 5 Evaluation System Baseline Declaration

## Status

The Phase 5 evaluation baseline is **NOT** the original Phase 4 merge
commit `49eb681` unchanged. The Phase 5 branch includes one declared
production hardening change to `src/services/retrieval.py`:

```
-import jieba_fast as jieba
+try:
+    import jieba_fast as jieba
+except ImportError:
+    import jieba
```

This change is a dependency-availability fallback — it does not add,
remove, or modify any retrieval functionality. When `jieba_fast` is
available (the Phase 4 production environment), behaviour is identical
to the original code. When `jieba_fast` is unavailable (the Phase 5
evaluation host, which lacks Cython headers), the pure-Python `jieba`
fallback is used.

## Parity Verification

Parity tests in `tests/evaluation/test_jieba_parity.py` verify:

1. **Tokenization parity**: `jieba_fast.cut_for_search` and
   `jieba.cut_for_search` produce identical tokens for common Chinese
   financial queries (when both libraries are available).
2. **Fallback correctness**: when `jieba_fast` is unavailable, the
   module still imports and produces valid tokenization.
3. **Retrieval module contract**: the imported `jieba` attribute is
   functional and has `cut_for_search`.

On the evaluation host (where `jieba_fast` cannot compile), the parity
test is skipped — the fallback is the only option, and correctness is
verified by the fallback correctness tests.

## Evaluation System Baseline Commit

The declared `evaluation_system_baseline` is the commit on the
`feat/nf-05-sealed-evaluation` branch that contains the jieba fallback.
The RC freeze manifest records this commit explicitly. It must NOT be
referred to as "Phase 4 original baseline" — it is the
"evaluation system baseline with declared jieba fallback".

## Why Not Revert (Option A)?

Reverting the jieba fallback would make the evaluation host unable to
run the RAG engine at all (ImportError on `jieba_fast`). This would
prevent any real end-to-end evaluation. Option B (declare the fix) is
the only viable path for real evaluation on the available hardware.
