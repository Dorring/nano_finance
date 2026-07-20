# Eval Contamination Notice

## Deprecated Metrics

The following evaluation results were produced using retrieval code that contained
hardcoded document-to-page mappings. These results are marked **deprecated_contaminated**
and MUST NOT be used as resume, README, or project quality indicators.

### Contamination Source

- Branch/commit range: All commits up to and including `329a01a` on `chore/nf-00-baseline-audit`
- Affected methods (now removed in Phase 1):
  - `_fallback_pages_for_query()` - hardcoded doc->page rules
  - `_supporting_pages_for_query()` - eval-specific page injection
  - `_force_supporting_page_coverage()` - forced top-k inclusion
  - `_augment_with_page_fallbacks()` - score floor injection
  - `_ensure_supporting_sources()` - supporting source propagation
- Affected metadata flag: `supporting_source_page`

### Contaminated Metrics

Any RAG evaluation result produced before Phase 1 (fix/nf-01-retrieval-integrity)
that used these methods is contaminated:

| Metric | Status | Replacement |
|--------|--------|-------------|
| Recall@K | deprecated_contaminated | Must re-measure after Phase 1 |
| MRR | deprecated_contaminated | Must re-measure after Phase 1 |
| Citation Precision | deprecated_contaminated | Must re-measure after Phase 1 |
| Citation Recall | deprecated_contaminated | Must re-measure after Phase 1 |
| Answer Accuracy | deprecated_contaminated | Must re-measure after Phase 5 |

### Clean Metrics

The following metrics will be valid AFTER Phase 5 completes:
- Oracle Context Answer Accuracy
- Oracle Generation Upper Bound
- All retrieval/citation/answer metrics measured on the sealed test set

### Audit Trail

- `artifacts/retrieval-integrity/before.json` - Snapshot of contaminated state
- `artifacts/retrieval-integrity/after.json` - Snapshot of clean state
- `artifacts/retrieval-integrity/diff.md` - Behavioral diff

### Instructions for Future Phases

1. Phase 5 must create a sealed test set with document-level isolation
2. All resume/README metrics must come from the sealed test set
3. Smoke fixtures remain for CI validation only, not quality claims
4. Do not restore any of the removed methods to improve metrics
