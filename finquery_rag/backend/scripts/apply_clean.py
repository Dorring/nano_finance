"""Apply clean method replacements to rag_engine.py."""
import re

def replace_method(content, sig, replacement_file):
    """Find method by signature and replace with content from file."""
    idx = content.find(sig)
    if idx < 0:
        print(f"  NOT FOUND: {sig[:60]}...")
        return content, False
    
    # Find next method/staticmethod/classmethod after this one
    next_def = None
    for pattern in ["\n    def ", "\n    @staticmethod", "\n    @classmethod"]:
        nd = content.find(pattern, idx + len(sig) + 1)
        if nd > 0 and (next_def is None or nd < next_def):
            next_def = nd
    
    if next_def is None or next_def < 0:
        print(f"  ERROR: cannot find end of method at offset {idx}")
        return content, False
    
    with open(replacement_file, encoding="utf-8") as f:
        replacement = f.read()
    
    result = content[:idx] + replacement + content[next_def:]
    print(f"  Replaced {next_def - idx} chars")
    return result, True

# Main
path = "src/services/rag_engine.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

orig_lines = len(content.split("\n"))

# 1. Replace _expand_retrieval_query
sig1 = "\n    def _expand_retrieval_query(self, query: str) -> str:"
content, ok = replace_method(content, sig1, "scripts/clean_expand.txt")
print(f"1. _expand_retrieval_query: {'OK' if ok else 'FAILED'}")

# 2. Replace _targeted_numeric_summary
sig2 = "\n    @classmethod\n    def _targeted_numeric_summary(cls, query: str, selected: list[dict], *, context: str | None = None) -> str | None:"
content, ok = replace_method(content, sig2, "scripts/clean_targeted.txt")
print(f"2. _targeted_numeric_summary: {'OK' if ok else 'FAILED'}")

# 3. Replace _summarize_factual_evidence
sig3 = "\n    @staticmethod\n    def _summarize_factual_evidence(query: str, selected: list[dict], *, context: str | None = None) -> str | None:"
content, ok = replace_method(content, sig3, "scripts/clean_factual.txt")
print(f"3. _summarize_factual_evidence: {'OK' if ok else 'FAILED'}")

new_lines = len(content.split("\n"))
print(f"Lines: {orig_lines} -> {new_lines}")

# Audit for remaining leaks
for entity in ["FINAL Annual Report", "wipo_pub", "leac203", "basic and formal annual", "Amba Ltd", "Sunfill Ltd", "Black Swan Ltd"]:
    lines = [i for i, l in enumerate(content.split("\n"), 1) if entity.lower() in l.lower() and not l.strip().startswith("#") and "Phase 1" not in l and "removed" not in l.lower()]
    status = f"LEAK at {lines}" if lines else "GONE"
    print(f"  {entity}: {status}")

for val in ["143,540", "387,063", "468,272", "42.2 million"]:
    clean_val = val.replace(",", "").replace(" ", "").lower()
    lines = [i for i, l in enumerate(content.split("\n"), 1) if clean_val in l.replace(",", "").replace(" ", "").lower() and not l.strip().startswith("#")]
    status = f"LEAK at {lines}" if lines else "GONE"
    print(f"  {val}: {status}")

# Check benchmark terms in query expansion
exp_sig = "def _expand_retrieval_query"
exp_idx = content.find(exp_sig)
if exp_idx >= 0:
    next_m = content.find("\n    def ", exp_idx + 10)
    exp_text = content[exp_idx:next_m] if next_m > 0 else content[exp_idx:exp_idx+1000]
    leaks = []
    for term in ["wipo", "pct system", "madrid system", "leac", "amba", "sunfill", "black swan", "budget", "IPSAS", "GAAP", "Statement V"]:
        if term in exp_text.lower():
            leaks.append(term)
    if leaks:
        print(f"  EXPANSION LEAKS: {leaks}")
    else:
        print("  Query expansion: CLEAN")

print(f"supporting_source_page: {content.count('supporting_source_page')}")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Saved.")
