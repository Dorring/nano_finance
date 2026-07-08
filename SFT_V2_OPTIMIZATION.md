# SFT v2 Optimization

## Evidence

- SFT1147 improves the finance validation macro primary score from `0.3736`
  at SFT800 to `0.4432`, despite worse aggregate validation BPB.
- Numeric and table QA remain the main failures. SFT1147 reaches only `0.1136`
  numeric accuracy and `0.3333` table exact match.
- The original merged finance corpus assigns `74.1%` of assistant tokens to
  `finance_r1`, while FinQA receives `1.09%` and FinSen `0.24%`.
- Base hits the 256-token generation limit on 183/200 examples. SFT800 and
  SFT1147 reduce that to 24/200 and 25/200.

## Data mixture

`build_sft_v2.py` creates a deterministic one-million-assistant-token finance
set. The allocation is:

| Source | Share |
|---|---:|
| FinQA | 27.5% |
| TAT-QA | 22.5% |
| FinER | 10% |
| FinRED | 10% |
| FinSen | 2.5% |
| FiQA | 5% |
| ECTSum | 12.5% |
| Finance R1 CoT | 10% |

This favors the weak numeric/table tasks, retains extraction and summary
coverage, reduces already-strong sentiment repetition, and caps long CoT.

## Screening experiments

Run two 150-step screens from Base28000 with identical data and seeds:

```bash
bash runs/sft_v2_ablation.sh lr005
bash runs/sft_v2_ablation.sh lr010
```

Only `init_lr_frac` changes (`0.05` versus `0.10`). Both use gradient clipping,
25-step validation, early stopping, and an independent output model tag.

## Selection

Evaluate each run's `best.json` checkpoint on the same 200 validation IDs.
Select by finance macro primary score, with these constraints:

- numeric accuracy must exceed SFT1147's `0.1136`;
- extraction JSON validity must remain `1.0`;
- sentiment accuracy must not fall by more than 3 percentage points;
- empty outputs must remain zero;
- MMLU/GSM8K regression is checked before extending the winner.

Extend only the winning screen to roughly 300–400 total steps. Do not run the
sealed test until the final model and RAG configuration are fixed.

The extension is a low-learning-rate second stage. Resume the winning
checkpoint with `RESUME_LR_SCALE=0.25` so the short screening schedule does not
jump from zero back to the original peak learning rate.

## Results

| Model | Best BPB | Finance macro primary |
|---|---:|---:|
| SFT800 | 0.4783 | 0.3736 |
| SFT1147 | 0.4842 | **0.4432** |
| V2 lr005 step150 | 0.5715 | 0.1295 |
| V2 lr010 step150 | 0.5558 | 0.2297 |
| V2 lr010 step275 | 0.5527 | 0.2077 |

The lr010 screen beat lr005, so only lr010 entered the low-learning-rate
extension. Its BPB improved through step275, but the application score fell
from `0.2297` to `0.2077`. Relation extraction fell from `0.2162` to `0.0588`.
SFT1147 remains the application baseline.

## Findings

- Aggregate BPB is not a sufficient checkpoint-selection metric for this
  multi-task application. Task metrics must be evaluated at candidate steps.
- Repeating final-answer-only FinQA examples increases short supervised tokens
  but does not add the missing calculation or tool-use signal.
- Strict assistant-token quotas can downsample structured extraction corpora.
  FinER and FinRED need full unique-record coverage plus a minimum epoch count.
- Long CoT needed capping, but reducing CoT alone did not compensate for the
  loss of structured-task exposure.

No additional training should be launched from these v2 checkpoints. A future
v2.1 must add verifiable calculation traces/tool supervision and enforce
minimum source coverage before another small screen. The sealed test remains
untouched. Current work should proceed with SFT1147 and shift to RAG retrieval
and generation improvements.

## Next SFT direction

SFT is still useful, but the next run must not continue from the failed v2
recipe. The next model experiment should be treated as SFT v3 and gated by
application metrics:

- keep SFT1147 as the current production/application baseline;
- use validation-only finance metrics for checkpoint selection, not aggregate
  BPB alone;
- add normal-chat and finance-chat smoke regression before replacing SFT1147;
- preserve general dialogue data instead of running finance-only specialization;
- require full unique-record coverage or a minimum epoch floor for FinER and
  FinRED so structured extraction is not downsampled away;
- add verifiable calculation traces or tool-style supervision for FinQA/TAT-QA
  rather than repeating final-answer-only samples;
- keep Finance R1 capped, because it previously dominated assistant tokens
  without fixing numeric/table tasks.

The code-side support for this gate is:

```bash
python -m scripts.dialogue_generate \
  --source sft \
  --model-tag d24_final_mixdata \
  --step 1147 \
  --eval-set artifacts/dialogue_eval/dialogue_smoke_val.jsonl \
  --output artifacts/dialogue_eval/sft_1147_predictions.jsonl

python -m scripts.dialogue_compare \
  --eval-set artifacts/dialogue_eval/dialogue_smoke_val.jsonl \
  --prediction SFT1147=artifacts/dialogue_eval/sft_1147_predictions.jsonl \
  --json-output artifacts/dialogue_eval/dialogue_comparison.json \
  --markdown-output artifacts/dialogue_eval/dialogue_comparison.md
```

Model generation still belongs on the GPU execution thread. The current code
thread only owns evaluation scripts, report generation, and review.

## Dialogue smoke results

The validation-only dialogue smoke set contains 50 ASCII examples across
`normal_chat`, `finance_chat`, `format_following`, `refusal_no_evidence`, and
`multi_turn`. All evaluated checkpoints produced 50/50 unique predictions with
zero empty and zero contentless outputs.

| Model | Macro | finance_chat | format_following | multi_turn | normal_chat | refusal_no_evidence |
|---|---:|---:|---:|---:|---:|---:|
| SFT1147 | **0.5627** | **0.5236** | 0.5287 | **0.6596** | **0.6235** | **0.4782** |
| SFT800 | 0.5578 | 0.5086 | **0.5549** | 0.6411 | 0.6078 | 0.4764 |
| V2 lr010 step150 | 0.5467 | 0.5228 | 0.5278 | 0.6430 | 0.6022 | 0.4379 |
| V2 lr010 step275 | 0.5450 | 0.5227 | 0.5420 | 0.5839 | 0.6205 | 0.4561 |

SFT1147 remains the best overall dialogue checkpoint. SFT800 is close, while the
v2 checkpoints do not improve dialogue behavior and step275 regresses on
multi-turn handling. The smoke set also exposes weak refusal and format
following behavior: refusal detection is near zero for most models, and JSON
format validity is zero on the current format-following checks. This should be
handled in the application/RAG layer with stronger prompt contracts, evidence
gating, citation requirements, and structured output validation before another
SFT round is launched.
