#!/usr/bin/env bash
set -euo pipefail

variant="${1:?usage: bash runs/sft_v2_ablation.sh <lr005|lr010>}"
case "$variant" in
  lr005)
    init_lr_frac="0.05"
    output_tag="d24_finance_v2_lr005"
    ;;
  lr010)
    init_lr_frac="0.10"
    output_tag="d24_finance_v2_lr010"
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

if [[ -n "${RESUME_FROM_STEP:-}" ]]; then
  model_args=(
    --model-tag="$output_tag"
    --output-model-tag="$output_tag"
    --resume-from-step="$RESUME_FROM_STEP"
    --resume-lr-scale="${RESUME_LR_SCALE:-1.0}"
  )
else
  model_args=(
    --model-tag=d24_final_mixdata
    --model-step=28000
    --output-model-tag="$output_tag"
  )
fi

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if [[ "${SKIP_SFT_V2_BUILD:-0}" != "1" ]]; then
  python finance-data-process/scripts/build_sft_v2.py
fi

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
  "${model_args[@]}" \
  --finance-train-file=finance-data-process/data/processed/sft/train_v2_balanced.jsonl \
  --finance-epochs=1 \
  --finance-cot-epochs=0 \
  --smoltalk-size=30000 \
  --mmlu-epochs=1 \
  --gsm8k-epochs=2 \
  --num-iterations="${TOTAL_ITERATIONS:-150}" \
  --device-batch-size=4 \
  --init-lr-frac="$init_lr_frac" \
  --warmup-ratio=0.05 \
  --warmdown-ratio=0.50 \
  --eval-every=25 \
  --save-every=50 \
  --early-stopping-patience=4 \
  --early-stopping-min-delta=0.0001 \
  --grad-clip=1.0 \
  --run="$output_tag"
