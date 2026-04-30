# #!/bin/bash

# # 暂时执行 Setup、分词器训练 (Tokenizer) 和 基础模型预训练 (Base model pretraining)

# export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# export HF_ENDPOINT="https://hf-mirror.com"
# mkdir -p $NANOCHAT_BASE_DIR

# # -----------------------------------------------------------------------------
# # wandb setup
# if [ -z "$WANDB_RUN" ]; then
#     WANDB_RUN=dummy
# fi

# # -----------------------------------------------------------------------------
# # 重置并初始化报告内容
# python -m nanochat.report reset

# # -----------------------------------------------------------------------------
# # Tokenizer (分词器下载与训练) 修改为混合中英文文本的分词器训练

# # 下载前 8 个数据分片(~2B 字符)用于分词器训练| 下载前 170 个通用英文数据分片(~40B 字符)用于预训练基础模型+自构建中文通用和金融数据
# # python -m nanochat.dataset -n 8
# # python -m nanochat.dataset -n 170 &
# # DATASET_DOWNLOAD_PID=$!

# # 训练分词器 (vocab size = 32768)
# python -m scripts.tok_train
# # 评估分词器
# python -m scripts.tok_eval

# # -----------------------------------------------------------------------------
# # Base model (预训练基础模型)
# echo "Waiting for dataset download to complete..."
# export CUDA_VISIBLE_DEVICES=4
# wait $DATASET_DOWNLOAD_PID

# # d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
# torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# # evaluate the model: CORE metric, BPB on train/val, and draw samples
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# 开始预训练 d24 模型 
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=9.5 --device-batch-size=4 --run=$WANDB_RUN --window-pattern L

# 评估预训练模型 (CORE metric, BPB on train/val 等)
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=4

# -----------------------------------------------------------------------------
# 整合预训练阶段的统计数据，生成最终评估报告 (report.md)
python -m nanochat.report generate