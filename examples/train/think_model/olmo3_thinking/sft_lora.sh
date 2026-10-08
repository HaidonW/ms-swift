#!/usr/bin/env bash
set -euo pipefail

# Run with Bash on Linux / WSL2, after installing this checkout with `pip install -e .`.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-allenai/Olmo-3-7B-Think}"
DATASET="${DATASET:-${SCRIPT_DIR}/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/olmo3-7b-think-lora}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

swift sft \
    --model "${MODEL}" \
    --model_type olmo3 \
    --template olmo3_thinking \
    --use_hf true \
    --tuner_type lora \
    --dataset "${DATASET}" \
    --split_dataset_ratio 0 \
    --eval_strategy no \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --gradient_checkpointing true \
    --packing false \
    --padding_free false \
    --preserve_thinking true \
    --use_liger_kernel false \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_length 2048 \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers 0 \
    --load_from_cache_file false \
    --report_to none \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
