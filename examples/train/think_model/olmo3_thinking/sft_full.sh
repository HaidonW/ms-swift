#!/usr/bin/env bash
set -euo pipefail

# Full fine-tuning needs sufficient GPU memory and a compatible DeepSpeed installation.
# CUDA_VISIBLE_DEVICES and NPROC_PER_NODE must describe the same number of local GPUs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-allenai/Olmo-3-7B-Think}"
DATASET="${DATASET:-${SCRIPT_DIR}/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/olmo3-7b-think-full}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

swift sft \
    --model "${MODEL}" \
    --model_type olmo3 \
    --template olmo3_thinking \
    --use_hf true \
    --tuner_type full \
    --deepspeed zero3 \
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
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
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
