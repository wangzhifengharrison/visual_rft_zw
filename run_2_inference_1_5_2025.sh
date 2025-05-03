#!/bin/bash

# 1. Initialize environment modules FIRST
source ~/.bashrc
source /etc/profile.d/modules.sh  # Critical for Gadi cluster
module purge  # Clean environment
module load cuda/12.5.1
module load gcc/12.2.0
source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT


# Application script
APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/coco_evaluation/Qwen2_VL_coco_infere.py

# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}
# Logging
exec > "logs/log_inference_${1}_${2}.out" 2>&1
# Run PyTorch application
WANDB_MODE=offline torchrun \
    --nnodes=${1} \
    --nproc_per_node=${2} \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${3}:29400 \
    ${APPLICATION_SCRIPT} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 300 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k_inference \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4
