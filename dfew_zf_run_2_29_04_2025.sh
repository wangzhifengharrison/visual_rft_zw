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
APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1/grpo.py 
export DATA_PATH=./share_data/valid_dfew_dataset_qwen_2   ### your local dataset
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_dfew_dataset_640    ### save path
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_GRPO_dfew_0_1000_640_use_cache_false_valid_2.txt"
# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}
# Logging
exec > "logs/log_${1}_${2}_dfew_original_usecache_false_640_valid_2.out" 2>&1
# Run PyTorch application
WANDB_MODE=offline torchrun \
    --nnodes=${1} \
    --nproc_per_node=${2} \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${3}:29400 \
    ${APPLICATION_SCRIPT} \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_dfew_dataset_640 \
    --save_steps 50 \
    --save_only_model true \
    --num_generations 4
