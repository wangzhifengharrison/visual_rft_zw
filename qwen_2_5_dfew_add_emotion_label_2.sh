#!/bin/bash

# 1. Initialize environment modules FIRST
source ~/.bashrc
source /etc/profile.d/modules.sh  # Critical for Gadi cluster
module purge  # Clean environment
module load cuda/12.5.1  #12.5.1 #12.6.2 
module load gcc/12.2.0
source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT


# Application script grpo_qwen_2_5_dfew_reward_add_emotion_label.py without reward for emotion label
APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1/grpo_qwen_2_5_dfew_reward_add_reward_for_emotion_label.py #grpo_qwen_2_5_dfew.py  #grpo.py 
export DATA_PATH=./share_data/valid_dfew_dataset_qwen_2_5_add_emotion_label_2000 #valid_dfew_dataset_qwen_2_5_add_emotion_label #valid_dfew_dataset_qwen_2_5 #valid_partial_dfew_dataset_qwen_2_5 #ViRFT_COCO_base65  #dfew_dataset_qwen_2_5   #dfew_dataset   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2.5-VL-3B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2.5-VL-3B-Instruct_GRPO_dfew_train    ### save path
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_qwen2.5_3b_GRPO_dfew_use_cache_false_dfew_clip_offload_full_add_emotion_label_2000.txt"
# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}
# Logging
exec > "logs/log_${1}_${2}_qwen2.5_3b_dfew_usecache_false_dfew_clip_offload_full_add_emotion_label_2000.out" 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #环境变量设置来减少 CUDA 显存碎片化

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
    --deepspeed src/virft/local_scripts/zero3_offload_qwen_2_5.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B_GRPO_dfew_add_emotion_label_2000 \
    --save_steps 40 \
    --save_only_model true \
    --num_generations 4
    
