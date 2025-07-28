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
# APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1/grpo_qwen_2_5_dfew.py    #grpo.py 

APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/src/virft/src/open_r1/grpo_qwen_2_5_dfew_reward_add_reward_with_label_for_emotion_label_change_confidence.py #grpo_qwen_2_5_dfew.py  #grpo.py 

export DATA_PATH=./share_data/valid_partial_dfew_dataset_qwen_2_5 #ViRFT_COCO_base65  #dfew_dataset_qwen_2_5   #dfew_dataset   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2.5-VL-3B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2.5-VL-3B-Instruct_GRPO_dfew_train    ### save path
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_qwen2.5_3b_GRPO_dfew_use_cache_false_dfew.txt"
# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}
# Logging
exec > "logs/log_${1}_${2}_qwen2.5_3b_dfew_usecache_false_dfew.out" 2>&1
# Run PyTorch application
WANDB_MODE=offline torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=100 \
    --rdzv_backend=c10d \
    ${APPLICATION_SCRIPT} \
    --output_dir './share_models/Qwen2.5-VL-3B-Instruct_GRPO_dfew_train' \
    --model_name_or_path './share_models/Qwen2.5-VL-3B-Instruct' \
    --dataset_name './share_data/valid_partial_dfew_dataset_qwen_2_5' \
    --deepspeed 'src/virft/local_scripts/zero3_qwen_2_5.json' \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B_GRPO_dfew_change_back_grpo_set \
    --save_steps 20 \
    --save_only_model true \
    --num_generations 2
# qsub -I -P kf09 -q gpuvolta -l ngpus=4,ncpus=48,mem=128GB,walltime=00:50:00
# qstat -n1 
# ssh gadi-gpu-v100-0103
# cd.........