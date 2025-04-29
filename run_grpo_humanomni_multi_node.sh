#!/bin/bash

# Load required modules
module load cuda/12.5.1 
module load gcc/13.2.0

# Set environment variables
export DEBUG_MODE="true"
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/
mkdir -p ./logs

# Set paths
export DATA_PATH=./share_data/ViRFT_COCO_base65
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k

# Get node information
NODE_RANK=${PBS_NODENUM:-0}
NUM_NODES=${PBS_NUM_NODES:-1}
MASTER_ADDR=$(hostname -i)

# Print node information
echo "Node Rank: $NODE_RANK"
echo "Number of Nodes: $NUM_NODES"
echo "Master Address: $MASTER_ADDR"

# Run distributed training
WANDB_MODE=offline torchrun \
    --nproc_per_node=4 \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=12346 \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3_offload.json \
    --max_prompt_length 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation eager \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2 