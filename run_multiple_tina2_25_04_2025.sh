#!/bin/bash

# 1. Initialize environment modules FIRST
source /etc/profile.d/modules.sh  # Critical for Gadi cluster
module purge  # Clean environment
module load cuda/12.5.1
module load gcc/12.2.0
source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT

# 7. Set up logging with PBS variables
mkdir -p logs
exec > logs/log_${PBS_O_VNODENUM}_$(hostname).out 2>&1

echo "=== tina2.sh on $(hostname) ==="
echo "Inherited PBS_NODEFILE   = '$PBS_NODEFILE'"
echo "Inherited PBS_O_HOSTFILE = '$PBS_O_HOSTFILE'"
echo "Inherited PBS_O_VNODENUM = '$PBS_O_VNODENUM'"

MASTER_ADDR=$(head -n1 $PBS_NODEFILE)

MASTER_PORT=$(( 15000 + RANDOM % 17001 ))  # RANDOM is 0-32767 in bash
export MASTER_PORT

GPUS_PER_NODE=4
# Get unique nodes in order of first occurrence (avoids sorting)
UNIQUE_NODES=$(awk '!seen[$0]++' $PBS_NODEFILE)
NNODES=$(echo "$UNIQUE_NODES" | wc -l)
NODE_HOSTNAME=$(hostname)
# Find rank in the UNIQUE_NODES list (not raw PBS_NODEFILE)
NODE_RANK=$(echo "$UNIQUE_NODES" | grep -nxF "$NODE_HOSTNAME" | cut -d: -f1)
NODE_RANK=$((NODE_RANK - 1))  # Convert to 0-based index
export NODE_RANK
export MASTER_ADDR

echo "Node rank: $NODE_RANK of $NNODES"
echo "Master Addr: $MASTER_ADDR"

export DATA_PATH=./share_data/ViRFT_COCO_base65   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k    ### save path


# src/virft/src/open_r1/grpo.py \

WANDB_MODE=offline torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    test_ddp.py \
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
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 10 \
    --save_only_model true \
    --num_generations 2

echo "torchrun exited with code $? at $(date)"