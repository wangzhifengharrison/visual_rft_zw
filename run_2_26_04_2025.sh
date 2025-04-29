#!/bin/bash

# 1. Initialize environment modules FIRST
source /etc/profile.d/modules.sh  # Critical for Gadi cluster
module purge  # Clean environment
module load cuda/12.5.1
module load gcc/12.2.0
source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Safely handle PBS variables
export PBS_O_VNODENUM=${PBS_O_VNODENUM:-0}  # Default to 0 if unset

echo "=== tina2.sh on $(hostname) ==="
echo "Inherited PBS_NODEFILE   = '$PBS_NODEFILE'"
echo "Inherited PBS_O_HOSTFILE = '$PBS_O_HOSTFILE'"
echo "Inherited PBS_O_VNODENUM = '$PBS_O_VNODENUM'"

# Get unique sorted nodes from PBS_NODEFILE
UNIQUE_NODES=($(sort -u "$PBS_NODEFILE"))
NNODES=${#UNIQUE_NODES[@]}
GPUS_PER_NODE=4

# Determine current node's rank
CURRENT_NODE=$(hostname)
NODE_RANK=-1
for i in "${!UNIQUE_NODES[@]}"; do
  if [[ "${UNIQUE_NODES[$i]}" == "$CURRENT_NODE" ]]; then
    NODE_RANK=$i
    break
  fi
done

export WANDB_MODE=offline
export MASTER_ADDR=${UNIQUE_NODES[0]}
export MASTER_PORT  # Make sure this is set in tina.sh

# Logging
exec > "logs/log_${NODE_RANK}_$(hostname).out" 2>&1
echo "=== Node Rank: $NODE_RANK / $NNODES, Master Addr: $MASTER_ADDR, Port: $MASTER_PORT ==="

export DATA_PATH=./share_data/ViRFT_COCO_base65   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k    ### save path

# src/virft/src/open_r1/grpo.py \
    # --deepspeed src/virft/local_scripts/zero3_offload.json \
        # test_ddp.py \

WANDB_MODE=offline torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 100 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --fp16 \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 10 \
    --save_only_model true \
    --num_generations 2

echo "torchrun exited with code $? at $(date)"
