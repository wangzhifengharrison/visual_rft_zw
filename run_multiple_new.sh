#!/bin/bash
#PBS -P kf09
#PBS -l ncpus=96
#PBS -l mem=256GB
#PBS -l ngpus=4
#PBS -l walltime=02:00:00
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -q gpuvolta
#PBS -l software=pytorch
#PBS -P your_project
#PBS -N virftjob
#PBS -o out.log
#PBS -e err.log
module load cuda/12.5.1
module load gcc/12.2.0
module load openmpi

# get master node IP
MASTER_ADDR=$(head -n 1 $PBS_NODEFILE)
echo "Master address: $MASTER_ADDR"

# total number of GPUs
GPUS_PER_NODE=4
NNODES=2
NP=$(($GPUS_PER_NODE * $NNODES))

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$PBS_VNODENUM \
  --master_addr=$MASTER_ADDR \
  --master_port=12345 \
  src/virft/src/open_r1/grpo.py \
  --output_dir ./share_models/output \
  --model_name_or_path ./share_models/Qwen2-VL-2B-Instruct \
  --dataset_name ./share_data/ViRFT_COCO_base65 \
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
  --save_steps 100 \
  --save_only_model true \
  --num_generations 2
