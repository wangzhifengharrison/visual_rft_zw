#!/bin/bash
#SBATCH --job-name=qwen2.5_GRPO_dfew
#SBATCH --account=OD-220461
# 16 GPUs total ⇒ 4 nodes × 4 GPUs/node
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4

# 192 CPUs total ⇒ 4 nodes × 48 CPUs/node
#SBATCH --cpus-per-task=4

# 512 GB memory per node
#SBATCH --mem=512G

# walltime 5h
#SBATCH --time=10:00:00

# run from your project directory
#SBATCH --chdir=/home/zhe030/zhe030/ZFW/Visual-RFT
#
# logs
#SBATCH --output=logs/slurm_qwen2_5_grpo-%j.out
#SBATCH --error=logs/slurm_qwen2_5_grpo-%j.err

### — Load your environment just like in your PBS wrapper
source ~/.bashrc
source /etc/profile.d/modules.sh
# module purge
module load cuda/12.6.3 #12.5.1 #12.6.2
module load gcc/12.3.0
# dynamically get the conda base
CONDA_BASE=$(conda info --base)
# which will be /apps/miniconda3/23.5.2
source "${CONDA_BASE}/etc/profile.d/conda.sh" #source /apps/miniconda3/23.5.2/etc/profile.d/conda.sh
# source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT

### — Set up and launch your application
APPLICATION_SCRIPT=/home/zhe030/zhe030/ZFW/Visual-RFT/src/virft/src/open_r1/grpo_qwen_2_5_dfew_reward_add_reward_with_label_for_emotion_label_change_confidence_use_score_not_bertscore_more_than_0_6.py
export DATA_PATH=./share_data/valid_dfew_dataset_qwen_2_5_add_emotion_label_all_data_in_set1_train
export CKPT_PATH=./share_models/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=./share_models/Qwen2.5-VL-3B-Instruct_GRPO_dfew_train_slurm
export DEBUG_MODE="true"
export LOG_PATH="./slurm_log_qwen2.5_3b_GRPO_dfew_0_6_mean.txt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

##############
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONWARNINGS="ignore"
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL  # Avoid interconnect conflicts
export PYTHONUNBUFFERED=1


# ensure your script is executable
chmod +x ${APPLICATION_SCRIPT}

# Rendezvous via C10D
MASTER_ADDR=$(hostname -s)
MASTER_PORT=29400
echo "MASTER_ADDR="$MASTER_ADDR"${MASTER_ADDR}:${MASTER_PORT}, ${SLURM_GPUS_ON_NODE} "

# Logging
exec > "logs/log_qwen2.5_3b_dfew_0_6_mean_2.out" 2>&1
# run across all 4 nodes, 4 procs (GPUs) ${SLURM_NNODES}, SLURM_GPUS_ON_NODE, per node srun flash_attention_2, zero3_offload_qwen_2_5.json,sdpa
# eager,sdpa,flash_attention_2,zero3_offload_qwen_2_5.json, zero3.json
WANDB_MODE=offline srun torchrun \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --nproc_per_node=${SLURM_GPUS_ON_NODE} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    ${APPLICATION_SCRIPT} \
    --output_dir ${SAVE_PATH} \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --gradient_checkpointing false \
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2.5-VL-3B_GRPO_dfew_add_reward_for_emotion_label_all \
    --save_steps 40 \
    --save_only_model true \
    --num_generations 4
