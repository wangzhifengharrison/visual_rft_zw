module load cuda/12.5.1
module load gcc/12.2.0
# module load pytorch/1.10.0
# MASTER_ADDR=$(head -n1 $PBS_NODEFILE)
# GPUS_PER_NODE=4
# NNODES=$(uniq $PBS_NODEFILE | wc -l)
# NODE_RANK=$PBS_O_VNODENUM  eager
MASTER_ADDR=$(head -n1 $PBS_NODEFILE)
GPUS_PER_NODE=4
NNODES=$(uniq $PBS_NODEFILE | wc -l)

NODE_HOSTNAME=$(hostname)
NODE_RANK=$(grep -nx "$NODE_HOSTNAME" $PBS_NODEFILE | cut -d: -f1 | head -n1)
NODE_RANK=$((NODE_RANK - 1))
echo "Master address: $MASTER_ADDR"
echo "Node rank: $NODE_RANK of $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "PBS_NODEFILE: $PBS_NODEFILE"
mkdir -p ./logs
echo "19"
export DATA_PATH=./share_data/ViRFT_COCO_base65   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k    ### save path
# 打印日志路径以确认
# echo "Log path set to: $LOG_PATH"

WANDB_MODE=offline torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
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
    --attn_implementation sdpa \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 10 \
    --save_only_model true \
    --num_generations 2

