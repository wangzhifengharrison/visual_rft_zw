# There's no need for prolonged training. For a dataset with only a few hundred samples, 200 steps should be sufficient.
#qsub -I  -P kf09 -q gpuvolta -l ngpus=8,ncpus=96,mem=256GB,walltime=00:30:00
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_coco_base65cate_6k.txt"
module load cuda/12.5.1 #12.1
module load gcc/12.2.0
export DATA_PATH=./share_data/ViRFT_COCO_base65   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k    ### save path
# export WANDB_MODE=offline
export WANDB_MODE=disabled
# nproc_per_node=8
# num_generations =8
# --report_to wandb \
# --attn_implementation flash_attention_2 \
# --attn_implementation xformers \$NPROC_PER_NODE


torchrun --nproc_per_node=4 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
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
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2
