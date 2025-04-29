cd /scratch/kf09/zw4360/Visual-RFT
# qsub -I official_provided_train_scription_on_github.sh -P kf09 -q gpuvolta -l ngpus=8,ncpus=96,mem=128GB,walltime=00:20:00
# qstat -n1 to check the node.
# ssh gadi-gpu-v100-0097
# conda activate R1-Omni
module load cuda/12.5.1 
module load gcc/12.2.0
# module load wandb #conda install -c conda-forge wandb
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/humanomni_emotion_emer_1format_withpath_withchoice.txt"
export HF_HOME=/mnt/data/jiaxing.zjx/cache/huggingface/
mkdir -p ./logs
export DATA_PATH=./share_data/ViRFT_COCO_base65   ### your local dataset downloading from huggingface
export CKPT_PATH=./share_models/Qwen2-VL-2B-Instruct    ### Qwen2-VL-2B checkpoint path
export SAVE_PATH=./share_models/Qwen2-VL-2B-Instruct_GRPO_coco_base65cate_6k    ### save path
# 打印日志路径以确认
echo "Log path set to: $LOG_PATH"
# nproc_per_node-每个节点运行 4 个 GPU 进程（说明你在单机 4 GPU 上训练）
# model_name_or_path-加载的初始模型路径（比如 R1-Omni-0.5B 是你的预训练模型）。
# output_dir：训练输出保存目录（模型 checkpoint 和日志等）
# dataset_name -训练数据所在路径，指向你组织好的 DFEW_all 情感识别数据集。
#deepspeed-启用 DeepSpeed，配置文件路径是 zero3.json，用于内存优化
# per_device_train_batch_size-每个 GPU 的 batch size 是 1（小模型或内存紧张时使用）。
# max_pixels-图像输入的最大像素数限制，用于控制显存（例如 640x640 图像 ≈ 409600 pixels）
#使用 FlashAttention 2 来加速注意力机制计算，提高性能和节省显存。
# 在训练中生成的候选输出数为 2（用于 RL 中采样、排序、反馈）。eager
WANDB_MODE=offline torchrun --nproc_per_node=4 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
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

    # --nnodes="1" \
    # --node_rank="0" \
    # --master_addr="127.0.0.1" \
    # --master_port="12346" \
    # src/virft/src/open_r1/grpo.py \
    # --output_dir ./outputs/test_humanomni_emer_1format_withpath_withchoice/ \
    # --model_name_or_path ../../R1-Omni-0.5B/ \
    # --dataset_name ../../DFEW_all \
    # --deepspeed local_scripts/zero3.json \
    # --max_prompt_length 512 \
    # --max_completion_length 512 \
    # --per_device_train_batch_size 1 \
    # --gradient_accumulation_steps 2 \
    # --logging_steps 1 \
    # --bf16 \
    # --report_to wandb \
    # --gradient_checkpointing false \
    # --attn_implementation flash_attention_2 \
    # --max_pixels 401408 \
    # --num_train_epochs 2 \
    # --run_name Qwen2-VL-2B-GRPO-emotion \
    # --save_steps 1000 \
    # --save_only_model true \
    # --num_generations 4   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
#        --model_name_or_path /mnt/data/jiaxing.zjx/code/HumanOmni/work_dirs/humanomniqwen2_siglip/finetune_HumanOmni_1B_Omni_emer_withchoice \
#    --dataset_name /mnt/data/jiaxing.zjx/code/R1-V-Qwen/R1-V/leonardPKU/clevr_cogen_a_train \
