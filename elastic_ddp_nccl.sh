#!/bin/bash

# Load shell environment variables
source ~/.bashrc

# Load module, always specify version number.
module load pytorch/1.10.0

# Application script
APPLICATION_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/elastic_ddp_nccl.py

# Set execute permission
chmod u+x ${APPLICATION_SCRIPT}

# Run PyTorch application
torchrun --nnodes=${1} --nproc_per_node=${2} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${3}:29400 ${APPLICATION_SCRIPT} 