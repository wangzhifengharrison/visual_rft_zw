#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=8
#PBS -l ncpus=96
#PBS -l mem=256GB
#PBS -l walltime=00:10:00
#PBS -l wd
cd /scratch/kf09/zw4360/Visual-RFT

# Activate Conda
# NOTE : Replace <ENV > with your actual conda environment name
# export CONDA_ENV ='/scratch/kf09/zw4360/miniconda3/bin/activate'
# source $CONDA_ENV Visual-RFT
module load cuda/12.5.1
module load gcc/12.2.0
# module load pytorch/1.10.0

source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT

sh run_multiple_tina2_copy.sh
