#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=16
#PBS -l ncpus=192
#PBS -l mem=512GB  
#PBS -l jobfs=800GB   
#PBS -l walltime=01:50:00
#PBS -l wd
#PBS -j oe
#PBS -o logs/run_log_$PBS_JOBID.log

# Set variables
if [[ $PBS_NCPUS -ge $PBS_NCI_NCPUS_PER_NODE ]]
then
  NNODES=$((PBS_NCPUS / PBS_NCI_NCPUS_PER_NODE))
else
  NNODES=1
fi

PROC_PER_NODE=$((PBS_NGPUS / NNODES))

MASTER_ADDR=$(cat $PBS_NODEFILE | head -n 1)

# Launch script
LAUNCH_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/qwen_2_5_zf_run_2_coco.sh

# Set execute permission
chmod u+x ${LAUNCH_SCRIPT}

# Run PyTorch application
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  pbsdsh -n $inode ${LAUNCH_SCRIPT} ${NNODES} ${PROC_PER_NODE} ${MASTER_ADDR} &
done
wait

#https://opus.nci.org.au/spaces/Help/pages/184647980/PyTorch
# qsub zf_run_29_04_2025_coco.sh to run the experiment