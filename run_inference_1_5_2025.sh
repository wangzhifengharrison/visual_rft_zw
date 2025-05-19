#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=4
#PBS -l ncpus=48
#PBS -l mem=128GB  
#PBS -l jobfs=200GB   
#PBS -l walltime=00:30:00
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
LAUNCH_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/run_2_inference_1_5_2025.sh

# Set execute permission
chmod u+x ${LAUNCH_SCRIPT}

# Run PyTorch application
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  pbsdsh -n $inode ${LAUNCH_SCRIPT} ${NNODES} ${PROC_PER_NODE} ${MASTER_ADDR} &
done
wait




