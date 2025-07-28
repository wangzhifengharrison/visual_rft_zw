#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ncpus=96
#PBS -l ngpus=8
#PBS -l mem=256GB
#PBS -l jobfs=800GB
#PBS -l walltime=00:05:00
#PBS -l wd

# Must include `#PBS -l storage=scratch/ab12+gdata/yz98` if the job
# needs access to `/scratch/ab12/` and `/g/data/yz98/`. Details on:
# https://opus.nci.org.au/display/Help/PBS+Directives+Explained
# https://opus.nci.org.au/spaces/Help/pages/184647980/PyTorch
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
LAUNCH_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/elastic_ddp_nccl.sh 

# Set execute permission
chmod u+x ${LAUNCH_SCRIPT}

# Run PyTorch application
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  pbsdsh -n $inode ${LAUNCH_SCRIPT} ${NNODES} ${PROC_PER_NODE} ${MASTER_ADDR} &
done
wait