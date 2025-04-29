#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=8
#PBS -l ncpus=96
#PBS -l mem=256GB     
#PBS -l walltime=00:03:00
#PBS -l wd
#PBS -j oe
#PBS -o logs/run_log_$PBS_JOBID.log
cd /scratch/kf09/zw4360/Visual-RFT

# Set up logging directory
mkdir -p logs

if [ -n "$PBS_JOBID" ]; then
  JOB_NUM=$(echo $PBS_JOBID | cut -d '.' -f1)
  export MASTER_PORT=$(( 20000 + JOB_NUM % 10000 ))  # 20000-29999
else
  export MASTER_PORT=12345  # Fallback
fi


echo "=== Cluster Setup === "
echo $(cat $PBS_NODEFILE | head -n 1)
uniq "$PBS_NODEFILE"


# Build an array of unique hostnames
mapfile -t UNIQUE_NODES < <(awk '!seen[$0]++' "$PBS_NODEFILE")
NNODES=${#UNIQUE_NODES[@]}

echo "Detected NNODES=$NNODES"

MASTER_ADDR=$(head -n1 $PBS_NODEFILE)
export MASTER_ADDR

LAUNCH_SCRIPT=/scratch/kf09/zw4360/Visual-RFT/run_2_29_04_2025.sh

# Set execute permission
chmod u+x ${LAUNCH_SCRIPT}

# Run PyTorch application
for inode in $(seq 1 $PBS_NCI_NCPUS_PER_NODE $PBS_NCPUS); do
  /opt/pbs/default/bin/pbsdsh -n $inode ${LAUNCH_SCRIPT} ${NNODES} ${MASTER_ADDR} ${MASTER_PORT} ${PBS_NODEFILE} &
done
wait
