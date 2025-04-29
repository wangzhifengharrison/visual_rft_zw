#!/bin/bash
#PBS -P kf09
#PBS -q gpuvolta
#PBS -l ngpus=8
#PBS -l ncpus=96
#PBS -l mem=256GB     
#PBS -l walltime=00:30:00
#PBS -l wd
#PBS -j oe
#PBS -o logs/run_log_$PBS_JOBID.log
cd /scratch/kf09/zw4360/Visual-RFT

# Set up logging directory
mkdir -p logs

# Generate a unique port based on the PBS job ID to avoid conflicts
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

# Launch tasks with static port
export PBS_LAUNCHER=/opt/pbs/default/bin/pbsdsh
$PBS_LAUNCHER -- /bin/bash -c "cd $PBS_O_WORKDIR && env PBS_NODEFILE=\"$PBS_NODEFILE\" MASTER_ADDR=\"\$(head -n 1 $PBS_NODEFILE)\" MASTER_PORT=\"$MASTER_PORT\" PBS_O_HOSTFILE=\"$PBS_O_HOSTFILE\" PBS_O_VNODENUM=\"$PBS_O_VNODENUM\" bash run_2_26_04_2025.sh"

