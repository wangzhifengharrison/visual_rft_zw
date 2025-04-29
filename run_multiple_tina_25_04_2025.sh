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

echo "=== Cluster Setup ==="
echo "Allocated nodes:"
uniq "$PBS_NODEFILE"

echo "=== Node List ==="
cat "$PBS_NODEFILE"

# Load modules and activate conda

source /scratch/kf09/zw4360/miniconda3/etc/profile.d/conda.sh
conda activate Visual-RFT
module load cuda/12.5.1
module load gcc/12.2.0
# Launch across all nodes using pdsh
export PBS_LAUNCHER=/opt/pbs/default/bin/pbsdsh
$PBS_LAUNCHER -- /bin/bash -c "cd $PBS_O_WORKDIR && env PBS_NODEFILE=\"$PBS_NODEFILE\" PBS_O_HOSTFILE=\"$PBS_O_HOSTFILE\" PBS_O_VNODENUM=\"$PBS_O_VNODENUM\" bash run_multiple_tina2_25_04_2025.sh"

mkdir -p logs
exec > logs/log_node${PBS_O_VNODENUM}_$(hostname).out 2>&1

echo "=== tina.sh on $(hostname) ==="
echo "PBS_NODEFILE   = '$PBS_NODEFILE'"
echo "PBS_O_HOSTFILE = '$PBS_O_HOSTFILE'"
echo "PBS_O_VNODENUM = '$PBS_O_VNODENUM'"