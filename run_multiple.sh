#!/bin/bash
module load openmpi

echo "PBS_NODEFILE points to: $PBS_NODEFILE"
cat $PBS_NODEFILE  # See which nodes you're using

mpirun -np 2 --map-by ppr:1:node --hostfile "$PBS_NODEFILE" \
    sh official_provided_train_scription_on_github.sh
