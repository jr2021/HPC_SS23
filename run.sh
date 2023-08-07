#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:10:00
#SBATCH -J HPC_WITH_PYTHON
#SBATCH --mem=10GB
#SBATCH --export=ALL
#SBATCH --partition=single


module load devel/python/3.8.6_gnu_10.2
module load compiler/gnu/12.1
module load mpi/openmpi/4.1
module load devel/miniconda
source activate hpc

NP=$1

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
time mpirun -np $NP python Simulation.py