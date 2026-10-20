#!/bin/bash -l
#SBATCH -C gpu
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH -J GENE_gpu
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -q debug
#SBATCH --exclusive
#SBATCH -A m2116_g

source ~/simulators/GENE3_gpu/env_gpu.sh

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MEMORY_PER_CORE=40000
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

set -x
./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall="srun -l -K -n $SLURM_NTASKS --gpus-per-task=1 --ntasks-per-node=$SLURM_NTASKS_PER_NODE -c 32 ~/simulators/GENE3_gpu/bin/gene_perlmutter"
set +x