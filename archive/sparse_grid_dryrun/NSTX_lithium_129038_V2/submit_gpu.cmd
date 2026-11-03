#!/bin/bash -l
#SBATCH -C gpu                     # use GPU partition; set gpu&hbm80g for 80GB
#SBATCH -t 12:00:00                # wallclock limit (currently, 24 hrs max)
#SBATCH -N 1                       # number of nodes
#SBATCH --ntasks-per-node=4        # set to 4 for GPUs, 64 for CPUs only. Additional 2 hyperthreads available
#SBATCH -c 32                      # cpus per task, 32 if GPU and no OPENMP, 2 if full CPU. Adjust accordingly
#SBATCH --gpus-per-task=1          # GENE GPU usage
#SBATCH --gpu-bind=none
#SBATCH -J GENE
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -q regular
#SBATCH --exclusive
##set specific account to charge (must end with _g to use GPUs)
#SBATCH -A <myaccount>

export SLURM_CPU_BIND="cores"

## set openmp threads
export OMP_NUM_THREADS=1

## do not use file locking for hdf5
export HDF5_USE_FILE_LOCKING=FALSE

## 40GB cards, adjust if using 80GB
export MEMORY_PER_CORE=40000

## turn cuda aware on
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

set -x
# run GENE
srun -l -K -n $SLURM_NTASKS ./gene_perlmutter

# run scanscript
#./scanscript --np $SLURM_NTASKS --ppn SLURM_NTASKS_PER_NODE --mps 4 --syscall='srun -l -K -n $SLURM_NTASKS --gpus-per-task=1 --ntasks-per-node=4 --gpus-per-node 4 ./gene_perlmutter'

set +x
