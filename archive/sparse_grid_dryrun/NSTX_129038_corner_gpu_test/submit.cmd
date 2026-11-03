#!/bin/bash -l
#SBATCH -C cpu                     # use CPU partition
#SBATCH -t 12:00:00                # wallclock limit
#SBATCH -N 1                       # total number of nodes, 2 Milan CPUs with 64 rank each
#SBATCH --ntasks-per-node=128      # 64 per CPU. Additional 2 hyperthreads disabled
#SBATCH -c 2                       # cpus per task, 2 if full CPU. Adjust accordingly
#SBATCH -J GENE
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -q regular
##set specific account to charge
#SBATCH -A <myaccount>

export SLURM_CPU_BIND="cores"

## set openmp threads
export OMP_NUM_THREADS=1

#do not use file locking for hdf5
export HDF5_USE_FILE_LOCKING=FALSE

set -x
# run GENE
srun -l -K -n $SLURM_NTASKS ./gene_perlmutter

# run scanscript
#./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall='srun -l -K -n $SLURM_NTASKS --cpu_bind=cores ./gene_perlmutter'

set +x
