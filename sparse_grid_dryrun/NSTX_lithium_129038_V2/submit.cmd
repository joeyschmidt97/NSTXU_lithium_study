#!/bin/bash -l
#SBATCH -C cpu
#SBATCH -t 00:30:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=128
#SBATCH -c 2
#SBATCH -J GENE
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#SBATCH -q debug
#SBATCH -A m2116

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

./scanscript --np 384 --ppn 128 --mps 128 \
  --syscall='srun -l -K -n 384 --cpu_bind=cores ./gene_perlmutter'
