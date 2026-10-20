#!/bin/bash
# Run corner_gene_scan.py --gpu run dirs inside one interactive allocation.
#
# The allocation is split into lanes of NODES_PER nodes; each lane takes the
# next unclaimed run dir, runs its scanscript pinned to its own nodes (srun -w),
# and moves on, so corners of different lengths still keep every lane busy.
# Same environment and srun line as the template's submit_gpu.cmd.
#
#   salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m2116_g --gpus-per-node=4
#   bash NSTXU_lithium_study/scripts/interactive_corners.sh ~/simulators/GENE3_gpu/NSTX_129015_corner_x0_0.8425_*/
#
# A run dir with .interactive_ran is skipped (scanscript exited 0: converged or
# hit timelim; check the output). Delete it to rerun. A dir another lane has
# claimed in this job is skipped; a new allocation re-claims unfinished dirs.
# Log per run dir: interactive.log.

set -u
NODES_PER=${NODES_PER:-2}      # 8 ky x 1 GPU each = 2 nodes
PPN=4                          # one rank per A100
NP=$((NODES_PER * PPN))        # = n_parallel_sims = --mps

source ~/simulators/GENE3_gpu/env_gpu.sh
export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export MEMORY_PER_CORE=40000
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1

[ -n "${SLURM_JOB_ID:-}" ] || { echo "not inside an allocation; salloc first" >&2; exit 1; }
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NLANES=$((${#HOSTS[@]} / NODES_PER))
[ "$NLANES" -ge 1 ] || { echo "need >= $NODES_PER nodes, have ${#HOSTS[@]}" >&2; exit 1; }
DIRS=("$@")
echo "${#DIRS[@]} run dirs, $NLANES lane(s) of $NODES_PER nodes: ${HOSTS[*]}"

lane() {
    local i=$1 d nodes
    nodes=$(IFS=,; echo "${HOSTS[*]:$((i * NODES_PER)):$NODES_PER}")
    for d in "${DIRS[@]}"; do
        d=${d%/}
        [ -e "$d/.interactive_ran" ] && continue
        mkdir "$d/.claim_$SLURM_JOB_ID" 2>/dev/null || continue
        echo "[lane $i $nodes] $(date +%T) start $d"
        if (cd "$d" && ./scanscript --np $NP --ppn $PPN --mps $NP \
                --syscall="srun -l -K -N $NODES_PER -w $nodes -n $NP --gpus-per-task=1 --ntasks-per-node=$PPN -c 32 $HOME/simulators/GENE3_gpu/bin/gene_perlmutter" \
                > interactive.log 2>&1); then
            touch "$d/.interactive_ran"
            echo "[lane $i] $(date +%T) done  $d"
        else
            echo "[lane $i] $(date +%T) FAILED $d (see $d/interactive.log)"
        fi
    done
}

for ((i = 0; i < NLANES; i++)); do
    lane "$i" &
done
wait
