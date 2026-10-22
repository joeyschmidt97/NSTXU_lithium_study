#!/bin/bash
# Run corner_gene_scan.py --gpu run dirs inside one interactive allocation.
#
# The allocation is split into lanes of NODES_PER nodes; each lane takes the
# next unclaimed run dir, runs its scanscript pinned to its own nodes (srun -w),
# and moves on, so corners of different lengths still keep every lane busy.
# Same environment as the template's submit_gpu.cmd. Every rank sees all 4 GPUs
# of its node (--gpus-per-node, --gpu-bind=none) and GENE picks one by local
# rank: with --gpus-per-task=1 each rank sees only its own GPU and GPU-aware
# MPI aborts (cuIpcOpenMemHandle: invalid argument) on the first exchange.
#
#   salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m2116_g --gpus-per-node=4
#   bash NSTXU_lithium_study/scripts/interactive_corners.sh ~/simulators/GENE3_gpu/NSTX_129015_corner_x0_0.8425_*/
#
# A run dir with .interactive_ran is skipped (GENE finished: converged or hit
# timelim; check the output). Delete it to rerun. A dir another lane has
# claimed in this job is skipped; a new allocation re-claims unfinished dirs.
# Log per run dir: interactive.log.
#
# Finished means GENE printed "Total wallclock time for GENE" during this
# launch; scanscript exits 0 even when srun was refused, so its exit code
# alone would mark never-run dirs done. A lane stops taking dirs once the
# allocation has less time left than the next dir's timelim + START_MARGIN,
# so no corner is started only to be killed (and leave a partial scanfiles).

# Ranks per run dir come from its parameters (n_parallel_sims x n_procs_sim),
# so --res-scale or a short --kymin list sizes itself; a dir needing more nodes
# than a lane has is skipped.

set -u
NODES_PER=${NODES_PER:-2}      # lane size; 8 ky x 1 GPU each = 2 nodes
PPN=4                          # one rank per A100
START_MARGIN=${START_MARGIN:-300}   # s on top of timelim for startup and output
DONE_LINE="Total wallclock time for GENE"
FAIL_RE="aborting job|MPICH ERROR|CANCELLED AT|Exited with exit code [1-9]"

param() { sed -n "s/^[ \t]*$2[ \t]*=[ \t]*\([0-9]*\).*/\1/p" "$1/parameters" | head -1; }
diagdir() { sed -n "s/^[ \t]*diagdir[ \t]*=[ \t]*'\([^']*\)'.*/\1/p" "$1/parameters" | head -1; }

# Seconds left in this allocation (squeue %L: [D-]HH:MM:SS or MM:SS).
time_left() {
    local l d=0 h=0 m=0 s=0
    l=$(squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null)
    [[ $l =~ ^([0-9]+-)?([0-9:]+)$ ]] || { echo 0; return; }
    [ -n "${BASH_REMATCH[1]}" ] && d=${BASH_REMATCH[1]%-}
    IFS=: read -r -a p <<< "${BASH_REMATCH[2]}"
    case ${#p[@]} in
        3) h=${p[0]}; m=${p[1]}; s=${p[2]} ;;
        2) m=${p[0]}; s=${p[1]} ;;
        1) s=${p[0]} ;;
    esac
    echo $((10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s))
}

# scanscript points the run dir's diagdir at the scanfilesNNNN it opens, so a
# diagdir different from the one before launch ($2) holds this launch's
# geneerror.log; unchanged means scanscript never started GENE. Finished =
# GENE's wallclock line and no MPI/srun abort in that log or interactive.log.
finished() {
    local dd logs=("$1/interactive.log")
    dd=$(diagdir "$1")
    [ "$dd" != "$2" ] || return 1
    logs+=("$dd/geneerror.log")
    cat "${logs[@]}" 2>/dev/null | grep -qE "$FAIL_RE" && return 1
    cat "${logs[@]}" 2>/dev/null | grep -q "$DONE_LINE"
}

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
    local i=$1 d nodes np n mps need left before dd0
    for d in "${DIRS[@]}"; do
        d=${d%/}
        [ -e "$d/.interactive_ran" ] && continue
        need=$(($(param "$d" timelim) + START_MARGIN))
        left=$(time_left)
        if [ "$left" -lt "$need" ]; then
            echo "[lane $i] $(date +%T) stop: ${left}s left, next dir needs ${need}s"
            return
        fi
        mps=$(param "$d" n_parallel_sims)
        np=$((mps * $(param "$d" n_procs_sim)))
        n=$(((np + PPN - 1) / PPN))
        if [ "$n" -gt "$NODES_PER" ]; then
            echo "[lane $i] SKIP $d: $np ranks need $n nodes, lane has $NODES_PER (set NODES_PER)"
            continue
        fi
        mkdir "$d/.claim_$SLURM_JOB_ID" 2>/dev/null || continue
        nodes=$(IFS=,; echo "${HOSTS[*]:$((i * NODES_PER)):$n}")
        [ -f "$d/geneerror.log" ] && mv "$d/geneerror.log" "$d/geneerror.log.prev"
        dd0=$(diagdir "$d")
        echo "[lane $i $nodes] $(date +%T) start $d ($np ranks, $mps sims)"
        (cd "$d" && ./scanscript --np $np --ppn $PPN --mps $mps \
            --syscall="srun -l -K -N $n -w $nodes -n $np --gpus-per-node=$PPN --gpu-bind=none --ntasks-per-node=$PPN -c 32 $HOME/simulators/GENE3_gpu/bin/gene_perlmutter" \
            > interactive.log 2>&1)
        if finished "$d" "$dd0"; then
            touch "$d/.interactive_ran"
            echo "[lane $i] $(date +%T) done  $d"
        else
            echo "[lane $i] $(date +%T) NOT FINISHED $d (see interactive.log, $(diagdir "$d")/geneerror.log)"
        fi
    done
}

for ((i = 0; i < NLANES; i++)); do
    lane "$i" &
done
wait
