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
# A run dir is skipped as done (GENE finished: converged or hit timelim; check
# the output) when it has .interactive_ran, or when any scanfilesNNNN (in the
# run dir or beside its diagdir on pscratch) holds a complete scan: an omega_NNNN for every parameters_NNNN and at least
# n_parallel_sims of them, with every kymin of the run dir's own input among
# its members (so a scan of another run dir in the same diagdir, e.g. the
# original VURS run beside a vurs_fill FILL run, does not count). GENE writes omega only when a run ends, so a scan
# killed by SLURM or ^C fails this; runs from sbatch or by hand pass it.
# Delete .interactive_ran and move the complete scanfiles aside to rerun. A dir
# another lane has claimed in this job is skipped; a new allocation re-claims
# unfinished dirs, and a relaunch in the same allocation (after ^C) releases
# this job's claims on dirs not done. Log per run dir: interactive.log.
#
#   LIST_ONLY=1 bash NSTXU_lithium_study/scripts/interactive_corners.sh <dirs>
# prints which dirs are done and exits; no allocation needed. DRY_RUN=1 also
# prints the scanscript/srun call of each dir still to run, then exits.
#
# Finished means GENE printed "Total wallclock time for GENE" during this
# launch; scanscript exits 0 even when srun was refused, so its exit code
# alone would mark never-run dirs done. A lane stops taking dirs once the
# allocation has less time left than the next dir's timelim + START_MARGIN,
# so no corner is started only to be killed (and leave a partial scanfiles).

# Ranks per run dir come from its parameters (n_parallel_sims x n_procs_sim),
# so --res-scale or a short --kymin list sizes itself; a dir needing more nodes
# than a lane has is skipped.
#
# GPU packing (LANES_PER_GPU=L > 0): every A100 is L lanes instead of nodes
# being split into NODES_PER-node lanes, so an allocation of N nodes runs
# N x 4 x L run dirs at once, any mix of shots, radii and scalings. A packed
# dir runs all its ranks on its lane's one GPU: its srun step is --overlap
# (steps share the node), sees one device through CUDA_VISIBLE_DEVICES, and
# gets 128 / (4 L) hardware threads split over its ranks. L = 3 with 3-ky
# dirs puts 9 GENE ranks on each A100. MPS=1 starts the CUDA MPS daemon on
# every node first, so those ranks share the GPU concurrently instead of
# time-slicing it. A packed dir with more than MAX_PACKED_RANKS ranks is skipped.
#
#   LANES_PER_GPU=3 MPS=1 bash NSTXU_lithium_study/scripts/interactive_corners.sh <dirs>

set -u
NODES_PER=${NODES_PER:-2}      # lane size; 8 ky x 1 GPU each = 2 nodes
PPN=4                          # one rank per A100
LANES_PER_GPU=${LANES_PER_GPU:-0}       # > 0: GPU packing mode (header)
MPS=${MPS:-0}
MAX_PACKED_RANKS=${MAX_PACKED_RANKS:-16}
HW_THREADS=128                 # per Perlmutter GPU node
START_MARGIN=${START_MARGIN:-300}   # s on top of timelim for startup and output
DONE_LINE="Total wallclock time for GENE"
FAIL_RE="aborting job|MPICH ERROR|CANCELLED AT|Exited with exit code [1-9]"

param() { sed -n "s/^[ \t]*$2[ \t]*=[ \t]*\([0-9]*\).*/\1/p" "$1/parameters" | head -1; }
diagdir() { sed -n "s/^[ \t]*diagdir[ \t]*=[ \t]*'\([^']*\)'.*/\1/p" "$1/parameters" | head -1; }

# kymin values, one per line as %g, sorted: of run dir $1's input parameters
# (its !scanlist, else the single value), and of every member of scanfiles dir $1.
dir_ky() {
    local line
    line=$(grep -m1 -E "^[ \t]*kymin[ \t]*=" "$1/parameters" 2>/dev/null) || return 0
    if [[ $line == *'!scanlist:'* ]]; then line=${line#*!scanlist:}; else line=${line#*=}; line=${line%%!*}; fi
    echo "$line" | tr ',' '\n' | awk 'NF {printf "%g\n", $1}' | LC_ALL=C sort -u
}
scan_ky() {
    grep -h -m1 -E "^[ \t]*kymin[ \t]*=" "$1"/parameters_[0-9]* 2>/dev/null \
        | sed 's/^[^=]*=//; s/!.*//' | awk 'NF {printf "%g\n", $1}' | LC_ALL=C sort -u
}

# The scanfilesNNNN of run dir $1 holding a complete scan of its own kymin, if
# any (see header). GENE output sits under the diagdir in its parameters
# (pscratch), which scanscript rewrites to the scanfilesNNNN it opened; its
# parent holds them all, including scans of other run dirs sharing that
# diagdir (vurs_fill FILL runs, extra ky): a scan counts only if it holds
# every kymin of this run dir.
complete_scan() {
    local s np no mps base want
    mps=$(param "$1" n_parallel_sims)
    want=$(dir_ky "$1")
    base=$(diagdir "$1"); base=${base%/}
    [[ $(basename "$base") =~ ^scanfiles[0-9]+$ ]] && base=$(dirname "$base")
    for s in "$1"/scanfiles*/ ${base:+"$base"/scanfiles*/}; do
        [ -d "$s" ] || continue
        np=$(ls "$s" | grep -c '^parameters_[0-9]')
        no=$(ls "$s" | grep -c '^omega_[0-9]')
        if [ "$no" -gt 0 ] && [ "$no" -ge "$np" ] && [ "$no" -ge "${mps:-1}" ]; then
            if [ -n "$want" ] && \
               [ -n "$(LC_ALL=C comm -23 <(echo "$want") <(scan_ky "${s%/}"))" ]; then
                continue                 # another run dir's scan in the same diagdir
            fi
            echo "${s%/}"
            return 0
        fi
    done
    return 1
}
already_done() { [ -e "$1/.interactive_ran" ] || complete_scan "$1" > /dev/null; }

# "mps ranks nodes" of run dir $1: n_parallel_sims, x n_procs_sim, over PPN.
dir_size() {
    local mps np
    mps=$(param "$1" n_parallel_sims)
    np=$((mps * $(param "$1" n_procs_sim)))
    echo "$mps $np $(((np + PPN - 1) / PPN))"
}
# srun line scanscript launches GENE with: $1 nodes $2 (comma list), $3 ranks.
syscall() {
    echo "srun -l -K -N $1 -w $2 -n $3 --gpus-per-node=$PPN --gpu-bind=none --ntasks-per-node=$PPN -c 32 $HOME/simulators/GENE3_gpu/bin/gene_perlmutter"
}
# Packed: node $1, ranks $2, GPU index $3; env runs inside the task, after
# Slurm sets CUDA_VISIBLE_DEVICES, so GENE sees only that GPU (as device 0).
syscall_packed() {
    local c=$((HW_THREADS / (PPN * LANES_PER_GPU) / $2))
    [ "$c" -ge 1 ] || c=1
    echo "srun -l -K --overlap -N 1 -w $1 -n $2 -c $c --gpus-per-node=$PPN --gpu-bind=none env CUDA_VISIBLE_DEVICES=$3 $HOME/simulators/GENE3_gpu/bin/gene_perlmutter"
}

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

DIRS=("$@")
NDONE=0
for d in "${DIRS[@]}"; do
    d=${d%/}
    if [ -e "$d/.interactive_ran" ]; then
        echo "  done  $d (.interactive_ran)"; NDONE=$((NDONE + 1))
    elif s=$(complete_scan "$d"); then
        echo "  done  $d ($s)"; NDONE=$((NDONE + 1))
    fi
done
echo "$NDONE already done (skipped), $((${#DIRS[@]} - NDONE)) to run"
[ -z "${LIST_ONLY:-}" ] || exit 0

# DRY_RUN=1: the scanscript call each dir still to run would get, then exit.
# Lanes take dirs as they free up, so the nodes are shown as <lane nodes>.
if [ -n "${DRY_RUN:-}" ]; then
    for d in "${DIRS[@]}"; do
        d=${d%/}
        already_done "$d" && continue
        read -r mps np n < <(dir_size "$d")
        if [ "$LANES_PER_GPU" -gt 0 ]; then
            if [ "$np" -gt "$MAX_PACKED_RANKS" ]; then
                echo "  SKIP  $d: $np ranks, packed lanes take <= $MAX_PACKED_RANKS"
                continue
            fi
            echo "  run   $d (timelim $(param "$d" timelim) s, $np ranks on one GPU)"
            echo "        cd $d && ./scanscript --np $np --ppn $np --mps $mps --syscall=\"$(syscall_packed "<node>" "$np" "<gpu>")\" > interactive.log 2>&1"
            continue
        fi
        if [ "$n" -gt "$NODES_PER" ]; then
            echo "  SKIP  $d: $np ranks need $n nodes, lane has $NODES_PER (set NODES_PER)"
            continue
        fi
        echo "  run   $d (timelim $(param "$d" timelim) s, $n nodes)"
        echo "        cd $d && ./scanscript --np $np --ppn $PPN --mps $mps --syscall=\"$(syscall "$n" "<lane nodes>" "$np")\" > interactive.log 2>&1"
    done
    exit 0
fi

[ -n "${SLURM_JOB_ID:-}" ] || { echo "not inside an allocation; salloc first" >&2; exit 1; }
mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "$LANES_PER_GPU" -gt 0 ]; then
    NLANES=$((${#HOSTS[@]} * PPN * LANES_PER_GPU))
    echo "${#DIRS[@]} run dirs, $NLANES packed lane(s), $LANES_PER_GPU per GPU: ${HOSTS[*]}"
    if [ "$MPS" = 1 ]; then
        # One daemon per node, held in a background step that sleeps: Slurm kills
        # a step's processes when it ends, daemon included. Killed on exit.
        MPS_PIDS=()
        for h in "${HOSTS[@]}"; do
            srun --overlap -N 1 -n 1 -w "$h" --gpus-per-node=$PPN --gpu-bind=none bash -c \
                "nvidia-cuda-mps-control -d && echo '  MPS daemon up on $h' && exec sleep infinity" &
            MPS_PIDS+=($!)
        done
        trap 'kill "${MPS_PIDS[@]}" 2>/dev/null' EXIT
        sleep 10
    fi
else
    NLANES=$((${#HOSTS[@]} / NODES_PER))
    [ "$NLANES" -ge 1 ] || { echo "need >= $NODES_PER nodes, have ${#HOSTS[@]}" >&2; exit 1; }
    echo "${#DIRS[@]} run dirs, $NLANES lane(s) of $NODES_PER nodes: ${HOSTS[*]}"
fi
# A claim of this job left by an earlier launch in the same allocation (^C)
# would skip its dir silently; that launch is gone, so release it.
for d in "${DIRS[@]}"; do
    d=${d%/}
    if [ -d "$d/.claim_$SLURM_JOB_ID" ] && ! already_done "$d"; then
        rmdir "$d/.claim_$SLURM_JOB_ID" && echo "  released stale claim  $d"
    fi
done

lane() {
    local i=$1 d nodes gpu np n mps need left before dd0
    for d in "${DIRS[@]}"; do
        d=${d%/}
        already_done "$d" && continue
        need=$(($(param "$d" timelim) + START_MARGIN))
        left=$(time_left)
        if [ "$left" -lt "$need" ]; then
            echo "[lane $i] $(date +%T) stop: ${left}s left, next dir needs ${need}s"
            return
        fi
        read -r mps np n < <(dir_size "$d")
        if [ "$LANES_PER_GPU" -gt 0 ]; then
            if [ "$np" -gt "$MAX_PACKED_RANKS" ]; then
                echo "[lane $i] SKIP $d: $np ranks, packed lanes take <= $MAX_PACKED_RANKS"
                continue
            fi
        elif [ "$n" -gt "$NODES_PER" ]; then
            echo "[lane $i] SKIP $d: $np ranks need $n nodes, lane has $NODES_PER (set NODES_PER)"
            continue
        fi
        mkdir "$d/.claim_$SLURM_JOB_ID" 2>/dev/null || continue
        [ -f "$d/geneerror.log" ] && mv "$d/geneerror.log" "$d/geneerror.log.prev"
        dd0=$(diagdir "$d")
        if [ "$LANES_PER_GPU" -gt 0 ]; then
            # lane i: node i / (4 L), GPU (i / L) mod 4
            nodes=${HOSTS[$((i / (PPN * LANES_PER_GPU)))]}
            gpu=$(((i / LANES_PER_GPU) % PPN))
            echo "[lane $i $nodes gpu $gpu] $(date +%T) start $d ($np ranks, $mps sims)"
            (cd "$d" && ./scanscript --np $np --ppn $np --mps $mps \
                --syscall="$(syscall_packed "$nodes" "$np" "$gpu")" > interactive.log 2>&1)
        else
            nodes=$(IFS=,; echo "${HOSTS[*]:$((i * NODES_PER)):$n}")
            echo "[lane $i $nodes] $(date +%T) start $d ($np ranks, $mps sims)"
            (cd "$d" && ./scanscript --np $np --ppn $PPN --mps $mps \
                --syscall="$(syscall "$n" "$nodes" "$np")" > interactive.log 2>&1)
        fi
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
