#!/bin/bash
# Which GENE run dirs each Slurm job has, and which dirs nothing has run.
#
#   bash NSTXU_lithium_study/scripts/gene_jobs.sh                  # every NSTX_* run dir
#   bash NSTXU_lithium_study/scripts/gene_jobs.sh 'NSTX_*_rc*_*'   # a name glob: reschecks
#   bash NSTXU_lithium_study/scripts/gene_jobs.sh 'NSTX_*_rcquiv9u_*'
#   bash NSTXU_lithium_study/scripts/gene_jobs.sh -a 'NSTX_*_rc*_*'   # done dirs too
#
# One block per job ID, from squeue (queued, running) and from the .claim_<jobid>
# marks interactive_corners.sh leaves in each run dir it starts (so ended
# interactive jobs show too): the job's open run dirs. A batch job's run dir is
# the dir it was submitted from (%Z) or its script's dir (%o). Done dirs, and
# ended jobs whose dirs are all done, are hidden unless -a (--all): then each dir
# is marked done or open and an all-done ended job is one line. Then every dir no
# job has taken.
# Done is interactive_corners.sh's check: .interactive_ran, or a complete
# scanfilesNNNN (omega for every parameters).
#
# bashrc:
#   genejobs() { bash ~/tools/NSTXU_lithium_study/scripts/gene_jobs.sh "$@"; }

set -u
GENEDIR=${GENEDIR:-$HOME/simulators/GENE3_gpu}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PAT='NSTX_*'
ALL=0
for a in "$@"; do
    case $a in
        -a|--all) ALL=1 ;;
        *) PAT=$a ;;
    esac
done

DIRS=()
for d in "$GENEDIR"/$PAT/; do
    d=${d%/}
    [ -f "$d/parameters" ] && DIRS+=("$d")
done
[ ${#DIRS[@]} -gt 0 ] || { echo "no run dirs matching $GENEDIR/$PAT"; exit 1; }

declare -A DONE STATE WORKDIR KIND JOBDIRS TAKEN
while read -r d; do
    DONE[$d]=1
done < <(LIST_ONLY=1 bash "$HERE/interactive_corners.sh" "${DIRS[@]}" 2>/dev/null \
         | awk '$1 == "done" {print $2}')

# queued and running jobs; a batch job's run dir is its submit dir or its script's dir
while IFS='|' read -r id st wd cmd; do
    STATE[$id]=$st
    WORKDIR[$id]=$wd
    for p in "${wd%/}" "$(dirname "$cmd")"; do
        if [ -f "$p/parameters" ]; then
            KIND[$id]=batch
            JOBDIRS[$id]+="$p"$'\n'
            TAKEN[$p]=1
            break
        fi
    done
done < <(squeue -u "${USER:-$(whoami)}" -h -o "%i|%T|%Z|%o")

# interactive_corners.sh claims, any job, running or ended
for d in "${DIRS[@]}"; do
    for c in "$d"/.claim_*; do
        [ -e "$c" ] || continue
        id=${c##*.claim_}
        KIND[$id]=${KIND[$id]:-interactive}
        JOBDIRS[$id]+="$d"$'\n'
        TAKEN[$d]=1
    done
done

ndone=0
for d in "${DIRS[@]}"; do [ -n "${DONE[$d]:-}" ] && ndone=$((ndone + 1)); done
echo "${#DIRS[@]} run dirs matching $GENEDIR/$PAT: $ndone done"

hidden=0
for id in $(printf '%s\n' "${!JOBDIRS[@]}" "${!STATE[@]}" | sort -un); do
    st=${STATE[$id]:-ENDED}
    mapfile -t jd < <(printf '%s' "${JOBDIRS[$id]:-}" | sed '/^$/d' | sort -u)
    n=0
    for d in "${jd[@]}"; do [ -n "${DONE[$d]:-}" ] && n=$((n + 1)); done
    alldone=0
    [ "$st" = ENDED ] && [ ${#jd[@]} -gt 0 ] && [ "$n" -eq ${#jd[@]} ] && alldone=1
    if [ "$alldone" = 1 ] && [ "$ALL" = 0 ]; then
        hidden=$((hidden + 1))
        continue
    fi
    printf '\n%-10s %-9s %-12s %3d dir(s), %d done\n' "$id" "$st" "${KIND[$id]:-other}" \
        "${#jd[@]}" "$n"
    if [ ${#jd[@]} -eq 0 ]; then
        echo "    no run dir matching '$PAT'; workdir ${WORKDIR[$id]:-?}"
        continue
    fi
    [ "$alldone" = 1 ] && continue
    for d in "${jd[@]}"; do
        if [ -n "${DONE[$d]:-}" ]; then
            [ "$ALL" = 1 ] && printf '    done %s\n' "${d##*/}"
        else
            printf '    open %s\n' "${d##*/}"
        fi
    done
done
[ "$hidden" -gt 0 ] && echo -e "\n($hidden ended job(s) with every dir done hidden; -a shows them)"

echo -e "\nnot run (not done, no job, no claim):"
n=0
for d in "${DIRS[@]}"; do
    [ -n "${DONE[$d]:-}" ] || [ -n "${TAKEN[$d]:-}" ] && continue
    echo "    ${d##*/}"
    n=$((n + 1))
done
[ "$n" -gt 0 ] || echo "    none"
