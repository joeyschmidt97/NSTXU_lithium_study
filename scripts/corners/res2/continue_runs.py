"""Rerun the ky GENE did not converge in finished run dirs, longer, into the same diagdir.

For every run dir given (res_check NSTX_*_rc<code>_* dirs, corner_gene_scan
dirs such as NSTX_*_refkyhi_*), the kymin of its input (!scanlist or single
value) are checked against every scanfilesNNNN under its diagdir: a ky counts
as converged when some scan member at that ky has an omega_NNNN with gamma or
omega nonzero (GENE writes 0, 0 when it stops unconverged, e.g. at timelim).
The ky left over get a new run dir beside the old one,

    <run dir>_cont/   parameters of <run dir> with kymin = the unconverged ky,
                      n_parallel_sims = their number, timelim from --walltime,
                      diagdir = the same diagdir (its parent, not the scanfilesNNNN
                      scanscript pointed it at), &scan dropped
                      .continue_after   number of the last scanfilesNNNN there now

so the rerun opens the next scanfilesNNNN in the same diagdir. The readers
already merge scanfiles dirs by ky (plot_corners.gene_corner: GENE converged
beats a field fit, else the later dir wins), so update_results and
plot_res_check pick the new values up under the same batch code with nothing
else to change. interactive_corners.sh only counts scanfiles after
.continue_after as this dir's own, so the old complete scan does not make it
look done.

The runs restart from t = 0: no checkpoint is read (a scan's checkpoints are
per member and per scanfiles dir, and the new scan numbers its members afresh).
The gain is wall time and fewer dirs packed per GPU.

Each _cont dir also gets the run dir's submit_gpu.cmd (or submit.cmd) resized
as corner_gene_scan does it: -t --walltime, -q --qos, -N for one rank set per
ky (ceil(n_ky * n_procs_sim / ntasks-per-node)), --mps n_ky, -J <dir name>.
With --submit each is sbatch'ed as its own job, unpacked (one A100 per ky);
a job ends as soon as all its ky have converged.

    python -m NSTXU_lithium_study.scripts.corners.res2.continue_runs --list ~/simulators/GENE3_gpu/NSTX_*_rc*_*/
    python -m NSTXU_lithium_study.scripts.corners.res2.continue_runs ~/simulators/GENE3_gpu/NSTX_*_rc*_*/ \\
        --exclude nv0x2_x0_0.947 --walltime 03:00:00 --submit

or, packed, in an interactive allocation (--walltime under the allocation's):

    LANES_PER_GPU=2 MPS=1 bash NSTXU_lithium_study/scripts/interactive_corners.sh ~/simulators/GENE3_gpu/NSTX_*_cont/
"""

from __future__ import annotations

import glob
import os
import re
import math
import shutil
import subprocess
import sys

from NSTXU_lithium_study.scripts import corner_gene_scan as C

SUFFIX = "_cont"
MARKER = ".continue_after"
WALLTIME = "03:40:00"
SUBMITS = (C.GPU_SUBMIT, C.SUBMIT)              # first one present in the run dir is used
JOB = ".job_id"                                 # written on sbatch; an existing _cont dir
                                                # without it is submitted by a later --submit


def input_ky(text: str) -> list[float]:
    """kymin of a parameters text: its !scanlist, else the single value."""
    line = next(l for l in text.splitlines() if re.match(r"^[ \t]*kymin[ \t]*=", l))
    vals = line.split("!scanlist:", 1)[1] if "!scanlist:" in line else \
        line.split("=", 1)[1].split("!", 1)[0]
    return sorted({round(float(v), 6) for v in vals.split(",") if v.strip()})


def base_diagdir(text: str) -> str:
    d = C.read_param(text, "diagdir").rstrip("/")
    return os.path.dirname(d) if re.fullmatch(r"scanfiles\d+", os.path.basename(d)) else d


def scanfiles(base: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(base, "scanfiles[0-9]*")) if os.path.isdir(d))


def converged_ky(base: str) -> set[float]:
    """ky of every scan member under *base* whose omega file holds a nonzero value."""
    out = set()
    for s in scanfiles(base):
        for par in glob.glob(os.path.join(s, "parameters_[0-9]*")):
            om = os.path.join(s, "omega_" + par.rsplit("_", 1)[1])
            try:
                with open(par) as fh:
                    k = round(float(C.read_param(fh.read(), "kymin")), 6)
                with open(om) as fh:
                    vals = [float(v) for v in fh.read().split()]
            except (OSError, ValueError):
                continue
            if len(vals) >= 3 and (vals[1] != 0 or vals[2] != 0):
                out.add(k)
    return out


def cont_parameters(text: str, ky: list[float], base: str, timelim: int) -> str:
    text = C.drop_namelist(text, "scan")
    text = C.set_param(text, "kymin", f"{ky[0]:g}" + (f"  !scanlist: {C.scanlist(ky)}"
                                                       if len(ky) > 1 else ""))
    text = C.set_param(text, "n_parallel_sims", str(len(ky)))
    text = C.set_param(text, "diagdir", f"'{base}'")
    return C.set_param(text, "timelim", str(timelim))


def cont_submit(template: str, n_ky: int, n_procs_sim: int, walltime: str, qos: str,
                name: str) -> str:
    """The run dir's submit script resized for *n_ky* ky, one n_procs_sim set each."""
    ppn = int(re.search(r"--ntasks-per-node=(\d+)", template).group(1))
    ntasks = n_ky * n_procs_sim
    text = C.corner_submit(template, math.ceil(ntasks / ppn), ntasks, walltime, qos, n_ky)
    # GENE needs exactly n_ky * n_procs_sim ranks; $SLURM_NTASKS would be N * ppn
    text = re.sub(r"\$SLURM_NTASKS(?!_)", str(ntasks), text)
    return re.sub(r"^#SBATCH -J .*$", f"#SBATCH -J {name[-60:]}", text, flags=re.M)


def submit(out: str, jobs: list) -> bool:
    """sbatch the submit script of *out*, record the job id in out/.job_id."""
    sub = next((f for f in SUBMITS if os.path.isfile(os.path.join(out, f))), None)
    if not sub:
        print(f"    no {' or '.join(SUBMITS)} in {out}, NOT submitted")
        return False
    r = subprocess.run(["sbatch", sub], cwd=out, check=True, capture_output=True, text=True)
    jobs.append(r.stdout.split()[-1])
    with open(os.path.join(out, JOB), "w") as fh:
        print(jobs[-1], file=fh)
    print(f"    job {jobs[-1]}  {os.path.basename(out)}")
    return True


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", help="run dirs (any *_cont among them is ignored)")
    ap.add_argument("--walltime", default=WALLTIME,
                    help=f"sets GENE timelim = walltime - {C.TIMELIM_MARGIN} s [{WALLTIME}]")
    ap.add_argument("--exclude", nargs="+", default=[], metavar="TEXT",
                    help="leave out run dirs whose path contains any of these")
    ap.add_argument("--force", action="store_true", help="rewrite existing *_cont dirs")
    ap.add_argument("--qos", default=C.QOS, help=f"SLURM -q of the written submit script [{C.QOS}]")
    ap.add_argument("--submit", action="store_true",
                    help="sbatch each written *_cont dir as its own job")
    ap.add_argument("--list", action="store_true", help="print what would be rerun, write nothing")
    args = ap.parse_args()
    timelim = C.walltime_seconds(args.walltime) - C.TIMELIM_MARGIN

    todo_dirs, n_ky, jobs = 0, 0, []
    for d in sorted({os.path.normpath(p) for p in args.dirs}):
        name = os.path.basename(d)
        if name.endswith(SUFFIX) or any(x in d for x in args.exclude):
            continue
        par = os.path.join(d, "parameters")
        if not os.path.isfile(par):
            print(f"  no parameters: {d}")
            continue
        with open(par) as fh:
            text = fh.read()
        base = base_diagdir(text)
        runs = scanfiles(base)
        if not runs:
            print(f"  never ran (no scanfiles in {base}): {name}")
            continue
        want = input_ky(text)
        conv = converged_ky(base)
        todo = [k for k in want if k not in conv]
        if not todo:
            continue
        todo_dirs, n_ky = todo_dirs + 1, n_ky + len(todo)
        print(f"  {name}: ky {C.scanlist(todo)} of {C.scanlist(want)} unconverged")
        if args.list:
            continue
        out = d + SUFFIX
        job_file = os.path.join(out, JOB)
        if os.path.exists(out) and not args.force:
            if args.submit and not os.path.exists(job_file) and submit(out, jobs):
                continue
            print(f"    {out} exists, NOT rewritten (--force)"
                  + (" or submitted" if os.path.exists(job_file) else ""))
            continue
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "parameters"), "w") as fh:
            fh.write(cont_parameters(text, todo, base, timelim))
        with open(os.path.join(out, MARKER), "w") as fh:
            fh.write(re.sub(r"\D", "", os.path.basename(runs[-1])) + "\n")
        sub = next((f for f in SUBMITS if os.path.isfile(os.path.join(d, f))), None)
        if sub:                                   # sbatch-only; interactive_corners ignores it
            with open(os.path.join(d, sub)) as fh:
                cmd = cont_submit(fh.read(), len(todo), int(C.read_param(text, "n_procs_sim")),
                                  args.walltime, args.qos, os.path.basename(out))
            with open(os.path.join(out, sub), "w") as fh:
                fh.write(cmd)
            os.chmod(os.path.join(out, sub), 0o755)
        for link in C.LINKS:
            src, dst = os.path.join(d, link), os.path.join(out, link)
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
            elif os.path.exists(src):
                shutil.copy2(src, dst)
        if args.submit:
            submit(out, jobs)
    print(f"\n{todo_dirs} run dir(s), {n_ky} ky to rerun"
          + ("" if args.list else f" -> *{SUFFIX}/, timelim {timelim} s")
          + (f", {len(jobs)} job(s) submitted ({args.walltime} on {args.qos})" if jobs else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
