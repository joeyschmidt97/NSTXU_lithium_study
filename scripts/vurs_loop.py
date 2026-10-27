"""Closed VURS loop in one interactive GPU allocation: pick, build, run, repeat.

Each round:

    1. pick   ScanPointPicker (sparse_scanner's five axes: Te/ne ped and width
              scales, kymin) reads every GENE run under <diag-root>/x0_<x0>/,
              field fits included, and suggests --jobs GENE jobs: one set of
              profile scalings each and a --ky-per-job kymin scanlist; gamma
              is floored at 0 (every stable ky reads as 0) unless --no-clip
    2. build  ped_grid --cheasebs --case <tag> for each new equilibrium (one
              process per job, in parallel; a case with DONE.json is reused),
              then corner_gene_scan.build writes one GPU run dir per job, its
              output into the same x0_<x0>/ dir the picker reads
    3. run    interactive_corners.sh runs those dirs, NODES_PER nodes per lane

so round N+1 picks from everything round N wrote. Rounds are synchronous: a
round's GENE runs finish (converge or hit timelim) before the next pick, so
nothing is ever pending.

    salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m2116_g --gpus-per-node=4
    python -m NSTXU_lithium_study.scripts.vurs_loop --shot 129038 --rounds 4 \\
        --walltime 00:45:00 --ped-grid-args "<the flags the corners were built with>"
    python -m NSTXU_lithium_study.scripts.vurs_loop --shot 129038 --dry-run   # round 1 pick only

--jobs defaults to one per lane (allocation nodes / NODES_PER), so each round
is one wave. --ky-per-job must equal NODES_PER * 4 (one GPU per ky), which is
interactive_corners.sh's --np/--mps. --walltime sets GENE's timelim (minus
corner_gene_scan.TIMELIM_MARGIN), not a SLURM limit: rounds x (cheaseBS +
walltime) has to fit the allocation. A ky that hits timelim still counts
through its field fit and std.

Every suggested value (scalings and kymin) is rounded to 3 decimals
(0.9913 -> 0.991, 1.0338 -> 1.034, 0.06512 -> 0.065), which is what the case
name, the cheaseBS geomfile name and GENE's input hold, so the point suggested
is the point run.
Two jobs that round to the same case in one round would share an output dir;
the later one is dropped. A job whose cheaseBS fails is dropped and its points
held, so later rounds do not suggest it again.

Per round, <genedir>/<prefix>_<x0>_rounds/round<N>/ keeps the run table, the
picks (rounded, with their case tags) and the ped_grid logs.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys

import pandas as pd

from NSTXU_lithium_study.scripts import corner_gene_scan as C
from NSTXU_lithium_study.scripts.ped_grid import DONE, tag
from TPED.projects.GENE_pipelines.scripts.sparse_scanner import DEFAULT_AXES
from TPED.projects.GENE_pipelines.src.point_picker import ScanPointPicker
from TPED.projects.GENE_pipelines.src.scan_campaign import QoISpec

#: sparse_scanner axis -> ped_grid knob
KNOB = {"Te_ped_scale": "Teh", "Te_width_scale": "Tew",
        "ne_ped_scale": "neh", "ne_width_scale": "new"}
PPN = 4                                    # one rank per A100, as interactive_corners.sh
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interactive_corners.sh")


def time_left() -> int | None:
    """Seconds left in this allocation (squeue %L), or None outside one."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    days, _, hms = left.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def pick(diag_x0, n_jobs, ky_per_job, model, seed, hold, clip_min=0.):
    """(run table, job table with case tags) for one round."""
    picker = ScanPointPicker(model, DEFAULT_AXES, QoISpec("gamma", "max_over_ky"),
                             log_axes=["kymin"], clip_min=clip_min, decimals=3,
                             seed=seed)
    picker.fit(diag_x0, hold=hold)
    jobs = picker.suggest_jobs(n_jobs, ky_per_job, "kymin")
    jobs["tag"] = [tag({KNOB[a]: row[a] for a in KNOB}) for _, row in jobs.iterrows()]
    return picker.runs_, jobs


def cheasebs(tags, shot, ped_root, ped_args, logdir):
    """Run ped_grid --cheasebs for each tag without DONE.json, in parallel; failed tags."""
    savedir = os.path.dirname(os.path.normpath(ped_root))
    procs = {}
    for t in tags:
        if os.path.isfile(os.path.join(ped_root, t, DONE)):
            print(f"  cheaseBS {t}: DONE.json exists, reused", flush=True)
            continue
        cmd = [sys.executable, "-m", "NSTXU_lithium_study.scripts.ped_grid", "--shot", str(shot),
               "--cheasebs", "--no-plots", "--savedir", savedir, "--case", t, *ped_args]
        log = open(os.path.join(logdir, f"ped_grid_{t}.log"), "w")
        print(f"  cheaseBS {t}: {shlex.join(cmd)}", flush=True)
        procs[t] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log)
    failed = []
    for t, (proc, log) in procs.items():
        proc.wait()
        log.close()
        if proc.returncode or not os.path.isfile(os.path.join(ped_root, t, DONE)):
            failed.append(t)
            print(f"  cheaseBS {t}: FAILED (see {log.name})", flush=True)
    return failed


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", type=int, default=C.SHOT, choices=sorted(C.SHOTS))
    ap.add_argument("--x0", type=float, default=None, help="radius [Te-top Chapman radius]")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=None,
                    help="GENE jobs per round [allocation nodes / NODES_PER]")
    ap.add_argument("--ky-per-job", type=int, default=None,
                    help="kymin per job [NODES_PER * 4]; must equal NODES_PER * 4")
    ap.add_argument("--walltime", default="00:45:00",
                    help="per GENE run, HH:MM:SS; sets timelim [00:45:00]")
    ap.add_argument("--model", choices=("vurs", "vwrs", "gp"), default="vurs")
    ap.add_argument("--no-clip", action="store_true",
                    help="fit gamma as it is; default floors it at 0 (stable reads as 0)")
    ap.add_argument("--seed", type=int, default=0, help="round N uses seed + N")
    ap.add_argument("--ped-grid-args", default="",
                    help="extra ped_grid flags for cheaseBS, quoted, e.g. \"--max-iter 40\"")
    ap.add_argument("--template", default=C.GPU_TEMPLATE)
    ap.add_argument("--ped-root", default=None)
    ap.add_argument("--diag-root", default=None)
    ap.add_argument("--prefix", default=None, help="run dir prefix [<shot prefix>_vurs]")
    ap.add_argument("--dry-run", action="store_true",
                    help="round 1 pick only: print the jobs, build and run nothing")
    args = ap.parse_args(argv)

    d = C.shot_defaults(args.shot)
    x0 = args.x0 if args.x0 is not None else d["x0"][0]
    ped_root = args.ped_root or d["ped_root"]
    diag_root = args.diag_root or d["diag_root"]
    prefix = args.prefix or f"{d['prefix']}_vurs"
    diag_x0 = os.path.join(diag_root, C.x0_label((x0,)))
    nodes_per = int(os.environ.get("NODES_PER", 2))
    ky_per_job = args.ky_per_job or nodes_per * PPN
    if ky_per_job != nodes_per * PPN:
        ap.error(f"--ky-per-job {ky_per_job} != NODES_PER * {PPN} = {nodes_per * PPN}: "
                 "interactive_corners.sh runs one GPU per ky")
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 0))
    if not args.dry_run and not n_nodes:
        ap.error("not inside an allocation; salloc first (or --dry-run)")
    n_jobs = args.jobs or max(n_nodes // nodes_per, 1)
    rounds_dir = os.path.join(os.path.dirname(os.path.normpath(args.template)),
                              f"{prefix}_{C.x0_label((x0,))}_rounds")
    ped_args = shlex.split(args.ped_grid_args)

    left = time_left()
    need = args.rounds * C.walltime_seconds(args.walltime)
    if args.x0 is not None:
        x0_source = "--x0"
    else:
        names = [k for k, v in d["chapman"].items() if v == x0]
        x0_source = (f"{names[0]} Chapman radius of {args.shot}" if names
                     else f"default for {args.shot}") + ", corner_gene_scan.SHOTS; --x0 to change"
    others = ", ".join(f"{k} {v:g}" for k, v in d["chapman"].items() if v != x0)
    print(f"x0 = {x0:g} ({x0_source}); fixed for every round: picks from and writes GENE runs "
          f"at this radius" + (f". Other Chapman radii: {others}" if others else ""))
    print(f"{args.rounds} round(s) x {n_jobs} job(s) x {ky_per_job} kymin, model {args.model}, "
          f"GENE walltime {args.walltime}; picking from {diag_x0}"
          + (f"; {left / 3600:.2f} h left in the allocation" if left is not None else ""))
    if left is not None and need > left:
        print(f"WARNING: {args.rounds} x {args.walltime} of GENE alone exceeds the "
              f"{left / 3600:.2f} h left; the last round(s) will be cut off", flush=True)

    hold = []
    for r in range(1, args.rounds + 1):
        print(f"\n===== round {r}/{args.rounds} =====", flush=True)
        runs, jobs = pick(diag_x0, n_jobs, ky_per_job, args.model, args.seed + r, hold,
                          clip_min=None if args.no_clip else 0.)
        print(f"{len(runs)} point(s) on disk: "
              + ", ".join(f"{n} {s}" for s, n in runs.status.value_counts().items()))
        dup = jobs.tag.duplicated()
        for t in jobs.tag[dup]:
            print(f"  job at {t} dropped: rounds onto another job's case this round")
        jobs = jobs[~dup].reset_index(drop=True)
        with pd.option_context("display.width", 200, "display.max_colwidth", 80):
            print(jobs.assign(kymin=[", ".join(f"{k:g}" for k in ks) for ks in jobs.kymin])
                  .to_string())
        if args.dry_run:
            return 0

        logdir = os.path.join(rounds_dir, f"round{r}")
        os.makedirs(logdir, exist_ok=True)
        runs.to_csv(os.path.join(logdir, "runs.csv"), index=False)
        jobs.to_csv(os.path.join(logdir, "jobs.csv"), index=False)

        failed = cheasebs(list(jobs.tag), args.shot, ped_root, ped_args, logdir)
        for _, job in jobs[jobs.tag.isin(failed)].iterrows():
            hold += [{**{a: job[a] for a in KNOB}, "kymin": k} for k in job.kymin]
        run_dirs = []
        for j, job in jobs[~jobs.tag.isin(failed)].iterrows():
            manifest, skipped = C.build(args.template, ped_root, diag_root, f"{prefix}_r{r}j{j}",
                                        x0=(x0,), walltime=args.walltime, kymin=list(job.kymin),
                                        cases=[job.tag], gpu=True)
            run_dirs += [rec["run_dir"] for rec in manifest["runs"].values()
                         if rec["case_dir"].endswith(job.tag)]
        if not run_dirs:
            print("no run dirs this round; stopping", flush=True)
            return 1
        print(f"\nrunning {len(run_dirs)} run dir(s): {RUNNER}", flush=True)
        subprocess.run(["bash", RUNNER, *run_dirs], check=False)
    print(f"\n{args.rounds} round(s) done; per-round tables in {rounds_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
