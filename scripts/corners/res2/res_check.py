"""Grid-doubling checks of the pick_points spot-check ky, one grid at a time, on GPU.

Every pick_points set (shot, x0, scaling, its three ky) is rerun once per grid
in --grids with that one grid doubled and everything else as in the 1x run:

    nx0  3 -> 7 (odd stays odd: 2n + 1, as corner_gene_scan --res-scale)
    nv0  n -> 2n
    nw0  n -> 2n
    nz0  n -> 2n (only if asked)

n_procs_* are left alone (unlike --res-scale), so each ky stays one rank, one
A100: a set is one run dir of three ky, three ranks, one node. Doubling one
grid at a time says which grid a ky is unresolved in, not only that it is.

The batch gets a six-character code from its picks and grids (same picks and
grids, same code), carried in every name:

    run dir  <genedir>/NSTX_<shot>_rc<code>_<grid>x2_x0_<x0>_<tag>/
    output   .../lithium_study/<shot>/rescheck_<code>/<grid>x2/x0_<x0>/<tag>/
    batch    <genedir>/rescheck_<code>.json   (picks, grids, 1x and 2x values)

so the one glob NSTX_*_rc<code>_*/ is the whole batch, and the same
interactive_corners.sh command can be run in allocation after allocation:
finished dirs are skipped, an existing run dir is never rewritten (--force to).
--code <code> reuses that batch's recorded picks, so a later update_results
cannot change what an old code means.

    python -m NSTXU_lithium_study.scripts.corners.res2.res_check --list         # code + jobs, no NERSC paths
    python -m NSTXU_lithium_study.scripts.corners.res2.res_check --dry-run      # first job's parameters
    python -m NSTXU_lithium_study.scripts.corners.res2.res_check                # write the run dirs
    python -m NSTXU_lithium_study.scripts.corners.res2.res_check --code a4f2ws  # same batch again
    python -m NSTXU_lithium_study.scripts.corners.res2.res_check --shot 132543 --grids nx0

then, in each allocation:

    salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m2116_g --gpus-per-node=4
    NODES_PER=1 bash NSTXU_lithium_study/scripts/interactive_corners.sh ~/simulators/GENE3_gpu/NSTX_*_rc<code>_*/

Compare with plot_corners --diag-root .../<shot>/rescheck_<code>/<grid>x2 --x0 <x0>.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from NSTXU_lithium_study.scripts import corner_gene_scan as C
from NSTXU_lithium_study.scripts.corners.res2.pick_points import GAMMA_CUT, RESULTS, pick

GRIDS = ("nx0", "nv0", "nw0", "nz0")
DEFAULT_GRIDS = ("nx0", "nv0", "nw0")
#: GENE timelim source; the 1x GPU corners ran at 01:30:00
WALLTIME = "01:30:00"
CODE_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def doubled(n: int, grid: str) -> int:
    return 2 * n + 1 if grid == "nx0" and n % 2 else 2 * n


def jobs_from_picks(picks) -> list[dict]:
    """One job per (shot, x0, case): its ky sorted, with the 1x values."""
    jobs = []
    for (shot, x0, case), g in picks.groupby(["shot", "x0", "case"], sort=True):
        g = g.sort_values("ky")
        jobs.append({"shot": int(shot), "x0": float(x0), "case": case, "set": g["set"].iloc[0],
                     "ky": [float(k) for k in g.ky],
                     "points": g[["role", "ky", "gamma", "omega", "mode"]].to_dict("records")})
    return jobs


def batch_code(jobs, grids) -> str:
    key = json.dumps([[j["shot"], f"{j['x0']:g}", j["case"], [f"{k:g}" for k in j["ky"]]]
                      for j in jobs] + [sorted(grids)])
    n = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    out = ""
    for _ in range(6):
        n, r = divmod(n, len(CODE_CHARS))
        out += CODE_CHARS[r]
    return out


def names(shot: int, code: str, grid: str) -> tuple[str, str]:
    """(run dir prefix, diag root) of one shot and grid in batch *code*."""
    base = os.path.dirname(C.shot_defaults(shot)["diag_root"].rstrip("/"))
    return f"NSTX_{shot}_rc{code}_{grid}x2", os.path.join(base, f"rescheck_{code}", f"{grid}x2")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grids", nargs="+", choices=GRIDS, default=list(DEFAULT_GRIDS),
                    help="grids to double, one run dir each [nx0 nv0 nw0]")
    ap.add_argument("--shot", type=int, nargs="+", default=None, help="shots [all picked]")
    ap.add_argument("--x0", type=float, nargs="+", default=None, help="radii [all picked]")
    ap.add_argument("--code", default=None,
                    help="reuse the picks recorded for this batch code")
    ap.add_argument("--results", default=RESULTS, help="sparse_grid_results root")
    ap.add_argument("--gamma-cut", type=float, default=GAMMA_CUT)
    ap.add_argument("--walltime", default=WALLTIME, help="sets GENE timelim [01:30:00]")
    ap.add_argument("--template", default=C.GPU_TEMPLATE)
    ap.add_argument("--force", action="store_true", help="rewrite existing run dirs")
    ap.add_argument("--list", action="store_true",
                    help="print the code and jobs and exit; reads nothing on NERSC")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the first job's parameters; write nothing")
    args = ap.parse_args()

    genedir = os.path.dirname(os.path.normpath(args.template))
    if args.code:
        path = os.path.join(genedir, f"rescheck_{args.code}.json")
        if not os.path.isfile(path):
            ap.error(f"no batch {path}")
        with open(path) as fh:
            batch = json.load(fh)
        jobs, grids, code = batch["jobs"], batch["grids"], args.code
    else:
        picks, notes = pick(args.results, args.shot, args.x0, args.gamma_cut)
        for shot, x0, case, why in notes:
            print(f"  not picked: {shot} x0 {x0:g} {case}: {why}")
        if picks.empty:
            sys.exit("no picks")
        jobs, grids = jobs_from_picks(picks), sorted(args.grids)
        code = batch_code(jobs, grids)

    print(f"\nbatch {code}: {len(jobs)} set(s) x {len(grids)} grid(s) ({', '.join(grids)}) "
          f"= {len(jobs) * len(grids)} run dirs")
    for j in jobs:
        print(f"  {j['shot']} x0 {j['x0']:g} {j['set']:3s} {j['case']}  ky "
              + ", ".join(f"{k:g}" for k in j["ky"]))
    if args.list:
        sys.exit(0)

    with open(os.path.join(args.template, "parameters")) as fh:
        par = fh.read()
    one_x = {g: int(C.read_param(par, g)) for g in grids}
    print("1x -> 2x: " + ", ".join(f"{g} {n} -> {doubled(n, g)}" for g, n in one_x.items()))

    skipped = []
    for grid in grids:
        for j in jobs:
            d = C.shot_defaults(j["shot"])
            prefix, diag_root = names(j["shot"], code, grid)
            _, skip = C.build(args.template, d["ped_root"], diag_root, prefix,
                              force=args.force, dry_run=args.dry_run, x0=(j["x0"],),
                              walltime=args.walltime, kymin=j["ky"],
                              overrides={grid: str(doubled(one_x[grid], grid))},
                              cases=[j["case"]], gpu=True)
            skipped += [(grid, j["shot"], j["x0"], *s) for s in skip
                        if s[1] != ["run dir exists"]]
            if args.dry_run:
                sys.exit(0)

    path = os.path.join(genedir, f"rescheck_{code}.json")
    if not os.path.isfile(path):
        with open(path, "w") as fh:
            json.dump({"code": code, "grids": grids, "one_x": one_x,
                       "two_x": {g: doubled(n, g) for g, n in one_x.items()},
                       "walltime": args.walltime, "jobs": jobs}, fh, indent=1)
    for grid, shot, x0, label, why in skipped:
        print(f"  SKIPPED {grid} {shot} x0 {x0:g} {label}: {why}")
    print(f"\nbatch {code} -> {path}\nrun (again in each new allocation until all are done):\n"
          "  salloc -N 4 -C gpu -q interactive -t 04:00:00 -A m2116_g --gpus-per-node=4\n"
          "  NODES_PER=1 bash NSTXU_lithium_study/scripts/interactive_corners.sh "
          f"~/simulators/GENE3_gpu/NSTX_*_rc{code}_*/")
    sys.exit(1 if skipped else 0)
