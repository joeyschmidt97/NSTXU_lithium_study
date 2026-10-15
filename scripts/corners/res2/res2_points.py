"""2x-resolution reruns of chosen x0 = 0.6737 corner points: are the 1x (nx0 = 3) answers viable?

Each point is one ped_grid case and the ky worth checking there, picked from
the 1x scan (sparse_grid_dryrun/x0_0.6737/corners_modes.csv). Every point is
launched through corner_gene_scan.build with --res-scale 2: nx0 3 -> 7, nz0
64 -> 128, nv0 36 -> 72, nw0 32 -> 64, with n_procs_z/v/w doubled to match
(1024 ranks per ky, 8 nodes), nky0 1 as before. Output lands in
<diag-root>_res2/x0_0.6737/<tag>/, run dirs NSTX_129038_corner_res2_x0_0.6737_<tag>/,
so plot_corners.py --diag-root <diag-root>_res2 --x0 0.6737 plots them beside nothing
else, and the 1x and 2x values can be compared ky by ky.

    python -m NSTXU_lithium_study.scripts.corners.res2.res2_points --dry-run    # first point's files
    python -m NSTXU_lithium_study.scripts.corners.res2.res2_points --test --submit   # first point, debug queue
    python -m NSTXU_lithium_study.scripts.corners.res2.res2_points --submit
    python -m NSTXU_lithium_study.scripts.corners.res2.res2_points --only KBM MTM_low --submit

The debug test (30 min) will not converge at 2x; it checks the job starts
and gives GENE's time per step, from which to judge WALLTIME before the
real submit.
"""

from __future__ import annotations

import sys

from NSTXU_lithium_study.scripts.corner_gene_scan import DIAG_ROOT, PED_ROOT, PREFIX, TEMPLATE, build

X0 = (0.6737,)
RES_SCALE = 2
#: 1x converged runs took up to 4800 s on 128 ranks; 2x is ~19x the work per
#: step on 8x the ranks, and doubling nz0 and nv0 may shorten the time step.
WALLTIME = "08:00:00"

#: name: (ped_grid case, ky to rerun, why), 1x gamma in c_ref/L_ref.
POINTS = {
    "KBM": ("Teh1.20_Tew0.80_neh0.80_new1.20", (0.5, 1.0),
            "only KBM corner at this x0 (gamma 0.290, 0.503); ky 0.1, 0.3 unconverged"),
    "KBM_neighbour": ("Teh1.20_Tew0.80_neh0.80_new0.80", (0.5, 1.0),
                      "same but new 0.8: MTM (0.187, 0.407); is the MTM/KBM split real?"),
    "MTM_weak": ("Teh1.20_Tew0.80_neh1.20_new1.20", (0.3, 1.0),
                 "weakest corner (0.034 at ky 0.3); gamma falls 0.124 -> 0.090 from ky 0.7 to 1"),
    "MTM_strong": ("Teh0.80_Tew1.20_neh1.20_new0.80", (0.3, 1.0),
                   "largest ky 1 gamma (0.551); ky 0.1 unconverged"),
    "MTM_low": ("Teh0.80_Tew0.80_neh0.80_new0.80", (0.3, 1.0),
                "all-0.8 reference MTM (0.136, 0.517)"),
}


def run(names=None, submit=False, test=False, dry_run=False, walltime=WALLTIME,
        res_scale=RES_SCALE, template=TEMPLATE, ped_root=PED_ROOT, diag_root=DIAG_ROOT,
        prefix=PREFIX, force=False):
    names = list(names or POINTS)
    unknown = set(names) - set(POINTS)
    if unknown:
        raise SystemExit(f"unknown point(s) {sorted(unknown)}; known: {list(POINTS)}")
    if test or dry_run:
        names = names[:1]
    skipped = []
    for name in names:
        case, ky, why = POINTS[name]
        print(f"\n===== {name}: {case} ky {', '.join(f'{k:g}' for k in ky)} -- {why}")
        _, skip = build(template, ped_root, diag_root, prefix, force=force, submit=submit,
                        test=test, dry_run=dry_run, x0=X0, walltime=None if test else walltime,
                        kymin=ky, res_scale=res_scale, cases=[case])
        skipped += skip
    return skipped


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, choices=list(POINTS),
                    help="points to launch [all]")
    ap.add_argument("--submit", action="store_true", help="sbatch each run dir")
    ap.add_argument("--test", action="store_true", help="first point only, debug queue, 30 min")
    ap.add_argument("--dry-run", action="store_true", help="print the first point's files")
    ap.add_argument("--force", action="store_true", help="rewrite existing run dirs")
    ap.add_argument("--walltime", default=WALLTIME, help="SLURM wall time (not with --test)")
    ap.add_argument("--res-scale", type=int, default=RES_SCALE)
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--ped-root", default=PED_ROOT)
    ap.add_argument("--diag-root", default=DIAG_ROOT, help="1x root; _res<N> is appended")
    args = ap.parse_args()
    sys.exit(1 if run(args.only, args.submit, args.test, args.dry_run, args.walltime,
                      args.res_scale, args.template, args.ped_root, args.diag_root,
                      force=args.force) else 0)
