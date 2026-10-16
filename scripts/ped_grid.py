"""Pedestal height x width design for one discharge, through mtanh_full.

scalings.py scales one knob at a time. This scans the four electron-channel
campaign axes together -- Te and ne pedestal height (scale_height) and width
(scale_width) -- and picks which points of the 4D box to run:

    axes     each knob alone over the axis grid, the other three at 1.0
    corners  all 2^4 combinations of the axis end points
    centre   (1, 1, 1, 1), the unscaled reconstruction
    full     the whole tensor grid (5^4 = 625 at the default axis)

Default design is axes + corners + centre: 16 + 16 + 1 = 33 cases.

Every case starts from the same unscaled discharge and the same nominal fits
(scalings._fit). Te is transformed with the Te fit, then ne with the ne fit,
one apply_mtanh_full call per variable carrying both its knobs. The two calls
touch disjoint variables (ne carries ni/nz through quasineutrality, Te touches
neither), so the order does not matter.

    python ped_grid.py --shot 129038                                # plots only
    python ped_grid.py --shot 129038 --pilot --cheasebs ...         # 2 corners + centre
    python ped_grid.py --shot 129038 --cheasebs --shard 0 --nshards 4 ...
    python ped_grid.py --shot 132543 --cheasebs --coordinate rhop --tensbnd -1.0 ...  # 132xxx K/L

With --cheasebs each case goes through output_gfile(run_cheasebs=True), which
writes the reconstructed gfile, the GENE profiles_e/i/z and profiles.iterdb
into <savedir>/<shot>/<tag>/. A case with a DONE marker is skipped, so a killed
shard is resumed by re-running the same command. Shards split the case list
round-robin; give every shard the same --savedir.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import traceback

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import NSTXU_lithium_study.scripts.scalings as S

DONE = "DONE.json"

# knob name -> (variable, apply_mtanh_full keyword). Order fixes the tag order.
KNOBS = {
    "Teh": ("Te", "scale_height"),
    "Tew": ("Te", "scale_width"),
    "neh": ("ne", "scale_height"),
    "new": ("ne", "scale_width"),
}
DESIGNS = ("axes", "corners", "centre", "full")


def axis(lo: float, hi: float, step: float) -> list[float]:
    """lo, lo+step, ..., hi inclusive, built by index so 0.1 steps stay exact."""
    if step <= 0 or hi < lo:
        raise ValueError(f"need step > 0 and hi >= lo, got {lo}, {hi}, {step}")
    n = int(round((hi - lo) / step))
    if not math.isclose(lo + n * step, hi, abs_tol=1e-9):
        raise ValueError(f"step {step} does not land on hi={hi} from lo={lo}")
    return [round(lo + i * step, 6) for i in range(n + 1)]


def design(grid, parts=("axes", "corners", "centre"), knobs=tuple(KNOBS)):
    """Ordered, de-duplicated list of points, each a {knob: factor} dict."""
    unit = {k: 1.0 for k in KNOBS}
    pts = []
    if "centre" in parts:
        pts.append(dict(unit))
    if "axes" in parts:
        for k in knobs:
            pts += [{**unit, k: s} for s in grid if s != 1.0]
    if "corners" in parts:
        for combo in itertools.product((grid[0], grid[-1]), repeat=len(knobs)):
            pts.append({**unit, **dict(zip(knobs, combo))})
    if "full" in parts:
        for combo in itertools.product(grid, repeat=len(knobs)):
            pts.append({**unit, **dict(zip(knobs, combo))})
    seen, out = set(), []
    for p in pts:
        key = tuple(p[k] for k in KNOBS)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def kind(p) -> str:
    """Which part of the design a point belongs to, for grouping plots."""
    moved = [k for k in KNOBS if p[k] != 1.0]
    if not moved:
        return "centre"
    if len(moved) == 1:
        return "axis_" + moved[0]
    return "multi"


def tag(p) -> str:
    return "_".join("%s%.2f" % (k, p[k]) for k in KNOBS)


def scale_point(base, p):
    """One apply_mtanh_full per variable with any knob off 1.0."""
    q = base
    for var in ("Te", "ne"):
        kw = {kwarg: p[k] for k, (v, kwarg) in KNOBS.items() if v == var}
        if any(s != 1.0 for s in kw.values()):
            q = q.apply_mtanh_full(var, fit=S._fit(base, var), **kw,
                                   enforce_quasineutrality=True, qz=S.QZ)
    return q


def plot_design(base, scaled, savedir, shot):
    """One PNG pair per single-knob axis, multi-knob points by Te/ne height."""
    groups = {}
    for p, q in scaled:
        k = kind(p)
        if k == "centre":
            continue
        if k == "multi":
            k = "corners_Teh%.2f_neh%.2f" % (p["Teh"], p["neh"])
        groups.setdefault(k, []).append((tag(p), q))
    for name, cases in sorted(groups.items()):
        for view, xlim in S.PLOT_VIEWS.items():
            path = os.path.join(savedir, "%d_%s_%s.png" % (shot, name, view))
            S.plot_family(base, cases, path, xlim=xlim)
            print("  wrote   %s  (%d case(s))" % (path, len(cases)), flush=True)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Pedestal height x width mtanh_full design for one discharge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=129038, choices=sorted(S.SHOTS))
    ap.add_argument("--lo", type=float, default=0.8)
    ap.add_argument("--hi", type=float, default=1.2)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--design", nargs="+", choices=DESIGNS,
                    default=["axes", "corners", "centre"])
    ap.add_argument("--knobs", nargs="+", choices=tuple(KNOBS), default=list(KNOBS),
                    help="axes the design spans; the rest stay at 1.0")
    ap.add_argument("--pilot", action="store_true",
                    help="centre plus the all-low and all-high corners only")
    ap.add_argument("--list", action="store_true",
                    help="print the case list and exit")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--cheasebs", action="store_true",
                    help="reconstruct each case; writes gfile, profiles_e/i/z "
                         "and profiles.iterdb into <savedir>/<shot>/<tag>/")
    ap.add_argument("--no-iterdb", dest="iterdb", action="store_false")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-iter", type=int, default=S.MAX_ITER)
    ap.add_argument("--tol-bs", type=float, default=None)
    ap.add_argument("--tol-q", type=float, default=None)
    ap.add_argument("--tol-ip-rel", type=float, default=None)
    ap.add_argument("--amplitude-warmup-iters", type=int, default=None)
    ap.add_argument("--ncscal", type=int, default=None)
    ap.add_argument("--tensbnd", type=float, default=None,
                    help="CHEASE TENSBND (boundary spline tension; template -0.3). "
                         "-1.0 with --coordinate rhop is the 132xxx K/L configuration")
    ap.add_argument("--replay-representation", choices=("istar", "jparallel"),
                    default=None)
    ap.add_argument("--coordinate", choices=("rhop", "rhot"), default=None)
    ap.add_argument("--bootstrap-mix", type=float, default=None)
    ap.add_argument("--savedir", default=None)
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.nshards:
        ap.error("need 0 <= --shard < --nshards")

    grid = axis(args.lo, args.hi, args.step)
    pts = design(grid, args.design, args.knobs)
    if args.pilot:
        lo = {**{k: 1.0 for k in KNOBS}, **{k: grid[0] for k in args.knobs}}
        hi = {**{k: 1.0 for k in KNOBS}, **{k: grid[-1] for k in args.knobs}}
        pts = [{k: 1.0 for k in KNOBS}, lo, hi]
    todo = pts[args.shard::args.nshards]

    print("=== pedestal design: %d, axis %s, %s, knobs %s ==="
          % (args.shot, grid, "pilot" if args.pilot else "+".join(args.design),
             args.knobs), flush=True)
    print("%d point(s) in design, shard %d/%d runs %d"
          % (len(pts), args.shard, args.nshards, len(todo)), flush=True)
    if args.list:
        for p in todo:
            print("  %-8s %s" % (kind(p), tag(p)))
        return 0

    savedir = os.path.abspath(args.savedir or S.scratch_dir("ped_%d" % args.shot))
    os.makedirs(savedir, exist_ok=True)
    print("output -> %s" % savedir, flush=True)
    with open(os.path.join(savedir, "design_shard%d.json" % args.shard), "w") as fh:
        json.dump({"shot": args.shot, "axis": grid, "knobs": KNOBS,
                   "cases": [{"tag": tag(p), **p} for p in todo]}, fh, indent=2)

    base = S.load(args.shot)
    scaled = []
    for p in todo:
        scaled.append((p, scale_point(base, p)))
        print("  scaled  %s" % tag(p), flush=True)

    if args.plots:
        plot_design(base, scaled, savedir, args.shot)
    if not args.cheasebs:
        return 0

    gfile_kw = {"max_iter": args.max_iter or None,
                "cheasebs_strict": args.strict,
                "write_iterdb": args.iterdb}
    if gfile_kw["max_iter"] is None:
        del gfile_kw["max_iter"]
    for key in ("tol_bs", "tol_q", "tol_ip_rel", "amplitude_warmup_iters",
                "replay_representation", "coordinate", "bootstrap_mix"):
        if getattr(args, key) is not None:
            gfile_kw[key] = getattr(args, key)
    if args.tensbnd is not None:
        from NSTXU_lithium_study.scripts.cheasebs_132xxx_tests import namelist_with
        entries = {"NCSCAL": args.ncscal} if args.ncscal is not None else {}
        gfile_kw["chease_namelist"] = namelist_with({**entries, "TENSBND": args.tensbnd},
                                                    savedir)
    elif args.ncscal is not None:
        gfile_kw["chease_namelist"] = S.namelist_with_ncscal(args.ncscal, savedir)
    print("gfile_kw: %s" % gfile_kw, flush=True)

    failed = []
    for p, q in scaled:
        label = tag(p)
        case = os.path.join(savedir, str(args.shot), label)
        if os.path.isfile(os.path.join(case, DONE)):
            print("--- %s done, skipping ---" % label, flush=True)
            continue
        os.makedirs(case, exist_ok=True)
        print("--- %d %s ---" % (args.shot, label), flush=True)
        try:
            path = q.output_gfile(savedir=case, run_cheasebs=True,
                                  comment="%d_%s" % (args.shot, label), **gfile_kw)
            iterdb = os.path.join(case, "profiles.iterdb")
            with open(os.path.join(case, DONE), "w") as fh:
                json.dump({**p, "gfile": str(path),
                           "iterdb": iterdb if os.path.isfile(iterdb) else None},
                          fh, indent=2)
            print("  gfile   %s" % path, flush=True)
            if args.iterdb and not os.path.isfile(iterdb):
                print("  WARNING no profiles.iterdb in %s" % case, flush=True)
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
            failed.append(label)

    print("=== shard %d: %d/%d ok in %s ==="
          % (args.shard, len(todo) - len(failed), len(todo), savedir), flush=True)
    if failed:
        print("FAILED: %s" % ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
