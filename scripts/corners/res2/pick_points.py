"""Pick resolution spot-check points from the corner scans in sparse_grid_results.

For every <shot>/x0_<x0>/corners/data/corners_modes.csv (written by
update_results / plot_corners) two profile scalings are checked:

    ref    Teh1.00_Tew1.00_neh1.00_new1.00, always
    max    the scaling (corner or other case) with the largest converged gamma
           at that radius

and on each, three converged ky (status 'ok': GENE converged; field fits and
ambiguous points are left out):

    max     the largest gamma
    switch  the first ky whose mode label differs from the previous converged
            ky; else the first inflection of gamma(log ky); else a middle ky
    min     the smallest gamma

A point with gamma above --gamma-cut is a numerical blow-up (129015 x0 0.9515
1.00 ref: gamma 28011.6 at every ky, omega ~0), not a growth rate; it is dropped
before picking. A radius or scaling with no converged point is reported, not
picked. Mode labels come from TPED mode_identifier; a switch into or out of
'unidentified' is the classifier failing, not necessarily the physics changing.

    python -m NSTXU_lithium_study.scripts.corners.res2.pick_points
    python -m NSTXU_lithium_study.scripts.corners.res2.pick_points --shot 129038 --csv picks.csv
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "sparse_grid_results")
REF = "Teh1.00_Tew1.00_neh1.00_new1.00"
CONVERGED = ("ok",)
#: gamma (c_ref/L_ref) above which a converged point is a blow-up
GAMMA_CUT = 50.0


def three_points(s: pd.DataFrame) -> list[tuple[str, int]]:
    """(role, row index) for max, switch/inflection/mid and min of one scaling,
    *s* its converged rows sorted by ky; fewer when *s* has fewer rows."""
    ky, g, mode = s.ky.values, s.gamma.values, s["mode"].fillna("?").values
    imax, imin = int(np.argmax(g)), int(np.argmin(g))
    taken = {imax, imin}
    mid = None
    for i in range(1, len(s)):
        if mode[i] != mode[i - 1] and i not in taken:
            mid = (f"switch {mode[i - 1]}->{mode[i]}", i)
            break
    if mid is None and len(s) >= 4:
        d2 = np.diff(np.diff(g) / np.diff(np.log(ky)))
        for j in range(1, len(d2)):
            if np.sign(d2[j]) != np.sign(d2[j - 1]) and j + 1 not in taken:
                mid = ("inflection", j + 1)
                break
    if mid is None:
        rest = [i for i in range(len(s)) if i not in taken]
        if rest:
            mid = ("mid", rest[len(rest) // 2])
    out = [("max", imax)] + ([mid] if mid else []) + ([("min", imin)] if imin != imax else [])
    return out


def pick(results=RESULTS, shots=None, x0s=None, gamma_cut=GAMMA_CUT):
    """(picks, notes): one picks row per (shot, x0, case, ky) and a note per
    radius or scaling that could not be picked."""
    rows, notes = [], []
    for f in sorted(glob.glob(os.path.join(results, "*", "x0_*", "corners", "data",
                                           "corners_modes.csv"))):
        parts = os.path.normpath(f).split(os.sep)
        shot, x0 = int(parts[-5]), float(parts[-4][3:])
        if (shots and shot not in shots) or (x0s and x0 not in x0s):
            continue
        d = pd.read_csv(f)
        conv = d[d.status.isin(CONVERGED)]
        blown = conv[conv.gamma > gamma_cut]
        for case, n in blown.groupby("corner").size().items():
            notes.append((shot, x0, case, f"{n} converged point(s) with gamma > {gamma_cut:g} "
                                          f"dropped as blow-up (max {blown.gamma.max():g})"))
        conv = conv[conv.gamma <= gamma_cut]
        gmax = conv[conv.corner != REF].groupby("corner").gamma.max()
        best = gmax.idxmax() if len(gmax) else None
        if best is None:
            notes.append((shot, x0, "-", "no converged non-reference scaling"))
        for case, label in ((REF, "ref"), (best, "max")):
            if case is None:
                continue
            s = conv[conv.corner == case].sort_values("ky").reset_index(drop=True)
            if s.empty:
                st = {k: int(v) for k, v in d[d.corner == case].status.value_counts().items()}
                notes.append((shot, x0, case, f"{label}: no converged point below the "
                                              f"gamma cut (statuses {st or 'no run'})"))
                continue
            for role, i in three_points(s):
                r = s.iloc[i]
                rows.append({"shot": shot, "x0": x0, "case": case, "set": label, "role": role,
                             "ky": float(r.ky), "gamma": float(r.gamma),
                             "omega": float(r.omega), "mode": r["mode"],
                             "n_converged": len(s)})
    return pd.DataFrame(rows), notes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--results", default=RESULTS, help="sparse_grid_results root")
    ap.add_argument("--shot", type=int, nargs="+", default=None, help="shots [all]")
    ap.add_argument("--x0", type=float, nargs="+", default=None, help="radii [all]")
    ap.add_argument("--gamma-cut", type=float, default=GAMMA_CUT,
                    help="converged gamma above this is a blow-up and dropped")
    ap.add_argument("--csv", default=None, help="write the picks here")
    args = ap.parse_args()
    picks, notes = pick(args.results, args.shot, args.x0, args.gamma_cut)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(picks.to_string(index=False) if len(picks) else "no picks")
    for shot, x0, case, why in notes:
        print(f"  {shot} x0 {x0:g} {case}: {why}")
    if args.csv:
        picks.to_csv(args.csv, index=False)
        print(f"{len(picks)} pick(s) -> {args.csv}")
    sys.exit(0)
