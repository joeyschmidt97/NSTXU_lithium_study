"""Plot a res_check batch: each doubled-grid rerun against its 1x scan, same profile scaling only.

The rerun dirs of batch <code> are found from their folder names,

    <lithium_study>/<shot>/rescheck_<code>/<grid>x2/x0_<x0>/<tag>/   (res_check.names)

and, as a backup (a dir moved, a root not globbed), from the batch record
GENE3_gpu/rescheck_<code>.json: its "dirs" list, or for an older record without
one, the dirs rebuilt from its jobs and grids. A dir with no GENE output yet is
reported and left out.

Each (shot, x0, grid) gets plot_corners.plot_res2 on the 1x scan of that radius
(diag_root, ped_root, extra_roots from sparse_grid_results/gene_run_paths.json,
else corner_gene_scan's shot defaults), given only the profile scalings that
were rerun, so each 1x line has its own 2x diamonds and nothing else:

    sparse_grid_results/<shot>/x0_<x0>/rescheck_<code>/rescheck_<grid>x2.png
    sparse_grid_results/<shot>/x0_<x0>/rescheck_<code>/data/rescheck_<grid>x2.csv
    sparse_grid_results/rescheck_<code>.csv        every row of the batch, one table

A 2x ky GENE did not converge (timelim reached: these are spot checks) is read
from a live field_omega fit of its field file, hollow +-1 std, faded unless the
fit passes; the 1x side uses GENE where it converged, else its recorded field
fit (update_results writes those). The table printed at the end flags a gamma
change above --tol.

    python -m NSTXU_lithium_study.scripts.corners.res2.plot_res_check --code h3r94s --list
    python -m NSTXU_lithium_study.scripts.corners.res2.plot_res_check --code h3r94s
    python -m NSTXU_lithium_study.scripts.corners.res2.plot_res_check --code h3r94s --shot 129015 --grids nx0
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from NSTXU_lithium_study.scripts import corner_gene_scan as C
from NSTXU_lithium_study.scripts.corners.plots import plot_corners as pc
from NSTXU_lithium_study.scripts.corners.res2.pick_points import RESULTS
from NSTXU_lithium_study.scripts.corners.res2.res_check import GRIDS, names
from NSTXU_lithium_study.scripts.ped_grid import parse_tag
from NSTXU_lithium_study.sparse_grid_results.update_results import X0_TOL, load_entries

#: <lithium_study>, the parent of every <shot>/corners and <shot>/rescheck_<code>
ROOT = os.path.dirname(os.path.dirname(C.shot_defaults(C.SHOT)["diag_root"].rstrip("/")))
GENEDIR = os.path.dirname(os.path.normpath(C.GPU_TEMPLATE))
DIR_RE = re.compile(r"(?P<shot>\d+)[/\\]rescheck_(?P<code>[0-9a-z]+)[/\\](?P<grid>n[xvwz]0)x2"
                    r"[/\\]x0_(?P<x0>\d+(?:\.\d+)?)[/\\](?P<tag>" + pc.TAG_RE.pattern + r")$")
#: |gamma 2x - gamma 1x| / |gamma 1x| above this is flagged in the table
TOL = 0.1
SUMMARY_KEYS = ("shot", "x0", "grid", "corner", "ky", "status_1x", "gamma_1x", "omega_1x",
                "source_2x", "gamma_2x", "gamma_2x_std", "omega_2x", "omega_2x_std",
                "gamma_rel_diff", "omega_rel_diff", "dir_2x")


def folder_dirs(code: str, root: str = ROOT) -> list[str]:
    """Rerun dirs of batch *code* by folder name, under every <shot>/ of *root*."""
    return sorted(d for d in glob.glob(os.path.join(root, "*", f"rescheck_{code}", "*x2",
                                                    "x0_*", "*")) if os.path.isdir(d))


def json_dirs(code: str, genedir: str = GENEDIR) -> list[str]:
    """Rerun dirs of batch *code* from its record: "dirs", else rebuilt from jobs and grids."""
    path = os.path.join(genedir, f"rescheck_{code}.json")
    if not os.path.isfile(path):
        print(f"  no batch record {path}")
        return []
    with open(path) as fh:
        batch = json.load(fh)
    if batch.get("dirs"):
        return list(batch["dirs"])
    return [os.path.join(names(j["shot"], code, g)[1], C.x0_label((j["x0"],)), j["case"])
            for g in batch["grids"] for j in batch["jobs"]]


def find_dirs(code, root=ROOT, genedir=GENEDIR, shots=None, x0s=None, grids=None):
    """{(shot, x0, grid): {tag: [rerun dir]}} of the dirs with GENE output, folder
    names first, the batch record filling in what they miss."""
    by_folder = [os.path.normpath(d) for d in folder_dirs(code, root)]
    by_json = [os.path.normpath(d) for d in json_dirs(code, genedir)]
    extra = [d for d in by_json if d not in by_folder]
    print(f"batch {code}: {len(by_folder)} dir(s) by folder name under {root}, "
          f"{len(extra)} more from the batch record")
    found: dict[tuple, dict[str, list[str]]] = {}
    for d in dict.fromkeys(by_folder + extra):
        m = DIR_RE.search(d)
        if not m:
            print(f"  not a rescheck_{code} dir name, left out: {d}")
            continue
        shot, x0, grid = int(m["shot"]), float(m["x0"]), m["grid"]
        if ((shots and shot not in shots) or (grids and grid not in grids)
                or (x0s and not any(abs(x0 - x) < X0_TOL for x in x0s))):
            continue
        if not pc.gene_runs(d):
            print(f"  no GENE output yet, left out: {d}")
            continue
        found.setdefault((shot, x0, grid), {}).setdefault(m["tag"], []).append(d)
    return found


def scan_roots(shot: int, x0: float, entries) -> dict:
    """diag_root, ped_root, extra_roots of the 1x scan at (shot, x0): its
    gene_run_paths.json entry (what pick_points read), else the shot defaults."""
    d = C.shot_defaults(shot)
    e = next((e for e in entries if e["shot"] == shot and abs(e["x0"] - x0) < X0_TOL), None)
    if e is None:
        print(f"  {shot} x0 {x0:g}: no gene_run_paths.json entry, using the shot defaults")
        e = {}
    return {"diag_root": e.get("diag_root") or d["diag_root"],
            "ped_root": e.get("ped_root") or d["ped_root"],
            "extra_roots": tuple(e.get("extra_roots") or ())}


def grid_value(paths, grid: str):
    """*grid* from the parameters of the first of *paths* that has one, or None."""
    for p in paths:
        if p:
            v = pc.run_param(p, grid)
            if v is not None:
                return v
    return None


def summary_rows(path: str, shot: int, x0: float, grid: str, alt: str) -> list[dict]:
    """plot_res2's CSV at *path* with its 2x columns renamed to *_2x, plus shot, x0, grid."""
    ak = re.sub(r"\W", "", alt.replace("nz0", "nz"))      # plot_res2's CSV key
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [{"shot": shot, "x0": f"{x0:g}", "grid": grid,
             **{k: r[k.replace("_2x", f"_{ak}")] for k in SUMMARY_KEYS[3:]}} for r in rows]


def print_table(rows, tol=TOL):
    print(f"\n{'shot':>6} {'x0':>7} {'grid':4} {'corner':33} {'ky':>5} {'gamma 1x':>9} "
          f"{'gamma 2x':>16} {'change':>7}  source 2x")
    for r in rows:
        rel = float(r["gamma_rel_diff"])
        sd = float(r["gamma_2x_std"])
        g2 = f"{float(r['gamma_2x']):.4g}" + (f" +-{sd:.2g}" if np.isfinite(sd) else "")
        flag = "  <-- over tol" if np.isfinite(rel) and abs(rel) > tol else ""
        print(f"{r['shot']:>6} {r['x0']:>7} {r['grid']:4} {r['corner']:33} {float(r['ky']):>5g} "
              f"{float(r['gamma_1x']):>9.4g} {g2:>16} "
              f"{100 * rel:>6.1f}%  {r['source_2x']}{flag}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", required=True, help="res_check batch code, e.g. h3r94s")
    ap.add_argument("--shot", type=int, nargs="+", default=None, help="shots [all in the batch]")
    ap.add_argument("--x0", type=float, nargs="+", default=None, help="radii [all in the batch]")
    ap.add_argument("--grids", nargs="+", choices=GRIDS, default=None,
                    help="doubled grids [all in the batch]")
    ap.add_argument("--root", default=ROOT, help="parent of <shot>/rescheck_<code>/")
    ap.add_argument("--genedir", default=GENEDIR, help="where rescheck_<code>.json is")
    ap.add_argument("--results", default=RESULTS, help="sparse_grid_results root (output)")
    ap.add_argument("--tol", type=float, default=TOL, help="flag |gamma change| above this")
    ap.add_argument("--list", action="store_true", help="print the dirs found and exit")
    args = ap.parse_args()

    found = find_dirs(args.code, args.root, args.genedir, args.shot, args.x0, args.grids)
    for (shot, x0, grid), dirs in sorted(found.items()):
        print(f"  {shot} x0 {x0:g} {grid}x2: " + ", ".join(sorted(dirs)))
    if not found:
        sys.exit("no rerun dirs with GENE output")
    if args.list:
        sys.exit(0)

    entries = load_entries()
    written, summary = [], []
    by_radius: dict[tuple, dict] = {}
    for (shot, x0, grid), dirs in found.items():
        by_radius.setdefault((shot, x0), {})[grid] = dirs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows
        for (shot, x0), per_grid in sorted(by_radius.items()):
            print(f"\n===== {shot} x0 {x0:g} =====")
            roots = scan_roots(shot, x0, entries)
            tags = sorted({t for dirs in per_grid.values() for t in dirs})
            ref = pc.tag(pc.REF)
            points = [parse_tag(t) for t in tags if t != ref] or None   # None: ref + corners
            try:
                ds = pc.load_dataset(roots["diag_root"], roots["ped_root"], (x0,),
                                     field_fits=True, shot=shot, points=points,
                                     scan="res check", extra_roots=roots["extra_roots"])
            except SystemExit as exc:
                print(f"  no 1x scan: {exc}")
                continue
            out = os.path.join(args.results, str(shot), C.x0_label((x0,)),
                               f"rescheck_{args.code}")
            os.makedirs(out, exist_ok=True)
            for grid, dirs in sorted(per_grid.items()):
                alt = f"{grid}x2"
                n1 = grid_value([p for t in dirs if t in ds.corner.values
                                 for p in ds.path.sel(corner=t).values], grid)
                n2 = grid_value([pc.gene_runs(d)[0] for ds_ in dirs.values() for d in ds_], grid)
                what = f"{grid} {n1:g} -> {n2:g}" if n1 and n2 else f"{grid} doubled"
                n = len(written)
                pc.plot_res2(ds, out, written, dirs=dirs, base="1x", alt=alt,
                             stem=f"rescheck_{alt}", what=what)
                for path in written[n:]:
                    if path.endswith(".csv"):
                        summary += summary_rows(path, shot, x0, grid, alt)

    if summary:
        path = os.path.join(args.results, f"rescheck_{args.code}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SUMMARY_KEYS)
            w.writeheader()
            w.writerows(summary)
        written.append(path)
        print_table(summary, args.tol)
    print("\nwrote\n  " + "\n  ".join(written))
