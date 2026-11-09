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
    sparse_grid_results/<shot>/summary_rescheck_<grid>x2.png

Each summary (one per shot and doubled grid, every batch of the call that
doubled that grid on it, each batch its own 2x marker: diamond, pentagon,
hexagon, ... in the order the codes are given) has one column per radius of the
shot (every x0_<x0>/ folder under sparse_grid_results/<shot>/, smallest x0 left,
so the columns line up with update_results' summaries; "no res check" where a
radius was not rerun), gamma over omega: every rerun scaling at that radius on the same axes in
its plot_corners colour, its 1x line (circles for the reference) and its 2x
diamonds beside it, the gamma change over --tol written next to the diamond.
2x markers mark 2x points only. A summary holds only the batches of the call
(--all or several --code), and replaces the per-batch summary_rescheck_<code>_*
files of earlier versions.

A 2x ky GENE did not converge (timelim reached: these are spot checks) is read
from a live field_omega fit of its field file, hollow +-1 std, faded unless the
fit passes; the 1x side uses GENE where it converged, else its recorded field
fit (update_results writes those). The table printed at the end flags a gamma
change above --tol. With --x0 the summary is not rewritten (it would lose the
other radii). update_results runs this for every batch (batch_codes) after its
own plots.

    python -m NSTXU_lithium_study.scripts.corners.res2.plot_res_check --all
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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
FLAG_COLOR = "#e34948"
#: 2x marker per batch in a summary (code_markers); none is a 1x corner marker
CODE_MARKERS = ("D", "p", "h", "*", "8", "d")
SUMMARY_KEYS = ("shot", "x0", "grid", "corner", "ky", "status_1x", "gamma_1x", "omega_1x",
                "source_2x", "gamma_2x", "gamma_2x_std", "omega_2x", "omega_2x_std",
                "gamma_rel_diff", "omega_rel_diff", "dir_2x")


def folder_dirs(code: str, root: str = ROOT) -> list[str]:
    """Rerun dirs of batch *code* by folder name, under every <shot>/ of *root*."""
    return sorted(d for d in glob.glob(os.path.join(root, "*", f"rescheck_{code}", "*x2",
                                                    "x0_*", "*")) if os.path.isdir(d))


def batch_codes(root: str = ROOT, genedir: str = GENEDIR) -> list[str]:
    """Every res_check batch code: <root>/<shot>/rescheck_<code>/ folders and
    <genedir>/rescheck_<code>.json records."""
    found = {os.path.basename(d).removeprefix("rescheck_")
             for d in glob.glob(os.path.join(root, "*", "rescheck_*")) if os.path.isdir(d)}
    found |= {os.path.basename(f).removeprefix("rescheck_").removesuffix(".json")
              for f in glob.glob(os.path.join(genedir, "rescheck_*.json"))}
    return sorted(found)


def shot_radii(results: str, shot: int) -> list[float]:
    """x0 of every x0_<x0>/ folder under <results>/<shot>/ (update_results' radii)."""
    out = []
    for name in os.listdir(os.path.join(results, str(shot))) if os.path.isdir(
            os.path.join(results, str(shot))) else ():
        try:
            out.append(float(name.removeprefix("x0_")))
        except ValueError:
            continue
    return sorted(out)


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


def code_markers(codes) -> dict:
    """2x marker per batch code, in the order given (the first keeps the diamond)."""
    return {c: CODE_MARKERS[i % len(CODE_MARKERS)] for i, c in enumerate(codes)}


def plot_summary(shot, grid, panels, codes, out, written, tol=TOL, radii=()):
    """<out>/summary_rescheck_<grid>x2.png: every batch in *codes* that doubled
    *grid* on one figure, one column per radius, gamma (top) and omega (bottom).

    *panels* = [(x0, [(1x Dataset, summary rows of *grid*), ...])], one
    (Dataset, rows) per batch at that radius; each row carries its "code". A
    scaling's 1x line is drawn once per radius (first batch holding it) and every
    batch's 2x points go beside it, told apart by marker (code_markers) and a
    small extra ky offset. *radii* not in *panels* get an empty "no res check"
    column."""
    markers = code_markers(codes)
    present = {r["code"] for _, sets in panels for _, rows in sets for r in rows}
    order = {c: i for i, c in enumerate(c for c in codes if c in present)}   # ky offset
    have = [p[0] for p in panels]
    panels = list(panels) + [(x, []) for x in radii
                             if not any(abs(x - h) < X0_TOL for h in have)]
    panels = sorted(panels, key=lambda p: p[0])
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels) + 3, 8.5), squeeze=False)
    handles, faded, missing, used = {}, False, False, []
    for col, (x0, sets) in enumerate(panels):
        if not sets:
            for row in range(2):
                ax = axes[row, col]
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color(pc.GRID)
            axes[0, col].text(0.5, 0.5, "no res check", ha="center", va="center",
                              fontsize=10, color=pc.MUTED, transform=axes[0, col].transAxes)
            axes[0, col].set_title(f"x0 = {x0:g}", loc="left", fontsize=10, color=pc.INK)
            continue
        drawn = set()
        for ds, rows in sets:
            ky = ds.ky.values
            nudge = pc.RES2_NUDGE * (np.ptp(ky) if ky.size > 1 else 1.0)
            names = list(ds.corner.values)
            for c in [c for c in names if any(r["corner"] == c for r in rows)]:
                i = names.index(c)
                st = pc.rerun_style(ds, c)
                if c not in drawn:                    # one 1x line per scaling and radius
                    drawn.add(c)
                    lab = str(ds.label.values[i])
                    handles.setdefault(lab, Line2D([], [], lw=1.8, ms=5, **st, label=lab))
                    for row, var in enumerate(("gamma", "omega")):
                        ax = axes[row, col]
                        y, err, conv = pc.one_x(ds, i, var)
                        ok = np.isfinite(y)
                        ax.plot(ky[ok], y[ok], lw=1.2, color=st["color"], ls=st["ls"],
                                alpha=0.7, zorder=3)
                        ax.plot(ky[ok & conv], y[ok & conv], st["marker"], ms=4,
                                color=st["color"], zorder=4)
                        g = ok & ~conv
                        ax.errorbar(ky[g], y[g], yerr=err[g], fmt=st["marker"], ms=4,
                                    mfc=pc.SURFACE, mec=st["color"], color=st["color"],
                                    capsize=2, lw=0.8, zorder=4)
                for r in (r for r in rows if r["corner"] == c):
                    code = r["code"]
                    if code not in used:
                        used.append(code)
                    k = float(r["ky"])
                    x2 = k + nudge * (1 + 0.6 * order[code])
                    filled = r["source_2x"] == "GENE"
                    weak = r["source_2x"] not in ("GENE", "pass")
                    for row, var in enumerate(("gamma", "omega")):
                        ax = axes[row, col]
                        v1, v2, e2 = (float(r[f"{var}_1x"]), float(r[f"{var}_2x"]),
                                      float(r[f"{var}_2x_std"]))
                        if not np.isfinite(v2):
                            missing = True
                            ax.plot(x2, 0, "x", color=st["color"], ms=7, mew=1.6, zorder=6)
                            continue
                        faded |= weak
                        if np.isfinite(v1):
                            ax.plot([k, x2], [v1, v2], ":", color=st["color"], lw=1, zorder=4)
                        ax.errorbar(x2, v2, yerr=e2, fmt=markers[code], ms=7,
                                    mfc=st["color"] if filled else pc.SURFACE, mec=st["color"],
                                    mew=1.6, color=st["color"], capsize=2, lw=0.9,
                                    alpha=0.45 if weak else 1.0, zorder=7)
                        rel = float(r[f"{var}_rel_diff"])
                        if var == "gamma" and np.isfinite(rel) and abs(rel) > tol:
                            ax.annotate(f"{100 * rel:+.0f}%", (x2, v2), xytext=(4, 4),
                                        textcoords="offset points", fontsize=7,
                                        color=FLAG_COLOR)
        for row, sym in enumerate((r"\gamma", r"\omega")):
            ax = axes[row, col]
            ax.axhline(0, color=pc.MUTED, lw=0.8)
            ax.set_xlabel(r"$k_y\rho_{ref}$")
            if col == 0:
                ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
            pc.tidy(ax)
        axes[0, col].set_title(f"x0 = {x0:g}", loc="left", fontsize=10, color=pc.INK)
    used = sorted(used, key=order.get)
    m0 = CODE_MARKERS[0]
    extra = [Line2D([], [], ls="none", marker="o", color=pc.MUTED, ms=4,
                    label="1x GENE converged"),
             Line2D([], [], ls="none", marker="o", mfc=pc.SURFACE, mec=pc.MUTED, ms=4,
                    label="1x recorded field fit ±1σ")]
    extra += [Line2D([], [], ls="none", marker=markers[c], color=pc.MUTED, ms=7,
                     label=f"{grid}x2 batch {c}") for c in used]
    extra += [Line2D([], [], ls="none", marker=m0, color=pc.MUTED, ms=7,
                     label="2x filled: GENE converged"),
              Line2D([], [], ls="none", marker=m0, mfc=pc.SURFACE, mec=pc.MUTED, mew=1.6,
                     ms=7, label="2x hollow: field fit, passing ±1σ")]
    if faded:
        extra.append(Line2D([], [], ls="none", marker=m0, mfc=pc.SURFACE, mec=pc.MUTED,
                            mew=1.6, ms=7, alpha=0.45,
                            label="2x faded: field fit, ambiguous/decaying"))
    if missing:
        extra.append(Line2D([], [], ls="none", marker="x", color=pc.MUTED, ms=7, mew=1.6,
                            label=f"{grid}x2, no field output yet"))
    pc.legend_right(fig, handles.values(), extra)
    fig.suptitle(f"NSTX {shot} res check, {grid} doubled ({', '.join(used)}): 1x line, "
                 f"2x points by batch (red: gamma change > {100 * tol:g}%)", x=0.02,
                 ha="left", fontsize=12, color=pc.INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    pc.save(fig, out, f"summary_rescheck_{grid}x2.png", written)


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


def collect(code, shots=None, x0s=None, grids=None, root=ROOT, genedir=GENEDIR,
            results=RESULTS, list_only=False, entries=None, cache=None):
    """Per-radius plots and CSV rows of batch *code*: (paths written, summary rows
    each with its "code", [(shot, grid, x0, 1x Dataset, rows of that grid)]).
    *cache* shares 1x Datasets between batches of one run."""
    found = find_dirs(code, root, genedir, shots, x0s, grids)
    for (shot, x0, grid), dirs in sorted(found.items()):
        print(f"  {shot} x0 {x0:g} {grid}x2: " + ", ".join(sorted(dirs)))
    if not found:
        print(f"batch {code}: no rerun dirs with GENE output")
        return [], [], []
    if list_only:
        return [], [], []

    entries = load_entries() if entries is None else entries
    cache = {} if cache is None else cache
    written, summary, items = [], [], []
    by_radius: dict[tuple, dict] = {}
    for (shot, x0, grid), dirs in found.items():
        by_radius.setdefault((shot, x0), {})[grid] = dirs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows
        for (shot, x0), per_grid in sorted(by_radius.items()):
            print(f"\n===== {code}: {shot} x0 {x0:g} =====")
            roots = scan_roots(shot, x0, entries)
            tags = sorted({t for dirs in per_grid.values() for t in dirs})
            ref = pc.tag(pc.REF)
            points = [parse_tag(t) for t in tags if t != ref] or None   # None: ref + corners
            key = (shot, round(x0, 6), tuple(tags))
            if key not in cache:
                try:
                    cache[key] = pc.load_dataset(roots["diag_root"], roots["ped_root"], (x0,),
                                                 field_fits=True, shot=shot, points=points,
                                                 scan="res check",
                                                 extra_roots=roots["extra_roots"])
                except SystemExit as exc:
                    print(f"  no 1x scan: {exc}")
                    cache[key] = None
            ds = cache[key]
            if ds is None:
                continue
            out = os.path.join(results, str(shot), C.x0_label((x0,)),
                               f"rescheck_{code}")
            os.makedirs(out, exist_ok=True)
            first = len(summary)
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
                        summary += [{**r, "code": code}
                                    for r in summary_rows(path, shot, x0, grid, alt)]
            for grid in per_grid:
                rows = [r for r in summary[first:] if r["grid"] == grid]
                if rows:
                    items.append((shot, grid, x0, ds, rows))
    return written, summary, items


def run(codes, shots=None, x0s=None, grids=None, root=ROOT, genedir=GENEDIR,
        results=RESULTS, tol=TOL, list_only=False) -> list[str]:
    """Plot every batch in *codes* (one code or a list; see the module docstring)
    and one summary per shot and doubled grid holding all of them; the paths
    written. A batch failing is reported and the rest still run. Summaries are
    skipped under *x0s*, so they never lose radii; they hold only the batches of
    this call, so update_results passes every batch."""
    codes = [codes] if isinstance(codes, str) else list(codes)
    entries, cache = (None, {}) if list_only else (load_entries(), {})
    written, items, failed = [], [], []
    for code in codes:
        try:
            w, summary, it = collect(code, shots, x0s, grids, root, genedir, results,
                                     list_only, entries, cache)
        except (Exception, SystemExit) as exc:
            print(f"batch {code} failed: {type(exc).__name__}: {exc}")
            failed.append(code)
            continue
        written += w
        items += it
        if summary:
            path = os.path.join(results, f"rescheck_{code}.csv")
            with open(path, "w", newline="") as fh:
                dw = csv.DictWriter(fh, fieldnames=SUMMARY_KEYS, extrasaction="ignore")
                dw.writeheader()
                dw.writerows(summary)
            written.append(path)
            print_table(summary, tol)
    if list_only:
        return []

    panels: dict[tuple, dict] = {}
    for shot, grid, x0, ds, rows in items:
        panels.setdefault((shot, grid), {}).setdefault(round(x0, 6), []).append((ds, rows))
    done = [c for c in codes if c not in failed]
    for (shot, grid), by_x0 in sorted(panels.items()):
        if x0s:
            print(f"  --x0 given: summary_rescheck_{grid}x2.png for {shot} left as is")
            continue
        plot_summary(shot, grid, sorted(by_x0.items()), done,
                     os.path.join(results, str(shot)), written, tol, shot_radii(results, shot))
    if not x0s:                   # one summary per batch (before 2026-11-08): superseded
        for shot in sorted({s for s, _ in panels}):
            for code in done:
                for old in glob.glob(os.path.join(results, str(shot),
                                                  f"summary_rescheck_{code}_*x2.png")):
                    os.remove(old)
                    print(f"  removed superseded {old}")
    if failed:
        print(f"rescheck failed: {', '.join(failed)}")
    print("\nwrote\n  " + "\n  ".join(written))
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", nargs="+", default=None, help="res_check batch code(s), e.g. h3r94s")
    ap.add_argument("--all", action="store_true", help="every batch (batch_codes)")
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

    if not args.code and not args.all:
        ap.error("give --code or --all")
    codes = batch_codes(args.root, args.genedir) if args.all else args.code
    print(f"batches: {', '.join(codes) or 'none'}")
    run(codes, args.shot, args.x0, args.grids, args.root, args.genedir, args.results,
        args.tol, args.list)
