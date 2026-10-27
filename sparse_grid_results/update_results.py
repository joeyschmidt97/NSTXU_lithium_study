"""Regenerate sparse_grid_results/<shot>/x0_<x0>/ from the GENE runs in gene_run_paths.json.

Each (discharge, x0) entry of gene_run_paths.json names where its GENE corners
and cheaseBS profiles live. For each, this first fits every ky's field time
trace (field_omega.run with write=True: the fit and its verdict are appended to
the omega file, GENE's own line untouched, an earlier fit replaced), then runs
plot_corners.run on it, so corners_growth_field.png and --field-fits always
read fresh fits. --no-field-omega skips the fitting. The PNGs go to

    sparse_grid_results/<shot>/x0_<x0>/          plots
    sparse_grid_results/<shot>/x0_<x0>/data/     CSVs

An entry with nz_compare = N also gets the same field fits and plots for its
nz0 = N runs (nz_root, or <diag_root>_nz<N>), so that scan can be inspected on
its own (--no-nz-subdir skips it):

    sparse_grid_results/<shot>/x0_<x0>/nz<N>/    plots (+ data/)

Then, per discharge, the quick look across radii: each plot in SUMMARY of every
x0_<x0>/ folder of that discharge side by side (1 x N, smallest x0 left), from
the PNGs already there, so a radius not replotted this run still shows:

    sparse_grid_results/<shot>/summary_<plot>.png

--no-summary skips it; --summary-only rebuilds it from the PNGs on disk and
plots nothing else (runs anywhere, no /pscratch needed).

Every enabled entry by default; --shot and --x0 pick a subset (an x0 matches
any shot that has it). One entry failing is reported and the rest still run.

    python -m NSTXU_lithium_study.sparse_grid_results.update_results
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --list
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --shot 129038
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --shot 129038 --x0 0.6737
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --x0 0.8447 --no-modes
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --no-field-omega
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --shot 129038 --no-nz-subdir
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --paths my_paths.json
    python -m NSTXU_lithium_study.sparse_grid_results.update_results --shot 129038 --summary-only
"""

from __future__ import annotations

import json
import os
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS = os.path.join(HERE, "gene_run_paths.json")
#: plot_corners.run options an entry (or "defaults") may set
OPTIONS = ("modes", "mode_extent", "field_fits", "spread", "max_params")
#: on/off switches an entry (or "defaults", or the CLI) may set: the plot_corners
#: options above, field_omega, the field fits written before plotting, and
#: nz_subdir, the full plot set for the nz_compare runs in x0_<x0>/nz<N>/
SWITCHES = ("field_omega", "nz_subdir", "modes", "mode_extent", "field_fits")
#: two x0 are the same scan when closer than this (rho_tor)
X0_TOL = 1e-4
#: plots put side by side across radii in <shot>/summary_<plot>.png
SUMMARY = ("corners_growth", "corners_growth_field", "corners_mode_structure",
           "corners_mode_scaling", "corners_sampling")
SUMMARY_DPI = 100


def load_entries(path: str = PATHS) -> list[dict]:
    """Every (shot, x0) entry of the paths file, sorted, with "defaults" folded in."""
    with open(path) as fh:
        cfg = json.load(fh)
    defaults = cfg.get("defaults", {})
    out = []
    for shot, by_x0 in cfg["runs"].items():
        for x0, e in by_x0.items():
            out.append({"shot": int(shot), "x0": float(x0), "label": f"x0_{float(x0):g}",
                        "enabled": e.get("enabled", True), "diag_root": e.get("diag_root"),
                        "ped_root": e.get("ped_root"), "res2_roots": tuple(e.get("res2_roots") or ()),
                        "nz_compare": e.get("nz_compare"), "nz_root": e.get("nz_root"),
                        "field_omega": e.get("field_omega", defaults.get("field_omega", True)),
                        "nz_subdir": e.get("nz_subdir", defaults.get("nz_subdir", True)),
                        **{k: e.get(k, defaults.get(k)) for k in OPTIONS}})
    return sorted(out, key=lambda e: (e["shot"], e["x0"]))


def select(entries, shots=None, x0s=None) -> list[dict]:
    """Entries matching --shot and --x0; disabled ones only when both name them."""
    picked = []
    for e in entries:
        if shots and e["shot"] not in shots:
            continue
        if x0s and not any(abs(e["x0"] - x) < X0_TOL for x in x0s):
            continue
        if e["enabled"] or (shots and x0s):
            picked.append(e)
    return picked


def nz_root(e) -> str:
    """Root of an entry's nz_compare runs."""
    return e["nz_root"] or f"{e['diag_root']}_nz{e['nz_compare']}"


def describe(e) -> str:
    def where(p):
        return f"{p}  [{'ok' if p and os.path.isdir(p) else 'MISSING'}]"
    dirs = os.path.join(e["diag_root"] or "", e["label"])
    lines = [f"{e['shot']} {e['label']}" + ("" if e["enabled"] else "  (disabled)"),
             f"    GENE      {where(dirs)}", f"    profiles  {where(e['ped_root'])}"]
    lines += [f"    res2      {where(p)}" for p in e["res2_roots"]]
    if e["nz_compare"]:
        lines.append(f"    nz{e['nz_compare']:<7} {where(os.path.join(nz_root(e), e['label']))}")
    lines.append("    options   " + ", ".join(f"{k}={e[k]}" for k in ("field_omega", "nz_subdir", *OPTIONS)
                                              if e[k] is not None))
    return "\n".join(lines)


def update(e, out_root: str = HERE, nz: bool = False):
    """Field fits (unless field_omega is off), then plot_corners.run on one entry,
    into <out_root>/<shot>/x0_<x0>/. nz=True does the same for the entry's
    nz_compare runs, into <out_root>/<shot>/x0_<x0>/nz<N>/."""
    from NSTXU_lithium_study.scripts.corners.plots import plot_corners as pc
    d = pc.shot_defaults(e["shot"])
    diag_root = e["diag_root"] or d["diag_root"]
    out = os.path.join(out_root, str(e["shot"]), e["label"])
    if nz:
        diag_root = nz_root({**e, "diag_root": diag_root})
        out = os.path.join(out, f"nz{e['nz_compare']}")
    if e["field_omega"]:
        from NSTXU_lithium_study.scripts.corners.field_omega import field_omega as fo
        print("--- field fits (field_omega --write)")
        try:
            fo.run(diag_root, (e["x0"],), write=True)
        except Exception:                             # plot what is recorded rather than nothing
            traceback.print_exc()
            print("field fits failed; plotting with the fits already in the omega files")
    kw = {k: e[k] for k in OPTIONS if e[k] is not None}
    print("--- plots (plot_corners)")
    return pc.run(diag_root, e["ped_root"] or d["ped_root"], out=out, x0=(e["x0"],),
                  shot=e["shot"], res2_roots=() if nz else e["res2_roots"],
                  nz_compare=None if nz else e["nz_compare"], nz_root=e["nz_root"], **kw)


def summarize(shot: int, out_root: str = HERE) -> list[str]:
    """<out_root>/<shot>/summary_<plot>.png for each plot in SUMMARY: that PNG of
    every x0_<x0>/ folder of the discharge, side by side at its own resolution,
    smallest x0 left, each titled with its x0. A plot no radius has is skipped."""
    import matplotlib.pyplot as plt
    shot_dir = os.path.join(out_root, str(shot))
    folders = []
    for name in os.listdir(shot_dir) if os.path.isdir(shot_dir) else ():
        try:
            folders.append((float(name.removeprefix("x0_")), name))
        except ValueError:
            continue
    written = []
    for plot in SUMMARY:
        found = [(x0, os.path.join(shot_dir, name, f"{plot}.png")) for x0, name in sorted(folders)]
        found = [(x0, p) for x0, p in found if os.path.isfile(p)]
        if not found:
            continue
        imgs = [(x0, plt.imread(p)) for x0, p in found]
        title = 60                                  # px above each panel for its x0
        w = sum(im.shape[1] for _, im in imgs)
        h = max(im.shape[0] for _, im in imgs) + title
        fig = plt.figure(figsize=(w / SUMMARY_DPI, h / SUMMARY_DPI), facecolor="white")
        left = 0
        for x0, im in imgs:                         # each panel at 1 image px per figure px
            ax = fig.add_axes((left / w, (h - title - im.shape[0]) / h, im.shape[1] / w,
                               im.shape[0] / h))
            ax.imshow(im, interpolation="none")
            ax.set_axis_off()
            fig.text((left + 20) / w, 1 - 15 / h, f"x0 = {x0:g}", ha="left", va="top",
                     fontsize=20, fontweight="bold")
            left += im.shape[1]
        path = os.path.join(shot_dir, f"summary_{plot}.png")
        fig.savefig(path, dpi=SUMMARY_DPI)
        plt.close(fig)
        written.append(path)
        print(f"    {path}  ({', '.join(f'{x0:g}' for x0, _ in imgs)})")
    return written


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--paths", default=PATHS, help="JSON of GENE run paths per discharge and x0")
    ap.add_argument("--shot", type=int, nargs="+", default=None,
                    help="discharges to update [every one in --paths]")
    ap.add_argument("--x0", type=float, nargs="+", default=None,
                    help="radii to update, matched to the x0 keys [every one]")
    ap.add_argument("--out-root", default=HERE, help="results root; writes <shot>/x0_<x0>/ under it")
    ap.add_argument("--list", action="store_true",
                    help="print the selected entries and whether their paths exist; plot nothing")
    ap.add_argument("--summary", default=True, action=argparse.BooleanOptionalAction,
                    help="after plotting, <shot>/summary_<plot>.png: each plot of every x0 "
                         "side by side")
    ap.add_argument("--summary-only", action="store_true",
                    help="only rebuild the summaries from the PNGs on disk; plot nothing else")
    for k, what in (("field_omega", "fit every ky's field trace and append it to its omega file "
                                    "before plotting (field_omega.py --write)"),
                    ("nz_subdir", "full field fits and plots for the nz_compare runs, "
                                  "in x0_<x0>/nz<N>/"),
                    ("modes", "TPED mode_identifier per ky"),
                    ("mode_extent", "corners_mode_extent.png, corners_mode_structure.png"),
                    ("field_fits", "carry decaying/ambiguous field fits")):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None,
                        action=argparse.BooleanOptionalAction,
                        help=f"{what}; overrides the paths file [per entry]")
    args = ap.parse_args()

    entries = select(load_entries(args.paths), args.shot, args.x0)
    if not entries:
        raise SystemExit(f"no entry in {args.paths} matches --shot {args.shot} --x0 {args.x0}")
    for e in entries:
        e.update({k: getattr(args, k) for k in SWITCHES if getattr(args, k) is not None})
    print("\n".join(describe(e) for e in entries))
    if args.list:
        raise SystemExit(0)
    shots = sorted({e["shot"] for e in entries})
    if args.summary_only:
        for shot in shots:
            print(f"\n===== {shot} summary =====")
            summarize(shot, args.out_root)
        raise SystemExit(0)

    jobs = [(e, False) for e in entries]
    jobs += [(e, True) for e in entries if e["nz_compare"] and e["nz_subdir"]]
    failed = []
    for e, nz in jobs:
        name = f"{e['shot']} {e['label']}" + (f" nz{e['nz_compare']}" if nz else "")
        print(f"\n===== {name} =====")
        try:
            update(e, args.out_root, nz)
        except (Exception, SystemExit) as exc:        # load_dataset exits when nothing is found
            traceback.print_exc() if isinstance(exc, Exception) else print(exc)
            failed.append(name)
    if args.summary:
        for shot in shots:
            print(f"\n===== {shot} summary =====")
            summarize(shot, args.out_root)
    done = len(jobs) - len(failed)
    print(f"\nupdated {done}/{len(jobs)}" + (f"; failed: {', '.join(failed)}" if failed else ""))
    if failed:
        raise SystemExit(1)
