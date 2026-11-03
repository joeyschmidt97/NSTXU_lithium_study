"""Numerics variants of one case side by side: which setting gives a clean mode.

Each variant is a diag root that corner_gene_scan.py --prefix/--diag-root/--set
wrote for the same case (default the 1.00 reference) at the same x0, e.g. the
129015 x0 0.9515 debug matrix

    <lithium_study>/<shot>/corners_dbg_<variant>/x0_<x0>/<tag>/scanfilesNNNN

and is labelled by its folder name after the common prefix (ctrl, eo3, nz256...).
Writes to <out> (default <lithium_study>/<shot>/dbg_compare/x0_<x0>/):

    variants_mode_structure.png  one row per variant, one column per ky: |phi| and
                                 |Apar| / peak along the whole ballooning domain (last
                                 field time). Row label: the variant and its numerics
                                 (edge_opt, hyp_z, nz0, nx0, nv0, nw0, from its
                                 parameters). Cell text: gamma, omega and where they
                                 come from (GENE = converged; else the field fit and
                                 its verdict, as field_omega.py), end / peak, and the
                                 near-Nyquist phi power fraction (TPED outlier check).
                                 Border: green = clean (finite gamma below --gamma-cut,
                                 both ends below EXTENT_TOL, no grid-scale flag), red =
                                 any of those fails, grey = no field output.
    variants_growth.png          gamma and omega vs ky, one line per variant; filled =
                                 GENE converged, hollow = field fit; blow-ups (gamma
                                 above --gamma-cut) left out and named in the title
    data/variants.csv            every number above, one row per variant and ky

The runs need not be finished: a ky GENE has not converged gets the field fit of
its field_NNNN so far.

    python -m NSTXU_lithium_study.scripts.corners.plots.plot_variants --shot 129015 --x0 0.9515
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_variants --shot 129015 --x0 0.9515 --modes
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_variants --shot 129015 --x0 0.9515 --roots <dir> <dir> ...
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

from NSTXU_lithium_study.scripts.corner_gene_scan import shot_defaults, x0_label
from NSTXU_lithium_study.scripts.corners.plots.plot_corners import (
    EXTENT_TOL, GRID, INK, MODE_SHORT, MUTED, OUTLIER_NYQ, STRUCT_COLOR, SURFACE, classify,
    data_path, gene_runs, live_fit, mode_ends, mode_profile, outlier_check, read_run, run_param,
    save, tidy)

#: The 1.00 scaling on every knob.
REF_TAG = "Teh1.00_Tew1.00_neh1.00_new1.00"
#: Default variant roots under <lithium_study>/<shot>/.
DBG_GLOB = "corners_dbg_*"
#: Numerics shown in each row label, as they appear in the parameters file.
NUMERICS = ("edge_opt", "hyp_z", "nz0", "nx0", "nv0", "nw0")
#: Converged gamma above this is a numerical blow-up (as res_check --gamma-cut).
GAMMA_CUT = 50.0
CLEAN, FAILED, NO_DATA = "#008300", "#e34948", GRID
VARIANT_COLORS = ("#1f1f1e", "#2a78d6", "#eb6834", "#008300", "#6250d6", "#eda100",
                  "#e34948", "#3fb6c6", "#9a6b3f", "#b58fd0")


def variant_roots(shot: int, roots=None) -> dict[str, str]:
    """{label: diag root}: *roots* if given, else <lithium_study>/<shot>/corners_dbg_*.
    Labels are the folder names after their common prefix; ctrl first."""
    if not roots:
        parent = os.path.dirname(shot_defaults(shot)["diag_root"].rstrip("/"))
        roots = sorted(d for d in glob.glob(os.path.join(parent, DBG_GLOB)) if os.path.isdir(d))
    names = [os.path.basename(os.path.normpath(r)) for r in roots]
    cut = len(os.path.commonprefix(names)) if len(names) > 1 else 0
    cut = names[0][:cut].rfind("_") + 1 if cut else 0          # back to a word boundary
    out = {(n[cut:] or n): r for n, r in zip(names, roots)}
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "ctrl", kv[0])))


def read_variant(root: str, x0: float, tag: str, modes: bool, gamma_cut: float) -> dict:
    """{ky: record} for every run of *tag* at *x0* under *root*, and the numerics."""
    by_ky, numerics = {}, {}
    for path in gene_runs(os.path.join(root, x0_label((x0,)), tag)):
        try:
            rec = read_run(path)
        except Exception as exc:                 # one bad member never sinks the variant
            print(f"    unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        r = {"path": path, "gamma": rec["gamma"], "omega": rec["omega"], "source": "GENE",
             "blowup": False}
        if rec["status"] == "ok" and abs(rec["gamma"]) > gamma_cut:
            r["blowup"] = True
        if rec["status"] != "ok" or r["blowup"]:
            fit = live_fit(path)
            if fit is None:
                r.update(gamma=np.nan, omega=np.nan, source="none")
            else:
                g, _, w, _, verdict = fit
                r.update(gamma=g, omega=w, source=f"fit {verdict}")
        try:
            r["profile"] = mode_profile(path)
        except Exception as exc:
            print(f"    field unreadable {path}: {type(exc).__name__}: {exc}")
            r["profile"] = {}
        ends = mode_ends(r["profile"]) if r["profile"] else {}
        r["end"] = max((np.nanmax(e) for e in ends.values()), default=np.nan)
        try:
            o = outlier_check(path)
            r["nyquist"], r["grid_flag"] = o["nyquist_fraction"], o["flag_grid_oscillation"]
        except Exception:                        # needs files a short run may not have yet
            r["nyquist"], r["grid_flag"] = np.nan, False
        r["mode"] = ""
        if modes:
            try:
                r["mode"] = classify(path).get("mode", "")
            except Exception:
                pass
        r["clean"] = bool(r["profile"] and np.isfinite(r["gamma"]) and not r["blowup"]
                          and abs(r["gamma"]) <= gamma_cut and r["end"] <= EXTENT_TOL
                          and not r["grid_flag"])
        by_ky[round(rec["kymin"], 6)] = r
        if not numerics:
            numerics = {k: run_param(path, k) for k in NUMERICS}
    return {"runs": by_ky, "numerics": numerics}


def numerics_label(num: dict) -> str:
    return "  ".join(f"{k} {v:g}" for k, v in num.items() if v is not None)


def plot_structure(variants: dict, ky: list, title: str, out: str, written: list):
    nr, nc = len(variants), len(ky)
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc + 2.6, 1.35 * nr + 1.6), squeeze=False)
    fig.subplots_adjust(left=2.4 / (2.3 * nc + 2.6), right=0.99, top=1 - 1.0 / (1.35 * nr + 1.6),
                        bottom=0.7 / (1.35 * nr + 1.6), wspace=0.08, hspace=0.25)
    for r, (name, v) in enumerate(variants.items()):
        clean = sum(run["clean"] for run in v["runs"].values())
        axes[r, 0].set_ylabel(f"{name}  ({clean}/{len(v['runs'])} clean)\n"
                              + numerics_label(v["numerics"]).replace("  nz0", "\nnz0"),
                              rotation=0, ha="right", va="center", fontsize=7, color=INK)
        for c, k in enumerate(ky):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"ky = {k:g}", fontsize=9, color=INK)
            run = v["runs"].get(k)
            if run is None or not run["profile"]:
                for sp in ax.spines.values():
                    sp.set_color(NO_DATA)
                ax.text(0.5, 0.5, "no run" if run is None else "no field", ha="center",
                        va="center", fontsize=7, color=MUTED, transform=ax.transAxes)
                continue
            prof = run["profile"]
            z = prof["zgrid"]
            for f, colour in STRUCT_COLOR.items():
                if f in prof:
                    ax.plot(z, prof[f], color=colour, lw=0.8)
            ax.set_xlim(z[0], z[-1])
            ax.set_ylim(0, 1.45)
            for sp in ax.spines.values():
                sp.set_color(CLEAN if run["clean"] else FAILED)
                sp.set_linewidth(1.6)
            g = "BLOW-UP, " if run["blowup"] else ""
            mode = MODE_SHORT.get(run["mode"], run["mode"])
            text = (f"{g}γ {run['gamma']:.3g}  ω {run['omega']:.3g}  {mode}\n"
                    f"{run['source']}  end {run['end']:.2g}  nyq {run['nyquist']:.2g}")
            ax.text(0.02, 0.97, text, ha="left", va="top", fontsize=5.2, color=INK,
                    transform=ax.transAxes)
    handles = [Line2D([], [], color=STRUCT_COLOR["field_phi"], lw=1.5, label=r"$|\phi|$ / peak"),
               Line2D([], [], color=STRUCT_COLOR["field_apar"], lw=1.5,
                      label=r"$|A_\parallel|$ / peak"),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=CLEAN, lw=1.6, label="clean"),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=FAILED, lw=1.6,
                             label=f"blow-up, end > {EXTENT_TOL:g} or grid-scale "
                                   f"(nyq > {OUTLIER_NYQ:g})")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.suptitle(title + "\nmode structure along the whole ballooning domain (z/π), "
                 "last field time", x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, out, "variants_mode_structure.png", written)


def plot_growth(variants: dict, title: str, out: str, written: list):
    fig, (ag, aw) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.subplots_adjust(right=0.8, top=0.84)
    blown = []
    for (name, v), colour in zip(variants.items(), VARIANT_COLORS * 3):
        pts = sorted((k, r) for k, r in v["runs"].items() if np.isfinite(r["gamma"]))
        blown += [f"{name} ky {k:g}" for k, r in v["runs"].items() if r["blowup"]]
        if not pts:
            continue
        k = [p[0] for p in pts]
        for ax, var in ((ag, "gamma"), (aw, "omega")):
            ax.plot(k, [p[1][var] for p in pts], color=colour, lw=1.2, label=name)
            for kk, r in pts:
                ax.plot(kk, r[var], "o", color=colour, ms=5,
                        mfc=colour if r["source"] == "GENE" else "none")
    for ax, lab in ((ag, r"$\gamma$ [$c_{ref}/L_{ref}$]"), (aw, r"$\omega$ [$c_{ref}/L_{ref}$]")):
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(sorted({k for v in variants.values() for k in v["runs"]}))
        ax.set_xlabel(r"$k_y\rho_{ref}$", color=INK)
        ax.set_ylabel(lab, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        tidy(ax)
    handles, labels = ag.get_legend_handles_labels()
    handles += [Line2D([], [], color=MUTED, marker="o", ls="", label="GENE converged"),
                Line2D([], [], color=MUTED, marker="o", mfc="none", ls="", label="field fit")]
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=8)
    fig.suptitle(title + ("\nblow-ups (field fit shown instead): " + ", ".join(blown)
                          if blown else "\nno blow-ups"), x=0.01, ha="left", fontsize=10, color=INK)
    save(fig, out, "variants_growth.png", written)


def write_csv(variants: dict, out: str, written: list):
    path = data_path(out, "variants.csv")
    keys = ("gamma", "omega", "source", "blowup", "end", "nyquist", "grid_flag", "mode", "clean")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", *NUMERICS, "ky", *keys, "path"])
        for name, v in variants.items():
            for k, r in sorted(v["runs"].items()):
                w.writerow([name, *(v["numerics"].get(n) for n in NUMERICS), k,
                            *(r[x] for x in keys), r["path"]])
    written.append(path)


def run(shot: int, x0: float, tag=REF_TAG, roots=None, out=None, modes=False,
        gamma_cut=GAMMA_CUT) -> list[str]:
    found = variant_roots(shot, roots)
    if not found:
        raise SystemExit(f"no variant roots ({DBG_GLOB} for {shot}, or --roots)")
    variants = {}
    for name, root in found.items():
        print(f"--- {name}: {root}")
        variants[name] = read_variant(root, x0, tag, modes, gamma_cut)
        if not variants[name]["runs"]:
            print(f"    no runs under {os.path.join(root, x0_label((x0,)), tag)}")
    ky = sorted({k for v in variants.values() for k in v["runs"]})
    if not ky:
        raise SystemExit("no GENE runs in any variant yet")
    out = out or os.path.join(os.path.dirname(os.path.normpath(next(iter(found.values())))),
                              "dbg_compare", x0_label((x0,)))
    os.makedirs(out, exist_ok=True)
    title = f"NSTX {shot} x0 = {x0:g}, {tag}: {len(variants)} numerics variants"
    written: list[str] = []
    plot_structure(variants, ky, title, out, written)
    plot_growth(variants, title, out, written)
    write_csv(variants, out, written)
    print("\nclean ky per variant (finite gamma, no blow-up, ends and grid-scale below tolerance):")
    for name, v in sorted(variants.items(), key=lambda kv: -sum(r["clean"] for r in
                                                              kv[1]["runs"].values())):
        print(f"  {name:12s} {sum(r['clean'] for r in v['runs'].values())}/{len(v['runs'])}"
              f"   {numerics_label(v['numerics'])}")
    for p in written:
        print(f"wrote {p}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--x0", type=float, required=True)
    ap.add_argument("--tag", default=REF_TAG, help=f"case to compare [{REF_TAG}]")
    ap.add_argument("--roots", nargs="+", default=None,
                    help=f"variant diag roots [<lithium_study>/<shot>/{DBG_GLOB}]")
    ap.add_argument("--out", default=None, help="[<lithium_study>/<shot>/dbg_compare/x0_<x0>]")
    ap.add_argument("--modes", action="store_true", help="TPED mode label per run (slower)")
    ap.add_argument("--gamma-cut", type=float, default=GAMMA_CUT,
                    help=f"converged |gamma| above this is a blow-up [{GAMMA_CUT:g}]")
    a = ap.parse_args()
    run(a.shot, a.x0, a.tag, a.roots, a.out, a.modes, a.gamma_cut)
