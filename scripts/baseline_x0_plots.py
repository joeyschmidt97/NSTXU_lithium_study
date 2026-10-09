"""Baseline Te/ne profiles with the candidate GENE x0 radii marked.

Per discharge, two PNGs: a full view and a pedestal zoom. Top row is the
baseline profile (dots) and its Stefanikova fit; bottom row is the fitted
gradient. Vertical lines mark the Chapman radii (ne top, Te top, steep) from
TPED's chapman_points.py, the fixed q-check radii, and any recorded GENE radii.
Every panel's y-axis is fitted to the data inside its visible x window.

The baseline is the box scan's reference_profiles/REF_profiles_e -- the same
file compare_cheasebs_runs.py --chapman fits -- so the radii drawn here are the
ones the comparison plots use.

    python baseline_x0_plots.py                               # all four shots
    python baseline_x0_plots.py --shots 129038 --zoom 0.8 1.0
    python baseline_x0_plots.py --box-root /pscratch/sd/j/joeschm/cheaseBS_runs/jparallel_box \
        --outdir ./x0_plots
"""

from __future__ import annotations

import glob
import json
import os

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# TPED and this repo import by absolute path: the directories that contain
# TPED/ and NSTXU_lithium_study/ must be on PYTHONPATH (see README).
from TPED.projects.discharge_tools.src.chapman_points import chapman_dataset_from_profile

SHOTS = (129015, 129038, 132543, 132588)

# First path that exists wins, so the same code runs on NERSC and on a laptop.
BOX_ROOT_CANDIDATES = [
    "/pscratch/sd/j/joeschm/cheaseBS_runs/jparallel_box",
    "C:/Users/joesc/git/cheaseBS_tests/data/jparallel_box",
]

#: Pedestal window, shared by every shot so the zooms compare directly.
ZOOM = (0.75, 1.0)

#: Fixed radii the box scan's q check was evaluated at.
FIXED_RADII = (0.85, 0.90, 0.95)

#: Radii of existing GENE linear scans (132588: q=4 and q=5 surfaces, per the
#: 2026-08-16 canary; coordinate recorded as r -- confirm it is rho_tor).
GENE_RADII = {132588: (0.736, 0.825)}

# Marker colours: validated categorical slots 1-3 and 5; line style and direct
# labels carry identity as well, so colour is never the only cue.
CHAPMAN_STYLE = {"n_top": ("#2a78d6", "-", "ne top"),
                 "T_top": ("#eb6834", "--", "Te top"),
                 "steep": ("#1baf7a", "-.", "steep")}
GENE_STYLE = ("#e87ba4", (0, (6, 2, 1, 2)))
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"


def box_root(explicit=None):
    if explicit:
        return explicit
    for path in BOX_ROOT_CANDIDATES:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("no box-scan root found; pass --box-root")


def reference_profile(root, shot):
    """First REF_profiles_e under <root>/<shot>/; every case shares the baseline."""
    hits = sorted(glob.glob(os.path.join(root, str(shot), "**", "reference_profiles",
                                         "REF_profiles_e"), recursive=True))
    return hits[0] if hits else None


def fit_ylim(ax, x, ys, xlim, pad=0.05):
    """Scale y to the data inside the visible x window, top at its max."""
    m = (x >= xlim[0]) & (x <= xlim[1])
    vals = np.concatenate([np.asarray(y)[m] for y in ys])
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or abs(hi) or 1.0
    ax.set_ylim(lo - pad * span, hi + pad * span)


def draw(shot, rho, prof, ds, pts, xlim, path, title):
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for j, (var, unit) in enumerate((("Te", "keV"), ("ne", "10$^{19}$ m$^{-3}$"))):
        y = prof[var]
        fit = ds["fit"].sel(variable=var).values
        grad = -np.gradient(fit, rho)
        a, b = ax[0, j], ax[1, j]
        a.plot(rho, y, "o", ms=3, color=MUTED, alpha=0.6)
        a.plot(rho, fit, color=INK, lw=2)
        a.set_ylabel(f"{var} [{unit}]")
        b.plot(rho, grad, color=INK, lw=2)
        b.set_ylabel(rf"$-d{var}/d\rho_{{tor}}$ (fit)")
        b.set_xlabel(r"$\rho_{tor}$")
        for a_ in (a, b):
            for x in FIXED_RADII:
                a_.axvline(x, color=MUTED, ls=":", lw=1.2)
            for p, (c, ls, _) in CHAPMAN_STYLE.items():
                a_.axvline(pts[p], color=c, ls=ls, lw=2)
            for x in GENE_RADII.get(shot, ()):
                a_.axvline(x, color=GENE_STYLE[0], ls=GENE_STYLE[1], lw=2)
            a_.grid(color=GRID, lw=0.8)
            a_.set_xlim(*xlim)
            for s in ("top", "right"):
                a_.spines[s].set_visible(False)
        fit_ylim(a, rho, (y, fit), xlim)
        fit_ylim(b, rho, (grad,), xlim)
        rms = float(ds["rms_relative"].sel(variable=var))
        a.set_title(f"{var}  (fit rms {100 * rms:.1f}%)", fontsize=10, color=INK, loc="left")

    # Direct labels on the Te panel; a label within 3% of the window of one
    # already placed drops lower so the two stay legible.
    top = ax[0, 0]
    y0, y1 = top.get_ylim()
    placed = []
    for p in sorted(CHAPMAN_STYLE, key=lambda k: pts[k]):
        x = pts[p]
        if not xlim[0] <= x <= xlim[1]:
            continue
        crowded = any(abs(x - q) < 0.03 * (xlim[1] - xlim[0]) for q in placed)
        top.text(x, y1 - (0.45 if crowded else 0.0) * (y1 - y0),
                 f" {CHAPMAN_STYLE[p][2]} {x:.3f}", rotation=90, va="top", ha="right",
                 fontsize=8, color=INK)
        placed.append(x)

    handles = []
    for p, (c, ls, lab) in CHAPMAN_STYLE.items():
        off = "" if xlim[0] <= pts[p] <= xlim[1] else "  (off-window)"
        handles.append(Line2D([], [], color=c, ls=ls, lw=2,
                              label=f"Chapman {lab} = {pts[p]:.3f}{off}"))
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.2,
                          label="fixed q-check radii " + "/".join(f"{r:.2f}" for r in FIXED_RADII)))
    if shot in GENE_RADII:
        handles.append(Line2D([], [], color=GENE_STYLE[0], ls=GENE_STYLE[1], lw=2,
                              label="existing GENE radii " + ", ".join(map(str, GENE_RADII[shot]))))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def run(shots=SHOTS, root=None, outdir=".", zoom=ZOOM):
    """Fit, plot and return {shot: {point: rho_tor}} for each shot found."""
    root = box_root(root)
    os.makedirs(outdir, exist_ok=True)
    out = {}
    for shot in shots:
        ref = reference_profile(root, shot)
        if not ref:
            print(f"[x0] {shot}: no reference_profiles/REF_profiles_e under {root}")
            continue
        d = np.loadtxt(ref, comments="#")
        rho, prof = d[:, 0], {"Te": d[:, 2], "ne": d[:, 3]}
        ds = chapman_dataset_from_profile(ref)
        pts = {p: float(ds["rho_tor"].sel(point=p)) for p in CHAPMAN_STYLE}
        full = (min(0.5, min(pts.values()) - 0.05), 1.0)
        draw(shot, rho, prof, ds, pts, full,
             os.path.join(outdir, f"nstx{shot}-baseline-x0.png"),
             f"NSTX {shot} baseline — candidate x0")
        draw(shot, rho, prof, ds, pts, tuple(zoom),
             os.path.join(outdir, f"nstx{shot}-pedestal-zoom.png"),
             f"NSTX {shot} pedestal zoom, rho_tor {zoom[0]}-{zoom[1]} — candidate x0")
        out[shot] = pts
        print(f"[x0] {shot}: " + "  ".join(f"{p}={v:.4f}" for p, v in pts.items()))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    ap.add_argument("--box-root", default=None,
                    help="directory holding <shot>/ box-scan outputs; default: first of %s"
                         % BOX_ROOT_CANDIDATES)
    ap.add_argument("--outdir", default="x0_plots")
    ap.add_argument("--zoom", type=float, nargs=2, default=list(ZOOM), metavar=("LO", "HI"))
    ap.add_argument("--json", action="store_true", help="also print the radii as JSON")
    args = ap.parse_args()
    pts = run(args.shots, args.box_root, args.outdir, args.zoom)
    if args.json:
        print(json.dumps(pts, indent=1))
