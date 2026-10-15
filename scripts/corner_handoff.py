"""Scaled-profile package of the ped_grid corners, for someone else to run.

ped_grid.py's 4D box -- Te and ne pedestal height and width (Teh, Tew, neh,
new) -- at its 16 corners plus the unit-scale centre, written as plain GENE
profiles beside the frozen source equilibrium. Nothing here runs CHEASE; it
writes files and plots.

WHAT GOES IN

Per discharge:

    <shot>/base/                               source EQDSK + unscaled profiles_{e,i,z}
    <shot>/base/<shot>_overview_{ped,full}.png every case vs base, one row per case
    <shot>/Teh1.00_Tew1.00_neh1.00_new1.00/    mtanh_full fit at unit scale (see below)
    <shot>/Teh0.80_Tew0.80_neh0.80_new0.80/    profiles_{e,i,z} + <tag>_vs_base_{ped,full}.png
    ...

The EQDSK is written once, into base/, because the boundary is frozen across
the whole scan. Case directories carry ped_grid.tag names, the same as the
corner cheaseBS and GENE runs.

THE CENTRE IS THE FIT, NOT THE RAW PROFILE

Every corner is an apply_mtanh_full reconstruction, so it differs from base/
by the scaling *and* by the fit residual. The centre case is the same
reconstruction with every knob at 1.0, so corner minus centre is the scaling
alone. (ped_grid.scale_point returns the raw profile at the centre; this
script does not, on purpose.)

The scale factors multiply fit parameters -- b_height for *h, b_width for *w --
not measured pedestal tops or widths. The manifest records, per case, Te and ne
at each variable's nominal pedestal-top radius (b_pos - 2 b_width), against
both the centre fit and the raw base.

    python corner_handoff.py                          # 132543 + 132588, 0.8/1.2, SCRATCH
    python corner_handoff.py --shots 132588
    python corner_handoff.py --lo 0.7 --hi 1.3 --name corners_0.7_1.3
    python corner_handoff.py --no-plots               # profiles + manifest only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import traceback

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

import NSTXU_lithium_study.scripts.scalings as S
from NSTXU_lithium_study.scripts.ped_grid import KNOBS, axis, design, scale_point, tag
from TPED.projects.discharge_tools.src.writers.gene_writer import write_gene_profiles

DEFAULT_SHOTS = (132543, 132588)

# Overview columns. Ti is held by every knob, so it would be a column of
# identical lines; ni shows quasineutrality carrying the ne scaling.
OVERVIEW_VARS = ("Te", "ne", "ni")

README = """# Pedestal-corner scaled-profile package

Generated {stamp} from `NSTXU_lithium_study/scripts/corner_handoff.py`.

## What this is

GENE profiles for {n_shots} NSTX discharge(s) at the corners of a 4D pedestal
box: Te and ne pedestal **height** and **width**, each at {lo} or {hi}
(2^4 = 16 corners), plus the unit-scale centre. Every case shares the source
equilibrium in `base/`; nothing here has been reconstructed.

## Layout

```
<shot>/base/<gfile>                         source EQDSK (frozen boundary, all cases)
<shot>/base/profiles_{{e,i,z}}                unscaled measured profiles
<shot>/base/<shot>_overview_ped.png         all cases vs base, pedestal zoom
<shot>/base/<shot>_overview_full.png        same, full radius
<shot>/<case>/profiles_{{e,i,z}}              scaled profiles
<shot>/<case>/<case>_vs_base_{{ped,full}}.png  that case vs base
manifest.json                               every case, factors and measured change
```

Case names read `Teh<a>_Tew<b>_neh<c>_new<d>`: the factor on the Te pedestal
height, Te width, ne height, ne width.

## What the factors mean

Each profile is a Stefanikova full-profile mtanh fit (Nucl. Fusion 56, 066010)
with the pedestal-top parameter `b_height` (h) and/or quarter-width `b_width`
(w) multiplied by the factor. They are fit-parameter factors, not measured
pedestal-top or width ratios; `manifest.json` records the measured Te and ne at
the nominal pedestal top for each case.

Density scaling rewrites `ni` and `nz` through quasineutrality (`qz = 6`).
Temperature scaling moves `Te` only; `Ti` is held.

## Which baseline to compare against

- `base/` is the **measured** profile.
- `Teh1.00_Tew1.00_neh1.00_new1.00/` is the **fit** at unit scale.

Every corner is a fit reconstruction, so corner vs `base/` mixes the scaling
with the fit residual; corner vs the centre case is the scaling alone.

If you run cheaseBS on a case, point the *reference* profiles at `base/`, not
at the case's own: using the scaled profiles as reference rebuilds the
fast-pressure decomposition from them and silently discards the perturbation.

## Discharges

{shot_lines}
"""


def unit_fit(base):
    """The mtanh_full reconstruction with every knob at 1.0: the corners' zero."""
    q = base
    for var in ("Te", "ne"):
        q = q.apply_mtanh_full(var, fit=S._fit(base, var),
                               enforce_quasineutrality=True, qz=S.QZ)
    return q


def ped_top_radius(base, var):
    fp = S._fit(base, var)
    return float(fp.b_pos - 2.0 * fp.b_width)


def value_at(q, var, r):
    x = np.asarray(q.ds.coords["rhot"].values, dtype=float)
    return float(np.asarray(q.ds[var].values, dtype=float)[np.argmin(np.abs(x - r))])


def measure(base, centre, q):
    """Te and ne at their nominal pedestal-top radii, vs centre fit and raw base."""
    out = {}
    for var in ("Te", "ne"):
        r = ped_top_radius(base, var)
        v, vc, vb = (value_at(o, var, r) for o in (q, centre, base))
        out[var] = {"ped_top_rhot": r, "value": v,
                    "ratio_vs_centre": v / vc if vc else None,
                    "ratio_vs_base": v / vb if vb else None}
    return out


def _ylim(arrays, x, xlim):
    """y range of the visible part of every line, so a zoom is legible."""
    m = np.ones_like(x, dtype=bool) if xlim is None else (x >= xlim[0]) & (x <= xlim[1])
    lo = min(float(np.nanmin(a[m])) for a in arrays)
    hi = max(float(np.nanmax(a[m])) for a in arrays)
    pad = 0.05 * (hi - lo or abs(hi) or 1.0)
    return lo - pad, hi + pad


def plot_overview(base, cases, path, shot, xlim=None, vars=OVERVIEW_VARS):
    """One row per case, one column per variable, base dashed on every panel.

    y is shared down each column, so a glance down a column compares cases on
    the same scale.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(base.ds.coords["rhot"].values, dtype=float)
    n = len(cases)
    fig, axes = plt.subplots(n, len(vars), figsize=(3.6 * len(vars), 1.5 * n + 0.8),
                             sharex=True, sharey="col", squeeze=False)
    for j, var in enumerate(vars):
        yb = np.asarray(base.ds[var].values, dtype=float)
        ys = [np.asarray(q.ds[var].values, dtype=float) for _, q in cases]
        lim = _ylim([yb, *ys], x, xlim)
        unit = base.ds[var].attrs.get("units", "")
        axes[0, j].set_title("%s [%s]" % (var, unit) if unit else var, fontsize=10)
        for i, ((label, _), y) in enumerate(zip(cases, ys)):
            ax = axes[i, j]
            ax.plot(x, yb, "k--", lw=1.0, label="base")
            ax.plot(x, y, color="C0", lw=1.4, label="scaled")
            ax.set_ylim(*lim)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel(label.replace("_", "\n"), fontsize=7, rotation=0,
                              ha="right", va="center")
    for ax in axes[-1]:
        ax.set_xlabel(r"$\rho_{tor}$")
        if xlim:
            ax.set_xlim(*xlim)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle("%d  pedestal corners vs base%s"
                 % (shot, "" if xlim is None else "  (rhot %.2f-%.2f)" % tuple(xlim)),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_shot(shot, pts, shot_dir, plots=True):
    """base/ plus one directory per point. Returns the manifest record."""
    print("--- %d ---" % shot, flush=True)
    base = S.load(shot)
    bdir = os.path.join(shot_dir, "base")
    os.makedirs(bdir, exist_ok=True)
    # run_cheasebs=False: the frozen raw geometry and nothing else.
    gfile = base.output_gfile(savedir=bdir, run_cheasebs=False)
    write_gene_profiles(base.ds, bdir)
    print("  base    %s + profiles_{e,i,z}" % os.path.basename(gfile), flush=True)

    centre = unit_fit(base)
    rec = {"gfile": os.path.basename(str(gfile)),
           "ped_top_rhot": {v: ped_top_radius(base, v) for v in ("Te", "ne")},
           "fit_b_width": {v: S._fit(base, v).b_width for v in ("Te", "ne")},
           "cases": []}
    cases = []
    for p in pts:
        label = tag(p)
        is_centre = all(p[k] == 1.0 for k in KNOBS)
        try:
            q = centre if is_centre else scale_point(base, p)
            cdir = os.path.join(shot_dir, label)
            write_gene_profiles(q.ds, cdir)
            if plots:
                for view, xlim in S.PLOT_VIEWS.items():
                    S.plot_family(base, [(label, q)],
                                  os.path.join(cdir, "%s_vs_base_%s.png" % (label, view)),
                                  xlim=xlim)
            m = measure(base, centre, q)
            rec["cases"].append({"dir": label, **p,
                                 "kind": "centre (fit, unit scale)" if is_centre
                                 else "corner", "measured": m})
            cases.append((label, q))
            print("  %-36s Te x%.3f  ne x%.3f  (ped top vs centre)"
                  % (label, m["Te"]["ratio_vs_centre"], m["ne"]["ratio_vs_centre"]),
                  flush=True)
        except Exception as exc:                                 # noqa: BLE001
            traceback.print_exc()
            rec["cases"].append({"dir": label, **p,
                                 "error": "%s: %s" % (type(exc).__name__, exc)})

    if plots and cases:
        for view, xlim in S.PLOT_VIEWS.items():
            path = plot_overview(base, cases,
                                 os.path.join(bdir, "%d_overview_%s.png" % (shot, view)),
                                 shot, xlim=xlim)
            print("  wrote   %s" % path, flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write a ped_grid-corner scaled-profile handoff package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS),
                    choices=sorted(S.SHOTS))
    ap.add_argument("--lo", type=float, default=0.8)
    ap.add_argument("--hi", type=float, default=1.2)
    ap.add_argument("--no-centre", dest="centre", action="store_false",
                    help="corners only")
    ap.add_argument("--no-plots", dest="plots", action="store_false")
    ap.add_argument("--outroot", default=None,
                    help="parent directory (default: $SCRATCH, else $PSCRATCH, else cwd)")
    ap.add_argument("--name", default=None,
                    help="package directory name (default: corner_handoff_<stamp>)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the package directory if it already exists")
    args = ap.parse_args(argv)

    grid = axis(args.lo, args.hi, args.hi - args.lo)
    pts = design(grid, ("centre", "corners") if args.centre else ("corners",))

    outroot = (args.outroot or os.environ.get("SCRATCH")
               or os.environ.get("PSCRATCH") or os.getcwd())
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    pkg = os.path.join(outroot, args.name or ("corner_handoff_" + stamp))
    if os.path.exists(pkg):
        if not args.overwrite:
            raise SystemExit("%s already exists; pass --overwrite or --name" % pkg)
        shutil.rmtree(pkg)
    os.makedirs(pkg)

    print("=== pedestal-corner handoff package ===")
    print("destination : %s" % pkg)
    print("shots       : %s" % args.shots)
    print("factors     : %s on %s  (fit-parameter factors)" % (grid, list(KNOBS)))
    print("cases       : %d per shot\n" % len(pts), flush=True)

    manifest = {"created": stamp, "package": pkg, "knobs": KNOBS, "axis": grid,
                "scale_meaning": "apply_mtanh_full scale_height (*h) / scale_width "
                                 "(*w) on the frozen nominal Stefanikova fit",
                "centre": "mtanh_full reconstruction at unit scale, not the raw "
                          "base profile",
                "quasineutrality": "enforced, qz = %g" % S.QZ,
                "shots": {}}
    failed = []
    for shot in args.shots:
        try:
            rec = write_shot(shot, pts, os.path.join(pkg, str(shot)), args.plots)
        except Exception as exc:                                 # noqa: BLE001
            print("!!! %d RAISED" % shot)
            traceback.print_exc()
            rec = {"error": "%s: %s" % (type(exc).__name__, exc)}
        if "error" in rec or any("error" in c for c in rec.get("cases", [])):
            failed.append(shot)
        manifest["shots"][str(shot)] = rec
        print(flush=True)

    shot_lines = []
    for shot in args.shots:
        rec = manifest["shots"][str(shot)]
        ok = [c for c in rec.get("cases", []) if "measured" in c]
        line = "- **%d** — EQDSK `%s`, %d case(s)" % (shot, rec.get("gfile", "MISSING"),
                                                    len(ok))
        if ok:
            for var in ("Te", "ne"):
                r = [c["measured"][var]["ratio_vs_centre"] for c in ok]
                line += ", %s ped-top x%.3f–%.3f" % (var, min(r), max(r))
        shot_lines.append(line)

    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(stamp=stamp, n_shots=len(args.shots), lo=grid[0],
                               hi=grid[-1], shot_lines="\n".join(shot_lines)))
    with open(os.path.join(pkg, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    print("=== done: %s ===" % pkg)
    if failed:
        print("  PROBLEM shots: %s" % failed)
    print("  tar czf %s.tar.gz -C %s %s"
          % (os.path.basename(pkg), os.path.dirname(pkg), os.path.basename(pkg)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
