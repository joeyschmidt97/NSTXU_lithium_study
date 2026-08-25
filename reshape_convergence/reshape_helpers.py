"""Helpers for reshape_convergence.ipynb -- one discharge, four box edges.

Notebook owns presentation of the run; this owns the run itself, matching the
split pedestal_scan.py already uses for the fit and scaling notebooks.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent,
                        *pathlib.Path(__file__).resolve().parents]
            if (p / "pedestal_scan.py").exists())
sys.path.insert(0, str(ROOT))

from pedestal_scan import ANALYSIS_RADII, SCALE_SANITY  # noqa: E402

from TPED.projects.discharge_tools.src.cheasebs_runner import (  # noqa: E402
    CheasebsAcceptance, read_acceptance)
from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData  # noqa: E402
from TPED.projects.discharge_tools.src.filetypes.gfile_plots import GFilePlotsMixin  # noqa: E402

# The two height axes at the two edges of the frozen campaign box: four solves
# per discharge.  Width axes are out -- the open question is the 19-iteration
# ne height point from 2026-08-18.
AXES = ("Te_ped_scale", "ne_ped_scale")
SCALES = SCALE_SANITY                       # (0.7, 1.3)

# Campaign settings, so iteration counts transfer to the real runs unchanged.
CHEASEBS = dict(max_iter=25, tol_bs=1e-3, tol_q=1e-3, tol_ip_rel=0.02,
                bootstrap_mix=0.1, istar_mix=0.05, plot_errors=True)

OUTROOT = os.path.join(os.path.dirname(__file__), "runs")


def _solve(disc, axis, scale, savedir, radii, base_ds=None):
    """One reshape, one cheaseBS solve.  A raise is recorded, not propagated."""
    os.makedirs(savedir, exist_ok=True)
    row = {"axis": axis, "scale": scale, "savedir": savedir}
    try:
        row["metric"] = float(disc.metric(axis, scale))
        row["metric_ratio"] = row["metric"] / float(disc.metric(axis, 1.0))
    except Exception as exc:
        row["metric_error"] = f"{type(exc).__name__}: {exc}"

    phys = disc.scaled(axis, scale)
    t0 = time.time()
    try:
        phys.output_gfile(
            savedir=savedir, run_cheasebs=True,
            cheasebs_acceptance=CheasebsAcceptance.production(analysis_radii=radii),
            cheasebs_strict=False, comment=f"{axis}_{scale:.3f}", **CHEASEBS)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_s"] = time.time() - t0
        return row
    row["wall_s"] = time.time() - t0

    rec = read_acceptance(savedir) or {}
    spath = os.path.join(savedir, "convergence_summary.json")
    summ = json.load(open(spath)) if os.path.exists(spath) else {}
    row.update(
        accepted=rec.get("accepted"), reasons=rec.get("reasons"),
        ip_error_rel=rec.get("ip_error_rel"), q_errors_rel=rec.get("q_errors_rel"),
        q_edge_error_rel=rec.get("q_edge_error_rel"),
        iterations=summ.get("iterations"), converged=summ.get("final_converged"),
        final_ip_a=summ.get("final_ip_a"), target_ip_a=summ.get("target_ip_a"),
    )
    row["gfile"] = _find_gfile(savedir)
    if row["gfile"] and base_ds is not None:
        try:
            row.update(profile_deltas(base_ds, row["gfile"]))
        except Exception as exc:
            row["delta_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _find_gfile(savedir):
    """The RECONSTRUCTED EQDSK, not the source copy sitting beside it.

    output_gfile writes the frozen source geometry into the run directory under
    its original g<shot>.<time> name -- byte-identical to the input EFIT -- and
    cheaseBS writes its result as EQDSK*.OUT (513x513, header "FROM CHEASE").
    Globbing g* therefore picks the input, every delta comes out zero, and the
    plot reads as "the reshape never reached the equilibrium".  Prefer the OUT.
    """
    out = sorted(p for p in glob.glob(os.path.join(savedir, "EQDSK*.OUT"))
                 if os.path.isfile(p))
    if out:
        return out[0]
    return None


def profile_deltas(base_ds, recon_path, rho_max=0.995):
    """Whole-profile change from source EFIT to reconstruction.

    cheaseBS reshapes the entire equilibrium, so a q error read at the two GENE
    analysis radii describes the two points the acceptance gate cares about and
    says nothing about the other 127.  These are the numbers that answer "did
    the reshape move the equilibrium, and where".

    Stops at rho_max: the last flux surface carries a standing ~20% q error
    that is a known grid/X-point artifact, and it would dominate every maximum.
    """
    recon = GFileData(recon_path).gfile_to_xarray()
    rho_b = np.asarray(base_ds.coords["rho_tor"].values, dtype=float)
    rho_r = np.asarray(recon.coords["rho_tor"].values, dtype=float)

    m = rho_b <= rho_max
    rho = rho_b[m]
    out = {}
    for key, name in (("q", "q"), ("p", "p")):
        b = np.asarray(base_ds[key].values, dtype=float)[m]
        r_full = np.asarray(recon[key].values, dtype=float)
        o = np.argsort(rho_r)
        r = np.interp(rho, rho_r[o], r_full[o])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(r - b) / np.maximum(np.abs(b), 1e-30)
        rel = np.where(np.isfinite(rel), rel, np.nan)
        i = int(np.nanargmax(rel))
        out[f"d{name}_max"] = float(rel[i])
        out[f"d{name}_at_rho"] = float(rho[i])
        out[f"d{name}_rms"] = float(np.sqrt(np.nanmean(rel**2)))
    return out


def run_bounds(camp, shot, axes=AXES, scales=SCALES, outroot=OUTROOT):
    """Four solves for one discharge.  Returns (rows, workdir)."""
    disc, radii = camp[shot], ANALYSIS_RADII[shot]
    base_ds = disc.phys._tree["raw/gfile"].dataset
    stamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    workdir = os.path.abspath(os.path.join(outroot, f"{shot}_{stamp}"))
    print(f"{shot}  radii {radii}  {len(axes)*len(scales)} solves -> {workdir}")

    rows = []
    for axis in axes:
        for s in scales:
            print(f"  {axis} {s:.2f} ...", end="", flush=True)
            r = _solve(disc, axis, s,
                       os.path.join(workdir, f"{axis}_{s:.3f}"), radii,
                       base_ds=base_ds)
            if "error" in r:
                print(f" RAISED {r['error'][:60]}")
            else:
                print(f" {r.get('iterations')} iters, {r['wall_s']:.0f}s, "
                      f"accepted={r.get('accepted')}")
            rows.append(r)

    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "reshape_convergence.json"), "w") as f:
        json.dump({"shot": shot, "radii": list(radii), "cheasebs": CHEASEBS,
                   "rows": rows}, f, indent=1, default=str)
    return rows, workdir


def table(rows, shot=None):
    """Parameters and solver response, one line per solve."""
    import pandas as pd

    out = []
    for r in rows:
        q = r.get("q_errors_rel") or {}
        out.append({
            "axis": r["axis"],
            "scale": r["scale"],
            "ped_top": r.get("metric"),
            "ped_top/nom": r.get("metric_ratio"),
            "iters": r.get("iterations"),
            "converged": r.get("converged"),
            "accepted": r.get("accepted"),
            "Ip_err": r.get("ip_error_rel"),
            # Whole profile -- cheaseBS reshapes all of it.
            "dq_max": r.get("dq_max"),
            "dq_at": r.get("dq_at_rho"),
            "dq_rms": r.get("dq_rms"),
            "dp_max": r.get("dp_max"),
            # The two GENE analysis radii only: what the acceptance gate scores.
            "q_err@x0": max((abs(v) for v in q.values()), default=None),
            "q_edge_err": r.get("q_edge_error_rel"),
            "wall_s": r.get("wall_s"),
            "status": "RAISED" if "error" in r else "",
        })
    df = pd.DataFrame(out)
    if shot is not None:
        df.insert(0, "shot", shot)
    fmt = {"ped_top": "{:.3e}", "ped_top/nom": "{:.3f}", "Ip_err": "{:.2%}",
           "dq_max": "{:.2%}", "dq_at": "{:.3f}", "dq_rms": "{:.2%}",
           "dp_max": "{:.2%}", "q_err@x0": "{:.2%}", "q_edge_err": "{:.2%}",
           "wall_s": "{:.0f}"}
    return df.style.format(fmt, na_rep="--").hide(axis="index")


def plot_gfiles(rows, camp, shot, xcoord="rho_tor", **xlim):
    """Baseline gfile against every reconstruction, on the q/F/p/p'/FF' panels.

    Black is the source EFIT.  A reconstruction sitting on top of it means the
    reshape did not reach the equilibrium; a spread means it did.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    base = camp[shot].phys._tree["raw/gfile"].dataset
    fig = GFilePlotsMixin.plot_gfile_profiles(
        base, xcoord=xcoord, label="source EFIT", color="k", **xlim)

    # Hex, not RGBA: plot_gfile_profiles does `color or panel_color`, and an
    # RGBA array raises on the truth test.
    colors = [mcolors.to_hex(c)
              for c in plt.cm.coolwarm(np.linspace(0, 1, max(len(rows), 2)))]
    drawn = 0
    for r, c in zip(rows, colors):
        path = r.get("gfile")
        if not path or not os.path.isfile(path):
            continue
        ds = GFileData(path).gfile_to_xarray()
        fig = GFilePlotsMixin.plot_gfile_profiles(
            ds, fig=fig, xcoord=xcoord, color=c,
            label=f"{r['axis'].replace('_ped_scale','')} {r['scale']:.2f}",
            **xlim)
        drawn += 1

    if drawn == 0:
        print("no reconstructed gfiles found -- solves raised, nothing to plot")
    fig.suptitle(f"{shot} -- source vs cheaseBS reconstructions", y=1.0)
    fig.tight_layout()
    return fig
