"""Read the completed cheaseBS box-edge runs back out of the repo.

The run notebook produces; this reads. Every run directory keeps its summary
JSON (committed) and its per-point solve artifacts (gitignored, so they exist
only on the machine that ran them). Everything table-shaped therefore works
anywhere; the DischargePhysics overlays need the machine that holds the EQDSKs.
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent,
                        *pathlib.Path(__file__).resolve().parents]
            if (p / "pedestal_scan.py").exists())
sys.path.insert(0, str(ROOT))

RUNSDIR = os.path.join(os.path.dirname(__file__), "runs")


# ----------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------

def load_runs(runsdir=RUNSDIR):
    """Every completed run, newest last. One dict per run directory."""
    runs = []
    for f in sorted(glob.glob(os.path.join(runsdir, "*", "reshape_convergence.json"))):
        d = json.load(open(f))
        tag = os.path.basename(os.path.dirname(f))
        c = d.get("cheasebs", {})
        runs.append(dict(
            tag=tag, dir=os.path.dirname(f), shot=d["shot"],
            radii=d.get("radii"), rows=d["rows"],
            tol_q=c.get("tol_q"), tol_bs=c.get("tol_bs"),
            max_iter=c.get("max_iter"),
            bootstrap_mix=c.get("bootstrap_mix"), istar_mix=c.get("istar_mix"),
            stamp=tag.split("_", 1)[1] if "_" in tag else "",
        ))
    return runs


def inventory(runs):
    """One line per run: what was solved, at which tolerance, how it went."""
    import pandas as pd

    out = []
    for r in runs:
        iters = [x.get("iterations") for x in r["rows"]]
        out.append({
            "run": r["tag"], "shot": r["shot"], "tol_q": r["tol_q"],
            "tol_bs": r["tol_bs"], "max_iter": r["max_iter"],
            "bs_mix": r["bootstrap_mix"], "istar_mix": r["istar_mix"],
            "n": len(r["rows"]),
            "iters": ", ".join("--" if i is None else str(i) for i in iters),
            "capped": sum(1 for i in iters if i == r["max_iter"]),
            "rejected": sum(1 for x in r["rows"] if x.get("accepted") is False),
            "raised": sum(1 for x in r["rows"] if "error" in x),
            "has_deltas": any(x.get("dq_max") is not None for x in r["rows"]),
            "wall_min": round(sum(x.get("wall_s") or 0 for x in r["rows"]) / 60, 1),
        })
    return pd.DataFrame(out)


def frame(runs):
    """One row per solve across every run -- the table everything else groups."""
    import pandas as pd

    out = []
    for r in runs:
        for x in r["rows"]:
            q = x.get("q_errors_rel") or {}
            out.append({
                "shot": r["shot"], "tol_q": r["tol_q"], "run": r["tag"],
                "axis": x["axis"], "var": x["axis"].split("_")[0],
                "scale": x["scale"],
                "dir": "up" if x["scale"] > 1 else "down",
                "ped_top": x.get("metric"),
                # measured pedestal-top change, the quantity the axis actually buys
                "d_ped_top": (x.get("metric_ratio") - 1
                              if x.get("metric_ratio") is not None else None),
                "iters": x.get("iterations"),
                "capped": (x.get("iterations") is not None
                           and x.get("iterations") == r["max_iter"]),
                "converged": x.get("converged"), "accepted": x.get("accepted"),
                "ip_err": x.get("ip_error_rel"),
                "q_err_x0": max((abs(v) for v in q.values()), default=None),
                "q_edge_err": x.get("q_edge_error_rel"),
                "dq_max": x.get("dq_max"), "dq_at": x.get("dq_at_rho"),
                "dp_max": x.get("dp_max"), "dp_at": x.get("dp_at_rho"),
                "wall_s": x.get("wall_s"),
                "raised": "error" in x,
            })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Reaching the solve artifacts
# ----------------------------------------------------------------------

def local_savedir(run, row):
    """The point's directory on THIS machine.

    The JSON stores the absolute path of the machine that solved it (NERSC), so
    it is rebuilt from this run directory plus the point's own folder name
    rather than trusted verbatim.
    """
    return os.path.join(run["dir"], os.path.basename(row["savedir"]))


def artifacts_available(runs):
    """Which runs still have their per-point EQDSKs on this machine."""
    import pandas as pd

    out = []
    for r in runs:
        have = sum(1 for x in r["rows"]
                   if os.path.isdir(local_savedir(r, x)))
        out.append({"run": r["tag"], "shot": r["shot"], "tol_q": r["tol_q"],
                    "points": len(r["rows"]), "artifacts_present": have})
    df = pd.DataFrame(out)
    if df["artifacts_present"].sum() == 0:
        print("No per-point artifacts on this machine -- runs/*/*/ is gitignored, "
              "so the EQDSKs live only where the solves ran (NERSC). Tables below "
              "work regardless; the DischargePhysics overlays need that machine.")
    return df


def load_physics(run, row):
    """DischargePhysics for one solved point, or None when unreachable."""
    import reshape_helpers as rh

    sd = local_savedir(run, row)
    if not os.path.isdir(sd):
        return None
    try:
        return rh.load_run(sd)
    except Exception as exc:
        print(f"  {row['axis']} {row['scale']}: reload failed -- "
              f"{type(exc).__name__}: {exc}")
        return None


# ----------------------------------------------------------------------
# Overlays
# ----------------------------------------------------------------------

def overlay(camp, run, var=None, **kwargs):
    """Source against the reconstructions, via DischargePhysics.plot().

    var="Te" or "ne" restricts to one axis, which is the readable comparison:
    three curves rather than five, and the two curves that share a scan are the
    ones whose difference means something. var=None overlays all four.

    plot() draws geometry, T and n profiles, and the q / F / p / p' / FF'
    panels, so a profile change and the equilibrium change it caused sit in one
    figure. If the T/n panels separate and the gfile panels do not, the reshape
    did not reach the equilibrium.
    """
    import matplotlib.pyplot as plt

    shot = run["shot"]
    rows = [r for r in run["rows"]
            if var is None or r["axis"].startswith(var)]
    rows = sorted(rows, key=lambda r: r["scale"])

    fig = camp[shot].phys.plot(label="source EFIT", discharge_idx=0, **kwargs)
    drawn = 0
    for i, r in enumerate(rows, start=1):
        p = load_physics(run, r)
        if p is None:
            continue
        fig = p.plot(fig=fig, discharge_idx=i, **kwargs,
                     label=f"{r['axis'].replace('_ped_scale','')} {r['scale']:.2f}"
                           f"  [{r.get('iterations')} it]")
        drawn += 1
    if drawn == 0:
        print(f"{shot} {var or 'all'}: no reconstructions reachable -- "
              "source only (see artifacts_available)")
    fig.suptitle(f"{shot}  tol={run['tol_q']:g}  "
                 f"{var or 'all axes'} — source vs reconstructions", y=1.01)
    return fig


def profile_only(camp, run, var, **kwargs):
    """The same comparison on kinetic profiles alone, for a closer read."""
    shot = run["shot"]
    rows = sorted([r for r in run["rows"] if r["axis"].startswith(var)],
                  key=lambda r: r["scale"])
    fig = camp[shot].phys.plot_profiles(label="source EFIT", **kwargs)
    for r in rows:
        p = load_physics(run, r)
        if p is None:
            continue
        fig = p.plot_profiles(fig=fig, label=f"{var} {r['scale']:.2f}", **kwargs)
    return fig
