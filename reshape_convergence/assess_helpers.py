"""Read the completed cheaseBS box-edge runs back out of the repo.

The run notebook produces; this reads. Every run directory keeps its summary
JSON (committed) and its per-point solve artifacts (gitignored, so they exist
only on the machine that ran them). Everything table-shaped therefore works
anywhere; the DischargePhysics overlays need the machine that holds the EQDSKs.
"""

from __future__ import annotations

import glob
import re
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
                "final_ip_a": x.get("final_ip_a"),
                "target_ip_a": x.get("target_ip_a"),
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


# ----------------------------------------------------------------------
# Per-iteration trace
# ----------------------------------------------------------------------

_ITER_RE = re.compile(
    r"Iteration (\d+) complete: Ip=([\d.]+) A, rel_error=([\deE.+-]+), "
    r"bs_change=(n/a|[\deE.+-]+), q_change=(n/a|[\deE.+-]+)")
_POINT_RE = re.compile(r"-((?:Te|ne)_ped_scale_[\d.]+)/cheasebs_run_config")
_RUNDIR_RE = re.compile(r"runs/(\d{6}_[\d_-]+)/")


def iteration_trace(nbpath):
    """Per-iteration record, recovered from an executed run notebook.

    cheaseBS prints Ip, rel_error, bs_change and q_change every iteration, but
    the summary JSON keeps only the final scalars -- so the trace exists solely
    in the notebook's stored cell output. It is the direct evidence for what
    stops the loop, which no other artifact carries.
    """
    import pandas as pd

    nb = json.load(open(nbpath))
    txt = "".join(
        "".join(o.get("text") or o.get("data", {}).get("text/plain") or "")
        for c in nb["cells"] for o in c.get("outputs", []))

    rows, point, rundir = [], None, None
    for line in txt.splitlines():
        m = _RUNDIR_RE.search(line)
        if m:
            rundir = m.group(1)
        m = _POINT_RE.search(line)
        if m:
            point = m.group(1)
        m = _ITER_RE.search(line)
        if m and point:
            it, ip, rel, bs, q = m.groups()
            axis, scale = point.rsplit("_", 1)
            rows.append({
                "run": rundir, "axis": axis, "var": axis.split("_")[0],
                "scale": float(scale), "iter": int(it), "ip_a": float(ip),
                "rel_err": float(rel),
                "bs_change": None if bs == "n/a" else float(bs),
                "q_change": None if q == "n/a" else float(q),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Symmetry
# ----------------------------------------------------------------------

def symmetry(df):
    """Up-vs-down response per axis, with NO baseline assumed.

    Grad-Shafranov and the Sauter/Wesson bootstrap are linear operators, so to
    first order dIp(+f) = -dIp(-f). Write the two box-edge points as an odd
    (linear) part and an even (rectified) part about the unperturbed value:

        odd  = [Ip(up) - Ip(down)] / 2          <- the physical response
        mid  = [Ip(up) + Ip(down)] / 2 = Ip(0) + even

    These runs carry no scale=1.0 control, so Ip(0) is unknown -- but it is the
    SAME number for every axis of a discharge. So under a linear response every
    axis must share one `mid`, and the spread in `mid` across axes measures the
    rectification directly, with nothing assumed. That is the test.

    `rect_A` is each axis's mid minus the smallest mid in the discharge, i.e.
    how far that axis's response is from purely odd. Adding a nominal point per
    discharge would pin Ip(0) and make the even part absolute rather than
    relative.
    """
    import pandas as pd

    out = []
    for (shot, tol), g in df[df.final_ip_a.notna()].groupby(["shot", "tol_q"]):
        per_axis = {}
        for axis, ga in g.groupby("axis"):
            up = ga[ga.scale > 1]["final_ip_a"]
            dn = ga[ga.scale < 1]["final_ip_a"]
            if up.empty or dn.empty:
                continue
            u, d = float(up.iloc[0]), float(dn.iloc[0])
            # mean |pedestal-top change| the axis actually bought, so the odd
            # response can be normalised: a small response to a small
            # perturbation is not the same finding as a small response to a
            # large one.
            f = float(ga.d_ped_top.abs().mean())
            per_axis[axis] = dict(up_A=u, down_A=d, f=f,
                                  odd_A=(u - d) / 2.0, mid_A=(u + d) / 2.0)
        if not per_axis:
            continue
        mid0 = min(v["mid_A"] for v in per_axis.values())
        for axis, v in per_axis.items():
            out.append({"shot": shot, "tol_q": tol, "axis": axis,
                        "Ip_up_A": v["up_A"], "Ip_down_A": v["down_A"],
                        "d_ped_top": v["f"], "odd_A": v["odd_A"],
                        "sens_A_per_unit": (v["odd_A"] / v["f"]) if v["f"] else np.nan,
                        "mid_A": v["mid_A"],
                        "rect_A": v["mid_A"] - mid0,
                        "rect_over_odd": (abs(v["mid_A"] - mid0) / abs(v["odd_A"])
                                          if v["odd_A"] else np.nan)})
    return pd.DataFrame(out)
