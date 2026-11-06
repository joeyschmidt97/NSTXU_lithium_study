"""Backfill kymin on VURS equilibria that already ran: same cases, new ky, FILL run dirs.

VURS rounds that ran a short kymin scanlist (e.g. 8 ky over 0.1-10) can miss
the growth-rate peak. This picks --n-ky new kymin for each of those
equilibria from a 1D GP of gamma against log kymin, fitted to that
equilibrium's own runs only, and reruns the same cheaseBS case with them:

    peak   --n-peak picks (default half): where the maximum of gamma(ky) may
           be. 2000 GP posterior draws on a fine log-ky grid, the argmax of
           each binned into a density; the densest allowed ky is picked, the
           GP is conditioned on its mean there (kriging believer, kernel
           fixed), and the draws are redone, so the picks spread over the
           peak instead of stacking on its mode. Two peaks (an ion- and an
           electron-scale branch) split the density and both get picks.
    std    the rest: largest GP posterior std, conditioned on every pick so
           far, one at a time.

A ky is allowed if it sits at least a factor --min-ratio (log distance) from
every ky already on disk or already picked, after rounding to 3 decimals as
vurs_loop does. An equilibrium with fewer than MIN_GP usable values gets
maximin gap fills instead of GP picks (flag 'fallback'). A posterior-mean
peak on the edge of the kymin range is flagged 'peak_at_edge': the maximum
may sit outside the range.

Equilibria: every vurs_loop run dir <prefix>_r<R>j<J>_x0_<x0>_<tag> beside the
template (--cases to name them instead). Values: ScanPointPicker over
<diag-root>/x0_<x0>/, statuses converged and unconverged, field-fit std as
the point's noise; gamma floored at 0 unless --no-clip, as vurs_loop.

Output: run dir <prefix>_FILL_x0_<x0>_<tag> per case, written by
corner_gene_scan.build into the same diagdir <diag-root>/x0_<x0>/<tag>, so the
ky land in the next scanfilesNNNN and merge by ky with the original run
(plot_corners, the picker). Picks go to <prefix>_x0_<x0>_rounds/fill/
(picks.csv, picks.png).

    # pick, save and plot only: no run dirs, no allocation needed
    python -m NSTXU_lithium_study.scripts.vurs_fill --shot 129015 --x0 0.8745 --dry-run
    # in a 4-node interactive allocation: build and run the saved picks
    python -m NSTXU_lithium_study.scripts.vurs_fill --shot 129015 --x0 0.8745 --walltime 00:50:00
    # or sbatch one job per case on the regular queue instead
    python -m NSTXU_lithium_study.scripts.vurs_fill --shot 129015 --x0 0.8745 --walltime 00:50:00 --submit

A real run reuses fill/picks.csv when it exists (--repick to redo), so the
picks checked in the dry run are the ones run, and a rerun after an
allocation ends resumes: existing FILL run dirs are not rewritten, and
interactive_corners.sh skips the ones already done. --n-ky must equal
NODES_PER * 4 (one GPU per ky), as in vurs_loop.
"""

from __future__ import annotations

import glob
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

from NSTXU_lithium_study.scripts import corner_gene_scan as C
from NSTXU_lithium_study.scripts.ped_grid import parse_tag, tag
from NSTXU_lithium_study.scripts.vurs_loop import KNOB, PPN, RUNNER, time_left
from TPED.projects.GENE_pipelines.scripts.sparse_scanner import DEFAULT_AXES
from TPED.projects.GENE_pipelines.src.point_picker import (CONVERGED, UNCONVERGED,
                                                           ScanPointPicker)
from TPED.projects.GENE_pipelines.src.scan_campaign import QoISpec

GRID = 400            # fine log-ky grid the GP is evaluated on
DRAWS = 2000          # posterior draws per peak pick
MIN_GP = 4            # usable values needed for a GP fit
PEAK_FLOOR = 0.05     # a peak pick needs density >= this x the first draw's top; else std
DECIMALS = 3          # as vurs_loop / ScanPointPicker(decimals=3)


# -- picking (pure: arrays in, arrays out) -----------------------------------

def _gp(u, z, noise, kernel=None):
    """sklearn GP on normalised data; kernel fitted unless given (then fixed)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    if kernel is None:
        span = max(np.ptp(u), 1e-3)
        kernel = (ConstantKernel(1., (1e-2, 1e2))
                  * Matern(.3 * span, (.05 * span, 2 * span), nu=2.5)
                  + WhiteKernel(1e-3, (1e-6, 1e-1)))
        gp = GaussianProcessRegressor(kernel, alpha=noise + 1e-8, normalize_y=False,
                                      n_restarts_optimizer=3, random_state=0)
    else:
        gp = GaussianProcessRegressor(kernel, alpha=noise + 1e-8, optimizer=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(u[:, None], z)
    return gp


def pick_line(ky, gamma, std, n_new, n_peak, ky_range, min_ratio=1.06, seed=0):
    """New kymin for one equilibrium.

    ky, gamma, std  the values on disk (std NaN where GENE converged)
    Returns dict: peak (list), std (list), flags (list of str), and for the
    plot the grid, posterior mean/std of gamma and the first peak density.
    """
    ky, gamma = np.asarray(ky, float), np.asarray(gamma, float)
    std = np.asarray(std, float)
    lo, hi = np.log(ky_range[0]), np.log(ky_range[1])
    grid = np.linspace(lo, hi, GRID)
    sep = np.log(min_ratio)
    taken = list(np.log(ky))
    out = dict(peak=[], std=[], flags=[], grid=np.exp(grid), mean=None, sd=None, density=None)

    def allowed():
        ok = np.ones(GRID, bool)
        for t in taken:
            ok &= np.abs(grid - t) >= sep
        rounded = np.round(np.exp(grid), DECIMALS)
        ok &= ~np.isin(rounded, np.round(np.exp(taken), DECIMALS))
        return ok

    def take(j, kind):
        k = round(float(np.exp(grid[j])), DECIMALS)
        out[kind].append(k)
        taken.append(np.log(k))
        return np.log(k)

    if len(ky) < MIN_GP:
        out["flags"].append("fallback")
        for _ in range(n_new):
            ok = allowed()
            if not ok.any():
                break
            gap = np.min(np.abs(grid[:, None] - np.asarray(taken)[None, :]), axis=1)
            gap[~ok] = -np.inf
            take(int(np.argmax(gap)), "std")
        return out

    ymu, ysd = gamma.mean(), gamma.std() or 1.
    z = (gamma - ymu) / ysd
    noise = np.where(np.isfinite(std), (std / ysd) ** 2, 0.)
    u = np.log(ky)
    gp = _gp(u, z, noise)
    kernel = gp.kernel_
    mean, sd = gp.predict(grid[:, None], return_std=True)
    out["mean"], out["sd"] = ymu + ysd * mean, ysd * sd
    if np.argmax(mean) in (0, GRID - 1):
        out["flags"].append("peak_at_edge")

    rng = np.random.default_rng(seed)
    for i in range(n_peak + (n_new - n_peak)):
        kind = "peak" if i < n_peak else "std"
        ok = allowed()
        if not ok.any():
            out["flags"].append("ran_out_of_room")
            break
        if kind == "peak":
            draws = gp.sample_y(grid[:, None], DRAWS, random_state=int(rng.integers(2**31)))
            dens = np.bincount(np.argmax(draws, axis=0), minlength=GRID).astype(float)
            dens = np.convolve(dens, np.ones(5) / 5, mode="same")       # smooth the binning
            if out["density"] is None:
                out["density"] = dens / dens.sum()
                dens_top = dens.max()
            dens[~ok] = 0.
            if dens.max() < PEAK_FLOOR * dens_top:   # peak mass used up: the rest is noise
                kind = "std"
        if kind == "peak":
            j = int(np.argmax(dens))
        else:
            s = gp.predict(grid[:, None], return_std=True)[1]
            s[~ok] = -np.inf
            j = int(np.argmax(s))
        uj = take(j, kind)
        zj = gp.predict(np.array([[uj]]))[0]           # believer: the mean, measured exactly
        u, z, noise = np.append(u, uj), np.append(z, zj), np.append(noise, 0.)
        gp = _gp(u, z, noise, kernel=kernel)
    return out


# -- cases and data -----------------------------------------------------------

def vurs_cases(genedir, prefix, x0lab):
    """Case tags of every vurs_loop run dir <prefix>_r<R>j<J>_<x0lab>_<tag>."""
    tags = set()
    for d in glob.glob(os.path.join(genedir, f"{prefix}_r*j*_{x0lab}_*")):
        t = os.path.basename(d).split(f"_{x0lab}_", 1)[1]
        try:
            parse_tag(t)
        except ValueError:
            continue
        tags.add(t)
    return sorted(tags)


def curves(diag_x0, tags, ky_range, clip):
    """{tag: DataFrame(kymin, gamma, std)} of usable values per case."""
    axes = {**DEFAULT_AXES, "kymin": tuple(ky_range)}
    picker = ScanPointPicker("space_filling", axes, QoISpec("gamma", "max_over_ky"),
                             log_axes=["kymin"], clip_min=0. if clip else None)
    picker.fit(diag_x0)
    runs = picker.runs_
    runs = runs.assign(tag=[tag({KNOB[a]: r[a] for a in KNOB}) for _, r in runs.iterrows()])
    use = runs[runs.status.isin((CONVERGED, UNCONVERGED)) & np.isfinite(runs.qoi)]
    out = {}
    for t in tags:
        sub = use[use.tag == t].sort_values("kymin")
        out[t] = pd.DataFrame({"kymin": sub.kymin.to_numpy(float), "gamma": sub.qoi.to_numpy(float),
                               "std": sub.qoi_std.to_numpy(float)})
    n_all = runs[runs.tag.isin(tags)].groupby("tag").size()
    return out, n_all


def guess(r, ks, what="mean"):
    """GP mean (or sd) of gamma at ky *ks*, before any fill pick; NaN without a GP."""
    ks = np.asarray(ks, float)
    if r[what] is None:
        return np.full(len(ks), np.nan)
    return np.interp(np.log(ks), np.log(r["grid"]), r[what])


def plot(picks, data, results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(picks)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.3 * rows), squeeze=False)
    for ax, (_, p) in zip(axs.flat, picks.iterrows()):
        d, r = data[p["tag"]], results[p["tag"]]
        if r["mean"] is not None:
            ax.fill_between(r["grid"], r["mean"] - 2 * r["sd"], r["mean"] + 2 * r["sd"],
                            color="0.88", lw=0, label="GP ± 2 sd")
            ax.plot(r["grid"], r["mean"], color="0.45", lw=1, label="GP mean")
        ax.errorbar(d.kymin, d.gamma, yerr=np.nan_to_num(d["std"]), fmt="o", ms=4,
                    color="k", zorder=3, label="on disk")
        # each new ky at the GP's best guess of gamma there, numbered in pick order
        for kind, color, marker, lab in (("peak", "C3", "^", "P"), ("std", "C0", "s", "U")):
            ks = np.asarray(r[kind], float)
            if not len(ks):
                continue
            ys = guess(r, ks)
            ax.errorbar(ks, ys, yerr=2 * guess(r, ks, "sd"), fmt=marker, ms=5.5, color=color,
                        mec="k", mew=.5, elinewidth=.8, capsize=1.5, zorder=4,
                        label=("peak pick" if kind == "peak" else "max-std pick") + " (GP guess)")
            for i, (k, y) in enumerate(zip(ks, ys), 1):
                ax.annotate(f"{lab}{i}", (k, y), xytext=(3, 4), textcoords="offset points",
                            fontsize=6, color=color)
        # rug: every ky the case will have, to judge spacing at a glance
        lo = ax.get_ylim()[0]
        ax.plot(d.kymin, np.full(len(d), lo), "|", color="k", ms=9, mew=1)
        ax.plot(r["peak"], np.full(len(r["peak"]), lo), "|", color="C3", ms=9, mew=1.5)
        ax.plot(r["std"], np.full(len(r["std"]), lo), "|", color="C0", ms=9, mew=1.5)
        ax.set_ylim(bottom=lo)
        ax.set_xscale("log")
        ax.set_title(p["tag"] + (f"  [{p['flags']}]" if p["flags"] else ""), fontsize=7)
        ax.set_xlabel(r"$k_y\rho_s$", fontsize=8)
        ax.set_ylabel(r"$\gamma$", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=len(labels))
    fig.suptitle("FILL picks: P = peak, U = max std, numbered in pick order, drawn at the GP "
                 "guess ± 2 sd; ticks on the bottom axis = every ky the case will have", fontsize=9,
                 x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_picks(args, diag_x0, tags, n_ky, n_peak):
    data, n_all = curves(diag_x0, tags, args.kymin_range, not args.no_clip)
    rows, results = [], {}
    for i, t in enumerate(tags):
        d = data[t]
        r = pick_line(d.kymin, d.gamma, d["std"], n_ky, n_peak, args.kymin_range,
                      args.min_ratio, seed=args.seed + i)
        results[t] = r
        if len(r["peak"]) + len(r["std"]) < n_ky:
            r["flags"].append(f"only {len(r['peak']) + len(r['std'])} ky")
        peak_ky = (float(r["grid"][np.argmax(r["mean"])]) if r["mean"] is not None
                   else float(d.kymin[np.argmax(d.gamma)]) if len(d) else np.nan)
        rows.append({"tag": t, **parse_tag(t), "n_on_disk": int(n_all.get(t, 0)),
                     "n_usable": len(d), "ky_on_disk": " ".join(f"{k:g}" for k in d.kymin),
                     "gamma_max_on_disk": float(d.gamma.max()) if len(d) else np.nan,
                     "gp_peak_ky": round(peak_ky, DECIMALS),
                     "peak_picks": " ".join(f"{k:g}" for k in sorted(r["peak"])),
                     "std_picks": " ".join(f"{k:g}" for k in sorted(r["std"])),
                     "peak_gamma_guess": " ".join(f"{g:.3g}" for g in guess(r, sorted(r["peak"]))),
                     "std_gamma_guess": " ".join(f"{g:.3g}" for g in guess(r, sorted(r["std"]))),
                     "kymin": " ".join(f"{k:g}" for k in sorted(r["peak"] + r["std"])),
                     "flags": ",".join(r["flags"])})
    return pd.DataFrame(rows), data, results


# -- main ---------------------------------------------------------------------

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", type=int, default=C.SHOT, choices=sorted(C.SHOTS))
    ap.add_argument("--x0", type=float, default=None, help="radius [Te-top Chapman radius]")
    ap.add_argument("--n-ky", type=int, default=None,
                    help="new kymin per case [NODES_PER * 4]; must equal NODES_PER * 4")
    ap.add_argument("--n-peak", type=int, default=None, help="of those, peak picks [n-ky // 2]")
    ap.add_argument("--kymin-range", type=float, nargs=2, default=(0.1, 10.), metavar=("LO", "HI"),
                    help="kymin range the picks and the GP grid span [0.1 10]")
    ap.add_argument("--min-ratio", type=float, default=1.06,
                    help="closest a new ky may sit to another, as a ratio [1.06]")
    ap.add_argument("--cases", nargs="+", default=None,
                    help="case tags to fill [every vurs_loop run dir of this shot and x0]")
    ap.add_argument("--no-clip", action="store_true", help="fit gamma below 0 as it is")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--walltime", default="00:45:00", help="per GENE run, HH:MM:SS [00:45:00]")
    ap.add_argument("--template", default=C.GPU_TEMPLATE)
    ap.add_argument("--ped-root", default=None)
    ap.add_argument("--diag-root", default=None)
    ap.add_argument("--prefix", default=None, help="vurs run dir prefix [<shot prefix>_vurs]")
    ap.add_argument("--repick", action="store_true", help="redo the picks even if picks.csv exists")
    ap.add_argument("--submit", action="store_true",
                    help="sbatch one job per case (regular queue) instead of running here")
    ap.add_argument("--dry-run", action="store_true",
                    help="pick, write picks.csv and picks.png; build and run nothing")
    args = ap.parse_args(argv)

    d = C.shot_defaults(args.shot)
    x0 = args.x0 if args.x0 is not None else d["x0"][0]
    ped_root = args.ped_root or d["ped_root"]
    diag_root = args.diag_root or d["diag_root"]
    prefix = args.prefix or f"{d['prefix']}_vurs"
    x0lab = C.x0_label((x0,))
    diag_x0 = os.path.join(diag_root, x0lab)
    genedir = os.path.dirname(os.path.normpath(args.template))
    fill_dir = os.path.join(genedir, f"{prefix}_{x0lab}_rounds", "fill")
    nodes_per = int(os.environ.get("NODES_PER", 2))
    n_ky = args.n_ky or nodes_per * PPN
    if not args.submit and n_ky != nodes_per * PPN:
        ap.error(f"--n-ky {n_ky} != NODES_PER * {PPN} = {nodes_per * PPN}: "
                 "interactive_corners.sh runs one GPU per ky")
    n_peak = n_ky // 2 if args.n_peak is None else args.n_peak
    if not 0 <= n_peak <= n_ky:
        ap.error("--n-peak must be in [0, --n-ky]")
    if not 0 < args.kymin_range[0] < args.kymin_range[1]:
        ap.error("--kymin-range needs 0 < LO < HI")
    n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 0))
    if not (args.dry_run or args.submit) and not n_nodes:
        ap.error("not inside an allocation; salloc first (or --dry-run / --submit)")

    picks_csv = os.path.join(fill_dir, "picks.csv")
    if os.path.isfile(picks_csv) and not args.repick:
        picks = pd.read_csv(picks_csv, keep_default_na=False)
        if args.cases:
            picks = picks[picks.tag.isin(args.cases)]
        print(f"reusing {len(picks)} saved pick(s) from {picks_csv} (--repick to redo)")
    else:
        tags = args.cases or vurs_cases(genedir, prefix, x0lab)
        if not tags:
            ap.error(f"no vurs run dirs {prefix}_r*j*_{x0lab}_* in {genedir}; pass --cases")
        print(f"x0 = {x0:g}; {len(tags)} case(s) to fill with {n_ky} kymin each "
              f"({n_peak} peak, {n_ky - n_peak} std) in [{args.kymin_range[0]:g}, "
              f"{args.kymin_range[1]:g}], from {diag_x0}", flush=True)
        picks, data, results = make_picks(args, diag_x0, tags, n_ky, n_peak)
        os.makedirs(fill_dir, exist_ok=True)
        picks.to_csv(picks_csv, index=False)
        plot(picks, data, results, os.path.join(fill_dir, "picks.png"))
        print(f"wrote {picks_csv} and picks.png")
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(picks[["tag", "n_usable", "gamma_max_on_disk", "gp_peak_ky", "peak_picks",
                     "std_picks", "flags"]].to_string(index=False))
    if args.dry_run:
        return 0

    fill_prefix = f"{prefix}_FILL"
    run_dirs = []
    for _, p in picks.iterrows():
        ky = [float(k) for k in str(p.kymin).split()]
        if not ky:
            print(f"  {p.tag}: no picks, skipped")
            continue
        if len(ky) != n_ky and not args.submit:
            print(f"  {p.tag}: {len(ky)} ky, not {n_ky}; its run dir needs "
                  f"{math.ceil(len(ky) / PPN)} node(s)")
        manifest, _ = C.build(args.template, ped_root, diag_root, fill_prefix, x0=(x0,),
                              walltime=args.walltime, kymin=ky, cases=[p.tag], gpu=True,
                              submit=args.submit)
        run_dirs.append(os.path.join(genedir, f"{fill_prefix}_{x0lab}_{p.tag}"))
    if args.submit:
        print(f"\nsubmitted; FILL run dirs {fill_prefix}_{x0lab}_* in {genedir}")
        return 0
    lanes = max(n_nodes // nodes_per, 1)
    waves = math.ceil(len(run_dirs) / lanes)
    left = time_left()
    need = waves * (C.walltime_seconds(args.walltime) + 300)
    print(f"\n{len(run_dirs)} FILL run dir(s), {lanes} lane(s): {waves} wave(s) x {args.walltime}"
          + (f"; {left / 3600:.2f} h left" if left is not None else ""))
    if left is not None and need > left:
        print(f"WARNING: {waves} waves need ~{need / 3600:.2f} h; rerun the same command in a "
              "new allocation to finish (done dirs are skipped)", flush=True)
    subprocess.run(["bash", RUNNER, *run_dirs], check=False,
                   env={**os.environ, "NODES_PER": str(nodes_per)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
