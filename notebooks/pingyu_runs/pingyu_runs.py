"""Pingyu's NSTX ETG / KBM linear GENE runs (132xxx): tables and plots.

Script form of pingyu_etg_kbm_runs.ipynb. Reads the runs under
/global/cfs/cdirs/m2116/TPED_GENE_simulations/{NSTX_ETG, NSTX_KBM, NSTX_KBM_more}
through TPED's GENE readers. He scanned 132xxx discharges with omt / omne
profile scalings, and his grid differs from ours (128 where we use 64), so these
are a reference for both the physics and the resolution.

Writes to --outdir:
    runs.csv            one row per parameters_NNNN: shot, scaling, x0, ky, grid,
                        gamma, omega, fingerprint / Epar ratios, mode_ID
    resolution.csv      every distinct grid per group, with run counts
    x0.csv              x0 and kymin sampled per group / shot / scaling
    modes.csv           mode_ID counts per group / shot / scaling
    errors.csv          every caught read failure (stage, message, run)
    x0.png  profiles.png  gamma_omega_<group>.png  modes.png

shot and the omt / omne factors are parsed from the paths in each parameters
file (iterdb_file, then geomdir / geomfile, then the run dir), in the IFS
naming omt0p7_omne1p1 (p = decimal point). scale_source in runs.csv says which
path matched; 'unscaled?' means none did.

Read failures are expected and are not fixed here: each is caught, recorded in
errors.csv, and the run carries on without that quantity.

    python -m NSTXU_lithium_study.notebooks.pingyu_runs.pingyu_runs
    python -m NSTXU_lithium_study.notebooks.pingyu_runs.pingyu_runs --groups NSTX_KBM --no-modes
    python -m NSTXU_lithium_study.notebooks.pingyu_runs.pingyu_runs --outdir pingyu_out --our-nx0 64
    python -m NSTXU_lithium_study.notebooks.pingyu_runs.pingyu_runs --groups NSTX_KBM --list-dirs
    python -m NSTXU_lithium_study.notebooks.pingyu_runs.pingyu_runs --groups NSTX_KBM \
        --start <run dir name> --limit 100        # 100 consecutive runs, ky traces intact

--match / --start / --limit trim the run list before any file is read. Paths
are kept in sorted order, so consecutive runs of one scanfiles dir (its ky
scan) stay together; --list-dirs shows the dirs and counts to start from.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder as GFF
from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
from TPED.projects.utils.read_iterdb import read_iterdb

PARENT = "/global/cfs/cdirs/m2116/TPED_GENE_simulations"
GROUPS = ("NSTX_ETG", "NSTX_KBM", "NSTX_KBM_more")
#: How deep GeneFileFinder looks for parameters files under each group dir.
MAX_DEPTH = 6
#: Our scans' grid size, for the resolution comparison.
OUR_NX0 = 64

#: Grid columns the resolution table groups on. lx is left out: GENE ties it
#: to ky, so it would split the table into one row per ky.
RES_COLS = ["nx0", "nky0", "nz0", "nv0", "nw0", "lv", "lw", "n_spec"]
PAR_KEYS = ["x0", "kymin", "kx_center", *RES_COLS, "lx", "beta", "coll", "magn_geometry",
            "omt_e", "omn_e", "omt_i", "omn_i", "temp_e", "dens_e"]
#: iterdb quantity names.
PROF_KEYS = {"Te": "TE", "Ti": "TI", "ne": "NE"}
FP_COLS = ["D_e-Chi_e", "Chi_i-Chi_e", "Q_EM_e-|Q_ES|_tot", "D-chi_tot_e"]
EP_COLS = ["Epar_norm", "parity_phi", "parity_apar"]
MODE_C = {"ETG": "tab:blue", "KBM": "tab:red", "MTM": "tab:green", "ITG_TEM": "tab:orange",
          "unidentified": "tab:grey", "failed": "k", "not growing": "lightgrey"}

NUM = r"(\d+(?:[p.]\d+)?)"
SCALE_RE = {"omt_scale": re.compile(rf"omt_?{NUM}", re.I),
            "omne_scale": re.compile(rf"omne?_?{NUM}", re.I)}
SHOT_RE = re.compile(r"(?<!\d)(1[23]\d{4})(?!\d)")

ERRORS = []   # every caught failure: group, run, stage, message


def log_error(group, where, stage, exc):
    msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    # run path out of the message, so errors.csv groups identical failures
    msg = re.sub(r"'[^']*parameters[^']*'", "<run>", msg.strip().splitlines()[-1]).split(" load_errors:")[0]
    ERRORS.append({"group": group, "where": where, "stage": stage, "error": msg[:300]})


def par(ds, key):
    if key not in ds:
        return None
    v = ds[key].values
    v = v.item() if v.size == 1 else v
    return v.decode() if isinstance(v, bytes) else v


def resolve(path, sim_dir):
    """GENE reads relative paths from the run dir, which the diagdir does not
    record: try the scan dir and its parent, else keep the path as written."""
    if not path or os.path.isabs(path):
        return path
    for base in (sim_dir, os.path.dirname(sim_dir)):
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.exists(cand):
            return cand
    return path


def parse_path_tags(*paths):
    out = {"shot": None, "omt_scale": np.nan, "omne_scale": np.nan, "scale_source": None}
    for p in filter(None, paths):
        if out["shot"] is None and (m := SHOT_RE.findall(p)):
            out["shot"] = int(m[-1])
        if out["scale_source"] is None:
            hits = {k: rx.findall(p) for k, rx in SCALE_RE.items()}
            if all(hits.values()):
                for k, h in hits.items():
                    out[k] = float(h[-1].replace("p", "."))
                out["scale_source"] = p
    return out


def scalar(ds, k):
    return float(np.asarray(ds[k].values).squeeze()) if k in ds else np.nan


################################################################################
################################### readers ####################################
################################################################################

def progress(stage, i, n, every=50):
    if i == n or i % every == 0:
        print(f"  {stage}: {i}/{n}", flush=True)


def select(found, match=None, start=None, limit=None):
    """Trim [(group, path)] before anything is read. Paths stay in sorted order,
    so a slice keeps each scanfiles dir's ky points together.

    match  keep paths containing this substring
    start  begin at the first path containing this substring
    limit  keep this many consecutive paths
    """
    if match:
        found = [f for f in found if match in f[1]]
    if start:
        first = next((i for i, f in enumerate(found) if start in f[1]), None)
        if first is None:
            raise SystemExit(f"--start '{start}' matches none of the {len(found)} runs")
        found = found[first:]
    return found[:limit] if limit else found


def discover(roots, max_depth=MAX_DEPTH, match=None, start=None, limit=None, list_dirs=False):
    """One GeneSimulationPlus per parameters file, built one at a time so a bad
    file costs that run, not the whole group. Returns (sims, runs).

    list_dirs prints each run dir (after match / start / limit) with its run
    count and exits, for choosing --start.
    """
    found = []
    for group, root in roots.items():
        try:
            found += [(group, fp) for fp in
                      sorted(GFF(root).find_files(filetype="parameters", max_depth=max_depth))]
        except Exception as exc:
            log_error(group, root, "discover", exc)
    print(f"{len(found)} runs found", flush=True)
    found = select(found, match, start, limit)
    if list_dirs:
        counts = pd.Series([os.path.dirname(fp) for _, fp in found]).value_counts(sort=False)
        print(counts.rename("n_runs").to_string())
        raise SystemExit(0)
    print(f"{len(found)} runs selected" + (f", {found[0][1]} first" if found else ""), flush=True)

    sims, rows = {}, []
    for i, (group, fp) in enumerate(found, 1):
        progress("parameters", i, len(found))
        try:
            sim = GeneSimulationPlus(fp)
            ds = sim.parameters_xarray
        except Exception as exc:
            log_error(group, fp, "parameters", exc)
            continue
        iterdb = resolve(par(ds, "iterdb_file"), sim.directory)
        geomdir, geomfile = par(ds, "geomdir"), par(ds, "geomfile")
        sims[fp] = (group, sim)
        rows.append({"group": group, "run": fp,
                     **parse_path_tags(iterdb, geomdir, geomfile, sim.directory),
                     **{k: par(ds, k) for k in PAR_KEYS},
                     "iterdb_file": iterdb, "geomdir": geomdir, "geomfile": geomfile,
                     "suffix": sim.suffix, "directory": sim.directory})
    runs = pd.DataFrame(rows, columns=["group", "run", "shot", "omt_scale", "omne_scale",
                                       "scale_source", *PAR_KEYS, "iterdb_file", "geomdir",
                                       "geomfile", "suffix", "directory"])
    num = [c for c in PAR_KEYS if c != "magn_geometry"]
    runs[num] = runs[num].apply(pd.to_numeric, errors="coerce")
    runs["scaling"] = [f"omt{t:g}_omne{n:g}" if pd.notna(t) else "unscaled?"
                       for t, n in zip(runs.omt_scale, runs.omne_scale)]
    return sims, runs


def add_omega(sims, runs):
    rows = []
    for i, (fp, (group, sim)) in enumerate(sims.items(), 1):
        progress("omega", i, len(sims))
        try:
            sim.return_sim_xarray(["omega"])
            tree = sim.tree
            if "omega" not in tree.children:
                log_error(group, fp, "omega", tree.attrs["load_errors"].get("omega", "no omega node"))
                continue
            ds = tree["omega"].ds
            # An empty or incomplete omega file (run never finished) reads as an
            # empty Dataset carrying load_warning, not as an error.
            if "gamma" not in ds or "omega" not in ds:
                warning = ds.attrs.get("load_warning", "no gamma/omega in omega file")
                log_error(group, fp, "omega", warning.split(":")[0])
                continue
            rows.append({"run": fp, "gamma": float(ds["gamma"].values), "omega": float(ds["omega"].values)})
        except Exception as exc:
            log_error(group, fp, "omega", exc)
    return runs.merge(pd.DataFrame(rows, columns=["run", "gamma", "omega"]), on="run", how="left")


def add_modes(sims, runs):
    """fingerprints, epar, mode_identifier per growing run, each tried on its
    own so a run missing field still gets its fingerprint ratios."""
    growing = set(runs.run[runs.gamma > 0])
    rows = []
    for fp, (group, sim) in sims.items():
        if fp not in growing:
            continue
        progress("modes", len(rows) + 1, len(growing), every=20)
        row = {"run": fp}
        for name, cols in (("fingerprints", FP_COLS), ("epar", EP_COLS), ("mode_identifier", ["mode_ID"])):
            try:
                node = getattr(sim.analysis, name)().ds
                for k in cols:
                    row[k] = str(node[k].values) if k == "mode_ID" else scalar(node, k)
            except Exception as exc:
                log_error(group, fp, name, exc)
        rows.append(row)
    ids = pd.DataFrame(rows, columns=["run", *FP_COLS, *EP_COLS, "mode_ID"])
    runs = runs.merge(ids, on="run", how="left")
    fallback = pd.Series(np.where(runs.gamma > 0, "failed", "not growing"), index=runs.index)
    runs["mode_ID"] = runs["mode_ID"].fillna(fallback)
    return runs


def read_profiles(runs):
    """iterdb path -> {Te: (rho, values), ...} for every iterdb the runs point at."""
    profiles = {}
    for path, d in runs.dropna(subset=["iterdb_file"]).groupby("iterdb_file"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):   # read_iterdb prints its point count
                rhot, prof, _units = read_iterdb(path)
            profiles[path] = {lab: (np.asarray(rhot[k]), np.asarray(prof[k]))
                              for lab, k in PROF_KEYS.items() if k in prof}
        except Exception as exc:
            log_error(d.group.iloc[0], path, "iterdb", exc)
    return profiles


################################################################################
#################################### tables ####################################
################################################################################

def resolution_table(runs, our_nx0=OUR_NX0):
    res = runs.groupby(["group", *RES_COLS], dropna=False).size().rename("n_runs").reset_index()
    res["nx0_vs_ours"] = np.where(res.nx0 == our_nx0, "same", f"differs from {our_nx0}")
    return res


def x0_table(runs):
    return (runs.groupby(["group", "shot", "scaling"], dropna=False)
            .agg(x0=("x0", lambda s: sorted(s.dropna().round(4).unique())),
                 kymin=("kymin", lambda s: sorted(s.dropna().round(3).unique())),
                 n_runs=("run", "size")))


def error_table():
    errors = pd.DataFrame(ERRORS, columns=["group", "where", "stage", "error"])
    summary = errors.groupby(["stage", "error"]).agg(
        n=("where", "size"), groups=("group", lambda g: sorted(set(g))), example=("where", "first"))
    return errors, summary


################################################################################
#################################### plots #####################################
################################################################################

def plot_x0(runs, path):
    n = max(runs.groupby(["group", "shot"], dropna=False).ngroups, 1)
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.45 * n))
    labels = []
    for i, ((group, shot), d) in enumerate(runs.groupby(["group", "shot"], dropna=False)):
        ax.scatter(d.x0, np.full(len(d), i), c=d.omt_scale, cmap="viridis",
                   vmin=runs.omt_scale.min(), vmax=runs.omt_scale.max(), s=40, edgecolor="k", lw=0.3)
        labels.append(f"{group} | {shot}")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(r"x0 ($\rho_{tor}$)")
    ax.set_title("x0 sampled (colour = omt scale)")
    if ax.collections:
        fig.colorbar(ax.collections[0], ax=ax, label="omt scale")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_profiles(runs, profiles, path):
    """One row per shot, lines per scaling, dashed at the x0 values sampled."""
    prof_runs = runs[runs.iterdb_file.isin(profiles)]
    shots = sorted(prof_runs.shot.dropna().unique())
    if not shots:
        print("profiles: no readable iterdb, profiles.png not written")
        return
    fig, axes = plt.subplots(len(shots), len(PROF_KEYS), figsize=(4 * len(PROF_KEYS), 3 * len(shots)),
                             squeeze=False, sharex=True)
    for r, shot in enumerate(shots):
        d_shot = prof_runs[prof_runs.shot == shot]
        for (scaling, ipath), _d in d_shot.groupby(["scaling", "iterdb_file"]):
            for c, lab in enumerate(PROF_KEYS):
                if lab in profiles[ipath]:
                    axes[r, c].plot(*profiles[ipath][lab], label=scaling, lw=1)
        for c, lab in enumerate(PROF_KEYS):
            for x0 in d_shot.x0.dropna().unique():
                axes[r, c].axvline(x0, ls="--", c="grey", lw=0.6)
            axes[r, c].set_title(f"{shot}  {lab}")
        axes[r, 0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\rho_{tor}$")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gamma_omega(runs, outdir):
    """Per group: gamma (top) and omega (bottom) vs ky, one column per shot,
    one curve per (scaling, x0)."""
    for group, dg in runs.dropna(subset=["gamma"]).groupby("group"):
        shots = sorted(dg.shot.dropna().unique()) or [None]
        fig, axes = plt.subplots(2, len(shots), figsize=(5 * len(shots), 7), squeeze=False, sharex=True)
        for c, shot in enumerate(shots):
            d_shot = dg if shot is None else dg[dg.shot == shot]
            for (scaling, x0), d in d_shot.groupby(["scaling", "x0"], dropna=False):
                d = d.sort_values("kymin")
                lab = f"{scaling}, x0={x0:.3f}"
                axes[0, c].plot(d.kymin, d.gamma, "o-", ms=3, lw=1, label=lab)
                axes[1, c].plot(d.kymin, d.omega, "o-", ms=3, lw=1, label=lab)
            axes[0, c].set_title(f"{group}  {shot}")
            axes[1, c].set_xlabel(r"$k_y\rho_s$")
            axes[0, c].legend(fontsize=6)
        axes[0, 0].set_ylabel(r"$\gamma$")
        axes[1, 0].set_ylabel(r"$\omega$")
        log_ky = dg.kymin.max() / max(dg.kymin.min(), 1e-12) > 20
        for ax in axes.flat:
            ax.axhline(0, c="k", lw=0.5)
            if log_ky:
                ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"gamma_omega_{group}.png"), dpi=130)
        plt.close(fig)


def plot_modes(runs, path):
    groups = sorted(runs.group.unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, group in zip(axes[0], groups):
        d = runs[(runs.group == group) & runs.gamma.notna()]
        for mode, dm in d.groupby("mode_ID"):
            ax.scatter(dm.kymin, dm.gamma, s=18, c=MODE_C.get(mode, "m"), label=f"{mode} ({len(dm)})")
        ax.set_title(group)
        ax.set_xlabel(r"$k_y\rho_s$")
        ax.axhline(0, c="k", lw=0.5)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(r"$\gamma$")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


################################################################################
##################################### run ######################################
################################################################################

def run(roots, outdir, max_depth=MAX_DEPTH, our_nx0=OUR_NX0, modes=True,
        match=None, start=None, limit=None, list_dirs=False):
    ERRORS.clear()
    sims, runs = discover(roots, max_depth, match, start, limit, list_dirs)
    os.makedirs(outdir, exist_ok=True)
    if runs.empty:
        print(error_table()[1].to_string())
        return runs
    runs = add_omega(sims, runs)
    runs = add_modes(sims, runs) if modes else runs.assign(mode_ID="not run")
    profiles = read_profiles(runs)

    summary = runs.groupby("group").agg(
        n_runs=("run", "size"), shots=("shot", lambda s: sorted(s.dropna().unique())),
        n_scalings=("scaling", "nunique"), with_omega=("gamma", "count"),
        growing=("gamma", lambda g: int((g > 0).sum())))
    res = resolution_table(runs, our_nx0)
    x0s = x0_table(runs)
    mode_counts = pd.crosstab([runs.group, runs.shot, runs.scaling], runs.mode_ID)
    errors, err_summary = error_table()

    runs.sort_values(["group", "shot", "scaling", "x0", "kymin"]).to_csv(
        os.path.join(outdir, "runs.csv"), index=False)
    res.to_csv(os.path.join(outdir, "resolution.csv"), index=False)
    x0s.to_csv(os.path.join(outdir, "x0.csv"))
    mode_counts.to_csv(os.path.join(outdir, "modes.csv"))
    errors.to_csv(os.path.join(outdir, "errors.csv"), index=False)

    plot_x0(runs, os.path.join(outdir, "x0.png"))
    plot_profiles(runs, profiles, os.path.join(outdir, "profiles.png"))
    plot_gamma_omega(runs, outdir)
    if modes:
        plot_modes(runs, os.path.join(outdir, "modes.png"))

    with pd.option_context("display.width", 200, "display.max_colwidth", 80, "display.max_rows", 200):
        for title, table in (("runs per group", summary), ("resolution", res), ("x0 sampled", x0s),
                             ("modes", pd.crosstab([runs.group, runs.shot], runs.mode_ID, margins=True)),
                             ("read errors", err_summary.drop(columns="example"))):
            print(f"\n=== {title}\n{table.to_string()}")
    print(f"\nwritten to {os.path.abspath(outdir)}")
    return runs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--parent", default=PARENT, help="dir holding the group dirs")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), help="group dirs under --parent")
    ap.add_argument("--outdir", default="pingyu_runs_out")
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH,
                    help="how deep to look for parameters files under each group")
    ap.add_argument("--our-nx0", type=int, default=OUR_NX0, help="our nx0, for the resolution flag")
    ap.add_argument("--no-modes", action="store_true",
                    help="skip fingerprints / Epar / mode_identifier (needs nrg, field, geometry)")
    ap.add_argument("--match", default=None, metavar="SUBSTR",
                    help="keep only runs whose parameters path contains this")
    ap.add_argument("--start", default=None, metavar="SUBSTR",
                    help="begin at the first run (sorted path order) whose path contains this")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep this many consecutive runs from --start")
    ap.add_argument("--list-dirs", action="store_true",
                    help="print each run dir with its run count and exit (to choose --start)")
    args = ap.parse_args()
    # The readers log a full traceback per failed file; those land in errors.csv instead.
    logging.getLogger("TPED").setLevel(logging.CRITICAL)
    run({g: os.path.join(args.parent, g) for g in args.groups}, args.outdir,
        args.max_depth, args.our_nx0, not args.no_modes, args.match, args.start, args.limit,
        args.list_dirs)
