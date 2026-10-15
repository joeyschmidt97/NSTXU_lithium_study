"""Plots for the 16-corner linear GENE scan (corner_gene_scan.py): growth, profiles, s-alpha, locals.

Reads every corner corner_gene_scan.py launched,

    GENE output   <diag-root>/x0_<x0>/<tag>/scanfilesNNNN/{parameters,omega}_000K
                  (every scanfiles dir, merged by ky)
    profiles      <ped-root>/<tag>/profiles.iterdb   (what GENE read, prof_type -2)

into one xarray Dataset with dims (corner, ky, rho_tor), and writes to <out>/

    corners_growth.png          gamma and omega vs ky, all corners
    corners_growth_field.png    (--field-all) the field_omega fit of every run, converged
                                or not, gamma and omega +-1 std vs ky, GENE's value beside
    corners_profiles.png        ne, Te, ni, Ti and their -dln/drho_tor, x0 marked
    corners_profiles_ped.png    the same, zoomed to PED_ZOOM
    corners_mode_scaling.png    (--modes) mode per corner and ky as a table, and the
                                mode mix at each knob level; + corners_mode_scaling.csv
    corners_s_alpha.png         shat vs alpha_MHD at x0: each corner a small ring of
                                its converged ky (unconverged ones left out), each
                                coloured by that ky's gamma (marker = its mode under
                                --modes), numbered, with a key beside
    corners_local_params.png    the local parameters that vary most: % from the corner
                                mean, or the absolute difference for those that change sign
    corners_local_table.png     their values per corner, as a table
    corners_local_params.csv    every local parameter per corner (+ gamma_max)

Every legend lists a corner as its knob factors in the order (Tew, new, Teh, neh),
e.g. (0.8, 1.2, 0.8, 0.8). Colour hue is the width pair (Tew, new), shade
the height pair (Teh, neh); line style is the height pair too (solid,
dashed, dotted, dash-dot) and the marker neh, so the four corners sharing a
hue never share a line style and no corner is told apart by colour alone.

    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --out ./corner_plots
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --spread 0.02 --max-params 16
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737 --modes
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.8447 --modes --field-fits
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737 --field-all

With --field-all, corners_growth_field.png draws every row of the CSV that
field_omega.py --csv wrote (default <diag-root>/x0_<x0>/field_omega_fits.csv):
the field fit of every run with +-1 std bars, filled where GENE converged and
hollow where it did not, with GENE's converged value as a black tick at the
same ky. It shows where the unconverged runs are heading and how far the fit
sits from GENE where both exist. ky is log-scaled and each corner nudged
sideways so the bars do not overlap. corners_growth.png is unchanged.

With --modes each ky GENE converged (or the field estimate filled) is
classified by TPED's mode_identifier (GENE_sim_analysis: fingerprint transport
ratios from nrg, Epar and parities from field + geometry, omega/omega_star),
drawn with that mode's marker in corners_growth.png, and listed with the
inputs behind each label in corners_modes.csv. The fingerprint diffusivity is
Gamma / omn: where omn ~ 0 (the ne top) the D ratios, and so the ETG, KBM and
ITG/TEM checks, are unreliable; read the inputs before the label.

With --field-fits, a ky that GENE left 0/0 but field_omega.py recorded a
fit for with verdict 'decaying' (gamma + 2 std < 0) or 'ambiguous' (std too
large to tell a slow-growing mode from beating damped ones) is drawn at its
fitted gamma/omega with +-1 std error bars: green-edged for decaying, faint
grey for ambiguous, never joined into the corner's line, never classified,
and left out of gamma_max when ambiguous. Without the flag they stay x.

A ky whose omega file carries a TPED field estimate below GENE's 0/0 line
(field_omega.py --write) plots as a diamond with ±1 std error bars.
An empty omega file, or one holding gamma = omega = 0 exactly (GENE's
placeholder when a run ends without a converged growth rate: time limit, or
a decaying mode), reads as NaN and is drawn as an x on the zero line; a
missing omega or parameters file is NaN and left out. The ky axis is every
kymin found across the corners, so scans with different ky lists combine. A corner with neither GENE output nor profiles is
reported and skipped. Nothing here raises on a partial scan.

alpha_MHD = q0^2 * major_R * dpdx_pm, all in GENE units (major_R = R0/Lref,
dpdx_pm = -dbeta/dx with x in Lref, as GENE writes it in the run's parameters
file). When dpdx_pm is not there it is rebuilt as
beta * sum_s dens_s * temp_s * (omn_s + omt_s).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import warnings

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D

import pandas as pd

from TPED.projects.GENE_sim_reader.src.GENE_sim_data import GeneSimulation
from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder
from TPED.projects.utils.read_iterdb import read_iterdb
from NSTXU_lithium_study.scripts.corner_gene_scan import (DIAG_ROOT, KYMIN, PED_ROOT, X0,
                                                         corners, tag, x0_label)
from NSTXU_lithium_study.scripts.corners.field_omega.field_omega import csv_path


def default_out(x0=X0) -> str:
    return os.path.join(DIAG_ROOT, x0_label(x0), "plots")


#: Legend order of the knobs; the corner tag (ped_grid.tag) uses Teh, Tew, neh, new.
ORDER = ("Tew", "new", "Teh", "neh")
LEGEND_TITLE = "(" + ", ".join(ORDER) + ")"

#: Pedestal window of the zoomed profile figure (rho_tor).
PED_ZOOM = (0.6, 1.0)

#: A local parameter is "varying" when (max - min) / max|value| over the corners
#: exceeds this. max|value|, not |mean|: gradients like omn_e cross zero, and a
#: mean near zero would inflate their spread.
SPREAD = 0.05
MAX_PARAMS = 12

#: Local parameters read from each run's parameters file (GENE names; species
#: quantities carry TPED's _<species> suffix, e.g. omn_e, omt_i).
PARAM_KEYS = ("beta", "coll", "debye2", "q0", "shat", "trpeps", "major_R", "minor_r",
              "rhostar", "dpdx_pm", "Tref", "nref", "Bref", "Lref", "nu_ei",
              "nustar_i", "nustar_e")
SPECIES_PARAM = re.compile(r"^(omn|omt|temp|dens)_[A-Za-z]+$")

#: iterdb signal -> (variable, scale to plot units, unit label)
ITERDB_VARS = {"NE": ("ne", 1e-19, "10$^{19}$ m$^{-3}$"), "TE": ("Te", 1e-3, "keV"),
               "NM1": ("ni", 1e-19, "10$^{19}$ m$^{-3}$"), "TI": ("Ti", 1e-3, "keV")}

# Hues for the (Tew, new) width pair -- categorical slots 1, 2, 3, 7 of the
# validated palette; shades step the (Teh, neh) height pair dark to light.
WIDTH_HUES = {(0.8, 0.8): "#2a78d6", (0.8, 1.2): "#eb6834",
              (1.2, 0.8): "#1baf7a", (1.2, 1.2): "#c4338a"}
HEIGHT_SHADE = {(0.8, 0.8): -0.35, (0.8, 1.2): -0.12, (1.2, 0.8): 0.15, (1.2, 1.2): 0.38}
#: Line style per height pair (Teh, neh): solid, dashed, dotted, dash-dot.
HEIGHT_LS = {(0.8, 0.8): (0, ()), (0.8, 1.2): (0, (6, 2.5)), (1.2, 0.8): (0, (1.2, 1.8)),
             (1.2, 1.2): (0, (7, 2, 1.5, 2))}
NEH_MARKER = {0.8: "o", 1.2: "s"}
GAMMA_CMAP = LinearSegmentedColormap.from_list("gamma", ["#008300", "#f0efec", "#e34948"])
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
DPI = 150
#: Which reading of a ky wins when two scanfiles dirs hold it.
RANK = {"ok": 2, "field": 1, "decaying": 0.5, "ambiguous": 0.5}
#: --field-fits: recorded field fits that did not pass, by verdict.
FIT_STYLE = {"decaying": {"mec": "#008300", "alpha": 1.0, "label": "decaying fit (γ+2σ<0) ±1σ"},
             "ambiguous": {"mec": "#9b9a92", "alpha": 0.55,
                           "label": "ambiguous fit (σ too large) ±1σ"}}

#: --modes: TPED mode_identifier label -> marker, and the inputs it decided on,
#: (TPED node, variable) -> column name.
MODE_MARKER = {"ITG_TEM": "o", "ETG": "^", "KBM": "s", "MTM": "v", "unidentified": "*"}
#: Mode colours: validated categorical slots 7, 4, 1, 2 (violet, yellow, blue, orange)
#: of the reference palette; yellow is under 3:1 on the surface, so every mode
#: panel also carries its label (legend, table text).
MODE_COLOR = {"ITG_TEM": "#6250d6", "ETG": "#eda100", "KBM": "#2a78d6", "MTM": "#eb6834",
              "unidentified": "#9b9a92"}
#: Gaussian width (in ky) of the smoothed mode fraction under the growth plot.
MODE_KDE_BW = 0.12
MODE_SHORT = {"ITG_TEM": "IT", "ETG": "ETG", "KBM": "KBM", "MTM": "MTM", "unidentified": "?"}
MODE_INPUTS = {("fingerprints", "D_e-Chi_e"): "De_Xe", ("fingerprints", "Chi_i-Chi_e"): "Xi_Xe",
               ("fingerprints", "Q_EM_e-|Q_ES|_tot"): "QEM_e", ("epar", "Epar_norm"): "Epar_norm",
               ("epar", "parity_apar"): "parity_apar", ("epar", "parity_phi"): "parity_phi"}


# ----------------------------------------------------------------------------- labels, style

def label(p) -> str:
    return "(" + ", ".join(f"{p[k]:g}" for k in ORDER) + ")"


def ordered(points):
    return sorted(points, key=lambda p: tuple(p[k] for k in ORDER))


def shade(hex_color: str, amount: float):
    """Mix toward black (amount < 0) or white (amount > 0)."""
    c = np.array(to_rgb(hex_color))
    return tuple(c * (1 + amount) if amount < 0 else c + (1 - c) * amount)


def style(p) -> dict:
    return {"color": shade(WIDTH_HUES[(p["Tew"], p["new"])], HEIGHT_SHADE[(p["Teh"], p["neh"])]),
            "ls": HEIGHT_LS[(p["Teh"], p["neh"])], "marker": NEH_MARKER[p["neh"]]}


def point_of(ds, corner) -> dict:
    return {k: float(ds[k].sel(corner=corner)) for k in ORDER}


# ----------------------------------------------------------------------------- GENE output

def gene_runs(corner_dir: str) -> list[str]:
    """parameters_NNNN of every scanfiles dir under corner_dir (GeneFileFinder), oldest
    dir first, or []. A later scan (e.g. extra ky) lands in the next scanfilesNNNN."""
    try:
        found = GeneFileFinder(corner_dir).find_files("parameters")
    except FileNotFoundError:
        return []
    found = [f for f in found if re.search(r"parameters_\d{4}$", f)]
    return sorted(found)


def read_run(par_path: str, field_fits: bool = False) -> dict:
    """One scan member through TPED's GeneSimulation: kymin, gamma, omega, status, locals.

    field_fits: GENE's 0/0 with a recorded 'decaying'/'ambiguous' field fit reads
    as that status with the fit's value and std, instead of 'zero'."""
    sim = GeneSimulation(par_path)
    # No omega file at all: skip the load, TPED would log a traceback for the None path.
    tree = sim.return_sim_xarray(["omega"] if sim.omega_filepath else [])
    par = tree["parameters"].ds
    rec = {"kymin": float(par["kymin"]), "gamma": np.nan, "omega": np.nan, "status": "missing",
           "gamma_std": np.nan, "omega_std": np.nan}
    if "omega" in tree.children:
        om = tree["omega"].ds
        rec["status"] = "ok" if "gamma" in om else "empty"
        if "gamma" in om:
            rec["gamma"], rec["omega"] = float(om["gamma"]), float(om["omega"])
            if om.attrs.get("source") == "field":     # TPED field estimate under GENE's 0/0 line
                rec["status"] = "field"
                rec["gamma_std"] = float(om["gamma_std"]) if "gamma_std" in om else np.nan
                rec["omega_std"] = float(om["omega_std"]) if "omega_std" in om else np.nan
            elif rec["gamma"] == 0 and rec["omega"] == 0:   # GENE's "not converged" placeholder
                rec["gamma"] = rec["omega"] = np.nan
                rec["status"] = "zero"
                verdict = om.attrs.get("field_verdict")
                if field_fits and verdict in FIT_STYLE and "gamma_field" in om:
                    rec["status"] = verdict
                    rec["gamma"], rec["omega"] = float(om["gamma_field"]), float(om["omega_field"])
                    for k in ("gamma", "omega"):
                        rec[f"{k}_std"] = (float(om[f"{k}_field_std"]) if f"{k}_field_std" in om
                                           else np.nan)
    rec["params"] = {k: float(par[k]) for k in par.data_vars
                     if (k in PARAM_KEYS or SPECIES_PARAM.match(k))
                     and np.issubdtype(par[k].dtype, np.number)}
    return rec


def alpha_mhd(pr: dict) -> float:
    """q0^2 * major_R * dpdx_pm; dpdx_pm rebuilt from beta and the species gradients if absent."""
    dpdx = pr.get("dpdx_pm", np.nan)
    if not np.isfinite(dpdx) or dpdx < 0:      # GENE writes -1/-2 for "compute it" in inputs
        specs = {k.split("_", 1)[1] for k in pr if k.startswith("omn_")}
        try:
            dpdx = pr["beta"] * sum(pr[f"dens_{s}"] * pr[f"temp_{s}"]
                                    * (pr[f"omn_{s}"] + pr[f"omt_{s}"]) for s in specs)
        except KeyError:
            return np.nan
    return pr.get("q0", np.nan) ** 2 * pr.get("major_R", np.nan) * dpdx


def gene_corner(corner_dir: str, field_fits: bool = False):
    """({kymin: (gamma, omega, status, gamma_std, omega_std, parameters path)},
    local parameters) of one corner, or None."""
    runs = gene_runs(corner_dir)
    if not runs:
        return None
    by_ky: dict[float, tuple] = {}
    params: dict[str, list[float]] = {}
    for path in runs:
        try:
            rec = read_run(path, field_fits)
        except Exception as exc:                # one bad member never sinks the corner
            print(f"    unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        # The same ky in a later scanfiles dir replaces the earlier one unless the
        # earlier is better: GENE converged > TPED field estimate > neither.
        k = round(rec["kymin"], 6)
        if k not in by_ky or RANK.get(rec["status"], 0) >= RANK.get(by_ky[k][2], 0):
            by_ky[k] = (rec["gamma"], rec["omega"], rec["status"], rec["gamma_std"],
                        rec["omega_std"], path)
        for k, v in rec["params"].items():
            params.setdefault(k, []).append(v)
    local = {}
    for k, vals in params.items():                # equilibrium values: identical at every ky
        vals = np.array(vals)
        if np.ptp(vals) > 1e-6 * max(abs(vals).max(), 1e-30):
            print(f"    {os.path.basename(corner_dir)}: {k} differs across ky "
                  f"({vals.min():.4g}..{vals.max():.4g}); using the median")
        local[k] = float(np.median(vals))
    if local:
        local["alpha_MHD"] = alpha_mhd(local)
    return by_ky, local


def classify(par_path: str) -> dict:
    """TPED mode_identifier on one scan member: {'mode', inputs..., 'w_wstar'}."""
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    tped_log = logging.getLogger("TPED")
    level = tped_log.level
    tped_log.setLevel(logging.CRITICAL)             # a missing file is reported once, below
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = GeneSimulationPlus(par_path)
            node = sim.analysis.mode_identifier()
    finally:
        tped_log.setLevel(level)
    out = {"mode": str(node.ds["mode_ID"].values)}
    for (src, var), col in MODE_INPUTS.items():
        ds_ = sim.tree[src].ds
        out[col] = float(np.asarray(ds_[var].values).squeeze()) if var in ds_ else np.nan
    wstar = float(node.ds["omega_star"].values)
    out["w_wstar"] = float(sim.tree["omega"].ds["omega"].values) / wstar if wstar else np.nan
    return out


def add_modes(ds: xr.Dataset) -> xr.Dataset:
    """mode (corner, ky) and its inputs, for every ky with a growth rate (GENE or field)."""
    cols = ["mode", *MODE_INPUTS.values(), "w_wstar"]
    vals = {c: np.full(ds.gamma.shape, "" if c == "mode" else np.nan, dtype=object if c == "mode"
                       else float) for c in cols}
    for i, c in enumerate(ds.corner.values):
        for j, ky in enumerate(ds.ky.values):
            if ds.status.values[i, j] not in ("ok", "field"):
                continue
            try:
                res = classify(str(ds.path.values[i, j]))
            except Exception as exc:              # missing field/geometry: no label, no crash
                msg = str(exc).split(" load_errors")[0].splitlines()[0][:160]
                print(f"    {c} ky={ky:g}: no mode ({type(exc).__name__}: {msg})")
                continue
            for col in cols:
                vals[col][i, j] = res[col]
        labels = [f"{ky:g} {m}" for ky, m in zip(ds.ky.values, vals["mode"][i]) if m]
        print(f"--- {c}: " + ", ".join(labels))
    for col in cols:
        ds[f"mode_{col}" if col != "mode" else "mode"] = (("corner", "ky"), vals[col])
    ds["mode"] = ds["mode"].astype(str)
    return ds


def write_modes_csv(ds, out, written):
    path = os.path.join(out, "corners_modes.csv")
    cols = ["status", "gamma", "gamma_std", "omega", "omega_std", "mode",
            *(f"mode_{c}" for c in MODE_INPUTS.values()),
            "mode_w_wstar"]
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky"] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, ky in enumerate(ds.ky.values):
                row = [str(ds[k].values[i, j]) if ds[k].dtype.kind in "OUS"
                       else f"{ds[k].values[i, j]:.6g}" for k in cols]
                fh.write(",".join([c, *(f"{ds[k].values[i]:g}" for k in ORDER), f"{ky:g}"] + row)
                         + "\n")
    written.append(path)


# ----------------------------------------------------------------------------- profiles

def iterdb_profiles(case_dir: str):
    """{var: (rho_tor, values in plot units)} from <case_dir>/profiles.iterdb, or None."""
    path = os.path.join(case_dir, "profiles.iterdb")
    if not os.path.isfile(path):
        return None
    with contextlib.redirect_stdout(io.StringIO()):       # read_iterdb prints the point count
        rhot, prof, _ = read_iterdb(path)
    return {name: (np.asarray(rhot[sig]), np.asarray(prof[sig]) * scale)
            for sig, (name, scale, _) in ITERDB_VARS.items() if sig in prof}


# ----------------------------------------------------------------------------- dataset

def load_dataset(diag_root: str = DIAG_ROOT, ped_root: str = PED_ROOT, x0=X0,
                 modes=False, field_fits=False) -> xr.Dataset:
    """Every corner with GENE output or profiles, as one Dataset (corner, ky, rho_tor).

    corner is the ped_grid tag; Tew/new/Teh/neh ride along as coordinates, so
    ds.set_index(corner=["Tew", "new", "Teh", "neh"]).unstack("corner") gives
    the 4D box.
    """
    gene, prof, pts = {}, {}, []
    for p in ordered(corners()):
        name = tag(p)
        g = gene_corner(os.path.join(diag_root, x0_label(x0), name), field_fits)
        pr = iterdb_profiles(os.path.join(ped_root, name))
        if g is None and pr is None:
            print(f"--- {name}: SKIP, no GENE output and no profiles.iterdb")
            continue
        if g is None:
            print(f"--- {name}: no GENE output (profiles only)")
        if pr is None:
            print(f"--- {name}: no profiles.iterdb (GENE only)")
        if g is not None:
            st = {k: v[2] for k, v in sorted(g[0].items())}
            print(f"--- {name}: {sum(s == 'ok' for s in st.values())}/{len(st)} ky converged"
                  + "".join(f"; ky={k:g} {s}" for k, s in st.items() if s != "ok"))
            gene[name] = g
        if pr is not None:
            prof[name] = pr
        pts.append(p)
    if not pts:
        raise SystemExit(f"no corner found under {diag_root} or {ped_root}")

    names = [tag(p) for p in pts]
    ky = np.array(sorted({k for g in gene.values() for k in g[0]}) or KYMIN, dtype=float)

    def on_ky(n, j, fill):
        row = gene.get(n, ({}, {}))[0]
        return [row[k][j] if k in row else fill for k in ky]

    ds = xr.Dataset(coords={"corner": names, "ky": ("ky", ky, {"long_name": "kymin rho_ref"}),
                            **{k: ("corner", [p[k] for p in pts]) for k in ORDER},
                            "label": ("corner", [label(p) for p in pts])})
    ds["gamma"] = (("corner", "ky"), [on_ky(n, 0, np.nan) for n in names], {"units": "c_ref/L_ref"})
    ds["omega"] = (("corner", "ky"), [on_ky(n, 1, np.nan) for n in names], {"units": "c_ref/L_ref"})
    ds["status"] = (("corner", "ky"), [on_ky(n, 2, "missing") for n in names])
    ds["gamma_std"] = (("corner", "ky"), [on_ky(n, 3, np.nan) for n in names],
                       {"long_name": "std of the TPED field estimate (status 'field' only)"})
    ds["omega_std"] = (("corner", "ky"), [on_ky(n, 4, np.nan) for n in names])
    ds["path"] = (("corner", "ky"), [on_ky(n, 5, "") for n in names])
    keys = sorted({k for g in gene.values() for k in g[1]})
    for k in keys:
        ds[k] = ("corner", [gene[n][1].get(k, np.nan) if n in gene else np.nan for n in names])
    if "alpha_MHD" in ds:
        ds["alpha_MHD"].attrs["formula"] = "q0^2 * major_R * dpdx_pm (GENE units)"
    trusted = ds["gamma"].where(ds["status"] != "ambiguous")   # an ambiguous fit is no maximum
    finite = trusted.notnull().any("ky")
    ds["gamma_max"] = trusted.fillna(-np.inf).max("ky").where(finite)
    ds["ky_at_max"] = xr.DataArray(ky[trusted.fillna(-np.inf).argmax("ky").values],
                                   dims="corner").where(finite)

    if prof:                                       # profiles on the first corner's grid
        ref = next(iter(prof.values()))
        grid = ref.get("ne", next(iter(ref.values())))[0]
        ds = ds.assign_coords(rho_tor=grid)
        for var in [v for v, *_ in ITERDB_VARS.values()]:
            rows = []
            for n in names:
                r, y = prof.get(n, {}).get(var, (None, None))
                rows.append(np.interp(grid, r, y) if r is not None else np.full(grid.size, np.nan))
            arr = np.array(rows)
            if np.isnan(arr).all():
                continue
            ds[var] = (("corner", "rho_tor"), arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                ds[f"omg_{var}"] = (("corner", "rho_tor"),
                                    -np.gradient(arr, grid, axis=1) / arr,
                                    {"long_name": f"-dln{var}/drho_tor"})
    ds.attrs.update(x0=float(x0[0]), diag_root=diag_root, ped_root=ped_root,
                    field_fits=int(field_fits))
    if modes and gene:
        print("\nmode_identifier (TPED) per ky:")
        ds = add_modes(ds)
    return ds


# ----------------------------------------------------------------------------- plots

def tidy(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def corner_handles(ds):
    return [Line2D([], [], lw=1.8, ms=5, mfc="none", **style(point_of(ds, c)), label=str(lab))
            for c, lab in zip(ds.corner.values, ds.label.values)]


def legend_right(fig, handles, extra=()):
    fig.legend(handles=list(handles) + list(extra), title=LEGEND_TITLE, loc="center right",
               frameon=False, fontsize=8, title_fontsize=9)


def save(fig, out, name, written):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)


def plot_growth(ds, out, written):
    modes = "mode" in ds
    if modes:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True,
                                 height_ratios=(3, 1.5))
        axes, mode_axes = grid[0], grid[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    unconv = fielded = False
    fits_seen = set()
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        status = ds.status.sel(corner=c).values
        for ax, var in zip(axes, ("gamma", "omega")):
            line = ds[var].sel(corner=c).where(~np.isin(status, list(FIT_STYLE)))
            for verdict, fs in FIT_STYLE.items():
                sel = status == verdict
                if sel.any():
                    fits_seen.add(verdict)
                    ax.errorbar(ds.ky.values[sel], ds[var].sel(corner=c).values[sel],
                                yerr=ds[f"{var}_std"].sel(corner=c).values[sel], fmt="o", ms=5,
                                mfc="none", mec=fs["mec"], mew=1.4, color=st["color"],
                                alpha=fs["alpha"], capsize=2.5, lw=0.9, zorder=4)
            if modes:                   # line only; each point gets its mode's marker below
                ax.plot(ds.ky, line, lw=1.4, color=st["color"], ls=st["ls"])
                y = ds[var].sel(corner=c).values
                for j, m in enumerate(ds["mode"].sel(corner=c).values):
                    if m and np.isfinite(y[j]):
                        filled = status[j] == "ok"
                        ax.plot(ds.ky.values[j], y[j], MODE_MARKER.get(m, "*"), ms=7,
                                color=st["color"], mec=st["color"] if filled else INK,
                                mfc=st["color"] if filled else "none", mew=1.2, zorder=6)
            else:
                ax.plot(ds.ky, line, lw=1.6, ms=5, mfc="none", **st)
            bad = ds.ky.values[np.isin(status, ("empty", "zero"))]
            if bad.size:
                unconv = True
                ax.plot(bad, np.zeros(bad.size), "x", color=st["color"], ms=9, mew=2)
            est = status == "field"
            if est.any():
                fielded = True
                ax.errorbar(ds.ky.values[est], ds[var].sel(corner=c).values[est],
                            yerr=ds[f"{var}_std"].sel(corner=c).values[est],
                            fmt="none" if modes else "D",
                            color=st["color"], ms=6, mec=INK, mew=0.6, capsize=3, lw=1, zorder=5)
    if modes:
        mode_strip(mode_axes[0], ds, weighted=False)
        mode_strip(mode_axes[1], ds, weighted=True)
    for ax, var, sym in zip(axes, ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        if not modes:
            ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    axes[0].set_title("growth rate", loc="left", fontsize=10, color=INK)
    axes[1].set_title("real frequency", loc="left", fontsize=10, color=INK)
    extra = [Line2D([], [], ls="none", marker="x", color=INK, ms=8, mew=2,
                    label="unconverged (empty or 0/0 omega)")] if unconv else []
    if fielded:
        extra.append(Line2D([], [], ls="none", marker="o" if modes else "D", color=MUTED,
                            mec=INK, mfc="none" if modes else MUTED, ms=6,
                            label="TPED field estimate ±1σ" + (" (hollow)" if modes else "")))
    extra += [Line2D([], [], ls="none", marker="o", mfc="none", mec=FIT_STYLE[v]["mec"], mew=1.4,
                     ms=6, alpha=FIT_STYLE[v]["alpha"], label=FIT_STYLE[v]["label"])
              for v in FIT_STYLE if v in fits_seen]
    if modes:
        found = [m for m in MODE_MARKER if (ds["mode"].values == m).any()]
        extra += [Line2D([], [], ls="none", marker=MODE_MARKER[m], color=MUTED, ms=7,
                         label=f"mode: {m}") for m in found]
        handles = [Line2D([], [], lw=1.8, color=style(point_of(ds, c))["color"],
                          ls=style(point_of(ds, c))["ls"], label=str(l))
                   for c, l in zip(ds.corner.values, ds.label.values)]
    else:
        handles = corner_handles(ds)
    legend_right(fig, handles, extra)
    fig.suptitle(f"NSTX 129038 corners, linear GENE at x0 = {ds.x0:g}", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, "corners_growth.png", written)


def plot_growth_field(csv_file, x0, out, written, dodge=0.025):
    """gamma and omega vs ky from field_omega.py --csv: every run's field fit +-1 std,
    filled where GENE converged, hollow where not, GENE's converged value as a tick."""
    if not os.path.isfile(csv_file):
        print(f"no {csv_file} (field_omega.py --csv); corners_growth_field.png skipped")
        return
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"{csv_file} has no rows; corners_growth_field.png skipped")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    pts = ordered([dict(zip(ORDER, k)) for k in df.groupby(list(ORDER)).groups])
    ys = {"gamma": [], "omega": []}
    for n, p in enumerate(pts):
        st = style(p)
        d = df[np.logical_and.reduce([df[k] == p[k] for k in ORDER])].sort_values("ky")
        x = d["ky"].values * (1 + dodge * (n - (len(pts) - 1) / 2))
        conv = (d["gene_state"] == "converged").values
        for ax, var in zip(axes, ("gamma", "omega")):
            y, err = d[f"{var}_field"].values, d[f"{var}_std"].values
            ys[var] += list(y[np.isfinite(y)])
            ax.plot(x, y, lw=0.9, color=st["color"], ls=st["ls"], alpha=0.6)
            for sel, mfc in ((conv, st["color"]), (~conv, "none")):
                if sel.any():
                    ax.errorbar(x[sel], y[sel], yerr=err[sel], fmt=st["marker"], ms=5,
                                color=st["color"], mfc=mfc, mec=st["color"], mew=1.2,
                                capsize=2, lw=0.9, zorder=4)
            gene = d[f"{var}_gene"].values
            if conv.any():
                ax.plot(x[conv], gene[conv], "_", color=INK, ms=9, mew=1.6, zorder=5)
    for ax, var, sym in zip(axes, ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$k_y\rho_{ref}$ (corners nudged apart)")
        ax.set_ylabel(rf"${sym}$ field fit [$c_{{ref}}/L_{{ref}}$]")
        v = np.array(ys[var])
        if v.size:                              # frame the values, not the widest bars
            lo, hi = v.min(), v.max()
            pad = 0.15 * (hi - lo or abs(hi) or 1.0)
            ax.set_ylim(min(lo - pad, -pad / 3), hi + pad)
        tidy(ax)
    ky = np.sort(df["ky"].unique())
    axes[0].set_xticks(ky, [f"{k:g}" for k in ky])
    axes[0].minorticks_off()
    axes[1].minorticks_off()
    axes[0].set_title("growth rate, field fit ±1σ", loc="left", fontsize=10, color=INK)
    axes[1].set_title("real frequency, field fit ±1σ", loc="left", fontsize=10, color=INK)
    handles = [Line2D([], [], lw=1.8, ms=5, mfc="none", **style(p), label=label(p)) for p in pts]
    extra = [Line2D([], [], ls="none", marker="o", color=MUTED, ms=6,
                    label="GENE converged (filled)"),
             Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, ms=6,
                    label="GENE unconverged (hollow)"),
             Line2D([], [], ls="none", marker="_", color=INK, ms=9, mew=1.6,
                    label="GENE's converged value")]
    legend_right(fig, handles, extra)
    n_un = int((df["gene_state"] != "converged").sum())
    fig.suptitle(f"NSTX 129038 corners, field-trace fits of every run at x0 = {x0:g} "
                 f"({len(df)} runs, {n_un} not converged by GENE)", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, "corners_growth_field.png", written)


def mode_strip(ax, ds, weighted):
    """Mode make-up at each ky as stacked bars: the fraction of corners in each
    mode (with a kernel-smoothed fraction over ky as lines), or with *weighted*
    each mode's share of the positive growth summed over corners."""
    ky, m, g = ds.ky.values, ds["mode"].values, ds.gamma.values
    found = [x for x in MODE_COLOR if (m == x).any()]
    width = 0.6 * (np.diff(ky).min() if ky.size > 1 else 0.2)
    bottom = np.zeros(ky.size)
    gpos = np.where(np.isfinite(g) & (m != ""), np.clip(np.nan_to_num(g), 0, None), 0.0)
    total = gpos.sum(axis=0)
    for mode in found:
        if weighted:
            h = np.divide((gpos * (m == mode)).sum(axis=0), total,
                          out=np.zeros(ky.size), where=total > 0)
        else:
            h = (m == mode).sum(axis=0) / m.shape[0]
        ax.bar(ky, h, width, bottom=bottom, color=MODE_COLOR[mode], edgecolor=SURFACE, lw=1,
               label=mode, zorder=3)
        bottom += h
    if not weighted and found:
        x = np.linspace(ky.min() - 2 * MODE_KDE_BW, ky.max() + 2 * MODE_KDE_BW, 400)
        # Kernel-weighted average of each ky's fraction (Nadaraya-Watson): stays in
        # 0-1 however closely the ky are spaced, unlike a summed KDE.
        w = np.exp(-0.5 * ((x[None] - ky[:, None]) / MODE_KDE_BW) ** 2)
        for mode in found:
            frac = (m == mode).sum(axis=0) / m.shape[0]
            ax.plot(x, (w * frac[:, None]).sum(axis=0) / w.sum(axis=0), color=MODE_COLOR[mode],
                    lw=2, zorder=4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of corners")
        ax.set_title(f"mode fraction per ky (bars), kernel-smoothed over ky (lines, bw "
                     f"{MODE_KDE_BW:g})", loc="left", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7.5, ncols=len(found), loc="upper left",
                  bbox_to_anchor=(0, -0.28))
    else:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(r"share of $\Sigma\gamma_{>0}$")
        ax.set_title("each mode's share of the growth at each ky", loc="left", fontsize=9,
                     color=INK)
    ax.set_xlabel(r"$k_y\rho_{ref}$")
    tidy(ax)


def plot_mode_scaling(ds, out, written):
    """Mode at every (corner, ky) as a coloured table, and per knob level the mode
    make-up of all labelled ky: how each profile scaling moves the mode mix."""
    m, ky = ds["mode"].values, ds.ky.values
    status = ds.status.values
    n = ds.sizes["corner"]
    found = [x for x in MODE_COLOR if (m == x).any()]
    fig, (tab, bars) = plt.subplots(1, 2, figsize=(16, max(5.5, 0.34 * n + 2.2)),
                                    width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            mode = m[i, j]
            fc = (MODE_COLOR.get(mode, SURFACE) if mode else
                  {"decaying": "#dcefdc", "ambiguous": "#ecebe6"}.get(status[i, j], SURFACE))
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            text = MODE_SHORT.get(mode, mode) if mode else {
                "zero": "x", "empty": "x", "decaying": "dec", "ambiguous": "amb"}.get(
                status[i, j], "-")
            dark = mode and np.mean(to_rgb(fc)) < 0.55
            tab.text(j + 0.5, n - 0.5 - i, text, ha="center", va="center", fontsize=7.5,
                     color="white" if dark else INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(f"mode per corner and ky, rows {LEGEND_TITLE}  (x unconverged, - none"
                  + (", dec/amb recorded fit" if ds.attrs.get("field_fits") else "") + ")",
                  loc="left", fontsize=10, color=INK)

    groups = [(k, v) for k in ORDER for v in (0.8, 1.2)]
    xs = np.array([i + 0.6 * (i // 2) for i in range(len(groups))])
    bottom = np.zeros(len(groups))
    labelled = [(m[ds[k].values == v] != "").sum() for k, v in groups]
    rows = []
    for mode in found:
        frac = np.array([(m[ds[k].values == v] == mode).sum() / max(t, 1)
                         for (k, v), t in zip(groups, labelled)])
        bars.bar(xs, frac, 0.8, bottom=bottom, color=MODE_COLOR[mode], edgecolor=SURFACE,
                 lw=1, label=mode, zorder=3)
        for x, f, b in zip(xs, frac, bottom):
            if f >= 0.08:
                bars.text(x, b + f / 2, f"{100 * f:.0f}%", ha="center", va="center", fontsize=7,
                          zorder=4,
                          color="white" if np.mean(to_rgb(MODE_COLOR[mode])) < 0.55 else INK)
        bottom += frac
        rows.append((mode, frac))
    bars.set_xticks(xs, [f"{k}\n{v:g}" for k, v in groups], fontsize=8)
    bars.set_ylim(0, 1.02)
    bars.set_ylabel("share of labelled ky")
    bars.set_title("mode mix at each knob level (all other knobs pooled)", loc="left",
                   fontsize=10, color=INK)
    bars.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(bars)
    fig.suptitle(f"NSTX 129038 corners at x0 = {ds.x0:g}: profile scaling vs mode",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "corners_mode_scaling.png", written)

    path = os.path.join(out, "corners_mode_scaling.csv")
    with open(path, "w") as fh:
        fh.write("knob,level,labelled_ky," + ",".join(found) + "\n")
        for g_i, ((k, v), t) in enumerate(zip(groups, labelled)):
            fh.write(f"{k},{v:g},{t}," + ",".join(f"{fr[g_i]:.3f}" for _, fr in rows) + "\n")
    written.append(path)


def fit_ylim(ax, x, ys, xlim, pad=0.05):
    m = (x >= xlim[0]) & (x <= xlim[1])
    vals = np.concatenate([np.asarray(y)[m] for y in ys])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) or abs(hi) or 1.0
        ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_profiles(ds, out, written, xlim=None, name="corners_profiles.png"):
    vars_ = [v for v in ("ne", "Te", "ni", "Ti") if v in ds]
    if not vars_:
        print("no profiles to plot")
        return
    units = {v: u for v, _, u in ITERDB_VARS.values()}
    x = ds.rho_tor.values
    xlim = xlim or (float(x.min()), float(x.max()))
    fig, ax = plt.subplots(2, len(vars_), figsize=(3.6 * len(vars_) + 2.8, 6.4),
                           sharex=True, squeeze=False)
    for j, v in enumerate(vars_):
        top, bot = ax[0, j], ax[1, j]
        for c in ds.corner.values:
            st = style(point_of(ds, c))
            for a, y in ((top, ds[v]), (bot, ds[f"omg_{v}"])):
                a.plot(x, y.sel(corner=c), color=st["color"], ls=st["ls"], lw=1.5)
        fit_ylim(top, x, ds[v].values, xlim)
        # Gradient y range ignores the last points, where -dln/drho blows up at the separatrix.
        fit_ylim(bot, x, ds[f"omg_{v}"].values, (xlim[0], min(xlim[1], 0.98)))
        top.set_title(f"{v} [{units[v]}]", loc="left", fontsize=10, color=INK)
        bot.set_title(rf"$-d\ln {v[0]}_{v[1]}/d\rho_{{tor}}$", loc="left", fontsize=10, color=INK)
        bot.set_xlabel(r"$\rho_{tor}$")
        for a in (top, bot):
            a.axvline(ds.x0, color=INK, ls=":", lw=1.4)
            a.set_xlim(*xlim)
            tidy(a)
    handles = [Line2D([], [], color=style(point_of(ds, c))["color"],
                      ls=style(point_of(ds, c))["ls"], lw=1.8, label=str(l))
               for c, l in zip(ds.corner.values, ds.label.values)]
    legend_right(fig, handles, [Line2D([], [], color=INK, ls=":", lw=1.4,
                                       label=f"x0 = {ds.x0:g}")])
    fig.suptitle("NSTX 129038 corner profiles (profiles.iterdb)"
                 + (f", rho_tor {xlim[0]:g}-{xlim[1]:g}" if name.endswith("_ped.png") else ""),
                 x=0.04, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, name, written)


def corner_modes(ds):
    """(mode at each corner's max-gamma ky, mode sequence over ky e.g. 'IT>KBM'), or blanks."""
    n = ds.sizes["corner"]
    if "mode" not in ds:
        return np.array([""] * n, dtype=object), [""] * n
    at_max, seq = [], []
    for i in range(n):
        modes = ds["mode"].values[i]
        gam = ds["gamma"].values[i]
        has = np.isfinite(gam) & (modes != "")
        at_max.append(modes[np.flatnonzero(has)[np.argmax(gam[has])]] if has.any() else "")
        runs = [m for j, m in enumerate(modes) if m and (j == 0 or m != modes[j - 1])]
        runs = [m for j, m in enumerate(runs) if j == 0 or m != runs[j - 1]]
        seq.append(">".join(MODE_SHORT.get(m, m) for m in runs) or "-")
    return np.array(at_max, dtype=object), seq


def ky_ring(ax_span, s_span, n, radius=0.02, aspect=1.35):
    """Data offsets placing n ky points on a small ring, lowest ky at 12 o'clock, clockwise."""
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return radius * ax_span * np.cos(ang), radius * aspect * s_span * np.sin(ang)


def plot_s_alpha(ds, out, written):
    """shat vs alpha_MHD, one ring of points per corner, one point per ky coloured by its gamma."""
    if "shat" not in ds or "alpha_MHD" not in ds:
        print("no shat/alpha_MHD (no GENE parameters read); s-alpha plot skipped")
        return
    a, s = ds.alpha_MHD.values, ds.shat.values
    ok = np.isfinite(a) & np.isfinite(s)
    if not ok.any():
        print("shat/alpha_MHD all NaN; s-alpha plot skipped")
        return
    status, ky = ds.status.values, ds.ky.values
    # converged ky only: GENE's own value or a passing field estimate; unconverged,
    # missing and --field-fits decaying/ambiguous fits are left off the rings
    gam = np.where(np.isin(status, list(FIT_STYLE)), np.nan, ds.gamma.values)
    modes = ds["mode"].values if "mode" in ds else np.full(gam.shape, "", dtype=object)
    gfin = gam[ok][np.isfinite(gam[ok])]
    span = max(np.abs(gfin).max(), 1e-6) if gfin.size else 1.0
    norm = Normalize(-span, span)                # symmetric: the grey middle is gamma = 0
    wide = "mode" in ds
    fig, (ax, key) = plt.subplots(1, 2, figsize=(15 if wide else 13.5, 7.5),
                                  width_ratios=(3, 1.8 if wide else 1.3))
    dx, dy = ky_ring(np.ptp(a[ok]) or 1.0, np.ptp(s[ok]) or 1.0, ky.size)
    sc, lines = None, []
    for n, i in enumerate(np.flatnonzero(ok), start=1):
        x, y = a[i] + dx, s[i] + dy
        ax.plot(np.r_[x, x[:1]], np.r_[y, y[:1]], color=GRID, lw=0.8, zorder=2)
        for j in range(ky.size):
            if not np.isfinite(gam[i, j]):       # unconverged or missing: gap in the ring
                continue
            field = status[i, j] == "field"
            sc = ax.scatter(x[j], y[j], c=[gam[i, j]], cmap=GAMMA_CMAP, norm=norm, s=42,
                            marker=MODE_MARKER.get(modes[i, j], "o") if modes[i, j] else "o",
                            edgecolors=INK, linewidths=1.5 if field else 0.5, zorder=4)
        ax.annotate(str(n), (a[i], s[i]), ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.8))
        g = gam[i][np.isfinite(gam[i])]
        rng = f"{g.min():.2f}-{g.max():.2f}" if g.size else "-"
        lines.append(f"{n:>2}  {ds.label.values[i]:<21} {rng:>9}"
                     + (f"  {corner_modes(ds)[1][i]}" if wide else ""))
    handles = [Line2D([], [], ls="none", marker="o", color=MUTED, mec=INK, mew=1.8, ms=7,
                      label="TPED field estimate (thick edge)")]
    if wide:
        handles += [Line2D([], [], ls="none", marker=MODE_MARKER[m], color=MUTED, mec=INK, ms=7,
                           label=m) for m in MODE_MARKER if (modes == m).any()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    key.axis("off")
    head = f"{'#':>2}  {LEGEND_TITLE:<21} {'gamma':>9}" + ("  modes over ky" if wide else "")
    key.text(0, 1, head + "\n" + "\n".join(lines), family="monospace", fontsize=8, va="top",
             color=INK, transform=key.transAxes)
    foot = (f"each corner: a ring of its ky, {ky[0]:g} at 12 o'clock then clockwise\n"
            "(" + ", ".join(f"{k:g}" for k in ky) + "); number at the corner's own\n"
            "(alpha, shat), the ring offset is for display only;\n"
            "unconverged ky are left out (gaps in the ring)")
    if wide:
        foot += "\n\nmodes over ky, low to high:\n" + "\n".join(
            f"  {v} = {k}" for k, v in MODE_SHORT.items())
    key.text(0, 0, foot, family="monospace", fontsize=7, va="bottom", color=MUTED,
             transform=key.transAxes)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\gamma$ at each $k_y$ [$c_{ref}/L_{ref}$]")
    ax.set_xlabel(r"$\alpha_{MHD} = q_0^2\,(R_0/L_{ref})\,(-d\beta/dx)$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.margins(x=0.1, y=0.1)
    ax.set_title(f"s-alpha at x0 = {ds.x0:g}, every ky", loc="left", fontsize=10, color=INK)
    tidy(ax)
    fig.tight_layout()
    save(fig, out, "corners_s_alpha.png", written)


def local_params(ds) -> list[str]:
    return [k for k in ds.data_vars
            if ds[k].dims == ("corner",) and k not in ("gamma_max", "ky_at_max")
            and np.issubdtype(ds[k].dtype, np.number)]


def varying(ds, spread=SPREAD, max_params=MAX_PARAMS) -> list[tuple[str, float]]:
    """(name, (max-min)/max|value|) of the local parameters above *spread*, largest first."""
    out = []
    for k in local_params(ds):
        v = ds[k].values[np.isfinite(ds[k].values)]
        if v.size < 2:
            continue
        rel = np.ptp(v) / max(abs(v).max(), 1e-300)
        if rel > spread:
            out.append((k, float(rel)))
    return sorted(out, key=lambda t: -t[1])[:max_params]


def plot_local(ds, out, written, spread=SPREAD, max_params=MAX_PARAMS):
    keys = local_params(ds)
    if not keys:
        print("no local parameters (no GENE parameters read); bar chart and table skipped")
        return
    csv = os.path.join(out, "corners_local_params.csv")
    cols = ["gamma_max", "ky_at_max"] + keys
    with open(csv, "w") as fh:
        fh.write(",".join(["corner", *ORDER] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            fh.write(",".join([c, *(f"{ds[k].values[i]:g}" for k in ORDER)]
                              + [f"{ds[k].values[i]:.6g}" for k in cols]) + "\n")
    written.append(csv)

    picked = varying(ds, spread, max_params)
    if not picked:
        print(f"no local parameter varies by more than {spread:g} across corners")
        return
    print("varying local parameters, (max-min)/max|value|: "
          + ", ".join(f"{k} {r:.3f}" for k, r in picked))
    names = [k for k, _ in picked]
    # % from the mean means nothing when the mean sits near zero, so parameters
    # that change sign across the corners get their own panel in absolute units.
    crosses = [k for k in names if np.nanmin(ds[k].values) < 0 < np.nanmax(ds[k].values)]
    groups = [(r"% from the corner mean", [k for k in names if k not in crosses], True),
              ("value - corner mean", crosses, False)]
    groups = [grp for grp in groups if grp[1]]
    spread_of = dict(picked)
    n = ds.sizes["corner"]
    width = 0.84 / n
    fig, axes = plt.subplots(1, len(groups), figsize=(max(10, 1.25 * len(names) + 4), 5.8),
                             width_ratios=[len(grp[1]) + 0.6 for grp in groups], squeeze=False)
    for ax, (ylabel, keys_, pct) in zip(axes[0], groups):
        mean = [np.nanmean(ds[k].values) for k in keys_]
        for i, c in enumerate(ds.corner.values):
            dev = [100 * (ds[k].values[i] / m - 1) if pct else ds[k].values[i] - m
                   for k, m in zip(keys_, mean)]
            ax.bar(np.arange(len(keys_)) - 0.42 + (i + 0.5) * width, dev, width * 0.9,
                   color=style(point_of(ds, c))["color"],
                   label=str(ds.label.values[i]) if ax is axes[0, 0] else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(np.arange(len(keys_)),
                      [f"{k}\n{100 * spread_of[k]:.0f}%" for k in keys_], fontsize=8)
        ax.set_ylabel(ylabel)
        tidy(ax)
    axes[0, 0].set_title(f"local parameters at x0 = {ds.x0:g} with (max-min)/max|value| > "
                         f"{spread:g} (spread under each name)", loc="left", fontsize=10,
                         color=INK)
    if crosses:
        axes[0, -1].set_title("sign-changing, absolute", loc="left", fontsize=10, color=INK)
    fig.legend(title=LEGEND_TITLE, loc="center right", frameon=False, fontsize=8,
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    save(fig, out, "corners_local_params.png", written)

    tcols = names + ["gamma_max", "ky_at_max"]
    cells = [[f"{v:.4g}" if np.isfinite(v) else "-" for v in (ds[k].values[i] for k in tcols)]
             for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(tcols) + 2), 0.26 * n + 1.0))
    ax.axis("off")
    tab = ax.table(cellText=cells, rowLabels=[str(l) for l in ds.label.values],
                   colLabels=tcols, loc="center", cellLoc="right")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1, 1.25)
    ax.set_title(f"local parameters per corner, rows {LEGEND_TITLE}", loc="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "corners_local_table.png", written)


def run(diag_root=DIAG_ROOT, ped_root=PED_ROOT, out=None, spread=SPREAD, max_params=MAX_PARAMS,
        x0=X0, modes=False, field_fits=False, field_all=None):
    """field_all: field_omega.py --csv file to draw as corners_growth_field.png
    ('' for its default path), or None for no such plot."""
    x0 = tuple(x0)
    out = out or default_out(x0)
    ds = load_dataset(diag_root, ped_root, x0, modes, field_fits)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN corners and rows
        plot_growth(ds, out, written)
        if field_all is not None:
            plot_growth_field(field_all or csv_path(diag_root, x0), x0[0], out, written)
        plot_profiles(ds, out, written)
        plot_profiles(ds, out, written, PED_ZOOM, "corners_profiles_ped.png")
        plot_s_alpha(ds, out, written)
        plot_local(ds, out, written, spread, max_params)
        if "mode" in ds:
            write_modes_csv(ds, out, written)
            plot_mode_scaling(ds, out, written)
    print("\nwrote\n  " + "\n  ".join(written))
    return ds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--diag-root", default=DIAG_ROOT,
                    help="GENE output root; corners are under <diag-root>/x0_<x0>/<tag>/")
    ap.add_argument("--ped-root", default=PED_ROOT, help="ped_grid cheaseBS cases <tag>/")
    ap.add_argument("--x0", type=float, nargs="+", default=list(X0),
                    help="radius of the scan to plot (reads its x0_<x0>/ folder)")
    ap.add_argument("--out", default=None,
                    help="directory for the PNGs and CSV [<diag-root>/x0_<x0>/plots]")
    ap.add_argument("--field-fits", action="store_true",
                    help="draw recorded field fits that did not pass (decaying/ambiguous, "
                         "from field_omega.py --write) with their error bars")
    ap.add_argument("--field-all", nargs="?", const="", default=None, metavar="CSV",
                    help="also plot every run's field fit +-1 std from field_omega.py --csv "
                         "[<diag-root>/x0_<x0>/field_omega_fits.csv]")
    ap.add_argument("--modes", action="store_true",
                    help="classify each ky with TPED mode_identifier (reads nrg, field, "
                         "geometry; slower) and mark the growth plot by mode")
    ap.add_argument("--spread", type=float, default=SPREAD,
                    help="(max-min)/|mean| above which a local parameter counts as varying")
    ap.add_argument("--max-params", type=int, default=MAX_PARAMS,
                    help="most varying parameters to chart and tabulate")
    args = ap.parse_args()
    run(args.diag_root, args.ped_root, args.out, args.spread, args.max_params, args.x0,
        args.modes, args.field_fits, args.field_all)
