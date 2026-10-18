"""Plots for the 16-corner linear GENE scan (corner_gene_scan.py): growth, profiles, s-alpha, locals.

Reads every corner corner_gene_scan.py launched,

    GENE output   <diag-root>/x0_<x0>/<tag>/scanfilesNNNN/{parameters,omega}_000K
                  (every scanfiles dir, merged by ky)
    profiles      <ped-root>/<tag>/profiles.iterdb   (what GENE read, prof_type -2)
    q             <ped-root>/<tag>/*_CH_COCOS*.geqdsk (the cheaseBS equilibrium GENE read)

into one xarray Dataset with dims (corner, ky, rho_tor), and writes to <out>/

    corners_growth.png          gamma and omega vs ky, all corners, GENE-converged ky only
    corners_growth_field.png    the same with the field-fit guess (+-1 std) at every
                                unconverged ky, on its corner's own line
    corners_profiles.png        ne, Te, ni, Ti and their -dln/drho_tor, x0 marked; under
                                them |q| vs rho_tor, a faint line at each integer q and a
                                drop line where each corner crosses it (rational surfaces,
                                to line MTM drive up against q = m/n)
    corners_profiles_ped.png    the same, zoomed to PED_ZOOM
    corners_mode_scaling.png    (--modes) mode per corner and ky as a table, each cell
                                the mode's colour shaded by gamma (dark = fastest corner
                                at that ky, light = slowest) with gamma printed, and the
                                mode mix at each knob level; + corners_mode_scaling.csv
    corners_s_alpha.png         shat vs alpha_MHD at x0: each corner a small ring of
                                its converged ky (unconverged ones left out), each
                                coloured by that ky's gamma (marker = its mode under
                                --modes), numbered, with a key beside
    corners_local_params.png    the local parameters that vary most: % from the corner
                                mean, or the absolute difference for those that change sign
    corners_local_table.png     their values per corner, as a table
    corners_local_params.csv    every local parameter per corner (+ gamma_max)
    corners_q_offset.png        rational-surface offset per corner and ky: delta =
                                n q0 - round(n q0) (n = n0_global, q0 at x0, both from
                                the run's parameters), as a table, and gamma vs |delta|;
                                + corners_q_offset.csv
    corners_res2.png            (when 2x-resolution runs exist) the 1x corner beside
                                its 2x rerun: gamma and omega vs ky, and 2x gamma vs
                                1x gamma; + corners_res2.csv
    corners_nz<N>.png           (--nz-compare N) the same comparison against the
                                nz0 = N runs under <diag-root>_nz<N> (or --nz-root),
                                each matched to the corner of the same profile
                                scaling (tag); the line is this scan, at its own nz0
                                read from its parameters; + corners_nz<N>.csv

Every CSV is written to <out>/data/, beside the plots.

2x-resolution runs are found under <diag-root> (and any sibling <diag-root>_res2,
the root corner_gene_scan.py --res-scale 2 writes) as any directory holding
scanfiles* or parameters_NNNN whose path names a corner tag and carries 2XRES
(any case, also 2x_res, res2x) or _res2, at --x0 (from an x0_<x0> folder in the
path, else the run's parameters). Each is matched to the 1x corner of the same
tag. A 2x ky GENE has not converged (still running, or 0/0 at timelim) gets a
field fit computed here from its field_NNNN, as field_omega.py does it, so the
best guess is there while the runs are going; its verdict sets the marker.

Every legend, table and CSV lists a corner as its knob factors in the order
of its run name, (Teh, Tew, neh, new): (1.2, 0.8, 0.8, 1.2) is
Teh1.20_Tew0.80_neh0.80_new1.20. Corners are sorted in that order too. Colour hue is the width pair (Tew, new), shade
the height pair (Teh, neh); line style is the height pair too (solid,
dashed, dotted, dash-dot) and the marker neh, so the four corners sharing a
hue never share a line style and no corner is told apart by colour alone.

    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --out ./corner_plots
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --spread 0.02 --max-params 16
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737 --modes
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.8447 --modes --field-fits

Two growth plots, always:

    corners_growth.png        only what GENE converged (filled markers on each
                              corner's line); every other ky is an x on the zero line
    corners_growth_field.png  the same corners and lines, with each unconverged ky
                              (0/0 or empty omega file) filled in by the field fit
                              field_omega.py --write recorded there (TPED's gamma_field
                              +- gamma_field_std, whatever its verdict): a hollow
                              marker with +-1 std bars in the corner's own colour, line
                              style and marker, joined into its line, so every guess
                              reads as part of its family. 'ambiguous' fits are faded.
                              Skipped when no omega file holds a fit.

With --modes each ky GENE converged (or the field estimate filled) is
classified by TPED's mode_identifier (GENE_sim_analysis: fingerprint transport
ratios from nrg, Epar and parities from field + geometry, omega/omega_star),
drawn with that mode's marker in corners_growth.png, and listed with the
inputs behind each label in corners_modes.csv. The fingerprint diffusivity is
Gamma / omn: where omn ~ 0 (the ne top) the D ratios, and so the ETG, KBM and
ITG/TEM checks, are unreliable; read the inputs before the label.

With --field-fits, a ky that GENE left 0/0 but field_omega.py recorded a
fit for with verdict 'decaying' (gamma + 2 std < 0) or 'ambiguous' (std too
large to tell a slow-growing mode from beating damped ones) carries that fit
in the dataset: 'dec'/'amb' cells in the mode table, gamma_std/omega_std in
corners_modes.csv, and left out of gamma_max when ambiguous. Without the flag
they stay x. A passing fit (status 'field') is the point's gamma either way.

An empty omega file, or one holding gamma = omega = 0 exactly (GENE's
placeholder when a run ends without a converged growth rate: time limit, or
a decaying mode), reads as NaN; a missing omega or parameters file is NaN and
left out. The ky axis is every kymin found across the corners, so scans with
different ky lists combine. A corner with neither GENE output nor profiles is
reported and skipped. Nothing here raises on a partial scan.

delta = n q0 - round(n q0), in [-1/2, 1/2], places x0 between the q = m/n
surfaces of the run's toroidal mode n: 0 on a rational surface, +-1/2 midway.
|delta| is the distance to the nearest one as a fraction of their spacing
x0 / (n shat q0) (in x = rho_tor), which dx_rational in the CSV gives in rho_tor.
A flux tube does not see this offset (ballooning space assumes dense rational
surfaces); it is the local proxy for the q-overlap a global MTM depends on.
n is GENE's n0_global, or round(ky x0 / (q0 rhostar)) when that is absent.

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

from TPED.projects.GENE_sim_reader.src.GENE_sim_data import GeneSimulation
from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder
from TPED.projects.utils.read_iterdb import read_iterdb
from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData
from NSTXU_lithium_study.scripts.corner_gene_scan import (DIAG_ROOT, KYMIN, PED_ROOT, SHOT,
                                                         SHOTS, X0, corners, find_geqdsk,
                                                         shot_defaults, tag, x0_label)
from NSTXU_lithium_study.scripts.ped_grid import KNOBS
from NSTXU_lithium_study.scripts.corners.field_omega import field_omega


def default_out(x0=X0, diag_root=DIAG_ROOT) -> str:
    return os.path.join(diag_root, x0_label(x0), "plots")


#: Legend order of the knobs: the corner tag's (ped_grid.tag, the GENE run and diag dir
#: names), Teh, Tew, neh, new, so a label reads straight across to its run directory.
ORDER = tuple(KNOBS)
LEGEND_TITLE = "(" + ", ".join(ORDER) + ")"

#: Pedestal window of the zoomed profile figure (rho_tor).
PED_ZOOM = (0.6, 1.0)
#: q panels show x0 +- this in rho_tor (5% of the minor radius); the window is
#: shaded on the profile panels above.
Q_ZOOM = 0.05

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
#: The TPED field fit of a run (GeneOmega), any GENE state; corners_growth_field.png.
FIT_KEYS = ("gamma_field", "gamma_field_std", "omega_field", "omega_field_std")
FIT_NONE = (np.nan,) * len(FIT_KEYS) + ("",)
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
#: Mode table shading: gamma / (largest gamma at that ky) of 0 -> this toward white,
#: 1 -> this toward black (shade()); per ky, since gamma rises steeply with ky.
TABLE_SHADE = (0.82, -0.3)
#: Gaussian width (in ky) of the smoothed mode fraction under the growth plot.
MODE_KDE_BW = 0.12
#: corners_q_offset.png: |delta| / 0.5, dark on a rational surface to light midway
Q_OFFSET_CMAP = LinearSegmentedColormap.from_list("q_offset", ["#5b2a86", "#b58fd0", "#f4f1f8"])
#: 2x-resolution run dirs: 2XRES / 2x_res / res2x anywhere in the path, or the _res2
#: suffix of corner_gene_scan.py --res-scale 2 and the dashboard's 2x button
RES2_RE = re.compile(r"(?i)2x_?res|res_?2x|_res2(?!\d)")
TAG_RE = re.compile(r"Teh\d+\.\d+_Tew\d+\.\d+_neh\d+\.\d+_new\d+\.\d+")
X0_RE = re.compile(r"x0_(\d+(?:\.\d+)?)(?![\d.-])")
#: how deep below each root to look for 2x run dirs
RES2_DEPTH = 5
#: corners_res2.png: 2x points sit this fraction of the ky span right of their ky
RES2_NUDGE = 0.012
#: --nz-compare N: the nz0 = N corners are at <diag-root>_nz<N>/x0_<x0>/<tag>/ (e.g.
#: corners_nz64 beside the nz0 = 128 corners)
NZ_SUFFIX = "_nz{}"
#: subfolder of --out that every CSV is written to
DATA_DIR = "data"
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
    # a 2x rerun nested in the corner dir (<tag>/2XRES/...) is not this corner's
    found = [f for f in found if re.search(r"parameters_\d{4}$", f)
             and not RES2_RE.search(os.path.relpath(f, corner_dir))]
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
           "gamma_std": np.nan, "omega_std": np.nan, "fit": FIT_NONE}
    if "omega" in tree.children:
        om = tree["omega"].ds
        if "gamma_field" in om:                     # field_omega.py --write, any run
            rec["fit"] = tuple(float(om[k]) if k in om else np.nan for k in FIT_KEYS) + (
                om.attrs.get("field_verdict", ""),)
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
    if "n0_global" in par:
        rec["n0"] = float(par["n0_global"])
    else:                                     # GENE's n for nky0 = 1: ky Cy / rhostar, Cy = x0/q0
        try:
            rec["n0"] = float(np.round(rec["kymin"] * float(par["x0"])
                                       / (float(par["q0"]) * float(par["rhostar"]))))
        except (KeyError, ZeroDivisionError):
            rec["n0"] = np.nan
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
    """({kymin: (gamma, omega, status, gamma_std, omega_std, parameters path,
    *FIT_KEYS, field verdict, n0)}, local parameters) of one corner, or None."""
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
                        rec["omega_std"], path, *rec["fit"], rec["n0"])
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
    path = data_path(out, "corners_modes.csv")
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


def q_profile(case_dir: str):
    """(rho_tor, |q|) of the corner's cheaseBS geqdsk (TPED GFileData), or None."""
    geqdsk, problem = find_geqdsk(case_dir)
    if problem:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = GFileData(os.path.join(case_dir, geqdsk)).gfile_to_xarray()
    except Exception as exc:                      # a bad gfile never sinks the corner
        print(f"    unreadable {geqdsk}: {type(exc).__name__}: {exc}")
        return None
    rho, q = np.asarray(g["rho_tor"].values), np.abs(np.asarray(g["q"].values))
    order = np.argsort(rho)
    return rho[order], q[order]


def integer_crossings(x, y, n):
    """rho where y crosses n (linear between grid points)."""
    d = y - n
    i = np.flatnonzero(np.isfinite(d[:-1]) & np.isfinite(d[1:]) & (np.sign(d[:-1]) != np.sign(d[1:])))
    return x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])


# ----------------------------------------------------------------------------- dataset

def load_dataset(diag_root: str = DIAG_ROOT, ped_root: str = PED_ROOT, x0=X0,
                 modes=False, field_fits=False, shot=SHOT) -> xr.Dataset:
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
        q = q_profile(os.path.join(ped_root, name))
        if q is not None:
            pr = {**(pr or {}), "q": q}
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
    for j, k in enumerate(FIT_KEYS, start=6):      # recorded field fit, whatever GENE did
        ds[k] = (("corner", "ky"), [on_ky(n, j, np.nan) for n in names])
    ds["field_verdict"] = (("corner", "ky"), [on_ky(n, 6 + len(FIT_KEYS), "") for n in names])
    ds["n0"] = (("corner", "ky"), [on_ky(n, 7 + len(FIT_KEYS), np.nan) for n in names],
                {"long_name": "toroidal mode number n (GENE n0_global)"})
    keys = sorted({k for g in gene.values() for k in g[1]})
    for k in keys:
        ds[k] = ("corner", [gene[n][1].get(k, np.nan) if n in gene else np.nan for n in names])
    if "alpha_MHD" in ds:
        ds["alpha_MHD"].attrs["formula"] = "q0^2 * major_R * dpdx_pm (GENE units)"
    if "q0" in ds:                                 # q = m/n surfaces around x0
        nq = ds["n0"] * ds["q0"]
        ds["q_offset"] = (nq - np.round(nq)).assign_attrs(
            long_name="n q0 - round(n q0): 0 on a rational surface, +-1/2 midway")
        if "shat" in ds:
            ds["dx_rational"] = (abs(ds["q_offset"]) * float(x0[0])
                                 / (ds["n0"] * ds["shat"] * ds["q0"])).assign_attrs(
                long_name="rho_tor from x0 to the nearest q = m/n surface")
    trusted = ds["gamma"].where(ds["status"] != "ambiguous")   # an ambiguous fit is no maximum
    finite = trusted.notnull().any("ky")
    ds["gamma_max"] = trusted.fillna(-np.inf).max("ky").where(finite)
    ds["ky_at_max"] = xr.DataArray(ky[trusted.fillna(-np.inf).argmax("ky").values],
                                   dims="corner").where(finite)

    if prof:                                       # profiles on the first corner's grid
        ref = next(iter(prof.values()))
        grid = ref.get("ne", next(iter(ref.values())))[0]
        ds = ds.assign_coords(rho_tor=grid)
        for var in [v for v, *_ in ITERDB_VARS.values()] + ["q"]:
            rows = []
            for n in names:
                r, y = prof.get(n, {}).get(var, (None, None))
                rows.append(np.interp(grid, r, y) if r is not None else np.full(grid.size, np.nan))
            arr = np.array(rows)
            if np.isnan(arr).all():
                continue
            ds[var] = (("corner", "rho_tor"), arr)
            if var == "q":
                ds[var].attrs["long_name"] = "|q| from the cheaseBS geqdsk"
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ds[f"omg_{var}"] = (("corner", "rho_tor"),
                                    -np.gradient(arr, grid, axis=1) / arr,
                                    {"long_name": f"-dln{var}/drho_tor"})
    ds.attrs.update(x0=float(x0[0]), diag_root=diag_root, ped_root=ped_root,
                    field_fits=int(field_fits), shot=int(shot))
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


def data_path(out, name):
    """<out>/data/<name>: every CSV goes in data/ beside the plots."""
    os.makedirs(os.path.join(out, DATA_DIR), exist_ok=True)
    return os.path.join(out, DATA_DIR, name)


def plot_growth(ds, out, written, guesses=False):
    """gamma and omega vs ky, one line per corner through what GENE converged
    (corners_growth.png) or, with *guesses*, also through the recorded field fit
    at each unconverged ky, drawn hollow with +-1 std (corners_growth_field.png)."""
    name = "corners_growth_field.png" if guesses else "corners_growth.png"
    ky = ds.ky.values
    status = ds.status.values
    conv = status == "ok"
    guess = ~conv & np.isfinite(ds["gamma_field"].values) if guesses else np.zeros_like(conv)
    if guesses and not guess.any():
        print(f"no field fit at any unconverged ky (field_omega.py --write); {name} skipped")
        return
    modes = "mode" in ds
    mode = ds["mode"].values if modes else np.full(conv.shape, "", dtype=object)
    verdict = ds["field_verdict"].values
    if modes:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True,
                                 height_ratios=(3, 1.5))
        axes, mode_axes = grid[0], grid[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    unconv = faded = False
    for i, c in enumerate(ds.corner.values):
        st = style(point_of(ds, c))

        def marker(j):
            return MODE_MARKER.get(mode[i, j], "*") if modes and mode[i, j] else st["marker"]

        for ax, var in zip(axes, ("gamma", "omega")):
            y = np.where(conv[i], ds[var].values[i], np.nan)
            y = np.where(guess[i], ds[f"{var}_field"].values[i], y)
            err = ds[f"{var}_field_std"].values[i]
            ok = np.isfinite(y)
            ax.plot(ky[ok], y[ok], lw=1.4, color=st["color"], ls=st["ls"], zorder=3)
            for j in np.flatnonzero(conv[i] & ok):
                ax.plot(ky[j], y[j], marker(j), ms=7 if modes else 5.5, color=st["color"],
                        mec=st["color"], zorder=6)
            for j in np.flatnonzero(guess[i]):
                amb = verdict[i, j] == "ambiguous"
                faded |= bool(amb)
                ax.errorbar(ky[j], y[j], yerr=err[j], fmt=marker(j), ms=7 if modes else 6,
                            mfc=SURFACE, mec=st["color"], mew=1.4, color=st["color"],
                            capsize=2.5, lw=0.9, alpha=0.5 if amb else 1.0, zorder=5)
            bad = ~conv[i] & ~guess[i] & (status[i] != "missing")
            if bad.any():
                unconv = True
                ax.plot(ky[bad], np.zeros(bad.sum()), "x", color=st["color"], ms=9, mew=2)
    if modes:
        mode_strip(mode_axes[0], ds, weighted=False)
        mode_strip(mode_axes[1], ds, weighted=True)
    for ax, var, sym in zip(axes, ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        if not modes:
            ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    what = "GENE converged + field-fit guess" if guesses else "GENE converged only"
    axes[0].set_title(f"growth rate, {what}", loc="left", fontsize=10, color=INK)
    axes[1].set_title(f"real frequency, {what}", loc="left", fontsize=10, color=INK)
    extra = [Line2D([], [], ls="none", marker="o", color=MUTED, ms=6,
                    label="GENE converged (filled)")]
    if guesses:
        extra.append(Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.4,
                            ms=6, label="field-fit guess ±1σ (hollow)"))
        if faded:
            extra.append(Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.4,
                                ms=6, alpha=0.5, label="guess with ambiguous fit (faded)"))
    if unconv:
        extra.append(Line2D([], [], ls="none", marker="x", color=INK, ms=8, mew=2,
                            label="unconverged" + (", no fit" if guesses else "")))
    if modes:
        found = [m for m in MODE_MARKER if (mode == m).any()]
        extra += [Line2D([], [], ls="none", marker=MODE_MARKER[m], color=MUTED, ms=7,
                         label=f"mode: {m}") for m in found]
        handles = [Line2D([], [], lw=1.8, color=style(point_of(ds, c))["color"],
                          ls=style(point_of(ds, c))["ls"], label=str(l))
                   for c, l in zip(ds.corner.values, ds.label.values)]
    else:
        handles = [Line2D([], [], lw=1.8, ms=5.5, **style(point_of(ds, c)), label=str(l))
                   for c, l in zip(ds.corner.values, ds.label.values)]
    legend_right(fig, handles, extra)
    fig.suptitle(f"NSTX {ds.shot} corners, linear GENE at x0 = {ds.x0:g}", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, name, written)


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


def table_colour(mode, frac):
    """The mode's colour, light for frac 0 to dark for frac 1 (TABLE_SHADE)."""
    lo, hi = TABLE_SHADE
    f = float(np.clip(frac, 0, 1)) if np.isfinite(frac) else 0.0
    return shade(MODE_COLOR.get(mode, MODE_COLOR["unidentified"]), lo + (hi - lo) * f)


def plot_mode_scaling(ds, out, written):
    """Mode at every (corner, ky) as a table shaded by gamma, and per knob level the
    mode make-up of all labelled ky: how each profile scaling moves the mode mix.

    A cell is its mode's colour, from light (gamma near 0) to dark (the largest
    gamma of any corner at that ky), with gamma printed under the mode, so a
    column reads as which corners, and so which knobs, drive that ky hardest."""
    m, ky = ds["mode"].values, ds.ky.values
    status = ds.status.values
    gam = ds.gamma.values
    trusted = np.where((m != "") & np.isfinite(gam), gam, np.nan)
    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(np.where(trusted > 0, trusted, np.nan), axis=0)
    n = ds.sizes["corner"]
    found = [x for x in MODE_COLOR if (m == x).any()]
    fig, (tab, bars) = plt.subplots(1, 2, figsize=(16, max(5.5, 0.34 * n + 2.2)),
                                    width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            mode = m[i, j]
            fc = (table_colour(mode, trusted[i, j] / col_max[j]) if mode else
                  {"decaying": "#dcefdc", "ambiguous": "#ecebe6"}.get(status[i, j], SURFACE))
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            text = MODE_SHORT.get(mode, mode) if mode else {
                "zero": "x", "empty": "x", "decaying": "dec", "ambiguous": "amb"}.get(
                status[i, j], "-")
            if mode and np.isfinite(gam[i, j]):
                text += f"\n{gam[i, j]:.3f}"
            dark = mode and np.mean(to_rgb(fc)) < 0.55
            tab.text(j + 0.5, n - 0.5 - i, text, ha="center", va="center", fontsize=6.5,
                     linespacing=1.1, color="white" if dark else INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(f"mode and gamma per corner and ky, rows {LEGEND_TITLE}  (x unconverged, - none"
                  + (", dec/amb recorded fit" if ds.attrs.get("field_fits") else "") + ")\n"
                  "shade: gamma / largest gamma at that ky, light 0 to dark 1",
                  loc="left", fontsize=10, color=INK)
    for k, mode in enumerate(found):              # light-to-dark key per mode, under the table
        for q in range(5):
            tab.add_patch(plt.Rectangle((k * 1.9 + q * 0.3, -1.95), 0.3, 0.45,
                                        fc=table_colour(mode, q / 4), ec="none", clip_on=False))
        tab.text(k * 1.9 + 1.6, -1.72, MODE_SHORT.get(mode, mode), fontsize=7, va="center", color=INK,
                 clip_on=False)

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
    fig.suptitle(f"NSTX {ds.shot} corners at x0 = {ds.x0:g}: profile scaling vs mode",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "corners_mode_scaling.png", written)

    path = data_path(out, "corners_mode_scaling.csv")
    with open(path, "w") as fh:
        fh.write("knob,level,labelled_ky," + ",".join(found) + "\n")
        for g_i, ((k, v), t) in enumerate(zip(groups, labelled)):
            fh.write(f"{k},{v:g},{t}," + ",".join(f"{fr[g_i]:.3f}" for _, fr in rows) + "\n")
    written.append(path)


def plot_q_offset(ds, out, written):
    """delta = n q0 - round(n q0) per corner and ky as a table (dark on a rational
    surface, light midway), and gamma vs |delta| beside it; + corners_q_offset.csv."""
    if "q_offset" not in ds or not np.isfinite(ds.q_offset.values).any():
        print("no n0_global/q0 in the runs' parameters; corners_q_offset.png skipped")
        return
    d, n0, gam, ky = ds.q_offset.values, ds.n0.values, ds.gamma.values, ds.ky.values
    m = ds["mode"].values if "mode" in ds else np.full(d.shape, "", dtype=object)
    conv = ds.status.values == "ok"
    n = ds.sizes["corner"]
    fig, (tab, sc) = plt.subplots(1, 2, figsize=(16, max(5.5, 0.34 * n + 2.2)),
                                  width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            if not np.isfinite(d[i, j]):
                tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=SURFACE, ec=SURFACE, lw=2))
                tab.text(j + 0.5, n - 0.5 - i, "-", ha="center", va="center", fontsize=6.5,
                         color=MUTED)
                continue
            fc = Q_OFFSET_CMAP(abs(d[i, j]) / 0.5)
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            if m[i, j]:                            # mode band along the top, MODE_COLOR
                tab.add_patch(plt.Rectangle((j + 0.04, n - 1 - i + 0.8), 0.92, 0.16, ec="none",
                                            fc=MODE_COLOR.get(m[i, j], MODE_COLOR["unidentified"])))
            text = f"{d[i, j]:+.2f}\nn={n0[i, j]:g}"
            if np.isfinite(gam[i, j]):
                mode = f"{MODE_SHORT.get(m[i, j], m[i, j])} " if m[i, j] else ""
                text += f"\n{mode}{gam[i, j]:.3f}"
            tab.text(j + 0.5, n - 0.55 - i, text, ha="center", va="center", fontsize=6,
                     linespacing=1.05, color="white" if np.mean(fc[:3]) < 0.55 else INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(r"$\delta = nq_0 - \mathrm{round}(nq_0)$ per corner and ky, rows "
                  f"{LEGEND_TITLE}\ncell: delta, n, [mode] gamma, top band the mode's colour; "
                  "dark = on a q = m/n surface, light = midway", loc="left", fontsize=10, color=INK)
    for q in range(6):                             # dark-to-light key under the table
        tab.add_patch(plt.Rectangle((q * 0.35, -1.95), 0.35, 0.45, fc=Q_OFFSET_CMAP(q / 5),
                                    ec="none", clip_on=False))
    tab.text(2.2, -1.72, "|delta| 0 to 0.5", fontsize=7, va="center", color=INK, clip_on=False)
    for k, mode in enumerate(x for x in MODE_COLOR if (m == x).any()):   # mode band key
        tab.add_patch(plt.Rectangle((3.6 + k * 1.3, -1.85), 0.3, 0.25, fc=MODE_COLOR[mode],
                                    ec="none", clip_on=False))
        tab.text(3.95 + k * 1.3, -1.72, MODE_SHORT.get(mode, mode), fontsize=7, va="center",
                 color=INK, clip_on=False)

    # each point its mode's marker and colour (as in the growth and mode plots), ky beside it
    for i, j in zip(*np.nonzero(conv & np.isfinite(d))):
        mode = m[i, j]
        mk = MODE_MARKER.get(mode, "*") if mode else "o"
        c = MODE_COLOR.get(mode, MODE_COLOR["unidentified"]) if mode else MUTED
        sc.plot(abs(d[i, j]), gam[i, j], mk, color=c, ms=7, mec=INK, mew=0.4, zorder=3)
        sc.annotate(f"{ky[j]:g}", (abs(d[i, j]), gam[i, j]), xytext=(4, 3),
                    textcoords="offset points", fontsize=6, color=MUTED)
    for mode in [x for x in MODE_MARKER if (m == x).any()]:
        sc.plot([], [], MODE_MARKER[mode], color=MODE_COLOR[mode], mec=INK, mew=0.4, ms=7,
                label=mode)
    if not (m != "").any():
        sc.plot([], [], "o", color=MUTED, ms=7, label="no mode (run --modes)")
    sc.axhline(0, color=MUTED, lw=0.8)
    sc.set_xlim(-0.01, 0.51)
    sc.set_xlabel(r"$|\delta|$  (0 on a rational surface, 0.5 midway)")
    sc.set_ylabel(r"$\gamma$ [$c_{ref}/L_{ref}$]")
    sc.set_title("GENE-converged gamma vs rational-surface offset (number: ky)", loc="left",
                 fontsize=10,
                 color=INK)
    sc.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(sc)
    fig.suptitle(f"NSTX {ds.shot} corners at x0 = {ds.x0:g}: q = m/n overlap", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "corners_q_offset.png", written)

    path = data_path(out, "corners_q_offset.csv")
    cols = [c for c in ("n0", "q0", "shat", "q_offset", "dx_rational", "status", "gamma", "mode")
            if c in ds]
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky"] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, k in enumerate(ky):
                row = []
                for col in cols:
                    v = ds[col].values[i, j] if ds[col].ndim == 2 else ds[col].values[i]
                    row.append(str(v) if ds[col].dtype.kind in "OUS" else f"{v:.6g}")
                fh.write(",".join([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}"] + row)
                         + "\n")
    written.append(path)


def _listdir(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def run_x0(run_dir: str, rel: str):
    """x0 of a run dir: the last x0_<x0> folder in *rel*, else its parameters' x0."""
    m = X0_RE.findall(rel)
    if m:
        return float(m[-1])
    for path in gene_runs(run_dir)[:1]:
        with open(path) as fh:
            m = re.search(r"^\s*x0\s*=\s*([-+.\deE]+)", fh.read(), re.M)
        return float(m.group(1)) if m else None
    return None


def res2_dirs(diag_root: str, x0, extra_roots=(), depth=RES2_DEPTH) -> dict[str, list[str]]:
    """{corner tag: [2x run dirs at x0]} under diag_root, its 2x siblings and extra_roots."""
    parent, base = os.path.split(diag_root.rstrip("/\\"))
    roots = [diag_root] + [os.path.join(parent, d) for d in sorted(_listdir(parent))
                           if d != base and d.startswith(base) and RES2_RE.search(d)]
    roots += [r for r in extra_roots if r not in roots]
    found: dict[str, list[str]] = {}
    other = set()
    for root in roots:
        top = os.path.dirname(os.path.normpath(root))
        for d, subs, files in os.walk(root):
            rel = os.path.relpath(d, top).replace(os.sep, "/")   # starts at root's own name
            run = (any(x.startswith("scanfiles") for x in subs)
                   or any(re.fullmatch(r"parameters_\d{4}", f) for f in files))
            subs[:] = [x for x in subs if not x.startswith("scanfiles") and x != "plots"]
            if rel.count("/") >= depth:
                subs[:] = []
            m = TAG_RE.search(rel)
            if not (run and m and RES2_RE.search(rel)):
                continue
            subs[:] = []
            x = run_x0(d, rel)
            if x is not None and abs(x - float(x0[0])) < 5e-4:
                found.setdefault(m.group(), []).append(d)
            else:
                other.add(x)
    if other:
        print(f"2x runs at other x0 left out: {sorted(other, key=lambda v: (v is None, v))}")
    return found


def run_nz0(par_path: str):
    with open(par_path) as fh:
        m = re.search(r"^\s*nz0\s*=\s*(\d+)", fh.read(), re.M)
    return int(m.group(1)) if m else None


def scan_nz0(ds):
    """nz0 of this scan, from the first corner that has a run."""
    for c in ds.corner.values:
        runs = gene_runs(os.path.join(ds.diag_root, x0_label((ds.x0,)), c))
        if runs:
            return run_nz0(runs[0])
    return None


def nz_dirs(root: str, x0, nz0: int) -> dict[str, list[str]]:
    """{corner tag: [run dir]} of the runs under <root>/x0_<x0>/<tag>/. A dir whose
    parameters do not have nz0 = *nz0* is reported and left out, so a run at the
    wrong resolution can never pose as the comparison."""
    base = os.path.join(root, x0_label(x0))
    found: dict[str, list[str]] = {}
    for name in sorted(_listdir(base)):
        d = os.path.join(base, name)
        runs = gene_runs(d) if TAG_RE.fullmatch(name) else []
        if not runs:
            continue
        got = run_nz0(runs[0])
        if got != nz0:
            print(f"--- nz0 {name}: parameters nz0 = {got}, not {nz0}; left out")
            continue
        found[name] = [d]
    if not found:
        print(f"no nz0 = {nz0} runs under {base}")
    return found


def live_fit(par_path: str):
    """(gamma, gamma_std, omega, omega_std, verdict) of field_omega's fit to the
    field_NNNN next to *par_path*, or None when there is no field output yet."""
    field = re.sub(r"parameters_(\d{4})$", r"field_\1", par_path)
    if not os.path.isfile(field) or os.path.getsize(field) == 0:
        return None
    try:
        _, stats = field_omega.fit(field)
    except Exception as exc:                    # a field file still being written
        print(f"    field fit failed {field}: {type(exc).__name__}: {exc}")
        return None
    f = stats[field_omega.FIELDS]
    g, w = f["gamma_avg"], f["omega_avg"]
    gs, ws = np.sqrt(f["gamma_variance"]), np.sqrt(f["omega_variance"])
    return g, gs, w, ws, field_omega.verdict(g, gs, w, ws)


def load_res2(ds, extra_roots=(), dirs=None, name="2x") -> dict:
    """{(corner tag, ky): 2x record} for every 2x run matching a corner in *ds*
    (or, with *dirs* = {tag: [run dir]}, for those runs, labelled *name*).

    A record holds gamma, omega and their std, source ('GENE' where GENE
    converged, else the verdict of the live field fit: 'pass', 'decaying',
    'ambiguous'; 'no fit' without field output) and the parameters path."""
    if dirs is None:
        dirs = res2_dirs(ds.diag_root, (ds.x0,), extra_roots)
    alt, out = name, {}
    for name, paths in sorted(dirs.items()):
        if name not in ds.corner.values:
            print(f"--- {alt} {name}: no 1x corner at x0 = {ds.x0:g}; left out")
            continue
        for d in paths:
            g = gene_corner(d)
            if g is None:
                continue
            for k, v in sorted(g[0].items()):
                rec = {"dir": d, "path": v[5], "gamma": v[0], "omega": v[1],
                       "gamma_std": np.nan, "omega_std": np.nan, "source": "GENE"}
                if v[2] != "ok":
                    fitted = live_fit(v[5])
                    if fitted is None:
                        rec.update(gamma=np.nan, omega=np.nan, source="no fit")
                    else:
                        rec.update(zip(("gamma", "gamma_std", "omega", "omega_std", "source"),
                                       fitted))
                old = out.get((name, k))
                if old is None or old["source"] != "GENE":   # GENE converged wins
                    out[(name, k)] = rec
        print(f"--- {alt} {name}: " + "; ".join(
            f"ky={k:g} {r['source']}" for (n, k), r in sorted(out.items()) if n == name))
    return out


def one_x(ds, i, var):
    """1x value per ky of corner i: GENE converged, else the recorded field fit; + std, conv."""
    conv = ds.status.values[i] == "ok"
    y = np.where(conv, ds[var].values[i], ds[f"{var}_field"].values[i])
    err = np.where(conv, np.nan, ds[f"{var}_field_std"].values[i])
    return y, err, conv


def parity(ax, pts, var, base, alt):
    """alt vs base for one variable: each point a corner and ky (number = ky), the
    1:1 line and a +-10% band."""
    for q in pts:
        ax.errorbar(q["x"], q["y"], xerr=q["xerr"], yerr=q["yerr"], fmt="D", ms=7,
                    mfc=q["color"] if q["filled"] else SURFACE, mec=q["color"], mew=1.6,
                    color=q["color"], capsize=2, lw=0.8, alpha=0.45 if q["weak"] else 1.0,
                    zorder=5)
        ax.annotate(f"{q['k']:g}", (q["x"], q["y"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=MUTED)
    vals = [v for q in pts for v in (q["x"], q["y"])] + [0.0]
    span = (max(vals) - min(vals)) or 1.0
    lim = [min(vals) - 0.05 * span, max(vals) + 0.05 * span]
    xs = np.linspace(*lim, 50)
    ax.fill_between(xs, 0.9 * xs, 1.1 * xs, color=GRID, lw=0, zorder=1)
    ax.plot(lim, lim, color=MUTED, lw=0.9, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    sym = {"gamma": r"\gamma", "omega": r"\omega"}[var]
    ax.set_xlabel(rf"${sym}$ {base} [$c_{{ref}}/L_{{ref}}$]")
    ax.set_ylabel(rf"${sym}$ {alt} [$c_{{ref}}/L_{{ref}}$]")
    ax.set_title(f"{alt} vs {base} {var} (number: ky; band: ±10%)", loc="left", fontsize=10,
                 color=INK)
    tidy(ax)


def pct_violins(ax, ky, pct, var, base, alt):
    """% difference (alt - base) / |base| at each ky: a violin (KDE over the corners)
    where two or more differ, and every corner as its own point on top."""
    gaps = np.diff(np.unique(ky))
    width = 0.8 * gaps.min() if gaps.size else 0.2      # one width, so violins compare
    for k in sorted(pct):
        vals = np.array([v for v, _, _ in pct[k]])
        if vals.size >= 2 and np.ptp(vals) > 0:
            vp = ax.violinplot(vals, positions=[k], widths=width, showextrema=False,
                               showmedians=True)
            for body in vp["bodies"]:
                body.set_facecolor(GRID)
                body.set_edgecolor(MUTED)
                body.set_alpha(0.9)
            vp["cmedians"].set_color(INK)
        for v, color, weak in pct[k]:
            ax.plot(k, v, "D", ms=5, color=color, alpha=0.45 if weak else 1.0, zorder=5)
    ax.axhspan(-10, 10, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$k_y\rho_{ref}$")
    ax.set_ylabel(f"{var}: ({alt} - {base}) / |{base}| [%]")
    ax.set_title(f"{var} % difference per ky (violin: KDE over corners; band: ±10%)",
                 loc="left", fontsize=10, color=INK)
    tidy(ax)


def plot_res2(ds, out, written, extra_roots=(), dirs=None, base="1x", alt="2x",
              stem="corners_res2", what="2x resolution check"):
    """Each corner with a 2x rerun: its 1x line (converged filled, recorded field
    fit hollow) and the 2x values as diamonds beside them (GENE converged filled,
    live field fit hollow +-1 std, faded unless it passes), joined to the 1x value
    at the same ky (top row); the % difference per ky as a violin over the corners,
    on the same ky axis (middle); 2x vs 1x for gamma and omega (bottom).
    + data/corners_res2.csv.

    *dirs*, *base*, *alt*, *stem* and *what* turn it into any other rerun
    comparison: corners_nz128.png is the nz0 = 128 reruns (alt) against the
    nz0 = 64 scan (base)."""
    res = load_res2(ds, extra_roots, dirs, alt)
    if not res:
        print(f"no {alt} runs at this x0; {stem}.png skipped")
        return
    bk, ak = (re.sub(r"\W", "", v.replace("nz0", "nz")) for v in (base, alt))   # CSV keys
    names = [c for c in ds.corner.values if any(n == c for n, _ in res)]
    ky = ds.ky.values
    nudge = RES2_NUDGE * (np.ptp(ky) if ky.size > 1 else 1.0)
    fig = plt.figure(figsize=(14, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 0.75, 1.05))
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    pct_axes = [fig.add_subplot(gs[1, n], sharex=axes[n]) for n in range(2)]
    par_axes = [fig.add_subplot(gs[2, n]) for n in range(2)]
    pts = {"gamma": [], "omega": []}
    pct = {"gamma": {}, "omega": {}}
    rows, faded = [], False
    for c in names:
        i = int(np.flatnonzero(ds.corner.values == c)[0])
        st = style(point_of(ds, c))
        one = {v: one_x(ds, i, v) for v in ("gamma", "omega")}
        for ax, var in zip(axes[:2], ("gamma", "omega")):
            y, err, conv = one[var]
            ok = np.isfinite(y)
            ax.plot(ky[ok], y[ok], lw=1.2, color=st["color"], ls=st["ls"], alpha=0.7, zorder=3)
            ax.plot(ky[ok & conv], y[ok & conv], st["marker"], ms=5, color=st["color"], zorder=4)
            g = ok & ~conv
            ax.errorbar(ky[g], y[g], yerr=err[g], fmt=st["marker"], ms=5, mfc=SURFACE,
                        mec=st["color"], color=st["color"], capsize=2, lw=0.8, zorder=4)
        for (n, k), r in sorted(res.items()):
            if n != c:
                continue
            j = int(np.argmin(abs(ky - k)))
            on_ky = abs(ky[j] - k) < 1e-6
            weak = r["source"] not in ("GENE", "pass")
            faded |= weak and np.isfinite(r["gamma"])
            for ax, var in zip(axes[:2], ("gamma", "omega")):
                v2, e2 = r[var], r[f"{var}_std"]
                if not np.isfinite(v2):
                    ax.plot(k + nudge, 0, "x", color=st["color"], ms=8, mew=1.8, zorder=6)
                    continue
                if on_ky and np.isfinite(one[var][0][j]):
                    ax.plot([k, k + nudge], [one[var][0][j], v2], ":", color=st["color"],
                            lw=1, zorder=4)
                filled = r["source"] == "GENE"
                ax.errorbar(k + nudge, v2, yerr=e2, fmt="D", ms=8,
                            mfc=st["color"] if filled else SURFACE, mec=st["color"], mew=1.8,
                            color=st["color"], capsize=3, lw=1, alpha=0.45 if weak else 1.0,
                            zorder=7)
            rel = {}
            for var in ("gamma", "omega"):
                v1, e1 = (one[var][0][j], one[var][1][j]) if on_ky else (np.nan, np.nan)
                rel[var] = (r[var] - v1) / abs(v1) if np.isfinite(v1) and v1 else np.nan
                if np.isfinite(v1) and np.isfinite(r[var]):
                    pts[var].append({"x": v1, "y": r[var], "xerr": e1, "yerr": r[f"{var}_std"],
                                     "color": st["color"], "filled": r["source"] == "GENE",
                                     "weak": weak, "k": k})
                    pct[var].setdefault(k, []).append((100 * rel[var], st["color"], weak))
            g1 = one["gamma"][0][j] if on_ky else np.nan
            st1 = ds.status.values[i, j] if on_ky else "missing"
            rows.append([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}",
                         st1 if st1 == "ok" or not np.isfinite(g1) else
                         f"field {ds.field_verdict.values[i, j]}".strip(),
                         *(f"{one[v][0][j]:.6g}" if on_ky else "nan" for v in ("gamma", "omega")),
                         r["source"], *(f"{r[v]:.6g}" for v in ("gamma", "gamma_std",
                                                               "omega", "omega_std")),
                         *(f"{rel[v]:.4g}" for v in ("gamma", "omega")), r["dir"]])
    for ax, var, sym in zip(axes[:2], ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    axes[0].set_title(f"growth rate: {base} line, {alt} diamonds", loc="left", fontsize=10,
                      color=INK)
    axes[1].set_title(f"real frequency: {base} line, {alt} diamonds", loc="left", fontsize=10,
                      color=INK)
    for n, var in enumerate(("gamma", "omega")):
        pct_violins(pct_axes[n], ky, pct[var], var, base, alt)
        parity(par_axes[n], pts[var], var, base, alt)
    handles = [Line2D([], [], lw=1.8, ms=5, **style(point_of(ds, c)),
                      label=str(ds.label.values[list(ds.corner.values).index(c)]))
               for c in names]
    extra = [Line2D([], [], ls="none", marker="o", color=MUTED, ms=5, label=f"{base} GENE converged"),
             Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, ms=5,
                    label=f"{base} recorded field fit ±1σ"),
             Line2D([], [], ls="none", marker="D", color=MUTED, ms=7, label=f"{alt} GENE converged"),
             Line2D([], [], ls="none", marker="D", mfc=SURFACE, mec=MUTED, mew=1.8, ms=7,
                    label=f"{alt} field fit, passing ±1σ")]
    if faded:
        extra.append(Line2D([], [], ls="none", marker="D", mfc=SURFACE, mec=MUTED, mew=1.8,
                            ms=7, alpha=0.45, label=f"{alt} field fit, ambiguous/decaying"))
    if any(not np.isfinite(r["gamma"]) for r in res.values()):
        extra.append(Line2D([], [], ls="none", marker="x", color=MUTED, ms=7, mew=1.8,
                            label=f"{alt}, no field output yet"))
    legend_right(fig, handles, extra)
    fig.suptitle(f"NSTX {ds.shot} corners at x0 = {ds.x0:g}: {what} "
                 f"({alt} not converged: live field fit)", x=0.02, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.97))
    save(fig, out, f"{stem}.png", written)

    path = data_path(out, f"{stem}.csv")
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky", f"status_{bk}", f"gamma_{bk}", f"omega_{bk}",
                           f"source_{ak}", f"gamma_{ak}", f"gamma_{ak}_std", f"omega_{ak}",
                           f"omega_{ak}_std", "gamma_rel_diff", "omega_rel_diff",
                           f"dir_{ak}"]) + "\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
    written.append(path)


def plot_nz_compare(ds, out, written, nz_compare, nz_root=None):
    """corners_nz<N>.png: each corner this scan ran beside its nz0 = N run of the
    same profile scaling, from <diag-root>_nz<N> (or nz_root)."""
    root = nz_root or ds.diag_root.rstrip("/\\") + NZ_SUFFIX.format(nz_compare)
    own = scan_nz0(ds)
    dirs = nz_dirs(root, (ds.x0,), nz_compare)
    # ds lists every ped_grid corner, run or not: compare only the profile
    # scalings this scan actually ran, not every nz0 = N corner
    ran = {t for t in dirs if gene_runs(os.path.join(ds.diag_root, x0_label((ds.x0,)), t))}
    if set(dirs) - ran:
        print(f"--- nz0 {nz_compare}: {len(set(dirs) - ran)} corner(s) with no nz0 {own} run "
              "left out")
    plot_res2(ds, out, written, dirs={t: dirs[t] for t in sorted(ran)},
              base=f"nz0 {own}", alt=f"nz0 {nz_compare}", stem=f"corners_nz{nz_compare}",
              what=f"nz0 {own} vs {nz_compare}")


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
    has_q = "q" in ds
    fig = plt.figure(figsize=(3.6 * len(vars_) + 2.8, 9.4 if has_q else 6.4))
    gs = fig.add_gridspec(3 if has_q else 2, len(vars_),
                          height_ratios=(1, 1, 1.25) if has_q else (1, 1))
    ax = np.empty((2, len(vars_)), dtype=object)
    for j in range(len(vars_)):
        ax[0, j] = fig.add_subplot(gs[0, j], sharex=ax[0, 0] if j else None)
        ax[1, j] = fig.add_subplot(gs[1, j], sharex=ax[0, 0])
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
        if not has_q:
            bot.set_xlabel(r"$\rho_{tor}$")
        for a in (top, bot):
            a.axvline(ds.x0, color=INK, ls=":", lw=1.4)
            a.set_xlim(*xlim)
            tidy(a)
    if has_q:                                     # one under each column, zoomed on x0
        qlim = (max(float(x.min()), ds.x0 - Q_ZOOM), min(float(x.max()), ds.x0 + Q_ZOOM))
        for a in ax.ravel():
            a.axvspan(*qlim, color=GRID, alpha=0.45, lw=0, zorder=0)
        for j in range(len(vars_)):
            plot_q(fig.add_subplot(gs[2, j]), ds, x, qlim,
                   labels=j == len(vars_) - 1, title=j == 0)
    handles = [Line2D([], [], color=style(point_of(ds, c))["color"],
                      ls=style(point_of(ds, c))["ls"], lw=1.8, label=str(l))
               for c, l in zip(ds.corner.values, ds.label.values)]
    legend_right(fig, handles, [Line2D([], [], color=INK, ls=":", lw=1.4,
                                       label=f"x0 = {ds.x0:g}")])
    fig.suptitle(f"NSTX {ds.shot} corner profiles (profiles.iterdb)"
                 + (f", rho_tor {xlim[0]:g}-{xlim[1]:g}" if name.endswith("_ped.png") else ""),
                 x=0.04, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, name, written)


def plot_q(ax, ds, x, xlim, labels=True, title=True):
    """|q| vs rho_tor per corner, a faint line at each integer q in view and, per
    corner, a drop line from the axis to where it crosses that integer."""
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        ax.plot(x, ds.q.sel(corner=c), color=st["color"], ls=st["ls"], lw=1.5, zorder=3)
    # y range over the window, short of the separatrix where q runs away
    fit_ylim(ax, x, ds.q.values, (xlim[0], min(xlim[1], 0.99)))
    lo, hi = ax.get_ylim()
    ints = np.arange(np.ceil(max(lo, 0)), np.floor(hi) + 1)
    every = max(1, int(np.ceil(ints.size / 20)))      # label at most ~20 integers
    for k, n in enumerate(ints):
        ax.axhline(n, color=GRID, lw=0.7, zorder=1)
        if labels and k % every == 0:
            ax.text(1.003, n, f"{n:g}", transform=ax.get_yaxis_transform(), fontsize=7,
                    va="center", color=MUTED)
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        y = ds.q.sel(corner=c).values
        for n in ints:
            r = integer_crossings(x, y, n)
            r = r[(r >= xlim[0]) & (r <= xlim[1])]
            if r.size:
                ax.vlines(r, lo, n, color=st["color"], ls=st["ls"], lw=0.8, alpha=0.75,
                          zorder=2)
    ax.axvline(ds.x0, color=INK, ls=":", lw=1.4)
    ax.set_ylim(lo, hi)
    if 2 <= ints.size <= 20:                      # ticks on the integers themselves
        ax.set_yticks(ints)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\rho_{tor}$")
    if title:
        ax.set_title(rf"$|q|$ (cheaseBS geqdsk), $x_0 \pm {Q_ZOOM:g}$ (shaded above); drop "
                     r"lines at each corner's integer-$q$ surfaces", loc="left", fontsize=10,
                     color=INK)
    tidy(ax)


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
    csv = data_path(out, "corners_local_params.csv")
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
        x0=X0, modes=False, field_fits=False, shot=SHOT, res2_roots=(), nz_compare=None,
        nz_root=None):
    x0 = tuple(x0)
    out = out or default_out(x0, diag_root)
    ds = load_dataset(diag_root, ped_root, x0, modes, field_fits, shot)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN corners and rows
        plot_growth(ds, out, written)
        plot_growth(ds, out, written, guesses=True)
        plot_profiles(ds, out, written)
        plot_profiles(ds, out, written, PED_ZOOM, "corners_profiles_ped.png")
        plot_s_alpha(ds, out, written)
        plot_local(ds, out, written, spread, max_params)
        plot_q_offset(ds, out, written)
        plot_res2(ds, out, written, res2_roots)
        if nz_compare:
            plot_nz_compare(ds, out, written, nz_compare, nz_root)
        if "mode" in ds:
            write_modes_csv(ds, out, written)
            plot_mode_scaling(ds, out, written)
    print("\nwrote\n  " + "\n  ".join(written))
    return ds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=SHOT, choices=sorted(SHOTS),
                    help="discharge; sets the --diag-root, --ped-root and --x0 defaults")
    ap.add_argument("--diag-root", default=None,
                    help="GENE output root; corners are under <diag-root>/x0_<x0>/<tag>/ "
                         f"[{DIAG_ROOT}]")
    ap.add_argument("--ped-root", default=None,
                    help=f"ped_grid cheaseBS cases <tag>/ [{PED_ROOT}]")
    ap.add_argument("--x0", type=float, nargs="+", default=None,
                    help="radius of the scan to plot (reads its x0_<x0>/ folder) "
                         "[Te-top Chapman radius]")
    ap.add_argument("--out", default=None,
                    help="directory for the PNGs, CSVs in its data/ "
                         "[<diag-root>/x0_<x0>/plots]")
    ap.add_argument("--field-fits", action="store_true",
                    help="carry recorded field fits that did not pass (decaying/ambiguous, "
                         "from field_omega.py --write) into the mode table, modes CSV and "
                         "gamma_max (the growth plots show every fit regardless)")
    ap.add_argument("--modes", action="store_true",
                    help="classify each ky with TPED mode_identifier (reads nrg, field, "
                         "geometry; slower) and mark the growth plot by mode")
    ap.add_argument("--res2-root", nargs="+", default=(),
                    help="more dirs to search for 2x-resolution runs (2XRES / _res2 in the "
                         "path) [<diag-root> and its _res2 siblings]")
    ap.add_argument("--nz-compare", type=int, default=None, metavar="N",
                    help="compare each corner with its nz0 = N run of the same profile "
                         "scaling (corners_nz<N>.png) [off]")
    ap.add_argument("--nz-root", default=None,
                    help="diag root of the --nz-compare runs, <tag>s under x0_<x0>/ "
                         "[<diag-root>_nz<N>]")
    ap.add_argument("--spread", type=float, default=SPREAD,
                    help="(max-min)/|mean| above which a local parameter counts as varying")
    ap.add_argument("--max-params", type=int, default=MAX_PARAMS,
                    help="most varying parameters to chart and tabulate")
    args = ap.parse_args()
    d = shot_defaults(args.shot)
    run(args.diag_root or d["diag_root"], args.ped_root or d["ped_root"], args.out, args.spread,
        args.max_params, args.x0 or d["x0"], args.modes, args.field_fits, args.shot,
        args.res2_root, args.nz_compare, args.nz_root)
