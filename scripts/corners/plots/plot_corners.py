"""Plots for the 16-corner linear GENE scan (corner_gene_scan.py): growth, profiles, s-alpha, locals.

Reads every corner corner_gene_scan.py launched,

    GENE output   <diag-root>/x0_<x0>/<tag>/scanfilesNNNN/{parameters,omega}_000K
                  (every scanfiles dir, merged by ky)
    profiles      <ped-root>/<tag>/profiles.iterdb   (what GENE read, prof_type -2)

into one xarray Dataset with dims (corner, ky, rho_tor), and writes to <out>/

    corners_growth.png          gamma and omega vs ky, all corners
    corners_profiles.png        ne, Te, ni, Ti and their -dln/drho_tor, x0 marked
    corners_profiles_ped.png    the same, zoomed to PED_ZOOM
    corners_s_alpha.png         shat vs alpha_MHD at x0, coloured by max_ky gamma,
                                points numbered with a key beside the plot
    corners_local_params.png    the local parameters that vary most: % from the corner
                                mean, or the absolute difference for those that change sign
    corners_local_table.png     their values per corner, as a table
    corners_local_params.csv    every local parameter per corner (+ gamma_max)

Every legend lists a corner as its knob factors in the order (Tew, new, Teh, neh),
e.g. (0.8, 1.2, 0.8, 0.8). Colour hue is the width pair (Tew, new), shade
the height pair (Teh, neh); line style repeats Teh and marker neh, so no
corner is told apart by colour alone.

    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --out ./corner_plots
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --spread 0.02 --max-params 16
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737

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
from NSTXU_lithium_study.scripts.corner_gene_scan import (DIAG_ROOT, KYMIN, PED_ROOT, X0,
                                                         corners, tag, x0_label)


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
TEH_LS = {0.8: "-", 1.2: "--"}
NEH_MARKER = {0.8: "o", 1.2: "s"}
GAMMA_CMAP = LinearSegmentedColormap.from_list("gamma", ["#008300", "#f0efec", "#e34948"])
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
DPI = 150


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
            "ls": TEH_LS[p["Teh"]], "marker": NEH_MARKER[p["neh"]]}


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


def read_run(par_path: str) -> dict:
    """One scan member through TPED's GeneSimulation: kymin, gamma, omega, status, locals."""
    sim = GeneSimulation(par_path)
    # No omega file at all: skip the load, TPED would log a traceback for the None path.
    tree = sim.return_sim_xarray(["omega"] if sim.omega_filepath else [])
    par = tree["parameters"].ds
    rec = {"kymin": float(par["kymin"]), "gamma": np.nan, "omega": np.nan, "status": "missing"}
    if "omega" in tree.children:
        om = tree["omega"].ds
        rec["status"] = "ok" if "gamma" in om else "empty"
        if "gamma" in om:
            rec["gamma"], rec["omega"] = float(om["gamma"]), float(om["omega"])
            if rec["gamma"] == 0 and rec["omega"] == 0:   # GENE's "not converged" placeholder
                rec["gamma"] = rec["omega"] = np.nan
                rec["status"] = "zero"
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


def gene_corner(corner_dir: str):
    """({kymin: (gamma, omega, status)}, local parameters) of one corner, or None."""
    runs = gene_runs(corner_dir)
    if not runs:
        return None
    by_ky: dict[float, tuple] = {}
    params: dict[str, list[float]] = {}
    for path in runs:
        try:
            rec = read_run(path)
        except Exception as exc:                # one bad member never sinks the corner
            print(f"    unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        # The same ky in a later scanfiles dir replaces the earlier one, unless
        # the earlier converged and the later did not.
        k = round(rec["kymin"], 6)
        if k not in by_ky or rec["status"] == "ok" or by_ky[k][2] != "ok":
            by_ky[k] = (rec["gamma"], rec["omega"], rec["status"])
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

def load_dataset(diag_root: str = DIAG_ROOT, ped_root: str = PED_ROOT, x0=X0) -> xr.Dataset:
    """Every corner with GENE output or profiles, as one Dataset (corner, ky, rho_tor).

    corner is the ped_grid tag; Tew/new/Teh/neh ride along as coordinates, so
    ds.set_index(corner=["Tew", "new", "Teh", "neh"]).unstack("corner") gives
    the 4D box.
    """
    gene, prof, pts = {}, {}, []
    for p in ordered(corners()):
        name = tag(p)
        g = gene_corner(os.path.join(diag_root, x0_label(x0), name))
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
    keys = sorted({k for g in gene.values() for k in g[1]})
    for k in keys:
        ds[k] = ("corner", [gene[n][1].get(k, np.nan) if n in gene else np.nan for n in names])
    if "alpha_MHD" in ds:
        ds["alpha_MHD"].attrs["formula"] = "q0^2 * major_R * dpdx_pm (GENE units)"
    finite = ds["gamma"].notnull().any("ky")
    ds["gamma_max"] = ds["gamma"].fillna(-np.inf).max("ky").where(finite)
    ds["ky_at_max"] = xr.DataArray(ky[ds["gamma"].fillna(-np.inf).argmax("ky").values],
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
    ds.attrs.update(x0=float(x0[0]), diag_root=diag_root, ped_root=ped_root)
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    unconv = False
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        for ax, var in zip(axes, ("gamma", "omega")):
            ax.plot(ds.ky, ds[var].sel(corner=c), lw=1.6, ms=5, mfc="none", **st)
            bad = ds.ky.values[np.isin(ds.status.sel(corner=c).values, ("empty", "zero"))]
            if bad.size:
                unconv = True
                ax.plot(bad, np.zeros(bad.size), "x", color=st["color"], ms=9, mew=2)
    for ax, var, sym in zip(axes, ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    axes[0].set_title("growth rate", loc="left", fontsize=10, color=INK)
    axes[1].set_title("real frequency", loc="left", fontsize=10, color=INK)
    extra = [Line2D([], [], ls="none", marker="x", color=INK, ms=8, mew=2,
                    label="unconverged (empty or 0/0 omega)")] if unconv else []
    legend_right(fig, corner_handles(ds), extra)
    fig.suptitle(f"NSTX 129038 corners, linear GENE at x0 = {ds.x0:g}", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, "corners_growth.png", written)


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


def plot_s_alpha(ds, out, written):
    if "shat" not in ds or "alpha_MHD" not in ds:
        print("no shat/alpha_MHD (no GENE parameters read); s-alpha plot skipped")
        return
    a, s, g = ds.alpha_MHD.values, ds.shat.values, ds.gamma_max.values
    ok = np.isfinite(a) & np.isfinite(s)
    if not ok.any():
        print("shat/alpha_MHD all NaN; s-alpha plot skipped")
        return
    gfin = g[ok & np.isfinite(g)]
    span = max(np.abs(gfin).max(), 1e-6) if gfin.size else 1.0
    norm = Normalize(-span, span)                # symmetric: the grey middle is gamma = 0
    fig, (ax, key) = plt.subplots(1, 2, figsize=(12.5, 7), width_ratios=(3, 1.2))
    has = ok & np.isfinite(g)
    sc = ax.scatter(a[has], s[has], c=g[has], cmap=GAMMA_CMAP, norm=norm, s=190,
                    edgecolors=INK, linewidths=0.8, zorder=5)
    if (ok & ~np.isfinite(g)).any():
        ax.scatter(a[ok & ~np.isfinite(g)], s[ok & ~np.isfinite(g)], s=190, facecolors="none",
                   edgecolors=MUTED, linewidths=1.2, zorder=5, label="no converged ky")
        ax.legend(frameon=False, fontsize=8, loc="best")
    # Points carry a number; the key beside the plot maps it to the corner, so
    # points that sit close together never stack their labels.
    lines = []
    for n, i in enumerate(np.flatnonzero(ok), start=1):
        ax.annotate(str(n), (a[i], s[i]), ha="center", va="center", fontsize=7,
                    fontweight="bold", color=INK, zorder=6)
        gi = f"{g[i]:.3f}" if np.isfinite(g[i]) else "-"
        lines.append(f"{n:>2}  {ds.label.values[i]:<21} {gi:>6}")
    key.axis("off")
    key.text(0, 1, f"{'#':>2}  {LEGEND_TITLE:<21} {'max g':>6}\n" + "\n".join(lines),
             family="monospace", fontsize=8, va="top", color=INK, transform=key.transAxes)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"$\max_{k_y}\gamma$ [$c_{ref}/L_{ref}$]")
    ax.set_xlabel(r"$\alpha_{MHD} = q_0^2\,(R_0/L_{ref})\,(-d\beta/dx)$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.margins(x=0.08, y=0.08)
    ax.set_title(f"s-alpha at x0 = {ds.x0:g}", loc="left", fontsize=10, color=INK)
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
        x0=X0):
    x0 = tuple(x0)
    out = out or default_out(x0)
    ds = load_dataset(diag_root, ped_root, x0)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN corners and rows
        plot_growth(ds, out, written)
        plot_profiles(ds, out, written)
        plot_profiles(ds, out, written, PED_ZOOM, "corners_profiles_ped.png")
        plot_s_alpha(ds, out, written)
        plot_local(ds, out, written, spread, max_params)
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
    ap.add_argument("--spread", type=float, default=SPREAD,
                    help="(max-min)/|mean| above which a local parameter counts as varying")
    ap.add_argument("--max-params", type=int, default=MAX_PARAMS,
                    help="most varying parameters to chart and tabulate")
    args = ap.parse_args()
    run(args.diag_root, args.ped_root, args.out, args.spread, args.max_params, args.x0)
