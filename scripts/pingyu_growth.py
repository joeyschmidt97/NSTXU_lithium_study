"""Ping-Yu's 132588 GENE runs next to ours: growth, frequency and convergence cost by x0.

Reads the per-run netCDFs that build_discharge_data wrote,

    ours      data/simulation_data/132588/<scan>/x0_<x0>/<case>/scanfilesNNNN/*.nc
    Ping-Yu   data/simulation_data/pingyu/132588/NSTX_KBM/<scaling>/...      (x0 0.80/0.85/0.90)
              data/simulation_data/pingyu/132588/scan_all/*.nc               (x0 0.84-0.965,
                                                                             omt_e x beta scan)

and writes to sparse_grid_results/132588/pingyu/

    pingyu_baseline_growth.png  gamma and omega vs ky at every baseline x0: ours (corners
                                ref nx0 3 solid, corners_nx7 dashed) and Ping-Yu's org runs.
                                Marker = mode; filled = GENE converged; hollow = field fit
                                that passed +-1 std; green edge = decaying fit; faded =
                                ambiguous fit. Column title: beta, omt_e, omn_e there.
    pingyu_scan_growth.png      the same for scan_all, one line per (omt_e, beta) point,
                                coloured by beta
    pingyu_convergence.png      (a) how each x0's runs ended, ours and his; (b) wall time
                                against gamma, coloured by how the run ended; (c) simulated
                                time GENE needed to converge against gamma, with the fitted
                                power law and our marginal gamma marked; (d) beta, omt_e,
                                omn_e against x0, both sets
    data/pingyu_runs.csv        one row per run (duplicates dropped), every value plotted

Duplicate runs (Ping-Yu's org/ holds copies of org/scanfiles0000/ runs) are dropped on
(src, x0, kymin, beta, omt_e, omn_e, nx0, gamma_best, omega_best).
Simulated time is steps x dt_max; linear GENE keeps dt fixed, so this is the run's t.

    python -m NSTXU_lithium_study.scripts.pingyu_growth
    python -m NSTXU_lithium_study.scripts.pingyu_growth --out sparse_grid_results/132588/pingyu
"""

from __future__ import annotations

import argparse
import glob
import os

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OURS = os.path.join(HERE, "data", "simulation_data", "132588")
PINGYU = os.path.join(HERE, "data", "simulation_data", "pingyu", "132588")
OUT = os.path.join(HERE, "sparse_grid_results", "132588", "pingyu")

KEYS = ("x0", "kymin", "nx0", "nz0", "nv0", "nw0", "beta", "omn_e", "omt_e", "coll", "q0",
        "shat", "timelim", "ntimesteps", "number_of_computed_time_steps",
        "time_for_initial_value_solver", "dt_max", "cost_hit_timelim", "cost_hit_ntimesteps",
        "omega_source", "fit_verdict", "gamma_best", "omega_best", "fit_gamma", "fit_omega",
        "fit_gamma_std", "fit_omega_std", "mode_ID")
DEDUP = ("src", "x0", "kymin", "beta", "omt_e", "omn_e", "nx0", "gamma_best", "omega_best")

# plot_corners palette and mode markers (not imported: plot_corners pulls in TPED)
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
MODE_MARKER = {"ITG_TEM": "o", "ETG": "^", "KBM": "s", "MTM": "v", "unidentified": "*"}
DECAY_EDGE = "#008300"
DPI = 150
#: how a run ended, in plot order, and its colour in the convergence panels
ENDS = {"GENE converged": "#2a78d6", "field fit pass": "#1baf7a",
        "decaying fit": "#008300", "ambiguous": "#9b9a92"}
SOURCE_STYLE = {"ours nx0 3": ("#1f1f1e", "-"), "ours nx0 7": ("#6b6a63", "--"),
                "Ping-Yu org": ("#eb6834", "-")}


def read(pattern: str, src: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(pattern, recursive=True):
        f = f.replace("\\", "/")
        with netCDF4.Dataset(f) as d:
            r = {"src": src, "file": f}
            for k in KEYS:
                v = np.nan
                if k in d.variables:
                    v = d.variables[k][...]
                    try:
                        v = v.item()
                    except (AttributeError, ValueError):
                        v = str(v)
                    if isinstance(v, bytes):
                        v = v.decode()
                r[k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def label_runs(d: pd.DataFrame) -> pd.DataFrame:
    d = d.drop_duplicates(subset=list(DEDUP)).copy()
    d["x0"] = d.x0.astype(float).round(3)
    gene = d.omega_source == "GENE"
    d["end"] = np.select([gene, d.fit_verdict == "pass", d.fit_verdict == "decaying"],
                         ["GENE converged", "field fit pass", "decaying fit"], "ambiguous")
    d["gamma"] = np.where(gene, d.gamma_best, d.fit_gamma)
    d["omega"] = np.where(gene, d.omega_best, d.fit_omega)
    d["gamma_err"] = np.where(gene, np.nan, d.fit_gamma_std)
    d["omega_err"] = np.where(gene, np.nan, d.fit_omega_std)
    d["t_sim"] = d.number_of_computed_time_steps * d.dt_max
    d["wall_s"] = d.time_for_initial_value_solver
    d["mode_ID"] = d.mode_ID.fillna("unidentified").replace({"nan": "unidentified"})
    path = d.file.str.replace(r"^.*/simulation_data/", "", regex=True)
    d["set"] = np.select(
        [path.str.startswith("132588/corners_nx7/"), path.str.startswith("132588/corners/"),
         path.str.contains("/NSTX_KBM/org/"), path.str.contains("/scan_all/"),
         path.str.contains("/NSTX_KBM/")],
        ["ours nx0 7", "ours nx0 3", "Ping-Yu org", "Ping-Yu scan_all", "Ping-Yu scalings"], "")
    d["baseline"] = d.set.isin(["ours nx0 7", "Ping-Yu org"]) | (
        (d.set == "ours nx0 3") & path.str.contains("Teh1.00_Tew1.00_neh1.00_new1.00"))
    return d


def load() -> pd.DataFrame:
    d = pd.concat([read(os.path.join(OURS, "**", "*.nc"), "ours"),
                   read(os.path.join(PINGYU, "**", "*.nc"), "pingyu")], ignore_index=True)
    return label_runs(d)


def tidy(ax):
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_points(ax, g, var, color, ls, lw=1.4):
    """One line through g (sorted by ky), markers by mode and how each run ended."""
    g = g.sort_values("kymin")
    ok = np.isfinite(g[var].astype(float))
    ax.plot(g.kymin[ok], g[var][ok], color=color, ls=ls, lw=lw, zorder=3)
    for _, r in g[ok].iterrows():
        mk = MODE_MARKER.get(r.mode_ID, "*")
        if r.end == "GENE converged":
            ax.plot(r.kymin, r[var], mk, ms=6, color=color, mec=color, zorder=6)
            continue
        edge = DECAY_EDGE if r.end == "decaying fit" else color
        ax.errorbar(r.kymin, r[var], yerr=r[f"{var}_err"], fmt=mk, ms=6, mfc=SURFACE, mec=edge,
                    mew=1.3, color=edge, capsize=2, lw=0.8,
                    alpha=0.45 if r.end == "ambiguous" else 1.0, zorder=5)


def end_handles():
    return [Line2D([], [], ls="none", marker="o", color=MUTED, ms=6, label="GENE converged"),
            Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.3, ms=6,
                   label="field fit pass ±1σ"),
            Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=DECAY_EDGE, mew=1.3, ms=6,
                   label="decaying fit ±1σ"),
            Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.3, ms=6,
                   alpha=0.45, label="ambiguous fit ±1σ")] + \
           [Line2D([], [], ls="none", marker=m, color=MUTED, ms=6, label=f"mode: {k}")
            for k, m in MODE_MARKER.items()]


def column_title(g) -> str:
    return (f"x0 = {g.x0.iloc[0]:g}\nβ {g.beta.median():.3f}, omt_e {g.omt_e.median():.1f}, "
            f"omn_e {g.omn_e.median():.1f}")


def plot_baseline(d, out, written):
    b = d[d.baseline]
    x0s = sorted(b.x0.unique())
    fig, axes = plt.subplots(2, len(x0s), figsize=(3.3 * len(x0s) + 2.6, 6.6), squeeze=False)
    for j, x0 in enumerate(x0s):
        g = b[b.x0 == x0]
        for i, var in enumerate(("gamma", "omega")):
            ax = axes[i, j]
            for s, (c, ls) in SOURCE_STYLE.items():
                if (g.set == s).any():
                    draw_points(ax, g[g.set == s], var, c, ls)
            ax.axhline(0, color=MUTED, lw=0.8)
            tidy(ax)
            if i == 1:
                ax.set_xlabel(r"$k_y\rho_{ref}$")
        axes[0, j].set_title(column_title(g), loc="left", fontsize=9, color=INK)
    axes[0, 0].set_ylabel(r"$\gamma$ [$c_{ref}/L_{ref}$]")
    axes[1, 0].set_ylabel(r"$\omega$ [$c_{ref}/L_{ref}$]")
    handles = [Line2D([], [], color=c, ls=ls, lw=1.8, label=s)
               for s, (c, ls) in SOURCE_STYLE.items()] + end_handles()
    fig.legend(handles=handles, loc="center right", frameon=False, fontsize=8)
    fig.suptitle("NSTX 132588 baseline, linear GENE: ours (x0 ≤ 0.785) and Ping-Yu's org "
                 "(x0 ≥ 0.80)", x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1 - 2.4 / fig.get_figwidth(), 0.95))
    save(fig, out, "pingyu_baseline_growth.png", written)


def plot_scan(d, out, written):
    s = d[d.set == "Ping-Yu scan_all"]
    x0s = sorted(s.x0.unique())
    cmap = plt.get_cmap("viridis")
    lo, hi = s.beta.min(), s.beta.max()
    fig, axes = plt.subplots(2, len(x0s), figsize=(2.9 * len(x0s) + 1.6, 6.6), squeeze=False)
    for j, x0 in enumerate(x0s):
        g = s[s.x0 == x0]
        for i, var in enumerate(("gamma", "omega")):
            ax = axes[i, j]
            for _, p in g.groupby(["omt_e", "beta", "nx0"]):
                c = cmap(0.9 * (p.beta.iloc[0] - lo) / (hi - lo + 1e-12))
                draw_points(ax, p, var, c, "-", lw=0.9)
            ax.axhline(0, color=MUTED, lw=0.8)
            tidy(ax)
            if i == 1:
                ax.set_xlabel(r"$k_y\rho_{ref}$")
        n_amb = (g.end == "ambiguous").sum()
        axes[0, j].set_title(f"x0 = {x0:g}\n{len(g)} runs, {n_amb} ambiguous", loc="left",
                             fontsize=9, color=INK)
    axes[0, 0].set_ylabel(r"$\gamma$ [$c_{ref}/L_{ref}$]")
    axes[1, 0].set_ylabel(r"$\omega$ [$c_{ref}/L_{ref}$]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, lo + (hi - lo) / 0.9))
    fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.01, label="β (line colour)")
    fig.suptitle("NSTX 132588, Ping-Yu scan_all (omt_e × β per x0), linear GENE", x=0.02,
                 ha="left", fontsize=12, color=INK)
    save(fig, out, "pingyu_scan_growth.png", written)


def converge_fit(d):
    """t_sim = A gamma^-p over runs GENE converged with gamma > 0."""
    c = d[(d.end == "GENE converged") & (d.gamma > 0.005) & (d.t_sim > 0)]
    p, logA = np.polyfit(np.log(c.gamma), np.log(c.t_sim), 1)
    return c, np.exp(logA), -p


def plot_convergence(d, out, written):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    # (a) how runs ended per x0
    a = ax[0, 0]
    t = d.groupby(["x0", "end"]).size().unstack(fill_value=0).reindex(columns=list(ENDS),
                                                                       fill_value=0)
    share = t.div(t.sum(axis=1), axis=0)
    xs = np.arange(len(t))
    bottom = np.zeros(len(t))
    for e, c in ENDS.items():
        a.bar(xs, share[e], 0.75, bottom=bottom, color=c, edgecolor=SURFACE, label=e)
        bottom += share[e].values
    for x, (x0, n) in zip(xs, t.sum(axis=1).items()):
        a.text(x, 1.01, str(n), ha="center", fontsize=8, color=MUTED)
    ours = d[d.src == "ours"].x0.unique()
    a.set_xticks(xs, [f"{x:g}" + ("*" if x in ours else "") for x in t.index], rotation=45)
    a.set_ylabel("share of runs")
    a.set_xlabel("x0  (* ours, rest Ping-Yu; run count on top)")
    a.set_title("(a) how runs ended, every scan", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    tidy(a)
    # (b) wall time vs gamma
    b = ax[0, 1]
    for e, c in ENDS.items():
        g = d[d.end == e]
        for src, mk in (("pingyu", "o"), ("ours", "D")):
            h = g[g.src == src]
            b.scatter(h.gamma, h.wall_s, s=14 if src == "pingyu" else 26, marker=mk, color=c,
                      alpha=0.6, lw=0.3, edgecolor=INK if src == "ours" else "none")
    for tl in sorted(d.timelim[d.timelim <= 1e5].dropna().unique()):   # 13 scaling runs carry 8.6e6
        b.axhline(tl, color=MUTED, lw=0.7, ls=":")
        b.text(0.99, tl, f"timelim {tl:g} s", fontsize=7, color=MUTED, va="bottom",
               ha="right", transform=b.get_yaxis_transform())
    b.set_yscale("log")
    b.set_xscale("symlog", linthresh=0.05)
    b.set_xlabel(r"$\gamma$ (GENE, else field fit) [$c_{ref}/L_{ref}$]")
    b.set_ylabel("wall time of the time loop [s]")
    b.set_title("(b) wall time against γ (◆ ours, ● Ping-Yu)", loc="left", fontsize=10)
    tidy(b)
    # (c) simulated time to converge vs gamma
    c_ax = ax[1, 0]
    c, A, p = converge_fit(d)
    for src, mk in (("pingyu", "o"), ("ours", "D")):
        h = c[c.src == src]
        c_ax.scatter(h.gamma, h.t_sim, s=14 if src == "pingyu" else 30, marker=mk,
                     color=ENDS["GENE converged"], alpha=0.5,
                     edgecolor=INK if src == "ours" else "none")
    u = d[(d.end != "GENE converged") & (d.t_sim > 0)]
    c_ax.scatter(u.gamma.clip(lower=0.004), u.t_sim, s=10, marker="x", color=MUTED, alpha=0.5,
                 label="not GENE-converged: t reached (γ from fit, ≤0 drawn at 0.004)")
    gg = np.logspace(np.log10(0.004), np.log10(5), 50)
    c_ax.plot(gg, A * gg ** -p, color=INK, lw=1.2,
              label=rf"fit to converged: $t = {A:.1f}\,\gamma^{{-{p:.2f}}}$")
    for gm in (0.02, 0.05):
        c_ax.axvline(gm, color="#eb6834", lw=0.8, ls="--")
        c_ax.text(gm, 0.02, f" γ {gm}: t ≈ {A * gm ** -p:.0f}", rotation=90, fontsize=7,
                  color="#eb6834", va="bottom", transform=c_ax.get_xaxis_transform())
    c_ax.set_xscale("log")
    c_ax.set_yscale("log")
    c_ax.set_xlabel(r"$\gamma$ [$c_{ref}/L_{ref}$]")
    c_ax.set_ylabel(r"simulated time reached [$L_{ref}/c_{ref}$]")
    c_ax.set_title("(c) simulated time GENE needed to converge", loc="left", fontsize=10)
    c_ax.legend(frameon=False, fontsize=8, loc="upper right")
    tidy(c_ax)
    # (d) drive against x0
    dd = ax[1, 1]
    m = d[d.baseline | (d.set == "Ping-Yu scan_all")].groupby(["src", "x0"]).agg(
        beta=("beta", "median"), omt=("omt_e", "median"), omn=("omn_e", "median"),
        conv=("end", lambda e: (e != "ambiguous").mean())).reset_index()
    tw = dd.twinx()
    for src, mk in (("ours", "D"), ("pingyu", "o")):
        h = m[m.src == src]
        dd.plot(h.x0, h.omt, mk + "-", color="#eb6834", ms=6, label=f"omt_e ({src})")
        dd.plot(h.x0, h.omn, mk + "--", color="#2a78d6", ms=6, label=f"omn_e ({src})")
        tw.plot(h.x0, h.beta, mk + ":", color=INK, ms=6, label=f"β ({src})")
        for _, r in h.iterrows():
            dd.annotate(f"{r.conv:.0%}", (r.x0, r.omt), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7, color=MUTED)
    dd.set_xlabel("x0")
    dd.set_ylabel("omt_e, omn_e (median)")
    tw.set_ylabel("β (median)")
    dd.set_title("(d) drive against x0; % = runs not ambiguous (baseline + scan_all)",
                 loc="left", fontsize=10)
    h1, l1 = dd.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    dd.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="center left")
    tidy(dd)
    fig.suptitle("NSTX 132588 linear GENE: convergence cost, ours and Ping-Yu's", x=0.02,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, out, "pingyu_convergence.png", written)
    return A, p


def save(fig, out, name, written):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)


def run(out: str = OUT) -> list[str]:
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    d = load()
    written = []
    d.drop(columns="file").assign(file=d.file.str.replace(r"^.*/simulation_data/", "",
                                                          regex=True)) \
        .to_csv(os.path.join(out, "data", "pingyu_runs.csv"), index=False)
    written.append(os.path.join(out, "data", "pingyu_runs.csv"))
    plot_baseline(d, out, written)
    plot_scan(d, out, written)
    A, p = plot_convergence(d, out, written)
    print(f"converged runs: t_sim = {A:.2f} * gamma^-{p:.2f}")
    for w in written:
        print("wrote", w)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT, help="output folder [sparse_grid_results/132588/pingyu]")
    run(ap.parse_args().out)
