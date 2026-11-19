"""Why does a numerics variant fail? Time traces, log-scale structure, parallel spectrum.

Companion to plot_variants.py (same variant roots and labels). plot_variants shows the
last-time |phi| / |Apar| and a gamma, omega line; at a steep-gradient radius that is not
enough to tell a numerical blow-up from a damped mode from an unconverged one. This reads
every field_NNNN of each variant x ky and writes, to <out> (default
<lithium_study>/<shot>/dbg_compare/x0_<x0>/diag/):

    diag_summary.png    variant x ky matrix coloured by verdict (below), with gamma and
                        omega in each cell; right column = the numerics that differ
                        between variants. The page to read first.
    diag_traces.png     gamma(t) and omega(t) from consecutive field frames (symlog y),
                        GENE's converged gamma as a dashed line, the field_omega fit
                        window shaded. A blow-up shows as gamma climbing without bound
                        late in the run; a damped mode as gamma settling below 0; two
                        competing modes as gamma(t) / omega(t) still oscillating.
    diag_structure.png  |phi|, |Apar| / max|phi| on a LOG axis along the ballooning
                        angle, with the 2 pi kx-connection boundaries. Shows tails that
                        a linear axis hides and where grid-scale noise sits.
    diag_spectrum.png   parallel power spectrum of phi (FFT along the ballooning grid),
                        top half of k_z shaded: power there is grid-scale (hyp_z / nz0).
    data/diagnostics.csv every number per variant x ky, plus all numerics per variant.

Verdict per run, first that applies:

    no field     no field_NNNN output
    blow-up      |gamma| > --gamma-cut (GENE value, else last trace value)
    grid-scale   phi power above half the parallel Nyquist > --nyq-cut, or the |phi|
                 peak spans fewer than --min-fwhm grid points (z under-resolved)
    extent       |phi| or |Apar| at either ballooning end > EXTENT_TOL (needs nx0 up)
    unsettled    gamma(t) still moving in the last 30 % (relative std > --settle-tol)
    damped       settled with gamma < 0: initial-value GENE only finds the least
                 damped mode, so the number is not a growth rate; check that no other
                 ky / branch grows, or use the eigenvalue solver
    growing      none of the above

Also per run: parity of phi and Apar (|sum f| / sum |f|; ~1 even, ~0 odd: ballooning
twisting parity = phi even / Apar odd, tearing = phi odd / Apar even), max|Apar|/max|phi|,
and 'purely growing' (|omega| < 1 % of |gamma|, the usual look of a numerical instability).

Note: with edge_opt > 0 GENE's z grid is non-uniform in the poloidal angle, so the
structure and spectrum here are on the computational grid, like every TPED plot.

    python -m NSTXU_lithium_study.scripts.corners.plots.variant_diagnostics --shot 129038 --x0 0.95
    python -m NSTXU_lithium_study.scripts.corners.plots.variant_diagnostics --shot 129038 --x0 0.95 --roots "/pscratch/.../129038/corners_dbg_*"
    python -m NSTXU_lithium_study.scripts.corners.plots.variant_diagnostics --shot 129038 --x0 0.95 --ky 0.3 1 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import os
import re

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField
from NSTXU_lithium_study.scripts.corner_gene_scan import x0_label
from NSTXU_lithium_study.scripts.corners.plots.plot_corners import (
    EXTENT_TOL, INK, MUTED, STRUCT_COLOR, data_path, gene_runs, live_fit, read_run, save, tidy)
from NSTXU_lithium_study.scripts.corners.plots.plot_variants import REF_TAG, variant_roots

#: Converged |gamma| above this is a numerical blow-up (as plot_variants).
GAMMA_CUT = 50.0
#: phi power fraction above half the parallel Nyquist that marks grid-scale structure.
NYQ_CUT = 0.05
#: Fewest grid points across the |phi| > max/2 region before z counts as under-resolved.
MIN_FWHM = 4
#: Relative std of gamma(t) over the last SETTLE_WINDOW of the trace that still counts as settled.
SETTLE_TOL = 0.05
SETTLE_WINDOW = 0.3
#: Absolute gamma std that counts as settled for |gamma| near 0.
SETTLE_ABS = 0.005
#: Frame pairs read per run for the traces (pairs of consecutive frames, evenly spaced).
MAX_PAIRS = 150
#: A frame-to-frame norm drop larger than this is GENE renormalising the linear run.
RENORM_DROP = 1e-3
LOG_FLOOR = 1e-7
#: Parameters reported per variant (string or number, as in the parameters file).
NUMERICS = ("edge_opt", "hyp_z", "hyp_z_order", "hyp_z_with_dz_prefactor", "hyp_v", "hyp_x",
            "hyp_y", "nz0", "nx0", "nexc", "nv0", "nw0", "lv", "lw", "dt_max", "courant",
            "calc_dt", "collision_op", "coll", "beta", "PRECISION", "timelim", "simtimelim",
            "parscheme", "arakawa_zv")
PHYSICS = ("q0", "shat", "beta", "coll", "omn_ions", "omt_ions", "omn_electrons",
           "omt_electrons")

VERDICTS = {                      # name: (colour, meaning for the legend)
    "growing": ("#008300", "growing, settled, clean"),
    "damped": ("#86b6ef", "settled, γ < 0 (least-damped only)"),
    "unsettled": ("#eda100", "γ(t) still moving"),
    "extent": ("#b58fd0", f"ends > {EXTENT_TOL:g} (nx0)"),
    "grid-scale": ("#eb6834", "grid-scale (hyp_z / nz0)"),
    "blow-up": ("#e34948", "blow-up |γ| > cut"),
    "no field": ("#e4e3df", "no field output"),
}
VARIANT_COLORS = ("#1f1f1e", "#2a78d6", "#eb6834", "#008300", "#6250d6", "#eda100",
                  "#e34948", "#3fb6c6", "#9a6b3f", "#b58fd0")


# --------------------------------------------------------------------------- reading

def param_text(par_path: str) -> dict:
    """{key: value} of every `key = value` line in a parameters file; numbers as float,
    everything else as the stripped string (quotes removed). Last value of a key wins,
    except species keys, which get _<name> suffixes from the &species blocks."""
    out, species = {}, None
    with open(par_path) as fh:
        for line in fh:
            m = re.match(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.+?)\s*$", line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip().strip("'\"")
            if key == "name":
                species = val
                continue
            try:
                v = float(val)
            except ValueError:
                v = val
            if species and key in ("omn", "omt", "mass", "temp", "dens", "charge"):
                out[f"{key}_{species}"] = v
            else:
                out[key] = v
    return out


def _quiet_field(field_path: str):
    with contextlib.redirect_stdout(io.StringIO()):
        return GeneField(field_path)


def _ballooning(gf: GeneField, times: list, name: str) -> np.ndarray:
    """(len(times), nx*nz) complex ballooning arrays of one field via TPED's transform."""
    d = gf._extract_field_data({"field_time": times}, times, [name])
    return np.asarray(d[name])


def field_diagnostics(par_path: str, max_pairs=MAX_PAIRS) -> dict | None:
    """Traces and last-frame structure from the field_NNNN next to *par_path*, or None."""
    field = re.sub(r"parameters_(\d{4})$", r"field_\1", par_path)
    if not os.path.isfile(field) or os.path.getsize(field) == 0:
        return None
    gf = _quiet_field(field)
    times = list(gf.all_time_values)
    if not times:
        return None
    nz, nx = int(gf.nz), int(gf.nx)
    out = {"nz": nz, "nx": nx, "zgrid": np.asarray(gf.zgrid), "t_end": times[-1],
           "n_frames": len(times)}

    # gamma(t), omega(t): lambda = log(<a, b> / <a, a>) / dt for consecutive frames a, b,
    # the amplitude-weighted version of TPED's log_weight (same sign convention).
    starts = np.unique(np.linspace(0, len(times) - 2, min(max_pairs, len(times) - 1))
                       .astype(int)) if len(times) > 1 else np.array([], int)
    tt, gam, om = [], [], []
    for i in starts:
        a, b = _ballooning(gf, [times[i], times[i + 1]], "field_phi")
        dt = times[i + 1] - times[i]
        na = np.vdot(a, a).real
        if dt <= 0 or na <= 0:
            continue
        ratio = np.vdot(a, b) / na
        if np.linalg.norm(b) < RENORM_DROP * np.sqrt(na) or ratio == 0:
            continue                                   # GENE renormalised between a and b
        lam = np.log(ratio) / dt
        tt.append(times[i + 1])
        gam.append(lam.real)
        om.append(lam.imag)
    out.update(t=np.array(tt), gamma_t=np.array(gam), omega_t=np.array(om))

    # last frame: structure, spectrum, parity
    phi = _ballooning(gf, [times[-1]], "field_phi")[0]
    apar = None
    if int(gf.param_dict.get("n_fields", 1)) > 1:
        apar = _ballooning(gf, [times[-1]], "field_apar")[0]
    ref = np.abs(phi).max()
    if ref == 0:
        return out
    phase = np.exp(-1j * np.angle(phi[np.argmax(np.abs(phi))]))
    out["phi"] = phi * phase / ref
    if apar is not None:
        out["apar"] = apar * phase / ref
    out.update(structure_stats(out["phi"], out.get("apar"), nz))
    return out


def structure_stats(phi: np.ndarray, apar, nz: int) -> dict:
    """End amplitudes, FWHM in points, Nyquist power, parity, EM ratio of a ballooning mode."""
    s = {}
    aphi = np.abs(phi)
    s["end_phi"] = float(max(aphi[0], aphi[-1]) / aphi.max())
    s["fwhm_pts"] = int(np.count_nonzero(aphi >= 0.5 * aphi.max()))
    p = np.abs(np.fft.fft(phi)) ** 2
    k = np.abs(np.fft.fftfreq(phi.size))                    # 0 .. 0.5 (Nyquist)
    s["nyq_frac"] = float(p[k > 0.25].sum() / p.sum())
    s["parity_phi"] = float(np.abs(phi.sum()) / aphi.sum())
    s["seg_max"] = (aphi.reshape(-1, nz).max(axis=1) / aphi.max()
                    if aphi.size % nz == 0 else np.array([]))
    if apar is not None and np.abs(apar).max() > 0:
        aa = np.abs(apar)
        s["end_apar"] = float(max(aa[0], aa[-1]) / aa.max())
        s["parity_apar"] = float(np.abs(apar.sum()) / aa.sum())
        s["apar_over_phi"] = float(aa.max())                # phi already / max|phi|
    else:
        s.update(end_apar=np.nan, parity_apar=np.nan, apar_over_phi=np.nan)
    return s


def settled(t, g, window=SETTLE_WINDOW):
    """(mean, std) of gamma(t) over the last *window* of the run time, or (nan, nan)."""
    if len(t) < 3:
        return np.nan, np.nan
    sel = t >= t[0] + (1 - window) * (t[-1] - t[0])
    if sel.sum() < 3:
        sel = slice(-3, None)
    return float(np.mean(g[sel])), float(np.std(g[sel]))


def verdict(r: dict, gamma_cut=GAMMA_CUT, nyq_cut=NYQ_CUT, min_fwhm=MIN_FWHM,
            settle_tol=SETTLE_TOL) -> str:
    if r.get("fd") is None or "phi" not in r["fd"]:
        return "no field"
    if any(np.isfinite(g) and abs(g) > gamma_cut for g in (r["gamma"], r["g_settle"])):
        return "blow-up"
    fd = r["fd"]
    if fd["nyq_frac"] > nyq_cut or fd["fwhm_pts"] < min_fwhm:
        return "grid-scale"
    if max(fd["end_phi"], np.nan_to_num(fd["end_apar"])) > EXTENT_TOL:
        return "extent"
    if r["status"] != "ok":
        m, s = r["g_settle"], r["g_settle_std"]
        if not np.isfinite(m) or s > max(settle_tol * abs(m), SETTLE_ABS):
            return "unsettled"
    return "damped" if r["gamma"] < 0 else "growing"


def read_case(root: str, x0: float, tag: str, ky_filter, max_pairs: int, cuts: dict) -> dict:
    """{ky: record} for every run of *tag* at *x0* under one variant root, plus numerics."""
    runs, numerics, physics = {}, {}, {}
    for path in gene_runs(os.path.join(root, x0_label((x0,)), tag)):
        try:
            rec = read_run(path)
        except Exception as exc:
            print(f"    unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        ky = round(rec["kymin"], 6)
        if ky_filter and not any(np.isclose(ky, k, rtol=1e-3) for k in ky_filter):
            continue
        r = {"path": path, "status": rec["status"], "gamma_gene": rec["gamma"],
             "omega_gene": rec["omega"]}
        try:
            r["fd"] = field_diagnostics(path, max_pairs)
        except Exception as exc:                       # file still being written, etc.
            print(f"    field unreadable {path}: {type(exc).__name__}: {exc}")
            r["fd"] = None
        fd = r["fd"] or {}
        r["g_settle"], r["g_settle_std"] = settled(fd.get("t", np.array([])),
                                                   fd.get("gamma_t", np.array([])))
        r["w_settle"], _ = settled(fd.get("t", np.array([])), fd.get("omega_t", np.array([])))
        if rec["status"] == "ok":
            r["gamma"], r["omega"], r["source"] = rec["gamma"], rec["omega"], "GENE"
        else:
            fit = live_fit(path) if r["fd"] else None
            if fit is not None:
                r["gamma"], r["omega"], r["source"] = fit[0], fit[2], f"fit {fit[4]}"
            else:
                r["gamma"], r["omega"], r["source"] = r["g_settle"], r["w_settle"], "trace"
        r["purely_growing"] = bool(np.isfinite(r["gamma"]) and r["gamma"] > 0
                                   and abs(r["omega"]) < 0.01 * abs(r["gamma"]))
        r["verdict"] = verdict(r, **cuts)
        runs[ky] = r
        if not numerics:
            pt = param_text(path)
            numerics = {k: pt.get(k) for k in NUMERICS}
            physics = {k: pt.get(k) for k in PHYSICS}
    return {"runs": runs, "numerics": numerics, "physics": physics}


# --------------------------------------------------------------------------- plotting

def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def differing(variants: dict) -> list[str]:
    """Numerics keys whose value is not the same in every variant."""
    keys = []
    for k in NUMERICS:
        vals = {fmt(v["numerics"].get(k)) for v in variants.values()}
        if len(vals) > 1:
            keys.append(k)
    return keys


def grid(nr, nc, w=2.2, h=1.5, left=2.2, top=1.2, bottom=0.8):
    W, H = w * nc + left + 0.3, h * nr + top + bottom
    fig, axes = plt.subplots(nr, nc, figsize=(W, H), squeeze=False)
    fig.subplots_adjust(left=left / W, right=1 - 0.3 / W, top=1 - top / H, bottom=bottom / H,
                        wspace=0.28, hspace=0.35)
    return fig, axes


def row_labels(axes, variants):
    for r, (name, v) in enumerate(variants.items()):
        n = v["numerics"]
        axes[r, 0].set_ylabel(f"{name}\nnz0 {fmt(n.get('nz0'))} nx0 {fmt(n.get('nx0'))}\n"
                              f"hyp_z {fmt(n.get('hyp_z'))} eo {fmt(n.get('edge_opt'))}",
                              rotation=0, ha="right", va="center", fontsize=7, color=INK,
                              labelpad=8)


def plot_summary(variants, ky, title, out, written):
    keys = differing(variants)
    nr, nc = len(variants), len(ky)
    W = 1.25 * nc + 1.6 + 1.1 * max(len(keys), 1)
    W = max(W, 9.0)
    H = 0.62 * nr + 2.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.4 / W, 1.2 / H, 1.25 * nc / W, 0.62 * nr / H])
    names = list(variants)
    for i, name in enumerate(names):
        for j, k in enumerate(ky):
            r = variants[name]["runs"].get(k)
            v = r["verdict"] if r else "no field"
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fc=VERDICTS[v][0], ec="white", lw=1.5))
            if r is None:
                ax.text(j + 0.5, i + 0.5, "no run", ha="center", va="center", fontsize=6,
                        color=MUTED)
                continue
            g, w = r["gamma"], r["omega"]
            extra = " PG" if r["purely_growing"] else ""
            src = r["source"] if r["source"] == "GENE" else (
                f"{r['source']}\ntrace γ {r['g_settle']:.3g}")
            ax.text(j + 0.5, i + 0.5, f"γ {g:.3g}\nω {w:.3g}{extra}\n{src}",
                    ha="center", va="center", fontsize=5.6, color=INK)
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_xticks(np.arange(nc) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    ax.set_yticks(np.arange(nr) + 0.5, names, fontsize=8)
    ax.set_xlabel(r"$k_y\rho_{ref}$", color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    # numerics that differ, one column each
    x = (1.4 + 1.25 * nc + 0.2) / W
    for c, key in enumerate(keys):
        fig.text(x + c * 1.1 / W, (1.2 + 0.62 * nr + 0.12) / H, key, fontsize=7, color=INK,
                 rotation=30, ha="left", va="bottom")
        for i, name in enumerate(names):
            fig.text(x + c * 1.1 / W, (1.2 + 0.62 * (nr - i - 0.5)) / H,
                     fmt(variants[name]["numerics"].get(key)), fontsize=7, color=INK,
                     va="center")
    phys = next(iter(variants.values()))["physics"]
    fig.suptitle(title + "\n" + "  ".join(f"{k} {fmt(v)}" for k, v in phys.items()
                                            if v is not None),
                 x=0.01, ha="left", fontsize=9, color=INK)
    handles = [Patch(fc=c, ec="none", label=f"{n}: {m}") for n, (c, m) in VERDICTS.items()]
    handles.append(Patch(fc="none", ec="none", label="PG = purely growing (|ω| < 1% γ)"))
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, fontsize=7)
    save(fig, out, "diag_summary.png", written)


def plot_traces(variants, ky, title, out, written, gamma_cut):
    fig, axes = grid(len(variants), len(ky))
    for i, (name, v) in enumerate(variants.items()):
        for j, k in enumerate(ky):
            ax = axes[i, j]
            r = v["runs"].get(k)
            if i == 0:
                ax.set_title(f"ky = {k:g}", fontsize=9, color=INK)
            fd = (r or {}).get("fd")
            if not fd or len(fd.get("t", [])) == 0:
                ax.text(0.5, 0.5, "no trace", ha="center", va="center", fontsize=7,
                        color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            t = fd["t"]
            ax.axvspan(t[0] + (1 - SETTLE_WINDOW) * (t[-1] - t[0]), t[-1], color="#f0efec",
                       lw=0)
            ax.plot(t, fd["gamma_t"], color="#2a78d6", lw=0.9)
            ax.plot(t, fd["omega_t"], color="#eb6834", lw=0.9, ls="--")
            if r["status"] == "ok":
                ax.axhline(r["gamma_gene"], color="#2a78d6", lw=0.6, ls=":")
            ax.axhline(0, color=MUTED, lw=0.5)
            ax.set_yscale("symlog", linthresh=0.1)
            vals = np.concatenate([fd["gamma_t"], fd["omega_t"]])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                lo, hi = min(vals.min(), 0.0), max(vals.max(), 0.0)
                ax.set_ylim(lo * 3 if lo < -0.1 else -0.3, hi * 3 if hi > 0.1 else 0.3)
            ax.tick_params(labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(VERDICTS[r["verdict"]][0])
                sp.set_linewidth(1.6)
            ax.text(0.02, 0.97, r["verdict"], transform=ax.transAxes, fontsize=6, va="top",
                    color=INK)
    row_labels(axes, variants)
    handles = [Line2D([], [], color="#2a78d6", label=r"$\gamma(t)$"),
               Line2D([], [], color="#eb6834", ls="--", label=r"$\omega(t)$"),
               Line2D([], [], color="#2a78d6", ls=":", label="GENE converged γ"),
               Patch(fc="#f0efec", label=f"last {SETTLE_WINDOW:.0%} (settle check)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(title + f"\nγ, ω from consecutive φ frames vs time [L_ref/c_ref], symlog y;"
                 f" border = verdict (blow-up cut {gamma_cut:g})",
                 x=0.01, ha="left", fontsize=9, color=INK)
    save(fig, out, "diag_traces.png", written)


def plot_structure(variants, ky, title, out, written):
    fig, axes = grid(len(variants), len(ky))
    for i, (name, v) in enumerate(variants.items()):
        for j, k in enumerate(ky):
            ax = axes[i, j]
            r = v["runs"].get(k)
            if i == 0:
                ax.set_title(f"ky = {k:g}", fontsize=9, color=INK)
            fd = (r or {}).get("fd")
            if not fd or "phi" not in fd:
                ax.text(0.5, 0.5, "no field", ha="center", va="center", fontsize=7,
                        color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            z = fd["zgrid"]
            for b in range(1, fd["nx"]):
                ax.axvline(z[b * fd["nz"]], color="#e4e3df", lw=0.5)
            ax.semilogy(z, np.maximum(np.abs(fd["phi"]), LOG_FLOOR),
                        color=STRUCT_COLOR["field_phi"], lw=0.8)
            if "apar" in fd:
                ax.semilogy(z, np.maximum(np.abs(fd["apar"]), LOG_FLOOR),
                            color=STRUCT_COLOR["field_apar"], lw=0.8)
            ax.axhline(EXTENT_TOL, color=MUTED, lw=0.5, ls=":")
            ax.set_ylim(LOG_FLOOR, 3 * max(1, np.nan_to_num(fd["apar_over_phi"])))
            ax.set_xlim(z[0], z[-1])
            ax.tick_params(labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(VERDICTS[r["verdict"]][0])
                sp.set_linewidth(1.6)
            ax.text(0.02, 0.97,
                    f"P(φ) {fd['parity_phi']:.2f} P(A) {fd['parity_apar']:.2f}\n"
                    f"A/φ {fd['apar_over_phi']:.2g} fwhm {fd['fwhm_pts']}",
                    transform=ax.transAxes, fontsize=5.5, va="top", color=INK)
    row_labels(axes, variants)
    handles = [Line2D([], [], color=STRUCT_COLOR["field_phi"], label=r"$|\phi|$ / max$|\phi|$"),
               Line2D([], [], color=STRUCT_COLOR["field_apar"],
                      label=r"$|A_\parallel|$ / max$|\phi|$"),
               Line2D([], [], color=MUTED, ls=":", label=f"EXTENT_TOL {EXTENT_TOL:g}"),
               Line2D([], [], color="#e4e3df", label="2π kx-connection boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(title + "\nlast-frame mode structure on a log axis vs ballooning angle z/π "
                 "(computational grid); P = parity |Σf|/Σ|f| (1 even, 0 odd)",
                 x=0.01, ha="left", fontsize=9, color=INK)
    save(fig, out, "diag_structure.png", written)


def plot_spectrum(variants, ky, title, out, written, nyq_cut):
    fig, axes = grid(len(variants), len(ky))
    for i, (name, v) in enumerate(variants.items()):
        for j, k in enumerate(ky):
            ax = axes[i, j]
            r = v["runs"].get(k)
            if i == 0:
                ax.set_title(f"ky = {k:g}", fontsize=9, color=INK)
            fd = (r or {}).get("fd")
            if not fd or "phi" not in fd:
                ax.text(0.5, 0.5, "no field", ha="center", va="center", fontsize=7,
                        color=MUTED, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            phi = fd["phi"]
            p = np.fft.fftshift(np.abs(np.fft.fft(phi)) ** 2)
            kk = np.fft.fftshift(np.fft.fftfreq(phi.size)) * 2       # -1 .. 1 = Nyquist
            ax.axvspan(-1, -0.5, color="#fbe3dc", lw=0)
            ax.axvspan(0.5, 1, color="#fbe3dc", lw=0)
            ax.semilogy(kk, np.maximum(p / p.max(), 1e-16), color=INK, lw=0.7, marker=".",
                        ms=1.5)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(1e-14, 2)
            ax.tick_params(labelsize=6)
            for sp in ax.spines.values():
                sp.set_color(VERDICTS[r["verdict"]][0])
                sp.set_linewidth(1.6)
            ax.text(0.02, 0.97, f"P(|k|>½) {fd['nyq_frac']:.2g}", transform=ax.transAxes,
                    fontsize=6, va="top", color=INK)
    row_labels(axes, variants)
    fig.suptitle(title + f"\nparallel power spectrum of φ along the ballooning grid "
                 f"(k / Nyquist); shaded = grid-scale, cut {nyq_cut:g}",
                 x=0.01, ha="left", fontsize=9, color=INK)
    save(fig, out, "diag_spectrum.png", written)


def write_csv(variants, out, written):
    path = data_path(out, "diagnostics.csv")
    cols = ("status", "gamma_gene", "omega_gene", "gamma", "omega", "source", "g_settle",
            "g_settle_std", "w_settle", "purely_growing", "verdict")
    fdcols = ("n_frames", "t_end", "end_phi", "end_apar", "fwhm_pts", "nyq_frac", "parity_phi",
              "parity_apar", "apar_over_phi")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "ky", *cols, *fdcols, *NUMERICS, *PHYSICS, "path"])
        for name, v in variants.items():
            for k, r in sorted(v["runs"].items()):
                fd = r["fd"] or {}
                w.writerow([name, k, *(r[c] for c in cols), *(fd.get(c) for c in fdcols),
                            *(v["numerics"].get(c) for c in NUMERICS),
                            *(v["physics"].get(c) for c in PHYSICS), r["path"]])
    written.append(path)


def expand_roots(roots):
    if not roots:
        return roots
    out = []
    for r in roots:
        out += sorted(d for d in glob.glob(r) if os.path.isdir(d)) or [r]
    return out


def run(shot, x0, tag=REF_TAG, roots=None, out=None, ky=None, max_pairs=MAX_PAIRS,
        gamma_cut=GAMMA_CUT, nyq_cut=NYQ_CUT, min_fwhm=MIN_FWHM, settle_tol=SETTLE_TOL):
    found = variant_roots(shot, expand_roots(roots))
    if not found:
        raise SystemExit(f"no variant roots for {shot} (corners_dbg_*, or --roots)")
    cuts = dict(gamma_cut=gamma_cut, nyq_cut=nyq_cut, min_fwhm=min_fwhm, settle_tol=settle_tol)
    variants = {}
    for name, root in found.items():
        print(f"--- {name}: {root}")
        variants[name] = read_case(root, x0, tag, ky, max_pairs, cuts)
        if not variants[name]["runs"]:
            print(f"    no runs under {os.path.join(root, x0_label((x0,)), tag)}")
    kys = sorted({k for v in variants.values() for k in v["runs"]})
    if not kys:
        raise SystemExit("no GENE runs in any variant yet")
    out = out or os.path.join(os.path.dirname(os.path.normpath(next(iter(found.values())))),
                              "dbg_compare", x0_label((x0,)), "diag")
    os.makedirs(out, exist_ok=True)
    title = f"NSTX {shot} x0 = {x0:g}, {tag}: {len(variants)} numerics variants"
    written: list[str] = []
    plot_summary(variants, kys, title, out, written)
    plot_traces(variants, kys, title, out, written, gamma_cut)
    plot_structure(variants, kys, title, out, written)
    plot_spectrum(variants, kys, title, out, written, nyq_cut)
    write_csv(variants, out, written)

    print("\nverdict counts per variant (best first):")
    order = ("growing", "damped", "unsettled", "extent", "grid-scale", "blow-up", "no field")
    def score(v):
        return tuple(-sum(r["verdict"] == o for r in v["runs"].values()) for o in order[:2])
    for name, v in sorted(variants.items(), key=lambda kv: score(kv[1])):
        counts = {o: sum(r["verdict"] == o for r in v["runs"].values()) for o in order}
        print(f"  {name:14s} " + "  ".join(f"{o} {n}" for o, n in counts.items() if n))
    print("numerics that differ: " + ", ".join(differing(variants)))
    for p in written:
        print(f"wrote {p}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shot", type=int, required=True)
    ap.add_argument("--x0", type=float, required=True)
    ap.add_argument("--tag", default=REF_TAG, help=f"case to compare [{REF_TAG}]")
    ap.add_argument("--roots", nargs="+", default=None,
                    help="variant diag roots, globs allowed [<lithium_study>/<shot>/corners_dbg_*]")
    ap.add_argument("--out", default=None,
                    help="[<lithium_study>/<shot>/dbg_compare/x0_<x0>/diag]")
    ap.add_argument("--ky", type=float, nargs="+", default=None, help="only these ky")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS,
                    help=f"frame pairs read per run for the traces [{MAX_PAIRS}]")
    ap.add_argument("--gamma-cut", type=float, default=GAMMA_CUT,
                    help=f"|gamma| above this is a blow-up [{GAMMA_CUT:g}]")
    ap.add_argument("--nyq-cut", type=float, default=NYQ_CUT,
                    help=f"phi power above half Nyquist that is grid-scale [{NYQ_CUT:g}]")
    ap.add_argument("--min-fwhm", type=int, default=MIN_FWHM,
                    help=f"fewest points across the |phi| peak [{MIN_FWHM}]")
    ap.add_argument("--settle-tol", type=float, default=SETTLE_TOL,
                    help=f"relative gamma(t) std that counts as settled [{SETTLE_TOL:g}]")
    a = ap.parse_args()
    run(a.shot, a.x0, a.tag, a.roots, a.out, a.ky, a.max_pairs, a.gamma_cut, a.nyq_cut,
        a.min_fwhm, a.settle_tol)
