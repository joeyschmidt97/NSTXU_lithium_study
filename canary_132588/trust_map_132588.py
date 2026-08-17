"""Where can CHEASE-BS reconstruction of 132588 be trusted, and where not.

Reads the canary run tree (runs/<timestamp>/) and plots the radial structure of
the reconstruction error, so "the edge is bad" becomes a number at every radius
instead of a feeling.

Four panels:
  (a) q(rho_tor) — source EFIT vs reconstructions, absolute values, for orientation.
  (b) |dq|/q vs rho_tor, log scale, with trust bands and the radii the existing
      132588 GENE linear scans sit at (q=4 at rho_tor 0.736, q=5 at 0.825).
  (c) |dp|/p vs rho_tor, same bands. Masked where source pressure is near zero,
      since a relative error against ~0 is meaningless, not large.
  (d) The fixed-point evidence: gating error metrics vs max_iter. Flat lines mean
      iterating does not help — the reconstruction lands where it lands.

rho_tor comes from GFileData's spline mapping for both files: the source EFIT is
129 points and cheaseBS writes 513, so index-wise comparison is wrong.

Usage (anywhere the TPED env is available):
    python trust_map_132588.py [--run runs/20260816_20-06-37] [--out trust_map.png]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData

HERE = os.path.dirname(os.path.abspath(__file__))

# Two series, two hues — validated for CVD separation and lightness. The three
# identity runs share one color on purpose: they lie on top of each other to
# within 0.05%, which is panel (d)'s point, and three near-identical curves in
# three shades would imply a distinction the data does not contain.
C_IDENTITY = "#0072B2"
C_MTANH = "#E69F00"
C_SOURCE = "#767676"
INK = "#1a1a1a"
MUTED = "#6b6b6b"

# Trust bands. Status colors, kept out of the series palette and always labeled.
BANDS = [
    (0.0, 0.01, "#e8f4ea", "trusted   < 1%"),
    (0.01, 0.05, "#fdf3e0", "wary   1-5%"),
    (0.05, 10.0, "#fbe9e7", "untrusted   > 5%"),
]

GENE_RADII = {0.736: "q=4", 0.825: "q=5"}
EQDSK_NAME = "EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT"
PRESSURE_FLOOR_FRAC = 1e-3   # of on-axis pressure


def read_gfile(path):
    ds = GFileData(path).gfile_to_xarray()
    rho = np.asarray(ds["rho_tor"].values)
    order = np.argsort(rho)
    return {"rho": rho[order],
            "q": np.asarray(ds["q"].values)[order],
            "p": np.asarray(ds["p"].values)[order]}


def rel_error_on(rho_ref, y_ref, rho_new, y_new, floor=None):
    """|y_new - y_ref| / |y_ref| with y_new interpolated onto the reference grid."""
    y_i = np.interp(rho_ref, rho_new, y_new)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.abs(y_i - y_ref) / np.abs(y_ref)
    if floor is not None:
        err = np.where(np.abs(y_ref) < floor, np.nan, err)
    return err


def draw_bands(ax):
    for lo, hi, color, label in BANDS:
        ax.axhspan(lo, hi, color=color, zorder=0)
        ax.text(0.012, np.sqrt(max(lo, 1e-4) * hi), label, fontsize=7.5,
                color=MUTED, va="center", ha="left", zorder=1)


def draw_gene_radii(ax, y_text=None):
    for r, label in GENE_RADII.items():
        ax.axvline(r, color=INK, lw=0.8, ls=(0, (4, 3)), alpha=0.55, zorder=2)
        if y_text is not None:
            ax.text(r, y_text, f" {label}", fontsize=7.5, color=INK,
                    rotation=90, va="bottom", ha="left", zorder=3)


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d8d8")
    ax.grid(True, lw=0.5, color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="canary run directory; defaults to the newest under runs/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = args.run
    if run_dir is None:
        candidates = sorted(glob.glob(os.path.join(HERE, "runs", "*")))
        if not candidates:
            raise SystemExit(f"no run directories under {os.path.join(HERE, 'runs')}")
        run_dir = candidates[-1]
    run_dir = os.path.abspath(run_dir)

    summary = json.load(open(os.path.join(run_dir, "canary_summary.json")))
    src = read_gfile(summary["source_gfile"])
    results = summary["results"]

    cases = []
    for name in sorted(results, key=lambda n: results[n].get("max_iter", 0)):
        path = os.path.join(run_dir, name, EQDSK_NAME)
        if not os.path.isfile(path):
            print(f"skipping {name}: no {EQDSK_NAME}")
            continue
        is_mtanh = "mtanh" in name
        cases.append({
            "name": name,
            "data": read_gfile(path),
            "color": C_MTANH if is_mtanh else C_IDENTITY,
            "max_iter": results[name].get("max_iter"),
            "is_mtanh": is_mtanh,
            "result": results[name],
        })
    if not cases:
        raise SystemExit(f"no usable cases in {run_dir}")

    p_floor = PRESSURE_FLOOR_FRAC * abs(src["p"][0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")

    # --- (a) absolute q, for orientation -----------------------------------
    ax = axes[0, 0]
    style(ax, r"$\rho_\mathrm{tor}$", "q", "(a)  Safety factor — source vs reconstruction")
    ax.plot(src["rho"], src["q"], color=C_SOURCE, lw=2.4, zorder=3)
    ax.text(src["rho"][-1], src["q"][-1], "  source EFIT", color=C_SOURCE,
            fontsize=8.5, va="center")
    for c in cases:
        ax.plot(c["data"]["rho"], c["data"]["q"], color=c["color"], lw=1.6,
                ls="--" if c["is_mtanh"] else "-", zorder=4)
    ax.text(0.55, np.interp(0.55, cases[0]["data"]["rho"], cases[0]["data"]["q"]),
            "identity", color=C_IDENTITY, fontsize=8.5, va="bottom")
    mt = next((c for c in cases if c["is_mtanh"]), None)
    if mt:
        ax.text(0.35, np.interp(0.35, mt["data"]["rho"], mt["data"]["q"]),
                "mtanh unity", color=C_MTANH, fontsize=8.5, va="top")
    draw_gene_radii(ax, y_text=float(np.nanmin(src["q"])))

    # --- (b) q error --------------------------------------------------------
    ax = axes[0, 1]
    style(ax, r"$\rho_\mathrm{tor}$", "|Δq| / q", "(b)  Where q can be trusted")
    draw_bands(ax)
    for c in cases:
        err = rel_error_on(src["rho"], src["q"], c["data"]["rho"], c["data"]["q"])
        ax.plot(src["rho"], err, color=c["color"], lw=1.6,
                ls="--" if c["is_mtanh"] else "-", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(0, 1.0)
    draw_gene_radii(ax, y_text=1.5e-4)

    # --- (c) pressure error -------------------------------------------------
    ax = axes[1, 0]
    style(ax, r"$\rho_\mathrm{tor}$", "|Δp| / p", "(c)  Where pressure can be trusted")
    draw_bands(ax)
    for c in cases:
        err = rel_error_on(src["rho"], src["p"], c["data"]["rho"], c["data"]["p"],
                           floor=p_floor)
        ax.plot(src["rho"], err, color=c["color"], lw=1.6,
                ls="--" if c["is_mtanh"] else "-", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10.0)
    ax.set_xlim(0, 1.0)
    draw_gene_radii(ax, y_text=1.5e-4)
    ax.text(0.02, 4.0, f"masked where source p < {PRESSURE_FLOOR_FRAC:g} × p(0)",
            fontsize=7.5, color=MUTED)

    # --- (d) does iterating help ------------------------------------------
    ax = axes[1, 1]
    style(ax, "max_iter", "relative error",
          "(d)  Iterating does not move it — a fixed point, not a transient")
    identity = [c for c in cases if not c["is_mtanh"]]
    iters = [c["max_iter"] for c in identity]

    def series(key_fn, color, label, lw=1.8, ls="-"):
        vals = [key_fn(c["result"]) for c in identity]
        vals = [np.nan if v is None else abs(v) for v in vals]
        ax.plot(iters, vals, color=color, lw=lw, ls=ls, marker="o", ms=6, zorder=4)
        ax.text(iters[-1], vals[-1], f"  {label}", color=color, fontsize=8.5,
                va="center", zorder=5)
        return vals

    # The two quantities the acceptance gate keys on get the series colors;
    # everything else is context and stays recessive.
    series(lambda r: r["d_ip_pct"] / 100.0, C_IDENTITY, "Ip error")
    series(lambda r: r["dq_psiN0.736_pct"] / 100.0, C_MTANH, "q @ 0.736 (q=4)")
    series(lambda r: r["dq_psiN0.825_pct"] / 100.0, C_MTANH, "q @ 0.825 (q=5)", ls="--")
    series(lambda r: r["dq95_pct"] / 100.0, C_SOURCE, "q95", lw=1.2)
    series(lambda r: r["dq_edge_pct"] / 100.0, C_SOURCE, "q at last surface", lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_xticks(iters)
    ax.set_xlim(min(iters) - 0.5, max(iters) + 6)
    ax.axhline(0.03, color=INK, lw=0.9, ls=":", zorder=3)
    ax.text(min(iters) - 0.4, 0.032, "gate: Ip tolerance 3%", fontsize=7.5, color=INK)
    ax.axhline(0.02, color=INK, lw=0.9, ls=":", zorder=3)
    ax.text(min(iters) - 0.4, 0.0135, "gate: analysis-radius q tolerance 2%",
            fontsize=7.5, color=INK)

    fig.suptitle("CHEASE-BS reconstruction of NSTX 132588 — radial trust map",
                 fontsize=13, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.945,
             f"run {os.path.basename(run_dir)}   ·   source {os.path.basename(summary['source_gfile'])}"
             f"   ·   dashed = mtanh-unity transform, solid = identity",
             fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = args.out or os.path.join(run_dir, "trust_map.png")
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")

    # The table view the figure is read against — same numbers, no color needed.
    print(f"\n{'case':22s} {'max_iter':>8s} {'|dIp|':>8s} {'q@0.736':>9s} "
          f"{'q@0.825':>9s} {'q95':>8s} {'q edge':>8s}")
    for c in cases:
        r = c["result"]
        print(f"{c['name']:22s} {str(c['max_iter']):>8s} "
              f"{abs(r['d_ip_pct']):7.3f}% {abs(r['dq_psiN0.736_pct']):8.3f}% "
              f"{abs(r['dq_psiN0.825_pct']):8.3f}% {abs(r['dq95_pct']):7.3f}% "
              f"{abs(r['dq_edge_pct']):7.3f}%")


if __name__ == "__main__":
    main()
