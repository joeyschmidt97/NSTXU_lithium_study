"""Figures for the cheaseBS box-edge analysis, from artifacts already on disk.

Every claim in [[2026-08-25-cheasebs-box-edge-convergence]] that is a *trend*
rather than a scalar gets a figure here, so it can be looked at rather than
taken on trust. Nothing needs NERSC and nothing needs a solve.

Sources, all local:
  profiles.csv          cheaseBS_tests/data/<shot>_runs/baseline/  (pressure split,
                        current decomposition, source-vs-CHEASE profiles)
  EQDSKs                ST_research/NSTXU_discharges/<shot>/       (source)
                        NSTXU_lithium_study/canary_132588/runs/    (reconstructed)
  iteration trace       reshape_convergence.ipynb stored cell output

Usage:
    python make_figures.py                     # writes to the vault assets dir
    python make_figures.py --outdir ./figures  # or anywhere
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent,
                        *pathlib.Path(__file__).resolve().parents]
            if (p / "pedestal_scan.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reshape_convergence"))

GIT = pathlib.Path("C:/Users/joesc/git")
VAULT_ASSETS = pathlib.Path(
    "C:/Users/joesc/vaults/master_vault/LLM_wikis/LLMW_Vulcan/raw/assets/generated")

CHEASEBS_TESTS = GIT / "cheaseBS_tests" / "data"
DISCHARGES = GIT / "ST_research" / "NSTXU_discharges"
CANARY = GIT / "NSTXU_lithium_study" / "canary_132588" / "runs" / "20260816_20-06-37"

MU0 = 4e-7 * np.pi
PED = 0.9          # pedestal boundary used for the "where the axes act" shading


def read_profiles_csv(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


# ----------------------------------------------------------------------
# 1. p_fast is a clamped residual
# ----------------------------------------------------------------------

def fig_pfast_clamp(shot, outdir):
    d = read_profiles_csv(CHEASEBS_TESTS / f"{shot}_runs" / "baseline" / "profiles.csv")
    rho, p_eq, p_th, p_f = d["rhot"], d["p_eq_pa"], d["p_th_pa"], d["p_fast_pa"]
    resid = p_eq - p_th
    clamped = resid < 0

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    ax[0].plot(rho, p_eq, "k-", lw=2, label=r"$p_\mathrm{eq}$  (EFIT total)")
    ax[0].plot(rho, p_th, "C0-", lw=1.6, label=r"$p_\mathrm{th}$  (kinetic sum)")
    ax[0].plot(rho, p_f, "C3-", lw=1.6, label=r"$p_\mathrm{fast}$  (as written)")
    ax[0].plot(rho, np.maximum(resid, 0), "C1--", lw=2.4,
               label=r"$\max(p_\mathrm{eq}-p_\mathrm{th},\,0)$")
    ax[0].set_xlabel(r"$\rho_\mathrm{tor}$"); ax[0].set_ylabel("pressure (Pa)")
    ax[0].set_title(f"{shot}: the two curves coincide exactly")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].axhline(0, c="k", lw=1)
    ax[1].plot(rho, resid, "C2-", lw=1.6, label=r"$p_\mathrm{eq}-p_\mathrm{th}$")
    ax[1].fill_between(rho, resid, 0, where=clamped, color="C3", alpha=.35,
                       label="clamped to zero")
    ax[1].axvspan(PED, rho.max(), color="grey", alpha=.15,
                  label=r"pedestal ($\rho_\mathrm{tor}>0.9$)")
    ax[1].set_xlabel(r"$\rho_\mathrm{tor}$")
    ax[1].set_ylabel(r"$p_\mathrm{eq}-p_\mathrm{th}$  (Pa)")
    frac_all = clamped.mean()
    frac_ped = clamped[rho > PED].mean()
    ax[1].set_title(f"clamped: {frac_all:.1%} of all radii, "
                    f"{frac_ped:.1%} above {PED}")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    share = np.divide(p_f, p_th + p_f, out=np.zeros_like(p_f),
                      where=(p_th + p_f) > 0)
    ax[2].plot(rho, 100 * share, "C3-", lw=1.8)
    ax[2].axvspan(PED, rho.max(), color="grey", alpha=.15)
    ax[2].set_xlabel(r"$\rho_\mathrm{tor}$")
    ax[2].set_ylabel(r"$p_\mathrm{fast}/p_\mathrm{tot}$  (%)")
    ax[2].set_title("oscillatory, non-monotonic, zero in the pedestal")
    ax[2].grid(alpha=.3)

    fig.suptitle(f"{shot} baseline — $p_\\mathrm{{fast}}$ is a clamped residual, "
                 "not a fast-ion model", y=1.02, fontsize=12)
    fig.tight_layout()
    out = outdir / f"cheasebs-pfast-clamp-{shot}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    exact = np.allclose(p_f, np.maximum(resid, 0), rtol=1e-9, atol=1e-9)
    print(f"  {out.name}   p_fast == max(resid,0): {exact}   "
          f"clamped {frac_all:.1%} / {frac_ped:.1%} in pedestal")
    return out


# ----------------------------------------------------------------------
# 2. Current decomposition
# ----------------------------------------------------------------------

def fig_current_decomposition(shot, outdir):
    d = read_profiles_csv(CHEASEBS_TESTS / f"{shot}_runs" / "baseline" / "profiles.csv")
    rho = d["rhot"]
    jt = d["total_parallel_current_density"]
    jbs = d["bootstrap_parallel_current_density"]
    jdr = d["driven_parallel_current_density"]
    dV = d["dVdpsi_m3_per_wb"]
    w = np.gradient(d["psi_abs_wb"]) * dV

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(rho, jt / 1e6, "k-", lw=2, label=r"$j_\parallel$ total")
    ax[0].plot(rho, jdr / 1e6, "C0-", lw=1.6, label=r"$j_\parallel$ driven")
    ax[0].plot(rho, jbs / 1e6, "C3-", lw=1.6, label=r"$j_\parallel$ bootstrap")
    ax[0].axvspan(PED, rho.max(), color="grey", alpha=.15, label="pedestal")
    ax[0].set_xlabel(r"$\rho_\mathrm{tor}$")
    ax[0].set_ylabel(r"$j_\parallel$  (MA/m$^2$)")
    ax[0].set_title(f"{shot}: driven current dominates")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    cum = np.cumsum(jbs * w) / max(np.cumsum(jt * w)[-1], 1e-30)
    frac = np.divide(jbs, jt, out=np.zeros_like(jbs), where=np.abs(jt) > 1e-12)
    ax[1].plot(rho, 100 * frac, "C3-", lw=1.8, label="local $j_{BS}/j_{tot}$")
    ax[1].plot(rho, 100 * cum, "C1--", lw=1.8, label="cumulative $I_{BS}/I_p$")
    ax[1].axvspan(PED, rho.max(), color="grey", alpha=.15)
    fbs = float(np.sum(jbs * w) / np.sum(jt * w))
    ax[1].axhline(100 * fbs, ls=":", c="k", lw=1)
    ax[1].set_xlabel(r"$\rho_\mathrm{tor}$")
    ax[1].set_ylabel("bootstrap share (%)")
    ax[1].set_title(rf"volume-averaged $f_{{BS}} = {fbs:.3f}$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    fig.suptitle(f"{shot} baseline — current decomposition: the channel the "
                 "iteration loop actually varies", y=1.02, fontsize=12)
    fig.tight_layout()
    out = outdir / f"cheasebs-current-decomposition-{shot}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}   f_BS = {fbs:.4f}")
    return out


# ----------------------------------------------------------------------
# 3. Iteration trace
# ----------------------------------------------------------------------

def fig_iteration_trace(outdir):
    import assess_helpers as ah

    tr = ah.iteration_trace(ROOT / "reshape_convergence" / "reshape_convergence.ipynb")
    if tr.empty:
        print("  (no trace in the run notebook -- skipping)")
        return None

    runs = ah.load_runs()
    tags = set(tr.run.dropna())
    tol = next((r["tol_q"] for r in runs if r["tag"] in tags), None)
    target = next((x.get("target_ip_a") for r in runs if r["tag"] in tags
                   for x in r["rows"] if x.get("target_ip_a")), None)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for (axis, scale), g in tr.groupby(["axis", "scale"]):
        lab = f"{axis.split('_')[0]} {scale:.2f}"
        ax[0].plot(g["iter"], g.ip_a / 1e3, "o-", label=lab)
        ax[1].plot(g["iter"], g.q_change, "o-", label=lab)
    if target:
        ax[0].axhline(target / 1e3, ls="--", c="k", lw=1.2, label="target $I_p$")
    ax[0].set_xlabel("iteration"); ax[0].set_ylabel("$I_p$ (kA)")
    ax[0].set_title("the whole response is set at iteration 00")
    ax[0].set_xticks(sorted(tr["iter"].unique()))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    if tol:
        ax[1].axhline(tol, ls="--", c="k", lw=1.2, label=rf"$\mathrm{{tol}}_q={tol:g}$")
        ax[1].axhline(1e-4, ls=":", c="C3", lw=1.2, label=r"$10^{-4}$ (earlier runs)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("`q_change`")
    ax[1].set_title("iteration 00 has no `q_change`: 2 is the floor")
    ax[1].set_xticks(sorted(tr["iter"].unique()))
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

    fig.suptitle("132543 box edges — per-iteration trace, the only record of "
                 "what stops the loop", y=1.02, fontsize=12)
    fig.tight_layout()
    out = outdir / "cheasebs-iteration-trace-132543.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")
    return out


# ----------------------------------------------------------------------
# 4. Ip self-consistency
# ----------------------------------------------------------------------

def _ip_integral(path):
    from matplotlib.path import Path as MplPath
    from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData

    ds = GFileData(str(path)).gfile_to_xarray()
    R, Z = ds["R"].values, ds["Z"].values
    psi2d = ds["psi_RZ"].values
    psi1d = ds.coords["psi"].values
    RR, ZZ = np.meshgrid(R, Z)
    poly = MplPath(np.column_stack([ds["RBDRY"].values, ds["ZBDRY"].values]))
    mask = poly.contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
    o = np.argsort(psi1d)
    pp = np.interp(psi2d, psi1d[o], ds["pprime"].values[o])
    ffp = np.interp(psi2d, psi1d[o], ds["ffprime"].values[o])
    j = RR * pp + ffp / (MU0 * RR)
    return abs(np.sum(j[mask]) * (R[1] - R[0]) * (Z[1] - Z[0]))


def _header_current(path):
    with open(path) as f:
        lines = [next(f) for _ in range(4)]
    return abs(float(lines[3][0:16]))


def fig_ip_consistency(outdir):
    shots, hdr, integ = [], [], []
    for s in (129015, 129038, 132543, 132588):
        g = sorted(glob.glob(str(DISCHARGES / str(s) / "g*")))[0]
        shots.append(s); hdr.append(_header_current(g)); integ.append(_ip_integral(g))
    hdr, integ = np.array(hdr), np.array(integ)

    can_lbl, can_ip = [], []
    tgt = hdr[shots.index(132588)]
    for d in sorted(os.listdir(CANARY)):
        p = CANARY / d / "EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT"
        if p.is_file():
            can_lbl.append(d.replace("identity_", "id ").replace("mtanh_unity_", "mtanh "))
            can_ip.append(_ip_integral(p))
    can_ip = np.array(can_ip)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(shots))
    ax[0].bar(x, 100 * (integ - hdr) / hdr, color="C0")
    ax[0].axhline(0, c="k", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(shots)
    ax[0].set_ylabel(r"$\int j_\phi dA$ vs header $I_p$  (%)")
    ax[0].set_ylim(-2.0, 0.2)
    ax[0].set_title("source EFITs reproduce their own current\n(<0.03%)")
    ax[0].grid(alpha=.3, axis="y")

    xc = np.arange(len(can_lbl))
    ax[1].bar(xc, 100 * (can_ip - tgt) / tgt, color="C3")
    ax[1].axhline(0, c="k", lw=1)
    ax[1].axhline(-1.5, ls="--", c="k", lw=1.2, label="reported $-1.5\\%$")
    ax[1].set_xticks(xc); ax[1].set_xticklabels(can_lbl, fontsize=8, rotation=15)
    ax[1].set_ylabel(r"$\int j_\phi dA$ vs target $I_p$  (%)")
    ax[1].set_ylim(-2.0, 0.2)
    ax[1].set_title("132588 reconstructions genuinely miss it\n(not a measurement artifact)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")

    fig.suptitle(r"$I_p$ self-consistency: $I_p=\int (Rp' + FF'/\mu_0R)\,dA$ "
                 "over the stored boundary", y=1.03, fontsize=12)
    fig.tight_layout()
    out = outdir / "cheasebs-ip-consistency.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    for s, h, i in zip(shots, hdr, integ):
        print(f"  {s}: header {h:,.0f}  integral {i:,.0f}  ({(i-h)/h:+.3%})")
    for l, i in zip(can_lbl, can_ip):
        print(f"  canary {l:<16} {i:,.0f}  ({(i-tgt)/tgt:+.2%})")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(VAULT_ASSETS))
    ap.add_argument("--shot", type=int, default=129015,
                    help="which baseline profiles.csv to use (needs cheaseBS_tests data)")
    a = ap.parse_args()
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[figures] -> {out}")

    made = []
    for fn in (lambda: fig_pfast_clamp(a.shot, out),
               lambda: fig_current_decomposition(a.shot, out),
               lambda: fig_iteration_trace(out),
               lambda: fig_ip_consistency(out)):
        try:
            r = fn()
            if r:
                made.append(r)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    print(f"[figures] {len(made)} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
