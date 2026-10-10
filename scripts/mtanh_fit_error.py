"""Where the campaign mtanh_full fit misses the raw profiles.

Refits Te and ne for every shot with the campaign setting (scalings.FIT_KWARGS)
and writes, to --outdir:

    mtanh_fit_error.png         fit - raw against rho_tor, every shot and variable
    mtanh_fit_params.csv        the eight Stefanikova coefficients per fit, plus rms
    mtanh_fit_error_regions.csv mean and worst error per radial region

    python -m NSTXU_lithium_study.scripts.mtanh_fit_error --outdir <dir>

Only Te and ne are drawn because they are the only variables the campaign fits:
apply_mtanh_full replaces Te or ne, and with quasineutrality on ni and nz are
multiplied by the ne fit/raw ratio, so ni carries exactly the ne error shown
here (and keeps any structure of its own, like the 132588 kink). Ti is never
fitted.

Two normalisations, because each hides what the other shows. Over the profile
maximum is what curve_fit minimises (it fits vals / max), so it is the honest
picture of fit quality and stays finite in the SOL. Over the local value is
what CHEASE actually receives: a 15% core Te miss is small against the axis
value of the max-normalised panel but is still 15% of the pressure there.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from NSTXU_lithium_study.scripts.scalings import FIT_KWARGS, SHOTS, _fit, load
from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import _default_bounds_full

VARS = ("Te", "ne")

# Fit-weight regions of fit_mtanh_full (_build_sigma defaults, with the
# campaign pedestal_weight): core weight 1, pedestal weight FIT_KWARGS, edge 1.
PED_THRESHOLD = 0.85
EDGE_THRESHOLD = 0.99
REGIONS = {
    "core": (0.0, 0.6),
    "shoulder": (0.6, PED_THRESHOLD),
    "pedestal": (PED_THRESHOLD, EDGE_THRESHOLD),
    "edge": (EDGE_THRESHOLD, np.inf),
}

# Local-value panel is cut here: past the separatrix the raw value goes to
# the SOL floor and the ratio stops meaning anything.
LOCAL_RHOT_MAX = 0.99

COLORS = {129015: "tab:blue", 129038: "tab:orange",
          132543: "tab:green", 132588: "tab:red"}
STYLES = {"Te": "-", "ne": "--"}


def _raw(phys, var):
    da = phys.ds[var]
    return phys.ds.coords["rhot"].values, np.asarray(getattr(da, "values", da),
                                                     dtype=float)


def collect(shots=SHOTS):
    """[(shot, var, rhot, raw, fit_vals, profile)] for every shot and variable."""
    rows = []
    for shot in shots:
        print("fitting %d ..." % shot, flush=True)
        phys = load(shot)
        for var in VARS:
            rhot, raw = _raw(phys, var)
            prof = _fit(phys, var)
            rows.append((shot, var, rhot, raw, prof(rhot), prof))
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for ax in (a, b):
        ax.axvspan(PED_THRESHOLD, EDGE_THRESHOLD, color="0.9", zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.3)
    for shot, var, rhot, raw, fit, _ in rows:
        kw = dict(color=COLORS.get(shot), ls=STYLES[var], lw=1.4,
                  label="%d %s" % (shot, var))
        a.plot(rhot, 100 * (fit - raw) / np.max(np.abs(raw)), **kw)
        m = rhot <= LOCAL_RHOT_MAX
        b.plot(rhot[m], 100 * (fit[m] - raw[m]) / raw[m], **kw)
    a.set_ylabel("(fit - raw) / max(raw)  [%]")
    a.set_title("mtanh_full fit error, pedestal_weight=%g (shaded: weighted "
                "region %.2f-%.2f)" % (FIT_KWARGS["pedestal_weight"],
                                       PED_THRESHOLD, EDGE_THRESHOLD),
                fontsize=10)
    b.set_ylabel("(fit - raw) / raw  [%]")
    b.set_xlabel(r"$\rho_\mathrm{tor}$")
    b.set_xlim(0, 1.05)
    a.legend(ncol=4, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


FIELDS = ("b_height", "b_sol", "b_pos", "b_width", "b_slope",
          "a_height", "a_width", "a_exp")


def at_bounds(prof, rtol=1e-3):
    """Coefficients pinned to a fit_mtanh_full default bound.

    A pinned coefficient means the optimum lies outside the box, so the fit
    there is the box talking, not the data. The amplitude bounds are on
    vals / max(vals); only their zero lower bound is meaningful in physical
    units, so b_sol and a_height are checked against 0 relative to b_height.
    """
    lo, hi = _default_bounds_full()
    p = prof.as_dict()
    pinned = []
    for i, k in enumerate(FIELDS):
        if k in ("b_height", "b_sol", "a_height"):
            if p[k] <= rtol * p["b_height"]:
                pinned.append("%s=0" % k)
            continue
        span = hi[i] - lo[i]
        for b in (lo[i], hi[i]):
            if abs(p[k] - b) <= rtol * span:
                pinned.append("%s=%g" % (k, b))
    return pinned


def write_params(rows, path):
    fields = FIELDS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("shot", "var", "unit") + fields
                   + ("rms_residual", "rms_relative", "raw_axis", "fit_axis",
                      "at_bound"))
        for shot, var, rhot, raw, fit, prof in rows:
            p = prof.as_dict()
            w.writerow([shot, var, prof.unit] + ["%.6g" % p[k] for k in fields]
                       + ["%.6g" % prof.rms_residual, "%.6g" % prof.rms_relative,
                          "%.6g" % raw[0], "%.6g" % fit[0],
                          " ".join(at_bounds(prof))])
    return path


def write_regions(rows, path):
    """Mean and worst error per region, in % of max(raw) and % of local raw."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("shot", "var", "region", "rhot_lo", "rhot_hi",
                    "mean_pct_of_max", "worst_pct_of_max", "worst_at_rhot",
                    "mean_pct_local", "worst_pct_local"))
        for shot, var, rhot, raw, fit, _ in rows:
            err_max = 100 * (fit - raw) / np.max(np.abs(raw))
            err_loc = 100 * (fit - raw) / raw
            for name, (lo, hi) in REGIONS.items():
                m = (rhot >= lo) & (rhot < hi)
                if not m.any():
                    continue
                i = np.argmax(np.abs(err_max[m]))
                ml = m & (rhot <= LOCAL_RHOT_MAX)
                loc = err_loc[ml] if ml.any() else np.array([np.nan])
                w.writerow([shot, var, name, lo, min(hi, float(rhot.max())),
                            "%.2f" % err_max[m].mean(), "%.2f" % err_max[m][i],
                            "%.3f" % rhot[m][i], "%.2f" % np.nanmean(loc),
                            "%.2f" % loc[np.nanargmax(np.abs(loc))]
                            if np.isfinite(loc).any() else "nan"])
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--shots", type=int, nargs="+", default=list(SHOTS))
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    rows = collect(args.shots)
    for p in (plot(rows, os.path.join(args.outdir, "mtanh_fit_error.png")),
              write_params(rows, os.path.join(args.outdir, "mtanh_fit_params.csv")),
              write_regions(rows, os.path.join(args.outdir,
                                               "mtanh_fit_error_regions.csv"))):
        print("wrote %s" % p, flush=True)


if __name__ == "__main__":
    main()
