"""Write the frozen campaign mtanh_full fits that scalings.py uses.

The same fit_mtanh_full call lands in different minima on different machines
and pfile copies (132543 Te: b_slope +0.04 locally, -0.95 on NERSC), and the
fit decides what a ped_scale factor actually does to the profile. So the
coefficients are fitted once, here, and read by scalings._fit everywhere.

    python -m NSTXU_lithium_study.scripts.freeze_fits \
        --cheasebs-tests C:/Users/joesc/git/cheaseBS_tests

Sources (vault note 2026-10-10-mtanh-fit-error-campaign-shots, 2026-10-16).
All four shots load Pingyu's GENE profiles_e/i/z from data/discharges (TPED
reads them over the pfile):

    129015, 129038  the coefficients the NERSC scans used (TPED_transform_log.json
                    of the jparallel box scan), so published 129xxx results
                    stay reproducible; those scans already ran on profiles_*
    132543          Te with b_slope >= 0: the free fit lands at b_slope -0.95,
                    which cancels the pedestal (30% of a ped_scale at the top,
                    core moving the other way); constrained, rms 0.98% and 95%
                    ne at default bounds (90%)
    132588          b_pos >= 0.70 for Te and ne. With the default 0.85 floor the
                    ne fit pins b_pos and cuts the axis by 13%, and Te misses
                    the shoulder (rms 2.8%); constrained, rms 0.17% / 0.05%
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np

from NSTXU_lithium_study.scripts import scalings
from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import (
    _auto_p0_full, _default_bounds_full, fit_mtanh_full)

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_FITS = os.path.join(HERE, "frozen_mtanh_fits.json")

# Box-scan runs whose transform logs hold the 129xxx coefficients.
NERSC_129XXX = "data/129015_129038/pm0.2_ncscal4_warmup0/{shot}/{var}_ped_scale_1.000"

# Bound overrides per shot and variable: parameter -> (lower, upper), None
# keeps the default. Everything not listed fits at default bounds.
PARAMS = ("b_height", "b_sol", "b_pos", "b_width", "b_slope", "a_height", "a_width", "a_exp")
REFITS = {
    132543: {"Te": {"b_slope": (0.0, None)}},
    132588: {"Te": {"b_pos": (0.70, None)}, "ne": {"b_pos": (0.70, None)}},
}


def _from_nersc_log(repo, shot, var):
    path = os.path.join(repo, NERSC_129XXX.format(shot=shot, var=var),
                        "TPED_transform_log.json")
    with open(path) as f:
        log = json.load(f)
    rec = [r for r in log["transform_history"] if r.get("var") == var][-1]
    if rec["scale_height"] != 1.0:
        raise ValueError(f"{path}: expected an unscaled case")
    return dict(params=rec["fit_params"], rms_relative=rec["rms_relative"],
                rms_residual=rec["rms_residual"],
                source="NERSC box scan, %s (%s)" % (
                    os.path.relpath(path, repo).replace(os.sep, "/"), log["generated"]))


def _local_fit(phys, var, bounds=None):
    ds = phys.ds
    p0 = None
    if bounds is not None:
        rhot = ds.coords["rhot"].values
        vals = np.asarray(ds[var].values, dtype=float)
        lo, hi = np.asarray(bounds[0]), np.asarray(bounds[1])
        p0 = list(np.clip(_auto_p0_full(rhot, vals / np.abs(vals).max()), lo + 1e-6, hi - 1e-6))
    prof, rec = fit_mtanh_full(ds, var, p0=p0, bounds=bounds, **scalings.FIT_KWARGS)
    return dict(params=rec["fit_params"], rms_relative=rec["rms_relative"],
                rms_residual=rec["rms_residual"])


def build(cheasebs_tests):
    out = {}
    for shot in (129015, 129038):
        out[str(shot)] = {v: _from_nersc_log(cheasebs_tests, shot, v) for v in ("Te", "ne")}
    for shot in (132543, 132588):
        phys = scalings.load(shot)
        out[str(shot)] = {}
        for v in ("Te", "ne"):
            over = REFITS.get(shot, {}).get(v)
            if not over:
                out[str(shot)][v] = dict(_local_fit(phys, v), source="local fit, default bounds")
                continue
            lo, hi = map(list, _default_bounds_full())
            for name, (l, h) in over.items():
                k = PARAMS.index(name)
                lo[k] = lo[k] if l is None else l
                hi[k] = hi[k] if h is None else h
            out[str(shot)][v] = dict(
                _local_fit(phys, v, (lo, hi)),
                source="local refit, " + ", ".join(
                    "%s in [%s, %s]" % (n, l, h) for n, (l, h) in over.items()),
                bounds=[lo, [h if np.isfinite(h) else None for h in hi]])
    for shot in out.values():
        for entry in shot.values():
            entry["fit_kwargs"] = dict(scalings.FIT_KWARGS)
    return {"written": date.today().isoformat(), "fits": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cheasebs-tests", required=True,
                    help="cheaseBS_tests checkout (for the 129xxx coefficients)")
    ap.add_argument("--out", default=FROZEN_FITS)
    args = ap.parse_args(argv)
    frozen = build(args.cheasebs_tests)
    with open(args.out, "w", newline="\n") as f:
        json.dump(frozen, f, indent=2)
        f.write("\n")
    for shot, by_var in frozen["fits"].items():
        for var, e in by_var.items():
            print("%s %s rms %.2f%%  %s" % (shot, var, 100 * e["rms_relative"], e["source"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
