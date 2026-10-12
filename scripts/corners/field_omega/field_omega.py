"""Field time-trace growth rates for the corner scan's unconverged ky (TPED GENE_omega_from_field).

GENE writes gamma = omega = 0 into omega_000N when a run hits timelim before
its growth rate converges (and leaves the file empty if it died earlier).
For each such ky this fits gamma and omega to the last part of the field_000N
time trace with TPED's estimator and, with --write, appends the result as a

    # TPED OMEGA_FROM_FIELD // kymin= gamma= omega= gamma_std= omega_std= ...

line below GENE's own line, which is never touched. TPED's GeneOmega (and so
plot_corners.py and the dashboard) then read the estimate, with its std,
wherever GENE's line is 0/0 or missing, and keep GENE's value everywhere else.

An estimate is written only when both stds pass

    std <= max(REL_TOL * |value|, ABS_TOL)

so a growth rate of 0.03 needs std below 0.005, not the estimator's absolute
default of 0.1. A run whose last stretch is not one clean exponential (two
modes competing at low ky is the usual reason it timed out) fails, and stays
unconverged: that point needs a longer or restarted run, not a fit.

    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega               # report only
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --write
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --x0 0.8447 --write
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --validate     # check on converged ky

--validate also fits every ky GENE did converge and prints both, writing
nothing for those: a check of the estimator (window, sign of omega) on runs
where the answer is known, before trusting it where it is not.
"""

from __future__ import annotations

import contextlib
import glob
import io
import os
import re

import numpy as np

from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField
from TPED.projects.GENE_sim_reader.src.GENE_omega_data import parse_omega_text
from NSTXU_lithium_study.scripts.corner_gene_scan import DIAG_ROOT, X0, corners, tag, x0_label

#: Fraction of the run's field output the fit uses: its last 30%.
WINDOW = (0.7, 1.0)
FIELDS = "field_phi"
TRACE_MODE = "log_weight"
REL_TOL = 0.1
ABS_TOL = 0.005


def omega_files(corner_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(corner_dir, "scanfiles*", "omega_*"))
                  if re.search(r"omega_\d{4}$", f))


def gene_state(omega_path: str) -> tuple[str, dict]:
    """('empty' | 'unconverged' | 'converged', parsed values) of GENE's line."""
    with open(omega_path) as fh:
        values = parse_omega_text(fh.read(), omega_path)
    if "gamma_gene" not in values:
        return "empty", values
    if values["gamma_gene"] == 0 and values["omega_gene"] == 0:
        return "unconverged", values
    return "converged", values


def kymin_of(omega_path: str) -> float:
    """kymin from the scan member's parameters_NNNN (an empty omega file has none)."""
    try:
        with open(re.sub(r"omega_(\d{4})$", r"parameters_\1", omega_path)) as fh:
            m = re.search(r"^\s*kymin\s*=\s*([-+.\deE]+)", fh.read(), re.M)
        return float(m.group(1)) if m else np.nan
    except OSError:
        return np.nan


def fit(field_path: str, window=WINDOW):
    """(GeneField, stats dict) of TPED's field time-trace estimate."""
    with contextlib.redirect_stdout(io.StringIO()):
        est = GeneField(field_path)
        stats = est._omega_from_field_calc(list(window), FIELDS, "percentage", TRACE_MODE)
    return est, stats


def passes(value: float, std: float, rel_tol=REL_TOL, abs_tol=ABS_TOL) -> bool:
    return bool(np.isfinite(value) and np.isfinite(std) and std <= max(rel_tol * abs(value), abs_tol))


def run(diag_root=DIAG_ROOT, x0=X0, write=False, validate=False, window=WINDOW,
        rel_tol=REL_TOL, abs_tol=ABS_TOL):
    root = os.path.join(diag_root, x0_label(x0))
    counts = {"written": 0, "passed": 0, "failed": 0, "no field": 0}
    print(f"{root}\nfit window {window[0]:g}-{window[1]:g} of the field output, {FIELDS}, "
          f"{TRACE_MODE}; pass: std <= max({rel_tol:g}*|value|, {abs_tol:g})"
          + ("" if write else "; REPORT ONLY (--write to append)") + "\n")
    print(f"{'corner':34} {'ky':>5} {'GENE':>11} {'gamma_field':>18} {'omega_field':>18}  result")
    for p in corners():
        corner_dir = os.path.join(root, tag(p))
        for omega_path in omega_files(corner_dir):
            state, values = gene_state(omega_path)
            if state == "converged" and not validate:
                continue
            field_path = re.sub(r"omega_(\d{4})$", r"field_\1", omega_path)
            ky = values.get("kymin", kymin_of(omega_path))
            gene = ("empty" if state == "empty" else
                    f"{values['gamma_gene']:.3f}/{values['omega_gene']:.2f}")
            if not os.path.isfile(field_path) or os.path.getsize(field_path) == 0:
                counts["no field"] += 1
                print(f"{tag(p):34} {ky:5.2f} {gene:>11} {'-':>18} {'-':>18}  no field file")
                continue
            try:
                est, stats = fit(field_path, window)
            except Exception as exc:
                counts["failed"] += 1
                print(f"{tag(p):34} {ky:5.2f} {gene:>11}  fit failed: {type(exc).__name__}: {exc}")
                continue
            s = stats[FIELDS]
            g, w = s["gamma_avg"], s["omega_avg"]
            gs, ws = np.sqrt(s["gamma_variance"]), np.sqrt(s["omega_variance"])
            ok = passes(g, gs, rel_tol, abs_tol) and passes(w, ws, rel_tol, abs_tol)
            if state == "converged":                # --validate: compare, never write
                result = (f"check: d_gamma {g - values['gamma_gene']:+.4f}, "
                          f"d_omega {w - values['omega_gene']:+.4f}")
            elif ok:
                counts["passed"] += 1
                if write:
                    with contextlib.redirect_stdout(io.StringIO()):
                        est._write_to_omega_file(stats)
                    counts["written"] += 1
                result = "PASS" + (", appended" if write else "")
            else:
                counts["failed"] += 1
                result = "FAIL (std too large), left unconverged"
            print(f"{tag(p):34} {ky:5.2f} {gene:>11} {g:9.4f} ±{gs:7.1e} {w:9.4f} ±{ws:7.1e}  {result}")
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--diag-root", default=DIAG_ROOT, help="corners are under <diag-root>/x0_<x0>/<tag>/")
    ap.add_argument("--x0", type=float, nargs="+", default=list(X0), help="which x0_<x0>/ scan")
    ap.add_argument("--write", action="store_true", help="append passing estimates to the omega files")
    ap.add_argument("--validate", action="store_true",
                    help="also fit ky GENE converged and print the difference (never written)")
    ap.add_argument("--window", type=float, nargs=2, default=list(WINDOW),
                    help="fraction of the field output to fit, start end")
    ap.add_argument("--rel-tol", type=float, default=REL_TOL)
    ap.add_argument("--abs-tol", type=float, default=ABS_TOL)
    args = ap.parse_args()
    run(args.diag_root, tuple(args.x0), args.write, args.validate, tuple(args.window),
        args.rel_tol, args.abs_tol)
