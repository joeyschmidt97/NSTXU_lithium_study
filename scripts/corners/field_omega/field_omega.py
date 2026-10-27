"""Field time-trace growth rates for every ky of the corner scan (TPED GENE_omega_from_field).

Every case dir under <diag-root>/x0_<x0>/ named like a ped_grid tag is fitted:
the 16 corners, the 1.00 reference and the VURS picks.

GENE writes gamma = omega = 0 into omega_000N when a run hits timelim before
its growth rate converges (and leaves the file empty if it died earlier).
For every ky, converged or not, this fits gamma and omega to the last part
of the field_000N time trace with TPED's estimator and, with --write,
appends the result as a

    # TPED OMEGA_FROM_FIELD // kymin= gamma= omega= gamma_std= omega_std= verdict= ...

line below GENE's own line, which is never touched; a line from an earlier
--write is replaced, so a rerun never stacks them. The omega file is then the
one record of the run: GENE's line says whether GENE converged (a value, 0/0,
or nothing), the TPED line holds the fit and its verdict:

    pass       both stds <= max(REL_TOL * |value|, ABS_TOL) (so a growth rate of
               0.03 needs std < 0.005)
    decaying   fails that, but gamma + 2 std < 0: no growing mode in the window
    ambiguous  fails, and a slow-growing mode cannot be told from beating damped
               modes (two modes competing at low ky is the usual reason it timed out)

TPED's GeneOmega keeps GENE's value as gamma/omega wherever GENE converged,
uses the fit only where GENE's line is 0/0 or missing and the verdict is
'pass', and always exposes gamma_field +- gamma_field_std, field_verdict and
source ('GENE', 'field', 'GENE_unconverged'). So plot_corners.py, the
dashboard and samplers (VURS) read the fit of any run, and plot_corners.py
--field-all draws every one of them.

    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega               # report only
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --write
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --x0 0.6737 --write
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --path .     # any dir
    python -m NSTXU_lithium_study.scripts.corners.field_omega.field_omega --shot 129015 --write

--path skips the corner layout and fits every omega_NNNN found at any depth
under the given directory (e.g. several scanfiles* dirs, or a continued run
next to a backed-up earlier segment).

Where GENE converged, the report also prints fit minus GENE: a check of the
estimator (window, sign of omega) where the answer is known, before trusting
it where it is not. --passing-only appends only fits that pass.
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
from NSTXU_lithium_study.scripts.corner_gene_scan import (DIAG_ROOT, SHOT, SHOTS, X0,
                                                         shot_defaults, x0_label)
from NSTXU_lithium_study.scripts.ped_grid import parse_tag

#: Fraction of the run's field output the fit uses: its last 30%.
WINDOW = (0.7, 1.0)
FIELDS = "field_phi"
TRACE_MODE = "log_weight"
REL_TOL = 0.1
ABS_TOL = 0.005


def omega_files(corner_dir: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(corner_dir, "scanfiles*", "omega_*"))
                  if re.search(r"omega_\d{4}$", f))


def corner_targets(diag_root: str, x0) -> tuple[str, list[tuple[str, str]]]:
    """(root, [(case tag, omega path)]) for every case dir at x0 named like a tag:
    the corners, the 1.00 reference and VURS picks."""
    root = os.path.join(diag_root, x0_label(x0))
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return root, []
    cases = []
    for name in names:
        try:
            parse_tag(name)
        except ValueError:                          # plots/, *_rounds/, anything else
            continue
        cases += [(name, o) for o in omega_files(os.path.join(root, name))]
    return root, cases


def path_targets(path: str) -> tuple[str, list[tuple[str, str]]]:
    """(path, [(omega path relative to path, omega path)]) for every omega_NNNN at any depth."""
    found = sorted(f for f in glob.glob(os.path.join(path, "**", "omega_*"), recursive=True)
                   if re.search(r"omega_\d{4}$", f))
    return path, [(os.path.relpath(o, path), o) for o in found]


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


def verdict(g, gs, w, ws, rel_tol=REL_TOL, abs_tol=ABS_TOL) -> str:
    """'pass' (both stds within tolerance), 'decaying' (fails, but gamma + 2 std < 0:
    no growing mode in the window) or 'ambiguous' (a slow-growing mode and beating
    damped modes cannot be told apart)."""
    if passes(g, gs, rel_tol, abs_tol) and passes(w, ws, rel_tol, abs_tol):
        return "pass"
    return "decaying" if np.isfinite(g) and np.isfinite(gs) and g + 2 * gs < 0 else "ambiguous"


def run(diag_root=DIAG_ROOT, x0=X0, write=False, window=WINDOW,
        rel_tol=REL_TOL, abs_tol=ABS_TOL, passing_only=False, path=None):
    root, targets = path_targets(path) if path else corner_targets(diag_root, x0)
    counts = {"written": 0, "GENE converged": 0, "pass": 0, "decaying": 0, "ambiguous": 0,
              "fit failed": 0, "no field": 0}
    print(f"{root}\nfit window {window[0]:g}-{window[1]:g} of the field output, {FIELDS}, "
          f"{TRACE_MODE}; pass: std <= max({rel_tol:g}*|value|, {abs_tol:g})"
          + ("" if write else "; REPORT ONLY (--write to append)") + "\n")
    print(f"{'run':34} {'ky':>5} {'GENE':>11} {'gamma_field':>18} {'omega_field':>18}  result")
    for label, omega_path in targets:
        state, values = gene_state(omega_path)
        field_path = re.sub(r"omega_(\d{4})$", r"field_\1", omega_path)
        ky = values.get("kymin", kymin_of(omega_path))
        gene = ("empty" if state == "empty" else
                f"{values['gamma_gene']:.3f}/{values['omega_gene']:.2f}")
        if not os.path.isfile(field_path) or os.path.getsize(field_path) == 0:
            counts["no field"] += 1
            print(f"{label:34} {ky:5.2f} {gene:>11} {'-':>18} {'-':>18}  no field file")
            continue
        try:
            est, stats = fit(field_path, window)
        except Exception as exc:
            counts["fit failed"] += 1
            print(f"{label:34} {ky:5.2f} {gene:>11}  fit failed: {type(exc).__name__}: {exc}")
            continue
        s = stats[FIELDS]
        g, w = s["gamma_avg"], s["omega_avg"]
        gs, ws = np.sqrt(s["gamma_variance"]), np.sqrt(s["omega_variance"])
        v = verdict(g, gs, w, ws, rel_tol, abs_tol)
        if state == "converged":                # GENE's value stays gamma; fit is a check
            counts["GENE converged"] += 1
            result = (f"GENE converged, {v}; fit-GENE: d_gamma {g - values['gamma_gene']:+.4f}, "
                      f"d_omega {w - values['omega_gene']:+.4f}")
        else:
            counts[v] += 1
            result = {"pass": "PASS, used as gamma",
                      "decaying": "DECAYING (gamma+2std<0), recorded only",
                      "ambiguous": "AMBIGUOUS (std too large), recorded only"}[v]
        if write and (v == "pass" or not passing_only):
            with contextlib.redirect_stdout(io.StringIO()):
                est._write_to_omega_file(stats, v)
            counts["written"] += 1
            result += ", appended"
        print(f"{label:34} {ky:5.2f} {gene:>11} {g:9.4f} ±{gs:7.1e} {w:9.4f} ±{ws:7.1e}  {result}")
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return counts


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=SHOT, choices=sorted(SHOTS),
                    help="discharge; sets the --diag-root and --x0 defaults")
    ap.add_argument("--diag-root", default=None,
                    help=f"corners are under <diag-root>/x0_<x0>/<tag>/ [{DIAG_ROOT}]")
    ap.add_argument("--x0", type=float, nargs="+", default=None,
                    help=f"which x0_<x0>/ scan [Te-top Chapman radius, {X0[0]:g} for {SHOT}]")
    ap.add_argument("--path", help="fit every omega_NNNN at any depth under this dir "
                                   "instead of the corner scan (--diag-root/--x0 ignored)")
    ap.add_argument("--write", action="store_true",
                    help="append every fit, with its verdict, to its omega file "
                         "(converged ky too; GENE's line is never changed)")
    ap.add_argument("--passing-only", action="store_true",
                    help="with --write, append only estimates that pass")
    ap.add_argument("--window", type=float, nargs=2, default=list(WINDOW),
                    help="fraction of the field output to fit, start end")
    ap.add_argument("--rel-tol", type=float, default=REL_TOL)
    ap.add_argument("--abs-tol", type=float, default=ABS_TOL)
    args = ap.parse_args()
    d = shot_defaults(args.shot)
    run(args.diag_root or d["diag_root"], tuple(args.x0 or d["x0"]), args.write, tuple(args.window),
        args.rel_tol, args.abs_tol, args.passing_only, args.path)
