#!/usr/bin/env python3
"""Two ways of scaling the same DIII-D profiles, both put through cheaseBS.

The question this exists to answer: is the omn/omt gradient scaling more stable
through cheaseBS than the full-mtanh pedestal scaling? Everything else is held
fixed -- same base EQDSK, same base profiles, same reference set, same baseline
decomposition, same solver settings, same point grid, same acceptance gate -- so
the only thing that differs between the two campaigns is how the scaled profiles
were produced.

Why the question is open. Replaying profiles the older IFS scaling code wrote
worked: the driven amplitude moved and q and the shear responded, on this same
EQDSK (run_cheasebs_scaling_scan.py, which is kept as the record of that run and
supplies the shared helpers here). The reshape campaign, which builds its
profiles from an mtanh fit, did not behave. That leaves the profile construction
as the suspect and these two methods as the two candidates.

METHOD 1 -- `omn_omt`, gradient scaling in the discharge object

    phys.apply_omt(alpha=omt, ...)      # Te and Ti, and Tz with them
        .apply_omne(alpha=omn, ...)     # ne, with ni and nz held quasineutral

TPED's pedestal gradient transforms: the profile is re-exponentiated about its
mid-pedestal value, `T_new = T_mid * (T/T_mid)**alpha`, with alpha ramped from 1
in the core to its full value across [rhot_topped, rhot_midped]. No fit is
involved in the scaling -- the profile's own shape is the thing being rescaled.
The window itself is per discharge and tabulated in RAMP_WINDOWS. For 162940 it
is not a measured pedestal at all: it is the pair the handed-over IFS scan was
generated with (rhotTopPed/rhotMidPed = 0.8/1.0), so this campaign scales the
same way the set it is being compared against did.

METHOD 2 -- `mtanh_full`, the reshape campaign's machinery

    phys.apply_mtanh_full('Te', fit=..., scale_height=omt)   # and Ti, Tz
    phys.apply_mtanh_full('ne', fit=..., scale_height=omn)

A Stefanikova F_full profile is fitted to each variable and the fit's pedestal
parameter is scaled, then the profile is rebuilt from the modified fit. This is
the same transform and the same `pedestal_weight=8.0` fit setting the NSTX
pedestal_scan campaign uses for its Te_ped_scale / ne_ped_scale axes; only the
discharge is different. `--mtanh-knob` picks which fit parameter the scale
multiplies, defaulting to `scale_height` to match that campaign.

The two are NOT the same operation and are not meant to be: one scales a
gradient, the other a fitted pedestal height. A point tagged omt0p8_omne0p9
means "0.8 on the temperature knob, 0.9 on the density knob" in each method's
own terms. What is comparable is the solver's response -- iterations, Ip error,
q error, acceptance -- across the same grid.

Two asymmetries worth knowing before reading the results:

  * The fit is a failure mode the gradient scaling does not have. Every fit's
    relative rms is recorded per point and printed, so a method-2 point that
    misbehaves can be checked against how well its own fit described the profile
    in the first place. Fits are computed once on the base profiles and reused
    at every point, as pedestal_scan does.
  * mtanh_full on a temperature is fitted per variable, so Tz is fitted and
    scaled in its own right rather than being carried along with Ti the way
    apply_omt carries it.

WHERE THINGS GO

`--case-dir` is the discharge directory: base gfile, pfile and profiles_{e,i,z}
under the names TPED expects, handed to DischargeData as-is. Results default to
$SCRATCH, one subdirectory per method:

    <outroot>/omn_omt/<tag>/     EQDSK, records, end plots, profiles
    <outroot>/mtanh_full/<tag>/
    <outroot>/comparison.json    every row from both methods
    <outroot>/table.txt          both methods in one table, method first

cheaseBS itself runs on a scratch working tree and only the result and the
records are copied into those directories. Both methods share one baseline
decomposition: it is a function of the EQDSK and the untransformed reference
profiles, and neither depends on the method, so building it twice would only
buy two chances to build it differently.

The solve calls TPED's `run_cheasebs_workflow` -- the function `output_gfile`
itself calls -- rather than `output_gfile`, which hardcodes `chease_namelist_nstx`
as an explicit argument so a DIII-D namelist cannot be passed through its
**cheasebs_overrides without colliding with it.

USAGE

    # one point, both methods: the unity point, which should reproduce the source
    python -u run_cheasebs_selfscaled_scan.py \
        --case-dir $SCRATCH/DIIID162940/DIIID162940 --only omt1p0_omne1p0

    # the full comparison, detached
    nohup python -u run_cheasebs_selfscaled_scan.py \
        --case-dir $SCRATCH/DIIID162940/DIIID162940 > /dev/null 2>&1 &
    tail -f $SCRATCH/cheasebs_scaling_comparison/*/campaign_*.log

    # one method only
    python -u run_cheasebs_selfscaled_scan.py --case-dir ... --method mtanh_full

    # what each method does to the profiles, solving nothing
    python run_cheasebs_selfscaled_scan.py --case-dir ... --dry-run

`-u` matters: without it Python block-buffers stdout when it is not a terminal
and the log stays empty for hours.

Exit status is 0 only when every solve completed AND every point was accepted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback

# cheaseBS renders per-run PNGs and must not reach for a display.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cheasebs_scan_common import (  # noqa: E402
    Tee, copy_back_if_needed, fmt_alpha, render_plots, shot_of, text_table)

METHODS = ("omn_omt", "mtanh_full")

# The points the IFS scan directory holds for 162940, so this campaign lands
# tag-for-tag beside it. (omt, omn).
DEFAULT_PAIRS = (
    # The diagonal, which is the axis the comparison is really about: how far
    # each method can be pushed before the solver stops behaving. 1.0 is the
    # null test and the first thing to read -- mtanh_full does not reproduce the
    # source there, because it replaces the profile with its own fit even at
    # unity, while omn_omt does.
    (0.7, 0.7), (0.8, 0.8), (0.9, 0.9), (1.0, 1.0), (1.1, 1.1),
    # One opposed corner and one of each single-axis move, so a method that
    # only misbehaves when the two knobs disagree is not invisible.
    (0.8, 1.1), (1.1, 1.0), (1.0, 1.1),
)

# (rhot_topped, rhot_midped) per shot: the exponent ramp
#
#     alpha_profile = 1 + (alpha-1) * (tanh((rhot-topped)/(midped-topped)) + 1)/2
#
# with the power law pivoted at the value at rhot_midped. midped >= 1.0 pivots on
# the separatrix and holds it fixed; topped is where the ramp reaches half its
# travel, so the transform is the identity well inboard of it.
#
# 162940 is NOT a measured pedestal. It is the pair the handed-over IFS
# `modProfs` scan was generated with, taken from that scan's own fit record --
# the point of this campaign is to scale the way the set it is compared against
# was scaled. topped=0.8 leaves the core untouched (the ramp is 3e-4 of the way
# to alpha at the axis) and puts the whole change in rhot > 0.6.
#
# 129015 is a measured mtanh_full pe window (fitted 2026-09-07 on the bundled
# base profiles; full pe width 0.081, fit rms 0.29%), i.e. a different kind of
# number in the same table -- see WINDOW_PROVENANCE, which is what gets logged.
#
# The measured 162940 pe window, 0.954/0.971, is kept here for the record and
# deliberately not used: it is 0.017 wide, which makes alpha_profile exactly 1
# everywhere inboard of 0.94, so every alpha returned the same profile and the
# first campaign's omn/omt points could not move the equilibrium.
RAMP_WINDOWS = {
    "162940": (0.8, 1.0),          # DIII-D; IFS handoff ramp, separatrix pivot
    "129015": (0.885, 0.926),      # NSTX;   measured mtanh_full pe window
}

WINDOW_PROVENANCE = {
    "162940": "IFS modProfs handoff record (rhotTopPed/rhotMidPed)",
    "129015": "measured mtanh_full pe window",
}

MEASURED_PE_WINDOWS = {            # for the record; not used for scaling
    "162940": (0.954, 0.971),      # full pe width 0.033, fit rms 0.19%
}

# For a shot with no measured window. Deliberately the old NSTX value rather
# than something derived: an unmeasured discharge should get the number whose
# provenance is known, and a loud line in the log saying so.
FALLBACK_TOPPED = 0.90
FALLBACK_MIDPED = 0.95

# pedestal_scan.FIT_KWARGS, unchanged: the pedestal is upweighted eightfold so
# the core cannot dominate a least-squares fit whose pedestal is the point.
# Copied rather than imported because pedestal_scan's module scope loads the
# NSTX discharge table.
FIT_KWARGS = dict(pedestal_weight=8.0)

# Temperatures take the omt knob, densities the omn knob. Tz is listed for the
# fit path because apply_mtanh_full fits each variable in its own right; the
# gradient path gets Tz through apply_omt's apply_to_tz, passed explicitly below
# to mirror the handoff's `tz_eq_ti: true` rather than leaning on the default.
TZ_EQ_TI = True
TEMPERATURE_VARS = ("Te", "Ti", "Tz")
DENSITY_VAR = "ne"

# DIII-D validated solver path: j_parallel replay on rhot with QSPEC enforced.
DEFAULT_CONFIG = os.path.join(HERE, "diiid_cheasebs_config.json")

DEFAULT_OUTROOT_SUBDIR = "cheasebs_scaling_comparison"


def tag_of(omt, omn):
    return f"omt{fmt_alpha(omt)}_omne{fmt_alpha(omn)}"


def parse_pair(text):
    """'0.8,1.1' -> (0.8, 1.1). Accepts whitespace or a slash too."""
    parts = [p for p in text.replace("/", ",").replace(" ", ",").split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--pair takes two numbers, omt first: --pair 0.8,1.1")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--pair values must be numbers: {text!r}")


def default_outroot(shot, stamp):
    """$SCRATCH if the environment has one, else this directory."""
    scratch = os.environ.get("SCRATCH")
    root = os.path.join(scratch, DEFAULT_OUTROOT_SUBDIR) if scratch else HERE
    return os.path.join(root, f"runs_{shot}_{stamp}")


def load_base_discharge(case_dir, gfile=None):
    """The base DischargeData for a case directory.

    Finding the gfile is DischargeData's own job -- it sniffs the file's
    contents rather than pattern-matching its name -- so this only constructs it
    and reports what it picked up. Construction is filepath resolution only;
    nothing is parsed until the physics object asks for it.
    """
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData

    data = DischargeData(input_dir=case_dir, gfile=gfile)
    if not data.gfile_filepath:
        raise SystemExit(
            f"No gfile found in {case_dir}. Pass one explicitly with --gfile."
        )
    if not data.profiles_filepaths:
        raise SystemExit(
            f"No profiles_* files found in {case_dir}; the scan has nothing to scale."
        )
    return data


def fit_base_profiles(phys_base):
    """{var: (StefanikovaProfile, rms_relative)} for everything mtanh_full scales.

    Fitted once on the base profiles and reused at every point, as pedestal_scan
    does: the fit describes the unscaled plasma, and refitting per point would
    let the fit quality drift from point to point and confound exactly the
    stability comparison this script exists to make.

    A variable whose fit raises is recorded rather than propagated -- it is a
    result about the method, and the other variables still have something to say.
    """
    from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import (
        fit_mtanh_full)

    fits, quality = {}, {}
    for var in TEMPERATURE_VARS + (DENSITY_VAR,):
        if var not in phys_base.ds:
            continue
        try:
            profile, meta = fit_mtanh_full(phys_base.ds, var, **FIT_KWARGS)
            fits[var] = profile
            quality[var] = meta.get("rms_relative")
        except Exception as exc:
            quality[var] = f"FIT FAILED: {type(exc).__name__}: {exc}"
    return fits, quality


def scale_omn_omt(phys_base, omt, omn, midped, topped):
    """Method 1: scale the profile gradients directly."""
    return (phys_base
            .apply_omt(alpha=omt, rhot_midped=midped, rhot_topped=topped,
                       apply_to_tz=TZ_EQ_TI)
            .apply_omne(alpha=omn, rhot_midped=midped, rhot_topped=topped))


def scale_mtanh_full(phys_base, omt, omn, fits, knob):
    """Method 2: scale the fitted pedestal, one variable at a time.

    Per variable rather than by passing a list, because apply_mtanh_full reuses
    a single `fit` argument across every variable in a list -- correct only when
    they share a fit, which these do not.

    ne goes last and carries enforce_quasineutrality, so ni and nz are rebuilt
    from the final ne rather than from an intermediate one.
    """
    phys = phys_base
    for var in TEMPERATURE_VARS:
        if var not in phys.ds or var not in fits:
            continue
        phys = phys.apply_mtanh_full(var, fit=fits[var], **{knob: omt})
    if DENSITY_VAR in phys.ds and DENSITY_VAR in fits:
        phys = phys.apply_mtanh_full(DENSITY_VAR, fit=fits[DENSITY_VAR],
                                     enforce_quasineutrality=True, qz=6.0,
                                     **{knob: omn})
    return phys


def scale(phys_base, method, omt, omn, midped, topped, fits, knob):
    """The scaled discharge for one point under one method."""
    if method == "omn_omt":
        return scale_omn_omt(phys_base, omt, omn, midped, topped)
    if method == "mtanh_full":
        return scale_mtanh_full(phys_base, omt, omn, fits, knob)
    raise ValueError(f"unknown scaling method: {method!r}")


def resolve_window(shot, topped_arg, midped_arg):
    """(topped, midped, why) for a shot: explicit flags, else the measured window.

    Each flag overrides independently, so one can be pinned while the other
    keeps its measured value. `why` is printed, because a window silently
    falling back to the NSTX default on a discharge it does not describe is the
    failure this table exists to prevent.
    """
    tabulated = RAMP_WINDOWS.get(str(shot))
    if tabulated:
        topped, midped = tabulated
        why = "%s for %s" % (WINDOW_PROVENANCE.get(str(shot), "tabulated window"),
                             shot)
    else:
        topped, midped = FALLBACK_TOPPED, FALLBACK_MIDPED
        why = (f"NO measured window for shot {shot} -- falling back to the NSTX "
               f"default {FALLBACK_TOPPED}/{FALLBACK_MIDPED}, which may not "
               f"describe this pedestal. Fit it and add it to RAMP_WINDOWS")

    overridden = []
    if topped_arg is not None:
        topped = topped_arg
        overridden.append("topped")
    if midped_arg is not None:
        midped = midped_arg
        overridden.append("midped")
    if overridden:
        why += f"; {' and '.join(overridden)} overridden on the command line"
    return topped, midped, why


def detect_pedestal(phys, var="ne"):
    """(midped, topped) from the steepest gradient, for --auto-pedestal.

    Deliberately crude and fit-free: midped is where |d(var)/d rhot| peaks in the
    outer half, topped is the first point inboard of it where the gradient has
    fallen to a third of that peak. Taking the gradient method's own window from
    a fit would put the fit back in the loop it is being compared against. The
    numbers are printed; they are a starting point to sanity-check, not an
    authority.
    """
    import numpy as np

    da = getattr(phys, var)
    vals = da.pint.magnitude if hasattr(da, "pint") else da.values
    rhot = np.asarray(phys.rhot)
    vals = np.asarray(vals, dtype=float)

    grad = np.abs(np.gradient(vals, rhot))
    outer = rhot > 0.5
    idx = int(np.argmax(np.where(outer, grad, -np.inf)))
    midped = float(rhot[idx])

    peak = grad[idx]
    topped = midped - 0.05
    for j in range(idx, -1, -1):
        if grad[j] < peak / 3.0:
            topped = float(rhot[j])
            break
    if not (0.0 < topped < midped):
        topped = max(midped - 0.05, 0.0)
    return midped, topped


def gradient_report(phys, radii=(0.9, 0.95, 0.99)):
    """Gradient length L = -y/(dy/drhot) at a few radii, so a dry run shows the
    scaling actually bit.

    L, not a/L: TPED's gradient_length returns the length itself, so SMALLER is
    a STEEPER profile. Scaling a gradient down (alpha < 1) therefore makes these
    numbers larger. Read them at rhot_midped and inside the pedestal; at
    rhot_topped the ramp's own derivative contributes a term of its own and the
    response there is not a clean factor of alpha.
    """
    import numpy as np

    out = {}
    for name in ("Te", "Ti", "ne"):
        try:
            gl = phys.gradient_length(name)
            vals = gl.pint.magnitude if hasattr(gl, "pint") else gl.values
            rhot = np.asarray(phys.rhot)
            order = np.argsort(rhot)
            out[name] = {float(r): round(float(np.interp(
                r, rhot[order], np.asarray(vals, float)[order])), 5)
                for r in radii}
        except Exception as exc:
            out[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


def solve(phys, method, omt, omn, run_dir, final_dir, gfile, phys_base,
          args, paths, solver):
    """Run cheaseBS on one already-scaled discharge, score it, copy it back."""
    from TPED.projects.discharge_tools.src.cheasebs_runner import (
        CheasebsAcceptance, run_cheasebs_workflow)

    tag = tag_of(omt, omn)
    row = {"branch": method, "method": method, "tag": tag,
           "t": omt, "n": omn, "omt": omt, "omne": omn,
           "run_dir": run_dir, "final_dir": final_dir,
           "history": phys.history, "gradients": gradient_report(phys)}

    t0 = time.time()
    try:
        eqdsk, result = run_cheasebs_workflow(
            gfile_path=gfile,
            ds=phys.ds,
            reference_ds=phys_base.ds,
            savedir=run_dir,
            config_template=args.cheasebs_config,
            chease_binary=paths["chease_binary"],
            cheasebs_script=paths["cheasebs_script"],
            chease_namelist=paths["chease_namelist"],
            baseline_dir=paths["baseline_dir"],
            acceptance=CheasebsAcceptance.production(
                analysis_radii=tuple(args.analysis_radii)),
            strict=False,
            return_acceptance=True,
            **solver,
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_s"] = time.time() - t0
        return row
    row["wall_s"] = time.time() - t0
    row["eqdsk"] = eqdsk

    rec = result.to_dict()
    row["accepted"] = rec.get("accepted")
    row["reasons"] = rec.get("reasons")
    row["ip_error_rel"] = rec.get("ip_error_rel")
    row["q_errors_rel"] = rec.get("q_errors_rel")
    row["q_edge_error_rel"] = rec.get("q_edge_error_rel")
    row["iterations"] = rec.get("cheasebs_iterations")
    row["converged"] = rec.get("cheasebs_converged")
    row["final_ip_a"] = rec.get("final_ip_a")
    row["target_ip_a"] = rec.get("target_ip_a")

    # The merged config the solve actually read, so copy_back resolves the same
    # profile paths cheaseBS used rather than a set reconstructed here.
    cfg_path = os.path.join(run_dir, "cheasebs_run_config.json")
    cfg = {}
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            pass

    for note in render_plots(run_dir, cfg):
        row.setdefault("plot_notes", []).append(note)
        print(f"    {note}")

    return copy_back_if_needed(row, run_dir, final_dir, eqdsk, cfg)


def summarize(rows):
    """Per-method counts, which is the comparison in one block.

    Accepted / converged / failed and the worst Ip error among the points that
    completed. Not a verdict -- the per-point table and the run plots are -- but
    it is the number that says whether one method is holding together better
    than the other across the same grid.
    """
    lines = []
    for method in METHODS:
        sub = [r for r in rows if r.get("branch") == method]
        if not sub:
            continue
        done = [r for r in sub if "error" not in r]
        ips = [r["ip_error_rel"] for r in done
               if isinstance(r.get("ip_error_rel"), (int, float))]
        iters = [r["iterations"] for r in done
                 if isinstance(r.get("iterations"), (int, float))]
        parts = [f"{len(done)}/{len(sub)} completed",
                 f"{sum(1 for r in done if r.get('accepted'))} accepted",
                 f"{sum(1 for r in done if r.get('converged'))} converged"]
        if ips:
            parts.append(f"worst Ip err {max(ips):.2%}")
        if iters:
            parts.append(f"mean iters {sum(iters) / len(iters):.1f}")
        lines.append(f"  {method:<12} " + ", ".join(parts))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scale DIII-D profiles two ways and run cheaseBS on both.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case-dir", required=True,
                    help="the discharge directory: base gfile, pfile and "
                         "profiles_{e,i,z}, handed to DischargeData as-is")
    ap.add_argument("--gfile", default=None,
                    help="base EQDSK; default is the one DischargeData finds")

    ap.add_argument("--method", nargs="+", default=list(METHODS),
                    choices=list(METHODS),
                    help="which scaling method(s) to run; both by default, "
                         "which is the comparison")
    ap.add_argument("--mtanh-knob", default="scale_height",
                    choices=["scale_height", "scale_width", "scale_core_height",
                             "scale_sol", "scale_slope", "scale_core_width"],
                    help="the apply_mtanh_full parameter the scale multiplies; "
                         "scale_height matches the NSTX pedestal_scan axes")

    ap.add_argument("--pair", type=parse_pair, action="append", default=None,
                    metavar="OMT,OMN",
                    help="a point to run, omt first; repeatable. "
                         "Default is the IFS scan's own grid")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="run only these tags, e.g. --only omt1p0_omne1p0")

    ap.add_argument("--rhot-midped", type=float, default=None,
                    help="mid-pedestal rho_tor; omn_omt pins the profile here. "
                         "Default is this shot's tabulated ramp (RAMP_WINDOWS)")
    ap.add_argument("--rhot-topped", type=float, default=None,
                    help="top-of-pedestal rho_tor; alpha ramps in over [top, mid]. "
                         "Default is this shot's measured window")
    ap.add_argument("--auto-pedestal", action="store_true",
                    help="measure the pedestal window off the base ne gradient "
                         "instead of using the defaults, and print what it found")

    ap.add_argument("--outroot", default=None,
                    help="where results go (default: "
                         "$SCRATCH/cheasebs_scaling_comparison/runs_<shot>_<stamp>)")
    ap.add_argument("--baseline-dir", default=None,
                    help="shared cheaseBS baseline; default is one per campaign "
                         "under the scratch working tree")
    ap.add_argument("--scratch-root", default=None,
                    help="where cheaseBS actually runs (default: TPED OUTPUT_PATH). "
                         "Only the result and the record are copied to --outroot")
    ap.add_argument("--in-place", action="store_true",
                    help="run directly under --outroot instead of a scratch "
                         "working tree, keeping the full per-iteration tree there")
    ap.add_argument("--cheasebs-config", default=DEFAULT_CONFIG,
                    help="cheaseBS JSON template holding the solver settings")
    ap.add_argument("--cheasebs-dir", default=None,
                    help="cheaseBS repo (default: TPED CHEASEBS_PATH)")
    ap.add_argument("--chease-binary", default=None,
                    help="CHEASE executable (default: TPED CHEASE_PATH/src-f90/chease)")
    ap.add_argument("--chease-namelist", default=None,
                    help="CHEASE namelist template; the DIII-D one, not TPED's "
                         "NSTX default")

    ap.add_argument("--max-iter", type=int, default=None,
                    help="override the template's max_iter")
    ap.add_argument("--amplitude-warmup-iters", type=int, default=None,
                    help="override the template's amplitude_warmup_iters: hold "
                         "the driven amplitude fixed for the first N outer "
                         "iterations so the bootstrap settles before the "
                         "amplitude controller starts moving. 0 disables the "
                         "warm-up entirely")
    ap.add_argument("--analysis-radii", type=float, nargs="*", default=[],
                    help="rho_tor values where q is scored; empty skips the q checks")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not let a rejected point set a non-zero exit status")
    ap.add_argument("--dry-run", action="store_true",
                    help="scale the profiles and report the gradients, solve nothing")
    ap.add_argument("--log", default=None,
                    help="log file (default: <outroot>/campaign_<stamp>.log)")
    args = ap.parse_args(argv)

    case_dir = os.path.abspath(os.path.expanduser(args.case_dir))
    if not os.path.isdir(case_dir):
        raise SystemExit(f"--case-dir does not exist: {case_dir}")
    if not os.path.isfile(args.cheasebs_config):
        raise SystemExit(f"cheaseBS config template not found: {args.cheasebs_config}")

    data = load_base_discharge(
        case_dir,
        gfile=os.path.abspath(os.path.expanduser(args.gfile)) if args.gfile else None)
    gfile = os.path.abspath(data.gfile_filepath)
    shot = shot_of(gfile)

    methods = [m for m in METHODS if m in args.method]     # stable order
    pairs = [tuple(p) for p in (args.pair or DEFAULT_PAIRS)]
    if args.only:
        wanted = set(args.only)
        pairs = [p for p in pairs if tag_of(*p) in wanted]
        if not pairs:
            raise SystemExit(f"--only matched none of the points: {sorted(wanted)}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    outroot = os.path.abspath(args.outroot or default_outroot(shot, stamp))
    os.makedirs(outroot, exist_ok=True)

    log_path = args.log or os.path.join(outroot, f"campaign_{stamp}.log")
    if not args.dry_run:
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee
        sys.stderr = tee

    # The CHEASE locations come from the TPED user config when they are not
    # given, so this agrees with every other cheaseBS caller in the repo about
    # which binary and which driver are "the" ones. The namelist is the one
    # exception: TPED's default is chease_namelist_nstx, wrong for DIII-D, so it
    # defaults to the repo's plain chease_namelist here.
    #
    # A dry run solves nothing, so it needs none of it. Requiring a configured
    # CHEASE to look at what the scaling did would make the cheap check the
    # expensive one.
    namelist, scratch_root = args.chease_namelist, args.scratch_root
    cheasebs_dir, chease_binary = args.cheasebs_dir, args.chease_binary
    cheasebs_script = paths = workroot = baseline_dir = None
    try:
        from TPED.config.config_helper import Config
        cfg_paths = Config()
        cheasebs_dir = cheasebs_dir or cfg_paths.get_path("CHEASEBS_PATH")
        chease_binary = chease_binary or os.path.join(
            cfg_paths.get_path("CHEASE_PATH") or "", "src-f90", "chease")
        namelist = namelist or os.path.join(cheasebs_dir or "", "chease_namelist")
        scratch_root = scratch_root or cfg_paths.get_path("OUTPUT_PATH")
    except Exception as exc:
        if not args.dry_run:
            raise SystemExit(
                f"Could not read the TPED config ({exc}). Pass --cheasebs-dir, "
                f"--chease-binary, --chease-namelist and --scratch-root explicitly."
            )

    if not args.dry_run:
        if not cheasebs_dir or not os.path.isdir(cheasebs_dir):
            raise SystemExit(f"cheaseBS directory not found: {cheasebs_dir!r} "
                             f"(set CHEASEBS_PATH or pass --cheasebs-dir)")
        cheasebs_script = os.path.join(cheasebs_dir,
                                       "run_chease_iterative_profiles.py")
        if not os.path.isfile(cheasebs_script):
            raise SystemExit(f"cheaseBS driver not found: {cheasebs_script}")
        if not chease_binary or not os.path.isfile(chease_binary):
            raise SystemExit(f"CHEASE executable not found: {chease_binary!r} "
                             f"(set CHEASE_PATH or pass --chease-binary)")
        if not namelist or not os.path.isfile(namelist):
            raise SystemExit(f"CHEASE namelist not found: {namelist!r} "
                             f"(pass --chease-namelist)")
        paths = {"cheasebs_script": cheasebs_script,
                 "chease_binary": os.path.abspath(chease_binary),
                 "chease_namelist": os.path.abspath(namelist)}

        if args.in_place:
            workroot = outroot
        else:
            if not scratch_root:
                raise SystemExit(
                    "No scratch root: set OUTPUT_PATH in the TPED user config, "
                    "pass --scratch-root, or use --in-place to run under --outroot."
                )
            workroot = os.path.join(
                os.path.abspath(os.path.expanduser(scratch_root)),
                "cheaseBS_runs", f"{stamp}-{shot}-scaling_comparison")

        # One baseline for the whole comparison: it is a function of the EQDSK
        # and the untransformed reference profiles, and neither depends on the
        # method. Building it per method would only buy two chances to build it
        # differently.
        baseline_dir = args.baseline_dir or os.path.join(workroot, "baseline")
        paths["baseline_dir"] = baseline_dir

    solver = {}
    if args.max_iter is not None:
        solver["max_iter"] = args.max_iter
    if args.amplitude_warmup_iters is not None:
        # 0 is a meaningful value, not "unset", so the sentinel is None.
        solver["amplitude_warmup_iters"] = args.amplitude_warmup_iters

    print(f"=== cheaseBS scaling-method comparison {stamp} ===")
    print(f"case dir  : {case_dir}")
    print(f"base gfile: {gfile}  (shot {shot})")
    for path in data.profiles_filepaths:
        print(f"base prof : {path}")
    print(f"base pfile: {data.pfile_filepath}")
    print(f"methods   : {', '.join(methods)}")
    print(f"mtanh knob: {args.mtanh_knob}")
    print(f"template  : {args.cheasebs_config}")
    if args.dry_run:
        print("cheaseBS  : (not resolved; dry run solves nothing)")
    else:
        print(f"cheaseBS  : {cheasebs_script}")
        print(f"chease    : {chease_binary}")
        print(f"namelist  : {namelist}")
        print(f"workroot  : {workroot}"
              f"{'   (in place)' if args.in_place else '   (scratch; not preserved)'}")
        print(f"baseline  : {baseline_dir}   (shared by both methods)")
    print(f"outroot   : {outroot}   (results and records)")
    print(f"overrides : {solver or '(template as-is)'}")
    print(f"radii     : {args.analysis_radii or '(q checks skipped)'}")
    print(f"points    : {len(pairs)} x {len(methods)} method(s) -> "
          f"{', '.join(tag_of(*p) for p in pairs)}")
    print(f"log       : {log_path}")
    print(f"pid       : {os.getpid()}")
    print()

    # Harmonized once and reused: every point scales the source profiles, never
    # its predecessor, and both methods start from the same object. This
    # untransformed dataset is also what cheaseBS gets as the reference set.
    from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics

    phys_base = DischargePhysics(data)

    topped, midped, window_why = resolve_window(shot, args.rhot_topped,
                                                args.rhot_midped)
    if args.auto_pedestal:
        midped, topped = detect_pedestal(phys_base)
        window_why = "measured now off the base ne gradient (--auto-pedestal)"
    # midped == 1.0 is not an edge case to be rejected, it is the IFS handoff's
    # own choice: it pivots the power law on the separatrix and so holds the
    # separatrix values fixed. topped == 0.0 is likewise legal (ramp on from the
    # axis). What must hold is only that the ramp has positive width and that
    # both ends are inside the closed radial domain.
    if not 0.0 <= topped < midped <= 1.0:
        raise SystemExit(f"ramp window is not ordered: topped={topped}, "
                         f"midped={midped}; need 0 <= topped < midped <= 1")
    print(f"pedestal  : topped={topped:.4f}, midped={midped:.4f}  (omn_omt only)")
    print(f"            {window_why}")

    fits, fit_quality = ({}, {})
    if "mtanh_full" in methods:
        fits, fit_quality = fit_base_profiles(phys_base)
        print(f"mtanh fits: {json.dumps(fit_quality, default=str)}"
              f"   (relative rms of each base fit)")
    print(f"base L    : {json.dumps(gradient_report(phys_base), default=str)}"
          f"   (gradient length in rhot; smaller = steeper)")
    print()

    if args.dry_run:
        for method in methods:
            print(f"--- {method} ---")
            for omt, omn in pairs:
                try:
                    phys = scale(phys_base, method, omt, omn, midped, topped,
                                 fits, args.mtanh_knob)
                    print(f"  {tag_of(omt, omn)}: "
                          f"{json.dumps(gradient_report(phys), default=str)}")
                except Exception as exc:
                    print(f"  {tag_of(omt, omn)}: RAISED "
                          f"{type(exc).__name__}: {exc}")
        print("\ndry run: no equilibrium was solved")
        return 0

    rows, failed = [], []
    t_camp = time.time()
    for method in methods:
        for omt, omn in pairs:
            tag = tag_of(omt, omn)
            print(f"--- {method} / {tag}  (omt {omt:.2f}, omne {omn:.2f}) ---",
                  flush=True)
            try:
                phys = scale(phys_base, method, omt, omn, midped, topped,
                             fits, args.mtanh_knob)
                row = solve(phys, method, omt, omn,
                            os.path.join(workroot, method, tag),
                            os.path.join(outroot, method, tag),
                            gfile, phys_base, args, paths, solver)
            except Exception:
                # One point failing outright must not take the rest of the
                # comparison with it; the remaining points are independent
                # hours of work, and a method that fails here has said
                # something about itself that the other points still measure.
                print(f"!!! {method}/{tag} RAISED, continuing with the next point")
                traceback.print_exc()
                row = {"branch": method, "method": method, "tag": tag,
                       "t": omt, "n": omn, "omt": omt, "omne": omn,
                       "error": "runner raised, see traceback in the log"}
            rows.append(row)
            if "error" in row:
                failed.append(f"{method}/{tag}")
                print(f"    FAILED: {row['error']}", flush=True)
            else:
                print(f"    {row.get('iterations')} iters, {row['wall_s']:.0f}s, "
                      f"converged={row.get('converged')}, "
                      f"accepted={row.get('accepted')}", flush=True)
            # Written after every point, not at the end: a campaign that is
            # killed halfway still leaves a readable record of what it did.
            with open(os.path.join(outroot, "comparison.json"), "w") as fh:
                json.dump({"shot": shot, "case_dir": case_dir, "gfile": gfile,
                           "base_profiles": data.profiles_filepaths,
                           "base_pfile": data.pfile_filepath,
                           "methods": methods, "mtanh_knob": args.mtanh_knob,
                           "mtanh_fit_quality": fit_quality,
                           "fit_kwargs": FIT_KWARGS,
                           "pairs": [list(p) for p in pairs],
                           "rhot_midped": midped, "rhot_topped": topped,
                           "cheasebs_config": args.cheasebs_config,
                           "overrides": solver, "baseline_dir": baseline_dir,
                           "workroot": workroot, "in_place": bool(args.in_place),
                           "paths": paths,
                           "analysis_radii": args.analysis_radii, "rows": rows},
                          fh, indent=1, default=str)

    table = text_table(rows)
    with open(os.path.join(outroot, "table.txt"), "w") as fh:
        fh.write(table + "\n")
    print(f"\n=== {len(rows)} solve(s) in {time.time() - t_camp:.0f}s ===")
    print(table)
    print("\nper method:")
    print(summarize(rows))

    rejected = [f"{r['method']}/{r['tag']}" for r in rows
                if "error" not in r and r.get("accepted") is False]
    if failed:
        print(f"\nFAILED to complete: {', '.join(failed)}")
    if rejected:
        print(f"Completed but REJECTED by the acceptance gate: {', '.join(rejected)}")
        for r in rows:
            if r.get("accepted") is False:
                for reason in r.get("reasons") or []:
                    print(f"  {r['method']}/{r['tag']}: {reason}")
    print(f"\nrecord : {os.path.join(outroot, 'comparison.json')}")
    print(f"table  : {os.path.join(outroot, 'table.txt')}")
    print(f"results: {outroot}")
    if not args.in_place:
        print(f"scratch: {workroot}  (full per-iteration tree; purgeable)")

    if failed:
        return 1
    if rejected and not args.no_gate:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
