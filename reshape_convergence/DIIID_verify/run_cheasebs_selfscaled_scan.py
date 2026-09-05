#!/usr/bin/env python3
"""The same omt/omne scan, scaled by TPED instead of read off disk.

Companion to run_cheasebs_scaling_scan.py, which replays profiles the older IFS
scaling code already wrote. That replay worked: the driven amplitude moved and q
and the shear responded, on the same DIII-D EQDSK the reshape campaign uses. So
the solver path is not the suspect, and what separates the two is how the
profiles were made -- an mtanh fit and a pedestal reshape on one side, a direct
gradient scaling on the other.

This script closes that gap by keeping everything except the scaling identical.
Same base EQDSK, same base profiles, same cheaseBS settings, same output layout
and the same point tags, so the two campaigns can be compared directory by
directory. The only difference is that the scaled profiles are produced here,
through the discharge object:

    phys.apply_omt(alpha=omt, ...)      # Te and Ti, and Tz with them
        .apply_omne(alpha=omn, ...)     # ne, with ni and nz held quasineutral

Both are the pedestal gradient transforms in TPED's pedestal_transforms: the
profile is re-exponentiated about its mid-pedestal value,
`T_new = T_mid * (T/T_mid)**alpha`, with alpha ramped smoothly from 1 in the
core to its full value across [rhot_topped, rhot_midped]. No mtanh fit is
involved at any point, which is what makes this a test of the fit rather than
another run through it.

`--omt` scales both temperatures, `--omn` scales the electron density and
carries ni and nz with it. That mirrors the omt/omne convention the IFS scan
directories are named for, so a point here has the same name and the same
intended meaning as the point it is being compared against.

The solve goes through TPED's `run_cheasebs_workflow` -- the function
`output_gfile` itself calls -- on a scratch run directory, and only the
equilibrium, the records and the end plots are copied into --outroot, using the
same copy_back the file-driven runner uses. `output_gfile` would be the shorter
route but it hardcodes `chease_namelist_nstx` as an explicit argument, so a
DIII-D namelist cannot be passed through its **cheasebs_overrides without
colliding with it; calling one layer down is what buys the right namelist.

The reference profiles are the untransformed base dataset, so the baseline
decomposition is a fixed frame rather than one that moves with the scan -- the
failure that made every downward point a null test.

USAGE

    # one point first: the unity point, which should reproduce the source
    python -u run_cheasebs_selfscaled_scan.py \
        --gfile /data/DIIID/DIIID162940/DIIID162940/chease/g162940.02944_670 \
        --base-profile-dir <cheaseBS>/data/profiles --base-profile-stem profiles_162940 \
        --only omt1p0_omne1p0

    # the full grid, detached, matching the IFS scan's points
    nohup python -u run_cheasebs_selfscaled_scan.py --gfile ... > /dev/null 2>&1 &
    tail -f runs_162940_selfscaled/campaign_*.log

    # see the plan and the scaled profiles' gradients without solving anything
    python run_cheasebs_selfscaled_scan.py --gfile ... --dry-run

`-u` matters: without it Python block-buffers stdout when it is not a terminal
and the log stays empty for hours.

Exit status is 0 only when every solve completed AND every point was accepted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys
import time
import traceback

# cheaseBS renders per-run PNGs and must not reach for a display.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_cheasebs_scaling_scan import (  # noqa: E402
    Tee, copy_back, render_plots, shot_of, text_table)

# The points the IFS scan directory holds for 162940, so this campaign lands
# tag-for-tag beside it. (omt, omn).
DEFAULT_PAIRS = (
    (0.7, 0.7), (0.7, 0.8), (0.7, 0.9),
    (0.8, 0.8), (0.8, 0.9), (0.8, 1.0), (0.8, 1.1),
    (0.9, 0.9), (0.9, 1.0), (0.9, 1.1),
    (1.0, 1.0), (1.0, 1.1),
    (1.1, 1.0), (1.1, 1.1),
)

# TPED's own pedestal window for this transform pair (cheasebs_identity test).
# Overridable, and --auto-pedestal measures them off the profile instead.
DEFAULT_MIDPED = 0.95
DEFAULT_TOPPED = 0.90

# DIII-D validated solver path: j_parallel replay on rhot with QSPEC enforced.
# Lives in a template file so it is the same object the file-driven runner and
# any later re-run read, rather than a second copy that can drift.
DEFAULT_CONFIG = os.path.join(HERE, "diiid_cheasebs_config.json")


def fmt_alpha(v):
    """0.7 -> '0p7', 1.0 -> '1p0', 1.05 -> '1p05'. The scan's own tag spelling."""
    s = f"{v:.1f}" if abs(v - round(v, 1)) < 1e-9 else f"{v:.2f}".rstrip("0")
    return s.replace(".", "p")


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


def stage_base_profiles(profiles, workdir):
    """Copy the base profiles to canonical `profiles_{e,i,z}` names.

    Not cosmetic. ProfilesData reads the species straight out of the filename
    with `re.match(r"profiles[_\\-]?([A-Za-z0-9]+)")`, so the bundled
    `profiles_162940_e` parses its species as "162940" -- all three files land
    as the same unknown species and the harmonized dataset is wrong in a way
    that does not raise. The canonical stem is the only name that parses.
    """
    staged = {}
    os.makedirs(workdir, exist_ok=True)
    for spec, src in profiles.items():
        dst = os.path.join(workdir, f"profiles_{spec}")
        shutil.copy2(src, dst)
        staged[spec] = dst
    return staged


def resolve_base_profiles(args):
    """{spec: path} for the three base profiles, from a dir+stem or three paths."""
    if args.base_profiles:
        if len(args.base_profiles) != 3:
            raise SystemExit("--base-profiles takes exactly three paths, e/i/z in order")
        found = dict(zip(("e", "i", "z"), [os.path.abspath(p) for p in args.base_profiles]))
    else:
        if not args.base_profile_dir:
            raise SystemExit(
                "Give the base profiles: --base-profile-dir (with --base-profile-stem) "
                "or --base-profiles <e> <i> <z>."
            )
        found = {spec: os.path.join(os.path.abspath(args.base_profile_dir),
                                    f"{args.base_profile_stem}_{spec}")
                 for spec in ("e", "i", "z")}
    missing = [p for p in found.values() if not os.path.isfile(p)]
    if missing:
        raise SystemExit("Base profiles not found:\n  " + "\n  ".join(missing))
    return found


def detect_pedestal(phys, var="ne"):
    """(midped, topped) from the steepest gradient, for --auto-pedestal.

    Deliberately crude and fit-free: midped is where |d(var)/d rhot| peaks in the
    outer half, topped is the first point inboard of it where the gradient has
    fallen to a third of that peak. The whole point of this script is to take the
    mtanh fit out of the loop, so the pedestal window must not come from one
    either. The numbers are printed; they are a starting point to sanity-check,
    not an authority.
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
    """a/L_x at a few radii, so a dry run shows the scaling actually bit."""
    out = {}
    for name in ("Te", "Ti", "ne"):
        try:
            gl = phys.gradient_length(name)
            vals = gl.pint.magnitude if hasattr(gl, "pint") else gl.values
            import numpy as np
            rhot = np.asarray(phys.rhot)
            order = np.argsort(rhot)
            out[name] = {float(r): float(np.interp(r, rhot[order],
                                                   np.asarray(vals, float)[order]))
                         for r in radii}
        except Exception as exc:
            out[name] = f"unavailable: {type(exc).__name__}: {exc}"
    return out


def solve(phys_base, omt, omn, midped, topped, run_dir, final_dir, gfile,
          args, paths, solver):
    """One point: scale the profiles, then reconstruct the equilibrium.

    The transforms are applied to the base discharge every time rather than
    chained onto the previous point, so each point is a scaling of the source
    profiles and not of its predecessor.
    """
    from TPED.projects.discharge_tools.src.cheasebs_runner import (
        CheasebsAcceptance, run_cheasebs_workflow)

    tag = tag_of(omt, omn)
    row = {"tag": tag, "omt": omt, "omne": omn,
           "run_dir": run_dir, "final_dir": final_dir}

    phys = (phys_base
            .apply_omt(alpha=omt, rhot_midped=midped, rhot_topped=topped)
            .apply_omne(alpha=omn, rhot_midped=midped, rhot_topped=topped))
    row["history"] = phys.history
    row["gradients"] = gradient_report(phys)

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

    if os.path.normcase(os.path.realpath(run_dir)) != os.path.normcase(
            os.path.realpath(final_dir)):
        try:
            copied = copy_back(run_dir, final_dir, eqdsk, cfg)
            row["copied"] = copied
            row["eqdsk_final"] = os.path.join(final_dir, os.path.basename(eqdsk))
            print(f"    copied {len(copied)} file(s) -> {final_dir}")
        except Exception as exc:
            # The solve succeeded; failing to copy it out is worth reporting
            # loudly but is not the same as the point having failed.
            row["copy_error"] = f"{type(exc).__name__}: {exc}"
            print(f"    COPY-BACK FAILED: {row['copy_error']}")
    else:
        row["eqdsk_final"] = eqdsk
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Scale the profiles with TPED, then run cheaseBS on each point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--gfile", required=True, help="base EQDSK, the frozen geometry")
    ap.add_argument("--base-profile-dir", default=None,
                    help="directory holding the base profiles")
    ap.add_argument("--base-profile-stem", default="profiles_162940",
                    help="their filename stem; '<stem>_e' and so on")
    ap.add_argument("--base-profiles", nargs=3, default=None, metavar=("E", "I", "Z"),
                    help="the three base profile paths, e/i/z in order")

    ap.add_argument("--pair", type=parse_pair, action="append", default=None,
                    metavar="OMT,OMN",
                    help="a point to run, omt first; repeatable. "
                         "Default is the IFS scan's own grid")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="run only these tags, e.g. --only omt1p0_omne1p0")

    ap.add_argument("--rhot-midped", type=float, default=DEFAULT_MIDPED,
                    help="mid-pedestal rho_tor; the transforms pin the profile here")
    ap.add_argument("--rhot-topped", type=float, default=DEFAULT_TOPPED,
                    help="top-of-pedestal rho_tor; alpha ramps in over [top, mid]")
    ap.add_argument("--auto-pedestal", action="store_true",
                    help="measure the pedestal window off the base ne gradient "
                         "instead of using the defaults, and print what it found")

    ap.add_argument("--outroot", default=None,
                    help="where per-point results go "
                         "(default: <this dir>/runs_<shot>_selfscaled)")
    ap.add_argument("--baseline-dir", default=None,
                    help="shared cheaseBS baseline; default is one per campaign "
                         "under the scratch root")
    ap.add_argument("--scratch-root", default=None,
                    help="where cheaseBS actually runs (default: TPED OUTPUT_PATH). "
                         "Only the result and the record are copied to --outroot")
    ap.add_argument("--in-place", action="store_true",
                    help="run directly under --outroot instead of scratch, keeping "
                         "the full per-iteration tree there")
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
    ap.add_argument("--analysis-radii", type=float, nargs="*", default=[],
                    help="rho_tor values where q is scored; empty skips the q checks")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not let a rejected point set a non-zero exit status")
    ap.add_argument("--dry-run", action="store_true",
                    help="scale the profiles and report the gradients, solve nothing")
    ap.add_argument("--log", default=None,
                    help="log file (default: <outroot>/campaign_<stamp>.log)")
    args = ap.parse_args(argv)

    gfile = os.path.abspath(os.path.expanduser(args.gfile))
    if not os.path.isfile(gfile):
        raise SystemExit(f"base gfile not found: {gfile}")
    shot = shot_of(gfile)
    if not os.path.isfile(args.cheasebs_config):
        raise SystemExit(f"cheaseBS config template not found: {args.cheasebs_config}")

    base_profiles = resolve_base_profiles(args)

    pairs = [tuple(p) for p in (args.pair or DEFAULT_PAIRS)]
    if args.only:
        wanted = set(args.only)
        pairs = [p for p in pairs if tag_of(*p) in wanted]
        if not pairs:
            raise SystemExit(f"--only matched none of the points: {sorted(wanted)}")

    outroot = os.path.abspath(
        args.outroot or os.path.join(HERE, f"runs_{shot}_selfscaled"))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
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
    namelist, scratch_root = args.chease_namelist, args.scratch_root
    cheasebs_dir, chease_binary = args.cheasebs_dir, args.chease_binary
    try:
        from TPED.config.config_helper import Config
        cfg_paths = Config()
        cheasebs_dir = cheasebs_dir or cfg_paths.get_path("CHEASEBS_PATH")
        chease_binary = chease_binary or os.path.join(
            cfg_paths.get_path("CHEASE_PATH") or "", "src-f90", "chease")
        namelist = namelist or os.path.join(cheasebs_dir or "", "chease_namelist")
        scratch_root = scratch_root or cfg_paths.get_path("OUTPUT_PATH")
    except Exception as exc:
        raise SystemExit(
            f"Could not read the TPED config ({exc}). Pass --cheasebs-dir, "
            f"--chease-binary, --chease-namelist and --scratch-root explicitly."
        )
    if not cheasebs_dir or not os.path.isdir(cheasebs_dir):
        raise SystemExit(f"cheaseBS directory not found: {cheasebs_dir!r} "
                         f"(set CHEASEBS_PATH or pass --cheasebs-dir)")
    cheasebs_script = os.path.join(cheasebs_dir, "run_chease_iterative_profiles.py")
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

    # cheaseBS works in scratch and only the result travels back, as in every
    # other cheaseBS caller in this repo.
    if args.in_place:
        workroot = outroot
    else:
        if not scratch_root:
            raise SystemExit(
                "No scratch root: set OUTPUT_PATH in the TPED user config, pass "
                "--scratch-root, or use --in-place to run under --outroot."
            )
        workroot = os.path.join(os.path.abspath(os.path.expanduser(scratch_root)),
                                "cheaseBS_runs", f"{stamp}-{shot}-selfscaled")

    # One baseline for the campaign: it is a function of the EQDSK and the
    # reference profiles, and both are the same at every point.
    baseline_dir = args.baseline_dir or os.path.join(workroot, "baseline")

    paths["baseline_dir"] = baseline_dir

    solver = {}
    if args.max_iter is not None:
        solver["max_iter"] = args.max_iter

    print(f"=== cheaseBS self-scaled omt/omne scan {stamp} ===")
    print(f"base gfile: {gfile}  (shot {shot})")
    for spec in ("e", "i", "z"):
        print(f"base prof : {base_profiles[spec]}")
    print(f"template  : {args.cheasebs_config}")
    print(f"cheaseBS  : {cheasebs_script}")
    print(f"chease    : {chease_binary}")
    print(f"namelist  : {namelist}")
    print(f"outroot   : {outroot}   (results and records)")
    print(f"workroot  : {workroot}"
          f"{'   (in place)' if args.in_place else '   (scratch; not preserved)'}")
    print(f"baseline  : {baseline_dir}")
    print(f"overrides : {solver or '(template as-is)'}")
    print(f"radii     : {args.analysis_radii or '(q checks skipped)'}")
    print(f"points    : {len(pairs)} -> {', '.join(tag_of(*p) for p in pairs)}")
    print(f"log       : {log_path}")
    print(f"pid       : {os.getpid()}")
    print()

    # The base discharge, harmonized once and reused: every point scales the
    # source profiles, never its predecessor. Its untransformed _tree is what
    # output_gfile hands cheaseBS as the reference set.
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData
    from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics

    staged = stage_base_profiles(base_profiles, os.path.join(outroot, "base_profiles"))
    print(f"staged base profiles under canonical names: {os.path.dirname(staged['e'])}")
    phys_base = DischargePhysics(
        DischargeData(gfile=gfile, profiles=[staged[s] for s in ("e", "i", "z")]))

    midped, topped = args.rhot_midped, args.rhot_topped
    if args.auto_pedestal:
        midped, topped = detect_pedestal(phys_base)
        print(f"auto pedestal: rhot_topped={topped:.4f}, rhot_midped={midped:.4f} "
              f"(steepest ne gradient; check this)")
    if not 0.0 < topped < midped < 1.0:
        raise SystemExit(f"pedestal window is not ordered: topped={topped}, "
                         f"midped={midped}; both must lie in (0, 1) with top < mid")
    print(f"pedestal  : topped={topped:.4f}, midped={midped:.4f}")
    print(f"base a/L  : {json.dumps(gradient_report(phys_base), default=str)}")
    print()

    if args.dry_run:
        for omt, omn in pairs:
            phys = (phys_base
                    .apply_omt(alpha=omt, rhot_midped=midped, rhot_topped=topped)
                    .apply_omne(alpha=omn, rhot_midped=midped, rhot_topped=topped))
            print(f"{tag_of(omt, omn)}: "
                  f"{json.dumps(gradient_report(phys), default=str)}")
        print("\ndry run: no equilibrium was solved")
        return 0

    rows, failed = [], []
    t_camp = time.time()
    for omt, omn in pairs:
        tag = tag_of(omt, omn)
        print(f"--- {tag}  (omt {omt:.2f}, omne {omn:.2f}) ---", flush=True)
        try:
            row = solve(phys_base, omt, omn, midped, topped,
                        os.path.join(workroot, tag), os.path.join(outroot, tag),
                        gfile, args, paths, solver)
        except Exception:
            # One point failing outright must not take the rest of the scan with
            # it; the remaining points are independent hours of work.
            print(f"!!! {tag} RAISED, continuing with the next point")
            traceback.print_exc()
            row = {"tag": tag, "omt": omt, "omne": omn,
                   "error": "runner raised, see traceback in the log"}
        rows.append(row)
        if "error" in row:
            failed.append(tag)
            print(f"    FAILED: {row['error']}", flush=True)
        else:
            print(f"    {row.get('iterations')} iters, {row['wall_s']:.0f}s, "
                  f"converged={row.get('converged')}, accepted={row.get('accepted')}",
                  flush=True)
        # Written after every point, not at the end: a campaign that is killed
        # halfway still leaves a readable record of what it did.
        with open(os.path.join(outroot, "selfscaled_scan.json"), "w") as fh:
            json.dump({"shot": shot, "gfile": gfile, "base_profiles": base_profiles,
                       "staged_profiles": staged, "pairs": [list(p) for p in pairs],
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
    print(f"\n=== {len(rows)} point(s) in {time.time() - t_camp:.0f}s ===")
    print(table)

    rejected = [r["tag"] for r in rows
                if "error" not in r and r.get("accepted") is False]
    if failed:
        print(f"\nFAILED to complete: {', '.join(failed)}")
    if rejected:
        print(f"Completed but REJECTED by the acceptance gate: {', '.join(rejected)}")
        for r in rows:
            if r.get("accepted") is False:
                for reason in r.get("reasons") or []:
                    print(f"  {r['tag']}: {reason}")
    print(f"\nrecord : {os.path.join(outroot, 'selfscaled_scan.json')}")
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
