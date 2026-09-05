#!/usr/bin/env python3
"""Replay a pre-scaled omt/omne profile scan through cheaseBS.

The scan directory this drives was produced by the older IFS omn/omt scaling
code, not by TPED: the profiles already exist on disk as GENE `profiles_{e,i,z}`
files, one set per (omt, omne) point, beside the base EQDSK they were scaled
from. So there is nothing to reshape here -- every point is a straight cheaseBS
solve on files that are already written, and this script is the batch wrapper
that turns that directory into a run per point plus one campaign record.

Expected input layout (`--case-dir`), e.g.
/data/DIIID/DIIID162940/DIIID162940/chease:

    chease_namelist                     <- used if present, else the cheaseBS one
    g162940.02944_670                   <- base EQDSK, auto-discovered
    omt0p7_omne0p7/
        profiles_e_omt0p7_omne0p7       <- the active profiles for that point
        profiles_i_omt0p7_omne0p7
        profiles_z_omt0p7_omne0p7
    omt0p7_omne0p8/
        ...

`p` in a directory name is a decimal point: `omt0p7_omne1p1` is omt 0.7, omne 1.1.

REFERENCE PROFILES -- the thing to get right
--------------------------------------------
cheaseBS takes two profile sets and they are not interchangeable. The
`reference_*_profile` set must describe the plasma the base EQDSK *already
contains*: it builds the fast-pressure and bootstrap/driven-current
decomposition that the perturbation is then measured against. The
`*_profile` set is the scaled profiles CHEASE actually solves with.

Aliasing the two -- pointing the reference at the scaled profiles -- makes
p_fast swell by exactly what the scaling removed, so p_total is floored at the
source pressure and every downward point silently reproduces its own baseline.
That is a null test, not a scan, and it is invisible in the output. See
"Reference profiles" in TPED's cheasebs_runner for the measured version.

So the reference set defaults to cheaseBS's own bundled, validated
`data/profiles/profiles_162940_{e,i,z}` -- the profiles the DIII-D regression
tests use against this exact EQDSK. Override with `--reference-dir` /
`--reference-stem`, or use `--reference-variation omt1p0_omne1p0` to take the
unity point of the scan itself as the reference. The script refuses to alias the
reference onto a scaled point unless `--allow-aliased-reference` is passed.

WHERE IT RUNS -- scratch, then copy back
----------------------------------------
Same split as every other cheaseBS caller in this repo (see `output_gfile` in
TPED's discharge_io). cheaseBS itself runs against a timestamped directory under
the configured scratch path (TPED `OUTPUT_PATH`), because the run tree -- the
baseline decomposition plus per-iteration CHEASE artifacts for every point -- is
bulky, transient working data. Only the result and the record are copied back
into `--outroot`:

    <outroot>/<tag>/EQDSK*.OUT              the reconstruction
                   /profiles_{e,i,z}        the scaled profiles it solved with
                   /reference_profiles/REF_profiles_{e,i,z}
                   /cheasebs_run_config.json
                   /convergence_summary.json
                   /cheasebs_acceptance.json
                   /iteration_log.csv
                   /iteration_errors.png
                   /run_summary.png

The profile copies are the reason this is worth doing rather than just keeping
paths: TPED's `resolve_run_files` prefers a run directory's own local copies
over the absolute paths in the config, so a directory carrying them re-plots
correctly years later, after scratch is purged and after the case directory has
moved. Which profiles the baseline was built from is the difference between a
real scan point and a null test, and it is not recoverable from the gfile alone.
`iteration_log.csv` travels for the same reason: `convergence_summary.json` keeps
only the final scalars, so without it "did the loop settle, or did it just hit
max_iter" is unanswerable.

`--in-place` skips scratch and runs directly under `--outroot`, which is only
worth it when scratch is unavailable.

The baseline decomposition depends only on (EQDSK, reference profiles), which
are identical across points, so it is built once into a single scratch directory
and reused by every solve. That is one CHEASE preprocessing pass for the
campaign instead of one per point. It stays in scratch: it is rebuildable, and
its own CHEASE artifacts are most of the bulk.

USAGE

    # everything, detached, log to the campaign file
    cd <repo>/reshape_convergence/DIIID_verify
    nohup python -u run_cheasebs_scaling_scan.py \
        --case-dir /data/DIIID/DIIID162940/DIIID162940/chease \
        > /dev/null 2>&1 &
    tail -f runs_162940_omn_omt_scaling/campaign_*.log

    # one point first, which is what to do before committing the whole scan
    python -u run_cheasebs_scaling_scan.py \
        --case-dir /data/DIIID/DIIID162940/DIIID162940/chease \
        --only omt1p0_omne1p0

    # see what it would run, touching nothing
    python run_cheasebs_scaling_scan.py --case-dir ... --dry-run

`-u` matters: without it Python block-buffers stdout when it is not a terminal
and the log stays empty for hours.

Exit status is 0 only when every solve completed AND every point was accepted,
so a wrapper can tell "ran" from "worked". `--no-gate` drops the acceptance
requirement from the exit status but still records the verdict.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

# cheaseBS renders per-run PNGs and must not reach for a display.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))

# The DIII-D validated solver path, from cheaseBS's diiid_base_case.json: the
# j_parallel = <j.B>/B0 replay on rhot with QSPEC enforced. This is NOT the NSTX
# I* path -- do not copy the istar_* / mixing settings from the NSTX configs in
# here, they belong to a different replay representation.
SOLVER_DEFAULTS = {
    "coordinate": "rhot",
    "replay_representation": "jparallel",
    "enforce_qspec": True,
    "initial_amplitude": 1.0,
    "max_iter": 8,
    "tol_ip_rel": 0.001,
    "tol_bs": 0.001,
    "tol_q": 0.001,
    "tol_a": 0.0001,
}

TAG_RE = re.compile(r"^omt(\d+p\d+)_omne(\d+p\d+)$")

# What travels from the scratch run directory into the permanent one. The final
# EQDSK and the profile copies are handled separately; these are the files
# cheaseBS and the scorer write under their own names.
COPY_BACK = (
    "cheasebs_run_config.json",
    "convergence_summary.json",
    "cheasebs_acceptance.json",
    "iteration_log.csv",
    "iteration_errors.png",
    "run_summary.png",
)


class Tee:
    """stdout to the terminal and the log file at once, line-flushed.

    Everything downstream prints -- cheaseBS's own per-iteration echo included --
    lands in the log without those modules knowing about it. Flushing on every
    write is what makes `tail -f` useful on a run this slow.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def parse_tag(name):
    """'omt0p7_omne1p1' -> (0.7, 1.1), or None when the name is not a scan point."""
    m = TAG_RE.match(name)
    if not m:
        return None
    return tuple(float(g.replace("p", ".")) for g in m.groups())


def find_base_gfile(case_dir):
    """The base EQDSK at the case-dir root: g<shot>.<time>, one of them.

    Restricted to the root so a per-point `.eqdsk` written by the scaling code
    can never be mistaken for the source equilibrium.
    """
    cands = sorted(p for p in glob.glob(os.path.join(case_dir, "g[0-9]*"))
                   if os.path.isfile(p))
    if not cands:
        raise SystemExit(
            f"No base gfile (g<shot>.<time>) found at the root of {case_dir}. "
            f"Pass one explicitly with --gfile."
        )
    if len(cands) > 1:
        raise SystemExit(
            "Multiple candidate gfiles at the case-dir root; name one with --gfile:\n  "
            + "\n  ".join(cands)
        )
    return cands[0]


def shot_of(gfile):
    """'g162940.02944_670' -> '162940'. Used only to name the output root."""
    m = re.match(r"^g(\d+)\.", os.path.basename(gfile))
    return m.group(1) if m else "unknown"


def discover_points(case_dir, only=None):
    """The scan points, as [(tag, omt, omne, {spec: profile path})], sorted.

    A directory missing any of the three species is reported and skipped rather
    than failing the campaign: cheaseBS needs all three, and one malformed point
    is not a reason to drop the other twelve.
    """
    points, skipped = [], []
    for path in sorted(glob.glob(os.path.join(case_dir, "omt*_omne*"))):
        if not os.path.isdir(path):
            continue
        tag = os.path.basename(path)
        parsed = parse_tag(tag)
        if parsed is None:
            skipped.append((tag, "directory name does not parse as omt<x>_omne<y>"))
            continue
        if only and tag not in only:
            continue
        profiles, missing = {}, []
        for spec in ("e", "i", "z"):
            p = os.path.join(path, f"profiles_{spec}_{tag}")
            if os.path.isfile(p):
                profiles[spec] = p
            else:
                missing.append(os.path.basename(p))
        if missing:
            skipped.append((tag, f"missing {', '.join(missing)}"))
            continue
        points.append((tag, parsed[0], parsed[1], profiles))
    return points, skipped


def resolve_reference(args, case_dir, cheasebs_dir):
    """{spec: path} for the reference profiles, plus a line describing the choice.

    Precedence: explicit --reference-dir, then --reference-variation (a point of
    this scan), then cheaseBS's bundled profiles for this shot.
    """
    if args.reference_dir:
        ref_dir, stem = args.reference_dir, args.reference_stem
        why = "explicit --reference-dir"
    elif args.reference_variation:
        ref_dir = os.path.join(case_dir, args.reference_variation)
        stem = f"profiles_{{spec}}_{args.reference_variation}"
        why = f"--reference-variation {args.reference_variation} (unity point of this scan)"
    else:
        ref_dir = os.path.join(cheasebs_dir, "data", "profiles")
        stem = args.reference_stem
        why = "cheaseBS bundled reference profiles (DIII-D regression set)"

    refs, missing = {}, []
    for spec in ("e", "i", "z"):
        name = (stem.format(spec=spec) if "{spec}" in stem
                else f"{stem}_{spec}")
        p = os.path.join(ref_dir, name)
        refs[spec] = p
        if not os.path.isfile(p):
            missing.append(p)
    if missing:
        raise SystemExit(
            "Reference profiles not found:\n  " + "\n  ".join(missing)
            + "\n\nThese must describe the plasma the base EQDSK already contains "
              "(see the module docstring). Point --reference-dir / --reference-stem "
              "at them, or --reference-variation at the unity point of the scan."
        )
    return refs, why


def build_config(point, refs, gfile, baseline_dir, out_dir, chease_binary,
                 chease_namelist, solver):
    """The cheaseBS JSON config for one point. Keys are the driver's own set."""
    _tag, _omt, _omne, profiles = point
    cfg = {
        "eqdsk": os.path.abspath(gfile),
        "electron_profile": os.path.abspath(profiles["e"]),
        "deuterium_profile": os.path.abspath(profiles["i"]),
        "carbon_profile": os.path.abspath(profiles["z"]),
        "reference_electron_profile": os.path.abspath(refs["e"]),
        "reference_deuterium_profile": os.path.abspath(refs["i"]),
        "reference_carbon_profile": os.path.abspath(refs["z"]),
        "baseline_dir": os.path.abspath(baseline_dir),
        "output_dir": os.path.abspath(out_dir),
        "chease_binary": os.path.abspath(chease_binary),
        "chease_namelist": os.path.abspath(chease_namelist),
        "overwrite": True,
    }
    cfg.update(solver)
    return cfg


def guard_paths(cfg):
    """cheaseBS rmtree's baseline_dir on a rebuild -- keep it away from live data.

    realpath so NERSC home symlinks (/global/homes -> /global/u1) compare equal.
    """
    def norm(p):
        return os.path.normcase(os.path.realpath(p))

    bdir = norm(cfg["baseline_dir"])
    if bdir == norm(cfg["output_dir"]):
        raise SystemExit("baseline_dir must differ from output_dir: a rebuild would "
                         "delete the run's own outputs.")
    if bdir == norm(os.path.dirname(cfg["eqdsk"])):
        raise SystemExit("baseline_dir must differ from the base gfile's directory: "
                         "a rebuild would delete the gfile before it is read.")
    for key in ("electron_profile", "deuterium_profile", "carbon_profile",
                "reference_electron_profile", "reference_deuterium_profile",
                "reference_carbon_profile"):
        if bdir == norm(os.path.dirname(cfg[key])):
            raise SystemExit(f"baseline_dir must differ from the directory holding "
                             f"{key}: a rebuild would delete the profiles.")


def last_eqdsk(run_dir):
    """The final reconstruction, from the run's own iteration log."""
    log = os.path.join(run_dir, "iteration_log.csv")
    if not os.path.isfile(log):
        return None
    found = None
    with open(log, newline="") as fh:
        for row in csv.DictReader(fh):
            path = (row.get("eqdsk_path") or "").strip()
            if path:
                found = path
    return found


def read_summary(run_dir):
    path = os.path.join(run_dir, "convergence_summary.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def score(eqdsk, gfile, run_dir, cheasebs_script, radii):
    """Acceptance verdict via TPED, or a note explaining why there isn't one.

    The subprocess exiting 0 and an EQDSK existing is not evidence the
    equilibrium is usable, so this runs whenever TPED is importable. It is a
    diagnostic on top of the solve, never allowed to sink a completed run.
    """
    try:
        from TPED.projects.discharge_tools.src.cheasebs_runner import (
            ACCEPTANCE_FILENAME, CheasebsAcceptance, evaluate_acceptance)
    except ImportError as exc:
        return None, f"TPED not importable ({exc}); acceptance not scored"

    try:
        result = evaluate_acceptance(
            eqdsk_path=eqdsk, source_eqdsk=gfile, run_dir=run_dir,
            cheasebs_script=cheasebs_script,
            policy=CheasebsAcceptance.production(analysis_radii=tuple(radii)),
        )
        rec = result.to_dict()
        with open(os.path.join(run_dir, ACCEPTANCE_FILENAME), "w") as fh:
            json.dump(rec, fh, indent=4, sort_keys=True)
        return rec, None
    except Exception as exc:
        return None, f"acceptance scoring raised: {type(exc).__name__}: {exc}"


def render_plots(run_dir, cfg):
    """The end plots, rendered in the run directory before anything is copied.

    Diagnostics on top of a finished solve: never allowed to sink the run, since
    the equilibrium is already written and scored by the time these are drawn.
    run_summary in particular reads every input path back off disk, which is how
    a reconstruction built from the wrong reference profiles becomes visible.
    """
    notes = []
    try:
        from TPED.projects.discharge_tools.src.cheasebs_runner import (
            plot_iteration_errors, plot_run_summary)
    except ImportError as exc:
        return [f"TPED not importable ({exc}); no plots rendered"]

    for name, fn in (("iteration_errors.png", plot_iteration_errors),
                     ("run_summary.png", plot_run_summary)):
        try:
            fn(run_dir, os.path.join(run_dir, name), cfg=cfg, quiet=True)
        except Exception as exc:
            notes.append(f"{name} failed: {type(exc).__name__}: {exc}")
    return notes


def copy_back(run_dir, final_dir, eqdsk, cfg):
    """Move the result and the record out of scratch into the permanent dir.

    The profiles are copied under their canonical stems rather than left as
    paths, because TPED's resolve_run_files prefers a run directory's own local
    copies over the config's absolute paths -- so a directory carrying them
    re-plots correctly after scratch is purged or the case directory moves.
    """
    os.makedirs(final_dir, exist_ok=True)
    copied = []

    if eqdsk and os.path.isfile(eqdsk):
        dst = os.path.join(final_dir, os.path.basename(eqdsk))
        shutil.copy2(eqdsk, dst)
        copied.append(os.path.basename(dst))

    for fname in COPY_BACK:
        src = os.path.join(run_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, fname))
            copied.append(fname)

    # The scaled profiles this point solved with, and the reference set the
    # baseline was built from. Both are kilobytes of text and they are the only
    # record of which is which once the config's absolute paths go stale.
    for spec, longname in (("e", "electron"), ("i", "deuterium"), ("z", "carbon")):
        src = cfg.get(f"{longname}_profile")
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, f"profiles_{spec}"))
            copied.append(f"profiles_{spec}")

    ref_dir = os.path.join(final_dir, "reference_profiles")
    for spec, longname in (("e", "electron"), ("i", "deuterium"), ("z", "carbon")):
        src = cfg.get(f"reference_{longname}_profile")
        if src and os.path.isfile(src):
            os.makedirs(ref_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(ref_dir, f"REF_profiles_{spec}"))
            copied.append(f"reference_profiles/REF_profiles_{spec}")

    return copied


def solve(point, cfg, cheasebs_script, gfile, radii, final_dir):
    """One point: write the config, run cheaseBS, score it, plot it, copy it back.

    cfg["output_dir"] is where cheaseBS actually works (scratch, normally);
    final_dir is the permanent directory the result is copied into. They are the
    same directory under --in-place, and copy_back then no-ops on itself.
    """
    tag, omt, omne, _profiles = point
    run_dir = cfg["output_dir"]
    os.makedirs(run_dir, exist_ok=True)
    row = {"tag": tag, "omt": omt, "omne": omne,
           "run_dir": run_dir, "final_dir": final_dir}

    config_path = os.path.join(run_dir, "cheasebs_run_config.json")
    with open(config_path, "w") as fh:
        json.dump(cfg, fh, indent=4)

    t0 = time.time()
    rc = subprocess.Popen(
        [sys.executable, "-u", cheasebs_script, "--config", config_path]).wait()
    row["wall_s"] = time.time() - t0
    row["returncode"] = rc
    if rc != 0:
        row["error"] = f"cheaseBS exited {rc}"
        return row

    summary = read_summary(run_dir)
    row["iterations"] = summary.get("iterations")
    row["converged"] = summary.get("final_converged", summary.get("converged"))
    row["final_ip_a"] = summary.get("final_ip_a")
    row["target_ip_a"] = summary.get("target_ip_a")

    eqdsk = last_eqdsk(run_dir)
    row["eqdsk"] = eqdsk
    if not eqdsk:
        row["error"] = "cheaseBS exited 0 but iteration_log.csv has no eqdsk_path"
        return row

    rec, note = score(eqdsk, gfile, run_dir, cheasebs_script, radii)
    if note:
        row["acceptance_note"] = note
        print(f"    {note}")
    if rec:
        row["accepted"] = rec.get("accepted")
        row["reasons"] = rec.get("reasons")
        row["ip_error_rel"] = rec.get("ip_error_rel")
        row["q_errors_rel"] = rec.get("q_errors_rel")
        row["q_edge_error_rel"] = rec.get("q_edge_error_rel")

    for plot_note in render_plots(run_dir, cfg):
        row.setdefault("plot_notes", []).append(plot_note)
        print(f"    {plot_note}")

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


def text_table(rows):
    """The campaign result as plain text -- a log is the only place it is read."""
    cols = [("tag", "{}", 18), ("omt", "{:.2f}", 6), ("omne", "{:.2f}", 6),
            ("iters", "{}", 6), ("conv", "{}", 6), ("acc", "{}", 6),
            ("Ip_err", "{:.2%}", 9), ("q@x0", "{:.2%}", 9),
            ("wall_s", "{:.0f}", 8), ("status", "{}", 8)]
    lines = ["  " + "".join(name.rjust(w) for name, _, w in cols)]
    for r in rows:
        q = r.get("q_errors_rel") or {}
        vals = {
            "tag": r["tag"], "omt": r["omt"], "omne": r["omne"],
            "iters": r.get("iterations"), "conv": r.get("converged"),
            "acc": r.get("accepted"), "Ip_err": r.get("ip_error_rel"),
            "q@x0": max((abs(v) for v in q.values() if v is not None), default=None),
            "wall_s": r.get("wall_s"),
            "status": "FAILED" if "error" in r else "",
        }
        cells = []
        for name, fmt, w in cols:
            v = vals.get(name)
            try:
                s = "--" if v is None else fmt.format(v)
            except (TypeError, ValueError):
                s = str(v)
            cells.append(s.rjust(w))
        lines.append("  " + "".join(cells))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run cheaseBS over a pre-scaled omt/omne profile scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case-dir", required=True,
                    help="scan directory holding the base gfile and the omt*_omne* points")
    ap.add_argument("--gfile", default=None,
                    help="base EQDSK; default is the single g<shot>.* at the case-dir root")
    ap.add_argument("--outroot", default=None,
                    help="where per-point run directories go "
                         "(default: <this dir>/runs_<shot>_omn_omt_scaling)")
    ap.add_argument("--only", nargs="+", default=None, metavar="TAG",
                    help="run only these points, e.g. --only omt1p0_omne1p0")

    ap.add_argument("--reference-dir", default=None,
                    help="directory holding the reference profiles; see the module "
                         "docstring before setting this")
    ap.add_argument("--reference-variation", default=None, metavar="TAG",
                    help="use a point of this scan as the reference set, "
                         "e.g. omt1p0_omne1p0")
    ap.add_argument("--reference-stem", default="profiles_162940",
                    help="reference filename stem; '<stem>_e' or a pattern "
                         "containing {spec}")
    ap.add_argument("--allow-aliased-reference", action="store_true",
                    help="permit a point to use its own scaled profiles as the "
                         "reference. This makes that point a null test")

    ap.add_argument("--scratch-root", default=None,
                    help="where cheaseBS actually runs (default: TPED OUTPUT_PATH). "
                         "Only the result and the record are copied to --outroot")
    ap.add_argument("--in-place", action="store_true",
                    help="run directly under --outroot instead of scratch, keeping "
                         "the full per-iteration tree there")

    ap.add_argument("--chease-binary", default=None,
                    help="CHEASE executable (default: TPED CHEASE_PATH/src-f90/chease)")
    ap.add_argument("--cheasebs-dir", default=None,
                    help="cheaseBS repo (default: TPED CHEASEBS_PATH)")
    ap.add_argument("--chease-namelist", default=None,
                    help="namelist template (default: the case dir's own "
                         "chease_namelist, else the cheaseBS one)")

    ap.add_argument("--max-iter", type=int, default=SOLVER_DEFAULTS["max_iter"])
    ap.add_argument("--tol-ip-rel", type=float, default=SOLVER_DEFAULTS["tol_ip_rel"])
    ap.add_argument("--tol-bs", type=float, default=SOLVER_DEFAULTS["tol_bs"])
    ap.add_argument("--tol-q", type=float, default=SOLVER_DEFAULTS["tol_q"])
    ap.add_argument("--coordinate", default=SOLVER_DEFAULTS["coordinate"],
                    choices=["rhot", "rhop"])
    ap.add_argument("--replay-representation",
                    default=SOLVER_DEFAULTS["replay_representation"],
                    choices=["jparallel", "istar"])
    ap.add_argument("--no-enforce-qspec", action="store_true",
                    help="drop the QSPEC constraint (it is on by default for DIII-D)")
    ap.add_argument("--rebuild-baseline", action="store_true",
                    help="force a fresh baseline decomposition even if one exists")

    ap.add_argument("--analysis-radii", type=float, nargs="*", default=[],
                    help="rho_tor values where q is scored; empty skips the q checks")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not let a rejected point set a non-zero exit status")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the first config, run nothing")
    ap.add_argument("--log", default=None,
                    help="log file (default: <outroot>/campaign_<stamp>.log)")
    args = ap.parse_args(argv)

    case_dir = os.path.abspath(os.path.expanduser(args.case_dir))
    if not os.path.isdir(case_dir):
        raise SystemExit(f"--case-dir does not exist: {case_dir}")

    gfile = os.path.abspath(args.gfile) if args.gfile else find_base_gfile(case_dir)
    if not os.path.isfile(gfile):
        raise SystemExit(f"base gfile not found: {gfile}")
    shot = shot_of(gfile)

    # cheaseBS / CHEASE locations come from the TPED user config when they are
    # not given, so this script agrees with every other cheaseBS caller in the
    # repo about which binary and which driver are "the" ones.
    cheasebs_dir, chease_binary = args.cheasebs_dir, args.chease_binary
    scratch_root = args.scratch_root
    if cheasebs_dir is None or chease_binary is None or (
            scratch_root is None and not args.in_place):
        try:
            from TPED.config.config_helper import Config
            cfg_paths = Config()
            cheasebs_dir = cheasebs_dir or cfg_paths.get_path("CHEASEBS_PATH")
            chease_binary = chease_binary or os.path.join(
                cfg_paths.get_path("CHEASE_PATH") or "", "src-f90", "chease")
            scratch_root = scratch_root or cfg_paths.get_path("OUTPUT_PATH")
        except Exception as exc:
            raise SystemExit(
                f"Could not read TPED config for the CHEASE paths ({exc}). "
                f"Pass --cheasebs-dir, --chease-binary and --scratch-root explicitly."
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

    # The scan directory ships its own namelist; it describes this equilibrium
    # and takes precedence over the repo template.
    namelist = args.chease_namelist
    if namelist is None:
        local = os.path.join(case_dir, "chease_namelist")
        namelist = local if os.path.isfile(local) else os.path.join(
            cheasebs_dir, "chease_namelist")
    if not os.path.isfile(namelist):
        raise SystemExit(f"CHEASE namelist not found: {namelist}")

    outroot = os.path.abspath(
        args.outroot or os.path.join(HERE, f"runs_{shot}_omn_omt_scaling"))

    # cheaseBS works in scratch and only the result travels back, matching every
    # other cheaseBS caller in this repo. The per-iteration CHEASE tree for a
    # thirteen-point scan is the bulk, and it is regenerable; --outroot is meant
    # to hold the equilibria and the records, not the working files.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if args.in_place:
        workroot = outroot
    else:
        if not scratch_root:
            raise SystemExit(
                "No scratch root: set OUTPUT_PATH in the TPED user config, pass "
                "--scratch-root, or use --in-place to run under --outroot."
            )
        workroot = os.path.join(os.path.abspath(os.path.expanduser(scratch_root)),
                                "cheaseBS_runs", f"{stamp}-{shot}-omn_omt_scaling")
    # The baseline is shared by every point and stays where the work happens: it
    # is rebuildable, and its own CHEASE artifacts are most of the bulk.
    baseline_dir = os.path.join(workroot, "baseline")

    points, skipped = discover_points(case_dir, only=set(args.only) if args.only else None)
    if not points:
        raise SystemExit(f"No usable omt*_omne* points found under {case_dir}")

    refs, ref_why = resolve_reference(args, case_dir, cheasebs_dir)
    aliased = [t for t, _o, _n, prof in points
               if any(os.path.realpath(prof[s]) == os.path.realpath(refs[s])
                      for s in ("e", "i", "z"))]
    if aliased and not args.allow_aliased_reference:
        raise SystemExit(
            f"Point(s) {', '.join(aliased)} would use their own scaled profiles as "
            f"the reference set. That floors p_total at the source pressure and makes "
            f"the point a null test (see the module docstring). Pass "
            f"--allow-aliased-reference if that is what you want."
        )

    solver = dict(SOLVER_DEFAULTS,
                  coordinate=args.coordinate,
                  replay_representation=args.replay_representation,
                  enforce_qspec=not args.no_enforce_qspec,
                  max_iter=args.max_iter, tol_ip_rel=args.tol_ip_rel,
                  tol_bs=args.tol_bs, tol_q=args.tol_q)
    # The baseline is a function of (EQDSK, reference profiles) only, so it is
    # identical for every point. Build it on the first solve and reuse it: the
    # driver builds it itself when baseline_dir is incomplete, and rebuilding it
    # per point would be one wasted CHEASE preprocessing pass each.
    solver["rebuild_baseline"] = bool(args.rebuild_baseline)

    os.makedirs(outroot, exist_ok=True)
    log_path = args.log or os.path.join(outroot, f"campaign_{stamp}.log")
    if not args.dry_run:
        tee = Tee(sys.stdout, log_path)
        sys.stdout = tee
        sys.stderr = tee

    print(f"=== cheaseBS omt/omne scaling scan {stamp} ===")
    print(f"case dir  : {case_dir}")
    print(f"base gfile: {gfile}  (shot {shot})")
    print(f"reference : {ref_why}")
    for spec in ("e", "i", "z"):
        print(f"            {refs[spec]}")
    print(f"cheaseBS  : {cheasebs_script}")
    print(f"chease    : {chease_binary}")
    print(f"namelist  : {namelist}")
    print(f"outroot   : {outroot}   (results and records)")
    print(f"workroot  : {workroot}"
          f"{'   (in place)' if args.in_place else '   (scratch; not preserved)'}")
    print(f"baseline  : {baseline_dir}")
    print(f"solver    : {solver}")
    print(f"radii     : {args.analysis_radii or '(q checks skipped)'}")
    print(f"points    : {len(points)} -> {', '.join(t for t, _, _, _ in points)}")
    for tag, why in skipped:
        print(f"  SKIPPED {tag}: {why}")
    print(f"log       : {log_path}")
    print(f"pid       : {os.getpid()}")
    print()

    final_dirs = [os.path.join(outroot, p[0]) for p in points]
    configs = [build_config(p, refs, gfile, baseline_dir,
                            os.path.join(workroot, p[0]),
                            chease_binary, namelist, solver)
               for p in points]
    guard_paths(configs[0])

    if args.dry_run:
        print("--- dry run: config for the first point ---")
        print(json.dumps(configs[0], indent=4))
        return 0

    rows, failed = [], []
    t_camp = time.time()
    for point, cfg, final_dir in zip(points, configs, final_dirs):
        tag = point[0]
        print(f"--- {tag}  (omt {point[1]:.2f}, omne {point[2]:.2f}) ---", flush=True)
        try:
            row = solve(point, cfg, cheasebs_script, gfile, args.analysis_radii,
                        final_dir)
        except Exception:
            # One point failing outright must not take the rest of the scan with
            # it; the remaining points are independent hours of work.
            print(f"!!! {tag} RAISED, continuing with the next point")
            traceback.print_exc()
            row = {"tag": tag, "omt": point[1], "omne": point[2],
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
        with open(os.path.join(outroot, "scaling_scan.json"), "w") as fh:
            json.dump({"shot": shot, "case_dir": case_dir, "gfile": gfile,
                       "reference": refs, "reference_choice": ref_why,
                       "solver": solver, "analysis_radii": args.analysis_radii,
                       "workroot": workroot, "baseline_dir": baseline_dir,
                       "in_place": bool(args.in_place),
                       "skipped": skipped, "rows": rows},
                      fh, indent=1, default=str)
        # Every point after the first reuses the baseline the first one built.
        solver["rebuild_baseline"] = False
        for c in configs:
            c["rebuild_baseline"] = False

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
    print(f"\nrecord : {os.path.join(outroot, 'scaling_scan.json')}")
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
