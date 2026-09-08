#!/usr/bin/env python3
"""Re-run cheaseBS on a hatch run's own files, with a hatch run's own config.

`compare_hatch_runs.py` reads what those runs already produced. This one solves
them again, so a hatch result and one of ours can be compared on the same
inputs instead of on two different solves. It is deliberately *not* the reshape
campaign: no profile transform is applied here, nothing is scaled, and the
solver settings are whatever the hatch config already says. Scaling is the next
step and belongs in a scan script, not in a re-run.

WHAT IT REUSES

The hatch tree does not have the layout TPED's readers expect -- the
reconstruction is `output/g_final.eqdsk`, the iteration log sits under
`output/`, the reference profiles are the bare `profiles_{e,i,z}` at the run
root, and the set actually solved with is a suffixed sibling (`profiles_e_1.3T`,
`profiles_e_1.3n`). `compare_hatch_runs.py` already resolves all of that, and
identifies which suffix a run used by content rather than by name (scoring each
candidate against the `EXPTNZ` cheaseBS wrote). This script imports that
resolution wholesale, so the two cannot disagree about which files a run is.

WHAT IT CHANGES IN THE CONFIG, AND WHY ONLY THAT

The hatch config is the unit of truth and is passed to
`run_chease_iterative_profiles.py` directly rather than through TPED's wrapper,
which would re-derive the paths and reject keys it does not know. Only these are
rewritten, all to absolute paths:

    eqdsk                        <- the resolved source g<shot>.<time>
    electron/deuterium/carbon    <- the chosen profile stem
    reference_*                  <- the reference (bare) stem
    baseline_dir, output_dir     <- inside --outroot, never the hatch tree
    chease_binary, namelist      <- kept if absolute and present, else TPED config

Every solver key -- coordinate, replay representation, mixing, tolerances,
`max_iter`, `amplitude_warmup_iters` -- is carried through untouched unless a
flag overrides it, and the diff against the original config is printed before
anything runs. Relative paths in the original are resolved against the config's
own directory, which is the only anchor that survives; cheaseBS itself resolved
them against a working directory that is not recoverable after the fact.

The baseline is rebuilt into `--outroot` rather than reused in place, because
the reference profiles are what define the `p_fast` split and a decomposition
carried over from another directory cannot be checked against them. `--reuse-
baseline` points at the hatch baseline instead, for comparing against exactly
the footing that run used.

    # what is there, and which profile set each run solved with. Solves nothing.
    python run_hatch_cheasebs.py --list

    # re-run the run's own profile set
    python run_hatch_cheasebs.py --outroot $SCRATCH/hatch_rerun

    # the 1.3T and 1.3n sets explicitly, and the unscaled reference
    python run_hatch_cheasebs.py --stem 1.3T --stem 1.3n --stem "" \
        --outroot $SCRATCH/hatch_rerun

    # same inputs, our own solver settings on top
    python run_hatch_cheasebs.py --stem 1.3T --max-iter 25 --warmup 0 \
        --set bootstrap_mix=0.05 --set istar_mix=0.02

Nothing is ever written inside --root.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMPARE = os.path.join(os.path.dirname(HERE), "compare_hatch_runs.py")

# The tree Hatch handed over. Overridable, but the default is the path this
# script was written against so the common case needs no flags.
DEFAULT_ROOT = "/pscratch/sd/j/joeschm/cheaseBS_hatch_results/for_joey/test"

PROFILE_KEYS = {"e": "electron_profile", "i": "deuterium_profile",
                "z": "carbon_profile"}
DRIVER = "run_chease_iterative_profiles.py"
NSTX_NAMELIST = "chease_namelist_nstx"


def load_compare():
    """`compare_hatch_runs` imported by path, for its resolution helpers."""
    if not os.path.isfile(COMPARE):
        raise SystemExit(f"cannot find {COMPARE}")
    spec = importlib.util.spec_from_file_location("compare_hatch_runs", COMPARE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tped_paths():
    """(cheasebs_dir, chease_binary, nstx_namelist) from the TPED user config."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
        from TPED.config.config_helper import Config
        cfg = Config()
        cheasebs = cfg.get_path("CHEASEBS_PATH") or ""
        chease = cfg.get_path("CHEASE_PATH") or ""
    except Exception:
        return "", "", ""
    binary = os.path.join(chease, "src-f90", "chease") if chease else ""
    namelist = os.path.join(cheasebs, NSTX_NAMELIST) if cheasebs else ""
    return cheasebs, binary, namelist


def parse_set(items):
    """--set key=value pairs, JSON-decoded so numbers and booleans stay typed."""
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set wants key=value (got {item!r})")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


def stem_label(stem):
    return stem.lstrip("_") or "reference"


def build_config(spec, cm, stem, out_dir, baseline_dir, overrides, tped):
    """(config dict, list of (key, old, new)) for one re-run.

    The hatch config is copied and only the keys above are replaced, so a diff
    of two dicts is the honest record of what this script did to it.
    """
    cheasebs_dir, chease_binary, namelist = tped
    with open(spec["config_path"]) as fh:
        cfg = json.load(fh)
    original = dict(cfg)

    stems = cm.candidate_stems(spec["root"])
    if stem not in stems:
        raise SystemExit(
            "run %s has no profiles_{e,i,z}%s set (candidates: %s)"
            % (spec["name"], stem,
               ", ".join(stem_label(s) for s in sorted(stems)) or "none"))

    cfg["eqdsk"] = spec["gfile_before"]
    for sp, key in PROFILE_KEYS.items():
        cfg[key] = stems[stem][sp]
        ref = (spec["profiles_before"] or {}).get(sp)
        if ref:
            cfg["reference_" + key] = ref
    cfg["output_dir"] = out_dir
    cfg["baseline_dir"] = baseline_dir
    # An empty baseline directory has nothing to reuse, so the decomposition has
    # to be built; pointing at the hatch baseline is the --reuse-baseline case
    # and leaves the flag as the config had it.
    if not os.path.isdir(baseline_dir) or not os.listdir(baseline_dir):
        cfg["rebuild_baseline"] = True

    for key, resolved in (("chease_binary", chease_binary),
                          ("chease_namelist", namelist)):
        current = cfg.get(key) or ""
        if os.path.isabs(current) and os.path.isfile(current):
            continue
        if resolved and os.path.isfile(resolved):
            cfg[key] = resolved
    cfg.update(overrides)

    diff = [(k, original.get(k, "(absent)"), cfg[k])
            for k in cfg if original.get(k) != cfg[k]]
    return cfg, sorted(diff)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help=f"hatch results tree, or one run root (default {DEFAULT_ROOT})")
    ap.add_argument("--run", action="append", default=[], metavar="NAME",
                    help="run directory under --root; repeatable, default all")
    ap.add_argument("--stem", action="append", default=[], metavar="SUFFIX",
                    help="profile set to solve: 1.3T, 1.3n, \"\" for the "
                         "reference set, or 'all'. Repeatable. Default is the "
                         "set the original run was solved with")
    ap.add_argument("--outroot", default=None,
                    help="where the re-runs are written (default "
                         "$SCRATCH/hatch_rerun/<stamp>, cwd if SCRATCH is unset)")
    ap.add_argument("--reuse-baseline", action="store_true",
                    help="point baseline_dir at the hatch run's own baseline "
                         "instead of rebuilding one in --outroot")
    ap.add_argument("--config", default=None,
                    help="use this cheaseBS config instead of the one found at "
                         "the run root")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None, metavar="N",
                    help="amplitude_warmup_iters")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override any config key; value is JSON-decoded")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the EXPTNZ content match when deciding which "
                         "profile set a run used")
    ap.add_argument("--list", action="store_true",
                    help="resolve and report, write nothing, solve nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the configs and print the commands, solve nothing")
    args = ap.parse_args(argv)

    cm = load_compare()
    root = os.path.abspath(os.path.expandvars(os.path.expanduser(args.root)))
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")
    # A run root carries its own gfile or output/; anything else is a parent.
    if cm.find_source_gfile(root) or os.path.isdir(os.path.join(root, "output")):
        runs = [root]
    else:
        runs = cm.discover_runs(root, args.run)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if args.outroot:
        outroot = os.path.abspath(os.path.expandvars(os.path.expanduser(args.outroot)))
    elif os.environ.get("SCRATCH"):
        outroot = os.path.abspath(os.path.join(os.environ["SCRATCH"],
                                               "hatch_rerun", stamp))
    elif args.list:
        # --list writes nothing, so an unresolvable root is only a display
        # string here and must not stop the report.
        outroot = os.path.join("$SCRATCH", "hatch_rerun", stamp)
    else:
        # Never the working directory: this script is normally invoked from
        # inside the repo, and a run tree written there gets committed by
        # accident.
        raise SystemExit("SCRATCH is unset, so there is no default output root. "
                         "Pass --outroot explicitly (or --list, which writes "
                         "nothing).")

    overrides = parse_set(args.set)
    if args.max_iter is not None:
        overrides["max_iter"] = args.max_iter
    if args.warmup is not None:
        overrides["amplitude_warmup_iters"] = args.warmup

    tped = tped_paths()
    print(f"=== hatch re-run {stamp} ===")
    print(f"root    : {root}")
    print(f"runs    : {', '.join(os.path.basename(r) for r in runs)}")
    print(f"outroot : {outroot}")
    if overrides:
        print(f"override: {overrides}")
    print()

    jobs, failed = [], 0
    for run in runs:
        spec = cm.resolve_run(run, verify=not args.no_verify)
        cm.report(spec)

        if args.config:
            spec["config_path"] = os.path.abspath(args.config)
        if not spec.get("config_path"):
            print(f"  SKIPPED: no cheaseBS config at {run} and none passed "
                  f"with --config", file=sys.stderr)
            failed += 1
            continue
        if not spec.get("gfile_before"):
            print(f"  SKIPPED: no source g<shot>.<time> at {run}", file=sys.stderr)
            failed += 1
            continue

        stems = cm.candidate_stems(spec["root"])
        if args.stem and "all" in args.stem:
            wanted = sorted(stems)
        elif args.stem:
            wanted = ["" if s in ("", "reference") else
                      (s if s.startswith("_") else "_" + s) for s in args.stem]
        else:
            # The set this run was solved with, as resolved above -- reproducing
            # the hatch run is the default, scanning is opt-in.
            after = (spec.get("profiles_after") or {}).get("e", "")
            base = os.path.basename(after)
            wanted = [base[len("profiles_e"):]] if base else [""]

        for stem in wanted:
            label = stem_label(stem)
            out_dir = os.path.join(outroot, f"{spec['name']}_{label}")
            baseline = (os.path.join(spec["root"], "baseline")
                        if args.reuse_baseline
                        else os.path.join(outroot, f"{spec['name']}_baseline"))
            cfg, diff = build_config(spec, cm, stem, out_dir, baseline,
                                     overrides, tped)
            print(f"\n--- {spec['name']} / {label} ---")
            print(f"  output_dir   : {out_dir}")
            print(f"  baseline_dir : {baseline}"
                  f"{'   (hatch run, reused)' if args.reuse_baseline else '   (rebuilt)'}")
            print("  config changes vs %s:" % os.path.basename(spec["config_path"]))
            for key, old, new in diff:
                print(f"    {key}: {old} -> {new}")
            same = [sp for sp, key in PROFILE_KEYS.items()
                    if cfg.get(key) and cfg.get("reference_" + key)
                    and os.path.realpath(cfg[key])
                    == os.path.realpath(cfg["reference_" + key])]
            if same:
                # Legitimate as a null test, and the documented way a scan comes
                # out invisible -- so it is labelled either way.
                print("  ! run profiles ARE the reference profiles (%s): the "
                      "baseline decomposition is built from the same files, so "
                      "p_fast absorbs any difference. Correct only as an "
                      "identity check." % ",".join(same))

            if args.list:
                continue
            os.makedirs(out_dir, exist_ok=True)
            cfg_path = os.path.join(out_dir, "cheasebs_run_config.json")
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=4)
            # A copy of what the run was launched against, next to the config
            # this script derived, so the pair is auditable after the fact.
            shutil.copy2(spec["config_path"],
                         os.path.join(out_dir, "hatch_original_config.json"))
            jobs.append((f"{spec['name']}/{label}", cfg_path, cfg))

    if args.list:
        return 1 if failed else 0

    driver = os.path.join(tped[0] or "", DRIVER)
    if not os.path.isfile(driver):
        print(f"\ncheaseBS driver not found ({driver or 'CHEASEBS_PATH unset'}); "
              f"configs written, nothing run.", file=sys.stderr)
        for name, cfg_path, _ in jobs:
            print(f"  python <cheaseBS>/{DRIVER} --config {cfg_path}")
        return 1

    for name, cfg_path, _ in jobs:
        cmd = [sys.executable, "-u", driver, "--config", cfg_path]
        print(f"\n=== solving {name} ===\n{' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode:
            failed += 1
            print(f"  {name}: driver exited {result.returncode}", file=sys.stderr)

    print(f"\n{len(jobs) - failed} solved, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
