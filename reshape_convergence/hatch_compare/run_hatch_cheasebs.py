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

THE CONFIG SUPPLIES SOLVER SETTINGS. THE RUN DIRECTORY SUPPLIES PATHS.

Every path key is DISCARDED from the hatch config and rebuilt from the run
directory being solved:

    eqdsk                        <- the g<shot>.<time> at this run root
    electron/deuterium/carbon    <- profiles_{e,i,z}<suffix> at this run root
    reference_*                  <- profiles_{e,i,z} (bare) at this run root
    baseline_dir, output_dir     <- inside --outroot, never the hatch tree
    chease_binary, namelist      <- kept if absolute and present, else TPED config

The config's own path entries are not usable and are not consulted. They are
written relative to a working directory that is not recoverable after the fact
(`test`'s config says `../test/profiles_e_1.3n`, which only resolves because
that config happens to sit in `test/`), and a config borrowed from a sibling
carries paths that resolve into the SIBLING's directory -- `test2` solved with
`../test/profiles_*` would silently reconstruct `test`'s profiles under
`test2`'s name. Discarding them makes that class of error impossible rather
than merely unlikely.

What survives from the config is every solver key -- coordinate, replay
representation, mixing, tolerances, `max_iter`, `amplitude_warmup_iters` --
carried through untouched unless a flag overrides it, with the diff printed
before anything runs.

THE BARE PROFILE SET IS NEVER SOLVED

`profiles_{e,i,z}` feeds `reference_*`, which is what cheaseBS builds the
baseline decomposition and the frozen `p_fast` from. It is not a run. Solving it
hands the same files to both roles: the decomposition gets built from the
profiles being replayed, `p_fast` absorbs the whole difference, and the result is
a reconstruction of the reference that answers nothing this comparison asks. So
`--stem all` means every SCALED set, and naming the bare set is refused. Every
solve still gets a baseline -- it just does not get its own reconstruction run.

One baseline per hatch run, built by the first solve and reused by the rest: it
is a function of the reference profiles and the source EQDSK, and neither
changes between stems. It is built inside `--outroot` rather than read from the
hatch tree so it can be checked against the profiles it claims to come from;
`--reuse-baseline` points at the hatch run's own instead, for reproducing
exactly the footing that run had.

    # what is there, and which profile set each run solved with. Solves nothing.
    python run_hatch_cheasebs.py --list

    # re-run the run's own profile set
    python run_hatch_cheasebs.py --outroot $SCRATCH/hatch_rerun

    # both scaled sets ("all" means every scaled set, never the bare one)
    python run_hatch_cheasebs.py --stem 1.3T --stem 1.3n \
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

# Dropped from the hatch config and rebuilt from the run directory. See the
# module docstring: a config's path entries either do not resolve or resolve
# into the wrong run.
PATH_KEYS = (("eqdsk", "baseline_dir", "output_dir")
             + tuple(PROFILE_KEYS.values())
             + tuple("reference_" + k for k in PROFILE_KEYS.values()))
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


def find_sibling_config(cm, run_root):
    """(path, run name) of a sibling run's config, when exactly one exists.

    Not every hatch run keeps its config: `test2` has none, and its solver
    settings are not recoverable from anything it did write -- cheaseBS's
    `convergence_summary.json` records the coordinate, the replay representation
    and the Ip/Bt targets, but not the mixing, the tolerances, `max_iter` or the
    warm-up. A sibling's config is the only available stand-in, so it is used
    when there is exactly one candidate and refused when there are several,
    rather than picking one silently.
    """
    parent = os.path.dirname(os.path.abspath(run_root))
    found = []
    for entry in sorted(os.listdir(parent)):
        d = os.path.join(parent, entry)
        if not os.path.isdir(d) or os.path.samefile(d, run_root):
            continue
        path, _ = cm.read_run_config(d)
        if path:
            found.append((path, entry))
    if len(found) == 1:
        return found[0]
    return None, None


def settings_from_summary(run_root):
    """Config keys a run recorded about itself, for use over a borrowed config.

    Only the four that cheaseBS actually writes into convergence_summary.json
    and that are also config keys. The Ip/Bt targets are carried only when they
    differ from the source values, since that is the case where they were a
    deliberate override rather than the EQDSK's own numbers.
    """
    path = os.path.join(run_root, "output", "convergence_summary.json")
    if not os.path.isfile(path):
        path = os.path.join(run_root, "convergence_summary.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            summary = json.load(fh)
    except Exception:
        return {}
    out = {}
    for key in ("coordinate", "replay_representation"):
        if summary.get(key):
            out[key] = summary[key]
    for key, source in (("target_ip_a", "source_ip_a"),
                        ("target_bt_t", "source_bt_t")):
        value, src = summary.get(key), summary.get(source)
        if value is not None and src is not None and value != src:
            out[key] = value
    return out


def build_config(spec, stems, stem, out_dir, baseline_dir, overrides, tped,
                 rebuild_baseline=None):
    """(config dict, list of (key, old, new)) for one re-run.

    The hatch config is copied and only the keys above are replaced, so a diff
    of two dicts is the honest record of what this script did to it.
    """
    cheasebs_dir, chease_binary, namelist = tped
    with open(spec["config_path"]) as fh:
        original = json.load(fh)
    # Solver settings only. Every path is rebuilt below from the run directory,
    # so a stale or sibling-relative entry cannot survive into the new config.
    cfg = {k: v for k, v in original.items() if k not in PATH_KEYS}

    if "" not in stems:
        raise ValueError("no bare profiles_{e,i,z} at the run root to use as "
                         "the reference set")
    cfg["eqdsk"] = spec["gfile_before"]
    for sp, key in PROFILE_KEYS.items():
        cfg[key] = stems[stem][sp]
        cfg["reference_" + key] = stems[""][sp]
    cfg["output_dir"] = out_dir
    cfg["baseline_dir"] = baseline_dir
    # The decomposition is a function of the reference profiles and the source
    # EQDSK, so every stem of one run shares it: built by the first solve, read
    # by the rest. Rebuilding per stem is not just wasted time -- cheaseBS
    # rmtree's the directory first, so the later solves delete the baseline the
    # earlier ones built and each pays for it again.
    if rebuild_baseline is not None:
        cfg["rebuild_baseline"] = rebuild_baseline

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
                    help="scaled profile set to solve: 1.3T, 1.3n, or 'all' "
                         "for every scaled set present. Repeatable. Default is "
                         "the set the original run was solved with. The bare "
                         "profiles_{e,i,z} cannot be named: it is the reference "
                         "the baseline is built from, not a run")
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
    ap.add_argument("--no-borrow-config", action="store_true",
                    help="do not fall back to a sibling run's config when a run "
                         "has none of its own")
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
    baselines_built = set()
    # An explicit single --stem also answers resolve_run's own question about
    # which set the run was solved with, which it raises on when it cannot tell.
    forced = None
    if len(args.stem) == 1 and args.stem[0] != "all":
        forced = ("" if args.stem[0] in ("", "reference")
                  else args.stem[0] if args.stem[0].startswith("_")
                  else "_" + args.stem[0])
    for run in runs:
        try:
            spec = cm.resolve_run(run, forced_stem=forced,
                                  verify=not args.no_verify)
        except SystemExit as exc:
            # resolve_run exits when it cannot identify the run's profile set.
            # One such run must not cancel the others: pass a single --stem to
            # answer it, or --no-verify to skip the content match.
            print(f"  SKIPPED {os.path.basename(run)}: {exc}", file=sys.stderr)
            failed += 1
            continue
        cm.report(spec)

        run_overrides = dict(overrides)
        if args.config:
            spec["config_path"] = os.path.abspath(args.config)
        elif not spec.get("config_path") and not args.no_borrow_config:
            borrowed, from_run = find_sibling_config(cm, run)
            if borrowed:
                own = settings_from_summary(run)
                spec["config_path"] = borrowed
                # The run's own record wins over the sibling's for the keys it
                # actually holds; --set still wins over both.
                run_overrides = {**own, **overrides}
                print(f"  ! no config at this run; borrowing {from_run}/"
                      f"{os.path.basename(borrowed)}. Its mixing, tolerances, "
                      f"max_iter and warm-up are ASSUMED to match — cheaseBS "
                      f"records none of them per run."
                      + (f" Taken from this run's own convergence_summary.json: "
                         f"{own}" if own else ""), file=sys.stderr)
        if not spec.get("config_path"):
            print(f"  SKIPPED: no cheaseBS config at {run}, no single sibling "
                  f"to borrow from, and none passed with --config",
                  file=sys.stderr)
            failed += 1
            continue
        if not spec.get("gfile_before"):
            print(f"  SKIPPED: no source g<shot>.<time> at {run}", file=sys.stderr)
            failed += 1
            continue

        stems = cm.candidate_stems(spec["root"])
        if args.stem and "all" in args.stem:
            # Scaled sets only -- the bare set is the reference, not a run.
            wanted = [s for s in sorted(stems) if s]
        elif args.stem:
            named = [s for s in args.stem if s not in ("", "reference")]
            if len(named) != len(args.stem):
                raise SystemExit(
                    "the bare profiles_{e,i,z} set cannot be solved: it is what "
                    "reference_* points at, so solving it would build the "
                    "baseline decomposition from the profiles being replayed "
                    "and p_fast would absorb the whole difference. Name a "
                    "scaled set instead.")
            wanted = [s if s.startswith("_") else "_" + s for s in named]
        else:
            # The set this run was solved with -- reproducing the hatch run is
            # the default, scanning is opt-in. Taken from the resolution above,
            # but only when it names a set that is actually AT THIS RUN ROOT: a
            # config can name a path in a sibling run, or one that is gone.
            after = (spec.get("profiles_after") or {}).get("e", "")
            wanted = []
            if after:
                same_root = (os.path.dirname(os.path.abspath(after))
                             == os.path.abspath(spec["root"]))
                suffix = os.path.basename(after)[len("profiles_e"):]
                if same_root and suffix in stems:
                    wanted = [suffix]
                else:
                    print("  ! the resolved run profiles (%s) are not a "
                          "profiles_e<suffix> at this run root; ignoring them "
                          "for the default stem" % after, file=sys.stderr)
            if wanted == [""]:
                print("  ! this run was solved with the bare profile set, so "
                      "there is no scaled set to reproduce", file=sys.stderr)
                wanted = []
            if not wanted:
                scaled = [s for s in sorted(stems) if s]
                if len(scaled) == 1:
                    wanted = scaled
                    print("  ! defaulting to the only scaled set present: %s"
                          % stem_label(scaled[0]), file=sys.stderr)
                else:
                    print("  SKIPPED %s: cannot tell which set it solved with; "
                          "pass --stem (present: %s)"
                          % (spec["name"],
                             ", ".join(stem_label(s) for s in sorted(stems))),
                          file=sys.stderr)
                    failed += 1
                    continue

        # A requested stem that this run does not carry is skipped, not fatal:
        # the hatch tree does not hold the same scalings for every run (the test
        # run has 1.3n and no 1.3T), and one absent set should not cancel the
        # sets that are there.
        absent = [s for s in wanted if s not in stems]
        for stem in absent:
            print(f"  ! no profiles_{{e,i,z}}{stem} in {spec['name']}; skipped. "
                  f"Present: {', '.join(stem_label(s) for s in sorted(stems))}",
                  file=sys.stderr)
        wanted = [s for s in wanted if s in stems]
        if not wanted:
            print(f"  SKIPPED: none of the requested profile sets exist in "
                  f"{spec['name']}", file=sys.stderr)
            failed += 1
            continue

        for stem in wanted:
            label = stem_label(stem)
            out_dir = os.path.join(outroot, f"{spec['name']}_{label}")
            baseline = (os.path.join(spec["root"], "baseline")
                        if args.reuse_baseline
                        else os.path.join(outroot, f"{spec['name']}_baseline"))
            # First solve to claim this baseline directory builds it; the
            # rest reuse. --reuse-baseline points at the hatch run's own, which
            # already exists, so nothing is rebuilt at all.
            if args.reuse_baseline:
                rebuild = False
            else:
                rebuild = baseline not in baselines_built
                baselines_built.add(baseline)
            try:
                cfg, diff = build_config(spec, stems, stem, out_dir, baseline,
                                         run_overrides, tped,
                                         rebuild_baseline=rebuild)
            except ValueError as exc:
                print(f"  SKIPPED {spec['name']}/{label}: {exc}", file=sys.stderr)
                failed += 1
                continue
            print(f"\n--- {spec['name']} / {label} ---")
            print("  profiles     : modified %s   reference %s   (both from "
                  "this run root)"
                  % (os.path.basename(stems[stem]["e"]),
                     os.path.basename(stems[""]["e"]) if "" in stems
                     else "(absent)"))
            print(f"  output_dir   : {out_dir}")
            print("  baseline_dir : %s   (%s)"
                  % (baseline,
                     "hatch run's own, reused" if args.reuse_baseline
                     else "built by this solve" if rebuild
                     else "reused from the solve above"))
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
