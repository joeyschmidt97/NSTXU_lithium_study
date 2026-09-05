#!/usr/bin/env python3
"""Run-compare the cheaseBS "hatch" runs, the way a reshape scan is compared.

WHAT THIS IS FOR
----------------
The reshape campaign renders one `run_summary.png` per scan point and one
`plot_run_comparison` figure per campaign, both out of TPED's cheasebs_runner
(`plot_run_summary` / `plot_run_comparison`). Those readers expect the
directory layout `run_cheasebs_workflow` copies back:

    <run>/g<shot>.<time>                        the source EQDSK
         /EQDSK*.OUT                            the reconstruction
         /profiles_{e,i,z}                      the profiles the run solved with
         /reference_profiles/REF_profiles_{e,i,z}   the untransformed reference
         /cheasebs_run_config.json
         /iteration_log.csv

The hatch results tree does not have that shape. The iteration log and the
per-iteration artifacts sit one level down in `output/`, the reconstruction is
named `g_final.eqdsk` rather than `EQDSK*.OUT`, the reference profiles are the
bare `profiles_{e,i,z}` at the run root, and the profiles actually solved with
are a suffixed sibling set (`profiles_e_1.3n`, `profiles_e_1.3T`, ...) with
nothing in the run tree stating which suffix was used.

So this script does two things and then hands off:

1. RESOLVES each run -- source gfile, reconstruction, iteration log, run
   profiles, reference profiles -- and PRINTS every absolute path it chose,
   with size and mtime, so the tie between a run and its profile files can be
   checked by eye before any figure is trusted.
2. Builds a small conformant shim directory per run (symlinks, plus a
   synthesized cheasebs_run_config.json pointing at the REAL profile and source
   paths) and calls the same two TPED plotters the reshape campaign uses.

The shim carries links and one JSON, nothing else, and nothing in the hatch
tree is written to. Links are symlinks where the filesystem allows them and
copies where it does not (Windows without developer mode), so the shim is
disposable either way -- delete it and re-run.

WHICH PROFILE SET DID A RUN ACTUALLY SOLVE WITH
-----------------------------------------------
Guessing from the suffix is exactly the mistake that made the reshape
reference-profile bug survive twenty solves, so the choice is measured rather
than assumed. cheaseBS writes the kinetic profiles it handed CHEASE into an
`EXPTNZ` file (rhopsi, Te[eV], ne[m^-3], Zeff, Ti[eV] -- see
`build_exptnz_lines` in cheaseBS/preprocess_eqdsk.py), one per stage:

    output/iteration_00/artifacts/EXPTNZ        <- the RUN profiles
    baseline/chease_stage2_artifacts/EXPTNZ     <- the REFERENCE profiles

Each candidate `profiles_{e,i,z}<suffix>` set is interpolated onto that
EXPTNZ's rhopsi grid and scored by max relative deviation in Te and ne. The
best-scoring suffix wins, and every candidate's score is printed, so a decision
that was not decisive is visible rather than silent.

Most of these profile files carry rho_tor in column 1 and zeros where rho_pol
would be, so putting a candidate on an EXPTNZ's rhopsi axis needs the
rhop<->rhot pair from `baseline/profiles.csv` -- the same grid cheaseBS
interpolated onto when it wrote the file. Without that CSV the content check is
skipped and the choice falls back to the config or the naming convention, and
says so.

A workflow config JSON at the run root (e.g. `nstx_joey_test.json`) overrides
the match, since that is what the driver actually read. `--stem` overrides both.

USAGE
-----
    python compare_hatch_runs.py \
        --root /pscratch/sd/j/joeschm/cheaseBS_hatch_results/for_joey \
        --outdir ./hatch_compare

    # inspect the resolution only, draw nothing
    python compare_hatch_runs.py --root ... --dry-run

    # force the tie by hand
    python compare_hatch_runs.py --root ... \
        --stem test=_1.3n --stem test2=_1.3T

    # pedestal zoom on the comparison figure
    python compare_hatch_runs.py --root ... --xlim 0.8 1.0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import sys

import numpy as np

# The plotters render to file and must never reach for a display.
os.environ.setdefault("MPLBACKEND", "Agg")

SPECIES = ("e", "i", "z")

# Where each stage's EXPTNZ lands, relative to a hatch run root. First existing
# path wins; the bootstrap-diagnostic copy is the fallback because it is written
# from the same profiles when the primary artifact directory is incomplete.
RUN_EXPTNZ = (
    "output/iteration_00/artifacts/EXPTNZ",
    "output/iteration_00/bootstrap_diagnostic_artifacts/EXPTNZ",
)
REF_EXPTNZ = (
    "baseline/chease_stage2_artifacts/EXPTNZ",
)

# Solver keys worth carrying into the shim config: the plotters read them for
# the tolerance lines and the provenance block, and plot_run_comparison uses
# them to decide which knob differs between variants.
CFG_KEYS = (
    "coordinate", "replay_representation", "enforce_qspec", "initial_amplitude",
    "max_iter", "tol_ip_rel", "tol_bs", "tol_q", "tol_a",
    "bootstrap_mix", "istar_mix", "amplitude_warmup_iters",
    "istar_regularize", "target_ip_a", "target_bt_t",
)


# ----------------------------------------------------------------------
# Small readers
# ----------------------------------------------------------------------

def read_gene_profile(path):
    """(rho_tor, rho_pol, T[keV], n[1e19 m^-3]) from a GENE profiles_<spec> file.

    Column ORDER is relied on, not the header wording, which differs between
    writers -- same convention as cheasebs_runner._read_gene_profile.
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("%s: expected 4 columns, got shape %s" % (path, data.shape))
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def read_exptnz(path):
    """(rhopsi, Te[eV], ne[m^-3]) from a cheaseBS-written EXPTNZ file.

    Header line is ' N   rhopsi, Te, ne, Zeff, Ti, ni profiles' followed by 5*N
    values, one per line, in that block order.
    """
    with open(path) as fh:
        first = fh.readline()
        rest = fh.read()
    n = int(first.split()[0])
    vals = np.fromstring(rest.replace("E", "e"), sep="\n")
    if vals.size < 5 * n:
        raise ValueError("%s: expected %d values, found %d" % (path, 5 * n, vals.size))
    block = vals[:5 * n].reshape(5, n)
    return block[0], block[1], block[2]


def _first_existing(root, rels):
    for rel in rels:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def _stat_line(path):
    """'12.3 kB  2026-09-04 11:02' for a path, or a reason it is not usable."""
    if not path:
        return "(absent)"
    if not os.path.exists(path):
        return "(MISSING)"
    st = os.stat(path)
    when = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
    return "%8.1f kB  %s" % (st.st_size / 1024.0, when)


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------

def candidate_stems(run_root):
    """{suffix: {spec: path}} for every complete profiles_{e,i,z}<suffix> set.

    The unscaled reference set has suffix ''. A set is only a candidate when all
    three species are present, because cheaseBS needs all three and a partial
    set would score against whichever species happened to exist.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(run_root, "profiles_e*"))):
        if not os.path.isfile(path):
            continue
        suffix = os.path.basename(path)[len("profiles_e"):]
        trio = {s: os.path.join(run_root, "profiles_%s%s" % (s, suffix))
                for s in SPECIES}
        if all(os.path.isfile(p) for p in trio.values()):
            out[suffix] = trio
    return out


def read_baseline_grid(run_root):
    """(rhop, rhot) of the baseline decomposition grid, or None.

    `baseline/profiles.csv` is the grid CHEASE's mapping stage produced and the
    grid every EXPTNZ is written on; it is also the only place the rhop<->rhot
    correspondence for this equilibrium is recorded. Needed because the GENE
    profile files carry rho_tor in column 1 and, in most of the cheaseBS sets,
    a column of zeros where rho_pol would be -- so a candidate cannot be put on
    an EXPTNZ's rhopsi axis without it.
    """
    path = os.path.join(run_root, "baseline", "profiles.csv")
    if not os.path.isfile(path):
        return None
    import csv
    rhop, rhot = [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                rhop.append(float(row["rhop"]))
                rhot.append(float(row["rhot"]))
            except (KeyError, TypeError, ValueError):
                return None
    if not rhop:
        return None
    return np.asarray(rhop), np.asarray(rhot)


def score_stem(trio, exptnz, grid):
    """max relative deviation of a candidate's Te and ne from an EXPTNZ stage.

    The candidate is put on the EXPTNZ's own rhopsi axis using the same rule
    cheaseBS used to build that file (`interpolate_profile_onto_baseline` in
    run_chease_closure_test.py): the profile's second column is the abscissa
    when it has any spread, and its first column (rho_tor) otherwise -- which is
    the usual case, since most of these files carry zeros there. In the rho_tor
    case the EXPTNZ axis is carried across through the baseline grid's own
    rhop<->rhot pair, so no mapping is assumed that cheaseBS did not already
    write down.

    Deviations are normalised by the profile scale rather than taken pointwise,
    so a near-zero edge value cannot dominate the score.
    """
    rhopsi, te_ev, ne_m3 = exptnz
    rho_tor, rho_pol, t_kev, n_19 = read_gene_profile(trio["e"])

    if float(np.max(rho_pol) - np.min(rho_pol)) > 1.0e-8:
        source, target = rho_pol, rhopsi
    else:
        if grid is None:
            raise ValueError(
                "profile carries no rho_pol column and baseline/profiles.csv is "
                "absent, so there is no rhop<->rhot map to score against")
        source = rho_tor
        target = np.interp(rhopsi, grid[0], grid[1])

    order = np.argsort(source)
    resid = []
    for cand, ref in ((np.interp(target, source[order], t_kev[order]) * 1.0e3, te_ev),
                      (np.interp(target, source[order], n_19[order]) * 1.0e19, ne_m3)):
        scale = np.nanmax(np.abs(ref))
        resid.append(float("nan") if not np.isfinite(scale) or scale <= 0
                     else float(np.nanmax(np.abs(cand - ref)) / scale))
    return max(resid)


def match_stem(stems, exptnz_path, grid, label):
    """(best_suffix, {suffix: score}, note) for one EXPTNZ stage.

    Returns (None, {}, note) when the stage file is missing -- the caller falls
    back to the naming convention and says so, rather than picking silently.
    """
    if not exptnz_path:
        return None, {}, "no %s EXPTNZ under this run; cannot verify by content" % label
    exptnz = read_exptnz(exptnz_path)
    scores = {}
    failed = []
    for suffix, trio in stems.items():
        try:
            scores[suffix] = score_stem(trio, exptnz, grid)
        except Exception as exc:
            scores[suffix] = float("nan")
            failed.append("%s (%s: %s)" % (suffix or "(bare)", type(exc).__name__, exc))
    usable = {k: v for k, v in scores.items() if v == v}
    if not usable:
        return None, scores, ("no candidate could be scored against %s: %s"
                              % (exptnz_path, "; ".join(failed)))
    best = min(usable, key=usable.get)
    note = "; ".join(failed)
    ranked = sorted(usable.values())
    if len(ranked) > 1 and ranked[1] < 10.0 * max(ranked[0], 1.0e-12):
        extra = ("match is NOT decisive -- best %.3g vs runner-up %.3g; "
                 "pass --stem to fix it by hand" % (ranked[0], ranked[1]))
        note = (note + "; " + extra) if note else extra
    return best, scores, note


def read_run_config(run_root):
    """(path, cfg) of a workflow config JSON sitting at the run root, if any.

    Identified by content -- an `electron_profile` key -- not by name, since the
    file is named after the case (`nstx_joey_test.json`) and a run may also carry
    unrelated JSON.
    """
    for path in sorted(glob.glob(os.path.join(run_root, "*.json"))):
        try:
            with open(path) as fh:
                cfg = json.load(fh)
        except Exception:
            continue
        if isinstance(cfg, dict) and "electron_profile" in cfg:
            return path, cfg
    return None, {}


def _cfg_profiles(cfg, cfg_path, prefix):
    """{spec: abspath} for a config's profile trio, resolved against the config.

    cheaseBS resolves relative config paths against its own working directory,
    which is not recoverable after the fact; the config's own directory is the
    only reproducible anchor. Absolute entries are used as written.
    """
    if not cfg_path or not cfg:
        return {}
    keys = {"e": "electron_profile", "i": "deuterium_profile", "z": "carbon_profile"}
    base = os.path.dirname(os.path.abspath(cfg_path))
    out = {}
    for spec, key in keys.items():
        raw = cfg.get(prefix + key)
        if not raw:
            return {}
        out[spec] = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(base, raw))
    # A config path that does not resolve is worse than no config path: the run
    # was launched from a working directory that is not recoverable, or it read
    # scratch that has since been purged. Fall through to the content match
    # rather than hand the plotters a set of names that are not on disk.
    if not all(os.path.isfile(q) for q in out.values()):
        return {}
    return out


def find_source_gfile(run_root):
    """The input EFIT under its original g<shot>.<time> name.

    Strictly name-matched: globbing g* also catches `g_final.eqdsk`, which is the
    reconstruction, and using it as "before" makes every delta come out zero.
    """
    cands = [p for p in glob.glob(os.path.join(run_root, "g*"))
             if os.path.isfile(p) and re.match(r"^g\d+\.\d+", os.path.basename(p))]
    return sorted(cands)[0] if cands else None


def find_final_eqdsk(run_root):
    """The reconstruction: g_final.eqdsk, else the highest iteration's EQDSK.

    A run that stopped at max_iter without converging has no g_final.eqdsk, and
    its last-iteration equilibrium is exactly the one worth looking at.
    """
    direct = os.path.join(run_root, "g_final.eqdsk")
    if os.path.isfile(direct):
        return direct

    def _idx(p):
        m = re.search(r"iteration_(\d+)", p)
        return int(m.group(1)) if m else -1

    cands = [p for p in glob.glob(os.path.join(
        run_root, "output", "iteration_*", "artifacts", "EQDSK*.OUT"))
        if os.path.isfile(p)]
    if not cands:
        return None
    cands.sort(key=lambda p: (_idx(p), "POS" in os.path.basename(p)))
    return cands[-1]


def resolve_run(run_root, forced_stem=None, verify=True):
    """Everything one hatch run contributes, plus how each choice was made."""
    run_root = os.path.abspath(run_root)
    spec = {
        "name": os.path.basename(run_root),
        "root": run_root,
        "notes": [],
        "scores": {},
    }

    spec["gfile_before"] = find_source_gfile(run_root)
    spec["gfile_after"] = find_final_eqdsk(run_root)
    spec["iteration_log"] = _first_existing(run_root, ("output/iteration_log.csv",
                                                       "iteration_log.csv"))
    spec["summary"] = _first_existing(run_root, ("output/convergence_summary.json",
                                                 "convergence_summary.json"))
    spec["run_exptnz"] = _first_existing(run_root, RUN_EXPTNZ)
    spec["ref_exptnz"] = _first_existing(run_root, REF_EXPTNZ)

    cfg_path, cfg = read_run_config(run_root)
    spec["config_path"] = cfg_path
    spec["cfg"] = {k: cfg[k] for k in CFG_KEYS if k in cfg}
    if cfg_path and not _cfg_profiles(cfg, cfg_path, ""):
        spec["notes"].append(
            "%s names profile files that are not on disk (resolved against the "
            "config's own directory); falling back to the EXPTNZ content match"
            % os.path.basename(cfg_path))

    grid = read_baseline_grid(run_root)
    spec["grid"] = grid
    if grid is None:
        spec["notes"].append(
            "no baseline/profiles.csv; EXPTNZ verification is only possible for "
            "profile files that carry a real rho_pol column")

    stems = candidate_stems(run_root)
    spec["stems"] = stems
    if not stems:
        spec["notes"].append("no complete profiles_{e,i,z}* set at the run root")
        spec["profiles_after"] = {}
        spec["profiles_before"] = {}
        return spec

    # --- the run (transformed) set ---
    after = {}
    if forced_stem is not None:
        if forced_stem not in stems:
            raise SystemExit(
                "%s: --stem %r is not one of the candidate suffixes %s"
                % (spec["name"], forced_stem, sorted(stems)))
        after = stems[forced_stem]
        spec["after_how"] = "--stem %r" % (forced_stem or "(bare)")
    elif _cfg_profiles(cfg, cfg_path, ""):
        after = _cfg_profiles(cfg, cfg_path, "")
        spec["after_how"] = "config %s" % os.path.basename(cfg_path)
    elif verify:
        best, scores, note = match_stem(stems, spec["run_exptnz"], grid, "run")
        spec["scores"]["run"] = scores
        if note:
            spec["notes"].append("run profiles: " + note)
        if best is not None:
            after = stems[best]
            spec["after_how"] = ("EXPTNZ match on %s (residual %.3g)"
                                 % (os.path.relpath(spec["run_exptnz"], run_root),
                                    scores[best]))
    if not after:
        # Naming convention, last: the transformed set is the one that is not
        # the bare stem. Only usable when it is unique.
        suffixed = [s for s in stems if s]
        if len(suffixed) == 1:
            after = stems[suffixed[0]]
            spec["after_how"] = "only suffixed set present (%r)" % suffixed[0]
        else:
            raise SystemExit(
                "%s: cannot tell which profile set this run solved with. "
                "Candidates: %s. Re-run with --stem %s=<suffix>."
                % (spec["name"], sorted(stems), spec["name"]))

    # --- the reference (untransformed) set ---
    before = {}
    if _cfg_profiles(cfg, cfg_path, "reference_"):
        before = _cfg_profiles(cfg, cfg_path, "reference_")
        spec["before_how"] = "config %s" % os.path.basename(cfg_path)
    elif verify:
        best, scores, note = match_stem(stems, spec["ref_exptnz"], grid, "baseline")
        spec["scores"]["reference"] = scores
        if note:
            spec["notes"].append("reference profiles: " + note)
        if best is not None:
            before = stems[best]
            spec["before_how"] = ("EXPTNZ match on %s (residual %.3g)"
                                  % (os.path.relpath(spec["ref_exptnz"], run_root),
                                     scores[best]))
    if not before and "" in stems:
        before = stems[""]
        spec["before_how"] = "bare profiles_{e,i,z} at the run root"

    spec["profiles_after"] = after
    spec["profiles_before"] = before

    # The failure this whole resolution exists to expose: if the reference set
    # IS the run set, the baseline decomposition was rebuilt from the scaled
    # profiles, p_fast absorbs the scaling, and the "scan" never reached CHEASE.
    same = [s for s in SPECIES
            if before.get(s) and after.get(s)
            and os.path.realpath(before[s]) == os.path.realpath(after[s])]
    if same:
        spec["notes"].append(
            "reference profiles ARE the run profiles (%s) -- the baseline "
            "decomposition was built from the scaled profiles, so p_total is "
            "floored at the source pressure and the transform is invisible"
            % ",".join(same))
    if not spec["gfile_before"]:
        spec["notes"].append("no g<shot>.<time> at the run root; 'before' equilibrium unavailable")
    if not spec["gfile_after"]:
        spec["notes"].append("no g_final.eqdsk and no iteration EQDSK; no reconstruction to plot")
    if not spec["iteration_log"]:
        spec["notes"].append("no iteration_log.csv; the iteration-metric row will be empty")
    return spec


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def report(spec):
    """Everything resolved for one run, as absolute paths, to stdout."""
    print("=" * 100)
    print("RUN  %s" % spec["name"])
    print("  root : %s" % spec["root"])
    print("-" * 100)

    rows = [
        ("gfile before (source EFIT)", spec.get("gfile_before")),
        ("gfile after  (reconstruction)", spec.get("gfile_after")),
        ("iteration log", spec.get("iteration_log")),
        ("convergence summary", spec.get("summary")),
        ("workflow config", spec.get("config_path")),
        ("EXPTNZ (run stage)", spec.get("run_exptnz")),
        ("EXPTNZ (baseline stage)", spec.get("ref_exptnz")),
    ]
    for label, path in rows:
        print("  %-30s %s" % (label + ":", path or "(absent)"))
        print("  %-30s %s" % ("", _stat_line(path)))

    print()
    print("  PROFILES  after = what the run solved with | before = untransformed reference")
    print("  %-30s %s" % ("after chosen by:", spec.get("after_how", "(unresolved)")))
    print("  %-30s %s" % ("before chosen by:", spec.get("before_how", "(unresolved)")))
    for spc in SPECIES:
        for tag, key in (("after ", "profiles_after"), ("before", "profiles_before")):
            path = spec.get(key, {}).get(spc)
            print("    [%s] %s : %s" % (spc, tag, path or "(absent)"))
            print("    %-13s %s" % ("", _stat_line(path)))

    if spec.get("stems"):
        print()
        print("  CANDIDATE PROFILE SETS at the run root (suffix -> EXPTNZ residual):")
        for suffix in sorted(spec["stems"]):
            run_s = spec["scores"].get("run", {}).get(suffix)
            ref_s = spec["scores"].get("reference", {}).get(suffix)
            def _fmt(v):
                return "     n/a" if v is None else ("%8.3g" % v)
            print("    profiles_{e,i,z}%-12s  run:%s   baseline:%s"
                  % (suffix or "", _fmt(run_s), _fmt(ref_s)))

    for note in spec["notes"]:
        print("  ! %s" % note)
    if spec.get("cfg"):
        print("  solver: %s" % spec["cfg"])
    print()


# ----------------------------------------------------------------------
# Shim construction
# ----------------------------------------------------------------------

def _link(src, dst):
    """Symlink src -> dst, falling back to a copy where symlinks are refused."""
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def build_shim(spec, shim_root):
    """A directory TPED's readers understand, pointing at the real hatch files.

    Deliberately sparse. resolve_run_files() prefers files found IN the run
    directory and only consults the config when they are absent, so the profiles
    and the source gfile are left OUT of the shim and named in the config
    instead: that way the provenance block on the figure prints the real hatch
    paths rather than these symlinks. The reconstruction has no config fallback
    in that reader, so it is the one thing that must be linked in, under a name
    matching the EQDSK*.OUT glob.
    """
    run_dir = os.path.join(shim_root, spec["name"])
    os.makedirs(run_dir, exist_ok=True)

    if spec.get("gfile_after"):
        _link(spec["gfile_after"], os.path.join(run_dir, "EQDSK_COCOS_02_POS.OUT"))
    if spec.get("iteration_log"):
        _link(spec["iteration_log"], os.path.join(run_dir, "iteration_log.csv"))
    if spec.get("summary"):
        _link(spec["summary"], os.path.join(run_dir, "convergence_summary.json"))

    cfg = dict(spec.get("cfg") or {})
    if spec.get("gfile_before"):
        cfg["eqdsk"] = spec["gfile_before"]
    keys = {"e": "electron_profile", "i": "deuterium_profile", "z": "carbon_profile"}
    for spc, key in keys.items():
        if spec.get("profiles_after", {}).get(spc):
            cfg[key] = spec["profiles_after"][spc]
        if spec.get("profiles_before", {}).get(spc):
            cfg["reference_" + key] = spec["profiles_before"][spc]
    # Provenance for the one file the plotters cannot name for themselves: the
    # reconstruction is read out of the shim, so the figure's path block prints
    # the link rather than g_final.eqdsk. Record where it really came from.
    cfg["_hatch_source_run"] = spec["root"]
    if spec.get("gfile_after"):
        cfg["_hatch_reconstruction"] = spec["gfile_after"]
    with open(os.path.join(run_dir, "cheasebs_run_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, sort_keys=True)
    return run_dir


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Resolve and run-compare the cheaseBS hatch runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="directory holding the run directories "
                        "(e.g. /pscratch/sd/j/joeschm/cheaseBS_hatch_results/for_joey)")
    p.add_argument("--run", action="append", default=[],
                   help="run directory name under --root; repeatable. "
                        "Default: every subdirectory that looks like a run.")
    p.add_argument("--outdir", default="hatch_compare",
                   help="where the figures and the shim tree are written")
    p.add_argument("--stem", action="append", default=[], metavar="RUN=SUFFIX",
                   help="force the run's profile suffix, e.g. test2=_1.3T. "
                        "Use RUN= (empty) for the bare profiles_{e,i,z} set.")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the EXPTNZ content match and rely on config/naming only")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and print paths, draw nothing")
    p.add_argument("--xlim", nargs=2, type=float, metavar=("LO", "HI"),
                   help="restrict every rho_tor panel of the comparison, e.g. 0.8 1.0")
    p.add_argument("--alpha", action="store_true",
                   help="add the alpha / s-alpha panels (costs a flux-surface contour per run)")
    p.add_argument("--enclosed", action="store_true",
                   help="add the enclosed-current panel")
    p.add_argument("--tped",
                   help="path to prepend to sys.path so `import TPED` resolves")
    return p


def discover_runs(root, names):
    """The run directories to compare, in the order given or sorted by name."""
    if names:
        dirs = [os.path.join(root, n) for n in names]
        missing = [d for d in dirs if not os.path.isdir(d)]
        if missing:
            raise SystemExit("no such run director%s: %s"
                             % ("y" if len(missing) == 1 else "ies",
                                ", ".join(missing)))
        return dirs
    out = []
    for entry in sorted(os.listdir(root)):
        d = os.path.join(root, entry)
        if not os.path.isdir(d):
            continue
        # A run root is anything carrying an output/ tree or a source gfile.
        if os.path.isdir(os.path.join(d, "output")) or find_source_gfile(d):
            out.append(d)
    if not out:
        raise SystemExit("no run directories found under %s" % root)
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.tped:
        sys.path.insert(0, os.path.abspath(args.tped))

    forced = {}
    for item in args.stem:
        if "=" not in item:
            raise SystemExit("--stem expects RUN=SUFFIX, got %r" % item)
        name, suffix = item.split("=", 1)
        forced[name] = suffix

    root = os.path.abspath(args.root)
    run_dirs = discover_runs(root, args.run)
    print("[hatch] root: %s" % root)
    print("[hatch] runs: %s" % ", ".join(os.path.basename(d) for d in run_dirs))
    print()

    specs = [resolve_run(d, forced_stem=forced.get(os.path.basename(d)),
                         verify=not args.no_verify)
             for d in run_dirs]
    for spec in specs:
        report(spec)

    if args.dry_run:
        print("[hatch] --dry-run: resolution only, no figures written")
        return 0

    try:
        from TPED.projects.discharge_tools.src.cheasebs_runner import (
            plot_run_comparison, plot_run_summary)
    except ImportError as exc:
        raise SystemExit(
            "TPED is not importable (%s). Point at it with --tped "
            "/path/to/TPED-parent, or activate the environment the reshape "
            "campaign runs in." % exc)

    outdir = os.path.abspath(args.outdir)
    shim_root = os.path.join(outdir, "shim")
    os.makedirs(shim_root, exist_ok=True)

    shims = []
    for spec in specs:
        shim = build_shim(spec, shim_root)
        shims.append(shim)
        print("[hatch] %-12s shim: %s" % (spec["name"], shim))

    print()
    for spec, shim in zip(specs, shims):
        out_png = os.path.join(outdir, "run_summary_%s.png" % spec["name"])
        png = plot_run_summary(shim, out_png, cfg=spec.get("cfg"))
        print("[hatch] %-12s summary: %s" % (spec["name"], png or "(not written)"))

    title = "cheaseBS hatch runs: " + " vs ".join(s["name"] for s in specs)
    out_png = os.path.join(outdir, "run_comparison.png")
    png = plot_run_comparison(shims, out_png, title=title,
                              alpha=args.alpha, enclosed=args.enclosed,
                              xlim=tuple(args.xlim) if args.xlim else None)
    print("[hatch] comparison: %s" % (png or "(not written)"))

    # plot_run_comparison labels a variant by its leaf directory name plus only
    # the config keys that differ, so say which legend entry is which run.
    print()
    print("[hatch] legend key -> hatch run:")
    for spec, shim in zip(specs, shims):
        print("    %-12s -> %s" % (os.path.basename(shim), spec["root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
