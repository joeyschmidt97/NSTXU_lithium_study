#!/usr/bin/env python3
"""Self-scaled omt/omn cheaseBS scan for an NSTX / NSTX-U discharge.

The NSTX face of `DIIID_verify/run_cheasebs_selfscaled_scan.py`. It does not
reimplement that campaign -- it stages an NSTX discharge into the form that
script expects and then calls its `main()`, with the method fixed to `omn_omt`
and the NSTX solver template selected. One method, because the fit path is what
misbehaved on DIII-D and the gradient path is what reproduced the source exactly
at alpha = 1.0.

Two things make an NSTX discharge not directly usable by that script:

1. **No `profiles_*` files.** These discharge directories hold a gfile, an
   afile and a pfile; the profiles come out of the pfile. The DIII-D scan
   refuses a case directory with no `profiles_*` in it, and has no `--pfile`
   flag to offer instead. So this script writes GENE profiles from the loaded
   discharge into a staged case directory and points the scan at that. The
   staged files are the campaign's own record of what it scaled.

2. **No impurity species on some shots.** 129015 and 129038 carry `ne`, `ni`,
   `Te`, `Ti` and nothing else, while cheaseBS wants all three species. Staging
   synthesises the missing ones the same way the transforms do -- `nz` from
   quasineutrality at Z = 6, `Tz = Ti` -- and says so in the log rather than
   letting a zero-filled `profiles_z` reach CHEASE unremarked.

The solver settings come from `nstx_cheasebs_config.json`, which is the
validated NSTX path from cheaseBS's own `nstx_config.json`: `rhop` coordinate,
`istar` replay with regularisation, and the damped Picard loop
(`bootstrap_mix = 0.1`, `istar_mix = 0.05`, two warm-up iterations) -- not the
DIII-D `rhot` / `jparallel` path. The CHEASE namelist is `chease_namelist_nstx`.

    # one point, to see the machinery work end to end
    python run_nstx_omtomn_scan.py --shot 129015 --pair 0.7,0.7 \
        --outroot $SCRATCH/NSTX_omn_omt_cheaseBS

    # the diagonal
    python run_nstx_omtomn_scan.py --shot 129015 \
        --outroot $SCRATCH/NSTX_omn_omt_cheaseBS

    # stage and scale only; solve nothing
    python run_nstx_omtomn_scan.py --shot 132588 --dry-run

    # a VALUE scaling -- Te,Ti,Tz x 1.3 replayed against the unscaled reference,
    # which is what the hatch runs do. Solved by run_hatch_cheasebs.py, since a
    # multiply is not one of this campaign's methods.
    python run_nstx_omtomn_scan.py --shot 129015 --scale-t 1.3 \
        --outroot $SCRATCH/NSTX_hatch_mimic

Anything this script does not define itself is forwarded, so
`--analysis-radii`, `--baseline-dir` and the rest of the DIII-D scan's flags
still work:

    python run_nstx_omtomn_scan.py --shot 132588 -- --no-gate --max-iter 30
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SELFSCALED = os.path.join(os.path.dirname(HERE), "DIIID_verify",
                          "run_cheasebs_selfscaled_scan.py")

# Import TPED and the sibling campaign from this checkout rather than whatever
# is installed, so this script matches the source it ships next to.
for parent in (HERE, os.path.dirname(HERE)):
    root = os.path.dirname(os.path.dirname(parent))
    if root not in sys.path:
        sys.path.insert(0, root)

# First path that exists wins, so the same script runs on NERSC and on a laptop.
# Kept in step with pedestal_scan.DISCHARGE_ROOT_CANDIDATES and the notebook.
DISCHARGE_ROOT_CANDIDATES = [
    r"/global/homes/j/joeschm/data/ST_research/NSTXU_discharges",   # NERSC
    r"C:/Users/joesc/git/ST_research/NSTXU_discharges",             # local
]

SHOTS = (129015, 129038, 132543, 132588)

# 129038's directory holds five pfiles, so auto-discovery refuses to guess.
PFILES = {129038: "p129038.00400"}

# (rhot_topped, rhot_midped) per discharge -- the exponent ramp, NOT a measured
# pedestal. midped = 1.0 pivots the power law on the separatrix and holds it
# fixed; topped = 0.8 leaves the core untouched. Same convention and same
# default as the handed-over IFS modProfs scan for DIII-D 162940, so the two
# campaigns are comparable. Settle a shot's window in
# `nstx_scaling_check.ipynb` first, then record it here.
RAMP_WINDOWS = {
    129015: (0.8, 1.0),
    129038: (0.8, 1.0),
    132543: (0.8, 1.0),
    132588: (0.8, 1.0),
}

# Where GENE is actually run, from pedestal_scan.ANALYSIS_RADII: q is scored
# there, so a scaling that moves the equilibrium somewhere GENE never looks is
# visible as a q check that did not move.
ANALYSIS_RADII = {129015: (0.85,), 129038: (0.85,),
                  132543: (0.736, 0.825), 132588: (0.736, 0.825)}

DEFAULT_CONFIG = os.path.join(HERE, "nstx_cheasebs_config.json")
NSTX_NAMELIST = "chease_namelist_nstx"
QZ = 6.0                       # impurity charge, matching apply_omne's default


def load_selfscaled():
    """The DIII-D campaign module, imported by path.

    By path rather than by name because `DIIID_verify` is a plain directory with
    no package marker, and the module has to find its own `cheasebs_scan_common`
    sibling -- which it does through sys.path, so that directory goes on it here.
    """
    if not os.path.isfile(SELFSCALED):
        raise SystemExit(f"cannot find the scan script at {SELFSCALED}")
    sibling = os.path.dirname(SELFSCALED)
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    spec = importlib.util.spec_from_file_location("run_cheasebs_selfscaled_scan",
                                                  SELFSCALED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discharge_root():
    for d in DISCHARGE_ROOT_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise SystemExit(
        "no discharge root found; add this machine's path to "
        f"DISCHARGE_ROOT_CANDIDATES (tried {DISCHARGE_ROOT_CANDIDATES})")


def stage_case_dir(shot, dest, scale_t=None, scale_n=None):
    """Write a case directory the DIII-D scan can read: gfile + profiles_e/i/z.

    Returns (dest, notes). The profiles are written from the discharge as loaded,
    so this is also the only record of what the campaign treated as its base --
    the pfile it came from stays untouched.

    scale_t / scale_n multiply the temperature and density FAMILIES by a
    constant before writing. This is a different operation from apply_omt /
    apply_omne, not a special case of them: the gradient transforms hold the
    value at rhot_midped fixed and re-exponentiate the shape, so they cannot
    move the separatrix, while a multiply scales every radius including it.
    Hatch's runs are of this second kind, which is why mimicking them needs a
    knob here rather than an alpha. Quasineutrality survives because the whole
    family is scaled by the same factor: ne = ni + qz*nz is linear in it.
    """
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData
    from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics
    from TPED.projects.discharge_tools.src.writers.gene_writer import write_gene_profiles

    src = os.path.join(discharge_root(), str(shot))
    if not os.path.isdir(src):
        raise SystemExit(f"no directory for shot {shot}: {src}")

    kw = {"input_dir": src}
    if shot in PFILES:
        kw["pfile"] = os.path.join(src, PFILES[shot])
    data = DischargeData(**kw)
    if not data.gfile_filepath:
        raise SystemExit(f"no gfile found in {src}")
    phys = DischargePhysics(data)
    ds = phys.ds.copy()

    notes = []
    if "nz" not in ds:
        # Quasineutrality at Z = 6, the same closure apply_omne enforces after a
        # density scaling. Without it write_gene_profiles emits no profiles_z at
        # all and cheaseBS stops on the missing carbon profile.
        ds["nz"] = (ds["ne"] - ds["ni"]) / QZ
        notes.append(f"nz synthesised from quasineutrality at Z={QZ:g}")
    if "Tz" not in ds:
        ds["Tz"] = ds["Ti"]
        notes.append("Tz set equal to Ti")

    os.makedirs(dest, exist_ok=True)
    gdst = os.path.join(dest, os.path.basename(data.gfile_filepath))
    if os.path.abspath(gdst) != os.path.abspath(data.gfile_filepath):
        shutil.copy2(data.gfile_filepath, gdst)
    write_gene_profiles(ds, dest)

    missing = [f"profiles_{s}" for s in ("e", "i", "z")
               if not os.path.isfile(os.path.join(dest, f"profiles_{s}"))]
    if missing:
        raise SystemExit(f"staging wrote no {', '.join(missing)} into {dest}")

    # The bare set above is the REFERENCE and is never scaled. A value scaling
    # is written beside it as profiles_{e,i,z}<suffix>, the layout the hatch
    # runs use, so the solve replays the scaled set against the unscaled
    # reference. Multiplying the bare set instead -- which this script did until
    # 2026-09-08 -- makes the baseline decomposition and the replayed profiles
    # the same files: p_fast absorbs the whole multiply, the reconstruction
    # matches its own baseline, and the plots come out flat.
    suffix = scale_suffix(scale_t, scale_n)
    if suffix:
        scaled = ds.copy()
        for factor, family, label in ((scale_t, ("Te", "Ti", "Tz"), "temperatures"),
                                      (scale_n, ("ne", "ni", "nz"), "densities")):
            if factor is None:
                continue
            touched = [v for v in family if v in scaled]
            for v in touched:
                scaled[v] = scaled[v] * factor
            notes.append(f"{label} x {factor:g} ({', '.join(touched)}) written as "
                         f"profiles_*{suffix}; the bare set stays unscaled and is "
                         f"what reference_* points at")
        tmp = os.path.join(dest, "_scaled_tmp")
        os.makedirs(tmp, exist_ok=True)
        write_gene_profiles(scaled, tmp)
        for sp in ("e", "i", "z"):
            src_f = os.path.join(tmp, f"profiles_{sp}")
            if os.path.isfile(src_f):
                os.replace(src_f, os.path.join(dest, f"profiles_{sp}{suffix}"))
        if os.path.isdir(tmp) and not os.listdir(tmp):
            os.rmdir(tmp)
        absent = [f"profiles_{s}{suffix}" for s in ("e", "i", "z")
                  if not os.path.isfile(os.path.join(dest, f"profiles_{s}{suffix}"))]
        if absent:
            raise SystemExit(f"staging wrote no {', '.join(absent)} into {dest}")
    return dest, notes


def scale_suffix(scale_t, scale_n):
    """`_t1.3`, `_n1.3`, `_t1.3n1.3`, or "" when nothing is scaled."""
    parts = [("t%g" % scale_t) if scale_t is not None else "",
             ("n%g" % scale_n) if scale_n is not None else ""]
    tag = "".join(parts)
    return ("_" + tag) if tag else ""


def resolve_namelist(explicit):
    """The NSTX CHEASE namelist, or None to let the scan script complain.

    The DIII-D scan defaults to the plain `chease_namelist` next to the cheaseBS
    checkout, which is the DIII-D one. NSTX needs `chease_namelist_nstx`, so it
    has to be passed -- and the place that knows where the checkout is, is the
    TPED user config.
    """
    if explicit:
        return explicit
    try:
        from TPED.config.config_helper import Config
        cheasebs = Config().get_path("CHEASEBS_PATH")
    except Exception:
        return None
    if not cheasebs:
        return None
    candidate = os.path.join(cheasebs, NSTX_NAMELIST)
    return candidate if os.path.isfile(candidate) else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", type=int, required=True, choices=SHOTS,
                    help="NSTX / NSTX-U discharge to scan")
    ap.add_argument("--outroot", default=None,
                    help="campaign root (default $SCRATCH/NSTX_omn_omt_cheaseBS/"
                         "<shot>_<stamp>, or ./ when SCRATCH is unset)")
    ap.add_argument("--rhot-topped", type=float, default=None,
                    help="override this shot's ramp topped (RAMP_WINDOWS)")
    ap.add_argument("--rhot-midped", type=float, default=None,
                    help="override this shot's ramp midped (RAMP_WINDOWS)")
    ap.add_argument("--scale-t", type=float, default=None, metavar="C",
                    help="multiply Te, Ti and Tz by C before staging. A value "
                         "scaling, which apply_omt cannot express at any alpha: "
                         "it moves the separatrix, the gradient transform pins "
                         "it. Use this to mimic a hatch 1.3T set")
    ap.add_argument("--scale-n", type=float, default=None, metavar="C",
                    help="multiply ne, ni and nz by C before staging (mimics a "
                         "hatch 1.3n set); quasineutrality is preserved")
    ap.add_argument("--pair", action="append", default=None, metavar="OMT,OMN",
                    help="one scan point; repeatable. Default is the DIII-D "
                         "campaign's point list")
    ap.add_argument("--cheasebs-config", default=DEFAULT_CONFIG,
                    help="solver template (default the NSTX one beside this script)")
    ap.add_argument("--chease-namelist", default=None,
                    help=f"CHEASE namelist (default {NSTX_NAMELIST} in CHEASEBS_PATH)")
    ap.add_argument("--warmup", type=int, default=None, metavar="N",
                    help="amplitude_warmup_iters: hold the driven amplitude "
                         "fixed for the first N outer iterations so the "
                         "bootstrap settles first. The NSTX template uses 2; "
                         "--warmup 0 turns the warm-up off")
    ap.add_argument("--stage-dir", default=None,
                    help="where to write the staged case directory "
                         "(default <outroot>/base_<shot>)")
    ap.add_argument("--scratch-run", action="store_true",
                    help="solve under the TPED scratch root and copy results "
                         "back, instead of running in --outroot. Off by default: "
                         "a run whose record is in scratch loses the record when "
                         "scratch is purged, and the comparison loses its "
                         "iteration rows with it")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage and scale, report the gradients, solve nothing")
    ap.add_argument("rest", nargs="*", default=[],
                    help="extra flags forwarded verbatim to the DIII-D scan "
                         "script (put them after a bare --)")
    args = ap.parse_args(argv)

    topped, midped = RAMP_WINDOWS[args.shot]
    if args.rhot_topped is not None:
        topped = args.rhot_topped
    if args.rhot_midped is not None:
        midped = args.rhot_midped

    scan = load_selfscaled()
    stamp = scan.datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    outroot = args.outroot or os.path.join(
        os.environ.get("SCRATCH", os.getcwd()), "NSTX_omn_omt_cheaseBS",
        f"{args.shot}_{stamp}")
    outroot = os.path.abspath(os.path.expandvars(os.path.expanduser(outroot)))
    tag = "".join(("_t%g" % args.scale_t if args.scale_t is not None else "",
                   "_n%g" % args.scale_n if args.scale_n is not None else ""))
    stage = os.path.abspath(os.path.expandvars(
        args.stage_dir
        or os.path.join(outroot, f"base_{args.shot}{tag}".replace(".", "p"))))

    print(f"=== NSTX omn_omt scan: shot {args.shot} ===")
    print(f"discharge : {os.path.join(discharge_root(), str(args.shot))}")
    print(f"ramp      : topped={topped}, midped={midped}"
          f"{' (overridden)' if (args.rhot_topped, args.rhot_midped) != (None, None) else ''}")
    print(f"template  : {args.cheasebs_config}")
    print(f"outroot   : {outroot}")

    case_dir, notes = stage_case_dir(args.shot, stage,
                                     scale_t=args.scale_t, scale_n=args.scale_n)
    print(f"staged    : {case_dir}")
    for note in notes:
        print(f"            {note}")

    suffix = scale_suffix(args.scale_t, args.scale_n)
    if suffix:
        # A value scaling is not one of the campaign's methods, and it must not
        # be mistaken for one: the campaign derives reference_* from the case
        # directory's own base profiles, so a scaled base would be replayed
        # against itself. run_hatch_cheasebs.py already solves exactly this
        # layout -- a suffixed set against the bare one -- so it does the solve.
        hatch = os.path.join(os.path.dirname(HERE), "hatch_compare",
                             "run_hatch_cheasebs.py")
        if not os.path.isfile(hatch):
            raise SystemExit(f"cannot find the hatch runner at {hatch}")
        cmd = [sys.executable, "-u", hatch,
               "--root", case_dir,
               "--stem", suffix.lstrip("_"),
               "--config", args.cheasebs_config,
               "--outroot", outroot]
        if args.dry_run:
            cmd.append("--list")
        cmd += [a for a in args.rest if a != "--"]
        print(f"solver    : {args.cheasebs_config}")
        print(f"handing off to run_hatch_cheasebs.py (value scaling, replayed "
              f"against the unscaled reference)\n{' '.join(cmd)}\n", flush=True)
        return subprocess.call(cmd)

    namelist = resolve_namelist(args.chease_namelist)
    if namelist is None and not args.dry_run:
        raise SystemExit(
            f"could not locate {NSTX_NAMELIST}: set CHEASEBS_PATH in the TPED "
            f"user config or pass --chease-namelist explicitly")

    inner = ["--case-dir", case_dir,
             "--method", "omn_omt",
             "--rhot-topped", str(topped),
             "--rhot-midped", str(midped),
             "--cheasebs-config", args.cheasebs_config,
             "--outroot", outroot]
    if namelist:
        inner += ["--chease-namelist", namelist]
    if args.warmup is not None:
        inner += ["--amplitude-warmup-iters", str(args.warmup)]
    if not args.scratch_run:
        inner += ["--in-place"]
    if args.dry_run:
        inner += ["--dry-run"]
    for pair in args.pair or []:
        inner += ["--pair", pair]
    radii = ANALYSIS_RADII.get(args.shot, ())
    if radii and not any(a == "--analysis-radii" for a in args.rest):
        inner += ["--analysis-radii"] + [str(r) for r in radii]
    inner += [a for a in args.rest if a != "--"]

    print("forwarding: " + " ".join(inner) + "\n", flush=True)
    return scan.main(inner)


if __name__ == "__main__":
    raise SystemExit(main())
