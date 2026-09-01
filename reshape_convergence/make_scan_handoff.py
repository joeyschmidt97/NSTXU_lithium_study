#!/usr/bin/env python3
"""Build a self-contained scaled-profile package on SCRATCH, for someone else to run.

The point is to hand over the *inputs* to the cheaseBS scan -- the frozen source
equilibrium and the scaled GENE profiles -- so a second person can reproduce the
reconstruction independently and look at the driven amplitude and the Ip relative
error without needing this repo, its fits, or its campaign machinery. Nothing
here runs CHEASE; it writes files.

WHAT GOES IN

Per discharge:

    <shot>/base/                    the frozen source EQDSK + unscaled profiles
    <shot>/ne_ped_scale_0.700/      profiles_{e,i,z} at that scale factor
    <shot>/ne_ped_scale_0.900/      ...
    <shot>/Te_ped_scale_1.300/      ...

The EQDSK is written once, into base/, because the boundary is frozen across the
whole scan -- every case is the same geometry with different profiles, and
copying it per case would imply otherwise.

WHAT THE SCALE FACTOR MEANS -- READ THIS BEFORE QUOTING "+/-30%"

The axes are mtanh **step-amplitude** scale factors, not pedestal-top fractions:
`apply_mtanh_full(scale_height=1.3)` multiplies the fitted step amplitude a_y0,
and the pedestal top is y_sep + a_y0 + core.

Under the Stefanikova FULL-profile form the two turn out to be close but not
equal -- measured on 132543, scale 0.70/1.30 gives a pedestal-top ratio of
0.732/1.269 for ne and 0.717/1.283 for Te, a slope near 0.9 rather than 1.0.
(The much weaker 0.478/0.720 slopes on record from 2026-08-19 belong to
`apply_mtanh_ped`, the pedestal-only form retired on 2026-08-22, and do not
apply here.)

So the manifest records, for every case, the measured pedestal-top ratio
alongside the requested scale factor. Quote either; do not convert between them
by assumption, and do not reuse one discharge's slope for another.

Scaling uses the Stefanikova full-profile form (`apply_mtanh_full`), the same
transform the campaign froze on 2026-08-23, with quasineutrality enforced so ni
and nz follow ne.

USAGE

    python make_scan_handoff.py                       # all four shots, SCRATCH
    python make_scan_handoff.py --shots 132588        # one shot
    python make_scan_handoff.py --outroot /some/dir   # somewhere other than SCRATCH
    python make_scan_handoff.py --scales 0.7 0.9 1.1 1.3
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import sys
import traceback

# Headless: the fit path can reach for pyplot, and there is no display on a
# login node.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent,
                        *pathlib.Path(__file__).resolve().parents]
            if (p / "pedestal_scan.py").exists())
sys.path.insert(0, str(ROOT))

from pedestal_scan import ANALYSIS_RADII, AXES, Campaign, DISCHARGES  # noqa: E402

from TPED.projects.discharge_tools.src.writers.gene_writer import (  # noqa: E402
    write_gene_profiles)

DEFAULT_SCALES = (0.7, 0.9, 1.1, 1.3)
DEFAULT_AXES = ("ne_ped_scale", "Te_ped_scale")

README = """# Scaled-profile package for the cheaseBS scan

Generated {stamp} from `NSTXU_lithium_study/reshape_convergence/make_scan_handoff.py`.

## What this is

The **inputs** to a cheaseBS profile-scaling scan on {n_shots} NSTX discharge(s),
so the reconstruction can be reproduced independently. No equilibrium in here has
been reconstructed -- the EQDSK is the source EFIT, unmodified.

## Layout

```
<shot>/base/                   source EQDSK + unscaled GENE profiles
<shot>/<axis>_<scale>/         profiles_{{e,i,z}} at that scale factor
manifest.json                  every case, with the measured change it produced
```

The EQDSK lives only in `base/` and applies to every case: the boundary is frozen
across the scan, and only the profiles differ.

## What the scale factors mean

The axes multiply the **mtanh step amplitude** of a Stefanikova full-profile fit,
not the pedestal-top value: the pedestal top is `y_sep + a_y0 + core` and only
`a_y0` is scaled. The two are close but not equal -- on 132543 a scale factor of
1.30 moves the pedestal top by 26.9% (ne) and 28.3% (Te), and the mapping differs
per discharge.

`manifest.json` records both numbers per case: `scale` as requested, and
`ped_top_ratio` as measured from the scaled profile. Quote whichever you need;
they are not interchangeable.

Density scaling rewrites `ni` and `nz` through quasineutrality (`qz = 6`), so the
ion channel follows `ne` and needs no axis of its own. Temperature scaling moves
`Te` only -- `Ti` is held, which is why an equal scale factor on the two axes is
not an equal perturbation to the total thermal pressure.

## Running cheaseBS on a case

```bash
python run_chease_iterative_profiles.py \\
  --eqdsk                       <shot>/base/<gfile> \\
  --reference-electron-profile  <shot>/base/profiles_e \\
  --reference-deuterium-profile <shot>/base/profiles_i \\
  --reference-carbon-profile    <shot>/base/profiles_z \\
  --electron-profile            <shot>/<case>/profiles_e \\
  --deuterium-profile           <shot>/<case>/profiles_i \\
  --carbon-profile              <shot>/<case>/profiles_z \\
  --baseline-dir  <workdir>/baseline \\
  --output-dir    <workdir>/run \\
  --coordinate rhop --replay-representation istar \\
  --istar-source-grid-mode workflow --istar-regularize \\
  --bootstrap-mix 0.1 --istar-mix 0.02 --enforce-qspec \\
  --max-iter 45
```

**The reference profiles must be the `base/` ones, not the case's own.** Pointing
both at the same files rebuilds the fast-pressure decomposition from the scaled
profiles, which floors `p_total` at the scaled pressure and silently discards the
perturbation. That failure produced twenty converged-looking solves here in
August 2026 before it was found.

## Two things worth knowing before reading the results

- **`istar_mix` is a stability control, not just a rate.** At the older default
  0.05 the loop diverges on 132588 `ne` 0.70 to a +181% Ip error. 0.02 converges
  it to +0.47%. It is not universal: at 0.02, 129015 and 132543 still drift.
- **The driven amplitude never moves in our runs.** `amplitude_warmup_iters = 2`
  makes the first two amplitude-history entries identical, and the secant step in
  `next_amplitude_guess` is proportional to their difference, so it is pinned at
  1.0 forever. Since the amplitude is the only actuator on Ip, that is the leading
  suspect for the several-percent Ip offsets. Setting `amplitude_warmup_iters: 0`
  is the test.

## Discharges

{shot_lines}
"""


def case_dirname(axis, scale):
    return "%s_%.3f" % (axis, scale)


def write_base(disc, shot_dir, quiet=False):
    """Source EQDSK plus the unscaled profiles, in base/."""
    base = os.path.join(shot_dir, "base")
    os.makedirs(base, exist_ok=True)
    # run_cheasebs=False writes the frozen raw geometry and nothing else: the
    # transforms in this script never touch it, and the package must not imply
    # a reconstruction happened.
    gfile = disc.phys.output_gfile(savedir=base, run_cheasebs=False)
    write_gene_profiles(disc.phys.ds, base)
    if not quiet:
        print("    base: %s + profiles_{e,i,z}" % os.path.basename(gfile))
    return gfile


def write_case(disc, shot_dir, axis, scale, quiet=False):
    """One scaled case. Returns the manifest record, including what it measured."""
    name = case_dirname(axis, scale)
    cdir = os.path.join(shot_dir, name)
    os.makedirs(cdir, exist_ok=True)

    scaled = disc.scaled(axis, scale)
    write_gene_profiles(scaled.ds, cdir)

    rec = {"axis": axis, "scale": scale, "dir": name,
           "var": AXES[axis][0], "unit": AXES[axis][3]}
    # The measured change, so the package never has to be described in scale
    # factors alone. metric() re-reads the pedestal top off the scaled profile.
    try:
        nominal = disc.metric(axis, 1.0)
        moved = disc.metric(axis, scale, q=scaled)
        rec["ped_top_nominal"] = float(nominal)
        rec["ped_top_scaled"] = float(moved)
        rec["ped_top_ratio"] = float(moved / nominal) if nominal else None
    except Exception as exc:                                    # noqa: BLE001
        rec["metric_error"] = "%s: %s" % (type(exc).__name__, exc)

    if not quiet:
        r = rec.get("ped_top_ratio")
        print("    %-22s ped-top ratio %s"
              % (name, "--" if r is None else "%.4f" % r))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Write a scaled-profile handoff package to SCRATCH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="+", default=sorted(DISCHARGES),
                    choices=sorted(DISCHARGES))
    ap.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES),
                    choices=sorted(AXES))
    ap.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES),
                    help="mtanh step-amplitude scale factors, NOT pedestal-top "
                         "fractions; the manifest records both")
    ap.add_argument("--outroot", default=None,
                    help="parent directory (default: $SCRATCH, else $PSCRATCH, "
                         "else the current directory)")
    ap.add_argument("--name", default=None,
                    help="package directory name (default: "
                         "cheasebs_scan_handoff_<stamp>)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace the package directory if it already exists")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    outroot = (args.outroot or os.environ.get("SCRATCH")
               or os.environ.get("PSCRATCH") or os.getcwd())
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    pkg = os.path.join(outroot, args.name or ("cheasebs_scan_handoff_" + stamp))
    if os.path.exists(pkg):
        if not args.overwrite:
            raise SystemExit(
                "%s already exists; pass --overwrite or --name to pick another"
                % pkg)
        shutil.rmtree(pkg)
    os.makedirs(pkg)

    print("=== scaled-profile handoff package ===")
    print("destination : %s" % pkg)
    print("shots       : %s" % args.shots)
    print("axes        : %s" % args.axes)
    print("scales      : %s  (mtanh step amplitude, not pedestal-top fraction)"
          % args.scales)
    print("cases       : %d per shot, %d total\n"
          % (len(args.axes) * len(args.scales),
             len(args.shots) * len(args.axes) * len(args.scales)), flush=True)

    # Campaign(shots=...) rather than Campaign(): the bare form loads and fits
    # all four discharges regardless, which is minutes of work per shot.
    print("loading and fitting %d discharge(s) ..." % len(args.shots), flush=True)
    camp = Campaign(shots=args.shots)
    print("  done\n", flush=True)

    manifest = {"created": stamp, "package": pkg, "axes": args.axes,
                "scales": args.scales,
                "scale_meaning": "mtanh step-amplitude factor via "
                                 "apply_mtanh_full; ped_top_ratio is the "
                                 "measured pedestal-top change it produced",
                "quasineutrality": "enforced, qz = 6",
                "shots": {}}
    failed = []

    for shot in args.shots:
        print("--- %d ---" % shot, flush=True)
        shot_dir = os.path.join(pkg, str(shot))
        os.makedirs(shot_dir, exist_ok=True)
        disc = camp[shot]
        rec = {"analysis_radii": list(ANALYSIS_RADII[shot]), "cases": []}
        try:
            rec["gfile"] = os.path.basename(write_base(disc, shot_dir, args.quiet))
        except Exception:                                        # noqa: BLE001
            # One discharge failing must not take the rest of the package with
            # it; the manifest records what is actually on disk.
            print("!!! %d base write RAISED" % shot)
            traceback.print_exc()
            failed.append(shot)
            manifest["shots"][str(shot)] = {**rec, "error": "base write failed"}
            continue

        for axis in args.axes:
            for s in args.scales:
                try:
                    rec["cases"].append(write_case(disc, shot_dir, axis, s,
                                                   args.quiet))
                except Exception as exc:                          # noqa: BLE001
                    print("!!! %d %s %.3f RAISED: %s"
                          % (shot, axis, s, exc))
                    traceback.print_exc()
                    failed.append(shot)
                    rec["cases"].append({"axis": axis, "scale": s,
                                         "error": "%s: %s"
                                                  % (type(exc).__name__, exc)})
        manifest["shots"][str(shot)] = rec
        print(flush=True)

    shot_lines = []
    for shot in args.shots:
        rec = manifest["shots"].get(str(shot), {})
        ratios = [c.get("ped_top_ratio") for c in rec.get("cases", [])
                  if c.get("ped_top_ratio")]
        shot_lines.append(
            "- **%d** — EQDSK `%s`, analysis radii %s, %d case(s)%s"
            % (shot, rec.get("gfile", "MISSING"),
               rec.get("analysis_radii"), len(rec.get("cases", [])),
               "" if not ratios else
               ", pedestal-top ratios %.3f–%.3f" % (min(ratios), max(ratios))))

    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README.format(stamp=stamp, n_shots=len(args.shots),
                               shot_lines="\n".join(shot_lines)))
    with open(os.path.join(pkg, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    n_cases = sum(len(r.get("cases", [])) for r in manifest["shots"].values())
    n_err = sum(1 for r in manifest["shots"].values()
                for c in r.get("cases", []) if "error" in c)
    print("=== done ===")
    print("  %d case(s) written, %d failed" % (n_cases - n_err, n_err))
    print("  %s" % pkg)
    print("  README.md and manifest.json describe the package; the manifest "
          "carries the measured pedestal-top ratio per case")
    if failed:
        print("  PROBLEM shots: %s" % sorted(set(failed)))
    print("\n  tar it for the handoff:")
    print("    tar czf %s.tar.gz -C %s %s"
          % (os.path.basename(pkg), os.path.dirname(pkg), os.path.basename(pkg)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
