"""Frozen scrambled-Sobol grading slice: independent test equilibria for the VURS campaign.

v2 (2026-10-21). One 4D design per shot over the four mtanh scalings in
sparse_scanner's axes (0.8-1.2, linear): the first --n points of an
Owen-scrambled Sobol sequence (scipy.stats.qmc.Sobol), seeded from
SeedSequence(ENTROPY, spawn_key=(shot,)), so the points never depend on any
sampler's data. kymin is not a Sobol axis: every slice equilibrium runs the
full corner grid corner_gene_scan.KYMIN as one 8-way scanlist job, the same
job shape as a corner, so gamma_max and the spectrum come out directly.

Two waves per shot. Points 0..WAVE-1 are wave 1, run first; points WAVE..n-1
are wave 2, run only if the extension rule in the manifest fires. The first
2^m points of a Sobol sequence are balanced, so stopping after either wave
leaves a balanced design. Wave 1 runs at the Te_top Chapman radius; other
radii may reuse the same equilibria.

The slice grades the VURS-trained mode map without bias. Its outcomes must not
enter training, acquisition or tuning before grading; after grading they join
the database.

Profile scalings are also given snapped to ped_grid's two-decimal case names,
the values actually run.

    python -m NSTXU_lithium_study.scripts.sobol_slice            # write the frozen CSV + manifest
    python -m NSTXU_lithium_study.scripts.sobol_slice --check    # regenerate and compare to the frozen CSV

The frozen CSV is the record, not this script: a different scipy may scramble
differently, which --check reports instead of overwriting. v1 (192 points,
5D with kymin, per radius role) was superseded before any run; its generator
is at commit b016adc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import scipy
from scipy.stats import qmc

from TPED.projects.GENE_pipelines.scripts.sparse_scanner import DEFAULT_AXES
from NSTXU_lithium_study.scripts.corner_gene_scan import KYMIN

#: Seed entropy: the date v2 was frozen (decision 2026-10-21).
ENTROPY = 20261021
SHOTS = (129015, 129038, 132543, 132588)
AXES = {a: b for a, b in DEFAULT_AXES.items() if a != "kymin"}
#: Points per shot in wave 1; the rest of --n is wave 2.
WAVE = 8
#: Extension rule, fixed before any slice outcome exists.
EXTENSION_RULE = ("Run wave 2 (points 8-15 per shot) only if, after wave 1, the pooled "
                  "95% interval half-width on the Stage 1 primary score exceeds the "
                  "threshold set in the comparison manifest before wave-1 grading "
                  "(proposed 0.15 on the label-agreement rate). The rule reads interval "
                  "width only, never which sampler leads.")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "sobol_grading_slice")
CSV = os.path.join(OUT_DIR, "sobol_slice_v2.csv")
MANIFEST = os.path.join(OUT_DIR, "sobol_slice_v2_manifest.json")


def design(n):
    """Every shot's design as one table."""
    rows = []
    for shot in SHOTS:
        seed = np.random.SeedSequence(ENTROPY, spawn_key=(shot,))
        u = qmc.Sobol(d=len(AXES), scramble=True,
                      seed=np.random.default_rng(seed)).random(n)
        df = pd.DataFrame({f"u_{a}": u[:, j] for j, a in enumerate(AXES)})
        for j, (name, (lo, hi)) in enumerate(AXES.items()):
            df[name] = lo + u[:, j] * (hi - lo)
            df[f"{name}_snapped"] = np.round(df[name], 2)
        df.insert(0, "wave", np.where(np.arange(n) < WAVE, 1, 2))
        df.insert(0, "point", np.arange(n))
        df.insert(0, "shot", shot)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def write_csv(df, path):
    df.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=16,
                    help="points frozen per shot, both waves; a power of two")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare to the frozen CSV; write nothing")
    args = ap.parse_args(argv)
    if args.n & (args.n - 1) or args.n < 2 * WAVE:
        raise SystemExit(f"--n {args.n} must be a power of two >= {2 * WAVE} "
                         "(Sobol balance needs 2^m; two waves of WAVE)")

    df = design(args.n)
    if args.check:
        frozen = pd.read_csv(CSV)
        same = frozen.shape == df.shape and np.allclose(
            frozen.select_dtypes("number"), df.select_dtypes("number"), rtol=0, atol=1e-9)
        print(f"{'MATCH' if same else 'MISMATCH'}: {CSV} (scipy {scipy.__version__})")
        return 0 if same else 1

    if os.path.exists(CSV):
        raise SystemExit(f"{CSV} exists: the slice is frozen. Use --check, or a new version file.")
    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(df, CSV)
    manifest = {
        "frozen": date.today().isoformat(),
        "version": 2,
        "supersedes": "sobol_slice_v1.csv (b016adc), superseded 2026-10-21 before any run: "
                      "one kymin per point gave no gamma_max or spectrum, and 192 "
                      "equilibria was too many",
        "purpose": "independent grading slice for the VURS campaign; post-draft test",
        "n_per_shot": args.n,
        "wave_1_points_per_shot": WAVE,
        "extension_rule": EXTENSION_RULE,
        "shots": list(SHOTS),
        "radius": "wave 1 at the Te_top Chapman radius; ne_top / steep optional on the same equilibria",
        "kymin_per_equilibrium": list(KYMIN),
        "axes": {a: {"bounds": list(b), "scale": "linear"} for a, b in AXES.items()},
        "generator": "scipy.stats.qmc.Sobol(d=4, scramble=True) (Owen scrambling)",
        "seed": f"np.random.default_rng(np.random.SeedSequence({ENTROPY}, spawn_key=(shot,)))",
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "csv_sha256": sha256(CSV),
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {len(df)} points: {CSV}\nsha256 {manifest['csv_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
