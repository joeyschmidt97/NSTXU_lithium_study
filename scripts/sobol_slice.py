"""Frozen scrambled-Sobol grading slice: independent test points for the VURS campaign.

One 5D design per (shot, radius role), in sparse_scanner's axes: the four mtanh
scalings on 0.8-1.2 (linear) and kymin on 0.05-2.0 (log). Each design is the
first --n points of an Owen-scrambled Sobol sequence (scipy.stats.qmc.Sobol),
seeded from SeedSequence(ENTROPY, spawn_key=(shot, role index)), so the points
never depend on any sampler's data.

The slice grades the VURS-trained mode map without bias. Its outcomes must not
enter training, acquisition or tuning before grading; after grading they join
the database. Designs are keyed by radius role (ne_top / Te_top / steep), not
by rho_tor value, so a later Chapman refit does not change the points.

Profile scalings are also given snapped to ped_grid's two-decimal case names,
the values actually run; kymin is exact.

    python -m NSTXU_lithium_study.scripts.sobol_slice            # write the frozen CSV + manifest
    python -m NSTXU_lithium_study.scripts.sobol_slice --check    # regenerate and compare to the frozen CSV

The frozen CSV is the record, not this script: a different scipy may scramble
differently, which --check reports instead of overwriting.
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

#: Seed entropy: the date the slice was frozen (decision 2026-10-20).
ENTROPY = 20261020
SHOTS = (129015, 129038, 132543, 132588)
ROLES = ("ne_top", "Te_top", "steep")
LOG_AXES = {"kymin"}

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "sobol_grading_slice")
CSV = os.path.join(OUT_DIR, "sobol_slice_v1.csv")
MANIFEST = os.path.join(OUT_DIR, "sobol_slice_v1_manifest.json")


def to_physical(u):
    """Unit-box points (n, 5) -> dict of physical axis columns, in DEFAULT_AXES order."""
    cols = {}
    for j, (name, (lo, hi)) in enumerate(DEFAULT_AXES.items()):
        if name in LOG_AXES:
            cols[name] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo)))
        else:
            cols[name] = lo + u[:, j] * (hi - lo)
    return cols


def design(n):
    """Every (shot, role) design as one table."""
    rows = []
    for shot in SHOTS:
        for k, role in enumerate(ROLES):
            seed = np.random.SeedSequence(ENTROPY, spawn_key=(shot, k))
            u = qmc.Sobol(d=len(DEFAULT_AXES), scramble=True,
                          seed=np.random.default_rng(seed)).random(n)
            df = pd.DataFrame({f"u_{a}": u[:, j] for j, a in enumerate(DEFAULT_AXES)})
            for name, v in to_physical(u).items():
                df[name] = v
                if name not in LOG_AXES:
                    df[f"{name}_snapped"] = np.round(v, 2)
            df.insert(0, "point", np.arange(n))
            df.insert(0, "role", role)
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
    ap.add_argument("--n", type=int, default=16, help="points per (shot, role); a power of two")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and compare to the frozen CSV; write nothing")
    args = ap.parse_args(argv)
    if args.n & (args.n - 1):
        raise SystemExit(f"--n {args.n} is not a power of two (Sobol balance needs 2^m)")

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
        "purpose": "independent grading slice for the VURS campaign; post-draft test",
        "n_per_design": args.n,
        "shots": list(SHOTS),
        "roles": list(ROLES),
        "axes": {a: {"bounds": list(b), "scale": "log" if a in LOG_AXES else "linear"}
                 for a, b in DEFAULT_AXES.items()},
        "generator": "scipy.stats.qmc.Sobol(d=5, scramble=True) (Owen scrambling)",
        "seed": f"np.random.default_rng(np.random.SeedSequence({ENTROPY}, spawn_key=(shot, role_index)))",
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
