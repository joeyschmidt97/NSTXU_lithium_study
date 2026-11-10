"""Check that GENE vsp files still read correctly, e.g. after trim_vsp.

Per file, every check is printed PASS/FAIL:

    layout      size is a whole number of blocks; Fortran record markers of every
                block are (8, 8, 8*D, 8*D)
    times       GeneVsp finds one time per block, finite and strictly increasing
    xarray      GeneVsp.vsp_filepath_to_xarray('last', 'all', 'all') gives
                5 x n_spec variables of shape (1, nw0, nv0, nz0), all finite
                (files of <= ALL_MAX blocks also read every dump this way)
    vspace      plot_corners.vspace_edges (the corners vspace-edge check) returns
                ratios for every species, finite and within [0, 1]

With --ref (one file only), an untrimmed copy of the same run, also:

    ref-bytes   the file's blocks are byte-identical to the ref's last blocks
    ref-times   the file's times equal the ref's last times
    ref-xarray  the last-time xarrays are equal
    ref-vspace  vspace_edges are equal

Exit status 1 if any check fails on any file.

    # one trimmed file against an untrimmed copy of it
    python -m NSTXU_lithium_study.scripts.check_vsp_read /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129015/X_corners_backup/x0_0.8425/Teh0.80_Tew0.80_neh0.80_new0.80/scanfiles0000/vsp_0002 --ref /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129015/corners/x0_0.8425/Teh0.80_Tew0.80_neh0.80_new0.80/scanfiles0000/vsp_0002
    # every vsp file under a dir (no ref)
    python -m NSTXU_lithium_study.scripts.check_vsp_read /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129015/X_corners_backup
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys

import numpy as np

from NSTXU_lithium_study.scripts.trim_vsp import (edges, expected_markers, find_vsp, markers,
                                                  quiet_reader)

ALL_MAX = 10        # files with at most this many blocks are also read at every time


class Checks:
    def __init__(self):
        self.failed = 0

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:10s} {detail}")
        self.failed += not ok
        return ok


def last_blocks_sha1(path: str, block: int, k: int) -> str:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(size - k * block)
        return hashlib.sha1(f.read(k * block)).hexdigest()


def finite_edges(e) -> bool:
    ratios, _ = e
    return bool(ratios) and all(math.isfinite(v) and 0 <= v <= 1 for r in ratios.values() for v in r)


def check(path: str, ref: str | None, chk: Checks) -> None:
    print(f"\n{path}")
    r = quiet_reader(path)
    if not chk("params", r.param_valid, "parameters file beside it"):
        return
    d, block = r.data_size, r.block_bytes
    size = os.path.getsize(path)
    n = size // block
    whole = size % block == 0 and n > 0
    bad = []
    if whole:
        with open(path, "rb") as f:
            bad = [i for i in range(n) if markers(f, i * block, block) != expected_markers(d)]
    chk("layout", whole and not bad,
        f"{size} bytes = {n} blocks of {block}" + (f", bad markers in blocks {bad[:5]}" if bad else "")
        if whole else f"{size} bytes is not a whole number of {block}-byte blocks")

    t = r.all_time_values
    chk("times", len(t) == n and all(map(math.isfinite, t)) and all(np.diff(t) > 0),
        f"{len(t)} times, {t[:3]}{' ...' if len(t) > 3 else ''}")

    x = r.vsp_filepath_to_xarray("last", "all", "all")
    shape = (1, r.nw, r.nv, r.nz)
    n_var = 5 * r.n_spec
    ok = x is not None and len(x.data_vars) == n_var and all(
        x[v].shape == shape and np.isfinite(x[v].values).all() for v in x.data_vars)
    chk("xarray", ok, f"{0 if x is None else len(x.data_vars)}/{n_var} vars, shape {shape}, all finite")
    if n <= ALL_MAX:
        xa = r.vsp_filepath_to_xarray("all", "all", "all")
        chk("xarray-all", xa is not None and xa.sizes.get("vsp_time") == n,
            f"{None if xa is None else xa.sizes.get('vsp_time')} of {n} dumps read")

    e = edges(path)
    if e is not None:
        chk("vspace", finite_edges(e), f"{e}")

    if ref:
        rr = quiet_reader(ref)
        nr = os.path.getsize(ref) // block
        chk("ref-bytes", rr.block_bytes == block and nr >= n
            and last_blocks_sha1(ref, block, n) == last_blocks_sha1(path, block, n),
            f"last {n} of {nr} ref blocks")
        chk("ref-times", rr.all_time_values[-n:] == t, f"ref last {rr.all_time_values[-n:]}")
        xr_ = rr.vsp_filepath_to_xarray("last", "all", "all")
        chk("ref-xarray", x is not None and xr_ is not None and all(
            np.array_equal(x[v].values, xr_[v].values) for v in x.data_vars)
            and float(x.vsp_time[-1]) == float(xr_.vsp_time[-1]), "last-time arrays equal")
        if e is not None:
            er = edges(ref)
            chk("ref-vspace", er == e, f"ref {er}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="vsp files, or dirs searched for vsp_NNNN / vsp.dat")
    p.add_argument("--ref", help="untrimmed copy of the (single) file, compared byte by byte")
    a = p.parse_args(argv)
    files = find_vsp(a.paths)
    if a.ref and len(files) != 1:
        p.error(f"--ref needs exactly one file, got {len(files)}")
    chk = Checks()
    for path in files:
        check(path, a.ref, chk)
    print(f"\n{len(files)} files, {chk.failed} failed checks: {'PASS' if not chk.failed else 'FAIL'}")
    return 1 if chk.failed else 0


if __name__ == "__main__":
    sys.exit(main())
