"""Trim GENE vsp files to their last dump, in place, to free /pscratch space.

A vsp file is one fixed-size block per dump (every istep_vsp steps), each two
Fortran unformatted records:

    [int32 8] [float64 time] [int32 8]  [int32 D*8] [D float64 data] [int32 D*8]

with D = 5 * n_spec * nw0 * nv0 * nz0, so one block is 8*D + 24 bytes (17.7 MB
for the corner runs: n_spec 3, nw0 32, nv0 36, nz0 128) and a run with
istep_vsp = 400 to 750000 steps carries ~1900 of them (~33 GB). The only reader
of vsp in this repo is the corners vspace-edge check (plot_corners.vspace_edges),
which reads the last dump only. This keeps that last block and drops the rest:
the trimmed file is a valid one-dump vsp file that GeneVsp, vspace_edges and
run_cache read unchanged.

Per file: block size from the run's parameters_NNNN (TPED GeneVsp), record
markers checked on the first and last block, then the last block is copied over
block 0 and the file truncated to one block. No temporary copy is written (the
quota may already be full). Interrupted before the truncate, the file still ends
with its original last block, so a rerun gives the same result. After the trim,
block 0 is reread and its sha1 must match the last block's from before.

Skipped, with the reason in the log: no parameters file beside it, size not a
whole number of blocks (still running or cut off), unexpected record markers,
already one block, or modified within --min-age minutes (a running job).

--pilot FILE trims that one file with extra checks and prints them: the last
time and vspace_edges ratios before and after must be identical, and the
record markers are compared with the byte offset GeneVsp reads its data from.

Dry run is the default: nothing is changed without --apply. Every file is logged
to a TSV in $HOME (not /pscratch, which turns read-only over quota).

    # 1. pilot one file: trims it, prints before/after checks
    python -m NSTXU_lithium_study.scripts.trim_vsp --pilot /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129038/corners/x0_0.6737/Teh1.20_Tew0.80_neh0.80_new0.80/scanfiles0000/vsp_0001
    # 2. dry run a tree: what would be trimmed, and how much freed
    python -m NSTXU_lithium_study.scripts.trim_vsp /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129038/corners
    # 3. trim it
    python -m NSTXU_lithium_study.scripts.trim_vsp /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129038/corners --apply
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import sys
import time

from TPED.projects.GENE_sim_reader.src.GENE_vsp_data import GeneVsp

VSP_NAME = re.compile(r"^vsp(_\d{4}|\.dat)$")
TIME_REC = 8                    # bytes in the time record
HEAD = 4 + TIME_REC + 4 + 4     # bytes before the data in a block (= 20)
GENEVSP_DATA_OFFSET = 4         # where GeneVsp.vsp_filepath_to_dict starts reading data in a block


def find_vsp(roots: list[str]) -> list[str]:
    """Every vsp_NNNN / vsp.dat file below the roots (a root may be a file), sorted."""
    out = []
    for root in roots:
        if os.path.isfile(root):
            out.append(os.path.abspath(root))
            continue
        for d, _, files in os.walk(root):
            out += [os.path.join(d, f) for f in files if VSP_NAME.match(f)]
    return sorted(out)


def geometry(path: str) -> tuple[int, int] | str:
    """(data values per block D, block bytes) from the run's parameters, or a skip reason."""
    with open(os.devnull, "w") as null:                 # GeneBase prints when parameters are missing
        stdout, sys.stdout = sys.stdout, null
        try:
            reader = GeneVsp(path)
        finally:
            sys.stdout = stdout
    if not reader.param_valid:
        return "no parameters file"
    d = reader.data_size
    block = reader.block_bytes
    if block != 8 * d + 24:
        return f"GeneVsp block_bytes {block} != 8*D+24 = {8 * d + 24}"
    return d, block


def markers(f, offset: int, d: int, block: int) -> tuple[int, int, int, int]:
    """The four int32 record markers of the block at offset."""
    f.seek(offset)
    head = f.read(HEAD)
    f.seek(offset + block - 4)
    tail = f.read(4)
    a, = struct.unpack("<i", head[0:4])
    b, = struct.unpack("<i", head[12:16])
    c, = struct.unpack("<i", head[16:20])
    e, = struct.unpack("<i", tail)
    return a, b, c, e


def markers_ok(m: tuple[int, int, int, int], d: int) -> bool:
    return m == (TIME_REC, TIME_REC, 8 * d, 8 * d)


def block_time(f, offset: int) -> float:
    f.seek(offset + 4)
    return struct.unpack("<d", f.read(8))[0]


def inspect(path: str, min_age: float) -> dict:
    """Everything needed to decide on one file; 'skip' holds the reason when it is left alone."""
    info = {"path": path, "size": os.path.getsize(path), "n_blocks": 0, "block": 0,
            "last_time": float("nan"), "skip": ""}
    age_min = (time.time() - os.path.getmtime(path)) / 60
    if age_min < min_age:
        info["skip"] = f"modified {age_min:.0f} min ago (< --min-age {min_age:g})"
        return info
    geo = geometry(path)
    if isinstance(geo, str):
        info["skip"] = geo
        return info
    d, block = geo
    info["d"], info["block"] = d, block
    size = info["size"]
    if size < block:
        info["skip"] = f"smaller than one block ({size} < {block} bytes)"
        return info
    if size % block:
        info["skip"] = f"not a whole number of blocks ({size} bytes, block {block}): running or cut off"
        return info
    n = size // block
    info["n_blocks"] = n
    with open(path, "rb") as f:
        for i in (0, n - 1):
            m = markers(f, i * block, d, block)
            if not markers_ok(m, d):
                info["skip"] = f"record markers of block {i} are {m}, expected {(TIME_REC, TIME_REC, 8 * d, 8 * d)}"
                return info
        info["last_time"] = block_time(f, (n - 1) * block)
    if n == 1:
        info["skip"] = "already one block"
    return info


def trim(info: dict) -> None:
    """Copy the last block over block 0, truncate to one block, check block 0's sha1."""
    path, n, block = info["path"], info["n_blocks"], info["block"]
    with open(path, "r+b") as f:
        f.seek((n - 1) * block)
        last = f.read(block)
        if len(last) != block:
            raise OSError(f"short read of last block ({len(last)} of {block} bytes)")
        want = hashlib.sha1(last).hexdigest()
        f.seek(0)
        f.write(last)
        f.flush()
        os.fsync(f.fileno())
        f.truncate(block)
        f.flush()
        os.fsync(f.fileno())
    with open(path, "rb") as f:
        got = hashlib.sha1(f.read()).hexdigest()
    if os.path.getsize(path) != block or got != want:
        raise OSError(f"check failed after trim: size {os.path.getsize(path)}, sha1 {got} != {want}")


def edges(path: str):
    """plot_corners.vspace_edges for a vsp_NNNN (None for vsp.dat)."""
    from NSTXU_lithium_study.scripts.corners.plots import plot_corners
    if not re.search(r"vsp_\d{4}$", path):
        return None
    return plot_corners.vspace_edges(re.sub(r"vsp_(\d{4})$", r"parameters_\1", path))


def pilot(path: str, min_age: float) -> int:
    info = inspect(path, min_age)
    print(f"file      {path}")
    print(f"size      {info['size'] / 1e9:.2f} GB, {info['n_blocks']} blocks of {info['block'] / 1e6:.2f} MB")
    if info["skip"]:
        print(f"SKIPPED   {info['skip']}")
        return 1
    d, block = info["d"], info["block"]
    with open(path, "rb") as f:
        m = markers(f, 0, d, block)
    print(f"markers   {m} (time record {TIME_REC} bytes, data record {8 * d} bytes): OK")
    print(f"offsets   data record starts at byte {HEAD} of each block; GeneVsp reads data from byte "
          f"{GENEVSP_DATA_OFFSET}" + ("" if GENEVSP_DATA_OFFSET == HEAD else
          f", so its arrays are shifted by {(HEAD - GENEVSP_DATA_OFFSET) // 8} values "
          "(first values = time + markers). Trimming keeps the bytes it reads unchanged; check the reader."))
    t0, e0 = info["last_time"], edges(path)
    print(f"before    last time {t0!r}")
    print(f"          vspace_edges {e0}")
    trim(info)
    with open(path, "rb") as f:
        t1 = block_time(f, 0)
    e1 = edges(path)
    print(f"after     size {os.path.getsize(path) / 1e6:.2f} MB, last time {t1!r}")
    print(f"          vspace_edges {e1}")
    same = t1 == t0 and e1 == e0
    print("PASS: last time and vspace_edges identical, block sha1 matched" if same
          else "FAIL: before/after differ (the trimmed file still holds the original last block)")
    return 0 if same else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="*", help="dirs searched for vsp_NNNN / vsp.dat (or single files)")
    p.add_argument("--apply", action="store_true", help="trim (default: dry run, nothing changed)")
    p.add_argument("--pilot", metavar="FILE", help="trim this one vsp file with before/after checks")
    p.add_argument("--min-age", type=float, default=60.0,
                   help="skip files modified within this many minutes (default 60)")
    p.add_argument("--log", default=None,
                   help="TSV log path (default $HOME/trim_vsp_<date_time>.tsv)")
    a = p.parse_args(argv)

    if a.pilot:
        return pilot(os.path.abspath(a.pilot), a.min_age)
    if not a.roots:
        p.error("give one or more dirs, or --pilot FILE")

    log = a.log or os.path.join(os.path.expanduser("~"), time.strftime("trim_vsp_%Y-%m-%d_%H%M%S.tsv"))
    files = find_vsp(a.roots)
    print(f"{len(files)} vsp files under {', '.join(a.roots)}  ({'APPLY' if a.apply else 'dry run'})")
    freed, counts = 0, {}
    with open(log, "w") as out:
        out.write("status\tpath\tn_blocks\told_bytes\tnew_bytes\tlast_time\tnote\n")
        for i, path in enumerate(files, 1):
            try:
                info = inspect(path, a.min_age)
            except OSError as exc:
                info = {"path": path, "size": 0, "n_blocks": 0, "block": 0,
                        "last_time": float("nan"), "skip": f"unreadable: {exc}"}
            status, new, note = "skip", info["size"], info["skip"]
            if not info["skip"]:
                status, new = ("trimmed" if a.apply else "would-trim"), info["block"]
                if a.apply:
                    try:
                        trim(info)
                    except OSError as exc:
                        status, new, note = "error", os.path.getsize(path), str(exc)
                if status != "error":
                    freed += info["size"] - new
            counts[status] = counts.get(status, 0) + 1
            out.write(f"{status}\t{path}\t{info['n_blocks']}\t{info['size']}\t{new}\t"
                      f"{info['last_time']!r}\t{note}\n")
            out.flush()
            if status != "skip" or i % 200 == 0:
                print(f"  [{i}/{len(files)}] {status:10s} {info['size'] / 1e9:7.2f} GB  {path}"
                      + (f"  ({note})" if note else ""))
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print(f"{'freed' if a.apply else 'would free'} {freed / 1e12:.3f} TB   log: {log}")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
