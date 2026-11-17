"""Time each stage of the field fit on real GENE field files, to find where a fit stalls.

field_omega.run (what update_results calls first) fits every ky's field time
trace with TPED's GeneField. This times the same work stage by stage for each
field file, printing (and flushing) the stage name before it starts, so a stall
shows exactly where it sits:

    open     GeneField(): parameters, then fieldlib lists every time step
             (one small read + seek per step on /pscratch)
    sweep    set_time + phi() over the fit window (one read per step), as the fit does
    lookup   the old tfld.index() lookups alone over that window (pure CPU, the
             O(n^2) cost TPED 1985dbbf removed), only with --old
    fit      field_omega.fit itself, no cache, with its gamma/omega

and a final table: size, time steps, seconds per stage. It never writes to the
omega files or the fit cache. --timeout stops a file after that many seconds
(SIGALRM, Linux) and moves on, so one hung file does not hold up the rest.

First it checks the TPED in use has the O(1) set_time (fieldfile.tind_of); if it
says OLD, pull TPED (gene_datatree) before trusting the numbers.

    python -m NSTXU_lithium_study.scripts.corners.field_omega.time_field_fit --shot 129015 --x0 0.8425 --tag Teh0.80_Tew0.80_neh1.20_new1.20
    python -m NSTXU_lithium_study.scripts.corners.field_omega.time_field_fit --shot 129015 --x0 0.8425 --ky 1.5 --old
    python -m NSTXU_lithium_study.scripts.corners.field_omega.time_field_fit --shot 129015 --x0 0.8425 --largest 5 --timeout 300
    python -m NSTXU_lithium_study.scripts.corners.field_omega.time_field_fit --path /pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129015/corners/x0_0.8425/Teh0.80_Tew0.80_neh1.20_new1.20
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import signal
import subprocess
import time

import numpy as np

from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField
from TPED.projects.GENE_sim_reader.utils import fieldlib
from NSTXU_lithium_study.scripts.corners.field_omega import field_omega as fo
from NSTXU_lithium_study.sparse_grid_results.update_results import load_entries, select


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout


def tped_check() -> str:
    """Where TPED is loaded from, its commit, and whether fieldfile has the O(1) lookup."""
    src = os.path.dirname(fieldlib.__file__)
    try:
        commit = subprocess.run(["git", "-C", src, "log", "-1", "--format=%h %s"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "?"
    new = "self.tind_of" in open(fieldlib.__file__).read()
    return (f"TPED {src}\n  commit {commit or '?'}\n  set_time: "
            + ("O(1) dict (new)" if new else "OLD list.index - pull TPED gene_datatree"))


def targets(args) -> list[tuple[str, str]]:
    """[(label, field path)] with a non-empty field file, filtered by --tag and --ky."""
    if args.path:
        _, found = fo.path_targets(args.path)
    else:
        found = []
        for e in select(load_entries(), [args.shot], args.x0):
            if e["cases"]:
                continue
            _, t = fo.corner_targets(e["diag_root"], (e["x0"],))
            found += t
    out = []
    for label, omega in found:
        field = re.sub(r"omega_(\d{4})$", r"field_\1", omega)
        if not os.path.isfile(field) or os.path.getsize(field) == 0:
            continue
        if args.tag and not any(t in label for t in args.tag):
            continue
        ky = fo.kymin_of(omega)
        if args.ky and not any(abs(ky - k) < 1e-6 for k in args.ky):
            continue
        name = os.path.dirname(label) if label.endswith(os.path.basename(omega)) else label
        out.append((f"{name or label} ky {ky:g}", field))
    if args.largest:
        out = sorted(out, key=lambda t: -os.path.getsize(t[1]))[:args.largest]
    return out


def stage(name, label):
    print(f"  {name:6} ...", end="", flush=True)
    return time.perf_counter()


def time_one(label, field, window, old) -> dict:
    row = {"label": label, "MB": os.path.getsize(field) / 1e6}
    print(f"{label}\n  {field}  ({row['MB']:.1f} MB)", flush=True)
    t0 = stage("open", label)
    with contextlib.redirect_stdout(io.StringIO()):
        gf = GeneField(field)
    row["open"] = time.perf_counter() - t0
    tf = gf.field.tfld
    lo = int(window[0] * len(tf))
    win = tf[lo:int(np.ceil(window[1] * len(tf)))] or tf[-1:]
    row["steps"], row["window"] = len(tf), len(win)
    print(f" {row['open']:.2f} s, {len(tf)} time steps, {len(win)} in the window", flush=True)

    t0 = stage("sweep", label)
    for t in win:
        gf.field.set_time(t)
        gf.field.phi()
    row["sweep"] = time.perf_counter() - t0
    print(f" {row['sweep']:.2f} s", flush=True)

    if old:
        t0 = stage("lookup", label)
        for t in win:
            tf.index(t)
        row["lookup"] = time.perf_counter() - t0
        print(f" {row['lookup']:.2f} s (old tfld.index, what the fix removed)", flush=True)

    t0 = stage("fit", label)
    _, stats = fo.fit(field, window)
    row["fit"] = time.perf_counter() - t0
    s = stats[fo.FIELDS]
    print(f" {row['fit']:.2f} s, gamma {s['gamma_avg']:.4f}, omega {s['omega_avg']:.4f}",
          flush=True)
    return row


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shot", type=int, default=None, help="discharge in gene_run_paths.json")
    ap.add_argument("--x0", type=float, nargs="+", default=None, help="radii [every one]")
    ap.add_argument("--path", default=None,
                    help="every field file at any depth under this dir instead (--shot/--x0 ignored)")
    ap.add_argument("--tag", nargs="+", default=None, help="only case dirs containing these")
    ap.add_argument("--ky", type=float, nargs="+", default=None, help="only these kymin")
    ap.add_argument("--largest", type=int, default=None, help="only the N largest field files")
    ap.add_argument("--window", type=float, nargs=2, default=list(fo.WINDOW),
                    help="fit window, fraction of the field output")
    ap.add_argument("--old", action="store_true",
                    help="also time the old tfld.index lookups (slow on long runs)")
    ap.add_argument("--timeout", type=int, default=0,
                    help="seconds per field file before moving on (0: none)")
    args = ap.parse_args()
    if not args.path and args.shot is None:
        ap.error("give --shot (and --x0) or --path")

    print(tped_check() + "\n", flush=True)
    todo = targets(args)
    print(f"{len(todo)} field file(s)\n", flush=True)
    if args.timeout:
        signal.signal(signal.SIGALRM, _alarm)
    rows = []
    for label, field in todo:
        if args.timeout:
            signal.alarm(args.timeout)
        try:
            rows.append(time_one(label, field, tuple(args.window), args.old))
        except Timeout:
            print(f"\n  TIMEOUT after {args.timeout} s: stalled in the stage above", flush=True)
            rows.append({"label": label, "MB": os.path.getsize(field) / 1e6, "note": "timeout"})
        except Exception as exc:
            print(f"\n  failed: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"label": label, "MB": os.path.getsize(field) / 1e6,
                         "note": type(exc).__name__})
        finally:
            if args.timeout:
                signal.alarm(0)
        print(flush=True)

    cols = ("MB", "steps", "window", "open", "sweep", "lookup", "fit")
    print(f"{"run":60}" + "".join(f"{c:>9}" for c in cols) + "  note")
    for r in rows:
        cells = "".join(f"{r[c]:>9.2f}" if isinstance(r.get(c), float) else
                        f"{r[c]:>9d}" if isinstance(r.get(c), int) else f"{'-':>9}" for c in cols)
        print(f"{r["label"][-60:]:60}{cells}  {r.get('note', '')}")
