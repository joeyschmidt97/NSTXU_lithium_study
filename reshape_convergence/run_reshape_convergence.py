#!/usr/bin/env python3
"""Batch runner for the cheaseBS box-edge convergence campaign.

The notebook equivalent of this file is reshape_convergence.ipynb, which is fine
for reading one discharge but ties the run to a live kernel: four solves take
hours, and a dropped SSH session or a closed browser takes the run with it. This
does the same solves headless so it survives detachment, and writes everything
the notebook would have shown to disk.

Same helpers, same records. Each discharge produces the usual
runs/<shot>_<stamp>/reshape_convergence.json plus a plain-text table beside it,
so nothing here has to be re-derived to read the result in the notebook after.

Run it detached and follow the log:

    cd <repo>/reshape_convergence
    nohup python -u run_reshape_convergence.py --shots 132543 > /dev/null 2>&1 &
    tail -f runs/campaign_*.log

`-u` matters: without it Python block-buffers stdout when it is not a terminal
and the log stays empty for hours. The script also flushes after every solve, so
`tail -f` shows progress as it happens rather than at exit.

Exit status is 0 only when every solve completed AND every point was accepted by
the campaign gate, so a wrapper can tell "ran" from "worked".
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys
import time
import traceback

# Headless: cheaseBS writes iteration_errors.png per run and matplotlib must not
# reach for a display. Set before anything can import pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = next(p for p in [pathlib.Path(__file__).resolve().parent,
                        *pathlib.Path(__file__).resolve().parents]
            if (p / "pedestal_scan.py").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reshape_convergence"))

import reshape_helpers as rh  # noqa: E402
from pedestal_scan import ANALYSIS_RADII, Campaign, DISCHARGES  # noqa: E402


class Tee:
    """stdout to the terminal and the log file at once, line-flushed.

    Everything downstream prints -- reshape_helpers, cheaseBS's own subprocess
    echo -- lands in the log without those modules knowing about it. Flushing on
    every write is what makes `tail -f` useful on a run this slow; the volume is
    a few hundred lines an hour, so the cost is irrelevant.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        self.fh.close()


def text_table(rows, shot):
    """The notebook's table as plain text.

    rh.table returns a pandas Styler, which renders to HTML and is useless in a
    log. Same columns and the same units, formatted for a terminal.
    """
    cols = [
        ("axis", "{}", 13), ("scale", "{:.2f}", 6), ("ped_top", "{:.3e}", 11),
        ("ratio", "{:.3f}", 7), ("iters", "{}", 6), ("conv", "{}", 6),
        ("acc", "{}", 6), ("Ip_err", "{:.2%}", 8), ("dq_max", "{:.2%}", 8),
        ("dq_at", "{:.3f}", 7), ("dp_max", "{:.2%}", 8),
        ("q@x0", "{:.2%}", 8), ("wall_s", "{:.0f}", 8), ("status", "{}", 8),
    ]
    lines = [f"shot {shot}",
             "  " + "".join(name.rjust(w) for name, _, w in cols)]
    for r in rows:
        q = r.get("q_errors_rel") or {}
        vals = {
            "axis": r["axis"], "scale": r["scale"], "ped_top": r.get("metric"),
            "ratio": r.get("metric_ratio"), "iters": r.get("iterations"),
            "conv": r.get("converged"), "acc": r.get("accepted"),
            "Ip_err": r.get("ip_error_rel"), "dq_max": r.get("dq_max"),
            "dq_at": r.get("dq_at_rho"), "dp_max": r.get("dp_max"),
            "q@x0": max((abs(v) for v in q.values()), default=None),
            "wall_s": r.get("wall_s"),
            "status": "RAISED" if "error" in r else "",
        }
        cells = []
        for name, fmt, w in cols:
            v = vals.get(name)
            try:
                s = "--" if v is None else fmt.format(v)
            except (TypeError, ValueError):
                s = str(v)
            cells.append(s.rjust(w))
        lines.append("  " + "".join(cells))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the cheaseBS box-edge convergence solves headless.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="+", default=sorted(DISCHARGES),
                    choices=sorted(DISCHARGES),
                    help="discharges to solve, in order")
    ap.add_argument("--max-iter", type=int, default=rh.CHEASEBS["max_iter"],
                    help="cheaseBS max_iter")
    ap.add_argument("--axes", nargs="+", default=list(rh.AXES),
                    help="scan axes")
    ap.add_argument("--scales", type=float, nargs="+", default=list(rh.SCALES),
                    help="scale factors (box edges)")
    ap.add_argument("--outroot", default=rh.OUTROOT,
                    help="where per-shot run directories are written")
    ap.add_argument("--log", default=None,
                    help="log file (default: <outroot>/campaign_<stamp>.log)")
    args = ap.parse_args(argv)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    os.makedirs(args.outroot, exist_ok=True)
    log_path = args.log or os.path.join(args.outroot, f"campaign_{stamp}.log")

    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    sys.stderr = tee

    cheasebs = dict(rh.CHEASEBS, max_iter=args.max_iter)

    print(f"=== reshape convergence campaign {stamp} ===")
    print(f"log       : {log_path}")
    print(f"shots     : {args.shots}")
    print(f"axes      : {args.axes}")
    print(f"scales    : {args.scales}")
    print(f"cheaseBS  : {cheasebs}")
    print(f"solves    : {len(args.shots) * len(args.axes) * len(args.scales)}")
    print(f"pid       : {os.getpid()}")
    print()

    t_camp = time.time()
    # Only the requested shots: Campaign() with no argument loads and fits all
    # four, which is minutes of work per discharge that a one-shot run never uses.
    print(f"loading and fitting {len(args.shots)} discharge(s) ...", flush=True)
    camp = Campaign(shots=args.shots)
    print(f"  done in {time.time() - t_camp:.0f}s\n", flush=True)

    summary, failed = {}, []
    for shot in args.shots:
        print(f"--- {shot}  radii {ANALYSIS_RADII[shot]} ---", flush=True)
        t0 = time.time()
        try:
            rows, workdir = rh.run_bounds(
                camp, shot, axes=tuple(args.axes), scales=tuple(args.scales),
                outroot=args.outroot, cheasebs=cheasebs)
        except Exception:
            # One discharge failing outright must not take the rest of the
            # campaign with it -- the remaining shots are hours of independent
            # work. Record it and carry on; the exit status still reports it.
            print(f"!!! {shot} RAISED, continuing with the next shot")
            traceback.print_exc()
            failed.append(shot)
            continue

        tbl = text_table(rows, shot)
        print(tbl, flush=True)
        with open(os.path.join(workdir, "table.txt"), "w", encoding="utf-8") as f:
            f.write(tbl + "\n")

        n_bad = sum(1 for r in rows if "error" in r or not r.get("accepted"))
        summary[shot] = {
            "workdir": workdir,
            "wall_s": time.time() - t0,
            "iterations": [r.get("iterations") for r in rows],
            "accepted": [r.get("accepted") for r in rows],
            "hit_max_iter": [r.get("iterations") == args.max_iter for r in rows],
            "rejected_or_raised": n_bad,
        }
        if n_bad:
            failed.append(shot)
        print(f"  {shot} done in {time.time() - t0:.0f}s -> {workdir}\n", flush=True)

    print("=== campaign summary ===")
    for shot, s in summary.items():
        # hit_max_iter is called out because the acceptance gate cannot see it:
        # a point can exhaust max_iter and still be accepted, which reads as
        # converged in the record unless the iteration count is checked.
        flag = " HIT-MAX-ITER" if any(s["hit_max_iter"]) else ""
        print(f"  {shot}: iters={s['iterations']} accepted={s['accepted']} "
              f"{s['wall_s']:.0f}s{flag}")
    print(f"  total wall: {time.time() - t_camp:.0f}s")
    if failed:
        print(f"  PROBLEM shots (raised or not fully accepted): {sorted(set(failed))}")

    with open(os.path.join(args.outroot, f"campaign_{stamp}.json"), "w") as f:
        json.dump({"stamp": stamp, "shots": args.shots, "axes": args.axes,
                   "scales": args.scales, "cheasebs": cheasebs,
                   "summary": summary, "failed": sorted(set(failed))},
                  f, indent=1, default=str)

    print(f"\nlog: {log_path}")
    tee.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
