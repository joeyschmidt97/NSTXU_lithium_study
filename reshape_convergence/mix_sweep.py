#!/usr/bin/env python3
"""Sweep the cheaseBS under-relaxation factors and report what each one costs.

`bootstrap_mix` and `istar_mix` damp the Picard loop that carries the bootstrap
current and the replayed total I* between outer iterations. They are a stability
control first: measured 2026-08-31 on 132588 `ne_ped_scale` 0.70, the campaign
default 0.1/0.05 diverges to a +181% Ip error while 0.05/0.02 falls monotonically
and saturates at +1.6%. They are a cost control second: roughly 1/mix iterations
buy one full update, so 0.02 needs ~50 where 0.035 needs ~29, at ~56 s each.

So the production setting is the LARGEST mixing that still contracts, and finding
it is a search, not a guess. This runs that search.

WHY A PROBE IS ENOUGH

A candidate does not have to be run to convergence to be judged. Direction is
visible in 20-30 iterations (falling, ringing, or growing), and the remaining
cost follows from a log-linear fit to the tail of the Ip residual. Each setting
therefore costs ~25 iterations instead of ~90, and the whole sweep fits in an
afternoon rather than a week.

USAGE

    # three candidates on the worst known point, ~23 min each
    python mix_sweep.py --shot 132588 --axis ne_ped_scale --scale 0.7 \
        --mix 0.10,0.02 --mix 0.05,0.05 --mix 0.075,0.035 --max-iter 25

    # re-print the comparison without re-solving anything
    python mix_sweep.py --score-only --outroot runs_mixsweep

    # confirm the winner on the other discharges, one point each
    python mix_sweep.py --shot 129038 --axis ne_ped_scale --scale 1.3 \
        --mix 0.075,0.035 --max-iter 30

Reading the result: `dir` is the trace direction and is the only pass/fail. FALL
is a converging setting, RING is marginal (lower the mixing), GROW diverges and
the run is void whatever its endpoint says. `proj` is the projected additional
iterations to reach tol_ip_rel at that setting -- the number to minimise among
the settings that FALL.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "run_reshape_convergence.py")


def parse_mix(text):
    """'0.075,0.035' -> (0.075, 0.035). Accepts whitespace or a slash too."""
    parts = [p for p in text.replace("/", ",").replace(" ", ",").split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--mix takes two numbers, bootstrap first: --mix 0.075,0.035")
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError("--mix values must be numbers: %r" % text)


def read_trace(path):
    """The Ip residual per iteration from a run's iteration_log.csv."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        try:
            v = float((r.get("ip_error_rel") or "").strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def judge(trace, tol):
    """(direction, projected extra iterations, first, final) for one trace.

    direction is FALL / RING / GROW / FLAT. RING counts sign changes in the
    successive differences of the tail: a residual that oscillates is not
    converging even when its last value happens to be small.
    """
    if len(trace) < 3:
        return "SHORT", None, (trace[0] if trace else None), (trace[-1] if trace else None)
    first, final = trace[0], trace[-1]
    tail = trace[-min(10, len(trace)):]

    diffs = [b - a for a, b in zip(tail, tail[1:])]
    diffs = [d for d in diffs if abs(d) > 1e-12 * max(max(abs(x) for x in tail), 1e-30)]
    flips = sum(1 for a, b in zip(diffs, diffs[1:]) if (a > 0) != (b > 0))

    if final > first:
        direction = "GROW"
    elif flips >= 3:
        direction = "RING"
    elif final < first * 0.98:
        direction = "FALL"
    else:
        direction = "FLAT"

    # Log-linear fit over the tail, so the projection reflects the late-stage
    # rate rather than the transient the loop opens with.
    proj = None
    pos = [(i, v) for i, v in enumerate(tail) if v > 0]
    if len(pos) >= 4:
        n = len(pos)
        sx = sum(i for i, _ in pos)
        sy = sum(math.log(v) for _, v in pos)
        sxx = sum(i * i for i, _ in pos)
        sxy = sum(i * math.log(v) for i, v in pos)
        den = n * sxx - sx * sx
        if den != 0:
            rate = (n * sxy - sx * sy) / den
            if final <= tol:
                proj = 0
            elif rate < -1e-6:
                proj = int(math.ceil(math.log(tol / final) / rate))
    return direction, proj, first, final


def point_dir(outroot, shot, axis, scale):
    """The per-point run directory the campaign runner wrote under outroot."""
    hits = sorted(glob.glob(os.path.join(
        outroot, "%d_*" % shot, "%s_%.3f" % (axis, scale))))
    return hits[-1] if hits else None


def solve(shot, axis, scale, max_iter, mix, outroot, log_path):
    """One campaign run at one mixing setting. Never raises: a failed setting
    is a result, and the remaining settings are still worth their wall time."""
    b, i = mix
    cmd = [sys.executable, "-u", RUNNER,
           "--shots", str(shot), "--axes", axis, "--scales", "%g" % scale,
           "--max-iter", str(max_iter),
           "--bootstrap-mix", "%g" % b, "--istar-mix", "%g" % i,
           "--outroot", outroot]
    print("  $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=HERE, stdout=fh,
                              stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Sweep bootstrap_mix / istar_mix and compare convergence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=132588)
    ap.add_argument("--axis", default="ne_ped_scale")
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--mix", type=parse_mix, action="append", default=[],
                    metavar="B,I",
                    help="a bootstrap,istar pair; repeat for each candidate "
                         "(default: 0.10,0.02 0.05,0.05 0.075,0.035)")
    ap.add_argument("--max-iter", type=int, default=25,
                    help="iterations per probe; 25 is enough to read direction")
    ap.add_argument("--tol-ip-rel", type=float, default=0.02,
                    help="tolerance the projection targets")
    ap.add_argument("--outroot", default=os.path.join(HERE, "runs_mixsweep"),
                    help="parent directory for the per-setting run directories")
    ap.add_argument("--score-only", action="store_true",
                    help="re-read existing sweep directories, solve nothing")
    args = ap.parse_args(argv)

    mixes = args.mix or [(0.10, 0.02), (0.05, 0.05), (0.075, 0.035)]
    os.makedirs(args.outroot, exist_ok=True)

    print("=== mixing sweep: shot %d  %s %.2f  max_iter %d ==="
          % (args.shot, args.axis, args.scale, args.max_iter))
    print("settings: " + ", ".join("b%g/i%g" % m for m in mixes))
    print("outroot : %s" % args.outroot)
    print("probe cost ~ %d iterations each; direction is the verdict, proj is the cost\n"
          % args.max_iter, flush=True)

    results = []
    for b, i in mixes:
        tag = "b%g_i%g" % (b, i)
        sub = os.path.join(args.outroot, tag)
        rc, wall = None, None
        if not args.score_only:
            os.makedirs(sub, exist_ok=True)
            print("--- %s ---" % tag, flush=True)
            rc, wall = solve(args.shot, args.axis, args.scale, args.max_iter,
                             (b, i), sub, os.path.join(sub, "solve.log"))
            print("  exit %s in %.0f s" % (rc, wall), flush=True)

        pdir = point_dir(sub, args.shot, args.axis, args.scale)
        trace = read_trace(os.path.join(pdir, "iteration_log.csv")) if pdir else []
        direction, proj, first, final = judge(trace, args.tol_ip_rel)
        results.append(dict(bootstrap_mix=b, istar_mix=i, tag=tag, run_dir=pdir,
                            iterations=len(trace), direction=direction,
                            projected_extra_iters=proj, ip_first=first,
                            ip_final=final, exit_code=rc, wall_s=wall))

    print("\n=== comparison ===")
    head = ("%-14s %6s %10s %10s %6s %6s %8s"
            % ("mix b/i", "iters", "Ip first", "Ip final", "dir", "proj", "wall_s"))
    print(head)
    print("-" * len(head))
    for r in results:
        print("%-14s %6d %10s %10s %6s %6s %8s" % (
            "%g/%g" % (r["bootstrap_mix"], r["istar_mix"]),
            r["iterations"],
            "--" if r["ip_first"] is None else "%.4f" % r["ip_first"],
            "--" if r["ip_final"] is None else "%.4f" % r["ip_final"],
            r["direction"],
            "--" if r["projected_extra_iters"] is None else r["projected_extra_iters"],
            "--" if r["wall_s"] is None else "%.0f" % r["wall_s"]))

    ok = [r for r in results if r["direction"] == "FALL"
          and r["projected_extra_iters"] is not None]
    empty = [r for r in results if r["iterations"] == 0]
    print()
    if len(empty) == len(results):
        # No iteration_log.csv anywhere under outroot. Says nothing about the
        # physics, and must not be reported as if it did.
        print("no run data found under %s -- nothing was solved, or --outroot "
              "points somewhere else. Expected "
              "<outroot>/<tag>/<shot>_<stamp>/<axis>_<scale>/iteration_log.csv"
              % args.outroot)
    elif ok:
        best = min(ok, key=lambda r: r["projected_extra_iters"])
        print("pick: bootstrap_mix %g / istar_mix %g -- converging, and the "
              "cheapest of the converging settings (~%d more iterations)"
              % (best["bootstrap_mix"], best["istar_mix"],
                 best["projected_extra_iters"]))
        print("confirm it on the other discharges before the campaign re-run:")
        print("  python mix_sweep.py --shot 129015 --axis ne_ped_scale --scale 0.7 "
              "--mix %g,%g --max-iter 30" % (best["bootstrap_mix"], best["istar_mix"]))
    else:
        print("no setting converged. Every candidate rang or grew, so the loop is "
              "not merely under-damped at these values -- lower the mixing "
              "further (halve both) before concluding the profiles are at fault.")

    out = os.path.join(args.outroot, "mix_sweep.json")
    with open(out, "w") as fh:
        json.dump({"shot": args.shot, "axis": args.axis, "scale": args.scale,
                   "max_iter": args.max_iter, "tol_ip_rel": args.tol_ip_rel,
                   "results": results}, fh, indent=1, default=str)
    print("\nwrote %s" % out)

    # Nonzero when nothing converged or nothing was found, so a wrapper can tell
    # "swept and found a setting" from every other outcome.
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
