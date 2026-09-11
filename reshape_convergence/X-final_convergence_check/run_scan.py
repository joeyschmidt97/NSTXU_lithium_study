#!/usr/bin/env python3
"""Run scaled cases through cheaseBS. Same transforms the notebook plots.

    python run_scan.py --shot 132588 --kind mtanh_full --te 1.3 --savedir out/
    python run_scan.py --shot 132588 --kind omt_omne --te 0.7 --ne 0.7 --savedir out/
    python run_scan.py --shot 132588 --kind mtanh_full --scan te 0.7 1.3 --savedir out/
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scalings import KINDS, SHOTS, run, tag  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, required=True, choices=SHOTS)
    ap.add_argument("--kind", default="mtanh_full", choices=KINDS)
    ap.add_argument("--te", type=float, default=1.0)
    ap.add_argument("--ne", type=float, default=1.0)
    ap.add_argument("--scan", nargs="+", metavar=("AXIS", "VALUE"),
                    help="sweep one axis: --scan te 0.7 0.9 1.1 1.3")
    ap.add_argument("--savedir", default=os.environ.get("SCRATCH", "."))
    ap.add_argument("--dry-run", action="store_true",
                    help="scale only, no gfile and no cheaseBS")
    args = ap.parse_args(argv)

    if args.scan:
        axis, values = args.scan[0], [float(v) for v in args.scan[1:]]
        if axis not in ("te", "ne") or not values:
            raise SystemExit("--scan wants 'te' or 'ne' then one or more values")
        cases = [{axis: v} for v in values]
    else:
        cases = [{"te": args.te, "ne": args.ne}]

    failed = []
    for kw in cases:
        te, ne = kw.get("te", 1.0), kw.get("ne", 1.0)
        label = tag(args.kind, te, ne)
        print(f"--- {args.shot} {label} ---", flush=True)
        try:
            run(args.shot, args.kind, te=te, ne=ne,
                gfile=not args.dry_run, savedir=args.savedir)
        except Exception:                                        # noqa: BLE001
            traceback.print_exc()
            failed.append(label)

    print(f"=== {len(cases) - len(failed)}/{len(cases)} ok ===")
    if failed:
        print("FAILED: %s" % ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
