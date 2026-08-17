"""Replay the acceptance gate over already-completed runs — no CHEASE needed.

Exercises cheasebs_runner.evaluate_acceptance() against EQDSKs that already exist
on disk, so the gate's tolerances and its verdicts can be checked (and re-tuned)
without spending solver time. Point it at any directory tree whose leaves contain
an EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT.

Two trees worth replaying:
  canary_132588/runs/<ts>/                       — identity + mtanh at 1/5/15 iterations
  TPED/projects/discharge_tools/tests/cheasebs_identity/runs/<ts>/132588/
                                                 — the 9-case scaling suite, which
    includes deliberately-perturbed runs the gate should REJECT (E/H miss their
    Ip target by -27% / +42%).

Expected outcome on the canary tree: all four ACCEPTED — Ip error 1.5% under the
3% tolerance, analysis-radius q error under 1% against the 2% tolerance. If those
reject, the tolerance defaults in CheasebsAcceptance are wrong, not the runs.

Usage:
    python gate_replay.py --tree runs/20260816_20-06-37 --source <path to g132588.00650>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from TPED.config.config_helper import Config
from TPED.projects.discharge_tools.src.cheasebs_runner import (
    CheasebsAcceptance,
    evaluate_acceptance,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EQDSK_NAME = "EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT"
GENE_RADII = (0.736, 0.825)


def find_runs(tree):
    return sorted(glob.glob(os.path.join(tree, "*", EQDSK_NAME)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=None,
                    help="directory whose immediate subdirs are runs; defaults to newest canary run")
    ap.add_argument("--source", default=None,
                    help="source EFIT g-file; defaults to the one named in canary_summary.json")
    ap.add_argument("--mode", choices=("production", "identity"), default="production",
                    help="production: per-point survey gate, q is allowed to move. "
                         "identity: regression check, q must match the source.")
    ap.add_argument("--max-ip-error", type=float, default=0.03)
    ap.add_argument("--max-q-error", type=float, default=0.02,
                    help="identity mode only")
    args = ap.parse_args()

    tree = args.tree
    if tree is None:
        candidates = sorted(glob.glob(os.path.join(HERE, "runs", "*")))
        if not candidates:
            raise SystemExit("no canary runs found; pass --tree")
        tree = candidates[-1]
    tree = os.path.abspath(tree)

    source = args.source
    if source is None:
        summary_path = os.path.join(tree, "canary_summary.json")
        if not os.path.isfile(summary_path):
            raise SystemExit(f"no canary_summary.json in {tree}; pass --source")
        source = json.load(open(summary_path))["source_gfile"]
    if not os.path.isfile(source):
        raise SystemExit(f"source g-file not found: {source}")

    cheasebs_script = os.path.join(Config().get_path("CHEASEBS_PATH"),
                                   "run_chease_iterative_profiles.py")

    if args.mode == "identity":
        policy = CheasebsAcceptance.identity(
            analysis_radii=GENE_RADII,
            max_ip_error_rel=args.max_ip_error,
            max_q_error_rel=args.max_q_error,
        )
        q_rule = f"q <= {policy.max_q_error_rel:.1%} vs source"
    else:
        policy = CheasebsAcceptance.production(
            analysis_radii=GENE_RADII,
            max_ip_error_rel=args.max_ip_error,
        )
        q_rule = f"q blowup bound {policy.max_q_change_rel:.0%} (q is allowed to move)"

    eqdsks = find_runs(tree)
    if not eqdsks:
        raise SystemExit(f"no {EQDSK_NAME} under {tree}/*/")
    print(f"tree:   {tree}\nsource: {source}\n"
          f"policy: [{args.mode}] Ip <= {policy.max_ip_error_rel:.1%}, {q_rule}, "
          f"at rho_tor {GENE_RADII}\n")

    rows = []
    for eqdsk in eqdsks:
        run_dir = os.path.dirname(eqdsk)
        name = os.path.basename(run_dir)

        # A run that aimed at a different Ip must be scored against the Ip it
        # aimed at, not the source's — that target lives in the merged config the
        # run wrote next to itself.
        target_ip = target_bt = None
        cfg_path = os.path.join(run_dir, "cheasebs_run_config.json")
        if os.path.isfile(cfg_path):
            cfg = json.load(open(cfg_path))
            target_ip = cfg.get("target_ip_a")
            target_bt = cfg.get("target_bt_t")

        result = evaluate_acceptance(
            eqdsk_path=eqdsk,
            source_eqdsk=source,
            run_dir=run_dir,
            cheasebs_script=cheasebs_script,
            policy=policy,
            target_ip_a=target_ip,
            target_bt_t=target_bt,
        )
        rows.append((name, result))

    width = max(len(n) for n, _ in rows) + 2
    print(f"{'case':{width}s} {'verdict':9s} {'|dIp|':>8s} "
          + " ".join(f"{'q@' + str(r):>9s}" for r in GENE_RADII)
          + f" {'q edge':>9s}")
    for name, r in rows:
        def fmt(v):
            return f"{v:8.3%}" if v is not None else "     n/a"
        q_cols = " ".join(f"{fmt(r.q_errors_rel.get(rr)):>9s}" for rr in GENE_RADII)
        print(f"{name:{width}s} {'ACCEPT' if r.accepted else 'REJECT':9s} "
              f"{fmt(r.ip_error_rel)} {q_cols} {fmt(r.q_edge_error_rel)}")
        for reason in r.reasons:
            print(f"{'':{width}s}   - {reason}")

    n_ok = sum(1 for _, r in rows if r.accepted)
    print(f"\n{n_ok}/{len(rows)} accepted")


if __name__ == "__main__":
    main()
