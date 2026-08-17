"""132588 CHEASE-BS canary: does the thesis validation discharge reconstruct?

Runs on NERSC (needs the compiled CHEASE binary via TPED/config/user_config.yaml).

Background — what the 2026-07-15 cheasebs_identity run already showed for 132588
(TPED/projects/discharge_tools/tests/cheasebs_identity/runs/20260715_14-12-33/132588):

  * identity cases (A/B/C/D, max_iter=1, qspec on) reproduce Te/Ti/ne/ni exactly,
    hold q(0) exactly and q95 to <0.35%, but land Ip 1.5% low (tol_ip_rel is
    0.002), leave q at the last flux surface ~20% low, and move pedestal pressure
    by 23-31%.
  * with max_iter=1 those runs are unconverged *by construction*, so it is not yet
    known whether the edge error is an unconverged transient or a standing
    CHEASE-vs-EFIT offset. That is the one question this script answers.
  * separately: E (target_ip_scale_a=2.0) and H (target_ip_scale_a=1.03) wrote
    correct absolute targets into their configs (1966669.5 A / 1012834.79 A) and
    produced byte-identical output EQDSKs at Ip = 1436124.7 A. With
    enforce_qspec=False the Ip target is not honoured and the result does not
    depend on it. The Bt target path does land its target exactly (C/F/I, 0.000%).
    Tracked separately; not exercised here.

What this script does:
  1. Runs the pure-identity case (no profile transform, qspec on, explicit
     target_ip_a == source Ip) at each max_iter in ITER_SWEEP.
  2. Runs the mtanh-unity case (D) at the largest max_iter, so the whole-profile
     fit residual is measured against a converged equilibrium rather than a
     first-iteration one.
  3. For every run, reports cheaseBS's own convergence verdict (read from
     convergence_summary.json on the scratch output dir, which the TPED wrapper
     currently does not copy back or check) alongside the q/p deviation of the
     output EQDSK from the source EFIT g-file.

Acceptance question this feeds: what Ip-error / q-error tolerance can the sparse
scan's equilibrium gate actually be frozen at?

Usage (NERSC):
    python cheasebs_canary_132588.py [--dirpath /path/to/132588] [--outroot ./runs]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

from TPED.projects.discharge_tools.src.discharge_data import DischargeData
from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics

DEFAULT_DIRPATH = "/global/homes/j/joeschm/data/ST_research/NSTXU_discharges/132588"
ITER_SWEEP = (1, 5, 15)

# Radii the existing 132588 GENE linear scans sit at (this repo,
# high_triangularity/132588): r_0.736 (q=4) and r_0.825 (q=5). q there is what
# the alignment work depends on, so it is reported explicitly.
GENE_RADII = (0.736, 0.825)


# --------------------------------------------------------------------------
# Minimal EQDSK reader (stdlib only — no numpy/xarray needed, so this stays
# runnable from a bare login-node python if the TPED env is not loaded).
# --------------------------------------------------------------------------

def read_eqdsk(path):
    lines = open(path).read().splitlines()
    ints = re.findall(r"-?\d+", lines[0])
    nw, nh = int(ints[-2]), int(ints[-1])
    body = " ".join(lines[1:])
    toks = re.findall(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?|[-+]?\d+\.?\d*[EeDd][-+]?\d+", body)
    vals = [float(t.replace("D", "E").replace("d", "e")) for t in toks]
    i = 0

    def take(n):
        nonlocal i
        out = vals[i:i + n]
        i += n
        return out

    take(5)
    _rmaxis, _zmaxis, simag, sibry, bcentr = take(5)
    current = take(5)[0]
    take(5)
    fpol = take(nw)
    pres = take(nw)
    take(nw)          # ffprim
    take(nw)          # pprime
    take(nw * nh)     # psirz
    qpsi = take(nw)
    return dict(nw=nw, bcentr=bcentr, current=current, simag=simag, sibry=sibry,
                fpol=fpol, pres=pres, qpsi=qpsi)


def regrid(y, n_out):
    """Linear resample of a psi_N-uniform profile onto n_out points.

    Source EFIT g-files are nw=129; cheaseBS writes nw=513. Both are uniform in
    normalized poloidal flux, so index-fraction interpolation is the like-for-like
    comparison.
    """
    n_in = len(y)
    out = []
    for k in range(n_out):
        x = k * (n_in - 1) / (n_out - 1)
        lo = int(x)
        hi = min(lo + 1, n_in - 1)
        f = x - lo
        out.append(y[lo] * (1 - f) + y[hi] * f)
    return out


def pct(a, b):
    return 100.0 * (a - b) / b if b else float("nan")


def deviation(post, src, n=129):
    q_post, q_src = regrid(post["qpsi"], n), regrid(src["qpsi"], n)
    p_post, p_src = regrid(post["pres"], n), regrid(src["pres"], n)
    dq = [pct(q_post[k], q_src[k]) for k in range(n)]
    dp = [pct(p_post[k], p_src[k]) for k in range(n) if abs(p_src[k]) > 1.0]
    i95 = int(0.95 * (n - 1))
    out = {
        "d_ip_pct": pct(abs(post["current"]), abs(src["current"])),
        "dq_axis_pct": dq[0],
        "dq95_pct": dq[i95],
        "dq_edge_pct": dq[-1],
        "max_abs_dq_pct": max(abs(x) for x in dq),
        "max_abs_dp_pct": max(abs(x) for x in dp),
    }
    # q at the radii the existing GENE scans use. psi_N index is an approximation
    # of rho_tor here (exact mapping needs the equilibrium); flagged as such.
    for r in GENE_RADII:
        k = min(n - 1, int(r * (n - 1)))
        out[f"dq_psiN{r}_pct"] = dq[k]
    return out


# --------------------------------------------------------------------------
# cheaseBS verdict — the data the TPED wrapper leaves behind on scratch
# --------------------------------------------------------------------------

def cheasebs_verdict(savedir):
    """cheaseBS's own convergence record for the run that wrote *savedir*.

    cheasebs_runner.py returns the last eqdsk_path in iteration_log.csv and never
    reads the convergence flag, so a non-converged run is indistinguishable from a
    converged one at the call site. The merged config it writes into savedir names
    the scratch output_dir, which is where convergence_summary.json lands.
    """
    cfg_path = os.path.join(savedir, "cheasebs_run_config.json")
    if not os.path.isfile(cfg_path):
        return {"error": f"no cheasebs_run_config.json in {savedir}"}
    cfg = json.load(open(cfg_path))
    out_dir = cfg.get("output_dir")
    summary_path = os.path.join(out_dir, "convergence_summary.json")
    if not os.path.isfile(summary_path):
        return {"error": f"no convergence_summary.json at {summary_path}",
                "output_dir": out_dir}
    summary = json.load(open(summary_path))
    return {
        "output_dir": out_dir,
        "converged": summary.get("final_converged"),
        "iterations": summary.get("iterations") or summary.get("n_iterations"),
        "target_ip_a": summary.get("target_ip_a"),
        "final_ip_a": (summary.get("final") or {}).get("ip_a"),
        "summary_path": summary_path,
    }


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def source_gfile_path(dirpath):
    phys = DischargePhysics(DischargeData(input_dir=dirpath))
    attrs = phys._get_raw_gfile().attrs
    return os.path.join(attrs["directory"], attrs["filename"])


def run_case(name, dirpath, out_root, *, transform, max_iter):
    savedir = os.path.join(out_root, name)
    os.makedirs(savedir, exist_ok=True)
    print(f"\n=== {name} (max_iter={max_iter}, transform={transform}) ===", flush=True)

    phys = DischargePhysics(DischargeData(input_dir=dirpath))
    if transform == "mtanh_unity":
        phys = phys.apply_mtanh_full(["Te", "ne"])

    final_eqdsk = phys.output_gfile(
        savedir=savedir,
        run_cheasebs=True,
        max_iter=max_iter,
        comment=f"canary132588-{name}",
    )
    return savedir, final_eqdsk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirpath", default=DEFAULT_DIRPATH,
                    help="132588 discharge directory (gfile + pfile)")
    ap.add_argument("--outroot", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    args = ap.parse_args()

    if not os.path.isdir(args.dirpath):
        sys.exit(f"discharge dirpath does not exist: {args.dirpath}")

    out_root = os.path.join(args.outroot, datetime.now().strftime("%Y%m%d_%H-%M-%S"))
    os.makedirs(out_root, exist_ok=True)
    print(f"Output root: {out_root}")

    src_path = source_gfile_path(args.dirpath)
    src = read_eqdsk(src_path)
    print(f"Source EFIT: {src_path}")
    print(f"  Ip={src['current']:.1f} A  Bt={src['bcentr']:+.6f} T  "
          f"q0={src['qpsi'][0]:.4f}  q95={src['qpsi'][int(0.95 * (src['nw'] - 1))]:.4f}")

    cases = [(f"identity_iter{n}", dict(transform="none", max_iter=n)) for n in ITER_SWEEP]
    cases.append((f"mtanh_unity_iter{max(ITER_SWEEP)}",
                  dict(transform="mtanh_unity", max_iter=max(ITER_SWEEP))))

    results = {}
    for name, kwargs in cases:
        try:
            savedir, final_eqdsk = run_case(name, args.dirpath, out_root, **kwargs)
        except Exception as exc:                      # a failed case must not kill the sweep
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        verdict = cheasebs_verdict(savedir)
        dev = deviation(read_eqdsk(final_eqdsk), src)
        results[name] = {"max_iter": kwargs["max_iter"], "eqdsk": final_eqdsk,
                         **verdict, **dev}
        print(f"  converged={verdict.get('converged')}  "
              f"dIp={dev['d_ip_pct']:+.3f}%  dq95={dev['dq95_pct']:+.3f}%  "
              f"dq_edge={dev['dq_edge_pct']:+.3f}%  max|dp|={dev['max_abs_dp_pct']:.2f}%")

    report = os.path.join(out_root, "canary_summary.json")
    with open(report, "w") as f:
        json.dump({"source_gfile": src_path, "dirpath": args.dirpath,
                   "results": results}, f, indent=2, sort_keys=True)

    print("\n=== summary ===")
    header = f"{'case':24s} {'conv':6s} {'dIp%':>9s} {'dq0%':>8s} {'dq95%':>9s} {'dq_edge%':>10s} {'max|dp|%':>10s}"
    print(header)
    for name, r in results.items():
        if "error" in r and "d_ip_pct" not in r:
            print(f"{name:24s} {r['error']}")
            continue
        print(f"{name:24s} {str(r.get('converged')):6s} {r['d_ip_pct']:+9.3f} "
              f"{r['dq_axis_pct']:+8.3f} {r['dq95_pct']:+9.3f} {r['dq_edge_pct']:+10.3f} "
              f"{r['max_abs_dp_pct']:10.2f}")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
