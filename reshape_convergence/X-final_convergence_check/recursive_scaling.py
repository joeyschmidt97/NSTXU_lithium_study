#!/usr/bin/env python3
"""Recursive cheaseBS scaling: chain single-knob scale steps.

Each step's cheaseBS reconstruction becomes the *baseline* for the next step
-- both the equilibrium geometry (the reconstructed EQDSK) and the reference
profiles cheaseBS's pressure decomposition is measured against (the GENE
profiles_e/i/z the previous step actually solved with, copied back by
output_gfile alongside the EQDSK). The transform in scalings.SCALINGS
re-fits/re-exponentiates against that new baseline every step.

Steps are an ADDITIVE grid on the cumulative scaling factor, not a repeated
multiplier: --step 0.05 with --target 1.2 walks the readable cumulative
sequence 1.05, 1.10, 1.15, 1.20. The number actually handed to scale() each
step is the ratio between consecutive grid points (1.05/1.00, 1.10/1.05, ...)
-- each a little under 1.05, because it is relative to a state already
scaled up, not relative to the original shot.

    # target 1.2 via 0.05-sized cumulative hops: 1.05, 1.10, 1.15, 1.20
    python recursive_scaling.py --shot 132588 --scaling omt --target 1.2 --step 0.05

    # explicit step count instead of a step-size guess (even spacing)
    python recursive_scaling.py --shot 132588 --scaling omt --target 1.2 --n-steps 4

    # just prove the plumbing works: run step 1, stop, inspect, then continue
    python recursive_scaling.py --shot 132588 --scaling omt --target 1.2 --pilot
    python recursive_scaling.py --resume <savedir printed above>

All of a chain's cheaseBS runs land together under one directory named
REC_<shot>_<scaling>_<stamp>, in scratch's own cheaseBS_runs/ tree, so they
are easy to spot among unrelated runs and easy to purge as a unit.

Every completed step is checkpointed to <savedir>/chain.json before the next
one starts, AND mirrored into <this repo>/runs/REC_.../chain.json -- the
scratch copy is the working record cheaseBS output actually lives beside; the
repo copy is small and travels with `git pull`, so the chain's status/results
can be checked from a different machine even though scratch never leaves the
machine it ran on. The next step is always rebuilt from disk (the previous
step's own EQDSK + profiles_e/i/z, via DischargeData -- the same loader path
a fresh shot goes through), so --resume after a killed process is a normal
case, not a special one; it only works on the machine that ran it, since it
reads the scratch copy's file paths.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import traceback

# Headless: cheaseBS's per-run plots reach for pyplot and there is no display
# on a login node or in a batch job.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import scalings as S  # noqa: E402 -- load(), scale(), tag(), SCALINGS, MAX_ITER

from TPED.projects.discharge_tools.src.discharge_data import DischargeData  # noqa: E402
from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics  # noqa: E402

MANIFEST = "chain.json"
SPECIES = ("e", "i", "z")
REPO_RECORD_ROOT = os.path.join(HERE, "runs")


# ---------------------------------------------------------------------------
# Step math -- additive cumulative grid
# ---------------------------------------------------------------------------

def build_grid(target: float, step: float) -> list[float]:
    """Cumulative scaling factors from 1.0 to `target`, in `step`-sized hops.

    e.g. target=1.2, step=0.05 -> [1.05, 1.10, 1.15, 1.20]. The final hop is
    clipped to land exactly on target rather than overshoot it.
    """
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")
    grid, val = [], 1.0
    while val < target - 1e-9:
        val = round(min(val + step, target), 6)
        grid.append(val)
    return grid


def build_grid_n(target: float, n: int) -> list[float]:
    """Cumulative scaling factors from 1.0 to `target` in n even hops."""
    step = (target - 1.0) / n
    return [round(1.0 + step * i, 6) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Rebasing -- turn one step's reconstructed output into the next step's input
# ---------------------------------------------------------------------------

def rebase(case_dir: str, eqdsk_path: str, shot: int, time_ms=None) -> DischargePhysics:
    """DischargePhysics for the next step, loaded the ordinary way.

    output_gfile copies the reconstructed EQDSK and the profiles_e/i/z it
    solved with into `case_dir` (discharge_io.py's copy-back list) -- exactly
    the (gfile, profiles) pair DischargeData already knows how to harmonize
    (DischargeHarmonizer falls back to profiles_filepaths when there is no
    pfile). Loading from disk rather than carrying the in-memory object
    forward is deliberate: it is the same call whether the chain is
    continuing in-process or resuming after --resume, so there is exactly one
    code path instead of two.
    """
    profiles = [os.path.join(case_dir, f"profiles_{s}") for s in SPECIES]
    missing = [p for p in profiles if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"cheaseBS did not copy back {missing}; cannot rebase the next "
            f"step on {case_dir}")
    data = DischargeData(gfile=eqdsk_path, profiles=profiles,
                         shot=shot, time_ms=time_ms)
    return DischargePhysics(data)


# ---------------------------------------------------------------------------
# Manifest (resume support + repo-tracked record)
# ---------------------------------------------------------------------------

def load_manifest(savedir: str) -> dict | None:
    path = os.path.join(savedir, MANIFEST)
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def write_manifest(manifest: dict, *dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, MANIFEST), "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Chain driver
# ---------------------------------------------------------------------------

def run_chain(shot: int, scaling: str, target: float, grid: list[float],
             savedir: str, repo_record_dir: str, gfile_kw: dict, *,
             pilot: bool = False, manifest: dict | None = None) -> dict:
    """Run (or resume) the recursive chain. Returns the updated manifest.

    Stops after one step when `pilot` is set, leaving the manifest ready for
    a later --resume to pick up the remaining steps.
    """
    os.makedirs(savedir, exist_ok=True)

    if manifest is None:
        base = S.load(shot)
        time_ms = base.ds.attrs.get("time_ms")
        manifest = {
            "shot": shot, "scaling": scaling, "target": target, "grid": grid,
            "savedir": savedir, "repo_record_dir": repo_record_dir,
            "time_ms": time_ms, "completed": [],
        }
        print(f"=== recursive {scaling}: {shot}, grid {grid} ===", flush=True)
        phys = base
    else:
        time_ms = manifest.get("time_ms")
        repo_record_dir = manifest.get("repo_record_dir", repo_record_dir)
        print(f"=== resuming {scaling}: {shot}, {len(manifest['completed'])}/"
              f"{len(manifest['grid'])} step(s) already done ===", flush=True)
        last = manifest["completed"][-1]
        phys = rebase(last["case_dir"], last["eqdsk"], shot, time_ms)

    n = len(manifest["grid"])
    done = len(manifest["completed"])
    if done >= n:
        print("chain already complete", flush=True)
        return manifest

    prev_cumulative = manifest["completed"][-1]["cumulative"] if done else 1.0

    for k in range(done, n):
        cumulative = manifest["grid"][k]
        r = cumulative / prev_cumulative
        label = f"step{k+1:02d}_{S.tag(scaling, cumulative)}"
        case_dir = os.path.join(savedir, label)
        os.makedirs(case_dir, exist_ok=True)
        print(f"--- step {k+1}/{n}: {scaling} -> cumulative {cumulative:.4f} "
              f"(x{r:.5f} relative to the last step) -> {case_dir} ---",
              flush=True)

        q = S.scale(phys, scaling, r)
        try:
            eqdsk_path = q.output_gfile(
                savedir=case_dir, run_cheasebs=True,
                comment=f"{shot}_{scaling}_step{k+1:02d}",
                **gfile_kw)
        except Exception:
            traceback.print_exc()
            manifest["error"] = f"step {k+1} raised, see traceback above"
            write_manifest(manifest, savedir, repo_record_dir)
            raise

        print(f"  reconstructed -> {eqdsk_path}", flush=True)
        manifest["completed"].append({
            "step": k + 1, "cumulative": cumulative, "relative_factor": r,
            "case_dir": case_dir, "eqdsk": eqdsk_path,
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_manifest(manifest, savedir, repo_record_dir)
        prev_cumulative = cumulative

        if pilot:
            print(f"--- pilot stop after step {k+1}; "
                  f"resume with --resume {savedir} ---", flush=True)
            return manifest

        phys = rebase(case_dir, eqdsk_path, shot, time_ms)

    print(f"=== chain complete: grid {manifest['grid']} reached, "
          f"{savedir} ===", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Recursively scale one discharge through cheaseBS, "
                    "each step starting from the previous step's reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, choices=sorted(S.SHOTS),
                    help="required unless --resume is given")
    ap.add_argument("--scaling", choices=sorted(S.SCALINGS),
                    help="required unless --resume is given")
    ap.add_argument("--target", type=float, default=1.2,
                    help="cumulative scaling factor to reach from 1.0")
    ap.add_argument("--step", type=float, default=0.05,
                    help="cumulative-grid step size, e.g. 0.05 -> "
                         "1.05, 1.10, 1.15, ... up to --target")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="use n evenly-spaced hops instead of --step")
    ap.add_argument("--pilot", action="store_true",
                    help="run one step then stop, for a functionality check "
                         "before committing to the full chain")
    ap.add_argument("--resume", metavar="SAVEDIR", default=None,
                    help="continue a chain from <SAVEDIR>/chain.json, "
                         "rebuilt from that step's own EQDSK + profiles on disk")
    ap.add_argument("--savedir", default=None,
                    help="default: <scratch>/cheaseBS_runs/REC_<shot>_<scaling>_<stamp>")
    ap.add_argument("--record-dir", default=None,
                    help=f"repo-tracked mirror of chain.json (default: "
                         f"{REPO_RECORD_ROOT}\\REC_<shot>_<scaling>_<stamp>)")
    ap.add_argument("--max-iter", type=int, default=S.MAX_ITER)
    ap.add_argument("--allow-rejected", action="store_true",
                    help="by default a rejected reconstruction aborts the "
                         "chain (cheasebs_strict=True) rather than silently "
                         "becoming the next step's baseline; pass this to "
                         "instead accept whatever cheaseBS returns")
    args = ap.parse_args(argv)

    manifest = None
    if args.resume:
        savedir = os.path.abspath(args.resume)
        manifest = load_manifest(savedir)
        if manifest is None:
            raise SystemExit(f"no {MANIFEST} found in {savedir}")
        shot, scaling = manifest["shot"], manifest["scaling"]
        target, grid = manifest["target"], manifest["grid"]
        repo_record_dir = manifest.get("repo_record_dir")
    else:
        if not args.shot or not args.scaling:
            raise SystemExit("--shot and --scaling are required (or pass --resume)")
        shot, scaling, target = args.shot, args.scaling, args.target
        grid = (build_grid_n(target, args.n_steps) if args.n_steps
               else build_grid(target, args.step))

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        rec_name = f"REC_{shot}_{scaling}_{stamp}"
        savedir = os.path.abspath(
            args.savedir or os.path.join(S.scratch_root(), "cheaseBS_runs", rec_name))
        repo_record_dir = os.path.abspath(args.record_dir or
                                          os.path.join(REPO_RECORD_ROOT, rec_name))

    gfile_kw = {"max_iter": args.max_iter,
               "cheasebs_strict": not args.allow_rejected}

    try:
        run_chain(shot, scaling, target, grid, savedir, repo_record_dir,
                 gfile_kw, pilot=args.pilot, manifest=manifest)
    except Exception:
        print(f"\n=== chain FAILED, partial record in {savedir} "
              f"and {repo_record_dir} ===", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
