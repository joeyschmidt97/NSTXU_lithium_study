"""132588 profile-base tests G/H/I: is the raw Te profile what breaks jparallel?

Design and reading table: vault note 2026-10-02-cheasebs-132xxx-convergence-plan,
"Profile-base tests". In short, on 132588 every box-scan case that kept the raw
Te failed (including omt/omne at 1.0, which are the raw profile exactly) and
every Te_ped_scale case, where Te is the mtanh fit, converged. The fit drops
axis Te to 0.853 of raw. These three tests separate the profile base from the
replay representation and from the core height.

    G  raw base (omt 1.0) vs fit-Te base (Te_ped_scale 1.0), both under I-star
    H  omt / omne / ne_ped_scale at 1.0, 0.8, 1.2 on the fit-Te base, jparallel
    I  fit-Te base with the core raised to the raw axis value, jparallel

Output goes straight into the cheaseBS_tests checkout, one directory per test:

    <repo>/data/profile_base_132588/<G_istar|H_fitbase|I_rawcore>/
        132588/<case>/            cheaseBS run, same layout as data/jparallel_box
        console.log               everything printed, tracebacks included
        summary.json              ok / failed per case, settings, commit
        cheasebs_comparison_*.png TPED compare_cheasebs_runs.py figures

Failed cases keep only their copied gfile, as in the box scan; console.log is
the record of why. After a test the comparison runs on its own; `compare`
reruns it alone.

    python -u -m NSTXU_lithium_study.scripts.profile_base_tests G
    python -u -m NSTXU_lithium_study.scripts.profile_base_tests H
    python -u -m NSTXU_lithium_study.scripts.profile_base_tests I
    python -u -m NSTXU_lithium_study.scripts.profile_base_tests compare G
    python -u -m NSTXU_lithium_study.scripts.profile_base_tests H --dry-run
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

from NSTXU_lithium_study.scripts.scalings import (
    discharge_root, load, namelist_with_ncscal, scale, tag)

SHOT = 132588

# First path that exists wins, as for the discharge root in scalings.py.
REPO_CANDIDATES = [
    "/global/homes/j/joeschm/projects/cheaseBS_tests",
    "C:/Users/joesc/git/cheaseBS_tests",
]
CAMPAIGN = "profile_base_132588"

# The approved jparallel configuration (2026-09-30 box scan). Every test
# changes exactly one thing against it.
JPARALLEL = dict(replay_representation="jparallel", coordinate="rhot",
                 bootstrap_mix=0.35, amplitude_warmup_iters=0,
                 tol_bs=1e-3, tol_q=1e-3, tol_ip_rel=1e-3, max_iter=40)
# G: the representation changed, with the I-star defaults it was run with in
# September (rho_pol, bootstrap_mix 0.1).
ISTAR = dict(JPARALLEL, replay_representation="istar", coordinate="rhop",
             bootstrap_mix=0.1)
NCSCAL = 4

# scale_core_height that puts the fitted axis Te back on the raw value:
# 1.17 gives 0.998 of raw (measured 2026-10-10; 1.0 gives 0.853).
RAW_CORE_HEIGHT = 1.17

# Test name -> (output directory, solver settings, cases).
# A case is (label, [(scaling, factor, extra kwargs), ...]) applied in order to
# the unscaled discharge, so ("Te_ped_scale", 1.0) first means "on the fit-Te
# base".
FIT_TE = ("Te_ped_scale", 1.0, {})
TESTS = {
    "G": ("G_istar", ISTAR, [
        (tag("omt", 1.0), [("omt", 1.0, {})]),
        (tag("Te_ped_scale", 1.0), [FIT_TE]),
    ]),
    "H": ("H_fitbase", JPARALLEL, [
        ("fitTe_" + tag(name, s), [FIT_TE, (name, s, {})])
        for name in ("omt", "omne", "ne_ped_scale")
        for s in (1.0, 0.8, 1.2)
    ]),
    "I": ("I_rawcore", JPARALLEL, [
        ("fitTe_rawcore_%.2f" % RAW_CORE_HEIGHT,
         [("Te_ped_scale", 1.0, {"scale_core_height": RAW_CORE_HEIGHT})]),
    ]),
}

# Box-scan runs drawn beside G so the figure shows the full 2x2: the same two
# unity cases under jparallel. omt_1.000 failed there and has no EQDSK, so the
# comparison skips it; it is listed anyway in case a rerun lands one.
G_JPARALLEL_REFERENCE = ["jparallel_box/%d/%d/omt_1.000" % (SHOT, SHOT),
                         "jparallel_box/%d/%d/Te_ped_scale_1.000" % (SHOT, SHOT)]


class _Tee:
    """Write to the terminal and a log file at once, flushing both."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "a", buffering=1)

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def repo_root(explicit=None) -> str:
    for d in ([explicit] if explicit else REPO_CANDIDATES):
        if d and os.path.isdir(os.path.join(d, "data")):
            return os.path.abspath(d)
    raise FileNotFoundError("no cheaseBS_tests checkout with a data/ directory "
                            "in %s; pass --repo" % (explicit or REPO_CANDIDATES))


def test_dir(repo, test) -> str:
    return os.path.join(repo, "data", CAMPAIGN, TESTS[test][0])


def _git_head(path) -> str:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def build(steps):
    """Apply each (scaling, factor, kwargs) in order to the unscaled shot."""
    q = load(SHOT)
    for name, s, kw in steps:
        q = scale(q, name, s, **kw)
    return q


def run_test(test, repo, dry_run=False):
    sub, settings, cases = TESTS[test]
    out = test_dir(repo, test)
    os.makedirs(out, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, os.path.join(out, "console.log"))
    sys.stderr = _Tee(sys.__stderr__, os.path.join(out, "console.log"))

    here = os.path.dirname(os.path.abspath(__file__))
    gfile_kw = dict(settings)
    if not dry_run:
        # Needs TPED's CHEASEBS_PATH config, which only the NERSC checkout has.
        gfile_kw["chease_namelist"] = namelist_with_ncscal(NCSCAL, out)
    print("=== test %s: %d case(s) -> %s ===" % (test, len(cases), out), flush=True)
    print("settings: %s" % json.dumps(settings), flush=True)

    results = {}
    for label, steps in cases:
        case = os.path.join(out, str(SHOT), label)
        print("--- %d %s : %s ---" % (SHOT, label,
              " -> ".join("%s %.3g %s" % (n, s, kw or "") for n, s, kw in steps)),
              flush=True)
        try:
            q = build(steps)
            if dry_run:
                print("  dry run: transforms %s" % [h.get("transform") for h in q._history],
                      flush=True)
                results[label] = "dry-run"
                continue
            os.makedirs(case, exist_ok=True)
            path = q.output_gfile(savedir=case, run_cheasebs=True,
                                  comment="%d_%s" % (SHOT, label), **gfile_kw)
            print("  gfile   %s" % path, flush=True)
            results[label] = "ok"
        except Exception as exc:                                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[label] = "failed: %r" % (exc,)

    n_ok = sum(v in ("ok", "dry-run") for v in results.values())
    print("=== test %s: %d/%d ok ===" % (test, n_ok, len(results)), flush=True)
    if not dry_run:
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"test": test, "shot": SHOT, "results": results,
                       "settings": settings, "ncscal": NCSCAL,
                       "cases": {lab: [[n, s, kw] for n, s, kw in st]
                                 for lab, st in cases},
                       "NSTXU_lithium_study": _git_head(here),
                       "finished": datetime.datetime.now().isoformat(timespec="seconds")},
                      f, indent=2)
        compare(test, repo)
    return 0 if n_ok == len(results) else 1


def compare(test, repo):
    """TPED compare_cheasebs_runs.py over one test, figures into its directory."""
    from TPED.projects.discharge_tools.src import discharge_physics
    script = os.path.join(os.path.dirname(os.path.dirname(discharge_physics.__file__)),
                          "scripts", "compare_cheasebs_runs.py")
    out = test_dir(repo, test)
    variations = [os.path.join(out, str(SHOT))]
    if test == "G":
        variations += [os.path.join(repo, "data", p) for p in G_JPARALLEL_REFERENCE
                       if os.path.isdir(os.path.join(repo, "data", p))]
    cmd = [sys.executable, script,
           "--baseline", os.path.join(discharge_root(), str(SHOT)),
           "--variations", *variations,
           "--outdir", out, "--plot-combine", "mtanh", "om",
           "--edge-by", "replay_representation",
           "--pedestal", "0.8,0.98", "--slices", "0.85", "0.9", "0.95", "0.975"]
    print("compare: %s" % " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    print("compare exit %d; figures in %s" % (rc, out), flush=True)
    return rc


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=sorted(TESTS) + ["compare"])
    ap.add_argument("test", nargs="?", choices=sorted(TESTS),
                    help="with compare: which test's runs to plot")
    ap.add_argument("--repo", default=None,
                    help="cheaseBS_tests checkout; default: first of %s" % REPO_CANDIDATES)
    ap.add_argument("--dry-run", action="store_true",
                    help="build every case's transforms, solve nothing")
    args = ap.parse_args(argv)
    repo = repo_root(args.repo)
    if args.action == "compare":
        if not args.test:
            ap.error("compare needs a test: G, H or I")
        return compare(args.test, repo)
    return run_test(args.action, repo, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
