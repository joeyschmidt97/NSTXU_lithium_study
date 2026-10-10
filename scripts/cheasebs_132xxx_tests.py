"""cheaseBS convergence tests for the two 132xxx discharges.

Design and reading tables: vault note 2026-10-02-cheasebs-132xxx-convergence-plan.

132588 -- profile base (the baseline itself does not reconstruct under jparallel;
every box-scan case on raw Te failed, every case on the mtanh-fit Te passed;
the fit drops axis Te to 0.853 of raw, and raw ni carries a kink):

    G  raw base (omt 1.0) vs fit-Te base (Te_ped_scale 1.0), both under I-star
    H  omt / omne / ne_ped_scale at 1.0, 0.8, 1.2 on the fit-Te base, jparallel
    I  fit-Te base with the core raised to the raw axis value, jparallel

132543 -- option ladder (profiles are clean and fit ~ raw within 2%; failures are
a deterministic CHEASE segfault in the magnetic-axis search, flipping on ~1%
input changes). One setting changed per rung against the approved jparallel
configuration, on the same five-case pilot: the passing Te_ped_scale 1.0 as a
control plus four box-scan failures inside the 0.8-1.2 campaign box.

    B  I-star replay on rho_pol (bootstrap_mix 0.1)
    C  NCSCAL=1
    D  jparallel on rho_pol
    T  TENSBND=-1.0 (boundary spline smoothing; template -0.3)

Output goes straight into the cheaseBS_tests checkout, one directory per test:

    <repo>/data/cheasebs_132xxx/<test>_<name>/
        <shot>/<case>/            cheaseBS run, same layout as data/jparallel_box
        console.log               everything printed, tracebacks included
        summary.json              ok / failed per case, settings, commit
        cheasebs_comparison_*.png TPED compare_cheasebs_runs.py figures

Failed cases keep only their copied gfile, as in the box scan; console.log is
the record of why. After a test the comparison runs on its own, with the
matching box-scan jparallel runs overlaid as the reference; `compare` reruns
it alone.

    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests G
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests B
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests compare B
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests H --dry-run
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

from NSTXU_lithium_study.scripts.scalings import discharge_root, load, scale, tag

# First path that exists wins, as for the discharge root in scalings.py.
REPO_CANDIDATES = [
    "/global/homes/j/joeschm/projects/cheaseBS_tests",
    "C:/Users/joesc/git/cheaseBS_tests",
]
CAMPAIGN = "cheasebs_132xxx"

# The approved jparallel configuration (2026-09-30 box scan). Every test
# changes exactly one thing against it: settings here, or a namelist entry.
JPARALLEL = dict(replay_representation="jparallel", coordinate="rhot",
                 bootstrap_mix=0.35, amplitude_warmup_iters=0,
                 tol_bs=1e-3, tol_q=1e-3, tol_ip_rel=1e-3, max_iter=40)
# I-star with the defaults it was run with in September (rho_pol, mix 0.1).
ISTAR = dict(JPARALLEL, replay_representation="istar", coordinate="rhop",
             bootstrap_mix=0.1)
RHOP = dict(JPARALLEL, coordinate="rhop")
NAMELIST = {"NCSCAL": 4}

# scale_core_height that puts 132588's fitted axis Te back on the raw value:
# 1.17 gives 0.998 of raw (measured 2026-10-10; 1.0 gives 0.853).
RAW_CORE_HEIGHT = 1.17

# A case is (label, [(scaling, factor, extra kwargs), ...]) applied in order to
# the unscaled discharge, so ("Te_ped_scale", 1.0) first means "on the fit-Te
# base".
FIT_TE = ("Te_ped_scale", 1.0, {})


def _single(name, s):
    return (tag(name, s), [(name, s, {})])


# 132543 pilot: control, then the box-scan failures inside the campaign box
# that cover both crash types (Te 1.1 dies in iteration 00, Te 1.2 in 02).
PILOT_132543 = [_single("Te_ped_scale", 1.0), _single("Te_ped_scale", 1.1),
                _single("Te_ped_scale", 1.2), _single("ne_ped_scale", 0.9),
                _single("ne_ped_scale", 1.2)]

# name -> dict(shot, dir, settings, namelist, cases, ref)
#   namelist : entries written into the CHEASE namelist copy
#   ref      : box-scan case labels overlaid on the comparison as jparallel
#              reference (only those that produced an EQDSK are drawn)
TESTS = {
    "G": dict(shot=132588, dir="G_istar", settings=ISTAR, namelist=NAMELIST,
              cases=[_single("omt", 1.0), _single("Te_ped_scale", 1.0)],
              ref=["omt_1.000", "Te_ped_scale_1.000"]),
    "H": dict(shot=132588, dir="H_fitbase", settings=JPARALLEL, namelist=NAMELIST,
              cases=[("fitTe_" + tag(name, s), [FIT_TE, (name, s, {})])
                     for name in ("omt", "omne", "ne_ped_scale")
                     for s in (1.0, 0.8, 1.2)],
              ref=["Te_ped_scale_1.000"]),
    "I": dict(shot=132588, dir="I_rawcore", settings=JPARALLEL, namelist=NAMELIST,
              cases=[("fitTe_rawcore_%.2f" % RAW_CORE_HEIGHT,
                      [("Te_ped_scale", 1.0, {"scale_core_height": RAW_CORE_HEIGHT})])],
              ref=["Te_ped_scale_1.000"]),
    "B": dict(shot=132543, dir="B_istar", settings=ISTAR, namelist=NAMELIST,
              cases=PILOT_132543, ref=["Te_ped_scale_1.000"]),
    "C": dict(shot=132543, dir="C_ncscal1", settings=JPARALLEL,
              namelist={"NCSCAL": 1}, cases=PILOT_132543,
              ref=["Te_ped_scale_1.000"]),
    "D": dict(shot=132543, dir="D_rhop", settings=RHOP, namelist=NAMELIST,
              cases=PILOT_132543, ref=["Te_ped_scale_1.000"]),
    "T": dict(shot=132543, dir="T_tensbnd", settings=JPARALLEL,
              namelist=dict(NAMELIST, TENSBND=-1.0), cases=PILOT_132543,
              ref=["Te_ped_scale_1.000"]),
}


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
    return os.path.join(repo, "data", CAMPAIGN, TESTS[test]["dir"])


def _git_head(path) -> str:
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return "unknown"


def namelist_with(entries, dest_dir) -> str:
    """A copy of the NSTX CHEASE namelist with each KEY=value replaced.

    Same route as scalings.namelist_with_ncscal, for any entry the template
    already carries. A key missing from the template is an error rather than an
    insertion: an inserted line can land in CHEASE's comment block and be
    silently ignored (the RELAX trap, 2026-09-30).
    """
    from TPED.config.config_helper import Config

    src = os.path.join(Config().get_path("CHEASEBS_PATH"), "chease_namelist_nstx")
    with open(src) as f:
        text = f.read()
    for key, value in entries.items():
        text, n = re.subn(r"\b%s\s*=\s*[-+.\dEe]+" % key,
                          "%s=%s" % (key, value), text, count=1)
        if not n:
            raise ValueError("no %s entry in %s to replace" % (key, src))
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "chease_namelist_nstx_" + "_".join(
        "%s%s" % (k.lower(), v) for k, v in entries.items()))
    with open(dest, "w") as f:
        f.write(text)
    return dest


def build(shot, steps):
    """Apply each (scaling, factor, kwargs) in order to the unscaled shot."""
    q = load(shot)
    for name, s, kw in steps:
        q = scale(q, name, s, **kw)
    return q


def run_test(test, repo, dry_run=False):
    spec = TESTS[test]
    shot, settings, cases = spec["shot"], spec["settings"], spec["cases"]
    out = test_dir(repo, test)
    os.makedirs(out, exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, os.path.join(out, "console.log"))
    sys.stderr = _Tee(sys.__stderr__, os.path.join(out, "console.log"))

    here = os.path.dirname(os.path.abspath(__file__))
    gfile_kw = dict(settings)
    if not dry_run:
        # Needs TPED's CHEASEBS_PATH config, which only the NERSC checkout has.
        gfile_kw["chease_namelist"] = namelist_with(spec["namelist"], out)
    print("=== test %s (%d): %d case(s) -> %s ===" % (test, shot, len(cases), out),
          flush=True)
    print("settings: %s  namelist: %s" % (json.dumps(settings),
                                          json.dumps(spec["namelist"])), flush=True)

    results = {}
    for label, steps in cases:
        case = os.path.join(out, str(shot), label)
        print("--- %d %s : %s ---" % (shot, label,
              " -> ".join("%s %.3g %s" % (n, s, kw or "") for n, s, kw in steps)),
              flush=True)
        try:
            q = build(shot, steps)
            if dry_run:
                print("  dry run: transforms %s" % [h.get("transform") for h in q._history],
                      flush=True)
                results[label] = "dry-run"
                continue
            os.makedirs(case, exist_ok=True)
            path = q.output_gfile(savedir=case, run_cheasebs=True,
                                  comment="%d_%s" % (shot, label), **gfile_kw)
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
            json.dump({"test": test, "shot": shot, "results": results,
                       "settings": settings, "namelist": spec["namelist"],
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
    spec = TESTS[test]
    shot, out = spec["shot"], test_dir(repo, test)
    box = os.path.join(repo, "data", "jparallel_box", str(shot), str(shot))
    variations = [os.path.join(out, str(shot))] + [
        os.path.join(box, lab) for lab in spec["ref"]
        if os.path.isdir(os.path.join(box, lab))]
    cmd = [sys.executable, script,
           "--baseline", os.path.join(discharge_root(), str(shot)),
           "--variations", *variations,
           "--outdir", out, "--plot-combine", "mtanh", "om",
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
            ap.error("compare needs a test: one of %s" % ", ".join(sorted(TESTS)))
        return compare(args.test, repo)
    return run_test(args.action, repo, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
