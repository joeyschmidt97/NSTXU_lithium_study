"""cheaseBS convergence tests for the two 132xxx discharges.

Design and reading tables: vault note 2026-10-02-cheasebs-132xxx-convergence-plan.

132588 -- profile base (the baseline itself does not reconstruct under jparallel;
every box-scan case on raw Te failed, every case on the mtanh-fit Te passed;
the fit drops axis Te to 0.853 of raw, and raw ni carries a kink):

    G  raw base (omt 1.0) vs fit-Te base (Te_ped_scale 1.0), both under I-star
    H  omt / omne / ne_ped_scale at 1.0, 0.8, 1.2 on the fit-Te base, jparallel
    I  fit-Te base with the core raised to the raw axis value, jparallel
    Q  H's ne cases with quasineutrality off, jparallel: 132588's pfile has
       ni/ne ~0.5 and 6nz/ne ~0.005, so the rebuild nz = (ne - ni)/6 raises nz
       ~100x even at omne 1.0, which crashed in H
    J  omt, and omne without the rebuild, at 0.8 and 1.2 on the raw base under
       I-star, the one 132588 base/representation pair that converged in G
    L  campaign mtanh knobs only, jparallel on rho_pol with TENSBND=-1.0 (K's
       settings): Te_ped_scale 1.0, 1.2, and ne_ped_scale 1.0, 0.8, 1.2 on the
       fit-Te base without the nz rebuild
    N  campaign outer bounds under L's settings: Te_ped_scale 0.8, 1.2, and
       ne_ped_scale 0.8, 1.2 on the fit-Te base without the nz rebuild, plus
       the fit-Te base itself as the 1.0 reference

132543 -- option ladder (profiles are clean and fit ~ raw within 2%; failures are
a deterministic CHEASE segfault in the magnetic-axis search, flipping on ~1%
input changes). One setting changed per rung against the approved jparallel
configuration, on the same five-case pilot: the passing Te_ped_scale 1.0 as a
control plus four box-scan failures inside the 0.8-1.2 campaign box.

    B  I-star replay on rho_pol (bootstrap_mix 0.1)
    C  NCSCAL=1
    D  jparallel on rho_pol
    T  TENSBND=-1.0 (boundary spline smoothing; template -0.3)
    K  D and T together: rho_pol and TENSBND=-1.0. D and T crash complementary
       pilot cases (D: Te 1.1; T: Te 1.2, ne 1.2)
    M  campaign outer bounds under K's settings: Te_ped_scale and ne_ped_scale
       at 0.8 and 1.2, plus Te_ped_scale 1.0 as the reference

Output goes straight into the cheaseBS_tests checkout, one directory per test:

    <repo>/data/cheasebs_132xxx/<test>_<name>/
        <shot>/<case>/            cheaseBS run, same layout as data/jparallel_box
        console.log               everything printed, cheaseBS's own output
                                  and tracebacks included
        summary.json              accepted / rejected / crashed per case,
                                  settings, commit
        cheasebs_comparison_*.png TPED compare_cheasebs_runs.py figures

Every case gets the scratch run directory's logs copied back (iteration_log.csv,
convergence_summary.json, profiles, ...), crashed ones included, so a case can
be diagnosed from the repo; console.log holds the crash itself. CHEASE's own
chease.log of the first and the last iteration lands in <case>/chease_logs/; for
a crash the last one is the solve that died. After a test the comparison runs on its own, with the
matching box-scan jparallel runs overlaid as the reference; `compare` reruns
it alone. `compare all` redraws every test that has a directory, and
--campaign profile_base_132588 points it at the directory the G/H/I tests
wrote to before the rename.

    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests G
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests B
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests Q
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests K
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests M
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests compare B
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests compare all
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests compare all --campaign profile_base_132588
    python -u -m NSTXU_lithium_study.scripts.cheasebs_132xxx_tests H --dry-run
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import time

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
TENSBND = {"TENSBND": -1.0}
NAMELIST = {"NCSCAL": 4}

# scale_core_height that puts 132588's fitted axis Te back on the raw value:
# 1.17 gives 0.998 of raw (measured 2026-10-10; 1.0 gives 0.853).
RAW_CORE_HEIGHT = 1.17

# A case is (label, [(scaling, factor, extra kwargs), ...]) applied in order to
# the unscaled discharge, so ("Te_ped_scale", 1.0) first means "on the fit-Te
# base".
FIT_TE = ("Te_ped_scale", 1.0, {})
# ne transforms without the ni/nz rebuild: ni and nz stay raw.
NO_QN = {"enforce_quasineutrality": False}


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
    "Q": dict(shot=132588, dir="Q_noqn", settings=JPARALLEL, namelist=NAMELIST,
              cases=[("fitTe_" + tag(name, s) + "_noqn",
                      [FIT_TE, (name, s, NO_QN)])
                     for name, s in (("omne", 1.0), ("omne", 0.8), ("omne", 1.2),
                                     ("ne_ped_scale", 1.0))],
              ref=["Te_ped_scale_1.000"]),
    "J": dict(shot=132588, dir="J_istar_raw", settings=ISTAR, namelist=NAMELIST,
              cases=[_single("omt", s) for s in (0.8, 1.2)]
                    + [(tag("omne", s) + "_noqn", [("omne", s, NO_QN)])
                       for s in (0.8, 1.2)],
              ref=["omt_1.000"]),
    "K": dict(shot=132543, dir="K_rhop_tensbnd", settings=RHOP,
              namelist=dict(NAMELIST, **TENSBND), cases=PILOT_132543,
              ref=["Te_ped_scale_1.000"]),
    "L": dict(shot=132588, dir="L_rhop_tensbnd", settings=RHOP,
              namelist=dict(NAMELIST, **TENSBND),
              cases=[_single("Te_ped_scale", 1.0), _single("Te_ped_scale", 1.2)]
                    + [("fitTe_" + tag("ne_ped_scale", s) + "_noqn",
                        [FIT_TE, ("ne_ped_scale", s, NO_QN)])
                       for s in (1.0, 0.8, 1.2)],
              ref=["Te_ped_scale_1.000"]),
    "M": dict(shot=132543, dir="M_bounds", settings=RHOP,
              namelist=dict(NAMELIST, **TENSBND),
              cases=[_single("Te_ped_scale", 1.0)]
                    + [_single(name, s) for name in ("Te_ped_scale", "ne_ped_scale")
                       for s in (0.8, 1.2)],
              ref=["Te_ped_scale_1.000"]),
    "N": dict(shot=132588, dir="N_bounds", settings=RHOP,
              namelist=dict(NAMELIST, **TENSBND),
              cases=[_single("Te_ped_scale", s) for s in (1.0, 0.8, 1.2)]
                    + [("fitTe_" + tag("ne_ped_scale", s) + "_noqn",
                        [FIT_TE, ("ne_ped_scale", s, NO_QN)])
                       for s in (0.8, 1.2)],
              ref=["Te_ped_scale_1.000"]),
}

# Outcome per case. "accepted" needs both a clean exit and cheaseBS's own
# acceptance verdict; an exit without raising is not enough, because a run that
# hits max_iter returns its last EQDSK unless cheasebs_strict is set (G's fit-Te
# case on 2026-10-10: 40 iterations, Ip +17.8%, recorded as "ok").
ACCEPTED = "accepted"

# Top-level files of the scratch run directory copied into the case directory.
RUN_DIR_FILES = ("profiles_e", "profiles_i", "profiles_z",
                 "cheasebs_run_config.json", "convergence_summary.json",
                 "cheasebs_acceptance.json", "iteration_log.csv",
                 "iteration_errors.png", "run_summary.png")


def _tee_fds(path):
    """Copy everything written to fd 1 and 2 into `path` as well.

    Done at the descriptor level, not by swapping sys.stdout, because cheaseBS
    runs as a subprocess that inherits the descriptors and writes to them
    directly: a Python-level tee never sees its output, which is where the
    crash reasons are. Needs `tee` on PATH; returns False without it.
    """
    import shutil
    tee = shutil.which("tee")
    if not tee:
        return False
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.Popen([tee, "-a", path], stdin=subprocess.PIPE)
    os.dup2(proc.stdin.fileno(), 1)
    os.dup2(proc.stdin.fileno(), 2)
    return True


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


def test_dir(repo, test, campaign=CAMPAIGN) -> str:
    return os.path.join(repo, "data", campaign, TESTS[test]["dir"])


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


def _run_dir(case, shot, label, since):
    """The scratch run directory output_gfile used for this case, or None.

    Read from the copied run config when it made it back; otherwise (a crash
    skips the copy-back) the newest <OUTPUT_PATH>/cheaseBS_runs/<date>/
    <stamp>-<shot>_<label> directory modified after `since`.
    """
    import glob
    cfg = os.path.join(case, "cheasebs_run_config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            d = json.load(f).get("output_dir")
        if d and os.path.isdir(d):
            return d
    try:
        from TPED.config.config_helper import Config
        root = Config().get_path("OUTPUT_PATH")
    except Exception:                                            # noqa: BLE001
        return None
    hits = [d for d in glob.glob(os.path.join(root, "cheaseBS_runs", "*",
                                              "*-%d_%s" % (shot, label)))
            if os.path.getmtime(d) >= since - 5]
    return max(hits, key=os.path.getmtime) if hits else None


def _copy_run_files(run_dir, case):
    import shutil
    copied = []
    for name in RUN_DIR_FILES:
        src = os.path.join(run_dir, name)
        if os.path.isfile(src) and not os.path.isfile(os.path.join(case, name)):
            shutil.copy2(src, os.path.join(case, name))
            copied.append(name)
    return copied


# chease.log files above this size are copied as their last CHEASE_LOG_TAIL
# bytes: the end of the log is where a crash reports.
CHEASE_LOG_MAX = 5_000_000
CHEASE_LOG_TAIL = 1_000_000


def _copy_chease_logs(run_dir, case):
    """CHEASE's chease.log of the first and last iteration into <case>/chease_logs/.

    cheaseBS runs CHEASE in iteration_NN/<stage dir>/stage/; a crash message
    names that log, but it stays on scratch. The first iteration shows the cold
    start, the last one is the solve that ended the run (the crashing one, for a
    crash). Each copy is named after its path below the run directory.
    """
    import glob
    import re as _re
    logs = glob.glob(os.path.join(run_dir, "iteration_*", "**", "chease.log"),
                     recursive=True)
    by_iter = {}
    for path in logs:
        m = _re.search(r"iteration_(\d+)", os.path.relpath(path, run_dir))
        if m:
            by_iter.setdefault(int(m.group(1)), []).append(path)
    if not by_iter:
        return []
    dest = os.path.join(case, "chease_logs")
    os.makedirs(dest, exist_ok=True)
    copied = []
    for it in sorted({min(by_iter), max(by_iter)}):
        for path in sorted(by_iter[it]):
            name = os.path.relpath(path, run_dir).replace(os.sep, "__")
            with open(path, "rb") as f:
                size = os.fstat(f.fileno()).st_size
                if size > CHEASE_LOG_MAX:
                    f.seek(size - CHEASE_LOG_TAIL)
                    data = b"[... first %d bytes cut ...]\n" % (size - CHEASE_LOG_TAIL) + f.read()
                else:
                    data = f.read()
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)
            copied.append("chease_logs/" + name)
    return copied


def _verdict(case):
    """accepted, or rejected: reasons, from the case's cheasebs_acceptance.json."""
    path = os.path.join(case, "cheasebs_acceptance.json")
    if not os.path.isfile(path):
        return "no acceptance record"
    with open(path) as f:
        acc = json.load(f)
    if acc.get("accepted") and acc.get("cheasebs_converged", True):
        return ACCEPTED
    reasons = list(acc.get("reasons") or [])
    if not acc.get("cheasebs_converged", True):
        reasons.insert(0, "not converged after %s iterations"
                       % acc.get("cheasebs_iterations"))
    return "rejected: " + "; ".join(reasons)


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
    log = os.path.join(out, "console.log")
    if not _tee_fds(log):
        print("no tee on PATH: console.log will miss cheaseBS's own output",
              flush=True)
        sys.stdout = _Tee(sys.__stdout__, log)
        sys.stderr = _Tee(sys.__stderr__, log)

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
        start = time.time()
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
            results[label] = None
        except Exception as exc:                                 # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[label] = "crashed: %r" % (exc,)
        run_dir = _run_dir(case, shot, label, start)
        if run_dir:
            copied = _copy_run_files(run_dir, case) + _copy_chease_logs(run_dir, case)
            print("  run dir %s (copied: %s)"
                  % (run_dir, ", ".join(copied) or "nothing new"), flush=True)
        if results[label] is None:
            results[label] = _verdict(case)
        print("  result  %s" % results[label], flush=True)

    n_ok = sum(v in (ACCEPTED, "dry-run") for v in results.values())
    print("=== test %s: %d/%d accepted ===" % (test, n_ok, len(results)),
          flush=True)
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


def compare(test, repo, campaign=CAMPAIGN):
    """TPED compare_cheasebs_runs.py over one test, figures into its directory."""
    from TPED.projects.discharge_tools.src import discharge_physics
    script = os.path.join(os.path.dirname(os.path.dirname(discharge_physics.__file__)),
                          "scripts", "compare_cheasebs_runs.py")
    spec = TESTS[test]
    shot, out = spec["shot"], test_dir(repo, test, campaign)
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
    ap.add_argument("test", nargs="?", choices=sorted(TESTS) + ["all"],
                    help="with compare: which test's runs to plot; all = every "
                         "test with a directory under --campaign")
    ap.add_argument("--repo", default=None,
                    help="cheaseBS_tests checkout; default: first of %s" % REPO_CANDIDATES)
    ap.add_argument("--campaign", default=CAMPAIGN,
                    help="with compare: data/ subdirectory holding the tests "
                         "(default %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every case's transforms, solve nothing")
    args = ap.parse_args(argv)
    repo = repo_root(args.repo)
    if args.action == "compare":
        if not args.test:
            ap.error("compare needs a test: all, or one of %s"
                     % ", ".join(sorted(TESTS)))
        if args.test != "all":
            return compare(args.test, repo, args.campaign)
        tests = [t for t in sorted(TESTS)
                 if os.path.isdir(test_dir(repo, t, args.campaign))]
        if not tests:
            ap.error("no test directories under %s"
                     % os.path.join(repo, "data", args.campaign))
        failed = [t for t in tests if compare(t, repo, args.campaign) != 0]
        print("compare all: %d test(s), failed: %s"
              % (len(tests), ", ".join(failed) or "none"), flush=True)
        return 1 if failed else 0
    return run_test(args.action, repo, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
