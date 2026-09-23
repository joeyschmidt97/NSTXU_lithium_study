"""Bare-bones scaling driver for the final convergence check.

One discharge -> one DischargePhysics -> one transform per scaling.

    python scalings.py                                    # all four shots
    python scalings.py --shots 129015 129038 132543
    python scalings.py --shot 132588 --scalings omt omne --scales 0.7 0.9 1.3
    python scalings.py --shot 132588 --scales 1.3 --cheasebs --strict
    python scalings.py --shots 129015 --scales 0.95 1.05 --cheasebs --ncscal 4

Plots are on by default (--no-plots to skip): two PNGs per transform family,
one full-profile and one zoomed on the pedestal. --cheasebs additionally hands
every case to output_gfile with run_cheasebs=True; the reconstruction itself is
output_gfile's job, this only picks the cases and the destination.

or from a notebook:

    from scalings import load, scale, run, SCALINGS

    phys = load(132588)
    q    = scale(phys, "Te_ped_scale", 1.3)     # mtanh_full step-amplitude
    q    = scale(phys, "omt", 0.7)              # gradient power law

    run(132588, plot_printouts=True)            # PNG per transform family
    run(132588, ["Te_ped_scale"], (1.3,), cheasebs=True)
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

# Headless as a script only: the fit path can reach for pyplot and there is no
# display on a login node. Left alone on import, so a notebook keeps its own
# backend and its figures stay interactive.
if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

# Import TPED from this checkout rather than whatever is installed.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _parent in (_HERE, os.path.dirname(_HERE)):
    _root = os.path.dirname(os.path.dirname(_parent))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from TPED.projects.discharge_tools.src.discharge_data import DischargeData
from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics
from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import fit_mtanh_full

# ---------------------------------------------------------------------------
# Discharges
# ---------------------------------------------------------------------------

# First path that exists wins, so the same code runs on NERSC and on a laptop.
DISCHARGE_ROOT_CANDIDATES = [
    "/global/homes/j/joeschm/data/ST_research/NSTXU_discharges",
    "C:/Users/joesc/git/ST_research/NSTXU_discharges",
]

SHOTS = (129015, 129038, 132543, 132588)

# 129038's directory holds five pfiles, so auto-discovery refuses to guess.
PFILES = {129038: "p129038.00400"}

# ---------------------------------------------------------------------------
# Scalings
# ---------------------------------------------------------------------------

# Campaign fit setting for mtanh_full. One value everywhere.
FIT_KWARGS = dict(pedestal_weight=8.0)
QZ = 6.0

# (rhot_topped, rhot_midped) for the omt/omne gradient transforms.
RAMP_WINDOW = (0.8, 1.0)

# cheaseBS outer-iteration cap, overriding the bundled NSTX template's 25.
#
# The template's 25 is ~1/istar_mix = 20 iterations to apply one full current
# update, plus margin (cheasebs_runner.py, "Iteration count"). That is a
# property of the under-relaxation weight, not of any discharge: the same cap
# is applied to all four regardless of how far each one's loop has to travel.
#
# The cap does not cause divergence and cannot cure it. A loop that blows up at
# 50 was already diverging by 20; 25 truncated it before the damage was visible
# and the endpoint then read as converged -- 132588 ne_ped_scale_0.700 is the
# case on record, +2.16% Ip at 25 against +181% at 50 (2026-08-31 truth-table
# audit). So 50 is the more honest setting, not the more dangerous one: it
# reveals which way the residual is going. Read the trace, not the endpoint.
#
# What actually converged that point (+0.47%) was damping -- istar_mix
# 0.05 -> 0.02 with bootstrap_mix 0.1 -> 0.05 -- which says the iteration gain
# is too high there, not merely that the initial equilibrium is too far away.
# Neither mix is changed here; both are reachable through gfile_kw.
MAX_ITER = 50

# name -> (apply_X key, transform-specific spec). `apply` selects the function in
# APPLY; the rest is what that function needs to pin down the single knob.
#   mtanh_full : (var, kwarg)  -- kwarg is the apply_mtanh_full keyword scaled
#   om         : (method,)     -- apply_omt and apply_omne are separate methods,
#                                 unlike apply_mtanh_full which takes var
SCALINGS = {
    "Te_ped_scale":   {"apply": "mtanh_full", "var": "Te", "kwarg": "scale_height"},
    "ne_ped_scale":   {"apply": "mtanh_full", "var": "ne", "kwarg": "scale_height"},
    "omt":            {"apply": "om", "method": "apply_omt"},
    "omne":           {"apply": "om", "method": "apply_omne"},
}

PLOT_VARS = ("Te", "Ti", "ne", "ni")

# Two views per figure, because each one hides the other's failure. A pedestal
# height change is a few percent of a core-scaled axis and is illegible on the
# full profile; the core drift these transforms can drag along with it is
# invisible in the zoom. None means the whole radius.
PLOT_VIEWS = {"full": None, "ped": (0.6, 1.0)}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def discharge_root() -> str:
    for d in DISCHARGE_ROOT_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"no discharge root in {DISCHARGE_ROOT_CANDIDATES}")


def load(shot: int) -> DischargePhysics:
    """DischargePhysics for one shot, raw tree kept so gfile output works."""
    d = os.path.join(discharge_root(), str(shot))
    kw = {"input_dir": d}
    if shot in PFILES:
        kw["pfile"] = os.path.join(d, PFILES[shot])
    return DischargePhysics(DischargeData(**kw))


# ---------------------------------------------------------------------------
# apply_X transforms
# ---------------------------------------------------------------------------

def _fit(phys: DischargePhysics, var: str):
    """Nominal StefanikovaProfile for one variable, cached on the object.

    Fitted once off the unscaled profiles so every scale factor is measured
    against the same fit rather than a refit of its own output.
    """
    cache = phys.__dict__.setdefault("_scaling_fits", {})
    if var not in cache:
        cache[var] = fit_mtanh_full(phys.ds, var, **FIT_KWARGS)[0]
    return cache[var]


def _apply_mtanh_full(phys, spec, s, **kw):
    """Stefanikova F_full reconstruction with one fit parameter scaled.

    Exactly the transform make_scan_handoff.py hands off: the nominal fit, one
    scaled keyword, quasineutrality on so ni and nz follow ne.
    """
    var, kwarg = spec["var"], spec["kwarg"]
    return phys.apply_mtanh_full(
        var, fit=_fit(phys, var), **{kwarg: s},
        enforce_quasineutrality=True, qz=QZ, **kw)


def _apply_om(phys, spec, s, *, window=None, **kw):
    """Gradient power law, re-exponentiated about rhot_midped.

    apply_omt and apply_omne are separate methods with different signatures
    (omt carries apply_to_tz, omne carries qz), so the scaling dict names which
    one to call rather than passing a var like apply_mtanh_full does.
    """
    top, mid = window or RAMP_WINDOW
    if spec["method"] == "apply_omne":
        kw.setdefault("qz", QZ)
    return getattr(phys, spec["method"])(
        s, rhot_midped=mid, rhot_topped=top, **kw)


APPLY = {"mtanh_full": _apply_mtanh_full, "om": _apply_om}


def scale(phys: DischargePhysics, scaling: str, s: float, **kw) -> DischargePhysics:
    """Apply one scaling at one factor. Returns a new DischargePhysics."""
    if scaling not in SCALINGS:
        raise ValueError(f"scaling must be one of {tuple(SCALINGS)}, got {scaling!r}")
    spec = SCALINGS[scaling]
    return APPLY[spec["apply"]](phys, spec, s, **kw)


def tag(scaling: str, s: float) -> str:
    return "%s_%.3f" % (scaling, s)


def iter_scaled(phys: DischargePhysics, scalings=None, scales=(0.7, 1.3), **kw):
    """Yield (scaling, factor, scaled_physics) over the requested grid.

    Each case starts from the same unscaled `phys`, so the cases are independent
    single-knob perturbations rather than a cumulative chain.
    """
    for name in (scalings or SCALINGS):
        for s in scales:
            yield name, s, scale(phys, name, s, **kw)


# ---------------------------------------------------------------------------
# Scaling check printouts
# ---------------------------------------------------------------------------

def scratch_root() -> str:
    """Parent directory for check output.

    $SCRATCH, else $PSCRATCH, else the platform temp dir. Both are checked with
    isdir rather than trusted: an exported-but-absent SCRATCH takes mkdtemp down
    with a FileNotFoundError that reads as a bug in the transforms.
    """
    for var in ("SCRATCH", "PSCRATCH"):
        root = os.environ.get(var)
        if root and os.path.isdir(root):
            return root
    return tempfile.gettempdir()


def scratch_dir(label) -> str:
    """A fresh temp directory for check output, under scratch_root().

    `label` names the whole invocation, not one shot: a multi-shot run resolves
    this once and shares it, so the grid lands in one directory instead of one
    temp directory per shot.
    """
    return tempfile.mkdtemp(prefix="scaling_check_%s_" % label,
                            dir=scratch_root())


def namelist_with_ncscal(ncscal: int, dest_dir: str) -> str:
    """A copy of the NSTX CHEASE namelist with NCSCAL set, written to dest_dir.

    NCSCAL lives in the CHEASE namelist, not in the cheaseBS JSON config -- the
    config's key list is closed and rejects anything it does not know -- so the
    only way to vary it per run is to hand cheaseBS an edited template. The copy
    is written beside the run output, which also records what was used.

    NCSCAL selects how CHEASE normalises the equilibrium it builds; see the
    CHEASE manual for the meaning of each value. The NSTX template ships with 1.
    """
    from TPED.config.config_helper import Config

    src = os.path.join(Config().get_path("CHEASEBS_PATH"), "chease_namelist_nstx")
    with open(src) as f:
        text = f.read()
    new, n = re.subn(r"NCSCAL\s*=\s*-?\d+", "NCSCAL=%d" % ncscal, text, count=1)
    if not n:
        raise ValueError("no NCSCAL entry in %s to replace" % src)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "chease_namelist_nstx_ncscal%d" % ncscal)
    with open(dest, "w") as f:
        f.write(new)
    return dest


def _case_legend(fig, labels):
    """Add a colour -> case legend naming each scaling.

    plot_profiles' own legend is species-only (black lines, linestyle per
    species) and never draws the `label` argument, so without this the figure
    shows which variable a line is but not which scaling produced it.
    """
    import matplotlib.lines as mlines

    first = {}
    for rec in getattr(fig, "_discharge_artists", []):
        first.setdefault(rec["discharge_idx"], rec["artist"])
    handles = [mlines.Line2D([], [], color=first[i].get_color(), linewidth=2.4,
                             label=lab)
               for i, lab in enumerate(labels) if i in first]
    if not handles:
        return fig
    for ax in fig.axes:
        if ax.get_legend():
            ax.add_artist(ax.get_legend())      # keep the species legend
        ax.legend(handles=handles, fontsize=7, loc="best")
    return fig


def plot_family(base, cases, path, vars=PLOT_VARS, xlim=None,
                xcoord="rhot"):
    """Every case sharing one transform on one figure. Returns the PNG path.

    Grouped by transform because that is the comparison worth making: the
    scalings of one family differ only in their knob, so they belong on shared
    axes, while mtanh_full and the gradient power law reshape the profile in
    different ways and would only clutter each other.

    The base is always drawn first --- a scaled profile on its own looks
    plausible at any factor, including one that did nothing.
    """
    import matplotlib.pyplot as plt

    kw = {xcoord: list(xlim)} if xlim else {}
    labels = ["base"] + [lab for lab, _ in cases]
    fig = base.plot_profiles(vars=vars, xcoord=xcoord, discharge_idx=0, **kw)
    for q in (q for _, q in cases):
        # discharge_idx is taken from the figure's own counter once fig is
        # passed, so each case lands in the next colour family on its own.
        fig = q.plot_profiles(vars=vars, xcoord=xcoord, fig=fig, **kw)
    _case_legend(fig, labels)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run(shot: int, scalings=None, scales=(0.7, 1.3), *, plot_printouts=False,
        cheasebs=False, savedir=None, gfile_kw=None, failures=None, **kw):
    """Scale one discharge across the grid. Returns {tag: DischargePhysics}.

    plot_printouts writes two PNGs per transform family --- full profile and
    pedestal zoom --- so the transforms can be eyeballed before anything
    expensive is launched on them.

    cheasebs hands each scaled case to output_gfile with run_cheasebs=True,
    writing to savedir/<shot>/<tag>/. Reconstruction itself is entirely
    output_gfile's: this only decides which cases it gets and where the results
    land. Plots are written first when both are asked for, so there is
    something to look at while the solves run.

    gfile_kw is forwarded to output_gfile (cheasebs_strict, cheasebs_config,
    cheasebs_acceptance, max_iter, istar_mix, ...). A list passed as `failures`
    collects "<shot> <tag>" for every case that raised, so a caller looping over
    shots can exit non-zero without re-reading the log.

    Every print flushes, and the destination is resolved and announced before
    the loading and fitting rather than after: block-buffered stdout under a
    batch scheduler otherwise holds the whole log until exit, so a run that is
    still fitting --- or that died in it --- looks like a run that wrote
    nothing and said nothing about where.
    """
    if plot_printouts or cheasebs:
        savedir = os.path.abspath(savedir or scratch_dir(str(shot)))
        os.makedirs(savedir, exist_ok=True)
        print("=== scaling check: %d ===" % shot, flush=True)
        print("output -> %s" % savedir, flush=True)

    print("loading %d ..." % shot, flush=True)
    base = load(shot)
    out, families = {}, {}
    for name, s, q in iter_scaled(base, scalings, scales, **kw):
        label = tag(name, s)
        out[label] = q
        families.setdefault(SCALINGS[name]["apply"], []).append((label, q))
        print("  scaled  %s" % label, flush=True)

    if plot_printouts:
        n = 0
        for family, cases in families.items():
            for view, xlim in PLOT_VIEWS.items():
                path = os.path.join(savedir,
                                    "%d_%s_%s.png" % (shot, family, view))
                plot_family(base, cases, path, xlim=xlim)
                n += 1
                print("  wrote   %s  (%d case(s): %s)"
                      % (path, len(cases), ", ".join(lab for lab, _ in cases)),
                      flush=True)
        print("=== %d PNG(s) in %s ===" % (n, savedir), flush=True)

    if cheasebs:
        failed = []
        # MAX_ITER is a default here, not a floor: an explicit max_iter in
        # gfile_kw wins, and passing max_iter=None falls back to the template.
        gfile_kw = dict(gfile_kw or {})
        gfile_kw.setdefault("max_iter", MAX_ITER)
        if gfile_kw.get("max_iter") is None:
            del gfile_kw["max_iter"]
        print("=== cheaseBS: %d, %d case(s), max_iter=%s ==="
              % (shot, len(out), gfile_kw.get("max_iter", "template")),
              flush=True)
        for label, q in out.items():
            case = os.path.join(savedir, str(shot), label)
            os.makedirs(case, exist_ok=True)
            print("--- %d %s ---" % (shot, label), flush=True)
            try:
                # run_cheasebs=True rather than None: the prompt is unanswerable
                # in a batch job, and a transform history is always present here.
                path = q.output_gfile(savedir=case, run_cheasebs=True,
                                      comment="%d_%s" % (shot, label),
                                      **gfile_kw)
                print("  gfile   %s" % path, flush=True)
            except Exception:                                    # noqa: BLE001
                # One rejected or diverged case must not take the rest of the
                # grid with it; what converged is still on disk.
                import traceback
                traceback.print_exc()
                failed.append(label)
                if failures is not None:
                    failures.append("%d %s" % (shot, label))
        print("=== %d: %d/%d ok in %s ==="
              % (shot, len(out) - len(failed), len(out), savedir), flush=True)
        if failed:
            print("FAILED: %s" % ", ".join(failed), flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Scale one discharge and check the transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Plural and variadic, defaulting to the whole campaign: the four shots are
    # the campaign, so a bare run should cover them, and naming three of them
    # is an ordinary request rather than an error.
    ap.add_argument("--shots", "--shot", dest="shots", type=int, nargs="+",
                    default=list(SHOTS), choices=sorted(SHOTS),
                    help="default: every shot in SHOTS")
    ap.add_argument("--scalings", nargs="+", default=None, choices=sorted(SCALINGS),
                    help="default: every scaling in SCALINGS")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.7, 1.3])
    # Plots are the point of a run without --cheasebs: nothing else is written,
    # so a bare run would scale the profiles and throw them away. On by default;
    # --plot-printouts is kept so the explicit form still works, and --no-plots
    # is the way to opt out --- worth doing alongside --cheasebs on a shot whose
    # scalings have already been eyeballed.
    ap.add_argument("--plot-printouts", "--plot-output", dest="plot_printouts",
                    action="store_true", default=True,
                    help="write full-profile and pedestal-zoom PNGs per "
                         "transform family (default)")
    ap.add_argument("--no-plots", dest="plot_printouts", action="store_false",
                    help="skip the check plots")
    ap.add_argument("--cheasebs", action="store_true",
                    help="reconstruct each case through output_gfile with "
                         "run_cheasebs=True, into <savedir>/<shot>/<tag>/")
    ap.add_argument("--strict", action="store_true",
                    help="with --cheasebs, raise on a rejected equilibrium "
                         "instead of returning it; set this when the output "
                         "feeds GENE runs")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER,
                    help="cheaseBS outer-iteration cap; 0 defers to the "
                         "bundled template (25). Read the MAX_ITER comment "
                         "before trusting a run that hits the cap")
    ap.add_argument("--ncscal", type=int, default=None,
                    help="CHEASE NCSCAL, applied by writing an edited copy of "
                         "the namelist template into the output directory. The "
                         "cheaseBS JSON config cannot carry it -- its key list "
                         "is closed -- so this is the only per-run route. "
                         "Default: leave the template's own value (1)")
    ap.add_argument("--savedir", default=None,
                    help="default: a fresh temp dir on $SCRATCH, else $PSCRATCH, "
                         "else the platform temp dir")
    args = ap.parse_args(argv)

    # Resolved once for the whole invocation rather than per shot, so a
    # multi-shot grid lands in one directory instead of four temp directories
    # that have to be collected by hand afterwards.
    savedir = args.savedir or scratch_dir("-".join(str(s) for s in args.shots))
    failures, broken = [], []
    gfile_kw = {"max_iter": args.max_iter or None}
    if args.strict:
        gfile_kw["cheasebs_strict"] = True
    if args.ncscal is not None:
        # Written once for the whole invocation: every case of the grid is then
        # solved against the same namelist, and the file sits beside the output
        # as the record of what that run used.
        gfile_kw["chease_namelist"] = namelist_with_ncscal(args.ncscal, savedir)
        print("NCSCAL=%d via %s" % (args.ncscal, gfile_kw["chease_namelist"]),
              flush=True)

    for shot in args.shots:
        try:
            run(shot, args.scalings, tuple(args.scales),
                plot_printouts=args.plot_printouts, cheasebs=args.cheasebs,
                savedir=savedir, failures=failures, gfile_kw=gfile_kw)
        except Exception:                                        # noqa: BLE001
            # A shot that cannot even be loaded or fitted must not take the
            # remaining shots with it.
            import traceback
            traceback.print_exc()
            broken.append(shot)

    print("\n=== %d/%d shot(s) ran, output in %s ==="
          % (len(args.shots) - len(broken), len(args.shots), savedir),
          flush=True)
    if broken:
        print("SHOTS RAISED: %s" % ", ".join(str(s) for s in broken), flush=True)
    if failures:
        print("CASES FAILED: %s" % "; ".join(failures), flush=True)
    return 1 if (broken or failures) else 0


if __name__ == "__main__":
    sys.exit(main())
