"""Bare-bones scaling driver for the final convergence check.

One discharge -> one DischargePhysics -> one transform per scaling.

    python scalings.py --shot 132588 --plot-printouts
    python scalings.py --shot 132588 --scalings omt omne --scales 0.7 0.9 1.3

or from a notebook:

    from scalings import load, scale, run, SCALINGS

    phys = load(132588)
    q    = scale(phys, "Te_ped_scale", 1.3)     # mtanh_full step-amplitude
    q    = scale(phys, "omt", 0.7)              # gradient power law

    run(132588, plot_printouts=True)            # PNG per transform family

Nothing here runs cheaseBS and nothing here writes a gfile --- call
``q.output_gfile(...)`` on a returned object for that.
"""

from __future__ import annotations

import os
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

# Plot window. A pedestal height change is a few percent of a core-scaled axis
# and is only legible zoomed, so the check plots come out zoomed by default.
PLOT_VARS = ("Te", "Ti", "ne", "ni")
PLOT_XLIM = (0.6, 1.0)


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


def scratch_dir(shot: int) -> str:
    """A fresh temp directory for check plots, under scratch_root()."""
    return tempfile.mkdtemp(prefix="scaling_check_%d_" % shot,
                            dir=scratch_root())


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


def plot_family(base, cases, path, vars=PLOT_VARS, xlim=PLOT_XLIM,
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
        savedir=None, **kw):
    """Scale one discharge across the grid. Returns {tag: DischargePhysics}.

    plot_printouts writes one PNG per transform family to a temp directory and
    runs nothing else --- no gfile, no cheaseBS --- so the transforms can be
    eyeballed before anything expensive is launched on them.

    Every print flushes, and the destination is resolved and announced before
    the loading and fitting rather than after: block-buffered stdout under a
    batch scheduler otherwise holds the whole log until exit, so a run that is
    still fitting --- or that died in it --- looks like a run that wrote
    nothing and said nothing about where.
    """
    if plot_printouts:
        savedir = os.path.abspath(savedir or scratch_dir(shot))
        os.makedirs(savedir, exist_ok=True)
        print("=== scaling check: %d ===" % shot, flush=True)
        print("plot printouts -> %s" % savedir, flush=True)

    print("loading %d ..." % shot, flush=True)
    base = load(shot)
    out, families = {}, {}
    for name, s, q in iter_scaled(base, scalings, scales, **kw):
        label = tag(name, s)
        out[label] = q
        families.setdefault(SCALINGS[name]["apply"], []).append((label, q))
        print("  scaled  %s" % label, flush=True)

    if plot_printouts:
        for family, cases in families.items():
            path = os.path.join(savedir, "%d_%s.png" % (shot, family))
            plot_family(base, cases, path)
            print("  wrote   %s  (%d case(s): %s)"
                  % (path, len(cases), ", ".join(lab for lab, _ in cases)),
                  flush=True)
        print("=== %d PNG(s) in %s ===" % (len(families), savedir), flush=True)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Scale one discharge and check the transforms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=132588, choices=SHOTS)
    ap.add_argument("--scalings", nargs="+", default=None, choices=sorted(SCALINGS),
                    help="default: every scaling in SCALINGS")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.7, 1.3])
    ap.add_argument("--plot-printouts", "--plot-output", dest="plot_printouts",
                    action="store_true",
                    help="write one PNG per transform family to a temp dir; "
                         "runs no gfile and no cheaseBS")
    ap.add_argument("--savedir", default=None,
                    help="default: a fresh temp dir on $SCRATCH")
    args = ap.parse_args(argv)

    run(args.shot, args.scalings, tuple(args.scales),
        plot_printouts=args.plot_printouts, savedir=args.savedir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
