"""Bare-bones scaling driver for the final convergence check.

One discharge -> one DischargePhysics -> one transform per scaling.

    from scalings import load, scale, run, SCALINGS

    phys = load(132588)
    q    = scale(phys, "Te_ped_scale", 1.3)     # mtanh_full step-amplitude
    q    = scale(phys, "omt", 0.7)              # gradient power law

    run(132588, plot_printouts=True)            # every scaling, PNG per case

Nothing here runs cheaseBS and nothing here writes a gfile --- call
``q.output_gfile(...)`` on a returned object for that.
"""

from __future__ import annotations

import os
import sys
import tempfile

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

def scratch_dir(shot: int) -> str:
    """A fresh temp directory for check plots, on SCRATCH when there is one."""
    root = os.environ.get("SCRATCH") or os.environ.get("PSCRATCH") or None
    return tempfile.mkdtemp(prefix="scaling_check_%d_" % shot, dir=root)


def plot_case(base, scaled, label, savedir, vars=PLOT_VARS, xlim=PLOT_XLIM,
              xcoord="rhot"):
    """Base vs scaled on one figure; returns the PNG path.

    The whole point is to see the perturbation next to what it was applied to,
    so the base is always drawn --- a scaled profile on its own looks plausible
    at any scale factor, including one that did nothing.
    """
    import matplotlib.pyplot as plt

    kw = {xcoord: list(xlim)} if xlim else {}
    fig = base.plot_profiles(vars=vars, label="base", xcoord=xcoord,
                             discharge_idx=0, **kw)
    fig = scaled.plot_profiles(vars=vars, label=label, xcoord=xcoord, fig=fig,
                               discharge_idx=1, **kw)
    for ax in fig.axes:
        ax.legend(fontsize=7, loc="best")
    path = os.path.join(savedir, label.replace(" ", "_") + ".png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def run(shot: int, scalings=None, scales=(0.7, 1.3), *, plot_printouts=False,
        savedir=None, **kw):
    """Scale one discharge across the grid. Returns {tag: DischargePhysics}.

    plot_printouts writes a base-vs-scaled PNG per case to a temp directory and
    runs nothing else --- no gfile, no cheaseBS --- so the transforms can be
    eyeballed before anything expensive is launched on them.
    """
    if plot_printouts:
        os.environ.setdefault("MPLBACKEND", "Agg")
        savedir = savedir or scratch_dir(shot)
        os.makedirs(savedir, exist_ok=True)
        print("plot printouts -> %s" % savedir)

    base = load(shot)
    out = {}
    for name, s, q in iter_scaled(base, scalings, scales, **kw):
        label = tag(name, s)
        out[label] = q
        if plot_printouts:
            print("  %-24s %s" % (label,
                                  os.path.basename(plot_case(base, q, label,
                                                             savedir))))
        else:
            print("  %s" % label)
    return out
