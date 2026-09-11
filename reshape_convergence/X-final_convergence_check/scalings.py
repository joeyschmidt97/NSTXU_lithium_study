"""Single source of truth for the final convergence check.

The notebook and any batch script import from here, so a scaling run in a
terminal and a scaling plotted in the notebook cannot differ.

    from scalings import load, scale, run, compare

    phys = load(132588)
    hi   = scale(phys, "mtanh_full", te=1.3)          # Hatch handoff transform
    lo   = scale(phys, "omt_omne",   te=0.7, ne=0.7)  # gradient transform
    compare(phys, hi, lo, labels=("base", "mtanh 1.3T", "omt/omne 0.7"))

    run(132588, "mtanh_full", te=1.3, gfile=True, savedir="out/")  # + cheaseBS
"""

from __future__ import annotations

import os
import sys

# Import TPED from this checkout rather than whatever is installed, matching
# NSTX_verify/run_nstx_omtomn_scan.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _parent in (_HERE, os.path.dirname(_HERE)):
    _root = os.path.dirname(os.path.dirname(_parent))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from TPED.projects.discharge_tools.src.discharge_data import DischargeData
from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics
from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import fit_mtanh_full

DISCHARGE_ROOT_CANDIDATES = [
    "/global/homes/j/joeschm/data/ST_research/NSTXU_discharges",
    "C:/Users/joesc/git/ST_research/NSTXU_discharges",
]
PFILES = {129038: "p129038.00400"}
SHOTS = (129015, 129038, 132543, 132588)

# (rhot_topped, rhot_midped) for omt/omne. IFS handoff pair; override per shot.
RAMP_WINDOWS = {s: (0.8, 1.0) for s in SHOTS}

# Campaign fit setting for mtanh_full. One value everywhere.
FIT_KWARGS = dict(pedestal_weight=8.0)

KINDS = ("omt_omne", "mtanh_full")


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


def _fits(phys, vars):
    """{var: StefanikovaProfile}, cached on the object so scales share one fit."""
    cache = phys.__dict__.setdefault("_scaling_fits", {})
    for v in vars:
        if v not in cache:
            cache[v] = fit_mtanh_full(phys.ds, v, **FIT_KWARGS)[0]
    return cache


def scale(phys: DischargePhysics, kind: str = "mtanh_full", *,
          te: float = 1.0, ne: float = 1.0, window=None, qz: float = 6.0,
          apply_to_tz: bool = True, **kw) -> DischargePhysics:
    """Apply one scaling. te/ne are the knobs; 1.0 leaves that channel alone.

    kind="omt_omne"   : gradient power law, re-exponentiated about rhot_midped.
                        te scales Te/Ti/Tz together, ne scales ne (ni, nz follow).
    kind="mtanh_full" : Stefanikova F_full reconstruction with b_height scaled.
                        te scales Te only, ne scales ne (ni, nz follow).
                        This is the transform the Hatch handoff package used.

    window : (rhot_topped, rhot_midped), omt_omne only. Defaults per shot.
    **kw   : forwarded to apply_mtanh_full (scale_width, shift_pos, ...).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    if kind == "omt_omne":
        top, mid = window or RAMP_WINDOWS.get(phys.ds.attrs.get("shot"), (0.8, 1.0))
        if te != 1.0:
            phys = phys.apply_omt(te, rhot_midped=mid, rhot_topped=top,
                                  apply_to_tz=apply_to_tz)
        if ne != 1.0:
            phys = phys.apply_omne(ne, rhot_midped=mid, rhot_topped=top, qz=qz)
        return phys

    fits = _fits(phys, [v for v, s in (("Te", te), ("ne", ne)) if s != 1.0])
    for var, s in (("Te", te), ("ne", ne)):
        if s != 1.0:
            phys = phys.apply_mtanh_full(var, fit=fits[var], scale_height=s,
                                         enforce_quasineutrality=True, qz=qz, **kw)
    return phys


def tag(kind: str, te: float, ne: float) -> str:
    return "%s_te%.3f_ne%.3f" % (kind, te, ne)


def run(shot: int, kind: str = "mtanh_full", *, te: float = 1.0, ne: float = 1.0,
        gfile: bool = False, savedir: str = ".", window=None, qz: float = 6.0,
        scale_kw=None, **gfile_kw) -> DischargePhysics:
    """Load, scale, and optionally write the gfile + run cheaseBS.

    gfile=False returns the scaled DischargePhysics and writes nothing --- the
    notebook path. gfile=True reconstructs the equilibrium and copies the result
    to savedir/<tag> --- the batch-script path. Same scaling either way.

    Extra keywords go to output_gfile (cheasebs_config, cheasebs_overrides, ...).
    """
    phys = scale(load(shot), kind, te=te, ne=ne, window=window, qz=qz,
                 **(scale_kw or {}))
    if gfile:
        out = os.path.join(savedir, str(shot), tag(kind, te, ne))
        os.makedirs(out, exist_ok=True)
        phys.output_gfile(savedir=out, run_cheasebs=True,
                          comment=f"{shot}_{tag(kind, te, ne)}", **gfile_kw)
    return phys


def compare(*phys, labels=None, vars=("Te", "Ti", "ne", "ni"), xcoord="rhot",
            xlim=None, fig=None):
    """Overlay any number of DischargePhysics on one T/n figure.

    xlim=(0.8, 1.0) zooms the pedestal; a pedestal change is a few percent of a
    core-scaled axis and is only legible zoomed.
    """
    labels = labels or [None] * len(phys)
    for i, (p, lab) in enumerate(zip(phys, labels)):
        kw = {xcoord: list(xlim)} if xlim else {}
        fig = p.plot_profiles(vars=vars, label=lab, xcoord=xcoord, fig=fig,
                              discharge_idx=i, **kw)
    return fig
