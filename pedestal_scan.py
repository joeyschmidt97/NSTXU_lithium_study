"""Single source of truth for the NSTX pedestal scan.

Everything the notebooks share lives here: where the discharges are, how they
are fitted, where each one's pedestal is, which axes the campaign scans, what
the literature bounds are, and how a scale factor is turned into a profile.

Imported by
  mtanh_fit_quality/mtanh_fit_quality.ipynb   -- is the fit trustworthy?
  profile_scaling/profile_scaling.ipynb       -- what do the scaled profiles look like?

Nothing here plots. Notebooks own presentation; this module owns the numbers, so
the two notebooks cannot drift into disagreeing about them.

Usage
-----
    from pedestal_scan import Campaign, AXES, SCALE_SANITY
    camp = Campaign()               # loads and fits all four discharges
    d = camp[132543]
    q = d.scaled("Te_ped_scale", 1.2)      # a DischargePhysics, transformed
    d.metric("Te_ped_scale", 1.2)          # the metric the bounds are quoted in
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import medfilt

from TPED.projects.discharge_tools.src.discharge_data import DischargeData
from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics
from TPED.projects.discharge_tools.src.transforms.mtanh_transforms import fit_mtanh_full

__all__ = [
    "DISCHARGE_ROOT_CANDIDATES", "DISCHARGES", "VARS", "PLOT_VARS", "COL",
    "FIT_KWARGS", "TRUST", "ANALYSIS_RADII", "BPOS_BOUND",
    "GRAD_FRAC", "GRAD_MEDFILT", "GRAD_SEARCH", "WINDOW_CAP", "WINDOW_MIN_HI",
    "SCALE_SANITY", "CORE_RHO", "CORE_FLOOR", "CORE_FRAC",
    "BOYLE_WIDTHS", "TARGETS", "REGIME", "SCAN_TI_TE", "AXES",
    "discharge_root", "Discharge", "Campaign",
    "window_stats", "scale_for",
]

# ---------------------------------------------------------------------------
# Where the data is
# ---------------------------------------------------------------------------

# First one that exists wins, so the same code runs on NERSC and on a laptop.
# Add a path rather than replacing one.
DISCHARGE_ROOT_CANDIDATES = [
    r"/global/homes/j/joeschm/data/ST_research/NSTXU_discharges",   # NERSC
    r"C:/Users/joesc/git/ST_research/NSTXU_discharges",             # local
]

# 129038's directory holds five pfiles, so auto-discovery refuses to guess.
DISCHARGES = {129015: {}, 129038: {"pfile": "p129038.00400"},
              132543: {}, 132588: {}}

# Where GENE is actually run. 132543/132588's radii are the q=4 and q=5
# surfaces and sit inside the pedestal top, so the verdict window has to reach
# them even when the pedestal proper starts further out.
ANALYSIS_RADII = {129015: (0.85,), 129038: (0.85,),
                  132543: (0.736, 0.825), 132588: (0.736, 0.825)}


def discharge_root() -> str:
    for d in DISCHARGE_ROOT_CANDIDATES:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(
        "no discharge root found; add this machine's path to "
        f"DISCHARGE_ROOT_CANDIDATES (tried {DISCHARGE_ROOT_CANDIDATES})")


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# Profiles the axes act on, plus the ones that follow through quasineutrality:
# a fit fine for ne and wrong for ni still reaches CHEASE.
VARS = ["Te", "Ti", "ne", "ni"]
PLOT_VARS = ["Te", "Ti", "ne", "ni", "pe"]
COL = {"Te": "tab:red", "Ti": "tab:orange", "ne": "tab:blue",
       "ni": "tab:cyan", "pe": "tab:purple"}

# One campaign setting, everywhere. Per-profile tuning (seeded p0, freed b_pos
# bounds, swept weight) was tried on 2026-08-22 and reverted: it won on
# in-window rms and distorted the profile outside the pedestal, which is what
# CHEASE-BS reshapes on.
FIT_KWARGS = dict(pedestal_weight=8.0)
TRUST = 0.01              # rms <= 1% of profile range, inside the window

# fit_mtanh_full bounds b_pos to [0.85, 0.999]. A fit landing exactly on 0.85
# has hit the bound, not found the pedestal -- reported, never silently used.
BPOS_BOUND = 0.85

# Pedestal-region detection, on |grad pe| from the DATA (the fit is the thing
# under test, so it does not get to choose the window that judges it). Quarter-
# maximum rather than half: a half-max cut on the raw gradient put 132588's
# inner edge at 0.625 while its pe is visibly falling from ~0.55, because one
# spiky point at 0.740 set the threshold. The median filter kills the spike and
# the looser fraction errs wide, which is the safe direction.
GRAD_FRAC = 0.25
GRAD_MEDFILT = 5
GRAD_SEARCH = (0.50, 0.999)     # 132543's core gradient dominates below this
WINDOW_CAP = 0.99               # never score past here: pfile SOL is unreliable
WINDOW_MIN_HI = 0.95            # always score at least out to here


# ---------------------------------------------------------------------------
# Scan axes and literature bounds
# ---------------------------------------------------------------------------

# +/-30% on the scaled profile. Two independent reasons that agree: the survey
# spec's concrete reference is Hatch's +/-10-30% grid around the experimental
# pre-ELM state, and nothing beyond ~1.4 has ever been through cheaseBS.
SCALE_SANITY = (0.7, 1.3)

# Core drift. These axes are meant to move the PEDESTAL, but Stefanikova's core
# Gaussian is anchored to a_height at r=0 only, so scaling b_height or b_width
# drags the core too -- for Te it drags it harder than the pedestal top. Measured
# and reported; enforcing it at any sane threshold empties every Te box.
CORE_RHO = 0.5
CORE_FLOOR = 0.05
CORE_FRAC = 0.30

# Boyle 2011 PPCF, Fig. 7: pedestal FULL widths in % psi_N.
#   7a = Delta_ne, 7d = Delta_Te, 7g = Delta_pe
BOYLE_WIDTHS = {
    "ELMy":     {"dne": (6, 12),  "dTe": (4, 7),  "dpe": (4, 8)},
    "ELM_free": {"dne": (14, 22), "dTe": (7, 10), "dpe": (8, 12)},
}

# Pedestal-top values and the ion/electron ratio; frozen as one box because
# together with the widths they set beta.
TARGETS = {"Te_ped": (0.2, 0.8),    # keV at B0 ~ 0.4 T
           "ne_ped": (3.0, 7.0),    # 1e19 m^-3
           "Ti_Te":  (1.0, 2.0)}

# "auto" classifies each discharge from its own nominal widths: the regime is a
# property of the plasma. Scored against ELMy, 129038 -- ne 18.6%, pe 9.0%,
# squarely ELM-free -- produced width scale factors of 0.20-0.37, an instruction
# to shrink an ELM-free pedestal to a fifth of its width.
REGIME = "auto"           # "auto", "ELMy", "ELM_free", or {shot: regime}

# Te and ne only, height and width each. The KBM drive is the electron pressure
# gradient, which these four axes set between them, and Boyle quotes exactly
# these quantities. Ti is not an axis: scaling ne already rewrites ni and nz
# through quasineutrality, and Ti/Te is a separate physics knob rather than part
# of matching Boyle.
SCAN_TI_TE = False

AXES = {
    "Te_ped_scale":   ("Te", "scale_height", "ped_top", "keV",   1e-3),
    "ne_ped_scale":   ("ne", "scale_height", "ped_top", "1e19",  1e-19),
    "Te_width_scale": ("Te", "scale_width",  "width",   "%psiN", 1.0),
    "ne_width_scale": ("ne", "scale_width",  "width",   "%psiN", 1.0),
}
if SCAN_TI_TE:
    AXES["Ti_Te_scale"] = ("Ti", "scale_height", "ti_te", "-", 1.0)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def window_stats(x, err, win):
    """rms and max |error| inside a window, as a fraction of profile range."""
    e = err[(x >= win[0]) & (x <= win[1])]
    return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))


def scale_for(m0, slope, target):
    """Invert the linear scale->metric map; nan when the axis cannot move it."""
    if not np.isfinite(slope) or abs(slope) < 1e-6:
        return float("nan")
    return 1.0 + ((target / m0) - 1.0) / slope


def _vals(q, var):
    """Profile array for one variable; pe derived, as CHEASE builds it."""
    if var == "pe":
        return (np.asarray(q.ds["ne"].values, dtype=float)
                * np.asarray(q.ds["Te"].values, dtype=float))
    return np.asarray(q.ds[var].values, dtype=float)


# ---------------------------------------------------------------------------
# One discharge: profiles, fits, pedestal, and the scaling it supports
# ---------------------------------------------------------------------------

class Discharge:
    """Nominal profiles, their fits, the measured pedestal, and the transforms.

    Attributes
    ----------
    phys   : DischargePhysics with pe = ne*Te added as a variable
    x      : rho_tor
    ped    : (lo, peak, hi) -- measured pedestal region, from |grad pe|
    win    : (lo, hi) -- the verdict window (pedestal widened to the GENE radii)
    fits   : {var: StefanikovaProfile} for the variables the axes act on
    var    : {var: dict} per-profile fit record (yhat, err, params, pinned, ...)
    """

    def __init__(self, shot: int):
        self.shot = shot
        self.phys = self._load()
        self.x = np.asarray(self.phys.rhot.values)
        self.ped = self._ped_region()
        self.win = (min([self.ped[0]] + list(ANALYSIS_RADII[shot])),
                    min(max(self.ped[2], WINDOW_MIN_HI), WINDOW_CAP))
        self.var = {v: self._fit(v) for v in VARS + ["pe"]}
        self.fits = {v: fit_mtanh_full(self.phys.ds, v, **FIT_KWARGS)[0]
                     for v in {AXES[a][0] for a in AXES}}
        self._ped_top_r: dict[str, float] = {}

    # -- construction ------------------------------------------------------

    def _load(self):
        d = os.path.join(discharge_root(), str(self.shot))
        kw = {"input_dir": d}
        if DISCHARGES[self.shot].get("pfile"):
            kw["pfile"] = os.path.join(d, DISCHARGES[self.shot]["pfile"])
        phys = DischargePhysics(DischargeData(**kw))
        ds = phys.ds.copy()
        ds["pe"] = phys.ds["ne"] * phys.ds["Te"]
        return DischargePhysics(ds)

    def _ped_region(self, var="pe"):
        """Quarter-maximum band of |grad var| -- the pedestal, from the data.

        Median-filtered first so one noisy point cannot set the maximum the
        threshold is measured against, and contiguous around the peak so a
        second steep feature further in does not annex the window.
        """
        y = _vals(self.phys, var)
        g = medfilt(np.abs(y / np.asarray(self.phys.gradient_length(var).values)),
                    GRAD_MEDFILT)
        m = (self.x >= GRAD_SEARCH[0]) & (self.x <= GRAD_SEARCH[1])
        xs, gs = self.x[m], g[m]
        i = int(np.nanargmax(gs))
        lo, hi = i, i
        while lo > 0 and gs[lo - 1] >= GRAD_FRAC * gs[i]:
            lo -= 1
        while hi < len(gs) - 1 and gs[hi + 1] >= GRAD_FRAC * gs[i]:
            hi += 1
        return float(xs[lo]), float(xs[i]), float(xs[hi])

    def _fit(self, var, **kw):
        """Fit one profile and return the record the notebooks read."""
        y = _vals(self.phys, var)
        rec = {"y": y, "scale": float(np.max(y) - np.min(y)) or 1.0}
        try:
            profile, meta = fit_mtanh_full(self.phys.ds, var,
                                           **{**FIT_KWARGS, **kw})
        except Exception as exc:                      # a failure is a result
            rec["error"] = f"{type(exc).__name__}: {exc}"
            return rec
        yhat = np.asarray(profile(self.x), dtype=float)
        fp = meta["fit_params"]
        rec.update(yhat=yhat, err=(yhat - y) / rec["scale"],
                   rms_global=meta["rms_relative"], params=fp,
                   pinned=abs(fp["b_pos"] - BPOS_BOUND) < 1e-6)
        return rec

    # -- geometry ----------------------------------------------------------

    def ped_top_radius(self, var):
        """Radius of this variable's fitted pedestal top, b_pos - 2*b_width.

        Per VARIABLE, since ne and Te pedestals sit 0.02-0.05 apart, and held
        fixed at the nominal fit so the metric measures the profile moving
        rather than the measuring point moving with it.
        """
        if var not in self._ped_top_r:
            fp = self.var[var]["params"]
            self._ped_top_r[var] = float(np.clip(
                fp["b_pos"] - 2.0 * fp["b_width"], GRAD_SEARCH[0], 0.98))
        return self._ped_top_r[var]

    def at(self, q, var, radius):
        """Profile value at a radius, off any (possibly transformed) object."""
        y = _vals(q, var)
        return float(y[int(np.argmin(np.abs(self.x - radius)))])

    # -- transforms --------------------------------------------------------

    def scaled(self, axis, s):
        """Apply one axis at one scale factor; returns a DischargePhysics.

        Density scaling rewrites ni and nz through quasineutrality inside
        DischargePhysics, which is why the ion channel needs no axis of its own.
        """
        var, kwarg = AXES[axis][0], AXES[axis][1]
        return self.phys.apply_mtanh_full(
            var, fit=self.fits[var], **{kwarg: s},
            enforce_quasineutrality=True, qz=6.0)

    def width_psin(self, q, var, with_flags=False):
        """Full pedestal width in % psi_N.

        Boyle quotes widths in poloidal flux, the fit works in rho_tor, and the
        conversion is not a constant -- it depends where the pedestal sits.
        Stefanikova's b_width is a QUARTER width (full = 4*b_width) about b_pos.

        with_flags also returns (b_pos, pinned). The width is a REFIT quantity,
        so a scaled profile whose refit lands on the b_pos bound reports a width
        that is an artefact of the bound rather than a measurement -- worth
        knowing before a pe-width number is read as physics.
        """
        ds = q.ds.copy()
        if var == "pe":
            ds["pe"] = q.ds["ne"] * q.ds["Te"]
        fp = fit_mtanh_full(ds, var, **FIT_KWARGS)[1]["fit_params"]
        rhop = np.asarray(q.rhop.values)
        w = 4.0 * fp["b_width"]
        pl, ph = np.interp([fp["b_pos"] - w / 2, fp["b_pos"] + w / 2],
                           self.x, rhop)
        width = 100.0 * (ph ** 2 - pl ** 2)
        if with_flags:
            return width, fp["b_pos"], abs(fp["b_pos"] - BPOS_BOUND) < 1e-6
        return width

    def metric(self, axis, s=1.0, q=None):
        """The physical quantity this axis's bound is quoted in.

        ped_top : profile value at the fitted pedestal top [raw units]
        width   : full pedestal width in %psi_N
        ti_te   : pedestal-top Ti/Te, both read at the Te radius
        """
        var, kwarg, kind, unit, disp = AXES[axis]
        q = self.scaled(axis, s) if q is None else q
        if kind == "ped_top":
            return self.at(q, var, self.ped_top_radius(var))
        if kind == "width":
            return self.width_psin(q, var)
        if kind == "ti_te":
            r = self.ped_top_radius("Te")
            return self.at(q, "Ti", r) / self.at(q, "Te", r)
        raise ValueError(kind)

    def response(self, axis, probe=1.2):
        """(nominal in bound units, relative slope in the scale factor).

        The slope is relative, so unit conversion does not touch it; the
        nominal must be converted or the inversion is meaningless.
        """
        disp = AXES[axis][4]
        m0 = self.metric(axis, 1.0)
        m1 = self.metric(axis, probe)
        return m0 * disp, (m1 - m0) / m0 / (probe - 1.0)

    def reach(self, axis, span=SCALE_SANITY):
        """Metric interval reachable inside span, MEASURED at both edges."""
        disp = AXES[axis][4]
        return tuple(sorted(self.metric(axis, s) * disp for s in span))

    def core_drift(self, axis, s):
        """(core drift, budget), both fractional.

        Drift is the largest fractional change inside CORE_RHO. Budget is
        CORE_FRAC of the change the axis produced at the pedestal top, floored
        at CORE_FLOOR. Over budget means the knob moved the core more than the
        thing it is named after.
        """
        var = AXES[axis][0]
        q = self.scaled(axis, s)
        y0, y = _vals(self.phys, var), _vals(q, var)
        r = self.ped_top_radius(var) if var in ("Te", "ne") else 0.9
        i = int(np.argmin(np.abs(self.x - r)))
        m = self.x <= CORE_RHO
        return (float(np.max(np.abs(y[m] / y0[m] - 1.0))),
                max(CORE_FLOOR, CORE_FRAC * abs(y[i] / y0[i] - 1.0)))

    # -- physicality -------------------------------------------------------

    def regime(self):
        """Boyle band the nominal widths sit in; votes across dTe, dne, dpe."""
        if isinstance(REGIME, dict):
            return REGIME[self.shot], {}, {}
        widths = {"dTe": self.width_psin(self.phys, "Te"),
                  "dne": self.width_psin(self.phys, "ne"),
                  "dpe": self.width_psin(self.phys, "pe")}
        if REGIME != "auto":
            return REGIME, widths, {}
        scores = {}
        for name, bands in BOYLE_WIDTHS.items():
            inside = sum(bands[k][0] <= w <= bands[k][1] for k, w in widths.items())
            dist = sum(0.0 if bands[k][0] <= w <= bands[k][1]
                       else min(abs(w - bands[k][0]), abs(w - bands[k][1]))
                       for k, w in widths.items())
            scores[name] = (inside, -dist)
        return max(scores, key=scores.get), widths, scores

    def physical(self, axis, s):
        """Is the profile at this scale factor a plasma at all?

        No literature constraint here on purpose. The sparse grid explores the
        box itself, so the box's job is to exclude profiles that are not
        physical or not fittable -- not to pre-select the ones Boyle happened to
        observe. Three tests:
          positive everywhere, monotonic outward through the pedestal (a core
          Gaussian pushed out of proportion can raise a bump near rho~0.9), and
          re-fittable, since CHEASE-BS and the writers refit downstream.
        """
        var = AXES[axis][0]
        try:
            q = self.scaled(axis, s)
            y = _vals(q, var)
            if float(np.min(y)) <= 0:
                return False
            m = (self.x >= 0.6) & (self.x <= 1.0)
            if np.any(np.diff(y[m]) > 0.02 * float(np.max(y[m]))):
                return False
            self.width_psin(q, "pe")          # refit must converge
        except Exception:
            return False
        return True

    def physical_span(self, axis, span=SCALE_SANITY, tol=0.01):
        """The sub-interval of span where this discharge stays physical.

        Walks each edge back toward nominal by bisection, so the limit lands
        where the physics stops rather than on a round number.
        """
        out = []
        for edge in span:
            if self.physical(axis, edge):
                out.append(float(edge))
                continue
            lo, hi = 1.0, edge                 # nominal passes by construction
            if not self.physical(axis, lo):
                return None
            while abs(hi - lo) > tol:
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if self.physical(axis, mid) else (lo, mid)
            out.append(round(float(lo), 4))
        return (min(out), max(out))

    def edge_ok(self, axis, s, regime=None, nominal_dpe=None):
        """Is this box edge a plasma worth submitting?

        Physical tests, not an arbitrary ceiling: positive, monotonic outward
        through the pedestal, and a pe width Boyle could have observed.
        """
        var = AXES[axis][0]
        try:
            q = self.scaled(axis, s)
            y = _vals(q, var)
            if float(np.min(y)) <= 0:
                return False
            # A large height scaling can push the core Gaussian and the pedestal
            # plateau out of proportion and raise a bump at rho~0.9 that the
            # width filter cannot see: a bump changes shape, not width.
            m = (self.x >= 0.6) & (self.x <= 1.0)
            if np.any(np.diff(y[m]) > 0.02 * float(np.max(y[m]))):
                return False
            w = self.width_psin(q, "pe")
        except Exception:
            return False
        lo_pe, hi_pe = BOYLE_WIDTHS[regime or self.regime()[0]]["dpe"]
        if lo_pe <= w <= hi_pe:
            return True
        # Some nominals start outside Boyle's band (129038 at 9.0%, 132588 at
        # 10.5%). Demanding the band outright would empty their boxes and say
        # nothing, so the test becomes "do not make it worse".
        if nominal_dpe is None or lo_pe <= nominal_dpe <= hi_pe:
            return False
        return (min(abs(w - lo_pe), abs(w - hi_pe))
                <= min(abs(nominal_dpe - lo_pe), abs(nominal_dpe - hi_pe)) * 1.25)

    def trim_edge(self, axis, s_target, regime=None, nominal_dpe=None,
                  s_from=1.0, tol=0.01):
        """Bisect an edge back toward nominal until it passes edge_ok.

        Bisection rather than a fixed clip, so the box ends where the physics
        stops rather than at a round number. None when even a hair's movement
        off nominal fails.
        """
        kw = dict(regime=regime, nominal_dpe=nominal_dpe)
        if self.edge_ok(axis, s_target, **kw):
            return s_target
        lo, hi = s_from, s_target
        if not self.edge_ok(axis, lo, **kw):
            return None
        while abs(hi - lo) > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.edge_ok(axis, mid, **kw) else (lo, mid)
        return round(lo, 4)


# ---------------------------------------------------------------------------
# All four discharges, and the box they support
# ---------------------------------------------------------------------------

class Campaign:
    """The four baselines, fitted once, plus the emitted scan box.

    Construction does the expensive work (load + fit each discharge); both
    notebooks then read the same objects instead of re-deriving them.
    """

    def __init__(self, shots=None):
        self.shots = sorted(shots or DISCHARGES)
        self.d = {s: Discharge(s) for s in self.shots}
        self.regime, self.widths = {}, {}
        for s in self.shots:
            reg, widths, _ = self.d[s].regime()
            self.regime[s], self.widths[s] = reg, widths

    def __getitem__(self, shot):
        return self.d[shot]

    def __iter__(self):
        return iter(self.shots)

    def target(self, shot, axis):
        """The literature bound this axis is scored against, in bound units."""
        var, _, kind, _, _ = AXES[axis]
        if kind == "width":
            return BOYLE_WIDTHS[self.regime[shot]]["dTe" if var == "Te" else "dne"]
        return TARGETS["Ti_Te" if kind == "ti_te" else f"{var}_ped"]

    def axis_table(self):
        """Per (shot, axis): nominal, slope, target, reachable interval, cover."""
        rows = []
        for shot in self.shots:
            d = self.d[shot]
            for axis in AXES:
                lo, hi = self.target(shot, axis)
                m0, slope = d.response(axis)
                r_lo, r_hi = d.reach(axis)
                overlap = max(0.0, min(r_hi, hi) - max(r_lo, lo))
                rows.append({
                    "shot": shot, "axis": axis, "nominal": m0, "slope": slope,
                    "unit": AXES[axis][3], "target": (lo, hi),
                    "reach": (r_lo, r_hi),
                    "cover": overlap / (hi - lo) if hi > lo else np.nan,
                    "scale": tuple(sorted((scale_for(m0, slope, lo),
                                           scale_for(m0, slope, hi)))),
                    "pinned": bool(d.var[AXES[axis][0]].get("pinned")),
                })
        return rows

    def shared_box(self, span=SCALE_SANITY):
        """ONE box for all four discharges, which is what a shared sampler needs.

        The sparse grid's axes are common to the campaign and it chooses its own
        points, so per-discharge bounds cannot be handed to it. Start from the
        +/-30% ceiling and shrink an axis only where some discharge stops being
        physical there -- the intersection across discharges, not an
        intersection with Boyle. Where the literature bands land is reported
        separately (see Campaign.axis_table) rather than used to cut the box.

        Returns (box, notes) with box = {axis: (lo, hi)}.
        """
        box, notes = {}, []
        for axis in AXES:
            spans = {}
            for shot in self.shots:
                sp = self.d[shot].physical_span(axis, span)
                if sp is None:
                    notes.append(f"{axis}: {shot} is not physical even at "
                                 "nominal — axis dropped")
                    spans = None
                    break
                spans[shot] = sp
            if not spans:
                continue
            lo = max(v[0] for v in spans.values())
            hi = min(v[1] for v in spans.values())
            if hi <= lo:
                notes.append(f"{axis}: no scale factor is physical for all "
                             f"discharges — dropped ({spans})")
                continue
            box[axis] = (round(float(lo), 4), round(float(hi), 4))
            for shot, sp in spans.items():
                if sp != (span[0], span[1]):
                    notes.append(f"{axis}: {shot} limits it to "
                                 f"{sp[0]:.3f}-{sp[1]:.3f}")
        return box, notes

    def emit_box(self):
        """PER-DISCHARGE box targeted at the Boyle bands, plus per-shot notes.

        Kept for the question "what would it take to put this discharge inside
        Boyle's band"; it is NOT what the sparse grid is handed, since that
        needs one box for the campaign (see shared_box).

        {shot: {axis: (lo, hi)}} plus per-shot notes.

        Each edge is clipped to SCALE_SANITY first -- nothing outside it has
        been run against cheaseBS, so trimming from an untested edge would
        bisect through profiles there is no basis to judge -- then trimmed back
        by the physical tests in Discharge.edge_ok.
        """
        rows = {(r["shot"], r["axis"]): r for r in self.axis_table()}
        box, notes = {}, {}
        for shot in self.shots:
            d, reg = self.d[shot], self.regime[shot]
            dpe = self.widths.get(shot, {}).get("dpe") or d.width_psin(d.phys, "pe")
            box[shot], notes[shot] = {}, []
            for axis in AXES:
                r = rows[(shot, axis)]
                sl, sh = r["scale"]
                if not (np.isfinite(sl) and np.isfinite(sh)):
                    notes[shot].append(f"{axis}: axis cannot move its metric — dropped")
                    continue
                sl, sh = max(sl, SCALE_SANITY[0]), min(sh, SCALE_SANITY[1])
                if sh <= sl:
                    notes[shot].append(
                        f"{axis}: Boyle band lies further than {SCALE_SANITY} "
                        f"allows — dropped (reaches {r['reach'][0]:.3g}-"
                        f"{r['reach'][1]:.3g} of {r['target'][0]}-{r['target'][1]} "
                        f"{r['unit']})")
                    continue
                cl = d.trim_edge(axis, sl, reg, dpe)
                ch = d.trim_edge(axis, sh, reg, dpe)
                if cl is None or ch is None or ch <= cl:
                    notes[shot].append(f"{axis}: no usable range — even small "
                                       "moves off nominal leave Boyle's pe band")
                    continue
                if (cl, ch) != (sl, sh):
                    notes[shot].append(
                        f"{axis}: trimmed to physics, reaches "
                        f"{r['nominal'] * (1 + r['slope'] * (cl - 1)):.3g}-"
                        f"{r['nominal'] * (1 + r['slope'] * (ch - 1)):.3g} "
                        f"{r['unit']} of {r['target'][0]}-{r['target'][1]}")
                box[shot][axis] = (round(float(cl), 4), round(float(ch), 4))
        return box, notes
