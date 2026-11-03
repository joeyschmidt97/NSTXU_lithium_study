"""Scaling checks for the NSTX-U projection of 132543 (cheaseBS test U) in GENE.

Test U (cheasebs_132xxx_tests.py) takes 132543 at 700 ms to B0 = 0.9 T:
lambda = 0.9 / 0.439262 = 2.049 on B0 and Ip (Ip/B0 held), density held, and
Te, Ti, Tz -> T_sep + lambda^2 (T - T_sep) with each separatrix value kept.
The reference is test U0: the same recipe at lambda = 1 (raw profiles, same
cheaseBS settings, B0 and Ip untouched), so U/U0 is the scaling and nothing
else. Test K's Te_ped_scale_1.000 (--ref K) is the fallback, but it is the
mtanh unity fit: at the Te top its omte is 1.15 against 0.27 raw, so a K-vs-U
GENE difference there mixes the fit with the scaling.

Both cases go through GENE at the same psi_N: the reference at x0 = 0.808
(132543 Te-top Chapman radius), U at its own rho_tor of that surface (0.8173;
U's q moved, so the same surface has a different rho_tor).

    prep   NERSC, before GENE: copies both cases to PREP_ROOT/<name>/ with the
           profiles_e/i/z rho_tor column recomputed from that case's own geqdsk
           (cheaseBS writes the source EFIT's rho_tor, which U's changed q puts
           0.010 off at x0), writes profiles.iterdb (TPED), and records the x0
           pair and the corner_gene_scan lines in prep.json.
    pre    no GENE output needed (runs anywhere the case dirs are): every
           quantity at x0, the U/ref ratio and its expected scaling.
    post   after GENE: reads GENE's parameters_NNNN per case and checks
           (a) GENE read the profiles and field it was given, (b) its normalized
           beta, coll, debye2 follow from its own Tref, nref, Bref, Lref,
           (c) the U/ref ratios match the scaling, (d) quasineutrality in density
           and density gradient of the species GENE holds.

    python -m NSTXU_lithium_study.scripts.nstxu_scaling_check pre
    python -m NSTXU_lithium_study.scripts.nstxu_scaling_check pre --ref K
    python -m NSTXU_lithium_study.scripts.nstxu_scaling_check prep --runs
    python -m NSTXU_lithium_study.scripts.nstxu_scaling_check post --out scaling_check.json

Expected U/ref ratios, s_B = s_I = lambda, s_T = T_U/T_ref at x0 (below
lambda^2 because of the separatrix clamp), density held:

    B (F_edge, bcentr), Ip  s_B, s_I exactly
    Bref (F/R on axis)      ~s_B; diamagnetic F and the R_axis shift move it
    q, shat                 ~1 (Ip/B0 held); the bootstrap current at lower nu*
                            moves them, which is physics, so WARN not FAIL
    Tref, nref              s_T, 1
    beta = 2 mu0 n T/B^2    s_T / s_B^2      (1 under pure similarity)
    coll ~ L n lnL / T^2    s_lnL / s_T^2    (the nu* drop)
    rhostar ~ sqrt(T)/(B L) sqrt(s_T) / s_B
    debye2 = eps0 B^2/(n m) s_B^2
    omn                     metric ratio (d rho_tor/d psi)_ref/(..)_U, same n(psi)
    omt                     U against clamp(raw), point by point

Exit code 1 if any FAIL. WARN is a number to read, not an error.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import sys

import numpy as np

# --------------------------------------------------------------------------- setup

#: First path that exists wins (NERSC, then the local checkouts).
TESTS_ROOTS = ["/pscratch/sd/j/joeschm/git_repos/cheaseBS_tests/data/cheasebs_132xxx",
               "C:/Users/joesc/git/cheaseBS_tests/data/cheasebs_132xxx"]
RAW_ROOTS = ["/global/homes/j/joeschm/tools/NSTXU_lithium_study/data/discharges/132543",
             os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "discharges", "132543")]
#: cheaseBS case dirs under TESTS_ROOTS, by the name GENE runs get.
CASES = {"U": "U_nstxu_0.9T/132543/T_scale_4.198_sep",
         "U0": "U0_unscaled_replay/132543/T_scale_1.000_sep",
         "K": "K_rhop_tensbnd/132543/Te_ped_scale_1.000"}
REF = "U0"
#: prep's case dirs (GENE geomdir + iterdb) and the GENE output root.
PREP_ROOT = "/pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/132543/nstxu_projection_eq"
DIAG_ROOT = "/pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/132543/nstxu_projection"
#: Files carried from a cheaseBS case dir into its prep copy (beside the geqdsk).
CARRY = ("cheasebs_acceptance.json", "cheasebs_run_config.json", "convergence_summary.json",
         "TPED_transform_log.json")

B0_SOURCE, B0_TARGET = 0.439262, 0.9          # bcentr in g132543.00700; Hatch 2026-10-23
LAMBDA = B0_TARGET / B0_SOURCE
X0 = 0.808                                   # 132543 Te-top Chapman radius (rho_tor)
#: Separatrix temperatures held by clamp_sep (eV): read from test U's
#: TPED_transform_log.json by load_t_sep(); these are its 2026-10-23 values.
T_SEP = {"e": 42.236, "i": 163.102, "z": 163.102}
CHARGE = {"e": -1, "i": 1, "z": 6}
GENE_SPECIES = {"e": "e", "i": "i", "z": "C"}

MU0, EPS0, QE, MP = 4e-7 * math.pi, 8.8541878128e-12, 1.602176634e-19, 1.67262192e-27
MREF = 2.0                                   # GENE mref (deuterium), template &units

#: Relative tolerance on a ratio that should hold exactly; on GENE's own numbers
#: against the profiles; on q/shat before the equilibrium shift is worth reading.
TOL_EXACT, TOL_GENE, TOL_Q = 0.01, 0.02, 0.10
#: Absolute floor on omt/omn comparisons: at the Te top omte ~0.3, where a 129-
#: against a 4000-point grid alone differs by ~0.1.
ATOL_OM = 0.15


def first_existing(paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    raise SystemExit(f"none of these exist: {paths}")


def load_t_sep(root: str | None = None) -> None:
    """Set T_SEP from test U's own transform log (the clamp values it used)."""
    log = os.path.join(case_path("U", root), "TPED_transform_log.json")
    for h in json.load(open(log)).get("transform_history", []):
        if h.get("transform") == "apply_t_scale" and h.get("t_sep"):
            T_SEP.update({"e": h["t_sep"]["Te"], "i": h["t_sep"]["Ti"], "z": h["t_sep"]["Tz"]})
            return
    raise SystemExit(f"{log}: no apply_t_scale t_sep entry")


def case_path(name: str, root: str | None = None) -> str:
    path = os.path.join(root or first_existing(TESTS_ROOTS), CASES[name])
    if not os.path.isdir(path):
        hint = (" -- run cheaseBS test U0 first (python -u -m NSTXU_lithium_study.scripts."
                "cheasebs_132xxx_tests U0), or pass --ref K") if name == "U0" else ""
        raise SystemExit(f"no case dir {path}{hint}")
    return path


# ------------------------------------------------------------------------- readers

def read_geqdsk(path: str) -> dict:
    """Standard G-EQDSK: header, 1D profiles on the uniform psi grid, boundary."""
    with open(path) as fh:
        lines = fh.readlines()
    nw, nh = (int(v) for v in lines[0].split()[-2:])
    tokens = re.findall(r"[+-]?\d*\.\d+(?:[eE][+-]?\d+)?|[+-]?\d+(?:[eE][+-]?\d+)?",
                        "".join(lines[1:]))
    vals = iter(float(t) for t in tokens)
    take = lambda n: np.array([next(vals) for _ in range(n)])   # noqa: E731
    take(5)                                                      # rdim zdim rcentr rleft zmid
    rmaxis, _, simag, sibry, bcentr = take(5)
    current = take(5)[0]
    take(5)
    g = {"rmaxis": rmaxis, "simag": simag, "sibry": sibry, "bcentr": bcentr,
         "current": current}
    g["fpol"], g["pres"], g["ffprim"], g["pprime"] = (take(nw) for _ in range(4))
    take(nw * nh)
    g["qpsi"] = take(nw)
    nbbbs = int(next(vals))
    next(vals)
    bb = take(2 * nbbbs).reshape(-1, 2)
    g["rbbbs"] = bb[:, 0]
    g["psin"] = np.linspace(0.0, 1.0, nw)
    # rho_tor from the toroidal flux Phi = int q dpsi (own trapezoid: TPED's numpy)
    dpsi = np.diff(g["psin"]) * (sibry - simag)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (g["qpsi"][1:] + g["qpsi"][:-1]) * dpsi)])
    g["rhot"] = np.sqrt(phi / phi[-1])
    g["minor_a"] = 0.5 * (g["rbbbs"].max() - g["rbbbs"].min())
    g["R_geo"] = 0.5 * (g["rbbbs"].max() + g["rbbbs"].min())
    return g


def find_geqdsk(case: str) -> str:
    """The case's cheaseBS geqdsk: TPED's *_CH_COCOS*.geqdsk, or CHEASE's own
    EQDSK_COCOS_*.OUT name in older runs (test K)."""
    gfiles = (glob.glob(os.path.join(case, "*_CH_COCOS*.geqdsk"))
              or glob.glob(os.path.join(case, "EQDSK_COCOS_*.OUT")))
    if len(gfiles) != 1:
        raise SystemExit(f"{case}: expected one *_CH_COCOS*.geqdsk or EQDSK_COCOS_*.OUT, "
                         f"found {gfiles}")
    return gfiles[0]


def read_profiles(case: str) -> dict:
    """profiles_e/i/z: rho_tor, rho_pol, T (keV), n (1e19 m^-3)."""
    out = {}
    for s in "eiz":
        a = np.loadtxt(os.path.join(case, f"profiles_{s}"))
        out[s] = {"rhot": a[:, 0], "rhop": a[:, 1], "T": a[:, 2], "n": a[:, 3]}
    return out


def remap(p: dict, g: dict) -> dict:
    """Profiles with rho_tor recomputed from geqdsk *g* at each point's rho_pol.

    rho_pol (sqrt psi_N) is the label the profiles live on (cheaseBS runs
    coordinate rho_pol); the rho_tor column is the source EFIT's and goes stale
    when q changes. GENE reads the iterdb against rho_tor of its own geomfile
    (inferred from the tracer_efit + iterdb route, not checked in GENE's source).
    """
    out = {}
    for s, d in p.items():
        d = dict(d)
        d["rhot_src"] = d["rhot"]
        d["rhot"] = np.interp(d["rhop"] ** 2, g["psin"], g["rhot"])
        out[s] = d
    return out


def read_gene_parameters(path: str) -> dict:
    """{namelist: {key: value}} plus 'species': {name: {..}} from a GENE parameters file."""
    out, species, block, name = {}, {}, None, None
    for raw in open(path):
        line = raw.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("&"):
            name, block = line[1:].strip().lower(), {}
            continue
        if line == "/":
            if name == "species":
                species[str(block.get("name"))] = block
            elif name is not None:
                out.setdefault(name, {}).update(block)
            name = block = None
            continue
        if block is not None and "=" in line:
            k, v = (t.strip() for t in line.split("=", 1))
            v = v.strip("'\"")
            try:
                v = float(v.replace("d", "e").replace("D", "e"))
            except ValueError:
                pass
            block[k.lower()] = v
    out["species"] = species
    return out


# ------------------------------------------------------------------------ physics

def at(x, xs, ys):
    return float(np.interp(x, xs, ys))


def dlog_drho(x, xs, ys):
    """-d ln y / d rho at x, central differences on the given grid."""
    return -at(x, xs, np.gradient(np.log(ys), xs))


def ln_lambda(n19, T_keV):
    """GENE's Coulomb log: 24 - ln(sqrt(n [cm^-3]) / T [eV])."""
    return 24.0 - math.log(math.sqrt(n19 * 1e13) / (T_keV * 1e3))


def gene_norms(nref, Tref, Bref, Lref):
    """GENE's beta, coll, debye2, rhostar from nref (1e19 m^-3), Tref (keV),
    Bref (T), Lref (m), mref = MREF proton masses (standard GENE definitions;
    coll is GENE's 2.3031e-5 Lref nref lnL / Tref^2)."""
    n, T, m = nref * 1e19, Tref * 1e3 * QE, MREF * MP
    return {"beta": 2 * MU0 * n * T / Bref ** 2,
            "coll": 2.3031e-5 * Lref * nref / Tref ** 2 * ln_lambda(nref, Tref),
            "debye2": EPS0 * Bref ** 2 / (n * m),
            "rhostar": math.sqrt(m * T) / (QE * Bref) / Lref}


def matched_x0(x0: float, ref: str, root: str | None = None) -> float:
    """U's rho_tor at the psi_N where *ref* has rho_tor = x0 (4 decimals)."""
    gr = read_geqdsk(find_geqdsk(case_path(ref, root)))
    gu = read_geqdsk(find_geqdsk(case_path("U", root)))
    return round(at(at(x0, gr["rhot"], gr["psin"]), gu["psin"], gu["rhot"]), 4)


def case_scalars(case: str, x0: float) -> dict:
    """Everything the checks need at x0 for one cheaseBS case, profiles remapped
    to the case's own rho_tor (the stale-column size kept as rhot_map_*)."""
    gfile = find_geqdsk(case)
    g = read_geqdsk(gfile)
    p = remap(read_profiles(case), g)
    e = p["e"]
    psin = at(x0, g["rhot"], g["psin"])
    q = at(x0, g["rhot"], g["qpsi"])
    jphi = -(g["R_geo"] * g["pprime"] + g["ffprim"] / (MU0 * g["R_geo"]))
    sc = {"case": case, "x0": x0, "geqdsk": os.path.basename(gfile),
          "B_F_edge": abs(g["fpol"][-1]), "B_axis": abs(g["fpol"][0]) / g["rmaxis"],
          "bcentr": abs(g["bcentr"]), "Ip_MA": abs(g["current"]) / 1e6,
          "R_axis": g["rmaxis"], "minor_a": g["minor_a"],
          "q0": g["qpsi"][0], "q95": at(0.95, g["psin"], g["qpsi"]),
          "q_x0": q, "shat_x0": x0 / q * at(x0, g["rhot"], np.gradient(g["qpsi"], g["rhot"])),
          "psin_x0": psin,
          "drhot_dpsin": at(psin, g["psin"], np.gradient(g["rhot"], g["psin"])),
          "jphi_x0": at(psin, g["psin"], jphi),
          "jphi_ped_max": float(np.max(np.abs(jphi[g["psin"] > 0.8])))}
    for s in "eiz":
        d = p[s]
        sc[f"T{s}"], sc[f"n{s}"] = at(x0, d["rhot"], d["T"]), at(x0, d["rhot"], d["n"])
        sc[f"omt{s}"], sc[f"omn{s}"] = dlog_drho(x0, d["rhot"], d["T"]), dlog_drho(x0, d["rhot"], d["n"])
    sc["qn_dens"] = sum(CHARGE[s] * sc[f"n{s}"] for s in "eiz") / sc["ne"]
    sc["qn_grad"] = (sum(CHARGE[s] * sc[f"n{s}"] * sc[f"omn{s}"] for s in "eiz")
                     / (sc["ne"] * sc["omne"]))
    rad = np.linspace(0.05, 0.99, 200)
    zn = sum(CHARGE[s] * np.interp(rad, p[s]["rhot"], p[s]["n"]) for s in "eiz")
    sc["qn_dens_max"] = float(np.max(np.abs(zn / np.interp(rad, e["rhot"], e["n"]))))
    mask = (e["rhop"] > 0.5) & (e["rhop"] < 0.99)
    shift = e["rhot"] - e["rhot_src"]
    sc["rhot_map_max"] = float(np.max(np.abs(shift[mask])))
    sc["rhot_map_x0"] = at(x0, e["rhot"], shift)
    sc["norms"] = gene_norms(sc["ne"], sc["Te"], sc["B_axis"], sc["minor_a"])
    sc["lnL"] = ln_lambda(sc["ne"], sc["Te"])
    pt = sum(sc[f"n{s}"] * sc[f"T{s}"] for s in "eiz") / (sc["ne"] * sc["Te"])
    dp = sum(sc[f"n{s}"] * sc[f"T{s}"] * (sc[f"omn{s}"] + sc[f"omt{s}"]) for s in "eiz") \
        / (sc["ne"] * sc["Te"])
    sc["beta_tot"], sc["beta_grad"] = sc["norms"]["beta"] * pt, sc["norms"]["beta"] * dp
    return sc


def raw_prediction(x0: float, case: str, lam2: float = LAMBDA ** 2) -> dict:
    """The case at x0 predicted from the raw source: each T through
    T_sep + lam2 (T - T_sep), placed on the case's own rho_tor by rho_pol."""
    g = read_geqdsk(find_geqdsk(case))
    raw = remap(read_profiles(first_existing(RAW_ROOTS)), g)
    out = {}
    for s in "eiz":
        d, Ts = raw[s], T_SEP[s] / 1e3
        T = Ts + lam2 * (d["T"] - Ts)
        out[f"T{s}"], out[f"omt{s}"] = at(x0, d["rhot"], T), dlog_drho(x0, d["rhot"], T)
        out[f"n{s}"], out[f"omn{s}"] = at(x0, d["rhot"], d["n"]), dlog_drho(x0, d["rhot"], d["n"])
    return out


# ------------------------------------------------------------------------- report

class Report:
    def __init__(self, ref: str):
        self.ref, self.rows = ref, []
        self.n = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0}

    def add(self, name, a, b, expected, verdict, note=""):
        ratio = b / a if isinstance(a, (int, float)) and a else float("nan")
        self.rows.append(dict(name=name, ref=a, U=b, ratio=ratio, expected=expected,
                              verdict=verdict, note=note))
        self.n[verdict] += 1

    def judge(self, name, a, b, expected, tol, note="", warn_only=False, atol=0.0):
        """PASS if b is within tol (relative) or atol (absolute) of expected * a."""
        bad = abs(b - expected * a) > max(tol * abs(expected * a), atol)
        self.add(name, a, b, expected, ("WARN" if warn_only else "FAIL") if bad else "PASS",
                 note)

    def flag(self, name, value, ok, note="", warn_only=False):
        self.add(name, None, value, None, "PASS" if ok else ("WARN" if warn_only else "FAIL"),
                 note)

    def info(self, name, a, b, note="", expected=None):
        self.add(name, a, b, expected, "INFO", note)

    def print(self, title):
        f = lambda v: f"{v:12.4g}" if isinstance(v, (int, float)) else f"{'-':>12}"  # noqa
        print(f"\n=== {title} ===")
        print(f"{'quantity':<24}{self.ref:>12}{'U':>12}{'ratio':>9}{'expect':>9}  verdict  note")
        for r in self.rows:
            rat = f"{r['ratio']:9.4f}" if r["ratio"] == r["ratio"] else f"{'':9}"
            exp = f"{r['expected']:9.4g}" if r["expected"] is not None else f"{'':9}"
            print(f"{r['name']:<24}{f(r['ref'])}{f(r['U'])}{rat}{exp}  {r['verdict']:<7}  "
                  f"{r['note']}")
        print("  " + "  ".join(f"{k} {v}" for k, v in self.n.items()))

    def failed(self):
        return self.n["FAIL"] > 0


# -------------------------------------------------------------------------- modes

def pre(x0: float, ref: str, out: str | None) -> int:
    root = first_existing(TESTS_ROOTS)
    x0U = matched_x0(x0, ref, root)
    R = case_scalars(case_path(ref, root), x0)
    U = case_scalars(case_path("U", root), x0U)
    sT, sBa = U["Te"] / R["Te"], U["B_axis"] / R["B_axis"]
    J = R["drhot_dpsin"] / U["drhot_dpsin"]
    pU = raw_prediction(x0U, case_path("U", root))
    pR = raw_prediction(x0, case_path(ref, root), 1.0)
    r = Report(ref)
    print(f"{ref} at x0 = {x0}, U at x0 = {x0U} (each own rho_tor; psi_N "
          f"{R['psin_x0']:.4f} / {U['psin_x0']:.4f}); lambda = {LAMBDA:.4f}, "
          f"lambda^2 = {LAMBDA ** 2:.4f}; metric ratio (drho_tor/dpsi) {ref}/U = {J:.4f}")
    print(f"{ref}: {R['case']}\nU:  {U['case']}")

    # imposed levers
    r.judge("B F_edge (T m)", R["B_F_edge"], U["B_F_edge"], LAMBDA, TOL_EXACT, "s_B")
    r.judge("bcentr (T)", R["bcentr"], U["bcentr"], LAMBDA, TOL_EXACT, "s_B")
    r.judge("Ip (MA)", R["Ip_MA"], U["Ip_MA"], LAMBDA, TOL_EXACT, "s_I")
    r.judge("B axis F/R (T)", R["B_axis"], U["B_axis"], LAMBDA, 0.03,
            "GENE Bref candidate; diamagnetic F + R_axis shift move it", warn_only=True)
    # profiles: each case against its recipe on raw; U against ref
    for s in "eiz":
        r.judge(f"T{s} (keV)", R[f"T{s}"], U[f"T{s}"], pU[f"T{s}"] / R[f"T{s}"], TOL_EXACT,
                f"U vs clamp(raw) {pU[f'T{s}']:.4g} keV; pure similarity x{LAMBDA ** 2:.3f}")
        r.judge(f"n{s} (1e19)", R[f"n{s}"], U[f"n{s}"], 1.0, TOL_EXACT, "density held")
        r.judge(f"omn{s}", R[f"omn{s}"], U[f"omn{s}"], J, 0.03,
                "same n(psi); ratio = metric", warn_only=True, atol=ATOL_OM)
        r.judge(f"omt{s} U vs recipe", pU[f"omt{s}"], U[f"omt{s}"], 1.0, 0.03,
                "U vs clamp(raw), same grid", warn_only=True, atol=ATOL_OM)
        r.judge(f"omt{s} {ref} vs raw", pR[f"omt{s}"], R[f"omt{s}"], 1.0, 0.03,
                f"{ref} is raw only if U0", warn_only=True, atol=ATOL_OM)
        Ts = T_SEP[s] / 1e3
        r.info(f"omt{s}", R[f"omt{s}"], U[f"omt{s}"], "clamp steepening x metric",
               (1 - Ts / U[f"T{s}"]) / (1 - Ts / R[f"T{s}"]) * J)
    # quasineutrality, both cases
    for c, d in ((ref, R), ("U", U)):
        r.flag(f"QN {c} sum Zn/ne", d["qn_dens"], abs(d["qn_dens"]) < 1e-2, "at x0")
        r.flag(f"QN {c} sum Zn omn", d["qn_grad"], abs(d["qn_grad"]) < 1e-2,
               "at x0, / (ne omne)")
        r.flag(f"QN {c} max |sum Zn|", d["qn_dens_max"], d["qn_dens_max"] < 1e-2,
               "rho_tor 0.05-0.99")
    # the stale rho_tor column prep replaces
    for c, d in ((ref, R), ("U", U)):
        r.flag(f"rho_tor stale {c}", d["rhot_map_max"], d["rhot_map_max"] < 5e-3,
               f"max shift on rho_pol 0.5-0.99, {d['rhot_map_x0']:+.4f} at x0; prep remaps",
               warn_only=True)
    # equilibrium response: the bootstrap current at lower nu*
    for key, lab in (("q_x0", "q at x0"), ("shat_x0", "shat at x0"), ("q95", "q95"),
                     ("q0", "q0")):
        r.judge(lab, R[key], U[key], 1.0, TOL_Q, "Ip/B0 held; bootstrap moves it",
                warn_only=True)
    r.info("R_axis (m)", R["R_axis"], U["R_axis"], "lower beta_p, smaller shift")
    r.info("j_phi(R_geo) at x0", R["jphi_x0"], U["jphi_x0"],
           "R p' + FF'/(mu0 R); s_I if the current shape held", LAMBDA)
    r.info("j_phi(R_geo) ped max", R["jphi_ped_max"], U["jphi_ped_max"], "psi_N > 0.8",
           LAMBDA)
    # what GENE should compute (Tref = Te, nref = ne, Bref = F/R axis, Lref = a)
    nR, nU = R["norms"], U["norms"]
    r.info("beta (GENE pred)", nR["beta"], nU["beta"], "s_T/s_B^2, s_B = Bref ratio",
           sT / sBa ** 2)
    r.info("coll (GENE pred)", nR["coll"], nU["coll"], "s_lnL/s_T^2: the nu* drop",
           (U["lnL"] / R["lnL"]) / sT ** 2)
    r.info("rhostar (GENE pred)", nR["rhostar"], nU["rhostar"], "sqrt(s_T)/s_B",
           math.sqrt(sT) / sBa)
    r.info("debye2 (GENE pred)", nR["debye2"], nU["debye2"], "s_B^2", sBa ** 2)
    r.info("beta_tot (all species)", R["beta_tot"], U["beta_tot"], "1 under pure similarity")
    r.info("beta' (all species)", R["beta_grad"], U["beta_grad"], "beta sum nT(omn+omt)/neTe")
    r.info("alpha ~ q^2 beta'", R["beta_grad"] * R["q_x0"] ** 2,
           U["beta_grad"] * U["q_x0"] ** 2, "ballooning drive proxy")
    r.info("lnLambda", R["lnL"], U["lnL"])
    r.print(f"pre-run scaling check: {ref} at {x0}, U at {x0U}")
    print(f"\ns_T(x0) = {sT:.4f} (lambda^2 = {LAMBDA ** 2:.4f}), s_B(Bref) = {sBa:.4f}")
    if out:
        with open(out, "w") as fh:
            json.dump({"x0": {ref: x0, "U": x0U}, "lambda": LAMBDA, ref: R, "U": U,
                       "rows": r.rows}, fh, indent=1, default=float)
        print(f"wrote {out}")
    return 1 if r.failed() else 0


def prep(x0: float, ref: str, dest: str, force: bool, runs: bool = False,
         walltime: str = "03:45:00", diag_root: str = DIAG_ROOT) -> int:
    """Copy ref and U into dest/<name>/ with rho_tor remapped, write their iterdb;
    with *runs*, also write both GENE3_gpu run dirs (corner_gene_scan --case-dir)."""
    from TPED.projects.discharge_tools.src.writers.iterdb_writer import profiles_to_iterdb

    root = first_existing(TESTS_ROOTS)
    x0s = {ref: x0, "U": matched_x0(x0, ref, root)}
    record = {"ref": ref, "lambda": LAMBDA, "cases": {}}
    for name, x in x0s.items():
        src, out = case_path(name, root), os.path.join(dest, name)
        if os.path.isdir(out) and not force:
            raise SystemExit(f"{out} exists; --force to rewrite")
        os.makedirs(out, exist_ok=True)
        gfile = find_geqdsk(src)
        shutil.copy2(gfile, out)
        for f in CARRY:
            if os.path.isfile(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), out)
        g = read_geqdsk(gfile)
        for s in "eiz":
            path = os.path.join(src, f"profiles_{s}")
            header = [l for l in open(path) if l.startswith("#")]
            a = np.loadtxt(path)
            a[:, 0] = np.interp(a[:, 1] ** 2, g["psin"], g["rhot"])
            with open(os.path.join(out, f"profiles_{s}"), "w") as fh:
                fh.writelines(header[:1])
                fh.write(f"# rho_tor recomputed from {os.path.basename(gfile)} at each "
                         f"rho_pol (nstxu_scaling_check prep); source {src}\n")
                np.savetxt(fh, a)
        iterdb = profiles_to_iterdb(out)
        shift = case_scalars(src, x)["rhot_map_max"]
        record["cases"][name] = {"x0": x, "source": src, "case_dir": out,
                                 "geomfile": os.path.basename(gfile), "iterdb": iterdb,
                                 "rhot_shift_max": shift}
        print(f"{name}: {out}  (x0 {x}, geomfile {os.path.basename(gfile)}, "
              f"rho_tor moved up to {shift:.4f})", flush=True)
    # the copies must reproduce the in-memory remap exactly
    for name, c in record["cases"].items():
        a, b = case_scalars(c["case_dir"], c["x0"]), case_scalars(c["source"], c["x0"])
        for k in ("Te", "ne", "omte", "omne", "Ti", "omti"):
            if abs(a[k] - b[k]) > 1e-6 * max(1.0, abs(b[k])):
                raise SystemExit(f"{name}: prep copy {k} {a[k]} != in-memory {b[k]}")
        if a["rhot_map_max"] > 1e-6:
            raise SystemExit(f"{name}: prep copy still has a stale rho_tor column")
    with open(os.path.join(dest, "prep.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"\nwrote {os.path.join(dest, 'prep.json')}")
    if not runs:
        print("GENE run dirs (or rerun prep with --runs):")
        for name, c in record["cases"].items():
            print(f"python -m NSTXU_lithium_study.scripts.corner_gene_scan --shot 132543 --gpu "
                  f"--case-dir {name}={c['case_dir']} --x0 {c['x0']} --walltime {walltime}")
        return 0
    from NSTXU_lithium_study.scripts import corner_gene_scan as cgs

    skipped = []
    for name, c in record["cases"].items():
        _, sk = cgs.build(template=cgs.GPU_TEMPLATE, ped_root=None, diag_root=diag_root,
                          prefix="NSTXU_132543_proj", force=force, x0=(c["x0"],),
                          walltime=walltime, gpu=True, case_dirs={name: c["case_dir"]})
        skipped += sk
    return 1 if skipped else 0


def latest_parameters(diagdir: str) -> str | None:
    """parameters_0000 of the newest scanfiles dir in diagdir (or diagdir itself)."""
    found = sorted(glob.glob(os.path.join(diagdir, "scanfiles*", "parameters_0000")))
    if not found and os.path.isfile(os.path.join(diagdir, "parameters_0000")):
        found = [os.path.join(diagdir, "parameters_0000")]
    return found[-1] if found else None


def gene_scalars(par: dict) -> dict:
    u, g, geo = par.get("units", {}), par.get("general", {}), par.get("geometry", {})
    out = {k: u.get(k) for k in ("bref", "tref", "nref", "lref")}
    out.update({k: g.get(k) for k in ("beta", "coll", "debye2")})
    out.update({k: geo.get(k) for k in ("q0", "shat", "rhostar", "minor_r", "major_r")})
    out["species"] = par["species"]
    return {k: (float(v) if isinstance(v, float) else v) for k, v in out.items()}


def post(x0: float, ref: str, diag_root: str, prep_root: str, out: str | None) -> int:
    pj = os.path.join(prep_root, "prep.json")
    if not os.path.isfile(pj):
        raise SystemExit(f"no {pj}: run prep first")
    cases = json.load(open(pj))["cases"]
    if ref not in cases:
        raise SystemExit(f"prep.json has {sorted(cases)}, not {ref}")
    S = {c: case_scalars(cases[c]["case_dir"], cases[c]["x0"]) for c in (ref, "U")}
    G = {}
    for c in (ref, "U"):
        d = os.path.join(diag_root, f"x0_{cases[c]['x0']:g}", c)
        p = latest_parameters(d)
        if not p:
            raise SystemExit(f"no scanfiles*/parameters_0000 under {d}: has {c} run?")
        print(f"{c}: {p}")
        G[c] = gene_scalars(read_gene_parameters(p))
    r = Report(ref)
    # (a) GENE read what it was given, and (b) its own normalizations are consistent
    for c in (ref, "U"):
        g, sc = G[c], S[c]
        for key, val, lab in (("tref", sc["Te"], "vs Te(x0) of the iterdb"),
                              ("nref", sc["ne"], "vs ne(x0) of the iterdb"),
                              ("q0", sc["q_x0"], "&geometry q0 is q(x0)")):
            if g[key] is None:
                r.flag(f"{c} GENE {key}", float("nan"), False, "missing", warn_only=True)
            else:
                r.judge(f"{c} GENE {key}", val, g[key], 1.0, TOL_GENE, lab)
        for key, val, lab, tol in (("bref", sc["B_axis"], "vs F/R on axis", 0.03),
                                   ("lref", sc["minor_a"], "vs LCFS half-width", 0.05),
                                   ("shat", sc["shat_x0"], "vs finite-difference shat", 0.10)):
            if g[key] is not None:
                r.judge(f"{c} GENE {key}", val, g[key], 1.0, tol, lab, warn_only=True)
        if None not in (g["nref"], g["tref"], g["bref"], g["lref"]):
            n = gene_norms(g["nref"], g["tref"], g["bref"], g["lref"])
            for key in ("beta", "coll", "debye2", "rhostar"):
                if g[key] is not None:
                    r.judge(f"{c} GENE {key}", n[key], g[key], 1.0, TOL_GENE,
                            "vs formula on GENE's own refs", warn_only=key in ("debye2",
                                                                                "rhostar"))
        sp = g["species"]
        zn = sum(s.get("charge", 0) * s.get("dens", 0) for s in sp.values())
        zg = sum(s.get("charge", 0) * s.get("dens", 0) * s.get("omn", 0) for s in sp.values())
        r.flag(f"{c} GENE QN dens", zn, abs(zn) < 1e-2, "sum charge*dens (nref units)")
        r.flag(f"{c} GENE QN grad", zg, abs(zg) < 1e-2 * max(1.0, abs(sp.get("e", {})
                                                                      .get("omn", 1.0))),
               "sum charge*dens*omn")
        for s, gs in GENE_SPECIES.items():
            if gs in sp and sc["ne"]:
                r.judge(f"{c} GENE {gs} temp", sc[f"T{s}"] / sc["Te"], sp[gs].get("temp", 0),
                        1.0, TOL_GENE, "T_s/Tref vs iterdb")
                r.judge(f"{c} GENE {gs} dens", sc[f"n{s}"] / sc["ne"], sp[gs].get("dens", 0),
                        1.0, TOL_GENE, "n_s/nref vs iterdb")
    # (c) U/ref ratios from GENE against the scaling
    gR, gU, sR, sU = G[ref], G["U"], S[ref], S["U"]
    sT = sU["Te"] / sR["Te"]
    if gR["bref"] and gU["bref"]:
        sB = gU["bref"] / gR["bref"]
        r.judge("Bref U/ref", gR["bref"], gU["bref"], LAMBDA, 0.05, "~s_B", warn_only=True)
        for key, exp, lab, warn in (
                ("tref", sT, "s_T", False), ("nref", 1.0, "density held", False),
                ("lref", 1.0, "frozen boundary", False),
                ("beta", sT / sB ** 2, "s_T/s_B^2", False),
                ("coll", (sU["lnL"] / sR["lnL"]) / sT ** 2, "s_lnL/s_T^2", False),
                ("debye2", sB ** 2, "s_B^2", True),
                ("rhostar", math.sqrt(sT) / sB, "sqrt(s_T)/s_B", True),
                ("q0", sU["q_x0"] / sR["q_x0"], "pre-table q(x0) ratio", True),
                ("shat", sU["shat_x0"] / sR["shat_x0"], "pre-table shat ratio", True)):
            if gR.get(key) is not None and gU.get(key) is not None:
                r.judge(f"{key} U/ref", gR[key], gU[key], exp, TOL_GENE, lab, warn_only=warn)
    J = sR["drhot_dpsin"] / sU["drhot_dpsin"]
    for s, gs in GENE_SPECIES.items():
        a, b = gR["species"].get(gs), gU["species"].get(gs)
        if a and b:
            r.judge(f"{gs} omn U/ref", a.get("omn", 0), b.get("omn", 0), J, 0.03,
                    "same n(psi); metric", warn_only=True, atol=ATOL_OM)
            r.info(f"{gs} omt U/ref", a.get("omt", 0), b.get("omt", 0), "clamp steepening")
    r.print(f"post-run GENE check: {ref} at {cases[ref]['x0']}, U at {cases['U']['x0']}")
    if out:
        with open(out, "w") as fh:
            json.dump({"cases": cases, "gene": G, "rows": r.rows}, fh, indent=1, default=str)
        print(f"wrote {out}")
    return 1 if r.failed() else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("mode", choices=("pre", "prep", "post"))
    ap.add_argument("--x0", type=float, default=X0, help="reference radius, rho_tor")
    ap.add_argument("--ref", default=REF, choices=("U0", "K"), help="unscaled reference case")
    ap.add_argument("--prep-root", default=PREP_ROOT, help="prep output: <root>/<name>/")
    ap.add_argument("--diag-root", default=DIAG_ROOT,
                    help="GENE output: <root>/x0_<x0>/<name>/scanfilesNNNN")
    ap.add_argument("--force", action="store_true",
                    help="prep: rewrite existing case dirs (and run dirs with --runs)")
    ap.add_argument("--runs", action="store_true",
                    help="prep: also write both GENE3_gpu run dirs (corner_gene_scan "
                         "--case-dir, GPU template, prefix NSTXU_132543_proj)")
    ap.add_argument("--walltime", default="03:45:00",
                    help="prep --runs: GENE timelim + 300 s. Two dirs on a 4-node 4 h "
                         "allocation run as one round of two 2-node lanes")
    ap.add_argument("--out", default=None, help="pre/post: write the table as JSON")
    a = ap.parse_args()
    load_t_sep()
    if a.mode == "pre":
        sys.exit(pre(a.x0, a.ref, a.out))
    if a.mode == "prep":
        sys.exit(prep(a.x0, a.ref, a.prep_root, a.force, a.runs, a.walltime, a.diag_root))
    sys.exit(post(a.x0, a.ref, a.diag_root, a.prep_root, a.out))
