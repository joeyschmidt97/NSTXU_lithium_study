"""Does an EQDSK's own p' and FF' integrate to the Ip in its header?

Ip = int j_phi dA,  j_phi(R,Z) = R p'(psi) + FF'(psi)/(mu0 R)

If the source EFIT does not reproduce its own stated current, cheaseBS is being
asked to hit a target its inputs cannot produce, and the standing offset is a
property of the input rather than of the solver.
"""
import sys, glob, os
sys.path.insert(0, "C:/Users/joesc/git")
import numpy as np
from matplotlib.path import Path
from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData

MU0 = 4e-7 * np.pi


def ip_from_profiles(path):
    ds = GFileData(path).gfile_to_xarray()
    R = ds["R"].values
    Z = ds["Z"].values
    psi2d = ds["psi_RZ"].values                    # (Z, R)
    psiax, psisep = ds.attrs["psiax"], ds.attrs["psisep"]
    psi1d = ds.coords["psi"].values
    pprime = ds["pprime"].values
    ffprime = ds["ffprime"].values

    RR, ZZ = np.meshgrid(R, Z)
    # inside the last closed surface, by the stored boundary polygon when present
    if "RBDRY" in ds and ds["RBDRY"].size > 3:
        poly = Path(np.column_stack([ds["RBDRY"].values, ds["ZBDRY"].values]))
        mask = poly.contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])
                                    ).reshape(RR.shape)
    else:
        psin = (psi2d - psiax) / (psisep - psiax)
        mask = psin <= 1.0

    order = np.argsort(psi1d)
    pp = np.interp(psi2d, psi1d[order], pprime[order])
    ffp = np.interp(psi2d, psi1d[order], ffprime[order])
    jphi = RR * pp + ffp / (MU0 * RR)

    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]
    ip = np.sum(jphi[mask]) * dR * dZ
    return ip, mask.sum(), ds


def header_current(path):
    with open(path) as f:
        lines = [next(f) for _ in range(5)]
    # EQDSK line 4, first field
    s = lines[3]
    return float(s[0:16])


print(f"{'file':<34} {'header Ip':>13} {'integral Ip':>13} {'rel diff':>10} {'cells':>7}")
print("-" * 82)
srcs = []
for shot in (129015, 129038, 132543, 132588):
    g = glob.glob(f"C:/Users/joesc/git/ST_research/NSTXU_discharges/{shot}/g*")[0]
    ip, n, ds = ip_from_profiles(g)
    hdr = header_current(g)
    srcs.append((shot, hdr, ip))
    print(f"{os.path.basename(g):<34} {hdr:13,.0f} {ip:13,.0f} "
          f"{(ip - hdr) / hdr:9.2%} {n:7d}")

print()
print("--- canary 132588 reconstructions (identity) ---")
base = "C:/Users/joesc/git/NSTXU_lithium_study/canary_132588/runs/20260816_20-06-37"
tgt = dict(srcs and [(s, h) for s, h, _ in srcs])[132588]
for d in sorted(os.listdir(base)):
    p = os.path.join(base, d, "EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT")
    if not os.path.isfile(p):
        continue
    ip, n, ds = ip_from_profiles(p)
    print(f"{d:<24} integral Ip = {ip:12,.0f}   vs source header {tgt:,.0f}  "
          f"({(ip - tgt) / tgt:+.2%})   cells {n}")
