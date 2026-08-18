"""Helpers for the 132588 sparse-grid parameter-generation test.

Kept out of the notebook so the notebook stays a sequence of decisions rather
than a wall of plumbing. Nothing here is TPED-general: the EQDSK retagging and
the dummy-QoI advance exist only because this test stops short of GENE, and
neither belongs in the campaign itself.
"""

from __future__ import annotations

import math
import os
import shutil


# ---------------------------------------------------------------- inputs

def check_inputs(inputs: dict) -> list:
    """Print a provided/missing table for the notebook's INPUTS block.

    Returns the list of names that are unset or point at something that does
    not exist. A path typo and an unset value are reported differently: the
    first means NERSC moved, the second means the notebook was not filled in.
    """
    missing = []
    width = max(len(k) for k in inputs)
    for name, value in inputs.items():
        if value is None:
            state, detail = "MISSING", "not set"
            missing.append(name)
        elif not isinstance(value, str):
            state, detail = "ok", repr(value)
        elif os.path.exists(value):
            kind = "dir" if os.path.isdir(value) else "file"
            state, detail = "ok", f"{kind}: {value}"
        else:
            state, detail = "NOT FOUND", value
            missing.append(name)
        print(f"  {name:<{width}}  {state:<9}  {detail}")
    if missing:
        print(f"\n{len(missing)} input(s) need attention: {', '.join(missing)}")
    else:
        print("\nall inputs present")
    return missing


def check_tped():
    """Report which TPED is actually imported, and whether it has the fixes.

    A notebook imports whatever TPED resolves to on sys.path, which need not be
    the clone you just pulled — and a live kernel keeps the module it imported
    first regardless of what the file on disk now says. Both failures look
    identical from the traceback: the bug you already fixed, again.

    Each probe below is a specific source fragment from a specific fix, so a
    missing one names the commit that has not arrived rather than just saying
    "out of date".
    """
    import inspect

    from TPED.projects.GENE_pipelines.src import scan_campaign
    from TPED.projects.GENE_pipelines.src import point_reconstruction

    print(f"scan_campaign      {scan_campaign.__file__}")
    print(f"point_reconstruction {point_reconstruction.__file__}")

    src = inspect.getsource(scan_campaign)
    psrc = inspect.getsource(point_reconstruction)
    probes = [
        ("kymin seeded before add_scan (a9d09ea)",
         'if "kymin" not in handler.to_dict()' in src),
        ("namelists passed explicitly (b2e5190)",
         'namelist="in_out"' in src),
        ("no hand-added quotes on paths (b2e5190)",
         "f\"'{os.path.basename(entry.eqdsk)}'\"" not in src),
        ("parameters file read back after write (b2e5190)",
         "does not carry" in src),
        ("per-point iterdb (8cfdf49)", "iterdb" in src and "iterdb" in psrc),
        ("scan axes kept out of the namelist (8cfdf49)",
         "is_equilibrium_axis" in src),
        ("pilot axes registered", bool(getattr(point_reconstruction,
                                               "MTANH_AXES", None))),
    ]
    missing = [name for name, ok in probes if not ok]
    print()
    for name, ok in probes:
        print(f"  [{'ok' if ok else 'MISSING'}] {name}")
    if missing:
        print(f"\n{len(missing)} fix(es) not present in the imported module.")
        print("Either that clone has not been pulled, or this kernel is still "
              "holding the module it imported first — restart the kernel before "
              "assuming the pull failed.")
    else:
        print("\nimported TPED has every fix this notebook depends on")
    return missing


# ------------------------------------------------------------ seed input

def seed_from_iterdb(iterdb_path: str, gfile_path: str, workdir: str):
    """Build a DischargeData from a seed iterdb plus its EQDSK.

    DischargeData reads pfiles, EQDSKs and GENE profiles files — there is no
    iterdb reader wired into discharge_tools (FILE_TYPES has three entries and
    iterdb is not one; TPED/projects/utils/read_iterdb.py is a standalone
    parser). This bridges the two by converting the iterdb to GENE profiles
    files once, up front.

    CAVEAT worth checking before trusting a scan built this way: an iterdb
    carries rho_tor only, so the rho_pol column of the written profiles files is
    filled with rho_tor. Anything downstream that reads rho_pol from the
    profiles gets rho_tor instead. The pfile path (seed_from_dirpath) has both
    and does not have this problem — prefer it when the pfile is available.
    """
    import numpy as np
    from TPED.projects.utils.read_iterdb import read_iterdb
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData

    rhot, prof, units = read_iterdb(iterdb_path)

    # write_iterdb's own key names, so a round trip through TPED's writer works.
    species = [("e", "TE", "NE", "Te", "ne"),
               ("i", "TI", "NM1", "Ti", "ni"),
               ("z", "TI", "NM2", "Tz", "nz")]

    profdir = os.path.join(workdir, "seed_profiles")
    os.makedirs(profdir, exist_ok=True)
    written = []
    for spec, tkey, nkey, tlabel, nlabel in species:
        if tkey not in prof or nkey not in prof:
            print(f"  [seed] no {nkey} in iterdb — skipping species '{spec}'")
            continue
        x = np.asarray(rhot[nkey], dtype=float)
        T = np.interp(x, np.asarray(rhot[tkey], dtype=float),
                      np.asarray(prof[tkey], dtype=float))
        n = np.asarray(prof[nkey], dtype=float)
        path = os.path.join(profdir, f"profiles_{spec}")
        with open(path, "w") as f:
            f.write(f"# 1.rho_tor 2.rho_pol 3.{tlabel}(keV) "
                    f"4.{nlabel}(10^19m^-3)\n#\n")
            np.savetxt(f, np.column_stack((x, x, T * 1e-3, n * 1e-19)))
        written.append(path)
        print(f"  [seed] wrote {path}  ({len(x)} points, "
              f"{tlabel} in {units.get(tkey)}, {nlabel} in {units.get(nkey)})")

    return DischargeData(gfile=gfile_path, profiles=written)


def seed_from_dirpath(dirpath: str):
    """Build a DischargeData by auto-discovery, as the CHEASE-BS canary does."""
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData

    return DischargeData(input_dir=dirpath)


# ------------------------------------------------------------- provenance

def tag_from_point(point: dict, axis_short: dict = None) -> str:
    """Compact filename tag encoding a point's transforms, e.g. Te1.10-ne0.90.

    Sorted by axis name so the same point always produces the same tag, and
    short enough to sit inside an EQDSK filename without pushing it past what
    GENE's geomfile field comfortably holds.
    """
    axis_short = axis_short or {}
    parts = []
    for name in sorted(point):
        label = axis_short.get(name, name)
        parts.append(f"{label}{float(point[name]):.3f}")
    return "-".join(parts)


def retag_eqdsks(campaign, batch: int, prefix: str, axis_short: dict = None):
    """Rename each accepted point's EQDSK to carry its transform tag.

    The campaign names directories by point_id (b0000p0000) because tell()
    takes values positionally and the ledger's ordering depends on that name
    sorting lexicographically — so the point_id cannot carry physics. The EQDSK
    filename can, and it is what ends up in the parameters file's geomfile
    field, which is where you actually want to read the transform back off a run
    that has been copied somewhere else.

    Must run after propose() and before submit_pending(), since geomfile is set
    from entry.eqdsk when the parameters file is written.
    """
    renamed = {}
    for entry in campaign.ledger.batch(batch):
        if not entry.eqdsk or not os.path.exists(entry.eqdsk):
            continue
        tag = tag_from_point(entry.point, axis_short)
        new = os.path.join(os.path.dirname(entry.eqdsk), f"{prefix}_{tag}")
        if new != entry.eqdsk:
            shutil.move(entry.eqdsk, new)
            campaign.ledger.update(entry.point_id, eqdsk=new)
        renamed[entry.point_id] = new
    return renamed


# ------------------------------------------------------------- dry submit

def write_parameters_only(campaign):
    """Submitter that writes the parameters file and stops short of sbatch.

    Returns (rundir, job_id) like the real submitter, with a sentinel job id so
    the ledger records a run directory without claiming a scheduler saw it.
    Going through the injection point rather than calling _write_parameters
    directly keeps the ledger bookkeeping on the production path.
    """
    def _submit(entry, **_kwargs):
        return campaign._write_parameters(entry), "DRYRUN"
    return _submit


def dummy_qoi(point: dict) -> float:
    """A smooth FABRICATED scalar used only to make the grid produce more points.

    Not physics. A constant would leave every sensitivity score at zero and the
    refinement would stop after one step, so this varies with every axis. Any
    session state advanced with it is meaningless as a scan.
    """
    return sum(math.sin(3.0 * v) * (i + 1)
               for i, v in enumerate(point[k] for k in sorted(point)))


# -------------------------------------------------------- profile summary

def convergence_report(campaign):
    """How hard cheaseBS actually worked per point, and how well it landed.

    An equilibrium that stopped at iteration 1 is not a converged equilibrium,
    it is the baseline current profile with the input profiles swapped
    underneath it -- cheaseBS under-relaxes (bootstrap_mix 0.1, istar_mix 0.05),
    so one iteration applies a twentieth of the current update. This table is
    how that gets noticed rather than inferred from a growth rate months later.

    Read the iteration count first. If every point stops at the same number and
    that number is max_iter, the tolerances were never met and the reported
    error is what the solver achieved, not what it promised.
    """
    rows = []
    for entry in sorted(campaign.ledger.entries.values(), key=lambda e: e.point_id):
        a = entry.acceptance or {}
        rows.append({
            "point_id": entry.point_id,
            "iterations": a.get("cheasebs_iterations"),
            "converged": a.get("cheasebs_converged"),
            "ip_error": a.get("ip_error_rel"),
            "accepted": a.get("accepted"),
        })
    head = (f"{'point_id':<12} {'iters':>6} {'solver conv':>12} "
            f"{'Ip err':>10} {'gate':>8}")
    print(head)
    print("-" * len(head))
    for r in rows:
        ip = f"{r['ip_error']:.3%}" if isinstance(r["ip_error"], (int, float)) else "n/a"
        print(f"{r['point_id']:<12} {str(r['iterations']):>6} "
              f"{str(r['converged']):>12} {ip:>10} "
              f"{'accept' if r['accepted'] else 'REJECT':>8}")

    iters = [r["iterations"] for r in rows if isinstance(r["iterations"], int)]
    cap = (campaign.cheasebs_overrides or {}).get("max_iter")
    if iters and cap and all(i >= cap for i in iters):
        print(f"\n!! every point ran to the cap (max_iter={cap}) and none "
              f"converged. The loop may still be moving, or tol_ip_rel may be "
              f"unreachable -- it is 0.002 by default against a reconstruction "
              f"that sits ~1.5% off target, which vetoes `converged` however "
              f"still the bootstrap and q iterations have gone. Check tol_bs / "
              f"tol_q / tol_a in the run config before raising max_iter.")
    if iters and max(iters) <= 1:
        print("\n!! every point stopped at one iteration. With bootstrap_mix 0.1 "
              "and istar_mix 0.05 that leaves ~95% of the BASELINE current "
              "profile in place, so the geometry never responded to the "
              "perturbed profiles. Raise max_iter -- see cheasebs_runner's "
              "module docstring.")
    elif iters:
        print(f"\niterations: min {min(iters)}, max {max(iters)}")
    return rows


def read_profiles(path):
    """Read a GENE profiles_X file written by discharge_tools. -> (rhot, T, n)."""
    import numpy as np

    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 2], data[:, 3]


def summarize_profiles(campaign, axis_short=None):
    """Table of every point's actual profiles, as written for CHEASE-BS.

    Reads each point's profiles_e / profiles_i off disk rather than recomputing
    the transform, so this is evidence about what was fed to CHEASE-BS and not a
    second opinion from the code that produced it.

    The duplicate check at the end is the point of the whole function: two
    points sharing a profile means the transform did not take, and every
    CHEASE-BS run would still have succeeded while the axes scanned nothing.
    """
    import numpy as np

    rows = []
    for entry in sorted(campaign.ledger.entries.values(), key=lambda e: e.point_id):
        pe = os.path.join(entry.savedir or "", "profiles_e")
        pi = os.path.join(entry.savedir or "", "profiles_i")
        if not os.path.exists(pe):
            rows.append({"point_id": entry.point_id, "point": entry.point,
                         "error": "no profiles_e (point rejected?)"})
            continue
        rhot, Te, ne = read_profiles(pe)
        Ti = read_profiles(pi)[1] if os.path.exists(pi) else None
        ped = (rhot >= 0.90) & (rhot <= 0.98)
        rows.append({
            "point_id": entry.point_id,
            "point": entry.point,
            "tag": tag_from_point(entry.point, axis_short),
            "rhot": rhot, "Te": Te, "ne": ne, "Ti": Ti,
            "Te_ped_max": float(np.max(Te[ped])) if ped.any() else float("nan"),
            "ne_ped_max": float(np.max(ne[ped])) if ped.any() else float("nan"),
            "Te_sep": float(Te[-1]), "ne_sep": float(ne[-1]),
            "gradTe_max": float(np.max(np.abs(np.gradient(Te, rhot)[ped])))
                          if ped.any() else float("nan"),
        })

    good = [r for r in rows if "error" not in r]
    ref = good[0] if good else None
    for r in good:
        if ref is None or len(r["Te"]) != len(ref["Te"]):
            r["dTe_vs_ref"] = float("nan")
        else:
            scale = max(float(np.max(np.abs(ref["Te"]))), 1e-30)
            r["dTe_vs_ref"] = float(np.max(np.abs(r["Te"] - ref["Te"])) / scale)

    if ref is not None:
        print(f"difference column is measured against {ref['point_id']} "
              f"({ref['tag']})")
        print()
    header = (f"{'point_id':<12} {'tag':<34} {'Te_ped':>9} {'ne_ped':>9} "
              f"{'|gradTe|':>9} {'dTe vs ref':>11}")
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['point_id']:<12} {r['error']}")
            continue
        print(f"{r['point_id']:<12} {r['tag']:<34} {r['Te_ped_max']:>9.4g} "
              f"{r['ne_ped_max']:>9.4g} {r['gradTe_max']:>9.4g} "
              f"{r['dTe_vs_ref']:>11.4f}")

    duplicates = []
    for i, a in enumerate(good):
        for b in good[i + 1:]:
            if (len(a["Te"]) == len(b["Te"])
                    and np.allclose(a["Te"], b["Te"], rtol=1e-12, atol=0)
                    and np.allclose(a["ne"], b["ne"], rtol=1e-12, atol=0)):
                duplicates.append((a["point_id"], b["point_id"]))
    print()
    if duplicates:
        print(f"!! {len(duplicates)} IDENTICAL profile pair(s) — the transform "
              f"did not take for these points:")
        for a, b in duplicates:
            print(f"     {a} == {b}")
    elif len(good) > 1:
        print(f"all {len(good)} profiles differ from each other")

    return rows


def plot_profiles(rows, xlim=(0.85, 1.0)):
    """Overlay every generated Te and ne profile, one colour per scan point."""
    import matplotlib.pyplot as plt

    good = [r for r in rows if "error" not in r]
    if not good:
        print("nothing to plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in good:
        axes[0].plot(r["rhot"], r["Te"], lw=1.3, label=r["tag"])
        axes[1].plot(r["rhot"], r["ne"], lw=1.3, label=r["tag"])
    for ax, lbl in zip(axes, ["Te (keV)", "ne (10^19 m^-3)"]):
        ax.set_xlim(*xlim)
        ax.set_xlabel("rho_tor")
        ax.set_ylabel(lbl)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("profiles written for CHEASE-BS, one line per scan point")
    plt.tight_layout()
    plt.show()


def point_discharges(campaign, axis_short=None):
    """Rebuild a DischargeData per point from that point's own written outputs.

    Reads the profiles_e/i/z and the retagged EQDSK out of each point's savedir,
    so these objects describe what CHEASE-BS produced rather than replaying the
    transform through the same code that produced it. Returns
    [(label, DischargeData)] in point_id order.
    """
    from TPED.projects.discharge_tools.src.discharge_data import DischargeData

    out = []
    for entry in sorted(campaign.ledger.entries.values(), key=lambda e: e.point_id):
        savedir = entry.savedir or ""
        profiles = [os.path.join(savedir, f"profiles_{s}") for s in ("e", "i", "z")]
        profiles = [p for p in profiles if os.path.exists(p)]
        if not profiles or not entry.eqdsk or not os.path.exists(entry.eqdsk):
            continue
        label = f"{entry.point_id} {tag_from_point(entry.point, axis_short)}"
        out.append((label, DischargeData(gfile=entry.eqdsk, profiles=profiles)))
    return out


def plot_discharge_overlay(campaign, axis_short=None, full=False,
                           vars=("Te", "Ti", "ne", "ni"), **kwargs):
    """Overlay every point using discharge_tools' own plotting.

    full=False -> DischargePhysics.plot_profiles: the harmonized profiles only,
                  one colour family per point.
    full=True  -> DischargePhysics.plot: the report layout, flux-surface geometry
                  from each point's OWN reconstructed EQDSK alongside the
                  profiles. Slower, and the only view that shows whether the
                  equilibria differ rather than just the profiles that made them.

    Pass an axis range as a coordinate keyword, e.g. rhot=[0.85, 1.0].
    """
    from TPED.projects.discharge_tools.src.discharge_physics import DischargePhysics

    pairs = point_discharges(campaign, axis_short)
    if not pairs:
        print("no point has both an EQDSK and profiles on disk")
        return None

    fig = None
    for i, (label, discharge) in enumerate(pairs):
        phys = DischargePhysics(discharge)
        plot = phys.plot if full else phys.plot_profiles
        fig = plot(vars=vars, label=label, fig=fig, discharge_idx=i, **kwargs)
    print(f"overlaid {len(pairs)} point(s)")
    return fig


# --------------------------------------------------------- gfile comparison

_GFILE_VARS = ("F", "p", "pprime", "ffprime", "q")


def _gfile_ds(path):
    from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData

    return GFileData(path).gfile_to_xarray()


def compare_gfiles(campaign, axis_short=None, ref_point_id=None):
    """Quantify how each point's EQDSK differs from the reference EQDSK.

    Flux-surface plots are the wrong instrument here and will mislead. The
    reconstruction preserves the source EFIT boundary by design, and a pedestal
    perturbation is a small fraction of total stored energy, so the separatrix
    cannot move and the interior surfaces move by far less than a contour plot
    can resolve. Identical-looking flux surfaces are the expected result, not
    evidence that nothing happened.

    What must change is the flux functions -- F, p, p', FF', q -- and the 2-D
    psi map by a small amount. This reports the max absolute and relative
    difference of each against the reference, and the normalized flux coordinate
    where that max occurs, which is the part a plot of overlapping curves cannot
    tell you.
    """
    import numpy as np

    entries = [e for e in sorted(campaign.ledger.entries.values(),
                                 key=lambda e: e.point_id)
               if e.eqdsk and os.path.exists(e.eqdsk)]
    if len(entries) < 2:
        print(f"need at least 2 reconstructed points, have {len(entries)}")
        return []

    ref_entry = next((e for e in entries if e.point_id == ref_point_id), entries[0])
    ref = _gfile_ds(ref_entry.eqdsk)
    print(f"reference: {ref_entry.point_id} "
          f"({tag_from_point(ref_entry.point, axis_short)})")
    # After GFileData.add_flux_coordinates the flux dimension is rho_idx --
    # psi, rho_pol and rho_tor are coordinates on it, not dimensions.
    print(f"boundary points: {ref.sizes.get('bndry', 'n/a')}, "
          f"flux grid: {ref.sizes.get('rho_idx', 'n/a')}")
    print()

    rows = []
    for entry in entries:
        ds = _gfile_ds(entry.eqdsk)
        row = {"point_id": entry.point_id,
               "tag": tag_from_point(entry.point, axis_short),
               "ds": ds}
        for var in _GFILE_VARS:
            if var not in ds or var not in ref:
                continue
            a, b = ds[var].values, ref[var].values
            if a.shape != b.shape:
                row[var] = {"absmax": float("nan"), "relmax": float("nan"),
                            "at": float("nan")}
                continue
            diff = np.abs(a - b)
            scale = max(float(np.max(np.abs(b))), 1e-300)
            i = int(np.argmax(diff))
            # Locate the max in rho_tor, not as a grid fraction: "the max sits
            # at rho_tor 0.96" is the statement that says the change stayed
            # pedestal-local, which is what this table is checking.
            at = (float(ds["rho_tor"].values[i]) if "rho_tor" in ds.coords
                  else float(i / max(len(diff) - 1, 1)))
            row[var] = {"absmax": float(diff[i]),
                        "relmax": float(diff[i] / scale),
                        "at": at}
        if "psi_RZ" in ds and "psi_RZ" in ref and ds["psi_RZ"].shape == ref["psi_RZ"].shape:
            d = np.abs(ds["psi_RZ"].values - ref["psi_RZ"].values)
            scale = max(float(np.max(np.abs(ref["psi_RZ"].values))), 1e-300)
            row["psi_RZ"] = {"absmax": float(np.max(d)),
                             "relmax": float(np.max(d) / scale),
                             "at": float("nan")}
        # The boundary is held fixed by the real-boundary reconstruction. If it
        # ever moves, that is a finding, so it is checked rather than assumed.
        for coord in ("RBDRY", "ZBDRY", "RLIM", "ZLIM"):
            if coord in ds and coord in ref and ds[coord].shape == ref[coord].shape:
                row.setdefault("boundary_max_move", 0.0)
                row["boundary_max_move"] = max(
                    row["boundary_max_move"],
                    float(np.max(np.abs(ds[coord].values - ref[coord].values))))
        rows.append(row)

    cols = [v for v in _GFILE_VARS if v in rows[0]] + (
        ["psi_RZ"] if "psi_RZ" in rows[0] else [])
    head = f"{'point_id':<12} {'tag':<30} " + " ".join(f"{c:>11}" for c in cols)
    print("max relative difference vs reference")
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = " ".join(f"{r[c]['relmax']:>11.3e}" for c in cols)
        print(f"{r['point_id']:<12} {r['tag']:<30} {cells}")

    print("\nwhere the max occurs (rho_tor, 0=axis 1=edge)")
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = " ".join(
            f"{r[c]['at']:>11.3f}" if r[c]["at"] == r[c]["at"] else f"{'n/a':>11}"
            for c in cols)
        print(f"{r['point_id']:<12} {r['tag']:<30} {cells}")

    moves = [r.get("boundary_max_move", 0.0) for r in rows]
    if any(m > 1e-9 for m in moves):
        print(f"\n!! the plasma boundary moved (max {max(moves):.3e} m). The "
              f"real-boundary reconstruction is supposed to hold it fixed.")
    else:
        print("\nplasma boundary identical across all points, as the "
              "real-boundary reconstruction intends")

    identical = [r["point_id"] for r in rows[1:]
                 if all(r[c]["relmax"] == 0.0 for c in cols)]
    if identical:
        print(f"!! EQDSK identical to the reference for: {', '.join(identical)}")
    return rows


def plot_gfile_differences(rows, ref_idx=0):
    """Plot each flux function's DIFFERENCE from the reference, not the overlay.

    Overlaid curves that differ by a fraction of a percent look like one curve.
    The difference is the only view where a sub-percent change is legible.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(rows) < 2:
        print("need at least 2 points")
        return
    ref = rows[ref_idx]["ds"]
    cols = [v for v in _GFILE_VARS if v in ref]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.4))
    axes = np.atleast_1d(axes)
    for r in rows:
        if r is rows[ref_idx]:
            continue
        for ax, var in zip(axes, cols):
            a, b = r["ds"][var].values, ref[var].values
            if a.shape != b.shape:
                continue
            x = (r["ds"]["rho_tor"].values if "rho_tor" in r["ds"].coords
                 else np.linspace(0, 1, len(a)))
            ax.plot(x, a - b, lw=1.3, label=r["tag"])
    for ax, var in zip(axes, cols):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("rho_tor")
        ax.set_title(f"delta {var}")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"EQDSK flux functions minus {rows[ref_idx]['tag']}")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------- verification

def check_template(path: str) -> list:
    """Warn about baseline-parameters settings the campaign will not fix."""
    text = open(path).read()
    notes = []
    if "iterdb_file" not in text:
        notes.append(
            "no iterdb_file line. The campaign sets one per point, but confirm "
            "GENE reads profiles from an iterdb here rather than from in-namelist "
            "omt/omn values — the scan does not touch those, so a template built "
            "that way would run every point on the seed profiles")
    if "scanlist" in text or "scan_dims" in text:
        notes.append(
            "template already carries a scan. The campaign adds a kymin scanlist "
            "on top, so check the resulting scan_dims is what you meant")
    if "x0" not in text:
        notes.append("no x0. This test is single-radius — set it")
    if "magn_geometry" not in text:
        notes.append("no magn_geometry. geomfile is only read for the "
                     "tracer_efit-style geometries")
    return notes


def verify_parameters(campaign, is_equilibrium_axis) -> list:
    """Read every written parameters file back and check it says what it should.

    The three failure modes worth catching automatically, all silent at runtime:
    a scan axis written into the namelist as a GENE key, an iterdb_file still
    pointing at the seed, and a geomfile that is not this point's EQDSK.
    """
    problems = []
    accepted = [e for e in campaign.ledger.entries.values() if e.eqdsk]
    with_rundir = [e for e in accepted if e.rundir]
    if accepted and not with_rundir:
        # Without this the function returns [] and reads as "all clear" while
        # having checked nothing. A vacuous pass is worse than a failure.
        problems.append(("ALL", f"{len(accepted)} point(s) reconstructed but no "
                                "parameters file was written for any of them"))
    for entry in campaign.ledger.entries.values():
        if not entry.rundir:
            continue
        path = os.path.join(entry.rundir, "parameters")
        if not os.path.exists(path):
            problems.append((entry.point_id, f"no parameters file at {path}"))
            continue
        text = open(path).read()
        for key in entry.point:
            if is_equilibrium_axis(key) and f"\n{key}" in text:
                problems.append((entry.point_id,
                                 f"scan axis {key!r} leaked into the namelist"))
        if entry.iterdb and f"'{entry.iterdb}'" not in text:
            problems.append((entry.point_id,
                             "iterdb_file is not this point's iterdb"))
        if entry.eqdsk and f"'{os.path.basename(entry.eqdsk)}'" not in text:
            problems.append((entry.point_id,
                             "geomfile is not this point's EQDSK"))
    return problems
