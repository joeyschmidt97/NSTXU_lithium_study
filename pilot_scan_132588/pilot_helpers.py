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
