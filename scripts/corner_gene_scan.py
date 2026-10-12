"""Linear GENE scans at the 16 corners of the ped_grid 4D box, one job per corner.

The corners are ped_grid.py's: every combination of 0.8 and 1.2 on Te and ne
pedestal height and width (Teh, Tew, neh, new). Each corner is one GENE problem
directory, copied from a template run, whose parameters point geomdir and
iterdb_file at that corner's cheaseBS output and scan

    kymin  0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0   (a scanlist: 8 parallel sims, one job)
    x0     a 129038 baseline Chapman radius, fixed (default Te top; --x0 picks another)

The equilibrium cannot be a scanlist (geomdir and iterdb_file are paths),
hence one job per corner. x0 is the same at every corner: the Chapman radii
are fitted to the baseline, so every variant is sliced at the same radius
(TPED chapman_points). Give X0 more values and it becomes a second scanlist.

Per corner this writes <genedir>/<prefix>_x0_<x0>_<tag>/ with
    parameters   template parameters, corner paths, scanlists, n_parallel_sims,
                 diagdir <diagroot>/x0_<x0>/<tag>; the template's &scan namelist (what
                 scanscript wrote into it last time) is dropped
    submit.cmd   template submit.cmd, regular queue, wall time, node count and
                 --np / srun -n sized to n_parallel_sims * n_procs_sim
    scanscript, gene_perlmutter   the template's symlinks, re-made
and records everything in <genedir>/<prefix>_x0_<x0>_manifest.json.
(The first scan, x0 = 0.6737, predates x0 in the names: its run dirs are
<prefix>_<tag>/ and its manifest <prefix>_manifest.json.)

    python corner_gene_scan.py --dry-run             # print one corner's files, write nothing
    python corner_gene_scan.py --test --submit       # one corner, debug queue, *_test dirs
    python corner_gene_scan.py                       # write the 16 run dirs, no sbatch
    python corner_gene_scan.py --submit              # write and sbatch each
    python corner_gene_scan.py --only Teh0.80_Tew0.80_neh0.80_new0.80 --submit
    python corner_gene_scan.py --x0 0.6737 --submit  # the ne-top radius instead

A corner whose cheaseBS case lacks DONE.json, profiles.iterdb or exactly one
*_CH_COCOS*.geqdsk (the geomfile) is skipped and reported. An existing run dir is left alone
unless --force.

--test writes a single corner (the first that passes the checks, or the
first --only) to <prefix>_test_x0_<x0>_<tag>/ with output under <diag-root>_test/,
on the debug queue for TEST_WALLTIME: quick through the queue, and never
collides with the real runs.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys

from NSTXU_lithium_study.scripts.ped_grid import DONE, design, tag

TEMPLATE = "/global/u1/j/joeschm/simulators/GENE3/NSTX_lithium_129038_V2"
PED_ROOT = "/pscratch/sd/j/joeschm/cheaseBS_runs/ped_grid/129038/129038"
DIAG_ROOT = "/pscratch/sd/j/joeschm/NSXTU_discharges/lithium_study/129038/corners"
PREFIX = "NSTX_129038_corner"

CORNER_GRID = (0.8, 1.2)
#: 1.5-3 added after the x0 = 0.6737 scan peaked at the top of 0.1-1.0.
KYMIN = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
#: 129038 baseline Chapman radii (rho_tor). From chapman_dataset_from_profile
#: on the box scan's REF_profiles_e, as baseline_x0_plots.py draws them.
CHAPMAN = {"ne_top": 0.6737, "Te_top": 0.8447, "steep": 0.8535}
#: Scanned radius; ne top (0.6737) was the first scan, Te top is the second.
X0 = (CHAPMAN["Te_top"],)

#: 1:30 timed out 22 of 80 x0 = 0.6737 points (all ky <= 0.7, t = timelim);
#: converged ones took up to 4800 s.
WALLTIME = "03:00:00"
QOS = "regular"
TEST_WALLTIME = "00:30:00"
TEST_QOS = "debug"
#: GENE stops itself this many seconds before the SLURM wall time, so it
#: writes its checkpoint and closes its files instead of being killed.
TIMELIM_MARGIN = 300

#: cheaseBS reconstruction as TPED names it (discharge_io); GENE reads it as geomfile.
GEQDSK = "*_CH_COCOS*.geqdsk"

LINKS = ("scanscript", "gene_perlmutter")


def corners():
    return design(list(CORNER_GRID), parts=("corners",))


def x0_label(x0=X0) -> str:
    """Output subfolder for the radius scanned, e.g. x0_0.8447 (x0_a-b-c for a scanlist)."""
    return "x0_" + "-".join(f"{v:g}" for v in x0)


def walltime_seconds(hms: str) -> int:
    h, m, s = (int(x) for x in hms.split(":"))
    return 3600 * h + 60 * m + s


def scanlist(values) -> str:
    return ", ".join(f"{v:g}" for v in values)


def set_param(text: str, key: str, value: str) -> str:
    """Replace the whole `key = ...` line (comments included); key must exist once."""
    pat = re.compile(rf"^([ \t]*){re.escape(key)}[ \t]*=.*$", re.M)
    if len(pat.findall(text)) != 1:
        raise ValueError(f"expected one '{key} =' line in the template parameters")
    return pat.sub(lambda m: f"{m.group(1)}{key} = {value}", text)


def drop_namelist(text: str, name: str) -> str:
    return re.sub(rf"^&{name}\b.*?^/[ \t]*\n?", "", text, flags=re.M | re.S | re.I)


def corner_parameters(template: str, case_dir: str, geomfile: str, diagdir: str,
                      n_parallel: int, timelim: int, x0=X0) -> str:
    text = drop_namelist(template, "scan")
    text = set_param(text, "kymin", f"{KYMIN[0]:g}  !scanlist: {scanlist(KYMIN)}")
    text = set_param(text, "x0", f"{x0[0]:g}" + (f"  !scanlist: {scanlist(x0)}"
                                                  if len(x0) > 1 else ""))
    text = set_param(text, "n_parallel_sims", str(n_parallel))
    text = set_param(text, "diagdir", f"'{diagdir}'")
    text = set_param(text, "iterdb_file", f"'{os.path.join(case_dir, 'profiles.iterdb')}'")
    text = set_param(text, "geomdir", f"'{case_dir.rstrip('/')}/'")
    text = set_param(text, "geomfile", f"'{geomfile}'")
    return set_param(text, "timelim", str(timelim))


def read_param(text: str, key: str) -> str:
    m = re.search(rf"^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n]*)", text, re.M)
    if not m:
        raise ValueError(f"no '{key}' in the template parameters")
    return m.group(1).strip().strip("'\"")


def corner_submit(template: str, nodes: int, ntasks: int, walltime: str, qos: str) -> str:
    text = re.sub(r"^#SBATCH -t .*$", f"#SBATCH -t {walltime}", template, flags=re.M)
    text = re.sub(r"^#SBATCH -q .*$", f"#SBATCH -q {qos}", text, flags=re.M)
    text = re.sub(r"^#SBATCH -N .*$", f"#SBATCH -N {nodes}", text, flags=re.M)
    text = re.sub(r"--np \d+", f"--np {ntasks}", text)
    return re.sub(r"srun (.*?)-n \d+", lambda m: f"srun {m.group(1)}-n {ntasks}", text)


def find_geqdsk(case_dir: str):
    """(file name, problem) for the corner's cheaseBS reconstruction.

    The TPED name, <gfile>_CH_COCOS02[_<transform>].geqdsk (discharge_io;
    cheasebs_runner._find_output_eqdsk globs the same pattern), e.g.
    g129038.00110_CH_COCOS02_mtanh_full_Te-sw1.2-sh1.2_ne-sw1.2-sh1.2.geqdsk.
    Exactly one is required: two would mean the case dir holds runs of
    different transforms and the choice is not this script's to make.
    """
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(case_dir, GEQDSK))
                   if os.path.isfile(p))
    if len(found) == 1:
        return found[0], None
    return None, f"{len(found)} files matching {GEQDSK}" + (f": {found}" if found else "")


def check_case(case_dir: str) -> tuple[str | None, list[str]]:
    geomfile, problem = find_geqdsk(case_dir)
    missing = [f for f in (DONE, "profiles.iterdb")
               if not os.path.isfile(os.path.join(case_dir, f))]
    if problem:
        missing.append(problem)
    acc = os.path.join(case_dir, "cheasebs_acceptance.json")
    if os.path.isfile(acc):
        with open(acc) as fh:
            if not json.load(fh).get("accepted", True):
                missing.append("cheasebs_acceptance: not accepted")
    return geomfile, missing


def build(template=TEMPLATE, ped_root=PED_ROOT, diag_root=DIAG_ROOT, prefix=PREFIX,
          only=None, force=False, submit=False, test=False, dry_run=False, x0=X0,
          walltime=None):
    """Write (and with *submit*, sbatch) the corner run dirs.

    walltime SLURM -t (HH:MM:SS); timelim follows it
    x0       radius (rho_tor) or radii to scan; more than one is a scanlist
    test     one corner, debug queue, TEST_WALLTIME, *_test dirs and manifest
    dry_run  print the first eligible corner's parameters and submit.cmd;
             write, create and submit nothing
    """
    walltime, qos = (walltime or TEST_WALLTIME, TEST_QOS) if test else (walltime or WALLTIME, QOS)
    if test:
        prefix, diag_root = f"{prefix}_test", f"{diag_root.rstrip('/')}_test"
    genedir = os.path.dirname(os.path.normpath(template))
    with open(os.path.join(template, "parameters")) as fh:
        par_template = fh.read()
    with open(os.path.join(template, "submit.cmd")) as fh:
        sub_template = fh.read()
    n_procs_sim = int(read_param(par_template, "n_procs_sim"))
    ppn = int(re.search(r"--ntasks-per-node=(\d+)", sub_template).group(1))
    x0 = tuple(x0)
    n_parallel = len(KYMIN) * len(x0)
    ntasks = n_parallel * n_procs_sim
    nodes = math.ceil(ntasks / ppn)
    timelim = walltime_seconds(walltime) - TIMELIM_MARGIN
    prefix = f"{prefix}_{x0_label(x0)}"

    manifest = {"template": template, "ped_root": ped_root, "kymin": KYMIN, "x0": x0,
                "walltime": walltime, "qos": qos, "test": test, "timelim": timelim, "nodes": nodes,
                "ntasks": ntasks, "n_parallel_sims": n_parallel, "runs": {}}
    skipped = []
    for p in corners():
        label = tag(p)
        if only and label not in only:
            continue
        case_dir = os.path.join(ped_root, label)
        geomfile, missing = check_case(case_dir)
        if missing:
            skipped.append((label, missing))
            print(f"--- {label}: SKIP, missing {missing}", flush=True)
            continue
        run_dir = os.path.join(genedir, f"{prefix}_{label}")
        diagdir = os.path.join(diag_root, x0_label(x0), label)
        parameters = corner_parameters(par_template, case_dir, geomfile, diagdir,
                                       n_parallel, timelim, x0)
        submit_cmd = corner_submit(sub_template, nodes, ntasks, walltime, qos)
        if dry_run:
            print(f"--- {label}: would write {run_dir}\n    output to {diagdir}\n")
            print("=== parameters\n" + parameters + "\n=== submit.cmd\n" + submit_cmd)
            print(f"{nodes} nodes x {walltime} on {qos} ({ntasks} ranks, {n_parallel} sims); "
                  "nothing written")
            return manifest, skipped
        if os.path.exists(run_dir) and not force:
            print(f"--- {label}: {run_dir} exists, NOT rewritten or submitted "
                  "(--force to rewrite with these settings)", flush=True)
            skipped.append((label, ["run dir exists"]))
            continue
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(diagdir, exist_ok=True)
        with open(os.path.join(run_dir, "parameters"), "w") as fh:
            fh.write(parameters)
        with open(os.path.join(run_dir, "submit.cmd"), "w") as fh:
            fh.write(submit_cmd)
        os.chmod(os.path.join(run_dir, "submit.cmd"), 0o755)
        for link in LINKS:
            src, dst = os.path.join(template, link), os.path.join(run_dir, link)
            if os.path.lexists(dst):
                os.remove(dst)
            if os.path.islink(src):
                os.symlink(os.readlink(src), dst)
            else:
                shutil.copy2(src, dst)
        record = {**p, "case_dir": case_dir, "run_dir": run_dir, "diagdir": diagdir,
                  "geomfile": geomfile, "job_id": None}
        if submit:
            out = subprocess.run(["sbatch", "submit.cmd"], cwd=run_dir, check=True,
                                 capture_output=True, text=True).stdout
            record["job_id"] = out.strip().split()[-1]
        manifest["runs"][label] = record
        print(f"--- {label}: {run_dir}" + (f"  job {record['job_id']}" if submit else ""),
              flush=True)
        if test:
            break

    path = os.path.join(genedir, f"{prefix}_manifest.json")
    if os.path.isfile(path):                    # keep corners written by earlier calls
        with open(path) as fh:
            manifest["runs"] = {**json.load(fh).get("runs", {}), **manifest["runs"]}
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest['runs'])} corner run(s) in {path}; "
          f"{nodes} nodes x {walltime} on {qos} each ({ntasks} ranks, {n_parallel} sims)")
    if skipped:
        print(f"{len(skipped)} skipped: " + ", ".join(l for l, _ in skipped))
    return manifest, skipped


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--template", default=TEMPLATE, help="GENE problem dir to copy")
    ap.add_argument("--ped-root", default=PED_ROOT, help="ped_grid cheaseBS cases <tag>/")
    ap.add_argument("--diag-root", default=DIAG_ROOT,
                    help="GENE output, one x0_<x0>/<tag>/ per corner")
    ap.add_argument("--x0", type=float, nargs="+", default=list(X0),
                    help=f"radius (rho_tor) to scan; Chapman points {CHAPMAN}")
    ap.add_argument("--prefix", default=PREFIX, help="run dir name prefix, beside the template")
    ap.add_argument("--walltime", default=None,
                    help=f"SLURM wall time HH:MM:SS [{WALLTIME}; {TEST_WALLTIME} with --test]")
    ap.add_argument("--only", nargs="+", default=None, help="corner tags to write")
    ap.add_argument("--force", action="store_true", help="rewrite existing run dirs")
    ap.add_argument("--submit", action="store_true", help="sbatch each run dir")
    ap.add_argument("--test", action="store_true",
                    help=f"one corner, {TEST_QOS} queue, {TEST_WALLTIME}, *_test dirs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print one corner's parameters and submit.cmd, write nothing")
    args = ap.parse_args()
    _, skipped = build(args.template, args.ped_root, args.diag_root, args.prefix,
                       args.only, args.force, args.submit, args.test, args.dry_run, args.x0,
                       args.walltime)
    sys.exit(1 if skipped else 0)
