"""Give finished cheaseBS runs a descriptively named copy of their EQDSK.

Runs made before TPED 0bdca16 saved the reconstruction as cheaseBS's generic
EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT. This walks a results tree and, next to
each such file, writes a copy under the name TPED now gives it:

    <source gfile>_CH_COCOS02[_<transform>].geqdsk
    e.g. g129038.00110_CH_COCOS02_mtanh_full_Te-sw1.2-sh0.8.geqdsk

The name is built by TPED's own _chease_gfile_name from the run's
TPED_transform_log.json, so old and new runs are named identically. The .OUT
file is left in place. A run with no transform log is reported and skipped,
never named from a guess.

Where the run directory is a ped_grid.py tag (Teh0.80_Tew1.20_neh1.00_new1.00),
the transform log is checked against it first and a mismatch is skipped: the
directory name is what the file gets filed under, so the two must agree.

    python -m NSTXU_lithium_study.scripts.rename_cheasebs_eqdsk <root> --dry-run
    python -m NSTXU_lithium_study.scripts.rename_cheasebs_eqdsk <root>
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil

# TPED and this repo import by absolute path: the directories that contain
# TPED/ and NSTXU_lithium_study/ must be on PYTHONPATH (see README).
from TPED.projects.discharge_tools.src.discharge_io import _chease_gfile_name

from NSTXU_lithium_study.scripts.ped_grid import KNOBS

TRANSFORM_LOG = "TPED_transform_log.json"
PREFERRED_OUT = "EQDSK_COCOS_02_POS_SOURCE_SIGNS.OUT"
TAG_RE = re.compile(r"^" + "_".join(r"%s(\d+\.\d+)" % k for k in KNOBS) + r"$")


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _output_eqdsk(run_dir):
    """The cheaseBS .OUT to copy, or None when there is none or it is ambiguous."""
    preferred = os.path.join(run_dir, PREFERRED_OUT)
    if os.path.isfile(preferred):
        return preferred
    outs = sorted(glob.glob(os.path.join(run_dir, "EQDSK*.OUT")))
    return outs[0] if len(outs) == 1 else None


def _source_gfile_name(run_dir):
    """g<shot>.<time> of the source geometry: the run config's eqdsk, else the local copy."""
    cfg = _read_json(os.path.join(run_dir, "cheasebs_run_config.json")) or {}
    if cfg.get("eqdsk"):
        return os.path.basename(cfg["eqdsk"].replace("\\", "/"))
    local = sorted(n for n in os.listdir(run_dir)
                   if re.match(r"^g\d+\.\d+", n) and "_CH_COCOS" not in n)
    return local[0] if local else None


def _tag_mismatch(run_dir, history):
    """Description of where a ped_grid tag and the transform log disagree, or None."""
    m = TAG_RE.match(os.path.basename(os.path.normpath(run_dir)))
    if not m:
        return None
    logged = {}
    for rec in history:
        if rec.get("transform") != "apply_mtanh_full":
            return f"unexpected transform {rec.get('transform')!r} in a ped_grid run"
        logged[rec.get("var")] = rec
    problems = []
    for (knob, (var, key)), value in zip(KNOBS.items(), m.groups()):
        got = float(logged.get(var, {}).get(key, 1.0))
        if not math.isclose(got, float(value), abs_tol=5e-3):
            problems.append(f"{knob}: dir {value}, log {got:g}")
    return "; ".join(problems) or None


def rename_run(run_dir, dry_run=False, overwrite=False):
    """(status, message) for one run directory."""
    src = _output_eqdsk(run_dir)
    if src is None:
        return "skip", "no single EQDSK*.OUT"
    log = _read_json(os.path.join(run_dir, TRANSFORM_LOG))
    if not log or log.get("transform_history") is None:
        return "skip", f"no {TRANSFORM_LOG}; not naming from a guess"
    history = log["transform_history"]
    mismatch = _tag_mismatch(run_dir, history)
    if mismatch:
        return "skip", f"directory tag disagrees with transform log ({mismatch})"
    gfile = _source_gfile_name(run_dir)
    if gfile is None:
        return "skip", "no source gfile name in run config or directory"

    dst = os.path.join(run_dir, _chease_gfile_name(gfile, history))
    if os.path.exists(dst) and not overwrite:
        return "exists", os.path.basename(dst)
    if not dry_run:
        shutil.copy2(src, dst)
    return "copied", f"{os.path.basename(src)} -> {os.path.basename(dst)}"


def run_dirs(root):
    """Every directory under root (root included) holding an EQDSK*.OUT."""
    for here, _dirs, files in os.walk(root):
        if any(f.startswith("EQDSK") and f.endswith(".OUT") for f in files):
            yield here


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="results tree to walk, e.g. <savedir> or <savedir>/129038")
    ap.add_argument("--dry-run", action="store_true", help="print the names, copy nothing")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a named copy that already exists")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    counts = {}
    for d in sorted(run_dirs(root)):
        status, msg = rename_run(d, dry_run=args.dry_run, overwrite=args.overwrite)
        counts[status] = counts.get(status, 0) + 1
        label = "would" if args.dry_run and status == "copied" else status
        print(f"[{label:6s}] {os.path.relpath(d, root)}: {msg}")
    verb = "would copy" if args.dry_run else "copied"
    print(f"\n{verb} {counts.get('copied', 0)}, already named {counts.get('exists', 0)}, "
          f"skipped {counts.get('skip', 0)}")
    return 1 if counts.get("skip") else 0


if __name__ == "__main__":
    raise SystemExit(main())
