#!/usr/bin/env python3
"""Shared machinery for the DIII-D cheaseBS scan runners.

Three things every runner in this directory needs and none of them should own: a
logger that survives detachment, the definition of what a finished solve leaves
behind, and the acceptance/plot/copy-back steps that turn a scratch run
directory into a permanent record.

They live here so the runners differ only in how they produce the profiles they
hand cheaseBS -- which is the whole experiment.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil

# What travels from a scratch run directory into the permanent one. The final
# EQDSK and the profile copies are handled separately in copy_back; these are
# the files cheaseBS and the scorer write under their own names.
COPY_BACK = (
    "cheasebs_run_config.json",
    "convergence_summary.json",
    "cheasebs_acceptance.json",
    "iteration_log.csv",
    "iteration_errors.png",
    "run_summary.png",
)

_SPECIES = (("e", "electron"), ("i", "deuterium"), ("z", "carbon"))


class Tee:
    """stdout to the terminal and the log file at once, line-flushed.

    Everything downstream prints -- cheaseBS's own per-iteration echo included --
    lands in the log without those modules knowing about it. Flushing on every
    write is what makes `tail -f` useful on a run this slow.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.fh.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def shot_of(gfile):
    """'g162940.02944_670' -> '162940'. Used only to name the output root."""
    m = re.match(r"^g(\d+)\.", os.path.basename(gfile))
    return m.group(1) if m else "unknown"


def fmt_alpha(v):
    """0.7 -> '0p7', 1.0 -> '1p0', 1.05 -> '1p05'. The scan's own tag spelling."""
    s = f"{v:.1f}" if abs(v - round(v, 1)) < 1e-9 else f"{v:.2f}".rstrip("0")
    return s.replace(".", "p")


def last_eqdsk(run_dir):
    """The final reconstruction, from the run's own iteration log."""
    log = os.path.join(run_dir, "iteration_log.csv")
    if not os.path.isfile(log):
        return None
    found = None
    with open(log, newline="") as fh:
        for row in csv.DictReader(fh):
            path = (row.get("eqdsk_path") or "").strip()
            if path:
                found = path
    return found


def read_summary(run_dir):
    path = os.path.join(run_dir, "convergence_summary.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def score(eqdsk, gfile, run_dir, cheasebs_script, radii):
    """Acceptance verdict via TPED, or a note explaining why there isn't one.

    The subprocess exiting 0 and an EQDSK existing is not evidence the
    equilibrium is usable, so this runs whenever TPED is importable. It is a
    diagnostic on top of the solve, never allowed to sink a completed run.
    """
    try:
        from TPED.projects.discharge_tools.src.cheasebs_runner import (
            ACCEPTANCE_FILENAME, CheasebsAcceptance, evaluate_acceptance)
    except ImportError as exc:
        return None, f"TPED not importable ({exc}); acceptance not scored"

    try:
        result = evaluate_acceptance(
            eqdsk_path=eqdsk, source_eqdsk=gfile, run_dir=run_dir,
            cheasebs_script=cheasebs_script,
            policy=CheasebsAcceptance.production(analysis_radii=tuple(radii)),
        )
        rec = result.to_dict()
        with open(os.path.join(run_dir, ACCEPTANCE_FILENAME), "w") as fh:
            json.dump(rec, fh, indent=4, sort_keys=True)
        return rec, None
    except Exception as exc:
        return None, f"acceptance scoring raised: {type(exc).__name__}: {exc}"


def render_plots(run_dir, cfg):
    """The end plots, rendered in the run directory before anything is copied.

    Diagnostics on top of a finished solve: never allowed to sink the run, since
    the equilibrium is already written and scored by the time these are drawn.
    run_summary in particular reads every input path back off disk, which is how
    a reconstruction built from the wrong reference profiles becomes visible.
    """
    notes = []
    try:
        from TPED.projects.discharge_tools.src.cheasebs_runner import (
            plot_iteration_errors, plot_run_summary)
    except ImportError as exc:
        return [f"TPED not importable ({exc}); no plots rendered"]

    for name, fn in (("iteration_errors.png", plot_iteration_errors),
                     ("run_summary.png", plot_run_summary)):
        try:
            fn(run_dir, os.path.join(run_dir, name), cfg=cfg, quiet=True)
        except Exception as exc:
            notes.append(f"{name} failed: {type(exc).__name__}: {exc}")
    return notes


def copy_back(run_dir, final_dir, eqdsk, cfg):
    """Move the result and the record out of scratch into the permanent dir.

    The profiles are copied under their canonical stems rather than left as
    paths, because TPED's resolve_run_files prefers a run directory's own local
    copies over the config's absolute paths -- so a directory carrying them
    re-plots correctly after scratch is purged or the case directory moves.
    """
    os.makedirs(final_dir, exist_ok=True)
    copied = []

    if eqdsk and os.path.isfile(eqdsk):
        dst = os.path.join(final_dir, os.path.basename(eqdsk))
        shutil.copy2(eqdsk, dst)
        copied.append(os.path.basename(dst))

    for fname in COPY_BACK:
        src = os.path.join(run_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, fname))
            copied.append(fname)

    # The scaled profiles this point solved with, and the reference set the
    # baseline was built from. Both are kilobytes of text and they are the only
    # record of which is which once the config's absolute paths go stale.
    for spec, longname in _SPECIES:
        src = cfg.get(f"{longname}_profile")
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, f"profiles_{spec}"))
            copied.append(f"profiles_{spec}")

    ref_dir = os.path.join(final_dir, "reference_profiles")
    for spec, longname in _SPECIES:
        src = cfg.get(f"reference_{longname}_profile")
        if src and os.path.isfile(src):
            os.makedirs(ref_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(ref_dir, f"REF_profiles_{spec}"))
            copied.append(f"reference_profiles/REF_profiles_{spec}")

    return copied


def copy_back_if_needed(row, run_dir, final_dir, eqdsk, cfg):
    """copy_back unless the run already happened in the permanent directory.

    Records what moved on *row*. A failure here is reported loudly but is not
    the same as the point having failed -- the equilibrium exists and is scored.
    """
    if os.path.normcase(os.path.realpath(run_dir)) == os.path.normcase(
            os.path.realpath(final_dir)):
        row["eqdsk_final"] = eqdsk
        return row
    try:
        copied = copy_back(run_dir, final_dir, eqdsk, cfg)
        row["copied"] = copied
        row["eqdsk_final"] = os.path.join(final_dir, os.path.basename(eqdsk))
        print(f"    copied {len(copied)} file(s) -> {final_dir}")
    except Exception as exc:
        row["copy_error"] = f"{type(exc).__name__}: {exc}"
        print(f"    COPY-BACK FAILED: {row['copy_error']}")
    return row


def text_table(rows):
    """The campaign result as plain text -- a log is the only place it is read.

    `t` and `n` are whatever the runner scaled: gradient alphas on one branch,
    mtanh fit-parameter factors on the other. `branch` names which, so the two
    can be read in one table without the numbers being mistaken for each other.
    """
    cols = [("branch", "{}", 10), ("tag", "{}", 16),
            ("t", "{:.2f}", 6), ("n", "{:.2f}", 6),
            ("iters", "{}", 6), ("conv", "{}", 6), ("acc", "{}", 6),
            ("Ip_err", "{:.2%}", 9), ("q@x0", "{:.2%}", 9),
            ("wall_s", "{:.0f}", 8), ("status", "{}", 8)]
    lines = ["  " + "".join(name.rjust(w) for name, _, w in cols)]
    for r in rows:
        q = r.get("q_errors_rel") or {}
        vals = {
            "branch": r.get("branch", "--"), "tag": r["tag"],
            "t": r.get("t", r.get("omt")), "n": r.get("n", r.get("omne")),
            "iters": r.get("iterations"), "conv": r.get("converged"),
            "acc": r.get("accepted"), "Ip_err": r.get("ip_error_rel"),
            "q@x0": max((abs(v) for v in q.values() if v is not None), default=None),
            "wall_s": r.get("wall_s"),
            "status": "FAILED" if "error" in r else "",
        }
        cells = []
        for name, fmt, w in cols:
            v = vals.get(name)
            try:
                s = "--" if v is None else fmt.format(v)
            except (TypeError, ValueError):
                s = str(v)
            cells.append(s.rjust(w))
        lines.append("  " + "".join(cells))
    return "\n".join(lines)
