"""Per-run cache of what the corner plots read from GENE output, so update_results
rereads only the runs whose files changed.

Cached, per scan member (parameters_NNNN / field_NNNN):

    read_run       growth rate, status, field fit, locals      parameters, omega
    classify       TPED mode_identifier (mode ID and inputs)   every <name>_NNNN
    outlier        TPED outlier analysis                       every <name>_NNNN
    mode_profile   |phi|, |Apar| along z (mode structure)      parameters, field
    vspace         vsp edge ratios                             parameters, vsp
    field_fit      field_omega's TPED field-trace fit          parameters, field

The right column is the files each value is read from: one pickle per scanfiles dir,
<dir>/.plot_cache.pkl, holds {(kind, VERSION, NNNN, extra): (signature, value)}, the
signature being those files' size and sha1 (under HASH_MAX: parameters, omega, which
field_omega rewrites with the same text every update) or size and mtime (field, nrg,
vsp, ...). A changed or new file, different extra (the options the value depends on)
or a bumped VERSION recomputes the value; FORCE (--force-reload) recomputes every value
and overwrites the stored one. ENABLED = False reads everything as before, storing
nothing. Values are written every FLUSH_EVERY new ones and when the process exits;
a dir that cannot be written is reported once and still read.
"""
from __future__ import annotations

import atexit
import copy
import hashlib
import os
import pickle
import re

NAME = ".plot_cache.pkl"
HASH_MAX = 1 << 20          # bytes; larger files are matched by size and mtime
FLUSH_EVERY = 50            # new values between writes
# bump one when what it computes changes: its stored values are then recomputed
VERSION = {"read_run": 1, "classify": 1, "outlier": 1, "mode_profile": 1, "vspace": 1,
           "field_fit": 1}
FILES = {"read_run": ("parameters", "omega"), "classify": None, "outlier": None,
         "mode_profile": ("parameters", "field"), "vspace": ("parameters", "vsp"),
         "field_fit": ("parameters", "field")}
RUN_RE = re.compile(r"^[A-Za-z]\w*_(\d{4})$")

FORCE = False
ENABLED = True
counts = {"reused": 0, "read": 0}
_stores: dict[str, dict] = {}
_dirty: set[str] = set()
_unwritable: set[str] = set()


def _store(d: str) -> dict:
    if d not in _stores:
        try:
            with open(os.path.join(d, NAME), "rb") as fh:
                _stores[d] = pickle.load(fh)
        except Exception:                        # none yet, or unreadable: start empty
            _stores[d] = {}
    return _stores[d]


def signature(d: str, n: str, names=None) -> tuple:
    """(file, size, sha1 or mtime) of every <name>_<n> in d (only *names* if given)."""
    sig = []
    for f in sorted(os.listdir(d)):
        name, _, num = f.rpartition("_")
        if num != n or (names is not None and name not in names):
            continue
        st = os.stat(os.path.join(d, f))
        if st.st_size <= HASH_MAX:
            with open(os.path.join(d, f), "rb") as fh:
                sig.append((f, st.st_size, hashlib.sha1(fh.read()).hexdigest()))
        else:
            sig.append((f, st.st_size, st.st_mtime_ns))
    return tuple(sig)


def memo(kind: str, path: str, compute, extra=()):
    """compute() for the run of *path* (any <name>_NNNN file of it), or its stored
    value when the run's FILES[kind] and *extra* are unchanged. An exception in
    compute() is raised and nothing is stored."""
    m = RUN_RE.match(os.path.basename(path))
    if not ENABLED or not m:
        return compute()
    d, n = os.path.dirname(path), m.group(1)
    key = (kind, VERSION[kind], n, tuple(extra))
    sig = signature(d, n, FILES[kind])
    store = _store(d)
    if not FORCE and key in store and store[key][0] == sig:
        counts["reused"] += 1
        return copy.deepcopy(store[key][1])
    value = compute()
    counts["read"] += 1
    store[key] = (sig, copy.deepcopy(value))
    _dirty.add(d)
    if counts["read"] % FLUSH_EVERY == 0:          # a killed job keeps most of its reads
        flush()
    return value


def flush():
    """Write every changed store, dropping values of an older VERSION."""
    for d in sorted(_dirty):
        store = {k: v for k, v in _stores[d].items() if VERSION.get(k[0]) == k[1]}
        tmp = os.path.join(d, NAME + ".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, os.path.join(d, NAME))
        except OSError as exc:
            if d not in _unwritable:
                print(f"cache not written in {d}: {exc}")
                _unwritable.add(d)
    _dirty.clear()


def report() -> str:
    return (f"run cache: {counts['reused']} reused, {counts['read']} read"
            + (" (--force-reload)" if FORCE else "") + ("" if ENABLED else " (off)"))


atexit.register(flush)
