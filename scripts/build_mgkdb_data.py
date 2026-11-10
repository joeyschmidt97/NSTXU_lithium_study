"""Pull MGKDB linear GENE runs and save one netCDF per run, NSTX-style.

MGKDB keeps, per linear run: nrg, omega, parameters, tracer_efit and the
uploader's quantity_info_NC as Files, and the final-time field and moments as
the Diagnostics 'Field Mom Snapshots' pickle (phi, apar, bpar and the moments
per species on (kx, ky, z)) with a 'Grid' pickle. No field or mom time history.

Per record this script downloads those, writes the snapshot back out as
one-time-step GENE binaries (field<suffix>, mom_<species><suffix>, GENE's own
record layout, precision and endianness from the parameters), and runs the same
TPED registry analyses as build_discharge_data (fingerprints, epar, kperp,
kperp_upar, ql_weight, mode_identifier, field_omega). So the netCDF has the
NSTX variable names and the NSTX mode label code path. Added on top:

- the quantity_info_NC variables with at most NC_MAX_SIZE values, prefixed nc_
  (the uploader's values from the TPED of the upload date; a cross-check, not a
  source). Its arrays (vsp f2 magnitudes, ~2.4 MB a run; kperp, omd_curv) are
  dropped;
- shot and campaign as variables, and build_discharge_data's cost_* (NaN where
  the parameters lack GENE's &info timing), so MGKDB and NSTX files concatenate;
- MGKDB bookkeeping in attrs (mgkdb_id, mgkdb_user, run_collection_name, ...)
  and attrs['snapshot_zero_filled']: moments the snapshot lacked, written as 0.

There is no field time history, so TPED's field_omega (a field fit) is
replaced by omega_from_file: the same variables, filled from the omega file
alone, attached before mode_identifier runs. omega_source:

    'GENE'              GENE's line, non-zero              converged
    'field_legacy'      '# POST-PROCESSED OMEGA_FROM_FIELD' converged (as TPED reads
                        line (field fit done before upload) pre-verdict lines), with std
    'GENE_unconverged'  0/0 (timelim placeholder)          not converged
    'GENE_sentinel'     -1/-1 (Hatch campaigns)            not converged
    'empty'             no data line (a 'kymin 0 0' line   not converged
                        is written so epar can run)

Runs that are not converged get no gamma_best and come out 'unidentified'.
fit_* are NaN, fit_verdict 'no_history'. The MGKDB IMAS eigenmode values
(imas_growth_rate_norm, imas_frequency_norm, imas_growth_rate_tolerance) are
saved alongside, raw: IMAS normalisation and sign, not GENE's.

Two steps, both resumable (files already present are skipped):

    <raw-root>/<run_collection_name>/{parameters,nrg,omega,tracer_efit}<suffix>,
        quantity_info_NC<suffix>.nc, field_mom_snapshot<suffix>.pkl, grid<suffix>.pkl,
        then the written field<suffix>, mom_<species><suffix>
    <out-root>/<campaign>/<scanfilesNNNN>/quantity_info_NC<suffix>.nc

Campaigns are matched on Metadata.DBtag.run_collection_name with an anchored
prefix, so hatch/pscans does not also catch drhatch/pscans2 and pscans3.

Run from the git parent dir (holding NSTXU_lithium_study and TPED) in an env
with both mgkdb and TPED. Raw downloads go to pscratch (re-downloadable, purge
is harmless); netCDFs go into this repo next to the NSTX data, then commit:

    python -m NSTXU_lithium_study.scripts.build_mgkdb_data --dry-run
    python -m NSTXU_lithium_study.scripts.build_mgkdb_data --limit 3
    python -m NSTXU_lithium_study.scripts.build_mgkdb_data
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import struct
import traceback

import numpy as np

from NSTXU_lithium_study.scripts.build_discharge_data import add_cost, netcdf_safe

#: campaign name -> run_collection_name prefix (matched as ^<prefix>/)
CAMPAIGNS = {
    'hatch_pscans': '/global/cfs/cdirs/m2116/hatch/pscans',
    'drhatch_pscans2': '/pscratch/sd/d/drhatch/pscans2',
    'drhatch_pscans3': '/pscratch/sd/d/drhatch/pscans3',
    'lleppin_153764': '/pscratch/sd/l/lleppin/DIIID153764/lin_loc',
    'lleppin_174082': '/pscratch/sd/l/lleppin/DIIID174082/lin_loc',
}
#: DIII-D discharge per campaign (Hatch pscans2/3 are equilibrium variations around 162940)
CAMPAIGN_SHOT = {'hatch_pscans': 162940, 'drhatch_pscans2': 162940, 'drhatch_pscans3': 162940,
                 'lleppin_153764': 153764, 'lleppin_174082': 174082}
#: defaults: raw downloads on pscratch, netCDFs in this repo (run from the git parent dir)
RAW_ROOT = '/pscratch/sd/j/joeschm/MGKDB_raw'
OUT_ROOT = 'NSTXU_lithium_study/data/simulation_data/mgkdb'
#: MGKDB Files keys downloaded per run (gfiles, profiles, autopar skipped)
FILES = ['parameters', 'nrg', 'omega', 'tracer_efit', 'quantity_info_NC']
#: MGKDB Diagnostics keys downloaded per run -> local file stem
DIAGNOSTICS = {'Field Mom Snapshots': 'field_mom_snapshot', 'Grid': 'grid'}
#: build_discharge_data's registry analyses, minus field_omega (replaced by
#: omega_from_file) and mode_identifier (run after it)
ANALYSES = ['fingerprints', 'epar', 'kperp', 'kperp_upar', 'ql_weight']
#: quantity_info_NC variables kept (prefixed nc_) only up to this many values
NC_MAX_SIZE = 16
#: IMAS eigenmode scalars saved as imas_<name>
IMAS_KEYS = ['growth_rate_norm', 'frequency_norm', 'growth_rate_tolerance']
#: GENE field and mom record order
FIELDS = ['phi', 'apar', 'bpar']
MOMS = ['dens', 'tpar', 'tperp', 'qpar', 'qperp', 'upar', 'densI1', 'TparI1', 'TppI1']
#: DBtag fields copied into attrs as mgkdb_<field>
DBTAG_ATTRS = ['user', 'run_collection_name', 'run_suffix', 'keywords',
               'confidence', 'comments', 'time_uploaded']


# --------------------------------------------------------------------------- #
# GENE binaries from the snapshot
# --------------------------------------------------------------------------- #
def _dtypes(precision: str, endianness: str):
    bo = '>' if str(endianness).upper() == 'BIG' else '<'
    double = str(precision).upper() == 'DOUBLE'
    return np.dtype(bo + ('f8' if double else 'f4')), np.dtype(bo + ('c16' if double else 'c8')), bo


def write_gene_binary(path: str, time: float, arrays, precision: str, endianness: str):
    """One time step in GENE's field/mom layout (Fortran records, as TPED's fieldlib/momlib read).

    Each array is (nz, ny, nx). Records: [int size][time][int size], then per
    array [int size][data, x fastest][int size]."""
    rt, ct, bo = _dtypes(precision, endianness)
    with open(path + '.part', 'wb') as fh:
        fh.write(struct.pack(bo + 'i', rt.itemsize))
        fh.write(np.array([time], dtype=rt).tobytes())
        fh.write(struct.pack(bo + 'i', rt.itemsize))
        for a in arrays:
            data = np.ascontiguousarray(a, dtype=ct).tobytes()
            fh.write(struct.pack(bo + 'i', len(data)))
            fh.write(data)
            fh.write(struct.pack(bo + 'i', len(data)))
    os.replace(path + '.part', path)


def read_last_step(path: str, nvars: int, shape, precision: str, endianness: str):
    """(time, [arrays (nz, ny, nx)]) of the last step of a GENE field/mom file."""
    rt, ct, _ = _dtypes(precision, endianness)
    nx, ny, nz = shape
    entry = nx * ny * nz * ct.itemsize
    step = (rt.itemsize + 8) + nvars * (entry + 8)
    size = os.path.getsize(path)
    with open(path, 'rb') as fh:
        fh.seek((size // step - 1) * step)
        raw = fh.read(step)
    t = np.frombuffer(raw, dtype=rt, count=1, offset=4)[0]
    off = rt.itemsize + 8
    arrays = []
    for _ in range(nvars):
        arrays.append(np.frombuffer(raw, dtype=ct, count=nx * ny * nz, offset=off + 4).reshape(nz, ny, nx))
        off += entry + 8
    return float(t), arrays


def read_pars(path: str) -> dict:
    """The few parameters scalars the writer needs, plus species names in order."""
    pars, species = {}, []
    with open(path) as fh:
        for line in fh:
            m = re.match(r'\s*(\w+)\s*=\s*(.+?)\s*$', line)
            if not m:
                continue
            k, v = m.group(1), m.group(2).strip().strip("'\"")
            if k == 'name':
                species.append(v)
            elif k in ('nx0', 'nky0', 'nz0', 'n_fields', 'n_moms'):
                pars[k] = int(float(v))
            elif k == 'kymin':
                pars[k] = float(v)
            elif k in ('PRECISION', 'ENDIANNESS'):
                pars[k] = v.upper()
    pars.setdefault('ENDIANNESS', 'LITTLE')
    pars['species'] = species
    return pars


def _kx_order(snap: dict, grid: dict | None):
    """Index that puts the snapshot's kx axis in GENE's native (FFT) order."""
    kx_snap = np.asarray(snap.get('kxgrid', []), dtype=float)
    if grid is None or kx_snap.size == 0 or 'kx_fftorder' not in grid:
        return None
    target = np.asarray(grid['kx_fftorder'], dtype=float)
    if kx_snap.shape != target.shape or np.allclose(kx_snap, target):
        return None
    return np.array([int(np.argmin(np.abs(kx_snap - k))) for k in target])


def snapshot_to_binaries(snap: dict, pars: dict, folder: str, suffix: str, grid: dict | None = None) -> list[str]:
    """Write field<suffix> and mom_<sp><suffix> from a Field Mom Snapshots dict.

    Snapshot arrays are (kx, ky, z); GENE files hold (z, ky, kx). Moments the
    snapshot lacks are written as zeros; their names are returned."""
    fm = snap['field_mom_final']
    t = float(np.ravel(snap['time'])[-1])
    shape = (pars['nx0'], pars['nky0'], pars['nz0'])
    order = _kx_order(snap, grid)

    def gene(a):
        a = np.asarray(a)
        if order is not None:
            a = a[order]
        return a.transpose(2, 1, 0)

    zero = np.zeros(shape[::-1], dtype=complex)
    missing = []
    flds = []
    for name in FIELDS[:pars['n_fields']]:
        if name in fm:
            flds.append(gene(fm[name]))
        else:
            flds.append(zero)
            missing.append(name)
    write_gene_binary(os.path.join(folder, f'field{suffix}'), t, flds, pars['PRECISION'], pars['ENDIANNESS'])
    for sp in pars['species']:
        arrs = []
        for m in MOMS[:pars['n_moms']]:
            if f'{m}{sp}' in fm:
                arrs.append(gene(fm[f'{m}{sp}']))
            else:
                arrs.append(zero)
                missing.append(f'{m}{sp}')
        write_gene_binary(os.path.join(folder, f'mom_{sp}{suffix}'), t, arrs, pars['PRECISION'], pars['ENDIANNESS'])
    return missing


# --------------------------------------------------------------------------- #
# MGKDB
# --------------------------------------------------------------------------- #
def connect():
    """LinearRuns collection and a GridFS handle, via TPED's MGKDB_AUTH_PKL."""
    import gridfs
    from mgkdb.support.mgk_login import f_login_dbase
    from TPED.config.config_helper import Config
    client, db = f_login_dbase(Config().get_path('MGKDB_AUTH_PKL')).connect()
    return db.LinearRuns, gridfs.GridFS(db)


def records(col, prefix: str, limit: int = 0):
    """Records under one campaign prefix: _id, DBtag, Files and Diagnostics only."""
    query = {'Metadata.DBtag.run_collection_name': {'$regex': '^' + re.escape(prefix.rstrip('/')) + '/'}}
    proj = {'Files': 1, 'Diagnostics': 1, 'Metadata.DBtag': 1}
    proj.update({f'gyrokineticsIMAS.linear.wavevector.eigenmode.{k}': 1 for k in IMAS_KEYS})
    cursor = col.find(query, proj).sort('_id', 1)
    return list(cursor.limit(limit) if limit else cursor)


def local_name(key: str, suffix: str) -> str:
    if key == 'quantity_info_NC':
        return f'{key}{suffix}.nc'
    if key in DIAGNOSTICS:
        return f'{DIAGNOSTICS[key]}{suffix}.pkl'
    return f'{key}{suffix}'


def run_dir(raw_root: str, rec: dict) -> str:
    return os.path.join(raw_root, rec['Metadata']['DBtag']['run_collection_name'].strip('/'))


def out_path(out_root: str, campaign: str, prefix: str, rec: dict) -> str:
    tag = rec['Metadata']['DBtag']
    sub = os.path.relpath(tag['run_collection_name'].rstrip('/'), prefix.rstrip('/'))
    return os.path.join(out_root, campaign, sub, f"quantity_info_NC{tag['run_suffix']}.nc")


def download(fs, rec: dict, raw_root: str) -> list[str]:
    """Write the FILES and DIAGNOSTICS of one record into its run dir; return the keys missing in MGKDB."""
    from bson import ObjectId
    dest = run_dir(raw_root, rec)
    os.makedirs(dest, exist_ok=True)
    suffix = rec['Metadata']['DBtag']['run_suffix']
    wanted = [(k, rec.get('Files', {}).get(k)) for k in FILES]
    wanted += [(k, rec.get('Diagnostics', {}).get(k)) for k in DIAGNOSTICS]
    missing = []
    for key, fid in wanted:
        if fid in (None, 'None', ''):
            missing.append(key)
            continue
        path = os.path.join(dest, local_name(key, suffix))
        if os.path.exists(path):
            continue
        data = fs.get(fid if isinstance(fid, ObjectId) else ObjectId(str(fid))).read()
        with open(path + '.part', 'wb') as fh:
            fh.write(data)
        os.replace(path + '.part', path)
    return missing


def imas_values(rec: dict) -> dict:
    """First wavevector's first eigenmode scalars from the IMAS branch (NaN if absent)."""
    try:
        em = rec['gyrokineticsIMAS']['linear']['wavevector'][0]['eigenmode'][0]
    except (KeyError, IndexError, TypeError):
        em = {}
    out = {}
    for k in IMAS_KEYS:
        try:
            out[k] = float(np.ravel(em.get(k))[0])
        except (TypeError, ValueError, IndexError):
            out[k] = np.nan
    return out


def fill_empty_omega(path: str, kymin: float) -> bool:
    """Write 'kymin 0 0' into an omega file with no data line (epar reads omega); True if done."""
    text = open(path).read() if os.path.exists(path) else ''
    if any(l.strip() and not l.strip().startswith('#') for l in text.splitlines()):
        return False
    with open(path, 'w') as fh:
        fh.write(f'{kymin:10.4f} {0.0:10.4f} {0.0:10.4f}\n' + text)
    return True


def omega_from_file(om, empty: bool = False) -> 'xr.Dataset':
    """field_omega's variables from the omega node alone (no field history)."""
    import xarray as xr
    nan = np.nan

    def val(name):
        return float(om[name].values) if name in om else nan

    g, w = val('gamma_gene'), val('omega_gene')
    best = (nan, nan, nan, nan)
    if not np.isnan(g):
        if g == 0 and w == 0:
            source = 'GENE_unconverged'
        elif g == -1 and w == -1:
            source = 'GENE_sentinel'
        else:
            source, best = 'GENE', (g, w, nan, nan)
    elif om.attrs.get('source') == 'field' and not np.isnan(val('gamma')):
        source, best = 'field_legacy', (val('gamma'), val('omega'), val('gamma_std'), val('omega_std'))
    else:
        source = 'empty'
    if empty:
        source, best = 'empty', (nan, nan, nan, nan)
    return xr.Dataset({
        'fit_gamma': nan, 'fit_omega': nan, 'fit_gamma_std': nan, 'fit_omega_std': nan,
        'fit_t_min': nan, 'fit_t_max': nan, 'fit_n_samples': 0, 'fit_verdict': 'no_history',
        'omega_source': source,
        'converged': source in ('GENE', 'field_legacy'),
        'gamma_best': best[0], 'omega_best': best[1],
        'gamma_best_std': best[2], 'omega_best_std': best[3],
    })


def build(rec: dict, raw_root: str, campaign: str):
    """Snapshot -> GENE binaries, TPED registry, then the nc_-prefixed quantity_info_NC."""
    import xarray as xr
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    from TPED.projects.MGKDB_tools.src.upload.quantity_info_xarray import prepare_attrs_for_netcdf

    tag = rec['Metadata']['DBtag']
    folder, suffix = run_dir(raw_root, rec), tag['run_suffix']
    param_file = os.path.join(folder, f'parameters{suffix}')
    with open(os.path.join(folder, local_name('Field Mom Snapshots', suffix)), 'rb') as fh:
        snap = pickle.load(fh)  # MGKDB's own pickle (group database)
    with open(os.path.join(folder, local_name('Grid', suffix)), 'rb') as fh:
        grid = pickle.load(fh)
    pars = read_pars(param_file)
    zero_filled = snapshot_to_binaries(snap, pars, folder, suffix, grid)
    empty = fill_empty_omega(os.path.join(folder, f'omega{suffix}'), pars.get('kymin', np.nan))

    sim = GeneSimulationPlus(param_file)
    sim.analysis(ANALYSES)
    sim.return_sim_xarray(['omega'])
    sim.add_derived('field_omega', omega_from_file(sim.tree['omega'].ds, empty), derived_from=['omega'],
                    method='omega_from_file')
    sim.analysis(['mode_identifier'])
    ds = netcdf_safe(sim.flat_xarray, list(sim.tree['parameters'].ds.data_vars))
    for k, v in imas_values(rec).items():
        ds[f'imas_{k}'] = xr.DataArray(v)
    ds = add_cost(ds)
    ds['shot'] = np.int64(CAMPAIGN_SHOT.get(campaign, -1))
    ds['campaign'] = campaign

    nc = xr.load_dataset(os.path.join(folder, local_name('quantity_info_NC', suffix)))
    nc = nc[[k for k in nc.data_vars if nc[k].size <= NC_MAX_SIZE]]
    nc = netcdf_safe(nc, list(nc.data_vars))
    nc = nc.rename({k: f'nc_{k}' for k in nc.data_vars})
    nc = nc.rename({d: f'nc_{d}' for d in nc.dims if d in ds.dims})
    ds = xr.merge([ds, nc.drop_vars([c for c in nc.coords if c not in nc.dims])],
                  combine_attrs='drop_conflicts')

    ds.attrs['campaign'] = campaign
    ds.attrs['mgkdb_id'] = str(rec['_id'])
    for k in DBTAG_ATTRS:
        ds.attrs[f'mgkdb_{k}'] = '' if tag.get(k) is None else str(tag.get(k))
    ds.attrs['snapshot_zero_filled'] = ','.join(zero_filled)
    ds.attrs = prepare_attrs_for_netcdf(ds.attrs)
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--raw-root', default=RAW_ROOT, help=f'where downloaded MGKDB files go (default {RAW_ROOT})')
    p.add_argument('--out-root', default=OUT_ROOT, help=f'where the per-run netCDFs go (default {OUT_ROOT})')
    p.add_argument('--campaign', nargs='+', choices=sorted(CAMPAIGNS), default=sorted(CAMPAIGNS))
    p.add_argument('--limit', type=int, default=0, help='first N records per campaign (0 = all)')
    p.add_argument('--download-only', action='store_true', help='download, do not build netCDFs')
    p.add_argument('--dry-run', action='store_true', help='count records per campaign, write nothing')
    args = p.parse_args(argv)

    col, fs = connect()
    jobs = {c: records(col, CAMPAIGNS[c], args.limit) for c in args.campaign}
    for c, recs in jobs.items():
        todo = sum(not os.path.exists(out_path(args.out_root, c, CAMPAIGNS[c], r)) for r in recs)
        print(f'{c}: {len(recs)} records, {todo} netCDFs to build -> {os.path.join(args.out_root, c)}')
    if args.dry_run:
        raise SystemExit(0)

    n, failed = 0, []
    for c, recs in jobs.items():
        print(f'\n===== {c} =====')
        for rec in recs:
            dest = out_path(args.out_root, c, CAMPAIGNS[c], rec)
            if os.path.exists(dest):
                continue
            n += 1
            try:
                missing = download(fs, rec, args.raw_root)
                if missing:
                    raise FileNotFoundError(f'not in MGKDB: {missing}')
                if args.download_only:
                    continue
                ds = build(rec, args.raw_root, c)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                ds.to_netcdf(dest + '.part', format='NETCDF4')
                os.replace(dest + '.part', dest)
                print(f"saved {dest}  [{ds['omega_source'].item()}, gamma_best {ds['gamma_best'].item():.3g}, "
                      f"{ds['mode_ID'].item()}]")
            except Exception:
                print(f"FAILED {rec['_id']} {rec['Metadata']['DBtag'].get('run_collection_name')}")
                traceback.print_exc()
                failed.append(str(rec['_id']))
    print(f'\ndone {n - len(failed)}/{n}' + (f', failed ids: {failed}' if failed else ''))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
