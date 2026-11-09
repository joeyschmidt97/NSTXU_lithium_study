"""Pull MGKDB linear GENE runs and save one netCDF per run, NSTX-style.

MGKDB keeps nrg, omega, parameters and the uploader's quantity_info_NC file per
run, but no field or mom files. So each run's netCDF is:

- TPED on the downloaded parameters + nrg + omega (current TPED): every
  parameters scalar, the last nrg step, GENE's gamma/omega, and the registry
  analyses that need only nrg (fingerprints, ql_weight). Same names as the NSTX
  files from build_discharge_data, ql_weight_EM included (fixed in TPED 9863466).
- the quantity_info_NC variables, prefixed nc_ (Epar, parity, widths, kperp and
  kx averages, ...). These are the uploader's last-time-slice values from the
  TPED of the upload date; they are the only source of field-derived quantities.
- gene_converged: 1 where GENE's omega file is not 0/0.
- MGKDB bookkeeping in attrs (mgkdb_id, mgkdb_user, run_collection_name, ...).

No mode label is written: TPED's mode_identifier needs the epar and field_omega
analyses, which need field files. Label downstream from nc_Epar, nc_parity_*,
the fingerprints and gamma/omega.

Two steps, both resumable (files already present are skipped):

    <raw-root>/<run_collection_name>/{parameters,nrg,omega}<suffix>, quantity_info_NC<suffix>.nc
    <out-root>/<campaign>/<scanfilesNNNN>/quantity_info_NC<suffix>.nc

Campaigns are matched on Metadata.DBtag.run_collection_name with an anchored
prefix, so hatch/pscans does not also catch drhatch/pscans2 and pscans3.

Run from the git parent dir (holding NSTXU_lithium_study and TPED) in an env
with both mgkdb and TPED:

    python -m NSTXU_lithium_study.scripts.build_mgkdb_data --raw-root /global/cfs/cdirs/m2116/MGKDB_raw --out-root /global/cfs/cdirs/m2116/MGKDB_netcdf --dry-run
    python -m NSTXU_lithium_study.scripts.build_mgkdb_data --raw-root /global/cfs/cdirs/m2116/MGKDB_raw --out-root /global/cfs/cdirs/m2116/MGKDB_netcdf --campaign drhatch_pscans2 --limit 5
    python -m NSTXU_lithium_study.scripts.build_mgkdb_data --raw-root /global/cfs/cdirs/m2116/MGKDB_raw --out-root /global/cfs/cdirs/m2116/MGKDB_netcdf
"""

from __future__ import annotations

import argparse
import os
import re
import traceback

from NSTXU_lithium_study.scripts.build_discharge_data import netcdf_safe

#: campaign name -> run_collection_name prefix (matched as ^<prefix>/)
CAMPAIGNS = {
    'hatch_pscans': '/global/cfs/cdirs/m2116/hatch/pscans',
    'drhatch_pscans2': '/pscratch/sd/d/drhatch/pscans2',
    'drhatch_pscans3': '/pscratch/sd/d/drhatch/pscans3',
    'lleppin_153764': '/pscratch/sd/l/lleppin/DIIID153764/lin_loc',
    'lleppin_174082': '/pscratch/sd/l/lleppin/DIIID174082/lin_loc',
}
#: MGKDB Files keys downloaded per run (gfiles, profiles, tracer_efit, autopar skipped)
FILES = ['parameters', 'nrg', 'omega', 'quantity_info_NC']
#: TPED registry analyses that need only nrg
ANALYSES = ['fingerprints', 'ql_weight']
#: DBtag fields copied into attrs as mgkdb_<field>
DBTAG_ATTRS = ['user', 'run_collection_name', 'run_suffix', 'keywords',
               'confidence', 'comments', 'time_uploaded']


def connect():
    """LinearRuns collection and a GridFS handle, via TPED's MGKDB_AUTH_PKL."""
    import gridfs
    from mgkdb.support.mgk_login import f_login_dbase
    from TPED.config.config_helper import Config
    client, db = f_login_dbase(Config().get_path('MGKDB_AUTH_PKL')).connect()
    return db.LinearRuns, gridfs.GridFS(db)


def records(col, prefix: str, limit: int = 0):
    """Records under one campaign prefix: _id, DBtag and Files only."""
    query = {'Metadata.DBtag.run_collection_name': {'$regex': '^' + re.escape(prefix.rstrip('/')) + '/'}}
    cursor = col.find(query, {'Files': 1, 'Metadata.DBtag': 1}).sort('_id', 1)
    return list(cursor.limit(limit) if limit else cursor)


def local_name(key: str, suffix: str) -> str:
    return f'{key}{suffix}.nc' if key == 'quantity_info_NC' else f'{key}{suffix}'


def run_dir(raw_root: str, rec: dict) -> str:
    return os.path.join(raw_root, rec['Metadata']['DBtag']['run_collection_name'].strip('/'))


def out_path(out_root: str, campaign: str, prefix: str, rec: dict) -> str:
    tag = rec['Metadata']['DBtag']
    sub = os.path.relpath(tag['run_collection_name'].rstrip('/'), prefix.rstrip('/'))
    return os.path.join(out_root, campaign, sub, f"quantity_info_NC{tag['run_suffix']}.nc")


def download(fs, rec: dict, raw_root: str) -> list[str]:
    """Write the FILES of one record into its run dir; return the keys missing in MGKDB."""
    from bson import ObjectId
    dest = run_dir(raw_root, rec)
    os.makedirs(dest, exist_ok=True)
    suffix = rec['Metadata']['DBtag']['run_suffix']
    missing = []
    for key in FILES:
        fid = rec.get('Files', {}).get(key)
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


def build(rec: dict, raw_root: str, campaign: str):
    """TPED (parameters + nrg + omega) merged with the nc_-prefixed quantity_info_NC."""
    import numpy as np
    import xarray as xr
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    from TPED.projects.MGKDB_tools.src.upload.quantity_info_xarray import prepare_attrs_for_netcdf

    tag = rec['Metadata']['DBtag']
    folder, suffix = run_dir(raw_root, rec), tag['run_suffix']
    sim = GeneSimulationPlus(os.path.join(folder, f'parameters{suffix}'))
    sim.analysis(ANALYSES)
    sim.return_sim_xarray(['omega'])
    ds = netcdf_safe(sim.flat_xarray, list(sim.tree['parameters'].ds.data_vars))
    ds['gene_converged'] = xr.DataArray(np.int8((ds['gamma'].item() != 0) or (ds['omega'].item() != 0)))

    nc = xr.load_dataset(os.path.join(folder, local_name('quantity_info_NC', suffix)))
    nc = netcdf_safe(nc, list(nc.data_vars))
    nc = nc.rename({k: f'nc_{k}' for k in nc.data_vars})
    nc = nc.rename_dims({d: f'nc_{d}' for d in nc.dims if d in ds.dims})
    ds = xr.merge([ds, nc.drop_vars([c for c in nc.coords if c not in nc.dims])],
                  combine_attrs='drop_conflicts')

    ds.attrs['campaign'] = campaign
    ds.attrs['mgkdb_id'] = str(rec['_id'])
    for k in DBTAG_ATTRS:
        ds.attrs[f'mgkdb_{k}'] = '' if tag.get(k) is None else str(tag.get(k))
    ds.attrs = prepare_attrs_for_netcdf(ds.attrs)
    return ds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--raw-root', required=True, help='where downloaded MGKDB files go')
    p.add_argument('--out-root', required=True, help='where the per-run netCDFs go')
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
                print(f"saved {dest}  [gamma {ds['gamma'].item():.3g}, omega {ds['omega'].item():.3g}]")
            except Exception:
                print(f"FAILED {rec['_id']} {rec['Metadata']['DBtag'].get('run_collection_name')}")
                traceback.print_exc()
                failed.append(str(rec['_id']))
    print(f'\ndone {n - len(failed)}/{n}' + (f', failed ids: {failed}' if failed else ''))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
