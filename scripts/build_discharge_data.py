"""Save TPED quantity_info xarrays of GENE runs into the NSTX_discharge_data repo.

Each GENE run under a diag dir is loaded with TPED (last time slice only, as in
MGKDB_tools quantity_info_xarray), run through the same analysis chain
(fingerprints, Epar, kperp, kperp(upar), QL weights), and saved as one netCDF:

    <out-root>/<shot>/<path of the run relative to the diag dir>/quantity_info_NC_<suffix>.nc

The relative path is kept so scanfiles0000/parameters_0001 and
scanfiles0001/parameters_0001 do not overwrite each other. --diag-dir may be an
x0_<x0> folder (its name is kept in the output path) or any dir above the x0_*
folders, e.g. .../129038/corners.

Besides the analysis quantities, every scalar from the run's parameters file is
saved (resolution nx0..nw0, n_procs_*, GENE's &info timing: step_time,
'number of computed time steps' -> number_of_computed_time_steps, ...), plus
cost_* variables derived from them (see add_cost).

    python -m NSTXU_lithium_study.scripts.build_discharge_data --shot 129038 --diag-dir /pscratch/.../129038/corners --out-root ~/NSTX_discharge_data --dry-run
    python -m NSTXU_lithium_study.scripts.build_discharge_data --shot 129038 --diag-dir /pscratch/.../129038/corners/x0_0.6737 --out-root ~/NSTX_discharge_data
    python -m NSTXU_lithium_study.scripts.build_discharge_data --from-paths-json --shot 129038 --x0 0.6737 --out-root ~/NSTX_discharge_data
    python -m NSTXU_lithium_study.scripts.build_discharge_data --from-paths-json --out-root ~/NSTX_discharge_data --skip-existing
"""

from __future__ import annotations

import os
import re
import traceback
from collections import defaultdict

#: GENE files loaded per run, and what is read from them (MGKDB quantity_info set)
FILETYPES = ['field', 'nrg', 'omega', 'geometry', 'mom', 'vsp']
INPUT_QUANTITIES = {
    'time_criteria': 'last',
    'species': 'all',
    'field_quantities': ['field_phi', 'field_apar', 'field_bpar'],
    'mom_quantities': ['mom_upar', 'mom_tpar', 'mom_tperp'],
    'nrg_quantities': ['Gamma_ES', 'Gamma_EM', 'Q_ES', 'Q_EM', 'n_mag', 'u_par_mag'],
}
#: a run counts as stopped by its wall-clock limit past this fraction of timelim
TIMELIM_FRAC = 0.98


def find_runs(diag_dir: str, max_depth: int = 4) -> list[tuple[str, str]]:
    """(parameters file, output subdir) for every GENE run under diag_dir.

    The output subdir is the run's folder relative to diag_dir, prefixed with
    diag_dir's own name when that is an x0_<x0> folder."""
    from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder
    diag_dir = os.path.abspath(diag_dir)
    base = os.path.dirname(diag_dir) if os.path.basename(diag_dir).startswith('x0_') else diag_dir
    files = GeneFileFinder(diag_dir).find_files('parameters', max_depth=max_depth)
    return [(f, os.path.relpath(os.path.dirname(f), base)) for f in files]


def out_path(out_root: str, shot: int, subdir: str, param_file: str) -> str:
    suffix = os.path.basename(param_file).split('_')[-1] if '_' in os.path.basename(param_file) else 'dat'
    return os.path.join(out_root, str(shot), subdir, f'quantity_info_NC_{suffix}.nc')


def parameter_scalars(param_ds):
    """Every 0-d parameter as a netCDF-safe variable: spaces in GENE's &info keys
    become underscores, bools become int8, objects become str."""
    import numpy as np
    out = {}
    for k, da in param_ds.data_vars.items():
        if da.ndim:
            continue
        name = re.sub(r'\W+', '_', str(k).strip()).strip('_')
        if da.dtype == bool:
            da = da.astype('i1')
        elif da.dtype == object:
            da = da.astype(str)
        elif da.dtype == np.longdouble:
            da = da.astype(np.float64)
        out[name] = da
    return out


def add_cost(ds):
    """cost_* variables from GENE's &info timing and the run's resolution.

    cost_core_hours       wall time of the time loop x MPI ranks x OMP threads
    cost_grid_points      nx0*nky0*nz0*nv0*nw0*n_spec
    cost_core_s_per_step_point  core-seconds per time step per grid point
    cost_hit_ntimesteps   1 if the run used every allowed time step
    cost_hit_timelim      1 if the time loop ran past TIMELIM_FRAC of timelim
    Missing inputs give NaN (-1 for the flags)."""
    import numpy as np

    def get(name):
        try:
            return float(ds[name].values)
        except (KeyError, TypeError, ValueError):
            return np.nan

    wall, steps = get('time_for_initial_value_solver'), get('number_of_computed_time_steps')
    ranks, threads = get('n_procs_sim'), get('OMP_NUM_THREADS')
    threads = 1.0 if np.isnan(threads) else threads
    grid = np.prod([get(n) for n in ('nx0', 'nky0', 'nz0', 'nv0', 'nw0', 'n_spec')])
    core_s = wall * ranks * threads
    ntmax, tlim = get('ntimesteps'), get('timelim')

    def flag(ok, cond):
        return np.int8(cond) if ok else np.int8(-1)

    ds['cost_core_hours'] = core_s / 3600
    ds['cost_grid_points'] = grid
    ds['cost_core_s_per_step_point'] = core_s / (steps * grid)
    ds['cost_hit_ntimesteps'] = flag(not np.isnan(steps + ntmax), steps >= ntmax)
    ds['cost_hit_timelim'] = flag(not np.isnan(wall + tlim), wall >= TIMELIM_FRAC * tlim)
    return ds


def build(param_files: list[str]) -> list:
    """Load, analyse and sample param_files into netCDF-ready Datasets
    (attrs['filepath'] names the parameters file each came from)."""
    import numpy as np
    from TPED.projects.GENE_sim_reader.src.GENE_sim_data import ParallelGeneMultiSimulations as PGMS
    from TPED.projects.MGKDB_tools.src.upload.quantity_info_xarray import (
        sampled_TPED_xarrays, prepare_attrs_for_netcdf)
    from TPED.projects.GENE_sim_analysis.src.analysis.fingerprints_quantities import fingerprint_analysis
    from TPED.projects.GENE_sim_analysis.src.analysis.E_par_calc import epar_calc
    from TPED.projects.GENE_sim_analysis.src.analysis.kperp_calc import kperp_analysis
    from TPED.projects.GENE_sim_analysis.src.analysis.ql_weight_calc import ql_weight_analysis
    from TPED.projects.GENE_sim_analysis.src.analysis.kperp_upar_calc import kperp_upar_analysis

    sims = PGMS(param_files)
    sims.return_sim_xarray_list(filetypes_list=FILETYPES, input_quantities=INPUT_QUANTITIES)
    # the analysis chain and the sampler read the pre-DataTree flat layout
    flats = [s.flat_xarray for s in sims.list_sim_objects]
    flats = fingerprint_analysis(flats, species='all')
    flats = epar_calc(flats)
    flats = kperp_analysis(flats)
    flats = kperp_upar_analysis(flats)
    flats = ql_weight_analysis(flats)
    sampled = sampled_TPED_xarrays(flats)

    out = []
    for sim, ds in zip(sims.list_sim_objects, sampled):
        for name, da in parameter_scalars(sim.tree['parameters'].to_dataset()).items():
            if name not in ds:
                ds[name] = da
        ds = add_cost(ds)
        for v in ds.data_vars:
            if ds[v].dtype == np.longdouble:
                ds[v] = ds[v].astype(np.float64)
        ds.attrs = prepare_attrs_for_netcdf(ds.attrs)
        out.append(ds)
    return out


def jobs_from_paths_json(path: str | None, shots, x0s) -> list[tuple[int, str]]:
    """(shot, x0 dir) for the gene_run_paths.json entries matching --shot/--x0."""
    from NSTXU_lithium_study.sparse_grid_results.update_results import PATHS, load_entries, select
    return [(e['shot'], os.path.join(e['diag_root'], e['label']))
            for e in select(load_entries(path or PATHS), shots, x0s) if e['diag_root']]


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--diag-dir', nargs='+',
                     help='x0_<x0> folder(s), or dir(s) above them, holding the GENE runs; needs one --shot')
    src.add_argument('--from-paths-json', nargs='?', const='', metavar='PATHS',
                     help='take <diag_root>/x0_<x0> of every gene_run_paths.json entry '
                          '[sparse_grid_results/gene_run_paths.json]; filter with --shot/--x0')
    ap.add_argument('--shot', type=int, nargs='+', default=None, help='discharge(s)')
    ap.add_argument('--x0', type=float, nargs='+', default=None, help='radii, with --from-paths-json')
    ap.add_argument('--out-root', required=True, help='NSTX_discharge_data clone; writes <shot>/... under it')
    ap.add_argument('--max-depth', type=int, default=4, help='folder depth searched below each diag dir')
    ap.add_argument('--skip-existing', action='store_true', help='leave runs whose .nc already exists')
    ap.add_argument('--dry-run', action='store_true', help='list runs and output paths; load nothing')
    args = ap.parse_args()

    if args.diag_dir is not None:
        if not args.shot or len(args.shot) != 1:
            raise SystemExit('--diag-dir needs exactly one --shot')
        jobs = [(args.shot[0], d) for d in args.diag_dir]
    else:
        jobs = jobs_from_paths_json(args.from_paths_json or None, args.shot, args.x0)
        if not jobs:
            raise SystemExit(f'no gene_run_paths.json entry matches --shot {args.shot} --x0 {args.x0}')

    # one load per run folder (a scanfiles dir), so a bad folder costs only itself
    groups = defaultdict(list)
    for shot, diag in jobs:
        try:
            runs = find_runs(diag, args.max_depth)
        except FileNotFoundError as exc:
            print(f'{shot} {diag}: {exc}')
            continue
        for f, sub in runs:
            dest = out_path(args.out_root, shot, sub, f)
            if args.skip_existing and os.path.exists(dest):
                continue
            groups[(shot, sub)].append((f, dest))

    n = sum(len(v) for v in groups.values())
    print(f'{n} runs in {len(groups)} folders')
    for (shot, sub), items in sorted(groups.items()):
        print(f'  {shot}/{sub}: {len(items)} runs -> {os.path.dirname(items[0][1])}')
    if args.dry_run:
        raise SystemExit(0)

    failed = []
    for (shot, sub), items in sorted(groups.items()):
        print(f'\n===== {shot}/{sub} =====')
        try:
            dests = {os.path.normpath(f): d for f, d in items}
            for ds in build([f for f, _ in items]):
                dest = dests[os.path.normpath(ds.attrs['filepath'])]
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                ds.to_netcdf(dest)
                print(f'saved {dest}')
        except Exception:
            traceback.print_exc()
            failed.append(f'{shot}/{sub}')
    print(f'\nbuilt {len(groups) - len(failed)}/{len(groups)} folders'
          + (f'; failed: {", ".join(failed)}' if failed else ''))
    if failed:
        raise SystemExit(1)
