"""Save TPED quantity_info xarrays of GENE runs into the NSTX_discharge_data repo.

Each GENE run under a diag dir is loaded with TPED (last time slice only) and
run through TPED's analysis registry (ANALYSES: fingerprints, epar, kperp,
kperp_upar, ql_weight, mode_identifier, field_omega). The run's tree is
flattened (sim.flat_xarray: raw parameters/omega/nrg/field/mom/geometry data
plus every analysis node) and saved as one netCDF:

    <out-root>/<shot>/<path of the run relative to the diag dir>/quantity_info_NC_<suffix>.nc

The relative path is kept so scanfiles0000/parameters_0001 and
scanfiles0001/parameters_0001 do not overwrite each other. --diag-dir may be an
x0_<x0> folder (its name is kept in the output path) or any dir above the x0_*
folders, e.g. .../129038/corners.

Parameters names are made netCDF-safe (GENE's &info 'number of computed time
steps' -> number_of_computed_time_steps, ...), the corner folder name
(Teh0.80_Tew0.80_neh0.80_new0.80) becomes Te_ped_scale, Te_width_scale,
ne_ped_scale, ne_width_scale and attrs['corner_tag'], complex arrays are split into
<name>_re / <name>_im, and cost_* variables are added (see add_cost). vsp is
not loaded: it was ~97% of each file and no analysis reads it.

field_omega gives every run a convergence measure: the field_phi fit over the
last 30% of the run (fit_gamma/fit_omega with *_std, fit_verdict), GENE's state
(omega_source), a converged flag, and gamma_best/omega_best for training.

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

#: TPED registry analyses run per run; each preloads the GENE files it declares
ANALYSES = ['fingerprints', 'epar', 'kperp', 'kperp_upar', 'ql_weight',
            'mode_identifier', 'field_omega']
#: a run counts as stopped by its wall-clock limit past this fraction of timelim
TIMELIM_FRAC = 0.98
#: corner folder names, e.g. Teh0.80_Tew0.80_neh0.80_new0.80 (h = pedestal height, w = width)
CORNER_TAG = re.compile(r'(?:(?:Te|Ti|ne)[hw]\d+(?:\.\d+)?_?)+')
CORNER_AXIS = re.compile(r'(Te|Ti|ne)([hw])(\d+(?:\.\d+)?)')
#: always written, NaN when the path has no corner tag, so runs concatenate
CORNER_AXES = ['Te_ped_scale', 'Te_width_scale', 'ne_ped_scale', 'ne_width_scale']


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


def netcdf_safe(ds, param_names):
    """ds with every variable writable to netCDF.

    Parameter names: non-word characters become underscores (GENE's &info keys
    have spaces). bool -> int8, object -> str, longdouble -> float64, complex ->
    <name>_re and <name>_im. Variable attrs that are not str/number/list become
    JSON strings."""
    import json
    import numpy as np
    ds = ds.rename({k: re.sub(r'\W+', '_', str(k).strip()).strip('_')
                    for k in param_names if k in ds.data_vars})
    for k in list(ds.data_vars):
        da = ds[k]
        if da.dtype == bool:
            ds[k] = da.astype('i1')
        elif da.dtype == object:
            ds[k] = da.astype(str)
        elif da.dtype == np.longdouble:
            ds[k] = da.astype(np.float64)
        elif np.iscomplexobj(da):
            ds[f'{k}_re'], ds[f'{k}_im'] = da.real, da.imag
            ds = ds.drop_vars(k)
            continue
        ds[k].attrs = {a: (v if isinstance(v, (str, int, float, np.number)) else json.dumps(v, default=str))
                       for a, v in ds[k].attrs.items()}
    return ds


def corner_scales(param_file: str) -> tuple[str, dict]:
    """(corner tag, {<prof>_<ped|width>_scale: value}) from the last folder in
    param_file's path named like Teh0.80_Tew0.80_neh0.80_new0.80; ('', {})
    without one."""
    for part in reversed(os.path.normpath(os.path.dirname(param_file)).split(os.sep)):
        if CORNER_TAG.fullmatch(part):
            return part, {f'{p}_{"ped" if hw == "h" else "width"}_scale': float(v)
                          for p, hw, v in CORNER_AXIS.findall(part)}
    return '', {}


def add_corner(ds, param_file: str):
    """Corner multipliers as variables (CORNER_AXES always, NaN if absent) and
    the folder name as attrs['corner_tag']."""
    import numpy as np
    tag, scales = corner_scales(param_file)
    for name in dict.fromkeys(CORNER_AXES + list(scales)):
        ds[name] = scales.get(name, np.nan)
    ds.attrs['corner_tag'] = tag
    return ds


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


def build(param_file: str):
    """Load and analyse one run into a netCDF-ready Dataset."""
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    from TPED.projects.MGKDB_tools.src.upload.quantity_info_xarray import prepare_attrs_for_netcdf

    sim = GeneSimulationPlus(param_file)
    sim.analysis(ANALYSES)
    ds = netcdf_safe(sim.flat_xarray, list(sim.tree['parameters'].ds.data_vars))
    ds = add_cost(ds)
    ds = add_corner(ds, param_file)
    ds.attrs = prepare_attrs_for_netcdf(ds.attrs)
    return ds


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

    # grouped by run folder (a scanfiles dir) for reporting; each run loads alone
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
        for f, dest in items:
            try:
                ds = build(f)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                ds.to_netcdf(dest)
                print(f'saved {dest}  [{ds["omega_source"].item()}, fit {ds["fit_verdict"].item()}, '
                      f'{ds["mode_ID"].item()}]')
            except Exception:
                traceback.print_exc()
                failed.append(f)
    print(f'\nbuilt {n - len(failed)}/{n} runs'
          + (''.join(f'\n  failed: {f}' for f in failed) if failed else ''))
    if failed:
        raise SystemExit(1)
