"""Plots for the 16-corner linear GENE scan (corner_gene_scan.py): growth, profiles, s-alpha, locals.

Reads every corner corner_gene_scan.py launched,

    GENE output   <diag-root>/x0_<x0>/<tag>/scanfilesNNNN/{parameters,omega}_000K
                  (every scanfiles dir, merged by ky)
    profiles      <ped-root>/<tag>/profiles.iterdb   (what GENE read, prof_type -2)
    q             <ped-root>/<tag>/*_CH_COCOS*.geqdsk (the cheaseBS equilibrium GENE read)

into one xarray Dataset with dims (corner, ky, rho_tor), and writes to <out>/corners/
(the same plots for each VURS batch go to <out>/batch<i>_<j>/, see below)

    corners_growth.png          gamma and omega vs ky, all corners, GENE-converged ky only
    corners_growth_field.png    the same with the field-fit guess (+-1 std) at every
                                unconverged ky, on its corner's own line
    corners_profiles.png        ne, Te, ni, Ti and their -dln/drho_tor, x0 marked; under
                                them |q| vs rho_tor, a faint line at each integer q and a
                                drop line where each corner crosses it (rational surfaces,
                                to line MTM drive up against q = m/n)
    corners_profiles_ped.png    the same, zoomed to PED_ZOOM
    corners_mode_scaling.png    (--modes) mode per corner and ky as a table, each cell
                                the mode's colour shaded by gamma (dark = fastest corner
                                at that ky, light = slowest) with gamma printed, and the
                                mode mix at each knob level; + corners_mode_scaling.csv
    corners_s_alpha.png         shat vs alpha_MHD at x0: each corner a small ring of
                                its converged ky (unconverged ones left out), each
                                coloured by that ky's gamma (marker = its mode under
                                --modes), the ion-scale peak circled, numbered, with
                                a key beside (gamma range, ion-peak ky and mode)
    corners_s_alpha_ion.png     shat vs alpha_MHD, one marker per case: its mode at the
                                ion-scale peak: the first local max of gamma over the
                                converged ky confirmed by a quadratic fit in log ky,
                                else (hollow) the max gamma
    corners_local_params.png    the local parameters that vary most: % from the corner
                                mean, or the absolute difference for those that change sign
    corners_local_table.png     their values per corner, as a table
    corners_local_params.csv    every local parameter per corner (+ gamma_max)
    corners_q_offset.png        rational-surface offset per corner and ky: delta =
                                n q0 - round(n q0) (n = n0_global, q0 at x0, both from
                                the run's parameters), as a table, and gamma vs |delta|;
                                + corners_q_offset.csv
    corners_mode_extent.png     (--mode-extent) has each eigenfunction decayed by the ends
                                of the ballooning domain? |phi| (top) and |Apar|
                                (bottom) at the left and right z ends over their peak
                                (last field time) against profile scaling, coloured by
                                ky, on one log scale; a large end value means the mode
                                is cut off and nx0 should grow; + corners_mode_extent.csv
    corners_mode_structure.png  (--mode-extent) the corners_mode_scaling.png table layout,
                                each cell |phi| and |Apar| / peak along the whole
                                ballooning domain (last field time): the mode structure
                                per corner and ky at a glance, outlined in its mode's
                                colour, dashed where an end is above EXTENT_TOL, grey
                                where not converged
    corners_mode_groups.png     (--mode-extent) the converged runs' mode structures grouped
                                by shape: earth mover's distance between |phi| (and
                                |Apar|) along z, average-linkage groups closer than
                                --group-tol, one column per group with its members'
                                |phi| (top) and |Apar| (bottom) overlaid, coloured by mode (under --modes), runs
                                in no group together in a last "outliers" panel;
                                + corners_mode_groups.csv
    corners_vspace_edge.png     (--vspace-edge) has f decayed by the edges of the velocity
                                box? GENE's vsp <|f|^2>^1/2 (per species, over x and ky)
                                at v_par = -lv (<) and +lv (>) (left) and at the top mu
                                point, mu ~ lw (right), over its peak (last vsp time, max
                                over z), against profile scaling, coloured by ky, one row
                                per species; a large edge value means f is cut off by the
                                box: grow lv (or nv0) for v_par, lw (or nw0) for mu;
                                + corners_vspace_edge.csv (per-species ratios, pass/fail
                                against --vspace-tol)
    corners_outlier.png         (--outliers) TPED's outlier analysis per corner and ky
                                (GENE_sim_analysis analyses/outlier.py): as a table, red
                                where any criterion trips, each cell its near-Nyquist
                                phi power fraction and the criteria that tripped (grid
                                = grid-scale oscillation along z, gamma < 0, D/chi_e,
                                chi_i/chi_e, x0); beside it that fraction vs ky per
                                corner, every run read, converged or not, with the
                                threshold drawn; + corners_outlier.csv
    corners_res2.png         (when 2x-resolution runs exist) the 1x corner beside
                                its 2x rerun: gamma and omega vs ky, and 2x gamma vs
                                1x gamma; + corners_res2.csv
    corners_nz<N>.png           (--nz-compare N) the same comparison against the
                                nz0 = N runs under <diag-root>_nz<N> (or --nz-root),
                                each matched to the corner of the same profile
                                scaling (tag); the line is this scan, at its own nz0
                                read from its parameters; + corners_nz<N>.csv
    s_alpha_ion.png (in <out>/) corners_s_alpha_ion.png for every case at x0: corners,
                                reference and every VURS batch on one s-alpha, marker
                                area = gamma at the ion peak; + s_alpha_ion.csv
    sampling.png (in <out>/)    where the scan has sampled: every case dir under
                                x0_<x0>/ (corners, reference, VURS picks, which the
                                other plots leave out) as a corner plot of the four
                                knobs, points over their kernel density, VURS picks
                                coloured by the order GENE first wrote them, hollow
                                when GENE has written nothing yet; beside it the Te
                                and ne pedestal of every case and the kymin density;
                                + sampling.csv

VURS picks (every case dir under x0_<x0>/ off the corner levels and not the
reference) are plotted in batches of BATCH in the order GENE first wrote them,
<out>/batch1_16/, <out>/batch17_32/, ...: the same plots as the corners, with
the reference, one line style per pick. The last batch keeps its name and fills
up as picks arrive. The 2x and nz0 comparisons are drawn for the corners only.

Every CSV is written to <out>/data/, beside the plots.

2x-resolution runs are found under <diag-root> (and any sibling <diag-root>_res2,
the root corner_gene_scan.py --res-scale 2 writes) as any directory holding
scanfiles* or parameters_NNNN whose path names a corner tag and carries 2XRES
(any case, also 2x_res, res2x) or _res2, at --x0 (from an x0_<x0> folder in the
path, else the run's parameters). Each is matched to the 1x corner of the same
tag. A 2x ky GENE has not converged (still running, or 0/0 at timelim) gets a
field fit computed here from its field_NNNN, as field_omega.py does it, so the
best guess is there while the runs are going; its verdict sets the marker.

Every legend, table and CSV lists a corner as its knob factors in the order
of its run name, (Teh, Tew, neh, new): (1.2, 0.8, 0.8, 1.2) is
Teh1.20_Tew0.80_neh0.80_new1.20. Corners are sorted in that order too. Colour hue is the width pair (Tew, new), shade
the height pair (Teh, neh); line style is the height pair too (solid,
dashed, dotted, dash-dot) and the marker neh, so the four corners sharing a
hue never share a line style and no corner is told apart by colour alone.

The reference case, 1.00 on every knob (Teh1.00_Tew1.00_neh1.00_new1.00, run
with corner_gene_scan.py --case into the same diag root), is read too when it
is there: listed first as (1, 1, 1, 1) ref, solid black with diamond markers
on every plot, table and CSV. It is no corner, so the corner statistics leave
it out: the mode fractions under the growth plot, the knob-level mode mix,
and the spread and mean of corners_local_params.png.

--case reads named case dirs instead of ped_grid tags (case_point; the NSTX-U
projection's U and U0 from nstxu_scaling_check prep): GENE under
<diag-root>/x0_<X0>/<NAME>/ and profiles under <ped-root>/<NAME>/, each case at
its own x0 (marked in its colour on the rho_tor axes). Same per-scan plots into
<out>/corners/, legends read "NAME (x0)", --ref-case is drawn solid black and
first but stays in the comparisons; no knob mix, batches, sampling or 2x check.

    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --out ./corner_plots
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --spread 0.02 --max-params 16
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.6737 --modes
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --x0 0.8447 --modes --field-fits
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --shot 132588 --x0 0.677 --outliers
    python -m NSTXU_lithium_study.scripts.corners.plots.plot_corners --shot 132543 --diag-root .../132543/nstxu_projection --ped-root .../132543/nstxu_projection_eq --case U0=0.808 U=0.8173 --ref-case U0 --modes --mode-extent

Two growth plots, always:

    corners_growth.png        only what GENE converged (filled markers on each
                              corner's line); every other ky is an x on the zero line
    corners_growth_field.png  the same corners and lines, with each unconverged ky
                              (0/0 or empty omega file) filled in by the field fit
                              field_omega.py --write recorded there (TPED's gamma_field
                              +- gamma_field_std, whatever its verdict): a hollow
                              marker with +-1 std bars in the corner's own colour, line
                              style and marker, joined into its line, so every guess
                              reads as part of its family. 'ambiguous' fits are faded.
                              Skipped when no omega file holds a fit.

With --modes each ky GENE converged (or the field estimate filled) is
classified by TPED's mode_identifier (GENE_sim_analysis: fingerprint transport
ratios from nrg, Epar and parities from field + geometry, omega/omega_star),
drawn with that mode's marker in corners_growth.png, and listed with the
inputs behind each label in corners_modes.csv. The fingerprint diffusivity is
Gamma / omn: where omn ~ 0 (the ne top) the D ratios, and so the ETG, KBM and
ITG/TEM checks, are unreliable; read the inputs before the label.

With --field-fits, a ky that GENE left 0/0 but field_omega.py recorded a
fit for with verdict 'decaying' (gamma + 2 std < 0) or 'ambiguous' (std too
large to tell a slow-growing mode from beating damped ones) carries that fit
in the dataset: 'dec'/'amb' cells in the mode table, gamma_std/omega_std in
corners_modes.csv, and left out of gamma_max when ambiguous. Without the flag
they stay x. A passing fit (status 'field') is the point's gamma either way.

An empty omega file, or one holding gamma = omega = 0 exactly (GENE's
placeholder when a run ends without a converged growth rate: time limit, or
a decaying mode), reads as NaN; a missing omega or parameters file is NaN and
left out. The ky axis is every kymin found across the corners, so scans with
different ky lists combine. A corner with neither GENE output nor profiles is
reported and skipped. Nothing here raises on a partial scan.

delta = n q0 - round(n q0), in [-1/2, 1/2], places x0 between the q = m/n
surfaces of the run's toroidal mode n: 0 on a rational surface, +-1/2 midway.
|delta| is the distance to the nearest one as a fraction of their spacing
x0 / (n shat q0) (in x = rho_tor), which dx_rational in the CSV gives in rho_tor.
A flux tube does not see this offset (ballooning space assumes dense rational
surfaces); it is the local proxy for the q-overlap a global MTM depends on.
n is GENE's n0_global, or round(ky x0 / (q0 rhostar)) when that is absent.

alpha_MHD = q0^2 * major_R * dpdx_pm, all in GENE units (major_R = R0/Lref,
dpdx_pm = -dbeta/dx with x in Lref, as GENE writes it in the run's parameters
file). When dpdx_pm is not there it is rebuilt as
beta * sum_s dens_s * temp_s * (omn_s + omt_s).
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import warnings

if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from TPED.projects.GENE_sim_reader.src.GENE_sim_data import GeneSimulation
from TPED.projects.GENE_sim_reader.src.GENE_field_data import GeneField
from TPED.projects.GENE_sim_reader.src.GENE_vsp_data import GeneVsp
from TPED.projects.GENE_sim_reader.utils.find_GENE_files import GeneFileFinder
from TPED.projects.utils.read_iterdb import read_iterdb
from TPED.projects.discharge_tools.src.filetypes.gfile_data import GFileData
from NSTXU_lithium_study.scripts.corner_gene_scan import (DIAG_ROOT, KYMIN, PED_ROOT, SHOT,
                                                         SHOTS, X0, corners, find_geqdsk,
                                                         shot_defaults, tag, x0_label)
from NSTXU_lithium_study.scripts.ped_grid import KNOBS, parse_tag
from NSTXU_lithium_study.scripts.corners.field_omega import field_omega


def default_out(x0=X0, diag_root=DIAG_ROOT) -> str:
    return os.path.join(diag_root, x0_label(x0), "plots")


#: Legend order of the knobs: the corner tag's (ped_grid.tag, the GENE run and diag dir
#: names), Teh, Tew, neh, new, so a label reads straight across to its run directory.
ORDER = tuple(KNOBS)
LEGEND_TITLE = "(" + ", ".join(ORDER) + ")"

#: Pedestal window of the zoomed profile figure (rho_tor).
PED_ZOOM = (0.6, 1.0)
#: q panels show x0 +- this in rho_tor (5% of the minor radius); the window is
#: shaded on the profile panels above.
Q_ZOOM = 0.05

#: A local parameter is "varying" when (max - min) / max|value| over the corners
#: exceeds this. max|value|, not |mean|: gradients like omn_e cross zero, and a
#: mean near zero would inflate their spread.
SPREAD = 0.05
MAX_PARAMS = 12

#: Local parameters read from each run's parameters file (GENE names; species
#: quantities carry TPED's _<species> suffix, e.g. omn_e, omt_i).
PARAM_KEYS = ("beta", "coll", "debye2", "q0", "shat", "trpeps", "major_R", "minor_r",
              "rhostar", "dpdx_pm", "Tref", "nref", "Bref", "Lref", "nu_ei",
              "nustar_i", "nustar_e")
SPECIES_PARAM = re.compile(r"^(omn|omt|temp|dens)_[A-Za-z]+$")

#: iterdb signal -> (variable, scale to plot units, unit label)
ITERDB_VARS = {"NE": ("ne", 1e-19, "10$^{19}$ m$^{-3}$"), "TE": ("Te", 1e-3, "keV"),
               "NM1": ("ni", 1e-19, "10$^{19}$ m$^{-3}$"), "TI": ("Ti", 1e-3, "keV")}

# Hues for the (Tew, new) width pair -- categorical slots 1, 2, 3, 7 of the
# validated palette; shades step the (Teh, neh) height pair dark to light.
WIDTH_HUES = {(0.8, 0.8): "#2a78d6", (0.8, 1.2): "#eb6834",
              (1.2, 0.8): "#1baf7a", (1.2, 1.2): "#c4338a"}
HEIGHT_SHADE = {(0.8, 0.8): -0.35, (0.8, 1.2): -0.12, (1.2, 0.8): 0.15, (1.2, 1.2): 0.38}
#: Line style per height pair (Teh, neh): solid, dashed, dotted, dash-dot.
HEIGHT_LS = {(0.8, 0.8): (0, ()), (0.8, 1.2): (0, (6, 2.5)), (1.2, 0.8): (0, (1.2, 1.8)),
             (1.2, 1.2): (0, (7, 2, 1.5, 2))}
NEH_MARKER = {0.8: "o", 1.2: "s"}
#: VURS batches: slot i of a batch takes hue i // 4, shade and line style i % 4 (the
#: corner scheme), and this marker per hue, so no two picks in a batch look alike.
SLOT_MARKER = ("o", "s", "^", "v")
GAMMA_CMAP = LinearSegmentedColormap.from_list("gamma", ["#008300", "#f0efec", "#e34948"])
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
#: The 1.00 scaling on every knob: no corner, the reference state the corners are
#: read against. Black, solid, diamonds, so it stands apart from every corner hue.
REF = {k: 1.0 for k in ORDER}
REF_STYLE = {"color": INK, "ls": (0, ()), "marker": "D"}
#: Named cases (case_point, run(cases=...)): explicit case dirs read by name instead of
#: ped_grid tags, e.g. the NSTX-U projection's U and U0 (nstxu_scaling_check prep), GENE
#: under <diag_root>/x0_<x0>/<name>/ and profiles under <ped_root>/<name>/, each at its
#: own x0. Knobs are NaN. The ref case is drawn in REF_STYLE but is still a row of every
#: comparison (local parameters, mode mix), since there are no corners to set it against.
NAMED_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#c4338a", "#6250d6", "#eda100")
NAMED_MARKERS = ("o", "s", "^", "v", "P", "X")
NAMED_LEGEND = "case (x0)"
DPI = 150
#: Which reading of a ky wins when two scanfiles dirs hold it.
RANK = {"ok": 2, "field": 1, "decaying": 0.5, "ambiguous": 0.5}
#: The TPED field fit of a run (GeneOmega), any GENE state; corners_growth_field.png.
FIT_KEYS = ("gamma_field", "gamma_field_std", "omega_field", "omega_field_std")
FIT_NONE = (np.nan,) * len(FIT_KEYS) + ("",)
#: --field-fits: recorded field fits that did not pass, by verdict.
FIT_STYLE = {"decaying": {"mec": "#008300", "alpha": 1.0, "label": "decaying fit (γ+2σ<0) ±1σ"},
             "ambiguous": {"mec": "#9b9a92", "alpha": 0.55,
                           "label": "ambiguous fit (σ too large) ±1σ"}}

#: --modes: TPED mode_identifier label -> marker, and the inputs it decided on,
#: (TPED node, variable) -> column name.
MODE_MARKER = {"ITG_TEM": "o", "ETG": "^", "KBM": "s", "MTM": "v", "unidentified": "*"}
#: Mode colours: validated categorical slots 3, 4, 1, 2 (green, yellow, blue, orange)
#: of the reference palette (ITG_TEM was violet, too close to the KBM blue); yellow
#: is under 3:1 on the surface, so every mode panel also carries its label (legend,
#: table text).
MODE_COLOR = {"ITG_TEM": "#1baf7a", "ETG": "#eda100", "KBM": "#2a78d6", "MTM": "#eb6834",
              "unidentified": "#9b9a92"}
#: Mode table shading: gamma / (largest gamma at that ky) of 0 -> this toward white,
#: 1 -> this toward black (shade()); per ky, since gamma rises steeply with ky.
TABLE_SHADE = (0.82, -0.3)
#: Gaussian width (in ky) of the smoothed mode fraction under the growth plot.
MODE_KDE_BW = 0.12
#: corners_q_offset.png: |delta| / 0.5, dark on a rational surface to light midway
Q_OFFSET_CMAP = LinearSegmentedColormap.from_list("q_offset", ["#5b2a86", "#b58fd0", "#f4f1f8"])
#: corners_mode_extent.png: end/peak above this is flagged as cut off (dashed line,
#: shaded above). A working threshold, not a GENE standard; tighten it if marginal
#: cells change gamma when nx0 grows.
EXTENT_TOL = 0.1
#: ky colour map; end values below the floor are drawn at it (log axis); the left /
#: right ends sit this far either side of their corner's column
EXTENT_CMAP = "viridis"
EXTENT_FLOOR = 1e-6
EXTENT_DX = 0.13
#: corners_vspace_edge.png: edge / peak of <|f|^2>^1/2 above this fails (--vspace-tol).
#: Tighter than EXTENT_TOL: F0 itself is down to exp(-lv^2) ~ 1e-4 at the default lv = 3,
#: so a perturbation still at 1% of its peak there is carried by the box edge. A working
#: threshold like EXTENT_TOL; tighten it if marginal runs change gamma when lv/lw grow.
VSP_TOL = 1e-2
#: the vsp quantity read (TPED GeneVsp name, species suffix added): <|f|^2>^1/2
VSP_QUANT = "vsp_f2_mag"
#: --outliers: thresholds of TPED's outlier analysis, as David's KBM_for_ASTRA criteria
#: set them (TPED QL_models/surrogate/mixing_length_kernel.py); the near-Nyquist
#: fraction keeps TPED's own default (OUTLIER_NYQ)
OUTLIER_KW = {"max_D_over_Chi": 2.0, "max_Chii_over_Chie": 5.0, "max_x0": 0.994}
OUTLIER_NYQ = 0.5
#: its criteria in cell order, and the short label a tripped one gets in its cell
OUTLIER_FLAGS = {"flag_grid_oscillation": "grid", "flag_gamma_negative": "γ<0",
                 "flag_D_over_Chi": "D/χe", "flag_Chii_over_Chie": "χi/χe", "flag_x0": "x0"}
#: corners_outlier.png cell fills: no criterion tripped, any tripped (GAMMA_CMAP's
#: green and red ends lightened 80% and 35% toward white, so the text stays INK)
OUTLIER_FILL = {False: "#cce6cc", True: "#ed8988"}
#: near-Nyquist fractions below this are drawn at it (log axis)
OUTLIER_FLOOR = 1e-8
#: the fields corners_mode_extent.png and corners_mode_structure.png read, and the
#: line colour of each in corners_mode_structure.png (palette slots 1 and 2)
FIELDS = ("field_phi", "field_apar")
STRUCT_COLOR = {"field_phi": "#2a78d6", "field_apar": "#eb6834"}
#: corners_mode_groups.png: two runs whose |phi| (and |Apar|) along z are closer than
#: this can share a group. Distance: earth mover's distance along z over the pair's
#: mean rms width (mode_emd), the largest over the fields both have, so 0.3 = the
#: weight moves 30% of a mode width: scale-free, a narrow ETG and a narrow KBM differ
#: as much as wide ones would. Average linkage, so a group's members are this close
#: on average, not by a chain. Panels per row; points of the common z grid.
GROUP_TOL = 0.3
GROUP_COLS = 4
GROUP_NZ = 801
#: corners_mode_structure.png: width of one eigenfunction box and of the figure, inches
STRUCT_BOX_IN = 0.95
STRUCT_MAX_W = 18
#: corners_mode_structure.png resolution block: the grid of each run (nky0 left out,
#: always 1 in a linear run) and its velocity box lv, lw (the vspace-edge check's
#: knobs), one cell per key (STRUCT_RES_IN wide), shaded by its rank among every run
#: in the plot on the reference palette's ordinal blue ramp (steps 250-700: the
#: lightest still clears 2:1 on the surface); RES_SAME when the plot has one value.
RES_KEYS = ("nx0", "nz0", "nv0", "nw0", "lv", "lw")
STRUCT_RES_IN = 0.36
RES_RAMP = ("#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#184f95", "#104281", "#0d366b")
RES_SAME = "#e4e3df"
#: 2x-resolution run dirs: 2XRES / 2x_res / res2x anywhere in the path, or the _res2
#: suffix of corner_gene_scan.py --res-scale 2 and the dashboard's 2x button
RES2_RE = re.compile(r"(?i)2x_?res|res_?2x|_res2(?!\d)")
TAG_RE = re.compile(r"Teh\d+\.\d+_Tew\d+\.\d+_neh\d+\.\d+_new\d+\.\d+")
X0_RE = re.compile(r"x0_(\d+(?:\.\d+)?)(?![\d.-])")
#: how deep below each root to look for 2x run dirs
RES2_DEPTH = 5
#: corners_res2.png: 2x points sit this fraction of the ky span right of their ky
RES2_NUDGE = 0.012
#: --nz-compare N: the nz0 = N corners are at <diag-root>_nz<N>/x0_<x0>/<tag>/ (e.g.
#: corners_nz64 beside the nz0 = 128 corners)
NZ_SUFFIX = "_nz{}"
#: extra-ky scans (corner_gene_scan --diag-root <diag-root>_kyhi, _kylo, ...) beside
#: the corners: their <tag>/scanfilesNNNN merge by ky with the corner's own. Only
#: this suffix: _nz64, _res2, _test siblings are other grids and must never mix in.
KY_SUFFIX = "_ky"
NOT_KY_RE = re.compile(r"(?i)_nz\d+|_test|2x_?res|res_?2x|_res2(?!\d)")
#: sampling.png: the corner knob levels (anything else off 1.00 is a VURS
#: pick), the knob axis range, the Gaussian KDE widths in knob and kymin units, and
#: the colour map of the VURS sampling order
CORNER_LEVELS = (0.8, 1.2)
SAMPLE_LIM = (0.75, 1.25)
SAMPLE_BW = 0.04
SAMPLE_KY_BW = 0.05
SAMPLE_CMAP = "viridis"
SAMPLE_LINE = "#6250d6"
KYMIN_RE = re.compile(r"^\s*kymin\s*=\s*([-+\d.eE]+)", re.M)
#: --out subfolder of the corner plots; VURS picks go in batch<i>_<j>/ beside it,
#: BATCH per folder in sampling order (16, as many as the corners)
CORNER_DIR = "corners"
BATCH = 16
#: subfolder of --out that every CSV is written to
DATA_DIR = "data"
MODE_SHORT = {"ITG_TEM": "IT", "ETG": "ETG", "KBM": "KBM", "MTM": "MTM", "unidentified": "?"}
MODE_INPUTS = {("fingerprints", "D_e-Chi_e"): "De_Xe", ("fingerprints", "Chi_i-Chi_e"): "Xi_Xe",
               ("fingerprints", "Q_EM_e-|Q_ES|_tot"): "QEM_e", ("epar", "Epar_norm"): "Epar_norm",
               ("epar", "parity_apar"): "parity_apar", ("epar", "parity_phi"): "parity_phi"}


# ----------------------------------------------------------------------------- labels, style

def case_point(name: str, x0=None, ref=False) -> dict:
    """A named case: GENE under <diag_root>/x0_<x0>/<name>/ (x0 None: the scan's x0),
    profiles under <ped_root>/<name>/; *ref* draws it in REF_STYLE, first."""
    return {**{k: np.nan for k in ORDER}, "name": name,
            "x0": None if x0 is None else float(x0), "ref": bool(ref)}


def case_name(p) -> str:
    """The case's dir name: its own for a named case, else the ped_grid tag."""
    return p.get("name") or tag(p)


def is_named(ds) -> bool:
    return bool(ds.attrs.get("named", 0))


def legend_title(ds) -> str:
    return NAMED_LEGEND if is_named(ds) else LEGEND_TITLE


def x0_text(ds) -> str:
    """'x0 = 0.808', or each named case's x0 when they differ."""
    if is_named(ds) and np.unique(ds["x0_case"].values).size > 1:
        return "x0 per case (" + ", ".join(f"{c} {x:g}" for c, x in
                                           zip(ds.corner.values, ds["x0_case"].values)) + ")"
    return f"x0 = {ds.x0:g}"


def x0_marks(ds) -> list[tuple[float, str]]:
    """(x0, colour) to mark on rho_tor axes: one per named case at its own x0, else ds.x0."""
    if not is_named(ds):
        return [(ds.x0, INK)]
    return [(float(x), style(point_of(ds, c))["color"])
            for c, x in zip(ds.corner.values, ds["x0_case"].values)]


def is_ref(p) -> bool:
    """The knob reference (1.00 on every knob). A named case never is: its ref flag
    only styles it, so it stays in the comparisons."""
    if "name" in p:
        return False
    return all(p[k] == REF[k] for k in ORDER)


def label(p) -> str:
    if "name" in p:
        return p["name"] + (f" ({p['x0']:g})" if p.get("x0") is not None else "") \
            + (" ref" if p.get("ref") else "")
    return "(" + ", ".join(f"{p[k]:g}" for k in ORDER) + ")" + (" ref" if is_ref(p) else "")


def ordered(points):
    return sorted(points, key=lambda p: tuple(p[k] for k in ORDER))


def shade(hex_color: str, amount: float):
    """Mix toward black (amount < 0) or white (amount > 0)."""
    c = np.array(to_rgb(hex_color))
    return tuple(c * (1 + amount) if amount < 0 else c + (1 - c) * amount)


def is_corner(p) -> bool:
    return all(p[k] in CORNER_LEVELS for k in ORDER)


def style(p) -> dict:
    if "name" in p:                                # a named case: ref black, others by slot
        if p.get("ref"):
            return dict(REF_STYLE)
        i = int(p.get("slot", 0))
        return {"color": NAMED_COLORS[i % len(NAMED_COLORS)], "ls": (0, ()),
                "marker": NAMED_MARKERS[i % len(NAMED_MARKERS)]}
    if is_ref(p):
        return dict(REF_STYLE)
    if not is_corner(p):                           # a VURS pick: its slot in the batch
        i = int(p.get("slot", 0))
        hue = list(WIDTH_HUES.values())[(i // 4) % 4]
        height = list(HEIGHT_LS)[i % 4]
        return {"color": shade(hue, HEIGHT_SHADE[height]), "ls": HEIGHT_LS[height],
                "marker": SLOT_MARKER[(i // 4) % 4]}
    return {"color": shade(WIDTH_HUES[(p["Tew"], p["new"])], HEIGHT_SHADE[(p["Teh"], p["neh"])]),
            "ls": HEIGHT_LS[(p["Teh"], p["neh"])], "marker": NEH_MARKER[p["neh"]]}


def rerun_style(ds, corner) -> dict:
    """style() of a 1x line in a rerun comparison: diamonds mark the reruns there,
    so the reference's diamond becomes a circle."""
    st = style(point_of(ds, corner))
    return {**st, "marker": "o"} if st["marker"] == "D" else st


def point_of(ds, corner) -> dict:
    p = {k: float(ds[k].sel(corner=corner)) for k in (*ORDER, "slot")}
    if is_named(ds):
        p.update(name=str(corner), x0=float(ds["x0_case"].sel(corner=corner)),
                 ref=bool(ds["case_ref"].sel(corner=corner)))
    return p


def corner_rows(ds) -> np.ndarray:
    """True for the 16 corners, False for the reference."""
    return ~ds["reference"].values


# ----------------------------------------------------------------------------- GENE output

def gene_runs(corner_dir: str) -> list[str]:
    """parameters_NNNN of every scanfiles dir under corner_dir (GeneFileFinder), oldest
    dir first, or []. A later scan (e.g. extra ky) lands in the next scanfilesNNNN."""
    try:
        found = GeneFileFinder(corner_dir).find_files("parameters")
    except FileNotFoundError:
        return []
    # a 2x rerun nested in the corner dir (<tag>/2XRES/...) is not this corner's
    found = [f for f in found if re.search(r"parameters_\d{4}$", f)
             and not RES2_RE.search(os.path.relpath(f, corner_dir))]
    return sorted(found)


def read_run(par_path: str, field_fits: bool = False) -> dict:
    """One scan member through TPED's GeneSimulation: kymin, gamma, omega, status, locals.

    field_fits: GENE's 0/0 with a recorded 'decaying'/'ambiguous' field fit reads
    as that status with the fit's value and std, instead of 'zero'."""
    sim = GeneSimulation(par_path)
    # No omega file at all: skip the load, TPED would log a traceback for the None path.
    tree = sim.return_sim_xarray(["omega"] if sim.omega_filepath else [])
    par = tree["parameters"].ds
    rec = {"kymin": float(par["kymin"]), "gamma": np.nan, "omega": np.nan, "status": "missing",
           "gamma_std": np.nan, "omega_std": np.nan, "fit": FIT_NONE}
    if "omega" in tree.children:
        om = tree["omega"].ds
        if "gamma_field" in om:                     # field_omega.py --write, any run
            rec["fit"] = tuple(float(om[k]) if k in om else np.nan for k in FIT_KEYS) + (
                om.attrs.get("field_verdict", ""),)
        rec["status"] = "ok" if "gamma" in om else "empty"
        if "gamma" in om:
            rec["gamma"], rec["omega"] = float(om["gamma"]), float(om["omega"])
            if om.attrs.get("source") == "field":     # TPED field estimate under GENE's 0/0 line
                rec["status"] = "field"
                rec["gamma_std"] = float(om["gamma_std"]) if "gamma_std" in om else np.nan
                rec["omega_std"] = float(om["omega_std"]) if "omega_std" in om else np.nan
            elif rec["gamma"] == 0 and rec["omega"] == 0:   # GENE's "not converged" placeholder
                rec["gamma"] = rec["omega"] = np.nan
                rec["status"] = "zero"
                verdict = om.attrs.get("field_verdict")
                if field_fits and verdict in FIT_STYLE and "gamma_field" in om:
                    rec["status"] = verdict
                    rec["gamma"], rec["omega"] = float(om["gamma_field"]), float(om["omega_field"])
                    for k in ("gamma", "omega"):
                        rec[f"{k}_std"] = (float(om[f"{k}_field_std"]) if f"{k}_field_std" in om
                                           else np.nan)
    if "n0_global" in par:
        rec["n0"] = float(par["n0_global"])
    else:                                     # GENE's n for nky0 = 1: ky Cy / rhostar, Cy = x0/q0
        try:
            rec["n0"] = float(np.round(rec["kymin"] * float(par["x0"])
                                       / (float(par["q0"]) * float(par["rhostar"]))))
        except (KeyError, ZeroDivisionError):
            rec["n0"] = np.nan
    rec["params"] = {k: float(par[k]) for k in par.data_vars
                     if (k in PARAM_KEYS or SPECIES_PARAM.match(k))
                     and np.issubdtype(par[k].dtype, np.number)}
    return rec


def alpha_mhd(pr: dict) -> float:
    """q0^2 * major_R * dpdx_pm; dpdx_pm rebuilt from beta and the species gradients if absent."""
    dpdx = pr.get("dpdx_pm", np.nan)
    if not np.isfinite(dpdx) or dpdx < 0:      # GENE writes -1/-2 for "compute it" in inputs
        specs = {k.split("_", 1)[1] for k in pr if k.startswith("omn_")}
        try:
            dpdx = pr["beta"] * sum(pr[f"dens_{s}"] * pr[f"temp_{s}"]
                                    * (pr[f"omn_{s}"] + pr[f"omt_{s}"]) for s in specs)
        except KeyError:
            return np.nan
    return pr.get("q0", np.nan) ** 2 * pr.get("major_R", np.nan) * dpdx


def ky_roots(diag_root: str, extra_roots=()) -> list[str]:
    """diag_root, its <diag_root>_ky* siblings (extra-ky scans; no _nz/_res2/_test
    grid), then extra_roots: every root whose x0_<x0>/<tag>/ holds this scan's runs."""
    parent, base = os.path.split(diag_root.rstrip("/\\"))
    roots = [diag_root] + [os.path.join(parent, d) for d in sorted(_listdir(parent))
                           if d.startswith(base + KY_SUFFIX) and not NOT_KY_RE.search(d[len(base):])
                           and os.path.isdir(os.path.join(parent, d))]
    return roots + [r for r in extra_roots if r not in roots]


def gene_corner(corner_dir, field_fits: bool = False):
    """({kymin: (gamma, omega, status, gamma_std, omega_std, parameters path,
    *FIT_KEYS, field verdict, n0)}, local parameters) of one corner, or None.

    corner_dir may be a list: the same corner under several roots (ky_roots),
    read in that order, so a ky run in two roots keeps the better one as below."""
    dirs = [corner_dir] if isinstance(corner_dir, str) else list(corner_dir)
    runs = [r for d in dirs for r in gene_runs(d)]
    if not runs:
        return None
    by_ky: dict[float, tuple] = {}
    params: dict[str, list[float]] = {}
    for path in runs:
        try:
            rec = read_run(path, field_fits)
        except Exception as exc:                # one bad member never sinks the corner
            print(f"    unreadable {path}: {type(exc).__name__}: {exc}")
            continue
        # The same ky in a later scanfiles dir replaces the earlier one unless the
        # earlier is better: GENE converged > TPED field estimate > neither.
        k = round(rec["kymin"], 6)
        if k not in by_ky or RANK.get(rec["status"], 0) >= RANK.get(by_ky[k][2], 0):
            by_ky[k] = (rec["gamma"], rec["omega"], rec["status"], rec["gamma_std"],
                        rec["omega_std"], path, *rec["fit"], rec["n0"])
        for k, v in rec["params"].items():
            params.setdefault(k, []).append(v)
    local = {}
    for k, vals in params.items():                # equilibrium values: identical at every ky
        vals = np.array(vals)
        if np.ptp(vals) > 1e-6 * max(abs(vals).max(), 1e-30):
            print(f"    {os.path.basename(dirs[0])}: {k} differs across ky "
                  f"({vals.min():.4g}..{vals.max():.4g}); using the median")
        local[k] = float(np.median(vals))
    if local:
        local["alpha_MHD"] = alpha_mhd(local)
    return by_ky, local


def classify(par_path: str) -> dict:
    """TPED mode_identifier on one scan member: {'mode', inputs..., 'w_wstar'}."""
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    tped_log = logging.getLogger("TPED")
    level = tped_log.level
    tped_log.setLevel(logging.CRITICAL)             # a missing file is reported once, below
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = GeneSimulationPlus(par_path)
            node = sim.analysis.mode_identifier()
    finally:
        tped_log.setLevel(level)
    out = {"mode": str(node.ds["mode_ID"].values)}
    for (src, var), col in MODE_INPUTS.items():
        ds_ = sim.tree[src].ds
        out[col] = float(np.asarray(ds_[var].values).squeeze()) if var in ds_ else np.nan
    wstar = float(node.ds["omega_star"].values)
    out["w_wstar"] = float(sim.tree["omega"].ds["omega"].values) / wstar if wstar else np.nan
    return out


def add_modes(ds: xr.Dataset) -> xr.Dataset:
    """mode (corner, ky) and its inputs, for every ky with a growth rate (GENE or field)."""
    cols = ["mode", *MODE_INPUTS.values(), "w_wstar"]
    vals = {c: np.full(ds.gamma.shape, "" if c == "mode" else np.nan, dtype=object if c == "mode"
                       else float) for c in cols}
    for i, c in enumerate(ds.corner.values):
        for j, ky in enumerate(ds.ky.values):
            if ds.status.values[i, j] not in ("ok", "field"):
                continue
            try:
                res = classify(str(ds.path.values[i, j]))
            except Exception as exc:              # missing field/geometry: no label, no crash
                msg = str(exc).split(" load_errors")[0].splitlines()[0][:160]
                print(f"    {c} ky={ky:g}: no mode ({type(exc).__name__}: {msg})")
                continue
            for col in cols:
                vals[col][i, j] = res[col]
        labels = [f"{ky:g} {m}" for ky, m in zip(ds.ky.values, vals["mode"][i]) if m]
        print(f"--- {c}: " + ", ".join(labels))
    for col in cols:
        ds[f"mode_{col}" if col != "mode" else "mode"] = (("corner", "ky"), vals[col])
    ds["mode"] = ds["mode"].astype(str)
    return ds


def outlier_check(par_path: str) -> dict:
    """TPED's outlier analysis on one scan member: every criterion's flag, the
    overall 'outlier', the near-Nyquist phi power fraction and the two transport
    ratios it judged (fingerprints D_e/Chi_e, Chi_i/Chi_e)."""
    from TPED.projects.GENE_sim_analysis.src.sim_assembler import GeneSimulationPlus
    tped_log = logging.getLogger("TPED")
    level = tped_log.level
    tped_log.setLevel(logging.CRITICAL)             # a missing file is reported once, below
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sim = GeneSimulationPlus(par_path)
            node = sim.analysis.outlier(**OUTLIER_KW, max_nyquist_fraction=OUTLIER_NYQ)
    finally:
        tped_log.setLevel(level)
    out = {k: bool(node.ds[k].values) for k in (*OUTLIER_FLAGS, "outlier")}
    out["nyquist_fraction"] = float(node.ds["nyquist_fraction"].values)
    fp = sim.tree["fingerprints"].ds
    out["D_over_Chi_e"] = float(fp["D-Chi_e"].values)
    out["Chii_over_Chie"] = float(fp["Chi_i"].values) / float(fp["Chi_e"].values)
    return out


def add_outliers(ds: xr.Dataset) -> xr.Dataset:
    """outlier (corner, ky), each criterion's flag and the numbers behind them, for
    every ky with GENE output, converged or not: a grid-scale oscillation is as
    much a reason a run never converged as a sign a converged one is wrong."""
    flags = [*OUTLIER_FLAGS, "outlier"]
    nums = ["nyquist_fraction", "D_over_Chi_e", "Chii_over_Chie"]
    # checked: the analysis ran; a flag without it reads False and is drawn as no data
    checked = np.zeros(ds.gamma.shape, dtype=bool)
    vals = {c: np.zeros(ds.gamma.shape, dtype=bool) for c in flags}
    vals.update({c: np.full(ds.gamma.shape, np.nan) for c in nums})
    for i, c in enumerate(ds.corner.values):
        for j, ky in enumerate(ds.ky.values):
            path = str(ds.path.values[i, j])
            if not path:
                continue
            try:
                res = outlier_check(path)
            except Exception as exc:              # missing nrg/field: no verdict, no crash
                msg = str(exc).split(" load_errors")[0].splitlines()[0][:160]
                print(f"    {c} ky={ky:g}: no outlier check ({type(exc).__name__}: {msg})")
                continue
            checked[i, j] = True
            for col in flags + nums:
                vals[col][i, j] = res[col]
        hits = [f"{ky:g} " + "+".join(OUTLIER_FLAGS[f] for f in OUTLIER_FLAGS if vals[f][i, j])
                for j, ky in enumerate(ds.ky.values) if vals["outlier"][i, j]]
        print(f"--- {c}: {int(checked[i].sum())} checked, "
              + (", ".join(hits) if hits else "no outliers"))
    ds["outlier_checked"] = (("corner", "ky"), checked)
    for col in flags:
        ds[col if col == "outlier" else f"outlier_{col}"] = (("corner", "ky"), vals[col])
    for col in nums:
        ds[f"outlier_{col}"] = (("corner", "ky"), vals[col])
    ds["outlier"].attrs.update({**OUTLIER_KW, "max_nyquist_fraction": OUTLIER_NYQ})
    return ds


def write_modes_csv(ds, out, written):
    path = data_path(out, "corners_modes.csv")
    cols = ["status", "gamma", "gamma_std", "omega", "omega_std", "mode",
            *(f"mode_{c}" for c in MODE_INPUTS.values()),
            "mode_w_wstar"]
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky"] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, ky in enumerate(ds.ky.values):
                row = [str(ds[k].values[i, j]) if ds[k].dtype.kind in "OUS"
                       else f"{ds[k].values[i, j]:.6g}" for k in cols]
                fh.write(",".join([c, *(f"{ds[k].values[i]:g}" for k in ORDER), f"{ky:g}"] + row)
                         + "\n")
    written.append(path)


# ----------------------------------------------------------------------------- profiles

def iterdb_profiles(case_dir: str):
    """{var: (rho_tor, values in plot units)} from <case_dir>/profiles.iterdb, or None."""
    path = os.path.join(case_dir, "profiles.iterdb")
    if not os.path.isfile(path):
        return None
    with contextlib.redirect_stdout(io.StringIO()):       # read_iterdb prints the point count
        rhot, prof, _ = read_iterdb(path)
    return {name: (np.asarray(rhot[sig]), np.asarray(prof[sig]) * scale)
            for sig, (name, scale, _) in ITERDB_VARS.items() if sig in prof}


def q_profile(case_dir: str):
    """(rho_tor, |q|) of the corner's cheaseBS geqdsk (TPED GFileData), or None."""
    geqdsk, problem = find_geqdsk(case_dir)
    if problem:
        return None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            g = GFileData(os.path.join(case_dir, geqdsk)).gfile_to_xarray()
    except Exception as exc:                      # a bad gfile never sinks the corner
        print(f"    unreadable {geqdsk}: {type(exc).__name__}: {exc}")
        return None
    rho, q = np.asarray(g["rho_tor"].values), np.abs(np.asarray(g["q"].values))
    order = np.argsort(rho)
    return rho[order], q[order]


def integer_crossings(x, y, n):
    """rho where y crosses n (linear between grid points)."""
    d = y - n
    i = np.flatnonzero(np.isfinite(d[:-1]) & np.isfinite(d[1:]) & (np.sign(d[:-1]) != np.sign(d[1:])))
    return x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i])


# ----------------------------------------------------------------------------- dataset

def load_dataset(diag_root: str = DIAG_ROOT, ped_root: str = PED_ROOT, x0=X0,
                 modes=False, field_fits=False, shot=SHOT, points=None,
                 scan="corners", extra_roots=(), outliers=False) -> xr.Dataset:
    """Every corner with GENE output or profiles, as one Dataset (corner, ky, rho_tor).

    corner is the ped_grid tag; Tew/new/Teh/neh ride along as coordinates, so
    ds.set_index(corner=["Tew", "new", "Teh", "neh"]).unstack("corner") gives
    the 4D box. The reference case (REF), when it has GENE output or profiles,
    comes first with reference = True.

    points: the cases to read instead of the corners, in that order (a VURS
    batch), the reference still first; scan names them in titles. slot is each
    case's place in that order (its style when it is no corner).

    GENE runs are read from every ky_roots(diag_root, extra_roots) root, merged by ky.

    points may instead be named cases (case_point): read by name, each at its own
    x0 (x0_case), the ref case first, no knob reference prepended.
    """
    gene, prof, pts = {}, {}, []
    roots = ky_roots(diag_root, extra_roots)
    if len(roots) > 1:
        print("GENE roots: " + ", ".join(os.path.basename(r.rstrip("/\\")) for r in roots))
    named = points is not None and any("name" in p for p in points)
    if named and not all("name" in p for p in points):
        raise ValueError("named cases and ped_grid points cannot be mixed")
    todo = (sorted(points, key=lambda p: not p.get("ref")) if named else
            [REF] + (ordered(corners()) if points is None else list(points)))
    for p in todo:
        name = case_name(p)
        if named and p.get("x0") is None:
            p = {**p, "x0": float(x0[0])}
        cx0 = (p["x0"],) if named else x0
        g = gene_corner([os.path.join(r, x0_label(cx0), name) for r in roots], field_fits)
        pr = iterdb_profiles(os.path.join(ped_root, name))
        q = q_profile(os.path.join(ped_root, name))
        if q is not None:
            pr = {**(pr or {}), "q": q}
        if g is None and pr is None:
            print(f"--- {name}: SKIP, no GENE output and no profiles.iterdb")
            continue
        if g is None:
            print(f"--- {name}: no GENE output (profiles only)")
        if pr is None:
            print(f"--- {name}: no profiles.iterdb (GENE only)")
        if g is not None:
            st = {k: v[2] for k, v in sorted(g[0].items())}
            used = [os.path.basename(r.rstrip("/\\")) for r in roots
                    if any(v[5].startswith(os.path.join(r, "")) for v in g[0].values())]
            print(f"--- {name}: {sum(s == 'ok' for s in st.values())}/{len(st)} ky converged"
                  + (f" (from {', '.join(used)})" if len(roots) > 1 else "")
                  + "".join(f"; ky={k:g} {s}" for k, s in st.items() if s != "ok"))
            gene[name] = g
        if pr is not None:
            prof[name] = pr
        pts.append(p)
    if not pts or (points is not None and not named and all(is_ref(p) for p in pts)):
        raise SystemExit(f"no {'named case' if named else 'corner'} found under "
                         f"{diag_root} or {ped_root}")

    names = [case_name(p) for p in pts]
    ky = np.array(sorted({k for g in gene.values() for k in g[0]}) or KYMIN, dtype=float)

    def on_ky(n, j, fill):
        row = gene.get(n, ({}, {}))[0]
        return [row[k][j] if k in row else fill for k in ky]

    ds = xr.Dataset(coords={"corner": names, "ky": ("ky", ky, {"long_name": "kymin rho_ref"}),
                            **{k: ("corner", [p[k] for p in pts]) for k in ORDER},
                            "label": ("corner", [label(p) for p in pts]),
                            "slot": ("corner", [sum(not is_ref(q) for q in pts[:i])
                                                for i in range(len(pts))]),
                            "reference": ("corner", [is_ref(p) for p in pts]),
                            "x0_case": ("corner", [p["x0"] if named else float(x0[0])
                                                   for p in pts]),
                            "case_ref": ("corner", [bool(p.get("ref")) for p in pts])})
    ds["gamma"] = (("corner", "ky"), [on_ky(n, 0, np.nan) for n in names], {"units": "c_ref/L_ref"})
    ds["omega"] = (("corner", "ky"), [on_ky(n, 1, np.nan) for n in names], {"units": "c_ref/L_ref"})
    ds["status"] = (("corner", "ky"), [on_ky(n, 2, "missing") for n in names])
    ds["gamma_std"] = (("corner", "ky"), [on_ky(n, 3, np.nan) for n in names],
                       {"long_name": "std of the TPED field estimate (status 'field' only)"})
    ds["omega_std"] = (("corner", "ky"), [on_ky(n, 4, np.nan) for n in names])
    ds["path"] = (("corner", "ky"), [on_ky(n, 5, "") for n in names])
    for j, k in enumerate(FIT_KEYS, start=6):      # recorded field fit, whatever GENE did
        ds[k] = (("corner", "ky"), [on_ky(n, j, np.nan) for n in names])
    ds["field_verdict"] = (("corner", "ky"), [on_ky(n, 6 + len(FIT_KEYS), "") for n in names])
    ds["n0"] = (("corner", "ky"), [on_ky(n, 7 + len(FIT_KEYS), np.nan) for n in names],
                {"long_name": "toroidal mode number n (GENE n0_global)"})
    keys = sorted({k for g in gene.values() for k in g[1]})
    for k in keys:
        ds[k] = ("corner", [gene[n][1].get(k, np.nan) if n in gene else np.nan for n in names])
    if "alpha_MHD" in ds:
        ds["alpha_MHD"].attrs["formula"] = "q0^2 * major_R * dpdx_pm (GENE units)"
    if "q0" in ds:                                 # q = m/n surfaces around x0
        nq = ds["n0"] * ds["q0"]
        ds["q_offset"] = (nq - np.round(nq)).assign_attrs(
            long_name="n q0 - round(n q0): 0 on a rational surface, +-1/2 midway")
        if "shat" in ds:
            ds["dx_rational"] = (abs(ds["q_offset"]) * ds["x0_case"]
                                 / (ds["n0"] * ds["shat"] * ds["q0"])).assign_attrs(
                long_name="rho_tor from x0 to the nearest q = m/n surface")
    trusted = ds["gamma"].where(ds["status"] != "ambiguous")   # an ambiguous fit is no maximum
    finite = trusted.notnull().any("ky")
    ds["gamma_max"] = trusted.fillna(-np.inf).max("ky").where(finite)
    ds["ky_at_max"] = xr.DataArray(ky[trusted.fillna(-np.inf).argmax("ky").values],
                                   dims="corner").where(finite)

    if prof:                                       # profiles on the first corner's grid
        ref = next(iter(prof.values()))
        grid = ref.get("ne", next(iter(ref.values())))[0]
        ds = ds.assign_coords(rho_tor=grid)
        for var in [v for v, *_ in ITERDB_VARS.values()] + ["q"]:
            rows = []
            for n in names:
                r, y = prof.get(n, {}).get(var, (None, None))
                rows.append(np.interp(grid, r, y) if r is not None else np.full(grid.size, np.nan))
            arr = np.array(rows)
            if np.isnan(arr).all():
                continue
            ds[var] = (("corner", "rho_tor"), arr)
            if var == "q":
                ds[var].attrs["long_name"] = "|q| from the cheaseBS geqdsk"
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ds[f"omg_{var}"] = (("corner", "rho_tor"),
                                    -np.gradient(arr, grid, axis=1) / arr,
                                    {"long_name": f"-dln{var}/drho_tor"})
    ds.attrs.update(x0=float(ds["x0_case"].values[0]), diag_root=diag_root, ped_root=ped_root,
                    gene_roots="; ".join(roots),
                    field_fits=int(field_fits), shot=int(shot), scan=scan, named=int(named))
    if modes and gene:
        print("\nmode_identifier (TPED) per ky:")
        ds = add_modes(ds)
    if outliers and gene:
        print("\noutlier (TPED) per ky:")
        ds = add_outliers(ds)
    return ds


# ----------------------------------------------------------------------------- plots

def tidy(ax):
    ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def corner_handles(ds):
    return [Line2D([], [], lw=1.8, ms=5, mfc="none", **style(point_of(ds, c)), label=str(lab))
            for c, lab in zip(ds.corner.values, ds.label.values)]


def legend_right(fig, handles, extra=(), title=LEGEND_TITLE):
    fig.legend(handles=list(handles) + list(extra), title=title, loc="center right",
               frameon=False, fontsize=8, title_fontsize=9)


def save(fig, out, name, written):
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)


def data_path(out, name):
    """<out>/data/<name>: every CSV goes in data/ beside the plots."""
    os.makedirs(os.path.join(out, DATA_DIR), exist_ok=True)
    return os.path.join(out, DATA_DIR, name)


def plot_growth(ds, out, written, guesses=False):
    """gamma and omega vs ky, one line per corner through what GENE converged
    (corners_growth.png) or, with *guesses*, also through the recorded field fit
    at each unconverged ky, drawn hollow with +-1 std (corners_growth_field.png)."""
    name = "corners_growth_field.png" if guesses else "corners_growth.png"
    ky = ds.ky.values
    status = ds.status.values
    conv = status == "ok"
    guess = ~conv & np.isfinite(ds["gamma_field"].values) if guesses else np.zeros_like(conv)
    if guesses and not guess.any():
        print(f"no field fit at any unconverged ky (field_omega.py --write); {name} skipped")
        return
    modes = "mode" in ds
    mode = ds["mode"].values if modes else np.full(conv.shape, "", dtype=object)
    verdict = ds["field_verdict"].values
    if modes:
        fig, grid = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True,
                                 height_ratios=(3, 1.5))
        axes, mode_axes = grid[0], grid[1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True)
    unconv = faded = False
    for i, c in enumerate(ds.corner.values):
        st = style(point_of(ds, c))

        def marker(j):
            return MODE_MARKER.get(mode[i, j], "*") if modes and mode[i, j] else st["marker"]

        for ax, var in zip(axes, ("gamma", "omega")):
            y = np.where(conv[i], ds[var].values[i], np.nan)
            y = np.where(guess[i], ds[f"{var}_field"].values[i], y)
            err = ds[f"{var}_field_std"].values[i]
            ok = np.isfinite(y)
            ax.plot(ky[ok], y[ok], lw=1.4, color=st["color"], ls=st["ls"], zorder=3)
            for j in np.flatnonzero(conv[i] & ok):
                ax.plot(ky[j], y[j], marker(j), ms=7 if modes else 5.5, color=st["color"],
                        mec=st["color"], zorder=6)
            for j in np.flatnonzero(guess[i]):
                amb = verdict[i, j] == "ambiguous"
                faded |= bool(amb)
                ax.errorbar(ky[j], y[j], yerr=err[j], fmt=marker(j), ms=7 if modes else 6,
                            mfc=SURFACE, mec=st["color"], mew=1.4, color=st["color"],
                            capsize=2.5, lw=0.9, alpha=0.5 if amb else 1.0, zorder=5)
            bad = ~conv[i] & ~guess[i] & (status[i] != "missing")
            if bad.any():
                unconv = True
                ax.plot(ky[bad], np.zeros(bad.sum()), "x", color=st["color"], ms=9, mew=2)
    if modes:
        mode_strip(mode_axes[0], ds, weighted=False)
        mode_strip(mode_axes[1], ds, weighted=True)
    for ax, var, sym in zip(axes, ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        if not modes:
            ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    what = "GENE converged + field-fit guess" if guesses else "GENE converged only"
    axes[0].set_title(f"growth rate, {what}", loc="left", fontsize=10, color=INK)
    axes[1].set_title(f"real frequency, {what}", loc="left", fontsize=10, color=INK)
    extra = [Line2D([], [], ls="none", marker="o", color=MUTED, ms=6,
                    label="GENE converged (filled)")]
    if guesses:
        extra.append(Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.4,
                            ms=6, label="field-fit guess ±1σ (hollow)"))
        if faded:
            extra.append(Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, mew=1.4,
                                ms=6, alpha=0.5, label="guess with ambiguous fit (faded)"))
    if unconv:
        extra.append(Line2D([], [], ls="none", marker="x", color=INK, ms=8, mew=2,
                            label="unconverged" + (", no fit" if guesses else "")))
    if modes:
        found = [m for m in MODE_MARKER if (mode == m).any()]
        extra += [Line2D([], [], ls="none", marker=MODE_MARKER[m], color=MUTED, ms=7,
                         label=f"mode: {m}") for m in found]
        handles = [Line2D([], [], lw=1.8, color=style(point_of(ds, c))["color"],
                          ls=style(point_of(ds, c))["ls"], label=str(l))
                   for c, l in zip(ds.corner.values, ds.label.values)]
    else:
        handles = [Line2D([], [], lw=1.8, ms=5.5, **style(point_of(ds, c)), label=str(l))
                   for c, l in zip(ds.corner.values, ds.label.values)]
    legend_right(fig, handles, extra, legend_title(ds))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan}, linear GENE at {x0_text(ds)}", x=0.06, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, name, written)


def mode_strip(ax, ds, weighted):
    """Mode make-up at each ky as stacked bars: the fraction of corners in each
    mode (with a kernel-smoothed fraction over ky as lines), or with *weighted*
    each mode's share of the positive growth summed over corners."""
    rows = corner_rows(ds)                         # the reference is no corner
    ky, m, g = ds.ky.values, ds["mode"].values[rows], ds.gamma.values[rows]
    found = [x for x in MODE_COLOR if (m == x).any()]
    width = 0.6 * (np.diff(ky).min() if ky.size > 1 else 0.2)
    bottom = np.zeros(ky.size)
    gpos = np.where(np.isfinite(g) & (m != ""), np.clip(np.nan_to_num(g), 0, None), 0.0)
    total = gpos.sum(axis=0)
    for mode in found:
        if weighted:
            h = np.divide((gpos * (m == mode)).sum(axis=0), total,
                          out=np.zeros(ky.size), where=total > 0)
        else:
            h = (m == mode).sum(axis=0) / m.shape[0]
        ax.bar(ky, h, width, bottom=bottom, color=MODE_COLOR[mode], edgecolor=SURFACE, lw=1,
               label=mode, zorder=3)
        bottom += h
    if not weighted and found:
        x = np.linspace(ky.min() - 2 * MODE_KDE_BW, ky.max() + 2 * MODE_KDE_BW, 400)
        # Kernel-weighted average of each ky's fraction (Nadaraya-Watson): stays in
        # 0-1 however closely the ky are spaced, unlike a summed KDE.
        w = np.exp(-0.5 * ((x[None] - ky[:, None]) / MODE_KDE_BW) ** 2)
        for mode in found:
            frac = (m == mode).sum(axis=0) / m.shape[0]
            ax.plot(x, (w * frac[:, None]).sum(axis=0) / w.sum(axis=0), color=MODE_COLOR[mode],
                    lw=2, zorder=4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction of corners")
        ax.set_title(f"mode fraction per ky (bars), kernel-smoothed over ky (lines, bw "
                     f"{MODE_KDE_BW:g})", loc="left", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7.5, ncols=len(found), loc="upper left",
                  bbox_to_anchor=(0, -0.28))
    else:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(r"share of $\Sigma\gamma_{>0}$")
        ax.set_title("each mode's share of the growth at each ky", loc="left", fontsize=9,
                     color=INK)
    ax.set_xlabel(r"$k_y\rho_{ref}$")
    tidy(ax)


def table_colour(mode, frac):
    """The mode's colour, light for frac 0 to dark for frac 1 (TABLE_SHADE)."""
    lo, hi = TABLE_SHADE
    f = float(np.clip(frac, 0, 1)) if np.isfinite(frac) else 0.0
    return shade(MODE_COLOR.get(mode, MODE_COLOR["unidentified"]), lo + (hi - lo) * f)


def plot_mode_scaling(ds, out, written):
    """Mode at every (corner, ky) as a table shaded by gamma, and per knob level the
    mode make-up of all labelled ky: how each profile scaling moves the mode mix.

    A cell is its mode's colour, from light (gamma near 0) to dark (the largest
    gamma of any corner at that ky), with gamma printed under the mode, so a
    column reads as which corners, and so which knobs, drive that ky hardest."""
    m, ky = ds["mode"].values, ds.ky.values
    status = ds.status.values
    gam = ds.gamma.values
    trusted = np.where((m != "") & np.isfinite(gam), gam, np.nan)
    with np.errstate(invalid="ignore"):
        col_max = np.nanmax(np.where(trusted > 0, trusted, np.nan), axis=0)
    n = ds.sizes["corner"]
    found = [x for x in MODE_COLOR if (m == x).any()]
    fig, (tab, bars) = plt.subplots(1, 2, figsize=(16, max(5.5, 0.34 * n + 2.2)),
                                    width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            mode = m[i, j]
            fc = (table_colour(mode, trusted[i, j] / col_max[j]) if mode else
                  {"decaying": "#dcefdc", "ambiguous": "#ecebe6"}.get(status[i, j], SURFACE))
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            text = MODE_SHORT.get(mode, mode) if mode else {
                "zero": "x", "empty": "x", "decaying": "dec", "ambiguous": "amb"}.get(
                status[i, j], "-")
            if mode and np.isfinite(gam[i, j]):
                text += f"\n{gam[i, j]:.3f}"
            dark = mode and np.mean(to_rgb(fc)) < 0.55
            tab.text(j + 0.5, n - 0.5 - i, text, ha="center", va="center", fontsize=6.5,
                     linespacing=1.1, color="white" if dark else INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(f"mode and gamma per corner and ky, rows {legend_title(ds)}  (x unconverged, - none"
                  + (", dec/amb recorded fit" if ds.attrs.get("field_fits") else "") + ")\n"
                  "shade: gamma / largest gamma at that ky, light 0 to dark 1",
                  loc="left", fontsize=10, color=INK)
    for k, mode in enumerate(found):              # light-to-dark key per mode, under the table
        for q in range(5):
            tab.add_patch(plt.Rectangle((k * 1.9 + q * 0.3, -1.95), 0.3, 0.45,
                                        fc=table_colour(mode, q / 4), ec="none", clip_on=False))
        tab.text(k * 1.9 + 1.6, -1.72, MODE_SHORT.get(mode, mode), fontsize=7, va="center", color=INK,
                 clip_on=False)

    if is_named(ds):                              # no knobs: the table is the whole story
        bars.axis("off")
        fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: mode per case",
                     x=0.02, ha="left", fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, out, "corners_mode_scaling.png", written)
        return
    groups = [(k, side) for k in ORDER for side in ("<1", ">1")]
    side = {g: (ds[g[0]].values < 1) if g[1] == "<1" else (ds[g[0]].values > 1) for g in groups}
    xs = np.array([i + 0.6 * (i // 2) for i in range(len(groups))])
    bottom = np.zeros(len(groups))
    labelled = [(m[side[g]] != "").sum() for g in groups]
    rows = []
    for mode in found:
        frac = np.array([(m[side[g]] == mode).sum() / max(t, 1)
                         for g, t in zip(groups, labelled)])
        bars.bar(xs, frac, 0.8, bottom=bottom, color=MODE_COLOR[mode], edgecolor=SURFACE,
                 lw=1, label=mode, zorder=3)
        for x, f, b in zip(xs, frac, bottom):
            if f >= 0.08:
                bars.text(x, b + f / 2, f"{100 * f:.0f}%", ha="center", va="center", fontsize=7,
                          zorder=4,
                          color="white" if np.mean(to_rgb(MODE_COLOR[mode])) < 0.55 else INK)
        bottom += frac
        rows.append((mode, frac))
    corner_scan = all(is_corner(point_of(ds, c)) or is_ref(point_of(ds, c))
                      for c in ds.corner.values)
    level = {"<1": "0.8", ">1": "1.2"} if corner_scan else {"<1": "<1", ">1": ">1"}
    bars.set_xticks(xs, [f"{k}\n{level[v]}" for k, v in groups], fontsize=8)
    bars.set_ylim(0, 1.02)
    bars.set_ylabel("share of labelled ky")
    bars.set_title("mode mix at each knob level (all other knobs pooled)", loc="left",
                   fontsize=10, color=INK)
    bars.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(bars)
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: profile scaling vs mode",
                 x=0.02, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "corners_mode_scaling.png", written)

    path = data_path(out, "corners_mode_scaling.csv")
    with open(path, "w") as fh:
        fh.write("knob,level,labelled_ky," + ",".join(found) + "\n")
        for g_i, ((k, v), t) in enumerate(zip(groups, labelled)):
            fh.write(f"{k},{level[v]},{t}," + ",".join(f"{fr[g_i]:.3f}" for _, fr in rows) + "\n")
    written.append(path)


def plot_q_offset(ds, out, written):
    """delta = n q0 - round(n q0) per corner and ky as a table (dark on a rational
    surface, light midway), and gamma vs |delta| beside it; + corners_q_offset.csv."""
    if "q_offset" not in ds or not np.isfinite(ds.q_offset.values).any():
        print("no n0_global/q0 in the runs' parameters; corners_q_offset.png skipped")
        return
    d, n0, gam, ky = ds.q_offset.values, ds.n0.values, ds.gamma.values, ds.ky.values
    m = ds["mode"].values if "mode" in ds else np.full(d.shape, "", dtype=object)
    conv = ds.status.values == "ok"
    n = ds.sizes["corner"]
    fig, (tab, sc) = plt.subplots(1, 2, figsize=(16, max(5.5, 0.34 * n + 2.2)),
                                  width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            if not np.isfinite(d[i, j]):
                tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=SURFACE, ec=SURFACE, lw=2))
                tab.text(j + 0.5, n - 0.5 - i, "-", ha="center", va="center", fontsize=6.5,
                         color=MUTED)
                continue
            fc = Q_OFFSET_CMAP(abs(d[i, j]) / 0.5)
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec=SURFACE, lw=2))
            if m[i, j]:                            # mode band along the top, MODE_COLOR
                tab.add_patch(plt.Rectangle((j + 0.04, n - 1 - i + 0.8), 0.92, 0.16, ec="none",
                                            fc=MODE_COLOR.get(m[i, j], MODE_COLOR["unidentified"])))
            text = f"{d[i, j]:+.2f}\nn={n0[i, j]:g}"
            if np.isfinite(gam[i, j]):
                mode = f"{MODE_SHORT.get(m[i, j], m[i, j])} " if m[i, j] else ""
                text += f"\n{mode}{gam[i, j]:.3f}"
            tab.text(j + 0.5, n - 0.55 - i, text, ha="center", va="center", fontsize=6,
                     linespacing=1.05, color="white" if np.mean(fc[:3]) < 0.55 else INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(r"$\delta = nq_0 - \mathrm{round}(nq_0)$ per corner and ky, rows "
                  f"{legend_title(ds)}\ncell: delta, n, [mode] gamma, top band the mode's colour; "
                  "dark = on a q = m/n surface, light = midway", loc="left", fontsize=10, color=INK)
    for q in range(6):                             # dark-to-light key under the table
        tab.add_patch(plt.Rectangle((q * 0.35, -1.95), 0.35, 0.45, fc=Q_OFFSET_CMAP(q / 5),
                                    ec="none", clip_on=False))
    tab.text(2.2, -1.72, "|delta| 0 to 0.5", fontsize=7, va="center", color=INK, clip_on=False)
    for k, mode in enumerate(x for x in MODE_COLOR if (m == x).any()):   # mode band key
        tab.add_patch(plt.Rectangle((3.6 + k * 1.3, -1.85), 0.3, 0.25, fc=MODE_COLOR[mode],
                                    ec="none", clip_on=False))
        tab.text(3.95 + k * 1.3, -1.72, MODE_SHORT.get(mode, mode), fontsize=7, va="center",
                 color=INK, clip_on=False)

    # each point its mode's marker and colour (as in the growth and mode plots), ky beside it
    for i, j in zip(*np.nonzero(conv & np.isfinite(d))):
        mode = m[i, j]
        mk = MODE_MARKER.get(mode, "*") if mode else "o"
        c = MODE_COLOR.get(mode, MODE_COLOR["unidentified"]) if mode else MUTED
        sc.plot(abs(d[i, j]), gam[i, j], mk, color=c, ms=7, mec=INK, mew=0.4, zorder=3)
        sc.annotate(f"{ky[j]:g}", (abs(d[i, j]), gam[i, j]), xytext=(4, 3),
                    textcoords="offset points", fontsize=6, color=MUTED)
    for mode in [x for x in MODE_MARKER if (m == x).any()]:
        sc.plot([], [], MODE_MARKER[mode], color=MODE_COLOR[mode], mec=INK, mew=0.4, ms=7,
                label=mode)
    if not (m != "").any():
        sc.plot([], [], "o", color=MUTED, ms=7, label="no mode (run --modes)")
    sc.axhline(0, color=MUTED, lw=0.8)
    sc.set_xlim(-0.01, 0.51)
    sc.set_xlabel(r"$|\delta|$  (0 on a rational surface, 0.5 midway)")
    sc.set_ylabel(r"$\gamma$ [$c_{ref}/L_{ref}$]")
    sc.set_title("GENE-converged gamma vs rational-surface offset (number: ky)", loc="left",
                 fontsize=10,
                 color=INK)
    sc.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    tidy(sc)
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: q = m/n overlap", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "corners_q_offset.png", written)

    path = data_path(out, "corners_q_offset.csv")
    cols = [c for c in ("n0", "q0", "shat", "q_offset", "dx_rational", "status", "gamma", "mode")
            if c in ds]
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky"] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, k in enumerate(ky):
                row = []
                for col in cols:
                    v = ds[col].values[i, j] if ds[col].ndim == 2 else ds[col].values[i]
                    row.append(str(v) if ds[col].dtype.kind in "OUS" else f"{v:.6g}")
                fh.write(",".join([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}"] + row)
                         + "\n")
    written.append(path)


def plot_outliers(ds, out, written):
    """TPED's outlier verdict per corner and ky as a table (red where any criterion
    trips, the near-Nyquist fraction and the tripped criteria in the cell), and the
    near-Nyquist fraction vs ky per corner beside it; + corners_outlier.csv."""
    checked, bad = ds.outlier_checked.values, ds.outlier.values
    if not checked.any():
        print("no run passed the outlier analysis; corners_outlier.png skipped")
        return
    nyq, st, ky = ds.outlier_nyquist_fraction.values, ds.status.values, ds.ky.values
    st_short = {"ok": "", "field": "field", "zero": "0/0", "empty": "empty",
                "decaying": "dec", "ambiguous": "amb"}
    n = ds.sizes["corner"]
    fig, (tab, sc) = plt.subplots(1, 2, figsize=(18, max(5.5, 0.34 * n + 2.2)),
                                  width_ratios=(1.15, 1))
    for i in range(n):
        for j in range(ky.size):
            if not checked[i, j]:
                tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=SURFACE, ec=SURFACE, lw=2))
                tab.text(j + 0.5, n - 0.5 - i, "-", ha="center", va="center", fontsize=6.5,
                         color=MUTED)
                continue
            tab.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, fc=OUTLIER_FILL[bool(bad[i, j])],
                                        ec=SURFACE, lw=2))
            tripped = "+".join(lab for f, lab in OUTLIER_FLAGS.items()
                               if ds[f"outlier_{f}"].values[i, j])
            text = "\n".join(x for x in (f"{nyq[i, j]:.1e}", tripped, st_short.get(st[i, j], st[i, j]))
                             if x)
            tab.text(j + 0.5, n - 0.5 - i, text, ha="center", va="center", fontsize=6,
                     linespacing=1.05, color=INK)
    tab.set_xlim(0, ky.size)
    tab.set_ylim(0, n)
    tab.set_xticks(np.arange(ky.size) + 0.5, [f"{k:g}" for k in ky], fontsize=8)
    tab.set_yticks(np.arange(n) + 0.5, [str(l) for l in ds.label.values[::-1]], fontsize=8)
    tab.set_xlabel(r"$k_y\rho_{ref}$")
    tab.tick_params(length=0)
    for sp in tab.spines.values():
        sp.set_visible(False)
    tab.set_title(f"TPED outlier per corner and ky, rows {legend_title(ds)}; red = outlier\n"
                  "cell: near-Nyquist fraction, tripped criteria, GENE status if not "
                  "converged", loc="left", fontsize=10, color=INK)
    crit = (f"grid: fraction > {OUTLIER_NYQ:g}   γ<0: GENE gamma < 0   "
            f"D/χe > {OUTLIER_KW['max_D_over_Chi']:g}   "
            f"χi/χe > {OUTLIER_KW['max_Chii_over_Chie']:g}   x0 > {OUTLIER_KW['max_x0']:g}")
    tab.text(0, -1.3, crit, fontsize=7.5, va="center", color=INK, clip_on=False)

    # every checked run, converged (filled) or not (hollow), in its corner's style
    for i, c in enumerate(ds.corner.values):
        sty = style(point_of(ds, c))
        ok = checked[i]
        if not ok.any():
            continue
        y = np.maximum(nyq[i], OUTLIER_FLOOR)
        sc.plot(ky[ok], y[ok], color=sty["color"], ls=sty["ls"], lw=1.3, zorder=2)
        conv = ok & (st[i] == "ok")
        sc.plot(ky[conv], y[conv], sty["marker"], color=sty["color"], ms=5, zorder=3)
        sc.plot(ky[ok & ~conv], y[ok & ~conv], sty["marker"], color=sty["color"], mfc="none",
                ms=5, zorder=3)
    sc.axhline(OUTLIER_NYQ, color="#e34948", ls=(0, (5, 3)), lw=1.2, zorder=1)
    sc.text(ky.min(), OUTLIER_NYQ * 1.3, f"outlier threshold {OUTLIER_NYQ:g}", fontsize=7.5,
            color="#e34948")
    sc.set_yscale("log")
    sc.set_xscale("log")
    sc.set_xticks(ky, [f"{k:g}" for k in ky])
    sc.minorticks_off()
    sc.set_xlabel(r"$k_y\rho_{ref}$")
    sc.set_ylabel("fraction of phi power above 0.9 x Nyquist in z")
    sc.set_title("near-Nyquist fraction (last field time, whole ballooning domain)\n"
                 "filled = GENE converged, hollow = not", loc="left", fontsize=10, color=INK)
    tidy(sc)
    legend_right(fig, corner_handles(ds), title=legend_title(ds))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: outlier check "
                 f"({int(bad.sum())} of {int(checked.sum())} runs flagged)", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.02, 0.87, 0.95))
    save(fig, out, "corners_outlier.png", written)

    path = data_path(out, "corners_outlier.csv")
    cols = ["status", "gamma", "outlier_checked", "outlier",
            *(f"outlier_{f}" for f in OUTLIER_FLAGS),
            "outlier_nyquist_fraction", "outlier_D_over_Chi_e", "outlier_Chii_over_Chie"]
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky"] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, k in enumerate(ky):
                row = [str(ds[col].values[i, j]) if ds[col].dtype.kind in "OUSb"
                       else f"{ds[col].values[i, j]:.6g}" for col in cols]
                fh.write(",".join([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}"] + row)
                         + "\n")
    written.append(path)


def mode_profile(par_path: str) -> dict:
    """{"zgrid": ballooning angle / pi, field: |f| / max|f|} on the ballooning grid
    (nx0 nz0 points), last field time; a field the run does not have is left out."""
    field = re.sub(r"parameters_(\d{4})$", r"field_\1", par_path)
    if not os.path.isfile(field) or os.path.getsize(field) == 0:
        return {}
    fx = GeneField(field).field_filepath_to_xarray("last", list(FIELDS))
    if fx is None:
        return {}
    out = {"zgrid": fx.zgrid.values}
    for name in FIELDS:
        if name in fx:
            f = np.abs(fx[name].values[-1])
            if f.max() > 0:
                out[name] = f / f.max()
    return out


def mode_ends(prof: dict) -> dict:
    """{field: (|f| at the left z end, |f| at the right z end) / max|f|}; NaN for a
    field *prof* (mode_profile) does not have."""
    return {f: (float(prof[f][0]), float(prof[f][-1])) if f in prof else (np.nan, np.nan)
            for f in FIELDS}


def load_profiles(ds) -> tuple[dict, set]:
    """{(corner index, ky index): mode_profile} for every run with field output,
    and the nx0 values those runs used. Reads every field file once, for both
    corners_mode_extent.png and corners_mode_structure.png."""
    profiles, nx0 = {}, set()
    for i in range(ds.sizes["corner"]):
        for j in range(ds.sizes["ky"]):
            path = str(ds.path.values[i, j])
            if not path:
                continue
            try:
                prof = mode_profile(path)
            except Exception as exc:                # one unreadable field never sinks the plot
                print(f"    field unreadable {path}: {type(exc).__name__}: {exc}")
                continue
            if any(f in prof for f in FIELDS):
                profiles[i, j] = prof
                nx0.add(run_param(path, "nx0"))
    return profiles, nx0


def dodge(x, w) -> np.ndarray:
    """Centres for boxes of width *w* at sorted positions *x*, moved as little as
    possible so none overlap: touching boxes form a group, spaced *w* apart and
    centred on the mean of their positions, until no two groups overlap."""
    groups = [[v] for v in x]                     # each group: its members' positions
    def first(g):
        return np.mean(g) - w * (len(g) - 1) / 2
    merged = True
    while merged:
        merged = False
        for k in range(len(groups) - 1):
            if first(groups[k]) + w * len(groups[k]) > first(groups[k + 1]) + 1e-12:
                groups[k:k + 2] = [groups[k] + groups[k + 1]]
                merged = True
                break
    return np.array([first(g) + w * q for g in groups for q in range(len(g))])


def res_colors(values) -> dict:
    """{value: colour} for one resolution key: the distinct values ranked, lowest
    on the light end of RES_RAMP, highest on the dark end; RES_SAME for one value."""
    vals = sorted({v for v in values if v is not None})
    if len(vals) < 2:
        return {v: RES_SAME for v in vals}
    return {v: RES_RAMP[round(k * (len(RES_RAMP) - 1) / (len(vals) - 1))]
            for k, v in enumerate(vals)}


def structure_rows(ds, profiles) -> list[tuple]:
    """(corner index, ky indices, RES_KEYS values) per row of corners_mode_structure.png:
    one row per case, split into one row per grid when its runs differ (an extra-ky
    scan at another resolution), rows of a case in order of their lowest ky."""
    rows = []
    for i in range(ds.sizes["corner"]):
        groups: dict[tuple, list[int]] = {}
        for j in range(ds.sizes["ky"]):
            if (i, j) in profiles:
                path = str(ds.path.values[i, j])
                groups.setdefault(tuple(run_param(path, k) for k in RES_KEYS), []).append(j)
        rows += sorted(((i, js, res) for res, js in groups.items()), key=lambda t: t[1][0])
    return rows


def plot_mode_structure(ds, out, written, profiles, tol=EXTENT_TOL):
    """|phi| and |Apar| along the field line per case and ky: one row per case,
    each run's ballooning eigenfunction a small box centred at its kymin on one
    log ky axis, so a case only draws the ky it ran (VURS picks each run their own
    set) and a row shows how the mode changes with ky. Each field over its own
    peak, last field time, whole z domain. Boxes are STRUCT_BOX_IN wide; where two
    ky of a row sit closer the boxes are pushed apart, and a caret under each box
    marks its true kymin (a leader joins them). Border: the cell's mode colour as
    in corners_mode_scaling.png (under --modes; grey without), dashed when an end
    is above *tol* (cut off, grow nx0; red dashed if no mode). Grey box: not
    converged (the last snapshot, not a settled mode). Corner text: mode (under
    --modes) and gamma.

    Left of each row, its grid (RES_KEYS), each key shaded by rank among all runs
    in the plot (res_colors), so a coarse run stands out. A case whose runs used
    different grids splits into one row per grid (structure_rows); each of those
    rows is boxed, block and runs, so the runs on each grid read at a glance."""
    ky = ds.ky.values
    rows = structure_rows(ds, profiles)
    if not rows:
        print("no field files; corners_mode_structure.png skipped")
        return
    n = ds.sizes["corner"]
    m = ds["mode"].values if "mode" in ds else np.full((n, ky.size), "", dtype=object)
    gam, status = ds.gamma.values, ds.status.values
    split = {i for i, _, _ in rows if sum(i == i2 for i2, _, _ in rows) > 1}
    shades = [res_colors(res[k] for _, _, res in rows) for k in range(len(RES_KEYS))]

    # kymin of each box in decades; boxes too close in a row are pushed apart (dodge)
    lk = {r: np.log10([ky[j] for j in js]) for r, (_, js, _) in enumerate(rows)}
    gaps = [g for x in lk.values() for g in np.diff(x)]
    span = max(np.ptp(np.concatenate(list(lk.values()))), 0.3)
    # the eigenfunction axes as before the block (0.15-0.98 of w0), the block added
    # between the row labels and them
    w0 = float(np.clip((span / min(gaps, default=span) + 1) * STRUCT_BOX_IN / 0.83 + 0.5,
                       9, STRUCT_MAX_W))
    blk_in = STRUCT_RES_IN * len(RES_KEYS)
    width = w0 + blk_in + 0.15
    label_in, ax_in = 0.15 * w0, 0.83 * w0
    left = (label_in + blk_in + 0.15) / width
    right = left + ax_in / width
    box = STRUCT_BOX_IN * (span + 0.1) / (ax_in - STRUCT_BOX_IN)   # one box, in decades
    at = {r: dodge(x, 1.06 * box) for r, x in lk.items()}           # box centres
    lo = min(x.min() for x in at.values()) - box / 2 - 0.05
    hi = max(x.max() for x in at.values()) + box / 2 + 0.05
    height = max(5.8, 0.62 * len(rows) + 2.5)        # inches; the bottom 0.85 holds two legend rows
    fig = plt.figure(figsize=(width, height))
    bottom = 0.85 / height + 0.04
    top = 1 - 1.3 / height                            # three title lines, then the RES_KEYS heads
    base = fig.add_axes((left, bottom, right - left, top - bottom))
    base.set_xscale("log")
    base.set_xlim(10 ** lo, 10 ** hi)
    base.set_ylim(len(rows), 0)
    base.set_yticks([])

    # resolution block: row labels on its left, the RES_KEYS as column heads
    blk = fig.add_axes((label_in / width, bottom, blk_in / width, top - bottom))
    blk.set_xlim(0, len(RES_KEYS))
    blk.set_ylim(len(rows), 0)
    blk.set_yticks([r + 0.5 for r in range(len(rows))],
                   [str(ds.label.values[i]) if r == 0 or rows[r - 1][0] != i else ""
                    for r, (i, _, _) in enumerate(rows)], fontsize=8, color=INK)
    blk.set_xticks([k + 0.5 for k in range(len(RES_KEYS))], RES_KEYS, fontsize=7, color=INK)
    blk.xaxis.tick_top()
    blk.tick_params(length=0)
    for sp in blk.spines.values():
        sp.set_visible(False)
    for r, (i, js, res) in enumerate(rows):
        for k, v in enumerate(res):
            c = shades[k].get(v, SURFACE)
            blk.add_patch(plt.Rectangle((k + 0.06, r + 0.12), 0.88, 0.76, fc=c, ec="none"))
            dark = c in RES_RAMP[4:]
            blk.text(k + 0.5, r + 0.5, "–" if v is None else f"{v:g}", ha="center", va="center",
                     fontsize=6.5, color="white" if dark else INK)
        if i in split:                                # box block and runs of this grid
            blk.add_patch(plt.Rectangle((0.01, r + 0.04), len(RES_KEYS) - 0.02, 0.92, fc="none",
                                        ec=INK, lw=1.0))
            x0b, x1b = 10 ** (at[r].min() - box / 2 - 0.02), 10 ** (at[r].max() + box / 2 + 0.02)
            base.add_patch(plt.Rectangle((x0b, r + 0.04), x1b - x0b, 0.92, fc="none", ec=INK,
                                         lw=1.0, zorder=0.5))
    base.tick_params(axis="y", length=0)
    base.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    base.xaxis.set_minor_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 3, 5) else ""))
    base.tick_params(axis="x", which="both", labelsize=8, colors=INK)
    base.grid(axis="x", which="both", color=GRID, lw=0.6)
    for r in range(1, len(rows)):                     # dotted between rows of one case
        same = rows[r][0] == rows[r - 1][0]
        for a in (base, blk):
            a.axhline(r, color=GRID, lw=0.6, ls=(0, (1, 2)) if same else "-")
    for side in ("top", "right", "left"):
        base.spines[side].set_visible(False)
    base.set_xlabel(r"$k_y\rho_{ref}$  (each box at its kymin: ballooning angle $z/\pi$ over the "
                    "whole domain, last field time)", fontsize=9, color=INK)

    n_cut = 0
    for r, (i, js, _) in enumerate(rows):
        for q, j in enumerate(js):
            x, xb = lk[r][q], at[r][q]
            # caret at the true kymin under the box, with a leader when the box was pushed
            base.plot(10 ** x, r + 0.985, "^", ms=3.5, color=MUTED, zorder=1, clip_on=False)
            if abs(xb - x) > 0.01:
                base.plot([10 ** xb, 10 ** x], [r + 0.93, r + 0.97], color=MUTED, lw=0.6)
            u = (xb - lo) / (hi - lo)
            ax = base.inset_axes((u - box / (hi - lo) / 2, 1 - (r + 0.93) / len(rows),
                                  box / (hi - lo), 0.86 / len(rows)))
            ax.set_xticks([])
            ax.set_yticks([])
            prof = profiles[i, j]
            z = prof["zgrid"]
            for f, colour in STRUCT_COLOR.items():
                if f in prof:
                    ax.plot(z, prof[f], color=colour, lw=0.9)
            ax.set_xlim(z[0], z[-1])
            ax.set_ylim(0, 1.08)
            ax.set_facecolor(SURFACE if status[i, j] == "ok" else "#ecebe6")
            cut = max(np.nanmax(e) for e in mode_ends(prof).values()) > tol
            n_cut += cut
            # border: the mode's colour as in corners_mode_scaling.png; dashed = cut off
            edge = MODE_COLOR.get(m[i, j], MODE_COLOR["unidentified"]) if m[i, j] else (
                "#e34948" if cut else GRID)
            for sp in ax.spines.values():
                sp.set_color(edge)
                sp.set_linewidth(1.8 if m[i, j] or cut else 0.8)
                sp.set_linestyle((0, (3, 1.5)) if cut else "-")
            text = MODE_SHORT.get(m[i, j], m[i, j]) if m[i, j] else ""
            if np.isfinite(gam[i, j]):
                text += f" {gam[i, j]:.3f}"
            if text:
                ax.text(0.03, 0.95, text.strip(), ha="left", va="top", fontsize=5.5, color=INK,
                        transform=ax.transAxes)
    handles = [Line2D([], [], color=STRUCT_COLOR["field_phi"], lw=1.5, label=r"$|\phi|$ / peak"),
               Line2D([], [], color=STRUCT_COLOR["field_apar"], lw=1.5,
                      label=r"$|A_\parallel|$ / peak"),
               plt.Rectangle((0, 0), 1, 1, fc=SURFACE, ec=INK, lw=1.8, ls=(0, (3, 1.5)),
                             label=f"dashed: end / peak above {tol:g} (cut off, grow nx0)"),
               plt.Rectangle((0, 0), 1, 1, fc="#ecebe6", ec=GRID, label="not converged")]
    handles += [plt.Rectangle((0, 0), 1, 1, fc=SURFACE, ec=MODE_COLOR[x], lw=1.8, label=x)
                for x in MODE_COLOR if (m == x).any()]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="lower center",
               ncol=min(len(handles), 5), bbox_to_anchor=(0.56, 0.0))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: mode structure along the field line,"
                 f" rows {legend_title(ds)}\n{n_cut} box(es) with an end above {tol:g}; "
                 "text: mode (under --modes) and gamma\nleft: each row's grid, darker = "
                 "higher among the runs here (grey: one value), boxed where a case's runs "
                 "differ", x=0.02, ha="left", fontsize=11,
                 color=INK)
    save(fig, out, "corners_mode_structure.png", written)


def mode_width(f, z) -> float:
    """rms width (z/pi) of |f| along *z*, taken as a distribution."""
    mu = (z * f).sum() / f.sum()
    return float(np.sqrt(((z - mu) ** 2 * f).sum() / f.sum()))


def mode_emd(a: dict, b: dict, z) -> float:
    """Distance between two mode_profile shapes on the common grid *z*: each field
    a distribution of |f| along z, earth mover's distance between the two over
    their mean rms width (mode_width), so the result is in mode widths, not z/pi;
    the largest over the fields both runs have, so a narrow |phi| does not hide a
    wide |Apar|."""
    d = []
    for f in FIELDS:
        if f in a and f in b:
            ca = np.cumsum(a[f]) / a[f].sum()
            cb = np.cumsum(b[f]) / b[f].sum()
            w = (mode_width(a[f], z) + mode_width(b[f], z)) / 2
            d.append(np.abs(ca - cb).sum() * (z[1] - z[0]) / max(w, z[1] - z[0]))
    return float(max(d)) if d else np.inf


def group_modes(dist: np.ndarray, tol=GROUP_TOL) -> list[list[int]]:
    """Average-linkage groups of the runs behind distance matrix *dist*: the two
    closest groups merge while their mean pairwise distance is <= *tol*.
    Largest group first."""
    groups = [[k] for k in range(len(dist))]
    while len(groups) > 1:
        best, pair = np.inf, None
        for p in range(len(groups)):
            for q in range(p + 1, len(groups)):
                d = dist[np.ix_(groups[p], groups[q])].mean()
                if d < best:
                    best, pair = d, (p, q)
        if best > tol:
            break
        p, q = pair
        groups[p] += groups.pop(q)
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def plot_mode_groups(ds, out, written, profiles, tol=GROUP_TOL):
    """Converged runs (GENE or a field fit) grouped by mode structure: each
    mode_profile put on one z grid (zero beyond a run's own domain), the earth
    mover's distance between every pair (mode_emd) and average-linkage groups
    under *tol* (group_modes). One column per group of two or more, its members'
    |phi| overlaid on top and |Apar| below (shared z), each in its mode's colour
    (grey without --modes); runs in no group share a last "outliers" panel. Panels are
    zoomed to where any member is above 1% of its peak. corners_mode_groups.csv
    holds each run's group (0: outlier) and distance to its group's medoid."""
    status, gam, ky = ds.status.values, ds.gamma.values, ds.ky.values
    m = ds["mode"].values if "mode" in ds else None
    runs = sorted(k for k in profiles if status[k] in ("ok", "field"))
    if not runs:
        print("no converged run with field output; corners_mode_groups.png skipped")
        return
    zmax = max(np.abs(profiles[k]["zgrid"]).max() for k in runs)
    z = np.linspace(-zmax, zmax, GROUP_NZ)
    shapes = [{f: np.interp(z, profiles[k]["zgrid"], profiles[k][f], left=0, right=0)
               for f in FIELDS if f in profiles[k]} for k in runs]
    n = len(runs)
    dist = np.zeros((n, n))
    for p in range(n):
        for q in range(p + 1, n):
            dist[p, q] = dist[q, p] = mode_emd(shapes[p], shapes[q], z)
    groups = group_modes(dist, tol)
    off = dist[np.triu_indices(n, 1)]
    if off.size:
        q = np.percentile(off, (10, 50, 90))
        print(f"    mode groups: pairwise distance 10/50/90% {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g} "
              f"mode widths; tol {tol:g} -> {sum(len(g) > 1 for g in groups)} group(s), "
              f"{sum(len(g) == 1 for g in groups)} outlier(s)")
    panels = [g for g in groups if len(g) > 1]
    lone = [g[0] for g in groups if len(g) == 1]
    if lone:
        panels.append(lone)
    group_of, medoid_d = {}, {}
    for g_id, g in enumerate(groups, 1):
        med = g[int(np.argmin(dist[np.ix_(g, g)].sum(axis=1)))]
        for k in g:
            group_of[k] = g_id if len(g) > 1 else 0
            medoid_d[k] = dist[k, med] if len(g) > 1 else np.nan

    def mode_of(k):
        return str(m[runs[k]]) if m is not None and m[runs[k]] else ""

    # each group a column pair of axes: |phi| on top, |Apar| below, sharing z
    ncol = min(len(panels), GROUP_COLS)
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4.2 * ncol + 2.6, 5.0 * nrow + 1.3),
                             squeeze=False, gridspec_kw={"hspace": 0.12})
    for p in range(len(panels), nrow * ncol):
        axes[2 * (p // ncol), p % ncol].set_visible(False)
        axes[2 * (p // ncol) + 1, p % ncol].set_visible(False)
    for c, g in enumerate(panels):
        top, bot = axes[2 * (c // ncol), c % ncol], axes[2 * (c // ncol) + 1, c % ncol]
        bot.sharex(top)
        hi = 0.2
        for k in g:
            colour = (MODE_COLOR.get(mode_of(k), MODE_COLOR["unidentified"]) if m is not None
                      else MUTED)
            for f, a in zip(FIELDS, (top, bot)):
                if f in shapes[k]:
                    a.plot(z, shapes[k][f], color=colour, lw=1.0, alpha=0.7)
                    above = z[shapes[k][f] > 0.01]
                    hi = max(hi, np.abs(above).max() if above.size else 0)
        top.set_xlim(-hi * 1.05, hi * 1.05)
        for f, a, sym in zip(FIELDS, (top, bot), (r"$|\phi|$", r"$|A_\parallel|$")):
            a.set_ylim(0, 1.08)
            if not any(f in shapes[k] for k in g):
                a.text(0.5, 0.5, f"no {sym}", ha="center", va="center", fontsize=9,
                       color=MUTED, transform=a.transAxes)
            if c % ncol == 0:
                a.set_ylabel(f"{sym} / peak")
            tidy(a)
        top.tick_params(labelbottom=False)
        bot.set_xlabel(r"ballooning angle $z/\pi$")
        ax = top
        kys = [ky[runs[k][1]] for k in g]
        cases = len({runs[k][0] for k in g})
        mix: dict[str, int] = {}
        for k in g:
            lab = MODE_SHORT.get(mode_of(k), mode_of(k)) or "–"
            mix[lab] = mix.get(lab, 0) + 1
        head = "outliers" if g is lone else f"group {group_of[g[0]]}"
        span = f"{min(kys):g}" if min(kys) == max(kys) else f"{min(kys):g}–{max(kys):g}"
        ax.set_title(f"{head}: {len(g)} run(s), {cases} case(s), ky {span}",
                     loc="left", fontsize=9, color=INK)
        if m is not None:
            ax.text(0.98, 0.95, ", ".join(f"{k} {v}" for k, v in mix.items()), ha="right",
                    va="top", fontsize=7, color=INK, transform=ax.transAxes)
    handles = ([Line2D([], [], color=MODE_COLOR[x], lw=2, label=x) for x in MODE_COLOR
                if any(mode_of(k) == x for k in range(n))] if m is not None
               else [Line2D([], [], color=MUTED, lw=2, label="no --modes")])
    legend_right(fig, handles, title="mode")
    skipped = sum(1 for k in profiles if k not in runs)
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: mode structures grouped by "
                 f"shape (earth mover's distance along z / mode width, average linkage <= {tol:g})\n"
                 f"{n} converged run(s) in {len(groups) - len(lone)} group(s)"
                 f" + {len(lone)} outlier(s); {skipped} unconverged left out",
                 x=0.02, ha="left", fontsize=11, color=INK)
    w, h = fig.get_size_inches()
    fig.subplots_adjust(left=0.75 / w, right=1 - 2.4 / w, bottom=0.7 / h, top=1 - 1.0 / h,
                        wspace=0.22)
    save(fig, out, "corners_mode_groups.png", written)

    path = data_path(out, "corners_mode_groups.csv")
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky", "status", "gamma", "mode", "group",
                           "emd_to_medoid"]) + "\n")
        for k, (i, j) in enumerate(runs):
            fh.write(",".join([str(ds.corner.values[i]),
                               *(f"{ds[q].values[i]:g}" for q in ORDER), f"{ky[j]:g}",
                               str(status[i, j]), f"{gam[i, j]:.6g}", mode_of(k),
                               str(group_of[k]), f"{medoid_d[k]:.4g}"]) + "\n")
    written.append(path)


def plot_mode_extent(ds, out, written, tol=EXTENT_TOL, profiles=None, nx0=()):
    """|phi| (top) and |Apar| (bottom) at the two ends of the ballooning domain over
    their peak, against profile scaling: one column per corner, the left end (<)
    just left of it and the right end (>) just right, coloured by ky, on one log
    scale, so a scaling that pushes the mode to the domain edge stands out.
    Hollow: not converged (the last snapshot, not a settled mode). Dashed: *tol*,
    above which the mode is cut off and nx0 should grow.
    *profiles*, *nx0*: load_profiles(ds), read here when not given.
    + data/corners_mode_extent.csv."""
    if profiles is None:
        profiles, nx0 = load_profiles(ds)
    ky, n = ds.ky.values, ds.sizes["corner"]
    ends = {f: np.full((n, ky.size, 2), np.nan) for f in FIELDS}
    for (i, j), prof in profiles.items():
        e = mode_ends(prof)
        for f in ends:
            ends[f][i, j] = e[f]
    # ds lists every ped_grid corner; plot only those with field output
    rows = [i for i in range(n) if any(np.isfinite(v[i]).any() for v in ends.values())]
    if not rows:
        print("no field files; corners_mode_extent.png skipped")
        return
    good = ky[[any(np.isfinite(v[rows, j]).any() for v in ends.values()) for j in range(ky.size)]]
    norm = LogNorm(good.min(), good.max()) if good.min() > 0 and np.ptp(good) > 0 \
        else Normalize(good.min() - 0.5, good.max() + 0.5)
    cmap = plt.get_cmap(EXTENT_CMAP)
    fig, axes = plt.subplots(2, 1, figsize=(max(11, 0.75 * len(rows) + 4), 9.5), sharex=True,
                             layout="constrained")
    for ax, (f, sym) in zip(axes, (("field_phi", r"$|\phi|$"), ("field_apar", r"$|A_\parallel|$"))):
        v = ends[f]
        for x, i in enumerate(rows):
            for j, k in enumerate(ky):
                conv = ds.status.values[i, j] == "ok"
                c = cmap(norm(k))
                for side, (dx, mk) in enumerate(((-EXTENT_DX, "<"), (EXTENT_DX, ">"))):
                    r = v[i, j, side]
                    if np.isfinite(r):
                        ax.plot(x + dx, max(r, EXTENT_FLOOR), mk, ms=7, mec=c, mew=1.4,
                                mfc=c if conv else SURFACE, zorder=4)
        ax.axhline(tol, color=INK, ls="--", lw=1, zorder=2)
        ax.axhspan(tol, 1.5, color="#e34948", alpha=0.07, lw=0, zorder=0)
        for x in range(1, len(rows)):
            ax.axvline(x - 0.5, color=GRID, lw=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_ylim(top=1.5)
        ax.set_ylabel(f"{sym} end / peak")
        n_bad = int((v[rows] > tol).sum())
        ax.set_title(f"{sym} at the left (<) and right (>) ends of the ballooning domain / peak;"
                     f" {n_bad} end(s) above {tol:g} (cut off, grow nx0)", loc="left",
                     fontsize=10, color=INK)
        tidy(ax)
    axes[-1].set_xticks(range(len(rows)), [str(ds.label.values[i]) for i in rows], rotation=35,
                        ha="right", fontsize=8)
    axes[-1].set_xlabel(f"profile scaling {legend_title(ds)}")
    axes[-1].set_xlim(-0.6, len(rows) - 0.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.03,
                      pad=0.02)
    cb.set_label(r"$k_y\rho_{ref}$")
    cb.set_ticks(good, labels=[f"{k:g}" for k in good])
    cb.minorticks_off()
    handles = [Line2D([], [], ls="none", marker="<", color=MUTED, ms=7, label="left end"),
               Line2D([], [], ls="none", marker=">", color=MUTED, ms=7, label="right end"),
               Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, ms=7,
                      label="hollow: not converged"),
               Line2D([], [], color=INK, ls="--", lw=1, label=f"cut-off threshold {tol:g}")]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="upper center", ncol=4,
                    bbox_to_anchor=(0.5, -0.32))
    nx0_txt = ", ".join(f"{x:g}" for x in sorted(x for x in nx0 if x is not None)) or "?"
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: mode extent along the field line "
                 f"(nx0 = {nx0_txt})\nend values below {EXTENT_FLOOR:g} are drawn at it",
                 x=0.02, ha="left", fontsize=12, color=INK)
    save(fig, out, "corners_mode_extent.png", written)

    path = data_path(out, "corners_mode_extent.csv")
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky", "status", "phi_end_over_peak",
                           "apar_end_over_peak", "phi_left", "phi_right", "apar_left",
                           "apar_right"]) + "\n")
        for i, c in enumerate(ds.corner.values):
            for j, k in enumerate(ky):
                e = {f: ends[f][i, j] for f in ends}
                if not any(np.isfinite(x).any() for x in e.values()):
                    continue
                worst = [np.nanmax(e[f]) if np.isfinite(e[f]).any() else np.nan for f in e]
                fh.write(",".join([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}",
                                   str(ds.status.values[i, j]), *(f"{w:.4g}" for w in worst),
                                   *(f"{x:.4g}" for f in e for x in e[f])]) + "\n")
    written.append(path)


def vspace_edges(par_path: str) -> tuple[dict, tuple]:
    """({species: (|f| at v_par = -lv, at v_par = +lv, at the top mu point) / max|f|},
    (nv0, nw0, lv, lw)) of one run from its vsp file (TPED GeneVsp, <|f|^2>^1/2 summed
    over x and ky, on (mu, v_par, z)), last vsp time; each edge value is its max over z
    and along the edge. GENE's v_par grid is equidistant on [-lv, lv] ends included, so
    its ends are the box edges; the top mu point is the box's lw end. ({}, ()) without
    a vsp file."""
    vsp = re.sub(r"parameters_(\d{4})$", r"vsp_\1", par_path)
    if not os.path.isfile(vsp) or os.path.getsize(vsp) == 0:
        return {}, ()
    reader = GeneVsp(vsp)
    vx = reader.vsp_filepath_to_xarray("last", "all", VSP_QUANT)
    if vx is None:
        return {}, ()
    out = {}
    for name in vx.data_vars:
        f = np.abs(vx[name].values[-1])                 # (nw, nv, nz)
        peak = f.max()
        if peak > 0:
            out[name.removeprefix(VSP_QUANT + "_")] = (float(f[:, 0].max() / peak),
                                                       float(f[:, -1].max() / peak),
                                                       float(f[-1].max() / peak))
    return out, tuple(reader.param_dict.get(k) for k in ("nv0", "nw0", "lv", "lw"))


def load_vspace_edges(ds) -> tuple[dict, dict]:
    """{(corner index, ky index): vspace_edges ratios} and {(i, j): (nv0, nw0, lv, lw)}
    for every run with a vsp file."""
    edges, grids = {}, {}
    for i in range(ds.sizes["corner"]):
        for j in range(ds.sizes["ky"]):
            path = str(ds.path.values[i, j])
            if not path:
                continue
            try:
                e, g = vspace_edges(path)
            except Exception as exc:                # one unreadable vsp never sinks the plot
                print(f"    vsp unreadable {path}: {type(exc).__name__}: {exc}")
                continue
            if e:
                edges[i, j], grids[i, j] = e, g
    return edges, grids


def plot_vspace_edge(ds, out, written, tol=VSP_TOL):
    """The velocity-space analog of corners_mode_extent.png: GENE's <|f|^2>^1/2 at the
    edges of the (v_par, mu) box over its peak, per species (rows), against profile
    scaling: left, v_par = -lv (<) just left of each corner's column and +lv (>) just
    right; right, the top mu point (^). Coloured by ky, one log scale. Hollow: not
    converged. Dashed: *tol*, above which f is cut off by the box (grow lv or nv0 for
    v_par, lw or nw0 for mu). + data/corners_vspace_edge.csv with pass/fail per run."""
    edges, grids = load_vspace_edges(ds)
    if not edges:
        print("no vsp files; corners_vspace_edge.png skipped (istep_vsp = 0?)")
        return
    ky = ds.ky.values
    species = list(dict.fromkeys(s for e in edges.values() for s in e))
    rows = sorted({i for i, _ in edges})
    good = np.array(sorted({ky[j] for _, j in edges}))
    norm = LogNorm(good.min(), good.max()) if good.min() > 0 and np.ptp(good) > 0 \
        else Normalize(good.min() - 0.5, good.max() + 0.5)
    cmap = plt.get_cmap(EXTENT_CMAP)
    fig, axes = plt.subplots(len(species), 2, figsize=(max(14, 1.3 * len(rows) + 6),
                                                       3.6 * len(species) + 2.2),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    # left column: the v_par ends (index 0, 1 of a vspace_edges tuple); right: top mu (2)
    panels = ((((0, -EXTENT_DX, "<"), (1, EXTENT_DX, ">")), r"$v_\parallel = \mp l_v$",
               "lv or nv0"),
              (((2, 0.0, "^"),), r"top $\mu$ ($\approx l_w$)", "lw or nw0"))
    for s, spec in enumerate(species):
        for ax, (sides, edge, grow) in zip(axes[s], panels):
            n_bad = 0
            for x, i in enumerate(rows):
                for j, k in enumerate(ky):
                    e = edges.get((i, j), {}).get(spec)
                    if e is None:
                        continue
                    c = cmap(norm(k))
                    conv = ds.status.values[i, j] == "ok"
                    for idx, dx, mk in sides:
                        ax.plot(x + dx, max(e[idx], EXTENT_FLOOR), mk, ms=7, mec=c, mew=1.4,
                                mfc=c if conv else SURFACE, zorder=4)
                    n_bad += max(e[idx] for idx, *_ in sides) > tol
            ax.axhline(tol, color=INK, ls="--", lw=1, zorder=2)
            ax.axhspan(tol, 1.5, color="#e34948", alpha=0.07, lw=0, zorder=0)
            for x in range(1, len(rows)):
                ax.axvline(x - 0.5, color=GRID, lw=0.8, zorder=1)
            ax.set_yscale("log")
            ax.set_ylim(top=1.5)
            ax.set_title(f"{spec}: |f| at {edge} / peak; {n_bad} run(s) above {tol:g} "
                         f"(grow {grow})", loc="left", fontsize=10, color=INK)
            tidy(ax)
        axes[s, 0].set_ylabel(r"$\langle|f|^2\rangle^{1/2}$ edge / peak")
    for ax in axes[-1]:
        ax.set_xticks(range(len(rows)), [str(ds.label.values[i]) for i in rows], rotation=35,
                      ha="right", fontsize=8)
        ax.set_xlabel(f"profile scaling {legend_title(ds)}")
        ax.set_xlim(-0.6, len(rows) - 0.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.03,
                      pad=0.02)
    cb.set_label(r"$k_y\rho_{ref}$")
    cb.set_ticks(good, labels=[f"{k:g}" for k in good])
    cb.minorticks_off()
    handles = [Line2D([], [], ls="none", marker="<", color=MUTED, ms=7,
                      label=r"$v_\parallel = -l_v$"),
               Line2D([], [], ls="none", marker=">", color=MUTED, ms=7,
                      label=r"$v_\parallel = +l_v$"),
               Line2D([], [], ls="none", marker="^", color=MUTED, ms=7, label=r"top $\mu$"),
               Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, ms=7,
                      label="hollow: not converged"),
               Line2D([], [], color=INK, ls="--", lw=1, label=f"threshold {tol:g}")]
    fig.legend(handles=handles, frameon=False, fontsize=8, loc="outside lower center", ncol=5)
    grid_txt = "; ".join(f"nv0 = {g[0]:g}, nw0 = {g[1]:g}, lv = {g[2]:g}, lw = {g[3]:g}"
                         for g in sorted({g for g in grids.values() if None not in g}))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: f at the velocity-box edges "
                 f"(vsp, last time, max over z)\n{grid_txt or 'grid ?'}; values below "
                 f"{EXTENT_FLOOR:g} are drawn at it", x=0.02, ha="left", fontsize=12, color=INK)
    save(fig, out, "corners_vspace_edge.png", written)

    path = data_path(out, "corners_vspace_edge.csv")
    per = [f"{sp}_{q}" for sp in species
           for q in ("vpar_edge", "mu_edge", "vpar_minus", "vpar_plus")]
    n_fail = 0
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky", "status", "nv0", "nw0", "lv", "lw", *per,
                           "vpar_edge_worst", "mu_edge_worst", "vspace_pass"]) + "\n")
        for (i, j), e in sorted(edges.items()):
            vals = []
            for sp in species:
                m, p, mu = e.get(sp, (np.nan,) * 3)
                vals += [max(m, p), mu, m, p]
            vpar, mu = np.nanmax(vals[0::4]), np.nanmax(vals[1::4])
            ok = vpar <= tol and mu <= tol
            n_fail += not ok
            fh.write(",".join([str(ds.corner.values[i]), *(f"{ds[x].values[i]:g}" for x in ORDER),
                               f"{ky[j]:g}", str(ds.status.values[i, j]),
                               *("" if g is None else f"{g:g}" for g in grids[i, j]),
                               *(f"{v:.4g}" for v in vals), f"{vpar:.4g}", f"{mu:.4g}",
                               "pass" if ok else "fail"]) + "\n")
    written.append(path)
    print(f"vspace edge: {n_fail}/{len(edges)} run(s) above {tol:g}")


def _listdir(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


def run_x0(run_dir: str, rel: str):
    """x0 of a run dir: the last x0_<x0> folder in *rel*, else its parameters' x0."""
    m = X0_RE.findall(rel)
    if m:
        return float(m[-1])
    for path in gene_runs(run_dir)[:1]:
        with open(path) as fh:
            m = re.search(r"^\s*x0\s*=\s*([-+.\deE]+)", fh.read(), re.M)
        return float(m.group(1)) if m else None
    return None


def res2_dirs(diag_root: str, x0, extra_roots=(), depth=RES2_DEPTH) -> dict[str, list[str]]:
    """{corner tag: [2x run dirs at x0]} under diag_root, its 2x siblings and extra_roots."""
    parent, base = os.path.split(diag_root.rstrip("/\\"))
    roots = [diag_root] + [os.path.join(parent, d) for d in sorted(_listdir(parent))
                           if d != base and d.startswith(base) and RES2_RE.search(d)]
    roots += [r for r in extra_roots if r not in roots]
    found: dict[str, list[str]] = {}
    other = set()
    for root in roots:
        top = os.path.dirname(os.path.normpath(root))
        for d, subs, files in os.walk(root):
            rel = os.path.relpath(d, top).replace(os.sep, "/")   # starts at root's own name
            run = (any(x.startswith("scanfiles") for x in subs)
                   or any(re.fullmatch(r"parameters_\d{4}", f) for f in files))
            subs[:] = [x for x in subs if not x.startswith("scanfiles") and x != "plots"]
            if rel.count("/") >= depth:
                subs[:] = []
            m = TAG_RE.search(rel)
            if not (run and m and RES2_RE.search(rel)):
                continue
            subs[:] = []
            x = run_x0(d, rel)
            if x is not None and abs(x - float(x0[0])) < 5e-4:
                found.setdefault(m.group(), []).append(d)
            else:
                other.add(x)
    if other:
        print(f"2x runs at other x0 left out: {sorted(other, key=lambda v: (v is None, v))}")
    return found


def run_param(par_path: str, key: str):
    """A number from a parameters file (e.g. nz0, nx0), or None."""
    with open(par_path) as fh:
        m = re.search(rf"^\s*{key}\s*=\s*([-+.\deE]+)", fh.read(), re.M)
    return float(m.group(1)) if m else None


def run_nz0(par_path: str):
    v = run_param(par_path, "nz0")
    return int(v) if v is not None else None


def scan_nz0(ds):
    """nz0 of this scan, from the first corner that has a run."""
    for c in ds.corner.values:
        runs = gene_runs(os.path.join(ds.diag_root, x0_label((ds.x0,)), c))
        if runs:
            return run_nz0(runs[0])
    return None


def nz_dirs(root: str, x0, nz0: int) -> dict[str, list[str]]:
    """{corner tag: [run dir]} of the runs under <root>/x0_<x0>/<tag>/. A dir whose
    parameters do not have nz0 = *nz0* is reported and left out, so a run at the
    wrong resolution can never pose as the comparison."""
    base = os.path.join(root, x0_label(x0))
    found: dict[str, list[str]] = {}
    for name in sorted(_listdir(base)):
        d = os.path.join(base, name)
        runs = gene_runs(d) if TAG_RE.fullmatch(name) else []
        if not runs:
            continue
        got = run_nz0(runs[0])
        if got != nz0:
            print(f"--- nz0 {name}: parameters nz0 = {got}, not {nz0}; left out")
            continue
        found[name] = [d]
    if not found:
        print(f"no nz0 = {nz0} runs under {base}")
    return found


def live_fit(par_path: str):
    """(gamma, gamma_std, omega, omega_std, verdict) of field_omega's fit to the
    field_NNNN next to *par_path*, or None when there is no field output yet."""
    field = re.sub(r"parameters_(\d{4})$", r"field_\1", par_path)
    if not os.path.isfile(field) or os.path.getsize(field) == 0:
        return None
    try:
        _, stats = field_omega.fit(field)
    except Exception as exc:                    # a field file still being written
        print(f"    field fit failed {field}: {type(exc).__name__}: {exc}")
        return None
    f = stats[field_omega.FIELDS]
    g, w = f["gamma_avg"], f["omega_avg"]
    gs, ws = np.sqrt(f["gamma_variance"]), np.sqrt(f["omega_variance"])
    return g, gs, w, ws, field_omega.verdict(g, gs, w, ws)


def load_res2(ds, extra_roots=(), dirs=None, name="2x") -> dict:
    """{(corner tag, ky): 2x record} for every 2x run matching a corner in *ds*
    (or, with *dirs* = {tag: [run dir]}, for those runs, labelled *name*).

    A record holds gamma, omega and their std, source ('GENE' where GENE
    converged, else the verdict of the live field fit: 'pass', 'decaying',
    'ambiguous'; 'no fit' without field output) and the parameters path."""
    if dirs is None:
        dirs = res2_dirs(ds.diag_root, (ds.x0,), extra_roots)
    alt, out = name, {}
    for name, paths in sorted(dirs.items()):
        if name not in ds.corner.values:
            print(f"--- {alt} {name}: no 1x corner at {x0_text(ds)}; left out")
            continue
        for d in paths:
            g = gene_corner(d)
            if g is None:
                continue
            for k, v in sorted(g[0].items()):
                rec = {"dir": d, "path": v[5], "gamma": v[0], "omega": v[1],
                       "gamma_std": np.nan, "omega_std": np.nan, "source": "GENE"}
                if v[2] != "ok":
                    fitted = live_fit(v[5])
                    if fitted is None:
                        rec.update(gamma=np.nan, omega=np.nan, source="no fit")
                    else:
                        rec.update(zip(("gamma", "gamma_std", "omega", "omega_std", "source"),
                                       fitted))
                old = out.get((name, k))
                if old is None or old["source"] != "GENE":   # GENE converged wins
                    out[(name, k)] = rec
        print(f"--- {alt} {name}: " + "; ".join(
            f"ky={k:g} {r['source']}" for (n, k), r in sorted(out.items()) if n == name))
    return out


def one_x(ds, i, var):
    """1x value per ky of corner i: GENE converged, else the recorded field fit; + std, conv."""
    conv = ds.status.values[i] == "ok"
    y = np.where(conv, ds[var].values[i], ds[f"{var}_field"].values[i])
    err = np.where(conv, np.nan, ds[f"{var}_field_std"].values[i])
    return y, err, conv


def parity(ax, pts, var, base, alt):
    """alt vs base for one variable: each point a corner and ky (number = ky), the
    1:1 line and a +-10% band."""
    for q in pts:
        ax.errorbar(q["x"], q["y"], xerr=q["xerr"], yerr=q["yerr"], fmt="D", ms=7,
                    mfc=q["color"] if q["filled"] else SURFACE, mec=q["color"], mew=1.6,
                    color=q["color"], capsize=2, lw=0.8, alpha=0.45 if q["weak"] else 1.0,
                    zorder=5)
        ax.annotate(f"{q['k']:g}", (q["x"], q["y"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=MUTED)
    vals = [v for q in pts for v in (q["x"], q["y"])] + [0.0]
    span = (max(vals) - min(vals)) or 1.0
    lim = [min(vals) - 0.05 * span, max(vals) + 0.05 * span]
    xs = np.linspace(*lim, 50)
    ax.fill_between(xs, 0.9 * xs, 1.1 * xs, color=GRID, lw=0, zorder=1)
    ax.plot(lim, lim, color=MUTED, lw=0.9, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    sym = {"gamma": r"\gamma", "omega": r"\omega"}[var]
    ax.set_xlabel(rf"${sym}$ {base} [$c_{{ref}}/L_{{ref}}$]")
    ax.set_ylabel(rf"${sym}$ {alt} [$c_{{ref}}/L_{{ref}}$]")
    ax.set_title(f"{alt} vs {base} {var} (number: ky; band: ±10%)", loc="left", fontsize=10,
                 color=INK)
    tidy(ax)


def pct_violins(ax, ky, pct, var, base, alt):
    """% difference (alt - base) / |base| at each ky: a violin (KDE over the corners)
    where two or more differ, and every corner as its own point on top."""
    gaps = np.diff(np.unique(ky))
    width = 0.8 * gaps.min() if gaps.size else 0.2      # one width, so violins compare
    for k in sorted(pct):
        vals = np.array([v for v, _, _ in pct[k]])
        if vals.size >= 2 and np.ptp(vals) > 0:
            vp = ax.violinplot(vals, positions=[k], widths=width, showextrema=False,
                               showmedians=True)
            for body in vp["bodies"]:
                body.set_facecolor(GRID)
                body.set_edgecolor(MUTED)
                body.set_alpha(0.9)
            vp["cmedians"].set_color(INK)
        for v, color, weak in pct[k]:
            ax.plot(k, v, "D", ms=5, color=color, alpha=0.45 if weak else 1.0, zorder=5)
    ax.axhspan(-10, 10, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"$k_y\rho_{ref}$")
    ax.set_ylabel(f"{var}: ({alt} - {base}) / |{base}| [%]")
    ax.set_title(f"{var} % difference per ky (violin: KDE over corners; band: ±10%)",
                 loc="left", fontsize=10, color=INK)
    tidy(ax)


def plot_res2(ds, out, written, extra_roots=(), dirs=None, base="1x", alt="2x",
              stem="corners_res2", what="2x resolution check"):
    """Each corner with a 2x rerun: its 1x line (converged filled, recorded field
    fit hollow) and the 2x values as diamonds beside them (GENE converged filled,
    live field fit hollow +-1 std, faded unless it passes), joined to the 1x value
    at the same ky (top row); the % difference per ky as a violin over the corners,
    on the same ky axis (middle); 2x vs 1x for gamma and omega (bottom).
    + data/corners_res2.csv.

    *dirs*, *base*, *alt*, *stem* and *what* turn it into any other rerun
    comparison: corners_nz128.png is the nz0 = 128 reruns (alt) against the
    nz0 = 64 scan (base)."""
    res = load_res2(ds, extra_roots, dirs, alt)
    if not res:
        print(f"no {alt} runs at this x0; {stem}.png skipped")
        return
    bk, ak = (re.sub(r"\W", "", v.replace("nz0", "nz")) for v in (base, alt))   # CSV keys
    names = [c for c in ds.corner.values if any(n == c for n, _ in res)]
    ky = ds.ky.values
    nudge = RES2_NUDGE * (np.ptp(ky) if ky.size > 1 else 1.0)
    fig = plt.figure(figsize=(14, 15.5))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 0.75, 1.05))
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    pct_axes = [fig.add_subplot(gs[1, n], sharex=axes[n]) for n in range(2)]
    par_axes = [fig.add_subplot(gs[2, n]) for n in range(2)]
    pts = {"gamma": [], "omega": []}
    pct = {"gamma": {}, "omega": {}}
    rows, faded = [], False
    for c in names:
        i = int(np.flatnonzero(ds.corner.values == c)[0])
        st = rerun_style(ds, c)
        one = {v: one_x(ds, i, v) for v in ("gamma", "omega")}
        for ax, var in zip(axes[:2], ("gamma", "omega")):
            y, err, conv = one[var]
            ok = np.isfinite(y)
            ax.plot(ky[ok], y[ok], lw=1.2, color=st["color"], ls=st["ls"], alpha=0.7, zorder=3)
            ax.plot(ky[ok & conv], y[ok & conv], st["marker"], ms=5, color=st["color"], zorder=4)
            g = ok & ~conv
            ax.errorbar(ky[g], y[g], yerr=err[g], fmt=st["marker"], ms=5, mfc=SURFACE,
                        mec=st["color"], color=st["color"], capsize=2, lw=0.8, zorder=4)
        for (n, k), r in sorted(res.items()):
            if n != c:
                continue
            j = int(np.argmin(abs(ky - k)))
            on_ky = abs(ky[j] - k) < 1e-6
            weak = r["source"] not in ("GENE", "pass")
            faded |= weak and np.isfinite(r["gamma"])
            for ax, var in zip(axes[:2], ("gamma", "omega")):
                v2, e2 = r[var], r[f"{var}_std"]
                if not np.isfinite(v2):
                    ax.plot(k + nudge, 0, "x", color=st["color"], ms=8, mew=1.8, zorder=6)
                    continue
                if on_ky and np.isfinite(one[var][0][j]):
                    ax.plot([k, k + nudge], [one[var][0][j], v2], ":", color=st["color"],
                            lw=1, zorder=4)
                filled = r["source"] == "GENE"
                ax.errorbar(k + nudge, v2, yerr=e2, fmt="D", ms=8,
                            mfc=st["color"] if filled else SURFACE, mec=st["color"], mew=1.8,
                            color=st["color"], capsize=3, lw=1, alpha=0.45 if weak else 1.0,
                            zorder=7)
            rel = {}
            for var in ("gamma", "omega"):
                v1, e1 = (one[var][0][j], one[var][1][j]) if on_ky else (np.nan, np.nan)
                rel[var] = (r[var] - v1) / abs(v1) if np.isfinite(v1) and v1 else np.nan
                if np.isfinite(v1) and np.isfinite(r[var]):
                    pts[var].append({"x": v1, "y": r[var], "xerr": e1, "yerr": r[f"{var}_std"],
                                     "color": st["color"], "filled": r["source"] == "GENE",
                                     "weak": weak, "k": k})
                    pct[var].setdefault(k, []).append((100 * rel[var], st["color"], weak))
            g1 = one["gamma"][0][j] if on_ky else np.nan
            st1 = ds.status.values[i, j] if on_ky else "missing"
            rows.append([c, *(f"{ds[x].values[i]:g}" for x in ORDER), f"{k:g}",
                         st1 if st1 == "ok" or not np.isfinite(g1) else
                         f"field {ds.field_verdict.values[i, j]}".strip(),
                         *(f"{one[v][0][j]:.6g}" if on_ky else "nan" for v in ("gamma", "omega")),
                         r["source"], *(f"{r[v]:.6g}" for v in ("gamma", "gamma_std",
                                                               "omega", "omega_std")),
                         *(f"{rel[v]:.4g}" for v in ("gamma", "omega")), r["dir"]])
    for ax, var, sym in zip(axes[:2], ("gamma", "omega"), (r"\gamma", r"\omega")):
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"$k_y\rho_{ref}$")
        ax.set_ylabel(rf"${sym}$ [$c_{{ref}}/L_{{ref}}$]")
        tidy(ax)
    axes[0].set_title(f"growth rate: {base} line, {alt} diamonds", loc="left", fontsize=10,
                      color=INK)
    axes[1].set_title(f"real frequency: {base} line, {alt} diamonds", loc="left", fontsize=10,
                      color=INK)
    for n, var in enumerate(("gamma", "omega")):
        pct_violins(pct_axes[n], ky, pct[var], var, base, alt)
        parity(par_axes[n], pts[var], var, base, alt)
    handles = [Line2D([], [], lw=1.8, ms=5, **rerun_style(ds, c),
                      label=str(ds.label.values[list(ds.corner.values).index(c)]))
               for c in names]
    extra = [Line2D([], [], ls="none", marker="o", color=MUTED, ms=5, label=f"{base} GENE converged"),
             Line2D([], [], ls="none", marker="o", mfc=SURFACE, mec=MUTED, ms=5,
                    label=f"{base} recorded field fit ±1σ"),
             Line2D([], [], ls="none", marker="D", color=MUTED, ms=7, label=f"{alt} GENE converged"),
             Line2D([], [], ls="none", marker="D", mfc=SURFACE, mec=MUTED, mew=1.8, ms=7,
                    label=f"{alt} field fit, passing ±1σ")]
    if faded:
        extra.append(Line2D([], [], ls="none", marker="D", mfc=SURFACE, mec=MUTED, mew=1.8,
                            ms=7, alpha=0.45, label=f"{alt} field fit, ambiguous/decaying"))
    if any(not np.isfinite(r["gamma"]) for r in res.values()):
        extra.append(Line2D([], [], ls="none", marker="x", color=MUTED, ms=7, mew=1.8,
                            label=f"{alt}, no field output yet"))
    legend_right(fig, handles, extra, legend_title(ds))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} at {x0_text(ds)}: {what} "
                 f"({alt} not converged: live field fit)", x=0.02, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.97))
    save(fig, out, f"{stem}.png", written)

    path = data_path(out, f"{stem}.csv")
    with open(path, "w") as fh:
        fh.write(",".join(["corner", *ORDER, "ky", f"status_{bk}", f"gamma_{bk}", f"omega_{bk}",
                           f"source_{ak}", f"gamma_{ak}", f"gamma_{ak}_std", f"omega_{ak}",
                           f"omega_{ak}_std", "gamma_rel_diff", "omega_rel_diff",
                           f"dir_{ak}"]) + "\n")
        for r in rows:
            fh.write(",".join(r) + "\n")
    written.append(path)


def plot_nz_compare(ds, out, written, nz_compare, nz_root=None):
    """corners_nz<N>.png: each corner this scan ran beside its nz0 = N run of the
    same profile scaling, from <diag-root>_nz<N> (or nz_root)."""
    root = nz_root or ds.diag_root.rstrip("/\\") + NZ_SUFFIX.format(nz_compare)
    own = scan_nz0(ds)
    dirs = nz_dirs(root, (ds.x0,), nz_compare)
    # ds lists every ped_grid corner, run or not: compare only the profile
    # scalings this scan actually ran, not every nz0 = N corner
    ran = {t for t in dirs if gene_runs(os.path.join(ds.diag_root, x0_label((ds.x0,)), t))}
    if set(dirs) - ran:
        print(f"--- nz0 {nz_compare}: {len(set(dirs) - ran)} corner(s) with no nz0 {own} run "
              "left out")
    plot_res2(ds, out, written, dirs={t: dirs[t] for t in sorted(ran)},
              base=f"nz0 {own}", alt=f"nz0 {nz_compare}", stem=f"corners_nz{nz_compare}",
              what=f"nz0 {own} vs {nz_compare}")


def fit_ylim(ax, x, ys, xlim, pad=0.05):
    m = (x >= xlim[0]) & (x <= xlim[1])
    vals = np.concatenate([np.asarray(y)[m] for y in ys])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = float(vals.min()), float(vals.max())
        span = (hi - lo) or abs(hi) or 1.0
        ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_profiles(ds, out, written, xlim=None, name="corners_profiles.png"):
    vars_ = [v for v in ("ne", "Te", "ni", "Ti") if v in ds]
    if not vars_:
        print("no profiles to plot")
        return
    units = {v: u for v, _, u in ITERDB_VARS.values()}
    x = ds.rho_tor.values
    xlim = xlim or (float(x.min()), float(x.max()))
    has_q = "q" in ds
    fig = plt.figure(figsize=(3.6 * len(vars_) + 2.8, 9.4 if has_q else 6.4))
    gs = fig.add_gridspec(3 if has_q else 2, len(vars_),
                          height_ratios=(1, 1, 1.25) if has_q else (1, 1))
    ax = np.empty((2, len(vars_)), dtype=object)
    for j in range(len(vars_)):
        ax[0, j] = fig.add_subplot(gs[0, j], sharex=ax[0, 0] if j else None)
        ax[1, j] = fig.add_subplot(gs[1, j], sharex=ax[0, 0])
    for j, v in enumerate(vars_):
        top, bot = ax[0, j], ax[1, j]
        for c in ds.corner.values:
            st = style(point_of(ds, c))
            for a, y in ((top, ds[v]), (bot, ds[f"omg_{v}"])):
                a.plot(x, y.sel(corner=c), color=st["color"], ls=st["ls"], lw=1.5)
        fit_ylim(top, x, ds[v].values, xlim)
        # Gradient y range ignores the last points, where -dln/drho blows up at the separatrix.
        fit_ylim(bot, x, ds[f"omg_{v}"].values, (xlim[0], min(xlim[1], 0.98)))
        top.set_title(f"{v} [{units[v]}]", loc="left", fontsize=10, color=INK)
        bot.set_title(rf"$-d\ln {v[0]}_{v[1]}/d\rho_{{tor}}$", loc="left", fontsize=10, color=INK)
        if not has_q:
            bot.set_xlabel(r"$\rho_{tor}$")
        for a in (top, bot):
            for xm, cm in x0_marks(ds):
                a.axvline(xm, color=cm, ls=":", lw=1.4)
            a.set_xlim(*xlim)
            tidy(a)
    if has_q:                                     # one under each column, zoomed on x0
        marks = [xm for xm, _ in x0_marks(ds)]
        qlim = (max(float(x.min()), min(marks) - Q_ZOOM), min(float(x.max()), max(marks) + Q_ZOOM))
        for a in ax.ravel():
            a.axvspan(*qlim, color=GRID, alpha=0.45, lw=0, zorder=0)
        for j in range(len(vars_)):
            plot_q(fig.add_subplot(gs[2, j]), ds, x, qlim,
                   labels=j == len(vars_) - 1, title=j == 0)
    handles = [Line2D([], [], color=style(point_of(ds, c))["color"],
                      ls=style(point_of(ds, c))["ls"], lw=1.8, label=str(l))
               for c, l in zip(ds.corner.values, ds.label.values)]
    legend_right(fig, handles, [Line2D([], [], color=INK, ls=":", lw=1.4,
                                       label=(f"x0 = {ds.x0:g}" if not is_named(ds) else
                                              "x0 (each case's colour)"))],
                 legend_title(ds))
    fig.suptitle(f"NSTX {ds.shot} {ds.scan} profiles (profiles.iterdb)"
                 + (f", rho_tor {xlim[0]:g}-{xlim[1]:g}" if name.endswith("_ped.png") else ""),
                 x=0.04, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 0.8, 0.95))
    save(fig, out, name, written)


def plot_q(ax, ds, x, xlim, labels=True, title=True):
    """|q| vs rho_tor per corner, a faint line at each integer q in view and, per
    corner, a drop line from the axis to where it crosses that integer."""
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        ax.plot(x, ds.q.sel(corner=c), color=st["color"], ls=st["ls"], lw=1.5, zorder=3)
    # y range over the window, short of the separatrix where q runs away
    fit_ylim(ax, x, ds.q.values, (xlim[0], min(xlim[1], 0.99)))
    lo, hi = ax.get_ylim()
    ints = np.arange(np.ceil(max(lo, 0)), np.floor(hi) + 1)
    every = max(1, int(np.ceil(ints.size / 20)))      # label at most ~20 integers
    for k, n in enumerate(ints):
        ax.axhline(n, color=GRID, lw=0.7, zorder=1)
        if labels and k % every == 0:
            ax.text(1.003, n, f"{n:g}", transform=ax.get_yaxis_transform(), fontsize=7,
                    va="center", color=MUTED)
    for c in ds.corner.values:
        st = style(point_of(ds, c))
        y = ds.q.sel(corner=c).values
        for n in ints:
            r = integer_crossings(x, y, n)
            r = r[(r >= xlim[0]) & (r <= xlim[1])]
            if r.size:
                ax.vlines(r, lo, n, color=st["color"], ls=st["ls"], lw=0.8, alpha=0.75,
                          zorder=2)
    for xm, cm in x0_marks(ds):
        ax.axvline(xm, color=cm, ls=":", lw=1.4)
    ax.set_ylim(lo, hi)
    if 2 <= ints.size <= 20:                      # ticks on the integers themselves
        ax.set_yticks(ints)
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"$\rho_{tor}$")
    if title:
        ax.set_title(rf"$|q|$ (cheaseBS geqdsk), $x_0 \pm {Q_ZOOM:g}$ (shaded above); drop "
                     r"lines at each corner's integer-$q$ surfaces", loc="left", fontsize=10,
                     color=INK)
    tidy(ax)


def corner_modes(ds):
    """(mode at each corner's max-gamma ky, mode sequence over ky e.g. 'IT>KBM'), or blanks."""
    n = ds.sizes["corner"]
    if "mode" not in ds:
        return np.array([""] * n, dtype=object), [""] * n
    at_max, seq = [], []
    for i in range(n):
        modes = ds["mode"].values[i]
        gam = ds["gamma"].values[i]
        has = np.isfinite(gam) & (modes != "")
        at_max.append(modes[np.flatnonzero(has)[np.argmax(gam[has])]] if has.any() else "")
        runs = [m for j, m in enumerate(modes) if m and (j == 0 or m != modes[j - 1])]
        runs = [m for j, m in enumerate(runs) if j == 0 or m != runs[j - 1]]
        seq.append(">".join(MODE_SHORT.get(m, m) for m in runs) or "-")
    return np.array(at_max, dtype=object), seq


#: s-alpha ion-scale peak: points either side of a local max of gamma in the
#: quadratic fit (in log ky) that confirms it as a cusp (ion_peak).
PEAK_HALF = 2
#: s-alpha ion plots: marker area (points^2) at the smallest and largest ion-peak gamma
ION_SIZE = (25, 260)


def converged_gamma(ds) -> np.ndarray:
    """gamma (corner, ky) of converged ky only: GENE's own value or a passing field
    estimate; unconverged, missing and --field-fits decaying/ambiguous fits NaN."""
    return np.where(np.isin(ds.status.values, list(FIT_STYLE)), np.nan, ds.gamma.values)


def ion_peak(gam_row, ky) -> tuple[int | None, str]:
    """(ky index, kind) of one case's ion-scale peak over its converged ky, whole
    ky range. kind "cusp": the first interior local max of gamma that a
    least-squares quadratic in log ky over it and PEAK_HALF converged ky either
    side confirms (concave, vertex between its two neighbours), so a one-point
    wiggle on a rising curve is no cusp. Without one, kind "max": the largest
    converged gamma. (None, "") with no converged ky."""
    js = np.flatnonzero(np.isfinite(gam_row))
    if not js.size:
        return None, ""
    x, g = np.log10(ky[js]), gam_row[js]
    for q in range(1, len(js) - 1):
        if not (g[q] > g[q - 1] and g[q] >= g[q + 1]):
            continue
        w = slice(max(q - PEAK_HALF, 0), q + PEAK_HALF + 1)
        a, b, _ = np.polyfit(x[w], g[w], 2)
        if a < 0 and x[q - 1] < -b / (2 * a) < x[q + 1]:
            return int(js[q]), "cusp"
    return int(js[np.argmax(g)]), "max"


def ion_peaks(ds) -> list[tuple]:
    """(ky index or None, gamma, mode, kind) at each case's ion_peak."""
    gam = converged_gamma(ds)
    modes = ds["mode"].values if "mode" in ds else np.full(gam.shape, "", dtype=object)
    out = []
    for i in range(ds.sizes["corner"]):
        j, kind = ion_peak(gam[i], ds.ky.values)
        out.append((j, np.nan, "", "") if j is None
                   else (j, gam[i, j], str(modes[i, j]), kind))
    return out


def ky_ring(ax_span, s_span, n, radius=0.02, aspect=1.35):
    """Data offsets placing n ky points on a small ring, lowest ky at 12 o'clock, clockwise."""
    ang = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    return radius * ax_span * np.cos(ang), radius * aspect * s_span * np.sin(ang)


def plot_s_alpha(ds, out, written):
    """shat vs alpha_MHD, one ring of points per corner, one point per ky coloured by its gamma."""
    if "shat" not in ds or "alpha_MHD" not in ds:
        print("no shat/alpha_MHD (no GENE parameters read); s-alpha plot skipped")
        return
    a, s = ds.alpha_MHD.values, ds.shat.values
    ok = np.isfinite(a) & np.isfinite(s)
    if not ok.any():
        print("shat/alpha_MHD all NaN; s-alpha plot skipped")
        return
    status, ky = ds.status.values, ds.ky.values
    gam = converged_gamma(ds)                    # unconverged ky are gaps in the ring
    modes = ds["mode"].values if "mode" in ds else np.full(gam.shape, "", dtype=object)
    peaks = ion_peaks(ds)
    gfin = gam[ok][np.isfinite(gam[ok])]
    span = max(np.abs(gfin).max(), 1e-6) if gfin.size else 1.0
    norm = Normalize(-span, span)                # symmetric: the grey middle is gamma = 0
    wide = "mode" in ds
    fig, (ax, key) = plt.subplots(1, 2, figsize=(15 if wide else 13.5, 7.5),
                                  width_ratios=(3, 1.8 if wide else 1.3))
    dx, dy = ky_ring(np.ptp(a[ok]) or 1.0, np.ptp(s[ok]) or 1.0, ky.size)
    sc, lines = None, []
    for n, i in enumerate(np.flatnonzero(ok), start=1):
        x, y = a[i] + dx, s[i] + dy
        ax.plot(np.r_[x, x[:1]], np.r_[y, y[:1]], color=GRID, lw=0.8, zorder=2)
        for j in range(ky.size):
            if not np.isfinite(gam[i, j]):       # unconverged or missing: gap in the ring
                continue
            field = status[i, j] == "field"
            sc = ax.scatter(x[j], y[j], c=[gam[i, j]], cmap=GAMMA_CMAP, norm=norm, s=42,
                            marker=MODE_MARKER.get(modes[i, j], "o") if modes[i, j] else "o",
                            edgecolors=INK, linewidths=1.5 if field else 0.5, zorder=4)
        jp = peaks[i][0]
        if jp is not None:                       # ring around the ion-scale peak, dashed: max
            ax.scatter(x[jp], y[jp], s=150, facecolors="none", edgecolors=INK, linewidths=1.2,
                       linestyles="-" if peaks[i][3] == "cusp" else "--", zorder=5)
        ax.annotate(str(n), (a[i], s[i]), ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color=INK, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE, ec="none", alpha=0.8))
        g = gam[i][np.isfinite(gam[i])]
        rng = f"{g.min():.2f}-{g.max():.2f}" if g.size else "-"
        lines.append(f"{n:>2}  {ds.label.values[i]:<21} {rng:>9}  " + peak_text(ds, peaks[i]))
    handles = [Line2D([], [], ls="none", marker="o", color=MUTED, mec=INK, mew=1.8, ms=7,
                      label="TPED field estimate (thick edge)"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=INK, mew=1.2, ms=11,
                      label="ion-scale peak (first cusp of gamma, quadratic-fit check)"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, mew=1.2, ms=11,
                      label="dashed ring: no cusp, max gamma instead")]
    if wide:
        handles += [Line2D([], [], ls="none", marker=MODE_MARKER[m], color=MUTED, mec=INK, ms=7,
                           label=m) for m in MODE_MARKER if (modes == m).any()]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    key.axis("off")
    head = f"{'#':>2}  {legend_title(ds):<21} {'gamma':>9}  ion peak ky" + (" mode" if wide else "")
    key.text(0, 1, head + "\n" + "\n".join(lines), family="monospace", fontsize=8, va="top",
             color=INK, transform=key.transAxes)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(r"$\gamma$ at each $k_y$ [$c_{ref}/L_{ref}$]")
    ax.set_xlabel(r"$\alpha_{MHD} = q_0^2\,(R_0/L_{ref})\,(-d\beta/dx)$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.margins(x=0.1, y=0.1)
    ax.set_title(f"s-alpha at {x0_text(ds)}, every ky", loc="left", fontsize=10, color=INK)
    tidy(ax)
    fig.tight_layout()
    save(fig, out, "corners_s_alpha.png", written)


def peak_text(ds, peak) -> str:
    """'0.7 KBM' for an ion_peaks entry, '20 ETG (max)' for the no-cusp fallback,
    'none' with no converged ky."""
    j, _, mode, kind = peak
    if j is None:
        return "none"
    return (f"{ds.ky.values[j]:g}" + (f" {MODE_SHORT.get(mode, mode)}" if mode else "")
            + (" (max)" if kind == "max" else ""))


def ion_rows(ds, scan=None) -> list[dict]:
    """One row per case with finite (alpha_MHD, shat): its ion_peaks entry, for
    corners_s_alpha_ion.png and the x0 summary."""
    if "shat" not in ds or "alpha_MHD" not in ds:
        return []
    ref = ds["case_ref"].values if "case_ref" in ds else np.zeros(ds.sizes["corner"], bool)
    rows = []
    for i, (j, g, mode, kind) in enumerate(ion_peaks(ds)):
        a, s = float(ds.alpha_MHD.values[i]), float(ds.shat.values[i])
        if np.isfinite(a) and np.isfinite(s):
            rows.append({"case": str(ds.label.values[i]), "scan": scan or ds.scan,
                         "ref": bool(ref[i]), "alpha_MHD": a, "shat": s,
                         "ky_ion": np.nan if j is None else float(ds.ky.values[j]),
                         "gamma_ion": g, "mode_ion": mode, "peak": kind})
    return rows


def ion_size(g, lo, hi):
    """Marker area for ion-peak gamma *g* on the plot's [lo, hi]: ION_SIZE linear
    in gamma, the smallest for NaN."""
    if not np.isfinite(g):
        return ION_SIZE[0]
    u = 0.5 if hi <= lo else float(np.clip((g - lo) / (hi - lo), 0, 1))
    return ION_SIZE[0] + u * (ION_SIZE[1] - ION_SIZE[0])


def ion_scatter(ax, rows, numbered=True):
    """One marker per row at (alpha_MHD, shat): shape and colour its ion-peak mode
    (grey with no mode label), area its ion-peak gamma (ion_size, over these
    rows), filled for a cusp, hollow for the no-cusp max; grey x with no
    converged ky; black edge on the reference, numbered in row order when
    *numbered*."""
    g = np.array([r["gamma_ion"] for r in rows], float)
    lo, hi = (np.nanmin(g), np.nanmax(g)) if np.isfinite(g).any() else (0.0, 1.0)
    for n, r in enumerate(rows, start=1):
        if not np.isfinite(r["ky_ion"]):
            ax.scatter(r["alpha_MHD"], r["shat"], marker="x", s=ION_SIZE[0], color=MUTED,
                       linewidths=1.2, zorder=3)
        else:
            m, c = r["mode_ion"], MODE_COLOR.get(r["mode_ion"], MUTED)
            cusp = r["peak"] == "cusp"
            ax.scatter(r["alpha_MHD"], r["shat"], marker=MODE_MARKER.get(m, "o") if m else "o",
                       s=ion_size(r["gamma_ion"], lo, hi), facecolors=c if cusp else "none",
                       edgecolors=INK if r["ref"] else (SURFACE if cusp else c),
                       linewidths=1.4 if r["ref"] or not cusp else 0.6, alpha=0.85, zorder=4)
        if numbered:
            ax.annotate(str(n), (r["alpha_MHD"], r["shat"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK, zorder=6)
    present = sorted({r["mode_ion"] for r in rows if np.isfinite(r["ky_ion"]) and r["mode_ion"]},
                     key=list(MODE_COLOR).index)
    handles = [Line2D([], [], ls="none", marker=MODE_MARKER.get(m, "o"), color=MODE_COLOR[m],
                      ms=8, label=m) for m in present]
    if any(np.isfinite(r["ky_ion"]) and not r["mode_ion"] for r in rows):
        handles.append(Line2D([], [], ls="none", marker="o", color=MUTED, ms=8,
                              label="ion peak, no mode (--modes)"))
    if any(r["peak"] == "max" for r in rows):
        handles.append(Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, mew=1.4,
                              ms=8, label="hollow: no cusp, max gamma instead"))
    if any(not np.isfinite(r["ky_ion"]) for r in rows):
        handles.append(Line2D([], [], ls="none", marker="x", color=MUTED, ms=7, mew=1.2,
                              label="no converged ky"))
    if any(r["ref"] for r in rows):
        handles.append(Line2D([], [], ls="none", marker="o", color=MUTED, mec=INK, mew=1.4,
                              ms=8, label="reference 1.00"))
    if np.isfinite(g).any():                     # size key: gamma at the ion peak
        for v in dict.fromkeys((lo, (lo + hi) / 2, hi)):
            handles.append(Line2D([], [], ls="none", marker="o", color=MUTED,
                                  ms=np.sqrt(ion_size(v, lo, hi)),
                                  label=f"gamma {v:.3g}"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="best")
    ax.set_xlabel(r"$\alpha_{MHD} = q_0^2\,(R_0/L_{ref})\,(-d\beta/dx)$")
    ax.set_ylabel(r"$\hat{s}$")
    ax.margins(x=0.1, y=0.1)
    tidy(ax)


def plot_s_alpha_ion(ds, out, written):
    """corners_s_alpha_ion.png: shat vs alpha_MHD, one marker per case, its mode at
    the ion-scale peak (ion_peak: first quadratic-confirmed cusp of gamma over ky,
    else the max); key: case, peak ky, gamma, mode, which of the two."""
    rows = ion_rows(ds)
    if not rows:
        print("no shat/alpha_MHD; corners_s_alpha_ion.png skipped")
        return
    fig, (ax, key) = plt.subplots(1, 2, figsize=(13.5, 7.5), width_ratios=(3, 1.5))
    ion_scatter(ax, rows)
    ax.set_title(f"s-alpha at {x0_text(ds)}, mode at the ion-scale peak", loc="left",
                 fontsize=10, color=INK)
    key.axis("off")
    lines = [f"{n:>2}  {r['case']:<21} " + ("none" if not np.isfinite(r["ky_ion"]) else
             f"{r['ky_ion']:>5g} {r['gamma_ion']:>7.3f} "
             f"{MODE_SHORT.get(r['mode_ion'], r['mode_ion']):<4} {r['peak']}")
             for n, r in enumerate(rows, start=1)]
    key.text(0, 1, f"{'#':>2}  {legend_title(ds):<21} {'ky':>5} {'gamma':>7} mode peak\n"
             + "\n".join(lines), family="monospace", fontsize=8, va="top", color=INK,
             transform=key.transAxes)
    fig.tight_layout()
    save(fig, out, "corners_s_alpha_ion.png", written)


def plot_s_alpha_x0(rows, x0, shot, out, written):
    """<out>/s_alpha_ion.png: every case at this x0 (corners, reference, every
    VURS batch) as in corners_s_alpha_ion.png, unnumbered; + data/s_alpha_ion.csv."""
    if not rows:
        print("no case with shat/alpha_MHD; s_alpha_ion.png skipped")
        return
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ion_scatter(ax, rows, numbered=False)
    n_max = sum(r["peak"] == "max" for r in rows)
    ax.set_title(f"NSTX {shot} s-alpha at x0 = {', '.join(f'{x:g}' for x in x0)}: mode at the "
                 f"ion-scale peak, marker area = its gamma" + chr(10)
                 + f"{len(rows)} case(s) (corners, reference, VURS picks), "
                 f"{n_max} without a cusp (max, hollow)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "s_alpha_ion.png", written)
    path = data_path(out, "s_alpha_ion.csv")
    keys = ("case", "scan", "ref", "alpha_MHD", "shat", "ky_ion", "gamma_ion", "mode_ion", "peak")
    with open(path, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                              for k in keys) + "\n")
    written.append(path)


def local_params(ds) -> list[str]:
    return [k for k in ds.data_vars
            if ds[k].dims == ("corner",) and k not in ("gamma_max", "ky_at_max")
            and np.issubdtype(ds[k].dtype, np.number)]


def varying(ds, spread=SPREAD, max_params=MAX_PARAMS) -> list[tuple[str, float]]:
    """(name, (max-min)/max|value|) of the local parameters above *spread*, largest first."""
    out = []
    rows = corner_rows(ds)
    for k in local_params(ds):
        v = ds[k].values[rows]
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        rel = np.ptp(v) / max(abs(v).max(), 1e-300)
        if rel > spread:
            out.append((k, float(rel)))
    return sorted(out, key=lambda t: -t[1])[:max_params]


def plot_local(ds, out, written, spread=SPREAD, max_params=MAX_PARAMS):
    keys = local_params(ds)
    if not keys:
        print("no local parameters (no GENE parameters read); bar chart and table skipped")
        return
    csv = data_path(out, "corners_local_params.csv")
    cols = ["gamma_max", "ky_at_max"] + keys
    with open(csv, "w") as fh:
        fh.write(",".join(["corner", *ORDER] + cols) + "\n")
        for i, c in enumerate(ds.corner.values):
            fh.write(",".join([c, *(f"{ds[k].values[i]:g}" for k in ORDER)]
                              + [f"{ds[k].values[i]:.6g}" for k in cols]) + "\n")
    written.append(csv)

    picked = varying(ds, spread, max_params)
    if not picked:
        print(f"no local parameter varies by more than {spread:g} across corners")
        return
    print("varying local parameters, (max-min)/max|value|: "
          + ", ".join(f"{k} {r:.3f}" for k, r in picked))
    names = [k for k, _ in picked]
    # % from the mean means nothing when the mean sits near zero, so parameters
    # that change sign across the corners get their own panel in absolute units.
    rows = corner_rows(ds)                         # mean and sign over the corners only
    crosses = [k for k in names
               if np.nanmin(ds[k].values[rows]) < 0 < np.nanmax(ds[k].values[rows])]
    groups = [(r"% from the corner mean", [k for k in names if k not in crosses], True),
              ("value - corner mean", crosses, False)]
    groups = [grp for grp in groups if grp[1]]
    spread_of = dict(picked)
    n = ds.sizes["corner"]
    width = 0.84 / n
    fig, axes = plt.subplots(1, len(groups), figsize=(max(10, 1.25 * len(names) + 4), 5.8),
                             width_ratios=[len(grp[1]) + 0.6 for grp in groups], squeeze=False)
    for ax, (ylabel, keys_, pct) in zip(axes[0], groups):
        mean = [np.nanmean(ds[k].values[rows]) for k in keys_]
        for i, c in enumerate(ds.corner.values):
            dev = [100 * (ds[k].values[i] / m - 1) if pct else ds[k].values[i] - m
                   for k, m in zip(keys_, mean)]
            ax.bar(np.arange(len(keys_)) - 0.42 + (i + 0.5) * width, dev, width * 0.9,
                   color=style(point_of(ds, c))["color"],
                   label=str(ds.label.values[i]) if ax is axes[0, 0] else None)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(np.arange(len(keys_)),
                      [f"{k}\n{100 * spread_of[k]:.0f}%" for k in keys_], fontsize=8)
        ax.set_ylabel(ylabel)
        tidy(ax)
    axes[0, 0].set_title(f"local parameters at {x0_text(ds)} with (max-min)/max|value| > "
                         f"{spread:g} (spread under each name)", loc="left", fontsize=10,
                         color=INK)
    if crosses:
        axes[0, -1].set_title("sign-changing, absolute", loc="left", fontsize=10, color=INK)
    fig.legend(title=legend_title(ds), loc="center right", frameon=False, fontsize=8,
               title_fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    save(fig, out, "corners_local_params.png", written)

    tcols = names + ["gamma_max", "ky_at_max"]
    cells = [[f"{v:.4g}" if np.isfinite(v) else "-" for v in (ds[k].values[i] for k in tcols)]
             for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(8, 1.05 * len(tcols) + 2), 0.26 * n + 1.0))
    ax.axis("off")
    tab = ax.table(cellText=cells, rowLabels=[str(l) for l in ds.label.values],
                   colLabels=tcols, loc="center", cellLoc="right")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7.5)
    tab.scale(1, 1.25)
    ax.set_title(f"local parameters per corner, rows {legend_title(ds)}", loc="left",
                 fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "corners_local_table.png", written)


# ----------------------------------------------------------------------------- sampling

def sampled_cases(diag_root: str, x0) -> list[dict]:
    """Every case dir under <diag_root>/x0_<x0>/ named like a ped_grid tag (the
    corners, the reference, VURS picks), oldest GENE output first: its knobs,
    kind ('corner', 'ref' or 'sampled'), the kymin of each run and the time of
    its first parameters file (nan when GENE has written nothing yet)."""
    root = os.path.join(diag_root, x0_label(x0))
    cases = []
    for name in _listdir(root):
        try:
            p = parse_tag(name)
        except ValueError:                          # plots/, *_rounds/, anything else
            continue
        runs = gene_runs(os.path.join(root, name))
        kymin = []
        for path in runs:
            try:
                with open(path) as fh:
                    m = KYMIN_RE.search(fh.read())
            except OSError:
                continue
            if m:
                kymin.append(float(m.group(1)))
        kind = ("ref" if is_ref(p) else
                "corner" if all(p[k] in CORNER_LEVELS for k in ORDER) else "sampled")
        cases.append({"tag": name, **p, "kind": kind, "kymin": sorted(kymin),
                      "time": min((os.path.getmtime(f) for f in runs), default=np.nan)})
    return sorted(cases, key=lambda c: (np.isnan(c["time"]), c["time"], c["tag"]))


def kde(points, grid, bw=SAMPLE_BW):
    """Gaussian kernel density of *points* (n, d) on *grid* (m, d), peak-normalised."""
    if not len(points):
        return np.zeros(len(grid))
    d2 = ((grid[:, None, :] - points[None, :, :]) / bw) ** 2
    dens = np.exp(-0.5 * d2.sum(axis=2)).sum(axis=1)
    return dens / dens.max()


def sample_colours(cases):
    """Per case: grey for corners, black for the reference, the sampling order on
    SAMPLE_CMAP (dark = first) for VURS picks; and that order's Normalize."""
    order = [i for i, c in enumerate(cases) if c["kind"] == "sampled"]
    norm = Normalize(1, max(len(order), 2))
    cmap = plt.get_cmap(SAMPLE_CMAP)
    rank = {i: n for n, i in enumerate(order, start=1)}
    cols = [INK if c["kind"] == "ref" else MUTED if c["kind"] == "corner" else cmap(norm(rank[i]))
            for i, c in enumerate(cases)]
    return cols, norm, cmap


def plot_sampling(cases, ped_root, x0, shot, out, written):
    """Where the scan has sampled the four profile scalings at this x0: every
    knob pair as a scatter over the kernel density of the points GENE has
    run (lower triangle), each knob's 1D density with a rug (diagonal), the Te
    and ne pedestals of every sampled case, and the kymin each one ran.

    Corners are grey squares, the reference a black diamond, VURS picks circles
    coloured by when GENE first wrote them (dark = first), hollow where GENE
    has written nothing yet; + sampling.csv."""
    if not cases:
        print("no case dirs to place; sampling.png skipped")
        return
    cols, norm, cmap = sample_colours(cases)
    ran = np.array([np.isfinite(c["time"]) for c in cases])
    pts = np.array([[c[k] for k in ORDER] for c in cases])
    marker = {"corner": "s", "ref": "D", "sampled": "o"}
    n = len(ORDER)
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(n, n, hspace=0.3, wspace=0.15)
    grid1 = np.linspace(*SAMPLE_LIM, 200)
    gx, gy = np.meshgrid(grid1, grid1)
    for i in range(n):
        for j in range(i + 1):
            ax = fig.add_subplot(gs[i, j])
            if i == j:                              # 1D density: all GENE runs, and VURS alone
                dens = kde(pts[ran][:, [i]], grid1[:, None])
                ax.fill_between(grid1, dens, color=GRID, lw=0)
                vurs = ran & np.array([c["kind"] == "sampled" for c in cases])
                if vurs.any():
                    ax.plot(grid1, kde(pts[vurs][:, [i]], grid1[:, None]),
                            color=SAMPLE_LINE, lw=1.6)
                for v, col in zip(pts[:, i], cols):
                    ax.plot([v, v], [0, 0.08], color=col, lw=1.2)
                ax.set_ylim(0, 1.05)
                ax.set_yticks([])
            else:                                   # knob pair: density + points
                dens = kde(pts[ran][:, [j, i]], np.column_stack([gx.ravel(), gy.ravel()]))
                ax.contourf(gx, gy, dens.reshape(gx.shape), levels=np.linspace(0.05, 1, 7),
                            cmap=LinearSegmentedColormap.from_list("kde", [SURFACE, "#b9b7ad"]),
                            zorder=0)
                for c, v, col, r in zip(cases, pts, cols, ran):
                    ax.plot(v[j], v[i], marker[c["kind"]], ms=6 if c["kind"] != "sampled" else 5,
                            mfc=col if r else "none", mec=col, mew=1.2, zorder=3)
                ax.set_ylim(*SAMPLE_LIM)
            ax.set_xlim(*SAMPLE_LIM)
            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(ORDER[j])
            if j > 0 and i != j:
                ax.set_yticklabels([])
            elif i != j:
                ax.set_ylabel(ORDER[i])
            tidy(ax)

    # upper triangle: the profiles behind the points, and the kymin each case ran
    units = {v: u for v, _, u in ITERDB_VARS.values()}
    prof = {c["tag"]: iterdb_profiles(os.path.join(ped_root, c["tag"])) for c in cases}
    for row, v in enumerate(("Te", "ne")):
        ax = fig.add_subplot(gs[row, 2:])
        lo, hi = np.inf, -np.inf
        for c, col in zip(cases, cols):
            pr = (prof[c["tag"]] or {}).get(v)
            if pr is None:
                continue
            r, y = pr
            ax.plot(r, y, color=col, lw=1.6 if c["kind"] == "ref" else 1.0,
                    alpha=1.0 if c["kind"] != "corner" else 0.7,
                    zorder=4 if c["kind"] == "ref" else 3)
            win = (r >= PED_ZOOM[0]) & (r <= PED_ZOOM[1])
            if win.any():
                lo, hi = min(lo, np.nanmin(y[win])), max(hi, np.nanmax(y[win]))
        ax.axvline(float(x0[0]), color=INK, ls=":", lw=1.4)
        ax.set_xlim(*PED_ZOOM)
        if np.isfinite(lo):
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(rf"{v} [{units[v]}] vs $\rho_{{tor}}$, every case (profiles.iterdb)",
                     loc="left", fontsize=10, color=INK)
        if not row:
            ax.set_xticklabels([])
        tidy(ax)
    ax = fig.add_subplot(gs[2, 3])
    kys = [(k, col) for c, col in zip(cases, cols) for k in c["kymin"]]
    if kys:
        ky = np.array([k for k, _ in kys])
        span = np.linspace(ky.min() - 0.1, ky.max() + 0.1, 300)
        ax.fill_between(span, kde(ky[:, None], span[:, None], SAMPLE_KY_BW), color=GRID, lw=0)
        for k, col in kys:
            ax.plot([k, k], [0, 0.1], color=col, lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xlabel(r"$k_y\rho_{ref}$ run")
    ax.set_title(f"kymin density (bw {SAMPLE_KY_BW:g})", loc="left", fontsize=9, color=INK)
    tidy(ax)

    count = {k: sum(c["kind"] == k for c in cases) for k in marker}
    handles = [Line2D([], [], ls="none", marker="s", color=MUTED, ms=6,
                      label=f"corner ({count['corner']})"),
               Line2D([], [], ls="none", marker="D", color=INK, ms=6,
                      label=f"(1, 1, 1, 1) ref ({count['ref']})"),
               Line2D([], [], ls="none", marker="o", color=cmap(0.5), ms=6,
                      label=f"VURS pick ({count['sampled']}), by order"),
               Line2D([], [], ls="none", marker="o", mfc="none", mec=MUTED, mew=1.2, ms=6,
                      label=f"no GENE output yet ({int((~ran).sum())})"),
               Line2D([], [], color=INK, ls=":", lw=1.4, label=f"x0 = {float(x0[0]):g}")]
    handles.insert(3, Line2D([], [], color=SAMPLE_LINE, lw=1.6, label="VURS picks' density"))
    handles.insert(4, Line2D([], [], color=GRID, lw=6, label="density of every case run"))
    key = fig.add_subplot(gs[0, 1])                 # empty upper-triangle cell: the key
    key.set_axis_off()
    key.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5)
    if count["sampled"]:
        cax = key.inset_axes((0.05, 0.08, 0.9, 0.06))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                          orientation="horizontal")
        cb.set_label("VURS sampling order (first GENE output)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(f"NSTX {shot} sampled profile scalings at x0 = {float(x0[0]):g}: "
                 f"{len(cases)} cases, {int(ran.sum())} with GENE output, "
                 f"{len(kys)} runs (shading: kernel density, bw {SAMPLE_BW:g})",
                 x=0.06, y=0.94, ha="left", fontsize=12, color=INK)
    save(fig, out, "sampling.png", written)

    path = data_path(out, "sampling.csv")
    with open(path, "w") as fh:
        fh.write(",".join(["case", *ORDER, "kind", "n_runs", "kymin", "first_output_utc"]) + "\n")
        for c in cases:
            when = ("" if np.isnan(c["time"]) else
                    np.datetime_as_string(np.datetime64(int(c["time"]), "s")))
            fh.write(",".join([c["tag"], *(f"{c[k]:g}" for k in ORDER), c["kind"],
                               str(len(c["kymin"])), " ".join(f"{k:g}" for k in c["kymin"]),
                               when]) + "\n")
    written.append(path)


def batches(cases, size=BATCH) -> list[tuple[str, list[dict]]]:
    """The VURS picks of sampled_cases in sampling order, in groups of *size*:
    [('batch1_16', points), ('batch17_32', points), ...]. A batch is named for
    its whole range, so the last one keeps its name while it fills up."""
    picks = [{k: c[k] for k in ORDER} for c in cases if c["kind"] == "sampled"]
    return [(f"batch{i + 1}_{i + size}", picks[i:i + size]) for i in range(0, len(picks), size)]


def plot_scan(ds, out, written, spread=SPREAD, max_params=MAX_PARAMS, res2_roots=(),
              nz_compare=None, nz_root=None, mode_extent=False, vspace_edge=False,
              vspace_tol=VSP_TOL, group_tol=GROUP_TOL):
    """Every per-scan plot of one dataset (the corners, or a VURS batch) into out/."""
    os.makedirs(out, exist_ok=True)
    plot_growth(ds, out, written)
    plot_growth(ds, out, written, guesses=True)
    plot_profiles(ds, out, written)
    plot_profiles(ds, out, written, PED_ZOOM, "corners_profiles_ped.png")
    plot_s_alpha(ds, out, written)
    plot_s_alpha_ion(ds, out, written)
    plot_local(ds, out, written, spread, max_params)
    plot_q_offset(ds, out, written)
    if not is_named(ds):                          # 2x reruns are found by ped_grid tag
        plot_res2(ds, out, written, res2_roots)
    if nz_compare:
        plot_nz_compare(ds, out, written, nz_compare, nz_root)
    if mode_extent:
        profiles, nx0 = load_profiles(ds)
        plot_mode_extent(ds, out, written, profiles=profiles, nx0=nx0)
        plot_mode_structure(ds, out, written, profiles)
        plot_mode_groups(ds, out, written, profiles, group_tol)
    if vspace_edge:
        plot_vspace_edge(ds, out, written, vspace_tol)
    if "outlier" in ds:
        plot_outliers(ds, out, written)
    if "mode" in ds:
        write_modes_csv(ds, out, written)
        plot_mode_scaling(ds, out, written)


def run(diag_root=DIAG_ROOT, ped_root=PED_ROOT, out=None, spread=SPREAD, max_params=MAX_PARAMS,
        x0=X0, modes=False, field_fits=False, shot=SHOT, res2_roots=(), nz_compare=None,
        nz_root=None, mode_extent=False, vspace_edge=False, vspace_tol=VSP_TOL, batch=BATCH,
        extra_roots=(), outliers=False, cases=None, scan=None, group_tol=GROUP_TOL):
    """The corners into <out>/corners/, each VURS batch into <out>/batch<i>_<j>/,
    and the sampling plot of all of them into <out>/. Returns the corner dataset
    (None when there are no corners but batches were plotted). GENE runs come from
    diag_root, its _ky* siblings and extra_roots (ky_roots).

    cases: named cases (case_point) to plot instead, into <out>/corners/ with the
    same per-scan plots (no batches, sampling, 2x or nz0 checks); scan titles them."""
    x0 = tuple(x0)
    out = out or default_out(x0, diag_root)
    os.makedirs(out, exist_ok=True)
    written: list[str] = []
    if cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ds = load_dataset(diag_root, ped_root, x0, modes, field_fits, shot, list(cases),
                              scan or "named cases", extra_roots, outliers)
            plot_scan(ds, os.path.join(out, CORNER_DIR), written, spread, max_params,
                      mode_extent=mode_extent, vspace_edge=vspace_edge, vspace_tol=vspace_tol,
                      group_tol=group_tol)
        print("\nwrote\n  " + "\n  ".join(written))
        return ds
    cases = sampled_cases(diag_root, x0)
    corner_ds, plotted, s_alpha = None, 0, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN corners and rows
        for name, pts in [(CORNER_DIR, None)] + batches(cases, batch):
            print(f"\n===== {name} =====")
            scan = "corners" if pts is None else f"VURS picks {name[5:].replace('_', '-')}"
            try:
                ds = load_dataset(diag_root, ped_root, x0, modes, field_fits, shot, pts, scan,
                                  extra_roots, outliers)
            except SystemExit as exc:                  # no corner (or pick) has anything yet
                print(exc)
                continue
            if pts is None:
                corner_ds = ds
                plot_scan(ds, os.path.join(out, name), written, spread, max_params, res2_roots,
                          nz_compare, nz_root, mode_extent, vspace_edge, vspace_tol, group_tol)
            else:                                      # 2x and nz0 checks are corner-only
                plot_scan(ds, os.path.join(out, name), written, spread, max_params,
                          mode_extent=mode_extent, vspace_edge=vspace_edge, vspace_tol=vspace_tol,
                          group_tol=group_tol)
            plotted += 1
            s_alpha += ion_rows(ds)
        plot_sampling(cases, ped_root, x0, shot, out, written)
        plot_s_alpha_x0(s_alpha, x0, shot, out, written)
    print("\nwrote\n  " + "\n  ".join(written))
    if not plotted:
        raise SystemExit(f"no corner or VURS pick found under {diag_root} or {ped_root}")
    return corner_ds


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--shot", type=int, default=SHOT, choices=sorted(SHOTS),
                    help="discharge; sets the --diag-root, --ped-root and --x0 defaults")
    ap.add_argument("--diag-root", default=None,
                    help="GENE output root; corners are under <diag-root>/x0_<x0>/<tag>/ "
                         f"[{DIAG_ROOT}]")
    ap.add_argument("--ped-root", default=None,
                    help=f"ped_grid cheaseBS cases <tag>/ [{PED_ROOT}]")
    ap.add_argument("--x0", type=float, nargs="+", default=None,
                    help="radius of the scan to plot (reads its x0_<x0>/ folder) "
                         "[Te-top Chapman radius]")
    ap.add_argument("--out", default=None,
                    help="directory for the PNGs, CSVs in its data/ "
                         "[<diag-root>/x0_<x0>/plots]")
    ap.add_argument("--field-fits", action="store_true",
                    help="carry recorded field fits that did not pass (decaying/ambiguous, "
                         "from field_omega.py --write) into the mode table, modes CSV and "
                         "gamma_max (the growth plots show every fit regardless)")
    ap.add_argument("--modes", action="store_true",
                    help="classify each ky with TPED mode_identifier (reads nrg, field, "
                         "geometry; slower) and mark the growth plot by mode")
    ap.add_argument("--res2-root", nargs="+", default=(),
                    help="more dirs to search for 2x-resolution runs (2XRES / _res2 in the "
                         "path) [<diag-root> and its _res2 siblings]")
    ap.add_argument("--extra-root", nargs="+", default=(),
                    help="more GENE roots whose x0_<x0>/<tag>/ runs merge by ky with the "
                         "corners [<diag-root>_ky* siblings]")
    ap.add_argument("--nz-compare", type=int, default=None, metavar="N",
                    help="compare each corner with its nz0 = N run of the same profile "
                         "scaling (corners_nz<N>.png) [off]")
    ap.add_argument("--nz-root", default=None,
                    help="diag root of the --nz-compare runs, <tag>s under x0_<x0>/ "
                         "[<diag-root>_nz<N>]")
    ap.add_argument("--mode-extent", action="store_true",
                    help="corners_mode_extent.png: |phi|, |Apar| at the ballooning z ends over "
                         "their peak, per corner and ky; corners_mode_structure.png: the whole "
                         "|phi|, |Apar| along z per corner and ky; corners_mode_groups.png: "
                         "converged runs grouped by mode-structure shape (reads every field "
                         "file)")
    ap.add_argument("--group-tol", type=float, default=GROUP_TOL,
                    help="corners_mode_groups.png: earth mover's distance along z, in mode "
                         "widths, under which mode structures share a group (average linkage)")
    ap.add_argument("--vspace-edge", action="store_true",
                    help="corners_vspace_edge.png: <|f|^2>^1/2 at the v_par = +-lv and top-mu "
                         "edges of the velocity box over its peak, per species, corner and ky "
                         "(reads every vsp file; needs istep_vsp > 0)")
    ap.add_argument("--outliers", action="store_true",
                    help="corners_outlier.png: TPED's outlier analysis per corner and ky "
                         "(grid-scale oscillation along z, gamma < 0, D/chi_e, chi_i/chi_e, x0; "
                         "reads nrg and field, every run converged or not)")
    ap.add_argument("--vspace-tol", type=float, default=VSP_TOL,
                    help="edge / peak above which --vspace-edge marks a run as failing")
    ap.add_argument("--spread", type=float, default=SPREAD,
                    help="(max-min)/|mean| above which a local parameter counts as varying")
    ap.add_argument("--max-params", type=int, default=MAX_PARAMS,
                    help="most varying parameters to chart and tabulate")
    ap.add_argument("--case", nargs="+", default=None, metavar="NAME[=X0]",
                    help="named case dirs to plot instead of the ped_grid corners: GENE under "
                         "<diag-root>/x0_<X0>/<NAME>/, profiles under <ped-root>/<NAME>/ "
                         "(X0 defaults to --x0), e.g. --case U0=0.808 U=0.8173")
    ap.add_argument("--ref-case", default=None, metavar="NAME",
                    help="--case drawn as the reference (black, first)")
    ap.add_argument("--scan-name", default=None, help="--case: the scan's name in titles")
    args = ap.parse_args()
    d = shot_defaults(args.shot)
    cases = None
    if args.case:
        cases = []
        for kv in args.case:
            name, _, cx0 = kv.partition("=")
            cases.append(case_point(name, float(cx0) if cx0 else None, name == args.ref_case))
        if args.ref_case and args.ref_case not in [c["name"] for c in cases]:
            ap.error(f"--ref-case {args.ref_case} is not one of --case")
    run(args.diag_root or d["diag_root"], args.ped_root or d["ped_root"], args.out, args.spread,
        args.max_params, args.x0 or d["x0"], args.modes, args.field_fits, args.shot,
        args.res2_root, args.nz_compare, args.nz_root, args.mode_extent, args.vspace_edge,
        args.vspace_tol, extra_roots=args.extra_root, outliers=args.outliers, cases=cases,
        scan=args.scan_name, group_tol=args.group_tol)
