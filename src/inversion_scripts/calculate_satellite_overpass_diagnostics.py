# Sample all diagnostics at satellite overpass time
import sys
import os
import glob
import tempfile
import shutil
import numpy as np
import xarray as xr
import yaml
from datetime import datetime, timedelta
import joblib
from joblib import Parallel, delayed
from contextlib import ExitStack

from src.utilities.generate_overpass_grids import generate_on_sim_grid

from src.inversion_scripts.utils import (
    check_is_OH_element,
    check_is_BC_element,
    build_pert_simulations_dict,
    get_shared_end_date,
    read_stage_marker,
    write_stage_marker,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"xarray.*"
)

warnings.filterwarnings(
    "ignore",
    message="Duplicate dimension names present.*"
)

MwAir = 28.97  # g/mol


def write_netcdf_atomic(output_ds, output_fpath):
    """Write a NetCDF file without exposing a partially written final file."""
    output_dir = os.path.dirname(output_fpath)
    output_basename = os.path.basename(output_fpath)

    os.makedirs(output_dir, exist_ok=True)

    # Undotted so a leftover from a hard kill is visible, and suffixed .tmp so
    # the S3 upload filters exclude it -- ".tmp.nc4" matched neither.
    fd, tmp_fpath = tempfile.mkstemp(
        prefix=f"{output_basename}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(fd)

    try:
        output_ds.to_netcdf(
            tmp_fpath,
            mode="w",
        )

        os.replace(
            tmp_fpath,
            output_fpath,
        )

    except Exception:
        try:
            os.remove(tmp_fpath)
        except FileNotFoundError:
            pass

        raise


def require_input_file(file_path, purpose):
    """Raise when a required OutputDir input is absent.

    An absent input used to be treated as "no data", leaving the affected
    cells NaN while the diagnostic was still written. That is correct only
    for dates outside [StartDate, EndDate), where no simulation output
    exists by construction. Once OutputDir is pruned for already-processed
    dates, an in-window input that has been deleted, or not restored, would
    otherwise be indistinguishable from a completed date, so it is a hard
    error instead.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Required {purpose} input is missing: {file_path}"
        )

    return file_path


# ---------------------------------------------------------------------------
# Helper: build date list
# ---------------------------------------------------------------------------
def local_end_exclusive_for(shared_end_date, utc_end):
    """First local date NOT coverable given a shared end date.

    While simulations are progressing, the latest local date is excluded
    because it may require UTC data from the following day. Once the shared
    end date reaches EndDate, local dates through EndDate - 1 are all covered.

    Shared by the current window and by the one a marker records, so the two
    can never disagree about what "finished" meant.
    """
    local_end_exclusive = datetime.strptime(shared_end_date, "%Y%m%d")

    if local_end_exclusive != utc_end:
        local_end_exclusive -= timedelta(days=1)

    return local_end_exclusive


def build_date_list(config):
    """Return local overpass dates needed to cover the UTC simulation window.

    StartDate and EndDate are UTC dates, with EndDate exclusive.

    Dates an existing marker already covers are dropped. The marker is written
    only after every run and date completed, so re-deriving its window and
    resuming from the end of it repeats no work. This is not only a saving:
    once prune_outputdir.py has removed the OutputDir dates both stages were
    finished with, the inputs for those dates are gone, and asking for them
    again would raise rather than quietly produce nothing.

    Set OVERPASS_IGNORE_MARKER=1 to process the full window regardless. That
    requires the OutputDir inputs to still be present.
    """
    StartDate = str(config["StartDate"])
    EndDate = str(config["EndDate"])

    start = datetime.strptime(StartDate, "%Y%m%d")
    end = datetime.strptime(EndDate, "%Y%m%d")  # exclusive UTC end

    RunName = config["RunName"]
    RunDirs = os.path.join(
        os.path.expandvars(config["OutputPath"]),
        config["RunName"],
    )

    shared_end_date = get_shared_end_date(
        jacobian_root=os.path.join(RunDirs, "jacobian_runs"),
        run_name=RunName,
        start_date=StartDate,
    )

    print(f"Latest shared date (exclusive): {shared_end_date}")

    local_start = start - timedelta(days=1)
    local_end_exclusive = local_end_exclusive_for(shared_end_date, end)

    n_process = (local_end_exclusive - local_start).days

    date_list = [
        (local_start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(n_process)
    ]

    if os.environ.get("OVERPASS_IGNORE_MARKER") == "1":
        print("OVERPASS_IGNORE_MARKER=1: processing the full window")
        return start, end, date_list, shared_end_date

    marker_end_date = read_stage_marker(RunDirs, "overpass", StartDate)

    if marker_end_date is not None:
        done_through = local_end_exclusive_for(marker_end_date, end)
        resume_from = done_through.strftime("%Y%m%d")
        already_done = [d for d in date_list if d < resume_from]
        date_list = [d for d in date_list if d >= resume_from]

        print(
            f"Marker S{marker_end_date}: {len(already_done)} local date(s) "
            f"through {(done_through - timedelta(days=1)).strftime('%Y%m%d')} "
            f"already complete, skipping them"
        )

        if not date_list:
            print("Nothing new to process")
        else:
            print(f"Resuming at {date_list[0]}, {len(date_list)} date(s) to go")

    return start, end, date_list, shared_end_date


# ---------------------------------------------------------------------------
# Helper: load or generate the overpass grid
# ---------------------------------------------------------------------------
def load_overpass_grid(
    config,
    JacobianRunDirs,
    CSgridDir,
    StartDate,
):
    """Load the overpass-time grid, generating it from a sample file if needed."""
    OverpassTime = config["OverpassTime"]
    OrbitDirection = config["OrbitDirection"]
    OrbitsPerDay = config["OrbitsPerDay"]
    RunName = config["RunName"]

    os.makedirs(CSgridDir, exist_ok=True)

    overpass_grid_fpath = os.path.join(
        CSgridDir,
        "overpass_sample_utc_hour.nc",
    )

    if not os.path.isfile(overpass_grid_fpath):
        Jacobian_RunDir = os.path.join(
            JacobianRunDirs,
            f"{RunName}_0001",
        )

        # Only lons/lats are read from this file, and those do not vary with
        # date, so any SpeciesConc file from the run serves. Falling back to
        # whatever is present keeps this working once OutputDir has been
        # pruned back to its most recent dates.
        grid_file = os.path.join(
            Jacobian_RunDir,
            f"OutputDir/GEOSChem.SpeciesConc.{StartDate}_0000z.nc4",
        )

        if not os.path.isfile(grid_file):
            available = sorted(glob.glob(os.path.join(
                Jacobian_RunDir,
                "OutputDir/GEOSChem.SpeciesConc.*.nc4",
            )))

            if not available:
                raise FileNotFoundError(
                    "Cannot build the overpass grid: no SpeciesConc file in "
                    f"{Jacobian_RunDir}/OutputDir, and "
                    f"{overpass_grid_fpath} does not exist. Restore it from "
                    "the archive."
                )

            grid_file = available[-1]

        overpass_ds = generate_on_sim_grid(
            grid_file,
            OverpassTime,
            overpass_grid_fpath,
            orbits_per_day=OrbitsPerDay,
            direction=OrbitDirection,
            isGCHP=config["UseGCHP"],
        )
    else:
        overpass_ds = xr.open_dataset(overpass_grid_fpath)

    return overpass_ds


# ---------------------------------------------------------------------------
# Helper: derive grid metadata from the overpass dataset
# ---------------------------------------------------------------------------
def get_grid_info(overpass_ds, use_gchp):
    """Return grid dimensions, coordinates, and overpass-time information."""
    if use_gchp:
        lon_name = "lons"
        lat_name = "lats"
        dims = ("nf", "Ydim", "Xdim")

        coords = dict(
            lats=(
                ["nf", "Ydim", "Xdim"],
                overpass_ds["lats"].values,
            ),
            lons=(
                ["nf", "Ydim", "Xdim"],
                overpass_ds["lons"].values,
            ),
        )
    else:
        lon_name = "lon"
        lat_name = "lat"
        dims = ("lat", "lon")

        coords = dict(
            lat=(
                ["lat"],
                overpass_ds["lat"].values,
            ),
            lon=(
                ["lon"],
                overpass_ds["lon"].values,
            ),
        )

    closest_hour = overpass_ds["closest_hour"].values
    day_offset = overpass_ds["day_offset"].values

    if np.issubdtype(day_offset.dtype, np.timedelta64):
        day_offset = (
            day_offset
            .astype("timedelta64[D]")
            .astype(int)
        )
    else:
        day_offset = day_offset.astype(int)

    assert not np.any(day_offset == -1), (
        "day_offset=-1 cells present; "
        "previous day's file lookup not implemented"
    )

    return (
        dims,
        coords,
        lon_name,
        lat_name,
        closest_hour,
        day_offset,
    )


# ---------------------------------------------------------------------------
# Helper: determine which tracer variables to read from a Jacobian run
# ---------------------------------------------------------------------------
def get_keepvars(
    sv_elems,
    n_elements,
    config,
    baserun=False,
):
    """Return the SpeciesConc variable names needed from one Jacobian run."""
    if sv_elems == [0]:
        return ["SpeciesConcVV_CH4"]

    local_indices = range(1, len(sv_elems) + 1)

    keepvars = [
        f"SpeciesConcVV_CH4_jac{idx:04d}"
        for idx in local_indices
    ]

    if baserun:
        keepvars.append("SpeciesConcVV_CH4")

    if len(keepvars) == 1:
        is_Regional = config["isRegional"]

        is_OH_element = check_is_OH_element(
            sv_elems[0],
            n_elements,
            config["OptimizeOH"],
            is_Regional,
        )

        is_BC_element = check_is_BC_element(
            sv_elems[0],
            n_elements,
            config["OptimizeOH"],
            config["OptimizeBCs"],
            is_OH_element,
            is_Regional,
        )

        if is_OH_element or is_BC_element:
            keepvars = ["SpeciesConcVV_CH4"]

    return keepvars


# ---------------------------------------------------------------------------
# Helper: read meteorological fields for one day
# ---------------------------------------------------------------------------
def load_met_fields(met_file):
    """Return AirDen and BxH arrays from a SpeciesConc file."""
    with xr.open_dataset(met_file) as met_ds:
        AirDen = met_ds["Met_AIRDEN"].values * 1e3
        BxH = met_ds["Met_BXHEIGHT"].values

    return AirDen, BxH


# ---------------------------------------------------------------------------
# Helper: pre-save met arrays as .npy for memory-mapped access
# ---------------------------------------------------------------------------
def _presave_one_date(
    date_str,
    utc_start,
    utc_end,
    JacobianRunDirs,
    RunName,
    run_id,
    prefix,
    tmp_dir,
):
    """Read one date's met fields and write them to tmp_dir as .npy.

    Returns (date_str, paths) with paths None for a date outside the window.
    Split out of presave_met_arrays so the dates can be read in parallel:
    each one opens a different file and writes different outputs, so there is
    nothing shared to serialise on.
    """
    date_dt = datetime.strptime(date_str, "%Y%m%d")

    if not utc_start <= date_dt < utc_end:
        return date_str, None

    met_file = os.path.join(
        JacobianRunDirs,
        f"{RunName}_{run_id:04d}",
        f"OutputDir/GEOSChem.{prefix}.{date_str}_0000z.nc4",
    )

    require_input_file(met_file, "met")

    AirDen, BxH = load_met_fields(met_file)

    airden_fpath = os.path.join(tmp_dir, f"AirDen_{date_str}.npy")
    bxh_fpath = os.path.join(tmp_dir, f"BxH_{date_str}.npy")

    np.save(airden_fpath, AirDen)
    np.save(bxh_fpath, BxH)

    del AirDen, BxH

    return date_str, {"AirDen": airden_fpath, "BxH": bxh_fpath}


def presave_met_arrays(
    date_list,
    utc_start,
    utc_end,
    JacobianRunDirs,
    RunName,
    tmp_dir,
    DisableRun0000=False,
    n_workers=1,
):
    """Pre-save each required UTC met date once for memory-mapped access."""
    if DisableRun0000:
        run_id = 1
        prefix = "BaseSpeciesConc"
    else:
        run_id = 0
        prefix = "SpeciesConc"

    required_dates = set(date_list)

    for date_str in date_list:
        date_dt = datetime.strptime(date_str, "%Y%m%d")
        required_dates.add(
            (date_dt + timedelta(days=1)).strftime("%Y%m%d")
        )

    ordered = sorted(required_dates)
    args = (utc_start, utc_end, JacobianRunDirs, RunName, run_id, prefix, tmp_dir)

    # The first in-window date is read here, alone, to find out how large one
    # date actually is on this grid. Everything after it runs in a pool sized
    # from that measurement. Guessing instead would mean picking a number that
    # is either wasteful at C36 or fatal at C360.
    saved_met = {}
    measured_mb = None
    rest = []

    for i, date_str in enumerate(ordered):
        date_str, paths = _presave_one_date(date_str, *args)
        saved_met[date_str] = paths
        if paths is not None:
            measured_mb = met_arrays_mb(paths)
            rest = ordered[i + 1:]
            break

    if rest:
        # Two dates are live at once in the step that follows -- a date and the
        # one after it -- alongside the simulation output being read against
        # them. OVERPASS_MEM_FACTOR scales the measurement to cover that.
        factor = float(os.environ.get("OVERPASS_MEM_FACTOR", "6"))
        per_worker = int(measured_mb * factor) if measured_mb else None

        pool = memory_capped_workers(n_workers, per_worker)
        if measured_mb:
            print(
                f"Met arrays are {measured_mb} MB/date; "
                f"presaving {len(rest)} more date(s) on {pool} worker(s)"
            )

        saved_met.update(
            dict(
                Parallel(
                    n_jobs=pool,
                    backend="loky",
                    pre_dispatch="2*n_jobs",
                )(
                    delayed(_presave_one_date)(date_str, *args)
                    for date_str in rest
                )
            )
        )

    met_paths = {}

    for date_str in date_list:
        date_dt = datetime.strptime(date_str, "%Y%m%d")
        next_date_str = (
            date_dt + timedelta(days=1)
        ).strftime("%Y%m%d")

        met_paths[date_str] = {
            "current": saved_met.get(date_str),
            "next": saved_met.get(next_date_str),
            "next_date_str": next_date_str,
        }

    return met_paths


# ---------------------------------------------------------------------------
# Helper: compute overpass CH4 columns for all variables
# ---------------------------------------------------------------------------
def compute_overpass_columns(
    keepvars,
    sim_utc_ds,
    sim_utc_ds_nextday,
    AirDen,
    AirDen_nextday,
    BxH,
    BxH_nextday,
    closest_hour,
    day_offset,
):
    """Sample CH4 columns at satellite overpass time."""
    n_vars = len(keepvars)
    spatial_shape = closest_hour.shape

    overpass_CH4_col = np.full(
        (n_vars, *spatial_shape),
        np.nan,
        dtype=np.float32,
    )

    if (
        sim_utc_ds is not None
        and AirDen is not None
        and BxH is not None
    ):
        valid0 = day_offset == 0

        for hr in np.unique(closest_hour[valid0]):
            hr_int = int(hr)
            mask = valid0 & (closest_hour == hr)

            col_weight = (
                AirDen[hr_int][:, mask]
                / MwAir
                * BxH[hr_int][:, mask]
            )

            for i, var in enumerate(keepvars):
                if var not in sim_utc_ds:
                    continue

                ch4 = (
                    sim_utc_ds[var]
                    .isel(time=hr_int)
                    .values[:, mask]
                )

                overpass_CH4_col[i, mask] = (
                    ch4 * col_weight
                ).sum(axis=0)

    if (
        sim_utc_ds_nextday is not None
        and AirDen_nextday is not None
        and BxH_nextday is not None
    ):
        valid1 = day_offset == 1

        for hr in np.unique(closest_hour[valid1]):
            hr_int = int(hr)
            mask = valid1 & (closest_hour == hr)

            col_weight = (
                AirDen_nextday[hr_int][:, mask]
                / MwAir
                * BxH_nextday[hr_int][:, mask]
            )

            for i, var in enumerate(keepvars):
                if var not in sim_utc_ds_nextday:
                    continue

                ch4 = (
                    sim_utc_ds_nextday[var]
                    .isel(time=hr_int)
                    .values[:, mask]
                )

                overpass_CH4_col[i, mask] = (
                    ch4 * col_weight
                ).sum(axis=0)

    return overpass_CH4_col


def sample_overpass_3D(
    sim_utc_ds,
    sim_utc_ds_nextday,
    closest_hour,
    day_offset,
    spatial_dims,
):
    """Sample time-dependent 3D fields at local satellite overpass time."""
    ref_ds = (
        sim_utc_ds
        if sim_utc_ds is not None
        else sim_utc_ds_nextday
    )

    if ref_ds is None:
        return xr.Dataset()

    out_ds = xr.Dataset()

    for cname, coord in ref_ds.coords.items():
        if cname != "time":
            out_ds = out_ds.assign_coords({
                cname: coord
            })

    all_vars = set()

    if sim_utc_ds is not None:
        all_vars.update(sim_utc_ds.data_vars)

    if sim_utc_ds_nextday is not None:
        all_vars.update(sim_utc_ds_nextday.data_vars)

    for var in sorted(all_vars):
        if (
            sim_utc_ds is not None
            and var in sim_utc_ds
        ):
            src_da = sim_utc_ds[var]

        elif (
            sim_utc_ds_nextday is not None
            and var in sim_utc_ds_nextday
        ):
            src_da = sim_utc_ds_nextday[var]

        else:
            continue

        if "time" not in src_da.dims:
            out_ds[var] = src_da
            continue

        if src_da.dims[0] != "time":
            raise ValueError(
                f"{var} has time dimension, but time is not "
                f"axis 0: {src_da.dims}"
            )

        out_dims = tuple(src_da.dims[1:])

        out_shape = tuple(
            src_da.sizes[d]
            for d in out_dims
        )

        if (
            tuple(out_dims[-len(spatial_dims):])
            != tuple(spatial_dims)
        ):
            continue

        sampled = np.full(
            out_shape,
            np.nan,
            dtype=np.float32,
        )

        if (
            sim_utc_ds is not None
            and var in sim_utc_ds
        ):
            valid0 = day_offset == 0

            for hr in np.unique(closest_hour[valid0]):
                hr_int = int(hr)
                mask = valid0 & (closest_hour == hr)

                arr = (
                    sim_utc_ds[var]
                    .isel(time=hr_int)
                    .values
                )

                sampled[..., mask] = arr[..., mask]

        if (
            sim_utc_ds_nextday is not None
            and var in sim_utc_ds_nextday
        ):
            valid1 = day_offset == 1

            for hr in np.unique(closest_hour[valid1]):
                hr_int = int(hr)
                mask = valid1 & (closest_hour == hr)

                arr = (
                    sim_utc_ds_nextday[var]
                    .isel(time=hr_int)
                    .values
                )

                sampled[..., mask] = arr[..., mask]

        out_ds[var] = xr.DataArray(
            sampled,
            dims=out_dims,
            attrs=dict(src_da.attrs),
        )

        out_ds[var].attrs["description"] = (
            "Sampled at local satellite overpass time "
            "without vertical integration."
        )

    return out_ds


def sample_baserun_file_type(
    Jacobian_RunDir,
    file_prefix,
    date_str,
    next_date_str,
    closest_hour,
    day_offset,
    dims,
    require_current=False,
    require_next=False,
):
    """Sample one base-run diagnostic file type at local overpass time.

    require_current/require_next mark the days that fall inside the UTC
    simulation window, whose OutputDir file must therefore be present.
    """
    file_current = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/{file_prefix}.{date_str}_0000z.nc4",
    )

    file_next = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/{file_prefix}.{next_date_str}_0000z.nc4",
    )

    if require_current:
        require_input_file(file_current, file_prefix)

    if require_next:
        require_input_file(file_next, file_prefix)

    with ExitStack() as stack:
        ds_current = (
            stack.enter_context(
                xr.open_dataset(file_current)
            )
            if os.path.isfile(file_current)
            else None
        )

        ds_next = (
            stack.enter_context(
                xr.open_dataset(file_next)
            )
            if os.path.isfile(file_next)
            else None
        )

        sampled_ds = sample_overpass_3D(
            ds_current,
            ds_next,
            closest_hour,
            day_offset,
            dims,
        )

    return sampled_ds


# ---------------------------------------------------------------------------
# Helper: process a single (day, run) pair
# ---------------------------------------------------------------------------
def process_run_day(
    run_i,
    date_str,
    met_paths_day,
    config,
    n_elements,
    JacobianRunDirs,
    pert_simulations_dict,
    overpass_ds,
    closest_hour,
    day_offset,
    dims,
    coords,
    lon_name,
    lat_name,
    DisableRun0000=False,
):
    """Process one Jacobian run for one local overpass date."""
    if DisableRun0000 and run_i == 0:
        return

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"xarray.*",
    )

    warnings.filterwarnings(
        "ignore",
        message="Duplicate dimension names present.*",
    )

    RunName = config["RunName"]
    OverpassTime = config["OverpassTime"]
    overpass_tag = OverpassTime.replace(":", "")

    run_num = str(run_i).zfill(4)
    sv_elems = pert_simulations_dict.get(run_num, [])
    next_date_str = met_paths_day["next_date_str"]

    Jacobian_RunDir = os.path.join(
        JacobianRunDirs,
        f"{RunName}_{run_i:04d}",
    )

    output_dir = os.path.join(
        Jacobian_RunDir,
        "OverpassDiagnostics",
    )

    do_sample_base_3d = (
        run_i == 0
        or (DisableRun0000 and run_i == 1)
    )

    if DisableRun0000 and run_i == 1:
        baserun_file_types = [
            "GEOSChem.BaseSpeciesConc",
            "GEOSChem.StateMetLevEdge",
        ]
    else:
        baserun_file_types = [
            "GEOSChem.SpeciesConc",
            "GEOSChem.StateMetLevEdge",
        ]

    expected_outputs = []

    if do_sample_base_3d:
        expected_outputs.extend(
            os.path.join(
                output_dir,
                (
                    f"{file_prefix}.overpass."
                    f"{date_str}_{overpass_tag}.nc4"
                ),
            )
            for file_prefix in baserun_file_types
        )

    if run_i != 0:
        expected_outputs.append(
            os.path.join(
                output_dir,
                (
                    f"GEOSChem.CH4col.overpass."
                    f"{date_str}_{overpass_tag}.nc4"
                ),
            )
        )

    if expected_outputs and all(
        os.path.isfile(fpath)
        for fpath in expected_outputs
    ):
        return

    if run_i != 0:
        keepvars = get_keepvars(
            sv_elems,
            n_elements,
            config,
            baserun=(run_i == 1),
        )

    current_met = met_paths_day["current"]
    next_met = met_paths_day["next"]

    AirDen = (
        np.load(
            current_met["AirDen"],
            mmap_mode="r",
        )
        if current_met is not None
        else None
    )

    BxH = (
        np.load(
            current_met["BxH"],
            mmap_mode="r",
        )
        if current_met is not None
        else None
    )

    AirDen_nextday = (
        np.load(
            next_met["AirDen"],
            mmap_mode="r",
        )
        if next_met is not None
        else None
    )

    BxH_nextday = (
        np.load(
            next_met["BxH"],
            mmap_mode="r",
        )
        if next_met is not None
        else None
    )

    os.makedirs(output_dir, exist_ok=True)

    def attach_overpass_grid_metadata(output_ds):
        """Attach grid coordinates without duplicating existing variables."""
        if (
            lat_name not in output_ds
            and lat_name in overpass_ds
        ):
            output_ds[lat_name] = overpass_ds[lat_name]

        if (
            lon_name not in output_ds
            and lon_name in overpass_ds
        ):
            output_ds[lon_name] = overpass_ds[lon_name]

        if config.get("UseGCHP", False):
            if (
                "corner_lons" not in output_ds
                and "corner_lons" in overpass_ds
            ):
                output_ds["corner_lons"] = (
                    overpass_ds["corner_lons"]
                )

            if (
                "corner_lats" not in output_ds
                and "corner_lats" in overpass_ds
            ):
                output_ds["corner_lats"] = (
                    overpass_ds["corner_lats"]
                )

        return output_ds

    def add_common_attrs(output_ds):
        """Add date-window metadata shared by all output files."""
        output_ds.attrs["date_interpretation"] = (
            "Filename date is local overpass date. Values are sampled only "
            "where the corresponding UTC simulation date falls within "
            "[StartDate, EndDate). Boundary cells outside the UTC window "
            "are NaN."
        )

        output_ds.attrs["utc_start_date"] = str(
            config["StartDate"]
        )

        output_ds.attrs["utc_end_date_exclusive"] = str(
            config["EndDate"]
        )

        return output_ds

    if do_sample_base_3d:
        for file_prefix in baserun_file_types:
            output_fpath = os.path.join(
                output_dir,
                (
                    f"{file_prefix}.overpass."
                    f"{date_str}_{overpass_tag}.nc4"
                ),
            )

            if os.path.isfile(output_fpath):
                continue

            output_ds = sample_baserun_file_type(
                Jacobian_RunDir,
                file_prefix,
                date_str,
                next_date_str,
                closest_hour,
                day_offset,
                dims,
                require_current=(current_met is not None),
                require_next=(next_met is not None),
            )

            output_ds = attach_overpass_grid_metadata(
                output_ds
            )

            output_ds = add_common_attrs(output_ds)

            write_netcdf_atomic(
                output_ds,
                output_fpath,
            )

        if run_i == 0:
            return

    sim_file_utc = os.path.join(
        Jacobian_RunDir,
        (
            f"OutputDir/GEOSChem.SpeciesConc."
            f"{date_str}_0000z.nc4"
        ),
    )

    sim_file_utc_nextday = os.path.join(
        Jacobian_RunDir,
        (
            f"OutputDir/GEOSChem.SpeciesConc."
            f"{next_date_str}_0000z.nc4"
        ),
    )

    output_fpath = os.path.join(
        output_dir,
        (
            f"GEOSChem.CH4col.overpass."
            f"{date_str}_{overpass_tag}.nc4"
        ),
    )

    if os.path.isfile(output_fpath):
        return

    # current_met/next_met are set only for days inside the UTC simulation
    # window, so those are exactly the days whose SpeciesConc file must exist.
    if current_met is not None:
        require_input_file(sim_file_utc, "SpeciesConc")

    if next_met is not None:
        require_input_file(sim_file_utc_nextday, "SpeciesConc")

    with ExitStack() as stack:
        sim_utc_ds = (
            stack.enter_context(
                xr.open_dataset(sim_file_utc)
            )
            if current_met is not None
            else None
        )

        sim_utc_ds_nextday = (
            stack.enter_context(
                xr.open_dataset(sim_file_utc_nextday)
            )
            if next_met is not None
            else None
        )

        overpass_all = compute_overpass_columns(
            keepvars,
            sim_utc_ds,
            sim_utc_ds_nextday,
            AirDen,
            AirDen_nextday,
            BxH,
            BxH_nextday,
            closest_hour,
            day_offset,
        )

        data_vars = {
            f"{var}_col": (
                dims,
                overpass_all[i],
                {"units": "mol/m2"},
            )
            for i, var in enumerate(keepvars)
        }

        output_ds = xr.Dataset(
            data_vars,
            coords=coords,
        )

        output_ds = attach_overpass_grid_metadata(
            output_ds
        )

        output_ds = add_common_attrs(output_ds)

        write_netcdf_atomic(
            output_ds,
            output_fpath,
        )


def slurm_memory_budget_mb():
    """The job's memory allocation in MB, or None when not under Slurm.

    This is the number the job is killed against, so it is the one worth
    respecting -- not the machine's total memory, which on a shared node
    belongs to other people too.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node and per_node.isdigit():
        return int(per_node)

    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get(
        "SLURM_JOB_CPUS_PER_NODE"
    )
    if per_cpu and per_cpu.isdigit() and cpus and cpus.isdigit():
        return int(per_cpu) * int(cpus)

    return None


def met_arrays_mb(paths):
    """Size of one date's presaved met arrays, in MB, or None if unreadable."""
    if not paths:
        return None
    try:
        total = os.path.getsize(paths["AirDen"]) + os.path.getsize(paths["BxH"])
    except (OSError, KeyError, TypeError):
        return None
    return max(1, total // (1024 * 1024))


def memory_capped_workers(requested, per_worker_mb=None):
    """Lower the worker count to what the memory allocation will hold.

    Each worker holds a date's met fields and the run output it is reading, so
    peak memory scales with how many run at once. With plenty of memory this
    binds on nothing; inside a small allocation it is the difference between
    finishing and being killed part-way, which on this step means losing the
    whole face's diagnostics.

    The per-worker figure should be measured rather than assumed: the arrays
    are a couple of hundred MB at C36 and two orders of magnitude larger at
    C360, so a constant that suits one resolution is dangerously wrong at the
    other. Callers pass what they measured; OVERPASS_MEM_PER_WORKER_MB
    overrides it, and 0 disables the cap.
    """
    override = os.environ.get("OVERPASS_MEM_PER_WORKER_MB")
    if override is not None and override.strip().isdigit():
        per_worker = int(override)
        if per_worker <= 0:
            return requested
    elif per_worker_mb:
        per_worker = per_worker_mb
    else:
        return requested

    budget = slurm_memory_budget_mb()
    if budget is None:
        return requested

    # Leave the parent its own share: it holds the datasets that get handed
    # to every worker, and it is still resident while they run.
    usable = max(0, budget - max(2000, budget // 10))
    allowed = max(1, usable // per_worker)

    if allowed < requested:
        print(
            f"Limiting to {allowed} worker(s): {budget} MB allocated, "
            f"~{per_worker} MB per worker "
            f"(OVERPASS_MEM_PER_WORKER_MB to change)"
        )
        return allowed

    return requested


def split_dates(date_list, n_chunks):
    """Split date_list into at most n_chunks contiguous, near-equal pieces.

    Contiguous rather than round-robin: consecutive dates read neighbouring
    files, and one worker walking a run of them is kinder to the filesystem
    than several interleaving across the whole year.
    """
    if n_chunks <= 1 or len(date_list) <= 1:
        return [date_list]

    n_chunks = min(n_chunks, len(date_list))
    size, extra = divmod(len(date_list), n_chunks)

    chunks = []
    start = 0
    for i in range(n_chunks):
        stop = start + size + (1 if i < extra else 0)
        chunks.append(date_list[start:stop])
        start = stop

    return chunks


def process_run_dates(
    run_i,
    date_list,
    met_paths,
    config,
    n_elements,
    JacobianRunDirs,
    pert_simulations_dict,
    overpass_ds,
    closest_hour,
    day_offset,
    dims,
    coords,
    lon_name,
    lat_name,
    DisableRun0000=False,
):
    """Process all requested dates for one Jacobian run."""
    for date_str in date_list:
        process_run_day(
            run_i,
            date_str,
            met_paths[date_str],
            config,
            n_elements,
            JacobianRunDirs,
            pert_simulations_dict,
            overpass_ds,
            closest_hour,
            day_offset,
            dims,
            coords,
            lon_name,
            lat_name,
            DisableRun0000,
        )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def calculate_satellite_overpass_diagnostics(
    config,
    n_elements,
    n_workers=-1,
):
    """Compute satellite overpass diagnostics for all runs and dates."""
    RunName = config["RunName"]
    OutputPath = os.path.expandvars(config["OutputPath"])
    DisableRun0000 = config.get("DisableRun0000", False)

    if DisableRun0000:
        start_run_num = 1
    else:
        start_run_num = 0

    JacobianRunDirs = os.path.join(
        OutputPath,
        f"{RunName}/jacobian_runs/",
    )

    CSgridDir = os.path.join(
        OutputPath,
        f"{RunName}/CS_grids/",
    )

    start, end, date_list, shared_end_date = build_date_list(config)
    StartDate = str(config["StartDate"])

    # Returned before the grid is loaded and before any OutputDir file is
    # touched. A face whose OutputDir has been pruned has nothing left to read,
    # so reaching further would fail on inputs that are gone by design.
    if not date_list:
        existing = read_stage_marker(
            os.path.join(OutputPath, RunName),
            "overpass",
            StartDate,
        )

        # A marker ahead of the current shared end date means the checkpoints
        # S is derived from went backwards. What it records still happened, so
        # rewriting it downward would discard a true claim; leave it and say so.
        if existing is not None and existing > shared_end_date:
            print(
                f"WARNING: marker S{existing} is ahead of the current shared "
                f"end date {shared_end_date}. Checkpoints appear to have been "
                f"removed. Leaving the marker as it stands."
            )
            return

        write_stage_marker(
            os.path.join(OutputPath, RunName),
            "overpass",
            StartDate,
            shared_end_date,
        )
        return

    overpass_ds = load_overpass_grid(
        config,
        JacobianRunDirs,
        CSgridDir,
        StartDate,
    )

    (
        dims,
        coords,
        lon_name,
        lat_name,
        closest_hour,
        day_offset,
    ) = get_grid_info(
        overpass_ds,
        config["UseGCHP"],
    )

    Jacobian_RunDir_list = [
        name
        for name in os.listdir(JacobianRunDirs)
        if os.path.isdir(
            os.path.join(JacobianRunDirs, name)
        )
    ]

    # Run indices are read from the directory names rather than inferred from
    # how many directories there are. The two agree only when the runs are
    # numbered from zero: with DisableRun0000 the directories are
    # _0001.._000N, so a count used as an exclusive range bound stops at N-1
    # and silently skips the last run. Reading the names is also unaffected by
    # any unrelated directory sitting alongside the runs.
    run_indices = sorted(
        run_i
        for run_i in (
            int(name.rsplit("_", 1)[1])
            for name in Jacobian_RunDir_list
            if name.startswith(f"{RunName}_")
            and name.rsplit("_", 1)[1].isdigit()
        )
        if run_i >= start_run_num
    )

    if not run_indices:
        raise FileNotFoundError(
            f"No Jacobian run directories matching {RunName}_#### found in "
            f"{JacobianRunDirs}"
        )

    print(
        f"Processing {len(run_indices)} Jacobian run(s): "
        f"{run_indices[0]:04d}..{run_indices[-1]:04d}"
    )

    pert_simulations_dict = build_pert_simulations_dict(
        config,
        n_elements,
    )

    tmp_dir = tempfile.mkdtemp(
        prefix="overpass_met_"
    )

    # n_workers is a joblib convention: -1 means every core, -2 all but one.
    # Both the chunk arithmetic and the message below need a real count.
    #
    # "Every core" means every core of the allocation, not of the machine.
    # joblib.cpu_count() reports the latter, so on a shared node -1 would open
    # 112 workers against 48 allocated cores and spend the difference in
    # context switching.
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n_cores = (
        int(slurm_cpus)
        if slurm_cpus and slurm_cpus.isdigit()
        else joblib.cpu_count()
    )

    effective_workers = (
        n_workers if n_workers > 0 else max(1, n_cores + 1 + n_workers)
    )
    n_workers = effective_workers

    try:
        met_paths = presave_met_arrays(
            date_list,
            start,
            end,
            JacobianRunDirs,
            RunName,
            tmp_dir,
            DisableRun0000,
            n_workers=n_workers,
        )

        # Reuse what the prologue measured. process_run_day holds a date and
        # the one after it, plus the simulation output read against them, so
        # it is the heavier of the two steps -- if either is going to exhaust
        # the allocation it is this one.
        sample = next(
            (v["current"] for v in met_paths.values() if v.get("current")),
            None,
        )
        measured_mb = met_arrays_mb(sample)
        if measured_mb:
            factor = float(os.environ.get("OVERPASS_MEM_FACTOR", "6"))
            effective_workers = memory_capped_workers(
                effective_workers, int(measured_mb * factor)
            )
            n_workers = effective_workers

        # Tasks are (run, slice of dates), not (run) alone. A face with one
        # Jacobian run used to produce exactly one task, so every core beyond
        # the first sat idle no matter what n_jobs said.
        #
        # Chunks rather than one task per date: loky pickles every argument to
        # every task, and overpass_ds and pert_simulations_dict go to each one.
        # Aiming at a few times n_workers keeps all the cores fed while paying
        # that cost tens of times rather than thousands.
        # Higher balances the load better; lower pickles the shared arguments
        # fewer times. Four is a starting point, not a measured optimum.
        chunk_factor = int(os.environ.get("OVERPASS_CHUNK_FACTOR", "4"))
        n_chunks = max(
            1,
            -(-(chunk_factor * effective_workers) // max(1, len(run_indices))),
        )
        date_chunks = split_dates(date_list, n_chunks)

        print(
            f"Parallel over {len(run_indices)} run(s) x "
            f"{len(date_chunks)} date chunk(s) "
            f"= {len(run_indices) * len(date_chunks)} task(s) "
            f"on {effective_workers} worker(s)"
        )

        Parallel(
            n_jobs=n_workers,
            backend="loky",
            batch_size=1,
            pre_dispatch="2*n_jobs",
        )(
            delayed(process_run_dates)(
                run_i,
                date_chunk,
                met_paths,
                config,
                n_elements,
                JacobianRunDirs,
                pert_simulations_dict,
                overpass_ds,
                closest_hour,
                day_offset,
                dims,
                coords,
                lon_name,
                lat_name,
                DisableRun0000,
            )
            for run_i in run_indices
            for date_chunk in date_chunks
        )

        # Reached only when every run and date above completed. Missing
        # OutputDir inputs now raise, so a marker cannot cover a date that
        # was silently written as all-NaN.
        write_stage_marker(
            os.path.join(OutputPath, RunName),
            "overpass",
            StartDate,
            shared_end_date,
        )

    finally:
        shutil.rmtree(
            tmp_dir,
            ignore_errors=True,
        )


if __name__ == "__main__":
    config_path = sys.argv[1]

    with open(config_path) as config_file:
        config = yaml.load(
            config_file,
            Loader=yaml.FullLoader,
        )

    n_elements = int(sys.argv[2])

    n_workers = (
        int(sys.argv[3])
        if len(sys.argv) > 3
        else -1
    )

    calculate_satellite_overpass_diagnostics(
        config,
        n_elements,
        n_workers,
    )