# Sample all diagnostics at satellite overpass time
import sys
import os
import tempfile
import shutil
import numpy as np
import xarray as xr
import yaml
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from contextlib import ExitStack

from src.utilities.generate_overpass_grids import generate_on_sim_grid

from src.inversion_scripts.utils import (
    check_is_OH_element,
    check_is_BC_element,
    build_pert_simulations_dict,
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


def output_file_is_complete(fpath):
    """Return True only if `fpath` exists AND holds valid data.

    A file counts as complete when it opens as a NetCDF dataset and a data
    variable whose name contains 'SpeciesConcVV_CH4' or 'Met_' is (a) a
    floating-point array and (b) entirely finite (no NaN, no inf). Checking a
    single such variable is enough, since all variables in a file are written
    together. A missing, unreadable, non-float, or NaN-containing file is
    treated as incomplete so it gets rewritten.
    """
    if not os.path.isfile(fpath):
        return False

    try:
        with xr.open_dataset(fpath) as ds:
            check_var = next(
                (v for v in ds.data_vars
                 if "SpeciesConcVV_CH4" in v or "Met_" in v),
                None,
            )
            if check_var is None:
                return False

            da = ds[check_var]

            # Must be floating-point simulation data
            if not np.issubdtype(da.dtype, np.floating):
                return False

            # Every value must be finite (rejects NaN and inf)
            return bool(np.isfinite(da).all())
    except Exception:
        # Corrupt or partially written file -> reprocess
        return False
    
# ---------------------------------------------------------------------------
# Helper: build date list
# ---------------------------------------------------------------------------
def build_date_list(config):
    """Return local overpass dates needed to cover the UTC simulation window.

    StartDate and EndDate are UTC dates, with EndDate exclusive.

    We write files named by local overpass date. For a UTC window
    [StartDate, EndDate), the relevant local dates can begin one day before
    StartDate, because some local-date overpasses occur on the next UTC day.

    Example:
        UTC window: [20240501, 20240601)
        local-date files written:
            20240430, 20240501, ..., 20240531

    Boundary files may be partial:
        - first local date has NaNs where UTC data before StartDate would be needed
        - last local date has NaNs where UTC data at/after EndDate would be needed
    """
    StartDate = str(config["StartDate"])
    EndDate = str(config["EndDate"])

    start = datetime.strptime(StartDate, "%Y%m%d")
    end = datetime.strptime(EndDate, "%Y%m%d")  # exclusive UTC end

    local_start = start - timedelta(days=1)
    local_end_exclusive = end

    n_process = (local_end_exclusive - local_start).days

    date_list = [
        (local_start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(n_process)
    ]

    return start, end, date_list


# ---------------------------------------------------------------------------
# Helper: load or generate the overpass grid
# ---------------------------------------------------------------------------
def load_overpass_grid(config, JacobianRunDirs, CSgridDir, StartDate):
    """Load the overpass-time grid, generating it from a sample file if needed."""
    OverpassTime   = config["OverpassTime"]
    OrbitDirection = config["OrbitDirection"]
    OrbitsPerDay   = config["OrbitsPerDay"]
    RunName        = config["RunName"]

    os.makedirs(CSgridDir, exist_ok=True)
    overpass_grid_fpath = os.path.join(CSgridDir, "overpass_sample_utc_hour.nc")

    if not os.path.isfile(overpass_grid_fpath):
        # a sample simulation output from base run to get the grid information
        Jacobian_RunDir = os.path.join(JacobianRunDirs, f"{RunName}_0001")
        grid_file = os.path.join(
            Jacobian_RunDir,
            f"OutputDir/GEOSChem.SpeciesConc.{StartDate}_0000z.nc4",
        )
        overpass_ds = generate_on_sim_grid(
            grid_file, OverpassTime, overpass_grid_fpath,
            orbits_per_day=OrbitsPerDay, direction=OrbitDirection,
            isGCHP=config["UseGCHP"],
        )
    else:
        overpass_ds = xr.open_dataset(overpass_grid_fpath)

    return overpass_ds


# ---------------------------------------------------------------------------
# Helper: derive grid metadata from the overpass dataset
# ---------------------------------------------------------------------------
def get_grid_info(overpass_ds, use_gchp):
    """Return (dims, coords, lon_name, lat_name, closest_hour, day_offset).

    ``coords`` is a dict suitable for passing to ``xr.Dataset(...)``
    constructor — matching the pattern used in ``generate_overpass_grids.py``.
    """
    if use_gchp:
        lon_name = 'lons'
        lat_name = 'lats'
        dims = ('nf', 'Ydim', 'Xdim')
        coords = dict(
            lats=(['nf', 'Ydim', 'Xdim'], overpass_ds['lats'].values),
            lons=(['nf', 'Ydim', 'Xdim'], overpass_ds['lons'].values),
        )
    else:
        lon_name = 'lon'
        lat_name = 'lat'
        dims = ('lat', 'lon')
        coords = dict(
            lat=(['lat'], overpass_ds['lat'].values),
            lon=(['lon'], overpass_ds['lon'].values),
        )

    closest_hour = overpass_ds["closest_hour"].values
    day_offset   = overpass_ds["day_offset"].values

    if np.issubdtype(day_offset.dtype, np.timedelta64):
        day_offset = day_offset.astype("timedelta64[D]").astype(int)
    else:
        day_offset = day_offset.astype(int)

    # For TROPOMI with overpass time around 13:30 local time, day_offset is either 0 or 1.
    assert not np.any(day_offset == -1), (
        "day_offset=-1 cells present; previous day's file lookup not implemented"
    )

    return dims, coords, lon_name, lat_name, closest_hour, day_offset


# ---------------------------------------------------------------------------
# Helper: determine which tracer variables to read from a Jacobian run
# ---------------------------------------------------------------------------
def get_keepvars(sv_elems, n_elements, config, baserun=False):
    """Return the list of SpeciesConc variable names for a Jacobian run."""
    if sv_elems == [0]:
        return ['SpeciesConcVV_CH4']

    # Construct the list of CH4 vars to request
    # Local tracer indices are 1..len(sv_elems) within this run
    local_indices = range(1, len(sv_elems) + 1)
    keepvars = [f"SpeciesConcVV_CH4_jac{idx:04d}" for idx in local_indices]
    if baserun:
        keepvars.append("SpeciesConcVV_CH4")
    if len(keepvars) == 1:
        is_Regional = config["isRegional"]
        is_OH_element = check_is_OH_element(
            sv_elems[0], n_elements, config["OptimizeOH"], is_Regional
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
    """Return (AirDen, BxH) arrays from a SpeciesConc file.

    AirDen is converted from kg/m3 to g/m3.
    """
    with xr.open_dataset(met_file) as met_ds:
        # convert to g/m3 (time, lev, nf, Ydim, Xdim)
        AirDen = met_ds['Met_AIRDEN'].values * 1e3
        # unit is m (time, lev, nf, Ydim, Xdim)
        BxH = met_ds['Met_BXHEIGHT'].values
    return AirDen, BxH


# ---------------------------------------------------------------------------
# Helper: pre-save met arrays as .npy for memory-mapped access
# ---------------------------------------------------------------------------
def presave_met_arrays(date_list, utc_start, utc_end, JacobianRunDirs, RunName, tmp_dir, DisableRun0000=False):
    """Pre-save met arrays for available UTC simulation dates only.

    date_list contains local overpass dates. For each local date D, sampling may
    require UTC files for D and D+1 depending on day_offset.

    If either UTC date is outside [utc_start, utc_end), we leave the
    corresponding met path as None. Later, those grid cells remain NaN.
    """
    
    if DisableRun0000:
        run_id = 1
        prefix = 'BaseSpeciesConc'
    else:
        run_id = 0
        prefix = 'SpeciesConc'
    met_paths = {}

    def utc_date_is_available(date_dt):
        return utc_start <= date_dt < utc_end

    def save_met_for_date(date_str, suffix):
        date_dt = datetime.strptime(date_str, "%Y%m%d")

        if not utc_date_is_available(date_dt):
            return None

        met_file = os.path.join(
            JacobianRunDirs, f"{RunName}_{run_id:04d}",
            f"OutputDir/GEOSChem.{prefix}.{date_str}_0000z.nc4",
        )

        if not os.path.isfile(met_file):
            return None

        AirDen, BxH = load_met_fields(met_file)

        airden_fpath = os.path.join(tmp_dir, f"AirDen_{suffix}_{date_str}.npy")
        bxh_fpath = os.path.join(tmp_dir, f"BxH_{suffix}_{date_str}.npy")

        np.save(airden_fpath, AirDen)
        np.save(bxh_fpath, BxH)

        del AirDen, BxH

        return {
            "date_str": date_str,
            "AirDen": airden_fpath,
            "BxH": bxh_fpath,
        }

    for date_str in date_list:
        date_dt = datetime.strptime(date_str, "%Y%m%d")
        next_date_str = (date_dt + timedelta(days=1)).strftime("%Y%m%d")

        current_paths = save_met_for_date(date_str, "current")
        next_paths = save_met_for_date(next_date_str, "next")

        met_paths[date_str] = {
            "current": current_paths,
            "next": next_paths,
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
    """Sample CH4 columns at satellite overpass hour for all variables.

    Missing boundary UTC files are allowed. Any grid cells requiring missing
    UTC data remain NaN.
    """
    n_vars = len(keepvars)
    spatial_shape = closest_hour.shape
    overpass_CH4_col = np.full((n_vars, *spatial_shape), np.nan, dtype=np.float32)

    # --- cells whose overpass falls on the local date's UTC day ---
    if sim_utc_ds is not None and AirDen is not None and BxH is not None:
        valid0 = day_offset == 0

        for hr in np.unique(closest_hour[valid0]):
            mask = valid0 & (closest_hour == hr)
            hr_int = int(hr)

            col_weight = AirDen[hr_int, ...] / MwAir * BxH[hr_int, ...]

            for i, var in enumerate(keepvars):
                if var not in sim_utc_ds:
                    continue

                ch4 = sim_utc_ds[var].isel(time=hr_int).values
                overpass_CH4_col[i, mask] = (ch4 * col_weight).sum(axis=0)[mask]

    # --- cells whose overpass falls on the next UTC day ---
    if sim_utc_ds_nextday is not None and AirDen_nextday is not None and BxH_nextday is not None:
        valid1 = day_offset == 1

        for hr in np.unique(closest_hour[valid1]):
            mask = valid1 & (closest_hour == hr)
            hr_int = int(hr)

            col_weight = AirDen_nextday[hr_int, ...] / MwAir * BxH_nextday[hr_int, ...]

            for i, var in enumerate(keepvars):
                if var not in sim_utc_ds_nextday:
                    continue

                ch4 = sim_utc_ds_nextday[var].isel(time=hr_int).values
                overpass_CH4_col[i, mask] = (ch4 * col_weight).sum(axis=0)[mask]

    return overpass_CH4_col

def sample_overpass_3D(
    sim_utc_ds,
    sim_utc_ds_nextday,
    closest_hour,
    day_offset,
    spatial_dims,
):
    """Sample all time-dependent variables with vertical dimension at local overpass time.

    Assumptions
    -----------
    - time is always axis 0.
    - horizontal grid dimensions are the trailing dimensions.
    - all non-time dimensions should be preserved exactly.
    - no vertical integration is performed.

    Returns
    -------
    xr.Dataset
        Dataset sampled at local overpass time.
    """
    ref_ds = sim_utc_ds if sim_utc_ds is not None else sim_utc_ds_nextday
    if ref_ds is None:
        return xr.Dataset()

    out_ds = xr.Dataset()

    # Copy all non-time coordinates.
    for cname, coord in ref_ds.coords.items():
        if cname != "time":
            out_ds = out_ds.assign_coords({cname: coord})

    # Use union of variables from current and next-day files.
    all_vars = set()
    if sim_utc_ds is not None:
        all_vars.update(sim_utc_ds.data_vars)
    if sim_utc_ds_nextday is not None:
        all_vars.update(sim_utc_ds_nextday.data_vars)

    for var in sorted(all_vars):
        if sim_utc_ds is not None and var in sim_utc_ds:
            src_da = sim_utc_ds[var]
        elif sim_utc_ds_nextday is not None and var in sim_utc_ds_nextday:
            src_da = sim_utc_ds_nextday[var]
        else:
            continue

        # Variables without time are copied directly.
        # Examples: contacts, anchor, lons, lats, corner_lons, corner_lats.
        if "time" not in src_da.dims:
            out_ds[var] = src_da
            continue

        # Time must be axis 0.
        if src_da.dims[0] != "time":
            raise ValueError(
                f"{var} has time dimension, but time is not axis 0: {src_da.dims}"
            )

        # Preserve every original dimension except time.
        out_dims = tuple(src_da.dims[1:])
        out_shape = tuple(src_da.sizes[d] for d in out_dims)

        # Only sample variables whose trailing dimensions match the model grid.
        # For GCHP: spatial_dims = ("nf", "Ydim", "Xdim")
        # For GCC:  spatial_dims = ("lat", "lon")
        if tuple(out_dims[-len(spatial_dims):]) != tuple(spatial_dims):
            # Skip time-dependent variables that are not on the model grid.
            # You can change this behavior later if needed.
            continue

        sampled = np.full(out_shape, np.nan, dtype=np.float32)

        # Sample from current UTC file.
        if sim_utc_ds is not None and var in sim_utc_ds:
            valid0 = day_offset == 0

            for hr in np.unique(closest_hour[valid0]):
                hr_int = int(hr)
                mask = valid0 & (closest_hour == hr)

                arr = sim_utc_ds[var].isel(time=hr_int).values

                # Since horizontal grid dims are trailing dims:
                sampled[..., mask] = arr[..., mask]

        # Sample from next UTC file.
        if sim_utc_ds_nextday is not None and var in sim_utc_ds_nextday:
            valid1 = day_offset == 1

            for hr in np.unique(closest_hour[valid1]):
                hr_int = int(hr)
                mask = valid1 & (closest_hour == hr)

                arr = sim_utc_ds_nextday[var].isel(time=hr_int).values

                sampled[..., mask] = arr[..., mask]

        out_ds[var] = xr.DataArray(
            sampled,
            dims=out_dims,
            attrs=dict(src_da.attrs),
        )

        out_ds[var].attrs["description"] = (
            "Sampled at local satellite overpass time without vertical integration."
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
):
    """Sample one baserun file type at local overpass time."""

    file_current = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/{file_prefix}.{date_str}_0000z.nc4",
    )

    file_next = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/{file_prefix}.{next_date_str}_0000z.nc4",
    )

    with ExitStack() as stack:
        ds_current = (
            stack.enter_context(xr.open_dataset(file_current))
            if os.path.isfile(file_current)
            else None
        )

        ds_next = (
            stack.enter_context(xr.open_dataset(file_next))
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
    """Process one Jacobian run for one day: compute overpass columns and write output.

    Met arrays are memory-mapped from pre-saved .npy files. The OS shares the
    underlying physical pages between all workers that process the same day, so
    memory cost is ~1 copy per day regardless of how many workers read it.

    If DisableRun0000=True, Run 0000 is skipped. In that case, Run 0001 is used
    to sample BaseSpeciesConc and StateMetLevEdge, then the normal CH4 column
    diagnostic is also computed from SpeciesConc.
    """

    # ------------------------------------------------------------------
    # Skip Run 0000 entirely when it is disabled
    # ------------------------------------------------------------------
    if DisableRun0000 and run_i == 0:
        return

    # Suppress xarray warnings in worker processes
    warnings.filterwarnings(
        "ignore", category=UserWarning, module=r"xarray.*"
    )
    warnings.filterwarnings(
        "ignore", message="Duplicate dimension names present.*"
    )

    RunName = config["RunName"]
    OverpassTime = config["OverpassTime"]
    overpass_tag = OverpassTime.replace(":", "")

    run_num = str(run_i).zfill(4)
    sv_elems = pert_simulations_dict.get(run_num, [])

    # Only nonzero runs need keepvars for column calculation.
    # For DisableRun0000=True, run_i == 1 acts as the base run.
    if run_i != 0:
        baserun = run_i == 1
        keepvars = get_keepvars(sv_elems, n_elements, config, baserun)

    # ------------------------------------------------------------------
    # Memory-map met arrays
    # ------------------------------------------------------------------
    current_met = met_paths_day["current"]
    next_met = met_paths_day["next"]

    AirDen = (
        np.load(current_met["AirDen"], mmap_mode="r")
        if current_met is not None
        else None
    )
    BxH = (
        np.load(current_met["BxH"], mmap_mode="r")
        if current_met is not None
        else None
    )

    AirDen_nextday = (
        np.load(next_met["AirDen"], mmap_mode="r")
        if next_met is not None
        else None
    )
    BxH_nextday = (
        np.load(next_met["BxH"], mmap_mode="r")
        if next_met is not None
        else None
    )

    next_date_str = met_paths_day["next_date_str"]

    Jacobian_RunDir = os.path.join(
        JacobianRunDirs,
        f"{RunName}_{run_i:04d}",
    )
    output_dir = os.path.join(Jacobian_RunDir, "OverpassDiagnostics")
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Helper: attach overpass grid metadata without duplicating variables
    # ------------------------------------------------------------------
    def attach_overpass_grid_metadata(output_ds):
        """Attach lat/lon/corner metadata only if not already present."""

        if lat_name not in output_ds and lat_name in overpass_ds:
            output_ds[lat_name] = overpass_ds[lat_name]

        if lon_name not in output_ds and lon_name in overpass_ds:
            output_ds[lon_name] = overpass_ds[lon_name]

        if config.get("UseGCHP", False):
            if "corner_lons" not in output_ds and "corner_lons" in overpass_ds:
                output_ds["corner_lons"] = overpass_ds["corner_lons"]

            if "corner_lats" not in output_ds and "corner_lats" in overpass_ds:
                output_ds["corner_lats"] = overpass_ds["corner_lats"]

        return output_ds

    def add_common_attrs(output_ds):
        """Add shared metadata attributes."""
        output_ds.attrs["date_interpretation"] = (
            "Filename date is local overpass date. Values are sampled only where "
            "the corresponding UTC simulation date falls within [StartDate, EndDate). "
            "Boundary cells outside the UTC window are NaN."
        )
        output_ds.attrs["utc_start_date"] = str(config["StartDate"])
        output_ds.attrs["utc_end_date_exclusive"] = str(config["EndDate"])
        return output_ds

    # ------------------------------------------------------------------
    # Sample 3D base-run files
    #
    # Cases:
    #   1. Normal mode:
    #        run_i == 0 samples SpeciesConc + StateMetLevEdge, then returns.
    #
    #   2. DisableRun0000 mode:
    #        run_i == 1 samples BaseSpeciesConc + StateMetLevEdge,
    #        then continues to compute CH4 column from SpeciesConc.
    # ------------------------------------------------------------------
    do_sample_base_3d = (run_i == 0) or (DisableRun0000 and run_i == 1)

    if do_sample_base_3d:
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

        for file_prefix in baserun_file_types:
            output_fpath = os.path.join(
                output_dir,
                f"{file_prefix}.overpass.{date_str}_{overpass_tag}.nc4",
            )

            # Skip this file if it already exists and is complete
            if output_file_is_complete(output_fpath):
                continue

            output_ds = sample_baserun_file_type(
                Jacobian_RunDir,
                file_prefix,
                date_str,
                next_date_str,
                closest_hour,
                day_offset,
                dims,
            )

            output_ds = attach_overpass_grid_metadata(output_ds)
            output_ds = add_common_attrs(output_ds)

            output_ds.to_netcdf(output_fpath, mode="w")

        # In normal mode, Run 0000 only produces base 3D diagnostics.
        # In DisableRun0000 mode, Run 0001 should continue to CH4 columns.
        if run_i == 0:
            return

    # ------------------------------------------------------------------
    # Compute CH4 column diagnostics for nonzero Jacobian runs
    # ------------------------------------------------------------------
    sim_file_utc = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/GEOSChem.SpeciesConc.{date_str}_0000z.nc4",
    )

    sim_file_utc_nextday = os.path.join(
        Jacobian_RunDir,
        f"OutputDir/GEOSChem.SpeciesConc.{next_date_str}_0000z.nc4",
    )

    output_fpath = os.path.join(
        output_dir,
        f"GEOSChem.CH4col.overpass.{date_str}_{overpass_tag}.nc4",
    )

    # Skip this file if it already exists and is complete
    if output_file_is_complete(output_fpath):
        return

    with ExitStack() as stack:
        sim_utc_ds = (
            stack.enter_context(xr.open_dataset(sim_file_utc))
            if current_met is not None and os.path.isfile(sim_file_utc)
            else None
        )

        sim_utc_ds_nextday = (
            stack.enter_context(xr.open_dataset(sim_file_utc_nextday))
            if next_met is not None and os.path.isfile(sim_file_utc_nextday)
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

        output_ds = xr.Dataset(data_vars, coords=coords)

        output_ds = attach_overpass_grid_metadata(output_ds)
        output_ds = add_common_attrs(output_ds)

        output_ds.to_netcdf(output_fpath, mode="w")

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def calculate_satellite_overpass_diagnostics(config, n_elements, n_workers=-1):
    """Compute satellite overpass CH4 column diagnostics for all runs and days.

    All (day, run) pairs are dispatched in a single ``joblib.Parallel``
    call.  Met arrays are pre-saved as ``.npy`` files and memory-mapped
    by each worker, so the OS shares physical pages between workers
    processing the same day.

    Parameters
    ----------
    config : dict
        YAML configuration dictionary.
    n_elements : int
        Number of state-vector elements.
    n_workers : int, optional
        Number of parallel worker processes.
        -1 (default) uses all available cores.  1 disables parallelism,
        which is useful for debugging.
    """
    RunName    = config["RunName"]
    OutputPath = os.path.expandvars(config["OutputPath"])
    DisableRun0000 = config.get("DisableRun0000", False)
    if DisableRun0000:
        start_run_num = 1
    else:
        start_run_num = 0
    JacobianRunDirs = os.path.join(OutputPath, f"{RunName}/jacobian_runs/")
    CSgridDir       = os.path.join(OutputPath, f"{RunName}/CS_grids/")

    start, end, date_list = build_date_list(config)
    StartDate = str(config["StartDate"])

    # ---- overpass grid ----
    overpass_ds = load_overpass_grid(config, JacobianRunDirs, CSgridDir, StartDate)
    dims, coords, lon_name, lat_name, closest_hour, day_offset = get_grid_info(
        overpass_ds, config["UseGCHP"]
    )

    # ---- Jacobian run inventory ----
    Jacobian_RunDir_list = [
        name for name in os.listdir(JacobianRunDirs)
        if os.path.isdir(os.path.join(JacobianRunDirs, name))
    ]
    num_jacobian_runs = len(Jacobian_RunDir_list)

    # from 1 to n_elements, e.g. 1 to 1000, with leading zeros
    pert_simulations_dict = build_pert_simulations_dict(config, n_elements)

    # ---- pre-save met arrays as .npy for memory-mapped access ----
    # Each day's AirDen/BxH are saved once.  Workers memory-map them
    # read-only, so the OS shares physical pages between processes
    # that work on the same day — no per-worker copies.
    tmp_dir = tempfile.mkdtemp(prefix="overpass_met_")
    try:
        met_paths = presave_met_arrays(
            date_list, start, end, JacobianRunDirs, RunName, tmp_dir, DisableRun0000
        )

        # ---- dispatch all (day, run) pairs in one parallel call ----
        Parallel(n_jobs=n_workers, backend="loky")(
            delayed(process_run_day)(
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
            for date_str in date_list
            for run_i in range(start_run_num, num_jacobian_runs)
        )
    finally:
        # clean up temporary memmap files
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    n_elements = int(sys.argv[2])
    n_workers  = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    calculate_satellite_overpass_diagnostics(config, n_elements, n_workers)