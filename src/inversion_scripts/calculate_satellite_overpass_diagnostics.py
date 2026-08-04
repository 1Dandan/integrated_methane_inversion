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
    get_shared_end_date,
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

    fd, tmp_fpath = tempfile.mkstemp(
        prefix=f".{output_basename}.",
        suffix=".tmp.nc4",
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


# ---------------------------------------------------------------------------
# Helper: build date list
# ---------------------------------------------------------------------------
def build_date_list(config):
    """Return local overpass dates needed to cover the UTC simulation window.

    StartDate and EndDate are UTC dates, with EndDate exclusive.

    While simulations are progressing, exclude the latest local date because
    it may require UTC data from the following day. Once shared_end_date reaches
    EndDate, include all local dates through EndDate - 1.
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
    local_end_exclusive = datetime.strptime(
        shared_end_date,
        "%Y%m%d",
    )

    if local_end_exclusive != end:
        local_end_exclusive -= timedelta(days=1)

    n_process = (local_end_exclusive - local_start).days

    date_list = [
        (local_start + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(n_process)
    ]

    return start, end, date_list


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

        grid_file = os.path.join(
            Jacobian_RunDir,
            f"OutputDir/GEOSChem.SpeciesConc.{StartDate}_0000z.nc4",
        )

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
def presave_met_arrays(
    date_list,
    utc_start,
    utc_end,
    JacobianRunDirs,
    RunName,
    tmp_dir,
    DisableRun0000=False,
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

    saved_met = {}

    for date_str in sorted(required_dates):
        date_dt = datetime.strptime(date_str, "%Y%m%d")

        if not utc_start <= date_dt < utc_end:
            saved_met[date_str] = None
            continue

        met_file = os.path.join(
            JacobianRunDirs,
            f"{RunName}_{run_id:04d}",
            f"OutputDir/GEOSChem.{prefix}.{date_str}_0000z.nc4",
        )

        if not os.path.isfile(met_file):
            saved_met[date_str] = None
            continue

        AirDen, BxH = load_met_fields(met_file)

        airden_fpath = os.path.join(
            tmp_dir,
            f"AirDen_{date_str}.npy",
        )

        bxh_fpath = os.path.join(
            tmp_dir,
            f"BxH_{date_str}.npy",
        )

        np.save(airden_fpath, AirDen)
        np.save(bxh_fpath, BxH)

        del AirDen, BxH

        saved_met[date_str] = {
            "AirDen": airden_fpath,
            "BxH": bxh_fpath,
        }

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
):
    """Sample one base-run diagnostic file type at local overpass time."""
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

    with ExitStack() as stack:
        sim_utc_ds = (
            stack.enter_context(
                xr.open_dataset(sim_file_utc)
            )
            if (
                current_met is not None
                and os.path.isfile(sim_file_utc)
            )
            else None
        )

        sim_utc_ds_nextday = (
            stack.enter_context(
                xr.open_dataset(sim_file_utc_nextday)
            )
            if (
                next_met is not None
                and os.path.isfile(sim_file_utc_nextday)
            )
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

    start, end, date_list = build_date_list(config)
    StartDate = str(config["StartDate"])

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

    num_jacobian_runs = len(Jacobian_RunDir_list)

    pert_simulations_dict = build_pert_simulations_dict(
        config,
        n_elements,
    )

    tmp_dir = tempfile.mkdtemp(
        prefix="overpass_met_"
    )

    try:
        met_paths = presave_met_arrays(
            date_list,
            start,
            end,
            JacobianRunDirs,
            RunName,
            tmp_dir,
            DisableRun0000,
        )

        Parallel(
            n_jobs=n_workers,
            backend="loky",
            batch_size=1,
            pre_dispatch="2*n_jobs",
        )(
            delayed(process_run_dates)(
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
                DisableRun0000,
            )
            for run_i in range(
                start_run_num,
                num_jacobian_runs,
            )
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