#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SBATCH -N 1

import os
import sys
import yaml
import warnings
import datetime
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import colorcet as cc
import cartopy.crs as ccrs
from scipy.ndimage import binary_dilation
import gc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from src.inversion_scripts.point_sources import get_point_source_coordinates
from src.inversion_scripts.utils import (
    sum_total_emissions,
    plot_field,
    plot_field_gchp, # note we need to set vmin and vmax to make it proper for all cubic faces
    filter_tropomi,
    filter_blended,
    calculate_superobservation_error,
    get_mean_emissions,
    get_posterior_emissions,
)
from src.inversion_scripts.operators.TROPOMI_operator import (
    read_tropomi,
    read_blended,
)
from src.inversion_scripts.classify_TROPOMI_obs_to_CSgrids import (
    latlon_to_cartesian,
    build_kdtree,
    classify_obs_to_cs_grid,
)

from src.inversion_scripts.regrid_precomputed_jacobian import(
    sum_and_sort_along_statevector,
)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_sensitivity_cache_path(
    preview_dir,
    filename="sensitivities.nc",
    kf_index=None,
):
    """
    Return the path for a cached 1-D sensitivity array.

    If kf_index is supplied, append ``_period{kf_index}`` before the
    filename extension, e.g. ``native_sensitivities_period3.nc``.
    """
    if kf_index is not None:
        stem, ext = os.path.splitext(filename)
        filename = f"{stem}_period{int(kf_index)}{ext}"

    return os.path.join(preview_dir, filename)


def save_sensitivities(
    sensitivities,
    preview_dir,
    filename="sensitivities.nc",
    kf_index=None,
    config=None,
    state_vector_path=None,
):
    """
    Save a 1-D averaging-kernel sensitivity vector.

    The caller determines whether the values are native or clustered by
    choosing the filename. For example:

      - ``sensitivities.nc`` for the current clustered StateVector.nc
      - ``native_sensitivities.nc`` for NativeStateVector.nc

    Kalman-filter periods are handled automatically through kf_index.
    """
    sensitivities = np.asarray(sensitivities, dtype=np.float64)

    if sensitivities.ndim != 1:
        raise ValueError(
            f"Sensitivities must be 1-D, got shape {sensitivities.shape}"
        )

    output_path = get_sensitivity_cache_path(
        preview_dir,
        filename=filename,
        kf_index=kf_index,
    )

    ds = xr.Dataset(
        data_vars={
            "Sensitivity": (
                ["state_vector_element"],
                sensitivities,
                {
                    "long_name": "Estimated averaging-kernel sensitivity",
                    "units": "1",
                },
            )
        },
        coords={
            "state_vector_element": np.arange(
                1, sensitivities.size + 1, dtype=np.int32
            )
        },
    )

    if config is not None:
        if "StartDate" in config:
            ds.attrs["StartDate"] = str(config["StartDate"])
        if "EndDate" in config:
            ds.attrs["EndDate"] = str(config["EndDate"])
        if "nBufferClusters" in config:
            ds.attrs["nBufferClusters"] = int(config["nBufferClusters"])

    if state_vector_path is not None:
        ds.attrs["StateVectorFile"] = os.path.abspath(state_vector_path)

    if kf_index is not None:
        ds.attrs["kf_index"] = int(kf_index)

    os.makedirs(preview_dir, exist_ok=True)

    # Atomic replacement prevents an interrupted run from leaving a partial cache.
    tmp_path = output_path + ".tmp"
    ds.to_netcdf(
        tmp_path,
        encoding={"Sensitivity": {"zlib": True, "complevel": 1}},
    )
    os.replace(tmp_path, output_path)
    ds.close()

    print(f"Saved {sensitivities.size} sensitivities to {output_path}")
    return output_path


def load_sensitivities(
    preview_dir,
    filename="sensitivities.nc",
    expected_size=None,
    kf_index=None,
):
    """
    Load a cached 1-D averaging-kernel sensitivity vector.

    Returns None when no matching cache exists. If expected_size is supplied,
    the cache must contain exactly that many state-vector elements.
    """
    cache_path = get_sensitivity_cache_path(
        preview_dir,
        filename=filename,
        kf_index=kf_index,
    )

    if not os.path.exists(cache_path):
        return None

    with xr.open_dataset(cache_path) as ds:
        sensitivities = ds["Sensitivity"].values.copy()
        cached_kf_index = ds.attrs.get("kf_index", None)

    if sensitivities.ndim != 1:
        raise ValueError(
            f"Cached sensitivities must be 1-D, got shape "
            f"{sensitivities.shape}: {cache_path}"
        )

    if expected_size is not None and sensitivities.size != int(expected_size):
        raise ValueError(
            f"Cached sensitivity size ({sensitivities.size}) does not match "
            f"the expected state-vector size ({int(expected_size)}): "
            f"{cache_path}"
        )

    if kf_index is not None and cached_kf_index is not None:
        if int(cached_kf_index) != int(kf_index):
            raise ValueError(
                f"Cached kf_index ({cached_kf_index}) does not match "
                f"requested kf_index ({kf_index}): {cache_path}"
            )

    print(f"Loaded {sensitivities.size} sensitivities from {cache_path}")
    return sensitivities


def get_TROPOMI_data(
    file_path, BlendedTROPOMI, xlim, ylim, startdate_np64, enddate_np64, use_water_obs
):
    """
    Returns a dict with the lat, lon, xch4, and albedo_swir observations
    extracted from the given tropomi file. Filters are applied to remove
    unsuitable observations
    Args:
        file_path : string
            path to the tropomi file
        BlendedTROPOMI : bool
            if True, use blended TROPOMI+GOSAT data
        xlim: list
            longitudinal bounds for region of interest
        ylim: list
            latitudinal bounds for region of interest
        startdate_np64: datetime64
            start date for time period of interest
        enddate_np64: datetime64
            end date for time period of interest
        use_water_obs: bool
            if True, use observations over water
    Returns:
         tropomi_data: dict
            dictionary of the extracted values
    """
    # Load the TROPOMI data
    assert isinstance(BlendedTROPOMI, bool), "BlendedTROPOMI is not a bool"
    if BlendedTROPOMI:
        TROPOMI = read_blended(file_path)
    else:
        TROPOMI = read_tropomi(file_path)
    if TROPOMI == None:
        print(f"Skipping {file_path} due to error")
        return TROPOMI

    if BlendedTROPOMI:
        # Only going to consider data within lat/lon/time bounds and without problematic coastal pixels
        sat_ind = filter_blended(
            TROPOMI, xlim, ylim, startdate_np64, enddate_np64, use_water_obs
        )
    else:
        # Only going to consider data within lat/lon/time bounds, with QA > 0.5, and with safe surface albedo values
        sat_ind = filter_tropomi(
            TROPOMI, xlim, ylim, startdate_np64, enddate_np64, use_water_obs
        )

    # Extract all valid observations at once using NumPy advanced indexing.
    # This avoids a Python loop over every individual TROPOMI pixel.
    return {
        "lat": np.asarray(TROPOMI["latitude"][sat_ind]),
        "lon": np.asarray(TROPOMI["longitude"][sat_ind]),
        "xch4": np.asarray(TROPOMI["methane"][sat_ind]),
        "swir_albedo": np.asarray(TROPOMI["swir_albedo"][sat_ind]),
        "time": np.asarray(TROPOMI["time"][sat_ind]),
    }


def imi_preview(
    config_path, state_vector_path, preview_dir, tropomi_cache, kf_index=None
):
    """
    Function to perform preview
    Requires preview simulation to have been run already (to generate HEMCO diags)
    Requires TROPOMI data to have been downloaded already
    """

    # ----------------------------------
    # Setup
    # ----------------------------------

    # Read config file
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    for key in config.keys():
        if isinstance(config[key], str):
            config[key] = os.path.expandvars(config[key])

    # Open the state vector file and squeeze time dimension
    state_vector = xr.load_dataset(state_vector_path).squeeze()
    state_vector_labels = state_vector["StateVector"]

    # Identify the last element of the region of interest
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - config["nBufferClusters"]
    )

    if config['UseGCHP']:
        basedir = os.path.expandvars(
            os.path.join(config["OutputPath"], config["RunName"])
        )
        gridfpath = f'{basedir}/CS_grids/grids.c{config["CS_RES"]}.nc'
        gridds = xr.open_dataset(gridfpath)
        corner_lons = gridds['corner_lons']
        corner_lats = gridds['corner_lats']
        
    # Set latitude/longitude bounds for plots
    if not config['UseGCHP']:
        # Trim 1-2.5 degrees to remove GEOS-Chem buffer zone
        if config["Res"] == "0.25x0.3125":
            degx = 4 * 0.3125
            degy = 4 * 0.25
        elif config["Res"] == "0.5x0.625":
            degx = 4 * 0.625
            degy = 4 * 0.5
        elif config["Res"] == "2.0x2.5":
            degx = 4 * 2.5
            degy = 4 * 2.0

        lon_bounds = [
            np.min(state_vector.lon.values) + degx,
            np.max(state_vector.lon.values) - degx,
        ]
        lat_bounds = [
            np.min(state_vector.lat.values) + degy,
            np.max(state_vector.lat.values) - degy,
        ]
    elif config['STRETCH_GRID']:
        buffer_bounds = 0.
        temp_lons = gridds['corner_lons'].values[5,...].copy()
        temp_lons[temp_lons>180] -= 360
        lon_min = max(temp_lons.min() - buffer_bounds, -180)
        lon_max = min(temp_lons.max() + buffer_bounds, 180)
        lat_min = max(gridds['corner_lats'].values[5,...].min(), -90)
        lat_max = min(gridds['corner_lats'].values[5,...].max(), 90)
        lon_bounds = [lon_min, lon_max]
        lat_bounds = [lat_min, lat_max]
    else:
        lon_bounds = [-180, 180]
        lat_bounds = [-90, 90]
    
    # # Define mask for ROI, to be used below
    a, df, num_days, prior, outstrings = estimate_averaging_kernel(
        config,
        state_vector_path,
        preview_dir,
        tropomi_cache,
        preview=True,
        kf_index=kf_index,
    )

    # Cache sensitivities for the current (possibly clustered) StateVector.
    # Native sensitivities are cached separately by aggregation.py.
    save_sensitivities(
        a,
        preview_dir,
        filename="sensitivities.nc",
        kf_index=kf_index,
        config=config,
        state_vector_path=state_vector_path,
    )

    mask = state_vector_labels <= last_ROI_element

    # ----------------------------------
    # Estimate dollar cost
    # ----------------------------------

    # Estimate cost by scaling reference cost of $20 for one-month Permian inversion
    # Reference number of state variables = 243
    # Reference number of days = 31
    # Reference cost for EC2 storage = $50 per month
    # Reference area = area of 24-39 N 95-111W
    # Note: calculate_area_in_km in src.inversion_scripts.utils cannot get the surface area correctly when it is nearly global coverage
    #       Thus, here we turn to get the ratio of the number of grid boxes relative to reference grid
    reference_cost = 20
    reference_num_compute_hours = 10
    ref_nbox = ((39 - 24) / 0.25) * ((-95 + 111) / 0.3125)

    hours_in_month = 31 * 24
    reference_storage_cost = 50 * reference_num_compute_hours / hours_in_month
    num_state_variables = np.nanmax(state_vector_labels.values)

    if config['UseGCHP']:
        nbox = 6 * config['CS_RES'] ** 2
    else:
        if config["Res"] == "0.125x0.15625":
            deltalat = 0.125
            deltalon = 0.15625
        if config["Res"] == "0.25x0.3125":
            deltalat = 0.25
            deltalon = 0.3125
        elif config["Res"] == "0.5x0.625":
            deltalat = 0.5
            deltalon = 0.625
        elif config["Res"] == "2.0x2.5":
            deltalat = 2.0
            deltalon = 2.5
        elif config["Res"] == "4.0x5.0":
            deltalat = 4.0
            deltalon = 5.0
        lats = [float(state_vector.lat.min()), float(state_vector.lat.max())]
        lons = [float(state_vector.lon.min()), float(state_vector.lon.max())]
        nbox = (lats[1] - lats[0]) / deltalat * (lons[1] - lons[0]) / deltalon
    nbox_factor = nbox / ref_nbox
    additional_storage_cost = ((num_days / 31) - 1) * reference_storage_cost
    expected_cost = (
        (reference_cost + additional_storage_cost)
        * (num_state_variables / 243)
        * nbox_factor
        * (num_days / 31)
    )

    outstring6 = (
        f"approximate cost = ${np.round(expected_cost,2)} for on-demand instance"
    )
    outstring7 = f"                 = ${np.round(expected_cost/3,2)} for spot instance"
    print(outstring6)
    print(outstring7)

    # ----------------------------------
    # Output
    # ----------------------------------

    # Write preview diagnostics to text file
    outputtextfile = open(os.path.join(preview_dir, "preview_diagnostics.txt"), "w+")
    outputtextfile.write("##" + outstring6 + "\n")
    outputtextfile.write("##" + outstring7 + "\n")
    outputtextfile.write(outstrings)
    outputtextfile.close()

    # Prepare plot data for prior
    prior_kgkm2h = prior * (1000**2) * 60 * 60  # Units kg/km2/h

    # Prepare plot data for observations
    df_means = df.copy(deep=True)
    df_means["lat"] = np.round(df_means["lat"], 1)  # Bin to 0.1x0.1 degrees
    df_means["lon"] = np.round(df_means["lon"], 1)
    df_means = df_means.groupby(["lat", "lon"]).mean()
    ds = df_means.to_xarray()

    # Prepare plot data for observation counts
    df_counts = df.copy(deep=True).drop(["xch4", "swir_albedo"], axis=1)
    df_counts["counts"] = 1
    df_counts["lat"] = np.round(df_counts["lat"], 1)  # Bin to 0.1x0.1 degrees
    df_counts["lon"] = np.round(df_counts["lon"], 1)
    df_counts = df_counts.groupby(["lat", "lon"]).sum()
    ds_counts = df_counts.to_xarray()

    plt.rcParams.update({"font.size": 18})

    # Plot prior emissions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    if config['UseGCHP']:
        plot_field_gchp(
            ax,
            corner_lons,
            corner_lats,
            prior_kgkm2h,
            cmap=cc.cm.linear_kryw_5_100_c67_r,
            plot_type="pcolormesh",
            vmin=0,
            vmax=14,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            levels=21,
            title="Prior emissions",
            point_sources=get_point_source_coordinates(config),
            cbar_label="Emissions (kg km$^{-2}$ h$^{-1}$)",
            only_ROI=False,
            is_regional=config["isRegional"],
        )
    else:
        plot_field(
            ax,
            prior_kgkm2h,
            cmap=cc.cm.linear_kryw_5_100_c67_r,
            plot_type="pcolormesh",
            vmin=0,
            vmax=14,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            levels=21,
            title="Prior emissions",
            point_sources=get_point_source_coordinates(config),
            cbar_label="Emissions (kg km$^{-2}$ h$^{-1}$)",
            mask=mask if config["isRegional"] else None,
            only_ROI=False,
            is_regional=config["isRegional"],
        )
    plt.savefig(
        os.path.join(preview_dir, "preview_prior_emissions.png"),
        bbox_inches="tight",
        dpi=150,
    )

    # simple function to find the dynamic range for colorbar
    dynamic_range = lambda vals: (
        np.round(np.nanmedian(vals) / 25.0) * 25 - 25,
        np.round(np.nanmedian(vals) / 25.0) * 25 + 25,
    )
    # Plot observations
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    xch4_min, xch4_max = dynamic_range(ds["xch4"].values)
    plot_field(
        ax,
        ds["xch4"],
        cmap="Spectral_r",
        plot_type="pcolormesh",
        vmin=xch4_min,
        vmax=xch4_max,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="TROPOMI $X_{CH4}$",
        cbar_label="Column mixing ratio (ppb)",
        mask=mask if config["isRegional"] else None,
        only_ROI=False,
        is_regional=config["isRegional"],
    )

    plt.savefig(
        os.path.join(preview_dir, "preview_observations.png"),
        bbox_inches="tight",
        dpi=150,
    )



    # Plot albedo
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        ds["swir_albedo"],
        cmap="magma",
        plot_type="pcolormesh",
        vmin=0,
        vmax=0.4,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="SWIR Albedo",
        cbar_label="Albedo",
        mask=mask if config["isRegional"] else None,
        only_ROI=False,
        is_regional=config["isRegional"],
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_albedo.png"), bbox_inches="tight", dpi=150
    )

    # Plot observation density
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    plot_field(
        ax,
        ds_counts["counts"],
        cmap="Blues",
        plot_type="pcolormesh",
        vmin=0,
        vmax=np.nanmax(ds_counts["counts"].values),
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title="Observation density",
        cbar_label="Number of observations",
        mask=mask if config["isRegional"] else None,
        only_ROI=False,
        is_regional=config["isRegional"],
    )
    plt.savefig(
        os.path.join(preview_dir, "preview_observation_density.png"),
        bbox_inches="tight",
        dpi=150,
    )

    # plot state vector
    num_colors = state_vector_labels.where(mask).max().item()
    sv_cmap = matplotlib.colors.ListedColormap(np.random.rand(int(num_colors), 3))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    if config['UseGCHP']:
        plot_field_gchp(
            ax,
            corner_lons,
            corner_lats,
            state_vector_labels,
            cmap=sv_cmap,
            vmin=1,
            vmax=num_colors,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            title="State Vector Elements",
            cbar_label="Element ID",
            only_ROI=True,
            state_vector_labels=state_vector_labels,
            last_ROI_element=last_ROI_element,
            is_regional=config["isRegional"],
        )
    else:
        plot_field(
            ax,
            state_vector_labels,
            cmap=sv_cmap,
            vmin=1,
            vmax=num_colors,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            title="State Vector Elements",
            cbar_label="Element ID",
            only_ROI=True,
            state_vector_labels=state_vector_labels,
            last_ROI_element=last_ROI_element,
            is_regional=config["isRegional"],
        )
    plt.savefig(
        os.path.join(preview_dir, "preview_state_vector.png"),
        bbox_inches="tight",
        dpi=150,
    )

    # plot estimated averaging kernel sensitivities
    sensitivities = map_sensitivities_to_sv(a, state_vector_labels, last_ROI_element)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    if config['UseGCHP']:
        plot_field_gchp(
            ax,
            corner_lons,
            corner_lats,
            sensitivities,
            cmap=cc.cm.CET_L19,
            vmin=0,
            vmax=np.nanpercentile(sensitivities.values, 95),
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            title="Estimated Averaging kernel sensitivities",
            cbar_label="Sensitivity",
            only_ROI=True,
            state_vector_labels=state_vector_labels,
            last_ROI_element=last_ROI_element,
            is_regional=config["isRegional"],
        )
    else:
        plot_field(
            ax,
            sensitivities,
            cmap=cc.cm.CET_L19,
            vmin=0,
            vmax=np.nanpercentile(sensitivities.values, 95),
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds,
            title="Estimated Averaging kernel sensitivities",
            cbar_label="Sensitivity",
            only_ROI=True,
            state_vector_labels=state_vector_labels,
            last_ROI_element=last_ROI_element,
            is_regional=config["isRegional"],
        )
    plt.savefig(
        os.path.join(preview_dir, "preview_estimated_sensitivities.png"),
        bbox_inches="tight",
        dpi=150,
    )



    # calculate expected DOFS
    expectedDOFS = np.round(sum(a), 5)
    if expectedDOFS < config["DOFSThreshold"]:
        print(
            f"\nExpected DOFS = {expectedDOFS} are less than DOFSThreshold = {config['DOFSThreshold']}. Exiting.\n"
        )
        print(
            "Consider increasing the inversion period, increasing the prior error, or using another prior inventory.\n"
        )
        # if run with sbatch this ensures the exit code is not lost.
        file = open(".error_status_file.txt", "w")
        file.write("Error Status: 1")
        file.close()
        sys.exit(1)


def map_sensitivities_to_sv(sensitivities, state_vector_lables, last_ROI_element):
    """
    Map 1D sensitivities onto a label grid.

    Parameters
    ----------
    sensitivities : array-like, shape (last_ROI_element,)
        Sensitivity value corresponding to ROI label 1..last_ROI_element.
    state_vector_lables : xr.DataArray
        The StateVector array containing integer labels and NaNs.
        Can have shape (lat, lon), (nf, Ydim, Xdim), (time, nf, Ydim, Xdim), etc.
    last_ROI_element : int

    Returns
    -------
    xr.DataArray with the same dims/coords as labels_da
    """
    labels = state_vector_lables.values
    sens = np.asarray(sensitivities)

    # Valid ROI labels: 1..last_ROI_element
    valid = np.isfinite(labels) & (labels <= last_ROI_element)

    # Output array
    out = np.full(labels.shape, np.nan, dtype=sens.dtype)

    # Convert labels → 0-based indices
    idx = labels[valid].astype(int) - 1

    # Fill output
    out[valid] = sens[idx]

    # Wrap back into a DataArray
    return xr.DataArray(
        out,
        coords=state_vector_lables.coords,
        dims=state_vector_lables.dims,
        name="Sensitivities"
    )

def get_sectoral_outputs(prior_ds, areas, mask, preview_dir):
    """
    Get sectoral emissions from the prior dataset
    """
    # Plot sectoral emissions
    sectors = [
        var
        for var in list(prior_ds.keys())
        if "EmisCH4" in var and not ("Total" in var or "Excl" in var)
    ]

    # Calculate total emissions for each sector
    prior_sector_vals = []
    positive_sectors = []
    for sector in sectors:
        prior_val = sum_total_emissions(prior_ds[sector], areas, mask)
        if prior_val > 0:
            prior_sector_vals.append(prior_val)
            positive_sectors.append(sector.replace("EmisCH4_", ""))

    # Combine the lists into tuples and sort them based on prior_sector_vals
    combined = list(zip(positive_sectors, prior_sector_vals))
    combined_sorted = sorted(combined, key=lambda x: x[1])
    positive_sectors, prior_sector_vals = zip(*combined_sorted)

    # Plot bars for prior emissions
    fig = plt.figure(figsize=(10, 5))
    ax = fig.subplots(1, 1)
    bar_height = 0.35
    ind = np.arange(len(positive_sectors))
    bars1 = ax.barh(
        ind,
        prior_sector_vals,
        bar_height,
        color="goldenrod",
        label="Prior Emissions",
    )

    # Add labels and title
    ax.set_xlabel(r"Emissions ($Tg\ a^{-1}$)")
    ax.set_ylabel("Sector")
    ax.set_title("Sectoral Emissions (Prior Inventory)")
    ax.set_yticks(ind)
    ax.set_yticklabels(positive_sectors)

    plt.savefig(f"{preview_dir}/prior_sectoral_emissions.png", bbox_inches="tight")

    sector_totals = {}

    for item in combined_sorted:
        category = item[0]
        sector_prior = item[1]
        sector_totals[f"{category}Prior"] = sector_prior

    # Save the statistics to a file
    stats_pd = pd.DataFrame(sector_totals, index=[0])
    stats_pd.to_csv(f"{preview_dir}/prior_sectoral_statistics.csv", index=False)

    return

def estimate_averaging_kernel(
    config, state_vector_path, preview_dir, tropomi_cache, preview=False, kf_index=None
):
    """
    Estimates the averaging kernel sensitivities using prior emissions
    and the number of observations available in each grid cell
    """

    # ----------------------------------
    # Setup
    # ----------------------------------

    # Open the state vector file and squeeze time dimension
    state_vector = xr.load_dataset(state_vector_path).squeeze()
    state_vector_labels = state_vector["StateVector"]

    # Identify the last element of the region of interest
    last_ROI_element = int(
        np.nanmax(state_vector_labels.values) - config["nBufferClusters"]
    )

    # Whether to use observations over water?
    use_water_obs = config["UseWaterObs"] if "UseWaterObs" in config.keys() else False

    # Define mask for ROI, to be used below
    mask = state_vector_labels <= last_ROI_element

    # ----------------------------------
    # Total prior emissions
    # ----------------------------------
    # Start and end dates of the inversion
    startday = str(config["StartDate"])
    endday = str(config["EndDate"])

    # Prior emissions
    prior_cache = os.path.expandvars(
        os.path.join(config["OutputPath"], config["RunName"], "hemco_prior_emis/OutputDir")
    )

    # adjustments for when performing for dynamic kf clustering
    if kf_index is not None:
        # use different date range for KF inversion if kf_index is not None
        rundir_path = preview_dir.split("preview")[0]
        periods = pd.read_csv(f"{rundir_path}periods.csv")
        startday = str(periods.iloc[kf_index - 1]["Starts"])
        endday = str(periods.iloc[kf_index - 1]["Ends"])

        # use the nudged (prior) emissions for generating averaging kernel estimate
        sf = xr.load_dataset(f"{rundir_path}archive_sf/prior_sf_period{kf_index}.nc")
        prior_ds = get_mean_emissions(startday, endday, prior_cache)
        prior_ds = get_posterior_emissions(prior_ds, sf, config["OptimizeSoil"])
    else:
        prior_ds = get_mean_emissions(startday, endday, prior_cache)

    prior = prior_ds["EmisCH4_Total"]

    # Compute total emissions in the region of interest
    if config['UseGCHP']:
        basedir = os.path.expandvars(
            os.path.join(config["OutputPath"], config["RunName"])
        )
        gridfpath = f'{basedir}/CS_grids/grids.c{config["CS_RES"]}.nc'
        gridds = xr.open_dataset(gridfpath)
        areas = gridds['area']
    else:
        areas = prior_ds["AREA"]
    total_prior_emissions = sum_total_emissions(prior, areas, mask)
    outstring1 = (
        f"Total prior emissions in region of interest = {total_prior_emissions} Tg/y \n"
    )
    print(outstring1)


    # calculate sectoral totals if running preview
    if preview:
        get_sectoral_outputs(prior_ds, areas, mask, preview_dir)

    # ----------------------------------
    # Observations in region of interest
    # ----------------------------------

    # Paths to tropomi data files
    tropomi_files = [f for f in os.listdir(tropomi_cache) if ".nc" in f]
    tropomi_paths = [os.path.join(tropomi_cache, f) for f in tropomi_files]

    if config['UseGCHP']:
        xlim = [-180, 180]
        ylim = [-90, 90]
    else:
        # Latitude/longitude bounds of the inversion domain
        sv_indomain = state_vector_labels.where(mask).stack(point=("lat", "lon")).dropna("point")
        xlim = [float(sv_indomain.lon.min()), float(sv_indomain.lon.max())]
        ylim = [float(sv_indomain.lat.min()), float(sv_indomain.lat.max())]
        
        if config["Res"] == "4.0x5.0":
            deg_lat, deg_lon = 4.0, 5.0
        elif config["Res"] == "2.0x2.5":
            deg_lat, deg_lon = 2.0, 2.5
        elif config["Res"] == "0.5x0.625":
            deg_lat, deg_lon = 0.5, 0.625
        elif config["Res"] == "0.25x0.3125":
            deg_lat, deg_lon = 0.25, 0.3125
        xlim = [xlim[0] - deg_lon/2, xlim[1] + deg_lon/2]
        ylim = [ylim[0] - deg_lat/2, ylim[1] + deg_lat/2]

    start = f"{startday[0:4]}-{startday[4:6]}-{startday[6:8]} 00:00:00"
    end = f"{endday[0:4]}-{endday[4:6]}-{endday[6:8]} 23:59:59"
    startdate_np64 = np.datetime64(
        datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    )
    enddate_np64 = np.datetime64(
        datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=1)
    )

    # Only consider tropomi files within date range (in case more are present)
    tropomi_paths = [
        p
        for p in tropomi_paths
        if int(p.split("____")[1][0:8]) >= int(startday)
        and int(p.split("____")[1][0:8]) < int(endday)
    ]
    tropomi_paths.sort()

    # Use blended TROPOMI+GOSAT data or operational TROPOMI data?
    BlendedTROPOMI = config["BlendedTROPOMI"]

    # Read in and filter tropomi observations (uses parallel processing).
    # Keep n_jobs=-1 so joblib uses the CPUs available to this Slurm job.
    observation_dicts = Parallel(n_jobs=-1)(
        delayed(get_TROPOMI_data)(
            file_path,
            BlendedTROPOMI,
            xlim,
            ylim,
            startdate_np64,
            enddate_np64,
            use_water_obs,
        )
        for file_path in tropomi_paths
    )
    # Remove any problematic observation dicts (eg. corrupted data file).
    observation_dicts = list(filter(None, observation_dicts))

    if observation_dicts:
        lat = np.concatenate([obs_dict["lat"] for obs_dict in observation_dicts])
        lon = np.concatenate([obs_dict["lon"] for obs_dict in observation_dicts])
        xch4 = np.concatenate([obs_dict["xch4"] for obs_dict in observation_dicts])
        albedo = np.concatenate(
            [obs_dict["swir_albedo"] for obs_dict in observation_dicts]
        )
        trtime = np.concatenate([obs_dict["time"] for obs_dict in observation_dicts])
    else:
        lat = np.array([], dtype=float)
        lon = np.array([], dtype=float)
        xch4 = np.array([], dtype=float)
        albedo = np.array([], dtype=float)
        trtime = np.array([], dtype="datetime64[ns]")

    # Assemble in dataframe.
    df = pd.DataFrame(
        {
            "lat": lat,
            "lon": lon,
            "obs_count": np.ones(len(lat), dtype=np.uint8),
            "swir_albedo": albedo,
            "xch4": xch4,
            "time": pd.to_datetime(trtime),
        }
    )

    # Set resolution-specific observation-count variables.
    # num_obs[i] and num_superobs[i] correspond to StateVector label i+1.
    if config['UseGCHP']:
        # Classify observations directly to cubed-sphere native grid cells.
        # No dense (date, nf, Ydim, Xdim) array is constructed.
        kdtree_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        print(
            f"Classifying {len(df)} observations to the cubed-sphere grid "
            f"with {kdtree_workers} KDTree worker(s)"
        )
        df_super = classify_obs_to_cs_grid(
            df,
            gridfpath,
            workers=kdtree_workers,
        )

        # build_kdtree() flattens the GCHP grid in C order, so sim_index maps
        # directly to the same flattened (nf, Ydim, Xdim) StateVector grid.
        sv_flat = np.asarray(state_vector_labels.values).reshape(-1)
        sim_index = df_super["sim_index"].to_numpy(dtype=np.int64)
        obs_sv_labels = sv_flat[sim_index]

        # Keep observations assigned to ROI state-vector elements only.
        valid_obs = (
            np.isfinite(obs_sv_labels)
            & (obs_sv_labels >= 1)
            & (obs_sv_labels <= last_ROI_element)
        )

        sv_idx = obs_sv_labels[valid_obs].astype(np.int64) - 1

        # Raw observation count per ROI state-vector element. Each row in df
        # currently represents one successful observation (obs_count == 1).
        num_obs = np.bincount(
            sv_idx,
            minlength=last_ROI_element,
        ).astype(np.int64)

        # One superobservation is one occupied native simulation grid cell on
        # one day. Deduplicate (date, sim_index), NOT (date, StateVector label),
        # because one SV element can contain many native grid cells.
        valid_pairs = df_super.loc[
            valid_obs,
            ["date", "sim_index"],
        ]
        first_in_superob = ~valid_pairs.duplicated().to_numpy()

        num_superobs = np.bincount(
            sv_idx[first_in_superob],
            minlength=last_ROI_element,
        ).astype(np.int64)

    else:
        # Set resolution specific variables
        # L_native = Rough length scale of native state vector element [m]
        if config["Res"] == "0.125x0.15625":
            lat_step = 0.125
            lon_step = 0.15625
        elif config["Res"] == "0.25x0.3125":
            lat_step = 0.25
            lon_step = 0.3125
        elif config["Res"] == "0.5x0.625":
            lat_step = 0.5
            lon_step = 0.625
        elif config["Res"] == "2.0x2.5":
            lat_step = 2.0
            lon_step = 2.5
        elif config["Res"] == "4.0x5.0":
            lat_step = 4.0
            lon_step = 5.0

        # Bin observations into native grid cells.
        to_lon = lambda x: np.floor(x / lon_step) * lon_step
        to_lat = lambda x: np.floor(x / lat_step) * lat_step

        df_super = df.rename(columns={"lon": "old_lon", "lat": "old_lat"})

        df_super["lat"] = to_lat(df_super.old_lat)
        df_super["lon"] = to_lon(df_super.old_lon)

        # Extract relevant fields and group by lat, lon, date.
        df_super = df_super[["lat", "lon", "time", "obs_count"]].copy()
        df_super["date"] = df_super["time"].dt.floor("D")
        grouped = (
            df_super.groupby(["lat", "lon", "date"])
            .size()
            .reset_index(name="obs_count")
        )

        # Convert the grouped DataFrame to a daily native-grid Dataset.
        daily_observation_counts = grouped.set_index(
            ["lat", "lon", "date"]
        ).to_xarray()
        daily_observation_counts = daily_observation_counts.reindex(
            lat=state_vector["lat"],
            lon=state_vector["lon"],
        ).transpose("date", "lat", "lon")

        daily_observation_counts["superobs_count"] = (
            daily_observation_counts["obs_count"]
        )
        daily_observation_counts["superobs_count"].values = np.where(
            np.isnan(np.asarray(daily_observation_counts["obs_count"].values)),
            0,
            1,
        )
        daily_observation_counts["obs_count"] = (
            daily_observation_counts["obs_count"].fillna(0)
        )

        # Collapse the date dimension first, then aggregate native-grid counts
        # by StateVector label.
        obs_count_grid = daily_observation_counts["obs_count"].sum(
            dim="date"
        ).values
        superobs_count_grid = daily_observation_counts["superobs_count"].sum(
            dim="date"
        ).values

        num_obs = sum_and_sort_along_statevector(
            val=obs_count_grid,
            sv=state_vector_labels.values,
        )[:last_ROI_element].astype(np.int64)

        num_superobs = sum_and_sort_along_statevector(
            val=superobs_count_grid,
            sv=state_vector_labels.values,
        )[:last_ROI_element].astype(np.int64)

    # Domain totals are retained for diagnostics and for the original AK
    # approximation below.
    tot_num_obs = int(num_obs.sum())
    tot_num_superobs = int(num_superobs.sum())

    flux_per_sv, L, num_sv_elements = compute_sv_element_stats(
        state_vector_labels=state_vector_labels,
        areas=areas,
        prior=prior,
        last_ROI_element=last_ROI_element,
        sum_and_sort_along_statevector=sum_and_sort_along_statevector,
    )
    if tot_num_obs < 1:
        sys.exit("Error: No observations found in region of interest")
    outstring2 = f"Found {tot_num_obs} observations ({tot_num_superobs} super observations) \
in the region of interest"
    print("\n" + outstring2)

    # ----------------------------------
    # Estimate information content
    # ----------------------------------

    time_delta = enddate_np64 - startdate_np64
    num_days = np.round((time_delta) / np.timedelta64(1, "D"))

    # If Kalman filter mode, count observations per inversion period
    if config["KalmanMode"]:
        startday_dt = datetime.datetime.strptime(startday, "%Y%m%d")
        endday_dt = datetime.datetime.strptime(endday, "%Y%m%d")
        if not config["MakePeriodsCSV"]:
            rundir_path = preview_dir.split("preview")[0]
            periods = pd.read_csv(f"{rundir_path}periods.csv")
            n_periods = periods.iloc[-1]["period_number"]
        else:
            n_periods = np.floor(
                (endday_dt - startday_dt).days / config["UpdateFreqDays"]
            )
        # average number of successful observation days in each inversion period
        outstring2 = f"Found {int(np.round(tot_num_obs / n_periods))} observations \
({int(np.round(tot_num_superobs / n_periods))} super observations) \
in the region of interest per inversion period, for {int(n_periods)} period(s)"
        print("\n" + outstring2)

    # Other parameters
    U = 5 * (1000 / 3600)  # 5 km/h uniform wind speed in m/s
    p = 101325  # Surface pressure [Pa = kg/m/s2]
    g = 9.8  # Gravity [m/s2]
    Mair = 0.029  # Molar mass of air [kg/mol]
    Mch4 = 0.01604  # Molar mass of methane [kg/mol]
    alpha = 0.4  # Simple parameterization of turbulence

    # Use the first element of the error list if multiple values are provided
    sigmaA = config["PriorError"][0] if isinstance(config["PriorError"], list) else config["PriorError"]
    # Error standard deviations with updated units
    sA = sigmaA * flux_per_sv
    sO = config["ObsError"][0] if isinstance(config["ObsError"], list) else config["ObsError"]

    # Calculate superobservation error per state-vector element using the
    # original IMI logic. P is the average number of raw observations
    # contributing to one native-grid superobservation within each SV element.
    num_obs = np.asarray(num_obs, dtype=float)
    num_superobs = np.asarray(num_superobs, dtype=float)

    P = np.divide(
        num_obs,
        num_superobs,
        out=np.zeros_like(num_obs),
        where=num_superobs > 0,
    )

    # Original behavior for cells with no observations (or any P < 1): use
    # the superobservation error corresponding to P=1, then explicitly set
    # their AK sensitivity to zero below.
    P_safe = np.where(P >= 1.0, P, 1.0)
    s_superO = calculate_superobservation_error(sO, P_safe) * 1e-9

    # Following eqn #6 from Estrada et al. 2025, but without accounting for
    # observations in the two concentric rings around each native grid cell.
    # num_superobs is therefore the number of superobservations associated
    # directly with each state-vector element.
    k = alpha * (Mair * L * g / (Mch4 * U * p))

    with np.errstate(divide="ignore", invalid="ignore"):
        a = sA**2 / (
            sA**2
            + (s_superO / k) ** 2
            / num_superobs
        )

    # Places with zero superobservations should have zero sensitivity.
    a = np.where(num_superobs == 0, 0.0, a)

    outstring3 = f"k = {np.round(k,5)} kg-1 m2 s"
    outstring4 = f"a = {np.round(a,5)} \n"
    outstring5 = f"expectedDOFS: {np.round(sum(a),5)}"

    if config["KalmanMode"]:
        outstring5 += " per inversion period"

    print(outstring3)
    print(outstring4)
    print(outstring5)

    if preview:
        outstrings = (
            f"##{outstring1}\n"
            + f"##{outstring2}\n"
            + f"##{outstring3}\n"
            + f"##{outstring4}\n"
            + outstring5
        )
        return a, df.drop(columns=["time"]), num_days, prior, outstrings
    else:
        return a

def compute_sv_element_stats(
    state_vector_labels,
    areas,
    prior,
    last_ROI_element,
    sum_and_sort_along_statevector,
):
    """
    Compute per-state-vector-element quantities:
      - mean prior flux within each state-vector element
      - L_native = sqrt(mean native grid-cell area)
      - number of native grid cells in each state-vector element

    For xarray inputs, `areas` and `prior` are explicitly aligned to the
    state-vector spatial dimensions before conversion to NumPy. This handles
    singleton non-spatial dimensions such as `time=1` in a cached
    mean-emissions file and avoids boolean-index shape mismatches.

    Parameters
    ----------
    state_vector_labels : xarray.DataArray or ndarray
        State vector label grid, with NaN for background and integer labels.
    areas : xarray.DataArray or ndarray
        Grid-cell areas, spatially matching the state vector after singleton
        non-spatial dimensions are removed.
    prior : xarray.DataArray or ndarray
        Prior flux field, spatially matching the state vector after singleton
        non-spatial dimensions are removed.
    last_ROI_element : int
        Largest label index in ROI (no buffers).
    sum_and_sort_along_statevector : callable
        Function (val, sv, fill_value=np.nan) -> per-label sums, in ascending
        label order.

    Returns
    -------
    emissions : np.ndarray, shape (last_ROI_element,)
        Mean prior flux in each state-vector element.
    L_native : np.ndarray, shape (last_ROI_element,)
        Characteristic native grid-cell length scale in each element.
    num_native_elements : np.ndarray, shape (last_ROI_element,)
        Number of native grid cells in each element.
    """

    def _align_to_state_vector(data, name, sv_dims, sv_shape):
        """Return `data` as a NumPy array matching state-vector dimension order."""
        if isinstance(data, xr.DataArray):
            arr = data

            # Remove only singleton dimensions that are not state-vector dims,
            # e.g. time=1 in a cached mean-emissions file.
            extra_dims = [dim for dim in arr.dims if dim not in sv_dims]
            for dim in extra_dims:
                if arr.sizes[dim] != 1:
                    raise ValueError(
                        f"{name} has unexpected non-spatial dimension "
                        f"{dim}={arr.sizes[dim]}; state-vector dims are {sv_dims}."
                    )
                arr = arr.squeeze(dim=dim, drop=True)

            missing_dims = [dim for dim in sv_dims if dim not in arr.dims]
            if missing_dims:
                raise ValueError(
                    f"{name} is missing state-vector dimension(s) {missing_dims}; "
                    f"{name} dims are {arr.dims}."
                )

            arr = arr.transpose(*sv_dims)
            out = np.asarray(arr.values)
        else:
            # For ndarray input, singleton dimensions are the only dimensions
            # we can safely remove because dimension names are unavailable.
            out = np.squeeze(np.asarray(data))

        if out.shape != sv_shape:
            raise ValueError(
                f"{name} shape {out.shape} does not match state-vector shape "
                f"{sv_shape}."
            )

        return out

    # Preserve xarray dimension names long enough to align GCHP fields.
    if isinstance(state_vector_labels, xr.DataArray):
        sv_dims = state_vector_labels.dims
        sv = np.asarray(state_vector_labels.values)
    else:
        sv = np.squeeze(np.asarray(state_vector_labels))
        sv_dims = None

    sv_shape = sv.shape

    if sv_dims is not None:
        areas_arr = _align_to_state_vector(areas, "areas", sv_dims, sv_shape)
        prior_arr = _align_to_state_vector(prior, "prior", sv_dims, sv_shape)
    else:
        areas_arr = np.squeeze(np.asarray(areas))
        prior_arr = np.squeeze(np.asarray(prior))
        if areas_arr.shape != sv_shape:
            raise ValueError(
                f"areas shape {areas_arr.shape} does not match state-vector "
                f"shape {sv_shape}."
            )
        if prior_arr.shape != sv_shape:
            raise ValueError(
                f"prior shape {prior_arr.shape} does not match state-vector "
                f"shape {sv_shape}."
            )

    # (a) total area per SV element (ROI + buffers, then slice ROI)
    area_per_sv_all = sum_and_sort_along_statevector(
        val=areas_arr,
        sv=sv,
    )
    area_per_sv = area_per_sv_all[:last_ROI_element]

    # (b) number of native grid cells per SV element
    ones_arr = np.ones_like(areas_arr, dtype=float)
    cell_count_per_sv_all = sum_and_sort_along_statevector(
        val=ones_arr,
        sv=sv,
    )
    cell_count_per_sv = cell_count_per_sv_all[:last_ROI_element]

    # (c) native length scale L_native = sqrt(mean cell area)
    mean_area_per_sv = area_per_sv / np.maximum(cell_count_per_sv, 1.0)
    L_native_per_sv = np.sqrt(mean_area_per_sv)

    # (d) mean prior flux per SV element
    emissions_per_sv_all = sum_and_sort_along_statevector(
        val=prior_arr * areas_arr,
        sv=sv,
    )
    flux_per_sv = (emissions_per_sv_all / area_per_sv_all)[:last_ROI_element]

    return flux_per_sv, L_native_per_sv, cell_count_per_sv

if __name__ == "__main__":
    try:
        config_path = sys.argv[1]
        state_vector_path = sys.argv[2]
        preview_dir = sys.argv[3]
        tropomi_cache = sys.argv[4]
        kf_index = int(sys.argv[5]) if len(sys.argv) > 5 else None

        imi_preview(
            config_path,
            state_vector_path,
            preview_dir,
            tropomi_cache,
            kf_index=kf_index,
        )
    except Exception as err:
        with open(os.path.join(preview_dir, ".preview_error_status.txt"), "w") as file1:
            # Writing data to a file
            file1.write(
                "This file is used to tell the controlling script that the imi_preview failed"
            )
        print(err)
        sys.exit(1)
