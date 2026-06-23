#!/usr/bin/env python3
"""
Generate overpass-time lookup grids for extracting hourly model
diagnostics at satellite overpass time.

The core computation is purely analytical (depends only on each
cell's longitude and latitude), so it can be evaluated directly
on any grid — no regridding needed.

Includes a latitude-dependent overpass offset based on orbital
geometry: the satellite takes finite time to travel from the
equator to higher latitudes, shifting the local solar time of
the overpass. Uses the proper arcsin relationship between
geographic latitude and orbital angle for a given inclination.
  
Simulation grid: specify --grid_file to read lon/lat from
     an existing model output or grid file

Default orbital parameters match Sentinel-5P / TROPOMI:
  - Ascending node equatorial crossing: 13:30 Mean Local Solar Time
  - Inclination: 98.7° (max latitude ~81.3°)
  - 14 orbits/day (227 orbits per 16-day cycle)
  - Reference altitude: ~824 km

Usage:
    # Directly on simulation grid
    python generate_overpass_grids.py --overpass_time 13:30 \
        --orbits_per_day 14 --direction ascending \
        --grid_file my_simulation.nc -o overpass_sim.nc (--gchp if GCHP grid)
    # or simply 
    python generate_overpass_grids.py --grid_file my_simulation.nc -o overpass_sim.nc (--gchp if GCHP grid)

Useful variables for extracting diagnostics at overpass time:
  - closest_hour: integer UTC hour of archived diagnostic closest to overpass time (0-23)
  - day_offset: which day's file to read relative to observation date (-1=prev day, 0=same day, +1=next day)
  # For TROPOMI, the day_offset is only 0 and +1, never -1, because the overpass time is in the afternoon. 
  # For a morning satellite (e.g., Aqua), you would get only 0 and -1.
"""

import argparse
import numpy as np
import xarray as xr

import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"xarray.*"
)

def compute_overpass(lon_arr, lat_arr, overpass_time_str='13:30',
                     orbits_per_day=14, direction='ascending',
                     inclination=98.7):
    """
    Core computation: given arrays of longitude and latitude (any shape),
    compute the overpass time fields.

    Parameters
    ----------
    lon_arr : np.ndarray
        Array of longitudes in degrees (-180 to 180 or 0 to 360).
        Can be any shape: 1D, 2D (lat x lon), 3D (nf x Ydim x Xdim), etc.
    lat_arr : np.ndarray
        Array of latitudes in degrees (-90 to 90). Same shape as lon_arr.
    overpass_time_str : str
        Local solar overpass time at the equator in HH:MM format.
    orbits_per_day : int
        Number of orbits per day.
    direction : str
        'ascending' or 'descending'.
    inclination : float
        Orbital inclination in degrees.

    Returns
    -------
    dict with keys: utc_offset, overpass_lst, closest_hour, day_offset,
                    max_lat, overpass_lst_equator
    """
    # Parse overpass time
    hh, mm = map(int, overpass_time_str.split(':'))
    overpass_lst = hh + mm / 60.0

    # Normalize longitudes to -180..180
    lon = lon_arr.copy().astype(np.float64)
    lon[lon > 180] -= 360

    # Solar UTC offset
    utc_offset = lon / 15.0

    # Orbital geometry
    incl_rad = np.radians(inclination)
    max_lat = np.degrees(np.arcsin(np.abs(np.sin(incl_rad))))
    T_quarter = 24.0 / orbits_per_day / 4.0

    lat_clamped = np.clip(np.abs(lat_arr), 0, max_lat - 0.01)
    orbital_angle = np.arcsin(
        np.sin(np.radians(lat_clamped)) / np.sin(incl_rad)
    )
    quarter_fraction = orbital_angle / (np.pi / 2)

    overpass_offset_hours = np.sign(lat_arr) * quarter_fraction * T_quarter
    if direction == 'descending':
        overpass_offset_hours = -overpass_offset_hours

    overpass_lst_2d = overpass_lst + overpass_offset_hours

    # Raw UTC and day boundary handling
    overpass_utc_raw = overpass_lst_2d - utc_offset
    closest_hour_raw = np.round(overpass_utc_raw).astype(int)

    day_offset = np.zeros_like(closest_hour_raw)
    day_offset[closest_hour_raw >= 24] = 1
    day_offset[closest_hour_raw < 0] = -1

    closest_hour = closest_hour_raw % 24

    return {
        'utc_offset': utc_offset.astype(np.float32),
        'overpass_lst': (overpass_lst_2d % 24.0).astype(np.float32),
        'closest_hour': closest_hour.astype(np.int16),
        'day_offset': day_offset.astype(np.int8),
        'max_lat': max_lat,
        'overpass_lst_equator': overpass_lst,
    }


def _add_attributes(ds, overpass_time_str, orbits_per_day, direction,
                    inclination, max_lat):
    """Add CF-style attributes to the dataset."""
    ds['utc_offset'].attrs = {
        'long_name': 'Solar time UTC offset',
        'units': 'hours',
        'description': 'Local solar time minus UTC (= longitude / 15)',
    }
    ds['overpass_lst'].attrs = {
        'long_name': 'Local solar time of overpass',
        'units': 'hours',
        'description': (
            f'Actual local solar time of the overpass at each grid cell, '
            f'accounting for latitude-dependent orbital shift. '
            f'Equals {overpass_time_str} only at the equator.'
        ),
    }
    ds['closest_hour'].attrs = {
        'long_name': 'Closest archived UTC hour index to overpass',
        'units': '1',
        'valid_range': [0, 23],
        'description': (
            'Integer UTC hour index (0-23) of the archived hourly diagnostic '
            'closest to the satellite overpass time. '
            'Use with day_offset to determine which day\'s file to read.'
        ),
    }

    ds['day_offset'].attrs = {
        'long_name': 'Day offset index for closest archived hour',
        'units': '1',
        'valid_range': [-1, 1],
        'description': (
            'Integer day index relative to the observation date. '
            '0 = same day, +1 = next day, -1 = previous day.'
        ),
    }
    ds.attrs = {
        'title': f'Overpass time grids for {overpass_time_str} LST (equatorial)',
        'overpass_local_solar_time_at_equator': overpass_time_str,
        'orbits_per_day': orbits_per_day,
        'direction': direction,
        'inclination_degrees': inclination,
        'max_latitude_degrees': float(np.round(max_lat, 1)),
        'description': (
            'Overpass time lookup grids for extracting hourly model '
            'diagnostics at satellite overpass time. Uses arcsin orbital '
            'model for latitude-dependent overpass shift.'
        ),
    }


def _print_summary(results, overpass_time_str, orbits_per_day,
                   direction, inclination, output_path):
    """Print summary statistics."""
    max_lat = results['max_lat']
    day_offset = results['day_offset']
    overpass_lst = results['overpass_lst']
    closest_hour = results['closest_hour']

    print(f'Wrote {output_path}')
    print(f'  Overpass LST at equator: {overpass_time_str}')
    print(f'  Orbits/day: {orbits_per_day}, direction: {direction}')
    print(f'  Inclination: {inclination}°, max latitude: {max_lat:.1f}°')
    print(f'  Lat-dependent LST range: '
          f'{overpass_lst.min():.2f}–{overpass_lst.max():.2f} h')
    print(f'  closest_hour range: '
          f'{int(closest_hour.min())}–{int(closest_hour.max())} UTC')
    n_next = int(np.sum(day_offset == 1))
    n_prev = int(np.sum(day_offset == -1))
    n_total = day_offset.size
    print(f'  day_offset: {n_prev} cells prev day, '
          f'{n_total - n_next - n_prev} same day, {n_next} next day')


def generate_on_sim_grid(grid_file, overpass_time_str, output_path,
                         orbits_per_day=14, direction='ascending',
                         inclination=98.7, isGCHP=False):
    """
    Generate overpass grids directly on a simulation grid.

    Parameters
    ----------
    grid_file : str
        Path to a NetCDF file containing the grid's longitude and
        latitude fields (any resolution, any grid type).
    lon_name, lat_name : str or None
        Variable names for longitude and latitude in grid_file.
        If None, auto-detected from common names.
    """
    src = xr.open_dataset(grid_file)

    # Auto-detect lon/lat variable names
    # GCC uses lat/lon, GCHP uses lats/lons
    if isGCHP:
        lon_name = 'lons'
        lat_name = 'lats'
    else:
        lon_name = 'lon'
        lat_name = 'lat'

    lon_da = src[lon_name]
    lat_da = src[lat_name]

    if isGCHP:
        lon_vals = lon_da.values
        lat_vals = lat_da.values
        dims = ('nf', 'Ydim', 'Xdim')
        coords=dict(lats=(['nf', 'Ydim', 'Xdim'], lat_da.values),
                    lons=(['nf', 'Ydim', 'Xdim'], lon_da.values))
    else:
        dims = ('lat', 'lon')
        lon_vals, lat_vals = np.meshgrid(lon_da.values, lat_da.values)
        coords=dict(lat=(['lat'], lat_da.values),
                    lon=(['lon'], lon_da.values))
    results = compute_overpass(lon_vals, lat_vals, overpass_time_str,
                               orbits_per_day, direction, inclination)

    ds = xr.Dataset(
        {k: (dims, results[k])
         for k in ['utc_offset', 'overpass_lst', 'closest_hour', 'day_offset']},
        coords=coords,
    )

    # add corner_lats/lons 
    if isGCHP:
        ds['corner_lons'] = src['corner_lons']
        ds['corner_lats'] = src['corner_lats']
    _add_attributes(ds, overpass_time_str, orbits_per_day, direction,
                    inclination, results['max_lat'])
    ds.attrs['source_grid'] = grid_file
    ds[lat_name].attrs = src[lat_name].attrs
    ds[lon_name].attrs = src[lon_name].attrs

    encoding = {k: {'dtype': str(ds[k].dtype), 'zlib': True, 'complevel': 4}
                for k in ['utc_offset', 'overpass_lst', 'closest_hour', 'day_offset']}
    ds.to_netcdf(output_path, encoding=encoding)

    shape_str = ' x '.join(str(s) for s in lon_vals.shape)
    print(f'  Grid: {shape_str}  (from {grid_file})')
    _print_summary(results, overpass_time_str, orbits_per_day,
                   direction, inclination, output_path)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate overpass-time lookup grids for extracting '
                    'hourly diagnostics at satellite overpass time.'
    )
    parser.add_argument(
        '--overpass_time', type=str, default='13:30',
        help='Local solar overpass time at equator in HH:MM (default: 13:30)'
    )
    parser.add_argument(
        '--orbits_per_day', type=int, default=14,
        help='Number of orbits per day (default: 14, e.g. TROPOMI/Sentinel-5P)'
    )
    parser.add_argument(
        '--direction', type=str, default='ascending',
        choices=['ascending', 'descending'],
        help='Orbit direction at equator crossing (default: ascending)'
    )
    parser.add_argument(
        '--inclination', type=float, default=98.7,
        help='Orbital inclination in degrees (default: 98.7 for Sentinel-5P)'
    )
    parser.add_argument(
        '-o', type=str, default='overpass_grids.nc',
        help='Output NetCDF filename (default: overpass_grids.nc)'
    )
    parser.add_argument(
        '--gchp',
        action='store_true',
        help='Input grid file is on a GCHP cubed-sphere grid'
    )

    # Grid source: --grid_file
    parser.add_argument(
        '--grid_file', type=str, default=None,
        help='NetCDF file to read simulation grid from (lon/lat auto-detected)'
    )

    args = parser.parse_args()

    generate_on_sim_grid(
        args.grid_file,
        args.overpass_time,
        args.o,
        args.orbits_per_day,
        args.direction,
        args.inclination,
        isGCHP=args.gchp,
    )
