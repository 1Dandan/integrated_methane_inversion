#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import glob
import numpy as np
import re
import os
import datetime
import yaml
import gc
from src.inversion_scripts.utils import (
    save_obj_atomic,
    get_shared_end_date,
    read_empty_granules,
    write_converted_manifest,
    write_stage_marker,
)
from src.inversion_scripts.operators.TROPOMI_operator import (
    apply_average_tropomi_operator,
    apply_tropomi_operator,
)
from joblib import Parallel, delayed


def apply_operator(operator, params, config):
    """
    Run the chosen operator based on selected instrument

    Arguments
        operator [str]    : Data conversion operator to use
        params   [dict]   : parameters to run the given operator
    Returns
        output   [dict]   : Dictionary with:
                            - obs_GC : GEOS-Chem and TROPOMI methane data
                            - TROPOMI methane
                            - GEOS-Chem methane
                            - TROPOMI lat, lon
                            - TROPOMI lat index, lon index
                              If build_jacobian=True, also include:
                                - K      : Jacobian matrix
    """
    if operator == "TROPOMI_average":
        return apply_average_tropomi_operator(
            params["filename"],
            params["BlendedTROPOMI"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["build_jacobian"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    elif operator == "TROPOMI":
        return apply_tropomi_operator(
            params["filename"],
            params["BlendedTROPOMI"],
            params["n_elements"],
            params["gc_startdate"],
            params["gc_enddate"],
            params["xlim"],
            params["ylim"],
            params["gc_cache"],
            params["period_i"],
            config,
            params["use_water_obs"],
        )
    else:
        raise ValueError("Error: invalid operator selected.")


if __name__ == "__main__":
    
    config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    startday = sys.argv[2]
    endday = sys.argv[3]
    lonmin = float(sys.argv[4])
    lonmax = float(sys.argv[5])
    latmin = float(sys.argv[6])
    latmax = float(sys.argv[7])
    n_elements = int(sys.argv[8])
    tropomi_cache = sys.argv[9]
    BlendedTROPOMI = sys.argv[10].lower() == "true"
    use_water_obs = sys.argv[11].lower() == "true"
    isPost = sys.argv[12]
    period_i = int(sys.argv[13])
    build_jacobian = sys.argv[14]
    viz_prior = sys.argv[15]

    # Reformat start and end days for datetime in configuration
    start = f"{startday[0:4]}-{startday[4:6]}-{startday[6:8]} 00:00:00"
    end = f"{endday[0:4]}-{endday[4:6]}-{endday[6:8]} 23:59:59"

    # Configuration
    workdir = "."
    if build_jacobian.lower() == "true":
        build_jacobian = True
    else:
        build_jacobian = False
    if isPost.lower() == "false":  # if sampling prior simulation
        gc_cache = f"{workdir}/data_geoschem"
        outputdir = f"{workdir}/data_converted"
        vizdir = f"{workdir}/data_visualization"

        # for lognormal, we also sample the prior simulation in a
        # separate call to jacobian.py solely for visualization purposes
        if viz_prior.lower() == "true":
            gc_cache = f"{gc_cache}_prior"
            outputdir = f"{outputdir}_prior"
            vizdir = f"{vizdir}_prior"

    else:  # if sampling posterior simulation
        gc_cache = f"{workdir}/data_geoschem_posterior"
        outputdir = f"{workdir}/data_converted_posterior"
        vizdir = f"{workdir}/data_visualization_posterior"
    xlim = [lonmin, lonmax]
    ylim = [latmin, latmax]
    gc_startdate = np.datetime64(datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S"))
    gc_enddate = np.datetime64(
        datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=1)
    )
    

    # gc_shared_enddate bounds granule selection while the perturbation runs
    # are still advancing: a granule is only processed once every run can
    # supply its columns, else its K row would come from a subset of the state
    # vector. It tracks progress, not intent, so it can exceed a lowered
    # EndDate; sub-range inversions need a reworked precomputed-Jacobian path.
    RunName = config["RunName"]
    RunDirs = os.path.join(os.path.expandvars(config["OutputPath"]), config["RunName"])

    shared_end_date = get_shared_end_date(
        jacobian_root=os.path.join(RunDirs, "jacobian_runs"),
        run_name=RunName,
        start_date=startday,
    )
    shared_end = f"{shared_end_date[0:4]}-{shared_end_date[4:6]}-{shared_end_date[6:8]} 23:59:59"

    # A granule dated D carries observations whose local date is D or D-1,
    # because local time is UTC + lon/15 and lon spans [-180, 180).
    #
    # Without SatDiagOperator the operator reads OutputDir, which holds every
    # UTC date through shared_end_date - 1, so granules run that far.
    #
    # With SatDiagOperator it reads one overpass file per local date instead,
    # and overpass output stops at its last local date: shared_end_date - 2
    # while the runs are still advancing, because sampling local date L needs
    # OutputDir for both L and L+1. A granule dated shared_end_date - 1 would
    # then ask for an overpass file that cannot exist yet. Once
    # shared_end_date reaches EndDate there is no further day to wait for, and
    # the two cases coincide. This mirrors build_date_list in
    # calculate_satellite_overpass_diagnostics.py.
    shared_end_offset = 1

    if config.get("SatDiagOperator", False) and shared_end_date != str(endday):
        shared_end_offset = 2

    gc_shared_enddate = np.datetime64(
        datetime.datetime.strptime(shared_end, "%Y-%m-%d %H:%M:%S")
        - datetime.timedelta(days=shared_end_offset)
    )
    print(f"Latest shared date (exclusive): {shared_end_date}")
    print(f"Latest granule date (inclusive): {str(gc_shared_enddate)[0:10]}")
    
    print("Start:", gc_startdate)
    print("End:", gc_enddate)
    
    # Get TROPOMI data filenames for the desired date range
    allfiles = glob.glob(f"{tropomi_cache}/*.nc")
    sat_files = []
    for index in range(len(allfiles)):
        filename = allfiles[index]
        shortname = re.split(r"\/", filename)[-1]
        shortname = re.split(r"\.", shortname)[0]
        strdate = re.split(r"\.|_+|T", shortname)[4]
        strdate = datetime.datetime.strptime(strdate, "%Y%m%d")
        if (strdate >= gc_startdate) and (strdate <= gc_shared_enddate):
            sat_files.append(filename)
    sat_files.sort()
    print("Found", len(sat_files), "TROPOMI data files.")

    # Whether a granule survives filtering depends only on these, so a
    # no_valid_obs verdict stays valid until one of them changes.
    filter_signature = [
        str(startday), str(endday),
        lonmin, lonmax, latmin, latmax,
        bool(use_water_obs), bool(BlendedTROPOMI),
    ]

    known_empty = read_empty_granules(workdir, filter_signature)

    if known_empty:
        print(
            f"Skipping {len(known_empty)} granule(s) previously found to hold "
            "no usable observation"
        )

    # Map GEOS-Chem to TROPOMI observation space
    # Also return Jacobian matrix if build_jacobian=True
    def process(filename):

        # Check if TROPOMI file has already been processed
        print("========================")
        shortname = re.split(r"\/", filename)[-1]
        print(shortname)
        date = re.split(r"\.", shortname)[0]

        if date in known_empty:
            # Re-reading it would filter every observation out again, which
            # costs a full granule read for a known answer.
            print("No usable observation last time; skipping.")
            return (date, "no_valid_obs")

        # If not yet processed, run apply_average_tropomi_operator()
        if not os.path.isfile(f"{outputdir}/{date}_GCtoTROPOMI.pkl"):
            print("Applying TROPOMI operator...")

            output = apply_operator(
                "TROPOMI_average",
                {
                    "filename": filename,
                    "BlendedTROPOMI": BlendedTROPOMI,
                    "n_elements": n_elements,
                    "gc_startdate": gc_startdate,
                    "gc_enddate": gc_enddate,
                    "xlim": xlim,
                    "ylim": ylim,
                    "gc_cache": gc_cache,
                    "build_jacobian": build_jacobian,
                    "period_i": period_i,
                    "use_water_obs": use_water_obs,
                    "isPost": isPost.lower() == "true"
                },
                config,
            )

            # we also save out the unaveraged tropomi operator for visualization purposes
            viz_output = apply_operator(
                "TROPOMI",
                {
                    "filename": filename,
                    "BlendedTROPOMI": BlendedTROPOMI,
                    "n_elements": n_elements,
                    "gc_startdate": gc_startdate,
                    "gc_enddate": gc_enddate,
                    "xlim": xlim,
                    "ylim": ylim,
                    "gc_cache": gc_cache,
                    "period_i": period_i,
                    "use_water_obs": use_water_obs,
                },
                config,
            )
            
            if output == None:
                # No valid observations in this granule, so no pickle is
                # written. Reported rather than silently skipped, so a
                # legitimately empty granule is distinguishable from one that
                # never got processed.
                return (date, "no_valid_obs")
            print("Saving .pkl file")
            save_obj_atomic(output, f"{outputdir}/{date}_GCtoTROPOMI.pkl")
            save_obj_atomic(viz_output, f"{vizdir}/{date}_GCtoTROPOMI.pkl")

            del output, viz_output
            gc.collect()
            return (date, "written")
        else:
            return (date, "cached")

    results = Parallel(n_jobs=-1)(delayed(process)(filename) for filename in sat_files)
    print(f"Wrote files to {outputdir}")

    # Record that data_converted covers [startday, shared_end_date). Only the
    # prior-simulation pass writes these: the posterior pass fills
    # data_converted_posterior, and the lognormal pass only refreshes
    # visualization output, so neither one completes this stage.
    if isPost.lower() == "false" and viz_prior.lower() == "false":
        # The manifest names every granule this window selected and what
        # became of it, so completeness can later be confirmed exactly
        # instead of inferred from which dates happen to have output.
        write_converted_manifest(
            workdir,
            startday,
            shared_end_date,
            results,
            filter_signature,
        )

        write_stage_marker(
            RunDirs,
            "inversion",
            startday,
            shared_end_date,
        )