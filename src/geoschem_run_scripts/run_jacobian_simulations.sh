#!/bin/bash
#SBATCH -J {RunName}

# Expected number of hourly records in each daily SpeciesConc file
EXPECTED_TIME_STEPS=24


# ----------------------------------------------------------------------
# Return the length of the unlimited time dimension
# ----------------------------------------------------------------------
get_time_length() {
    local file="$1"
    local time_len

    [[ -f "$file" && -s "$file" ]] || return 1

    time_len=$(
        ncdump -h "$file" 2>/dev/null |
        sed -nE \
            's/^[[:space:]]*time = UNLIMITED ;[[:space:]]*\/\/[[:space:]]*\(([0-9]+) currently\).*/\1/p' |
        head -n 1
    )

    [[ "$time_len" =~ ^[0-9]+$ ]] || return 1

    printf '%s\n' "$time_len"
}


# ----------------------------------------------------------------------
# Check whether a daily SpeciesConc file is complete
# ----------------------------------------------------------------------
is_valid_nc() {
    local file="$1"
    local time_len

    # File must exist and must not be empty
    [[ -f "$file" && -s "$file" ]] || return 1

    # Validate the NetCDF structure/header
    if ! ncks -m "$file" >/dev/null 2>&1; then
        return 1
    fi

    # Extract the length of the time dimension
    if ! time_len=$(get_time_length "$file"); then
        return 1
    fi

    # A complete daily file must contain exactly 24 hourly records
    [[ "$time_len" -eq "$EXPECTED_TIME_STEPS" ]]
}


# ----------------------------------------------------------------------
# Remove all output collections belonging to one incomplete date
# ----------------------------------------------------------------------
remove_output_date() {
    local output_dir="$1"
    local file_date="$2"
    local file
    local found=0

    echo "Removing output files for incomplete date: ${file_date}"

    while IFS= read -r -d '' file; do
        found=1
        echo "  Removing: $file"

        if ! rm -f -- "$file"; then
            echo "ERROR: Failed to remove $file" >&2
            return 1
        fi
    done < <(
        find "$output_dir" \
            -maxdepth 1 \
            -type f \
            -name "GEOSChem.*.${file_date}_0000z.nc4" \
            -print0
    )

    if [[ "$found" -eq 0 ]]; then
        echo "  No matching files found."
    fi
}


# ----------------------------------------------------------------------
# Find incomplete daily SpeciesConc files and remove every collection
# for those dates
# ----------------------------------------------------------------------
remove_incomplete_output_dates() {
    local output_dir="$1"
    local species_file
    local filename
    local file_date
    local time_len
    local date_to_remove

    declare -A incomplete_dates=()

    [[ -d "$output_dir" ]] || {
        echo "Output directory does not exist yet: $output_dir"
        return 0
    }

    echo "Checking existing SpeciesConc files for incomplete dates..."

    while IFS= read -r -d '' species_file; do
        filename=${species_file##*/}

        if [[ "$filename" =~ ^GEOSChem\.SpeciesConc\.([0-9]{8})_0000z\.nc4$ ]]; then
            file_date=${BASH_REMATCH[1]}
        else
            continue
        fi

        if ! time_len=$(get_time_length "$species_file"); then
            echo "Incomplete or unreadable file:"
            echo "  $species_file"
            incomplete_dates["$file_date"]=1
            continue
        fi

        if [[ "$time_len" -lt "$EXPECTED_TIME_STEPS" ]]; then
            echo "Incomplete file: ${time_len}/${EXPECTED_TIME_STEPS} records"
            echo "  $species_file"
            incomplete_dates["$file_date"]=1

        elif [[ "$time_len" -eq "$EXPECTED_TIME_STEPS" ]]; then
            echo "Complete file: ${file_date}, ${time_len}/${EXPECTED_TIME_STEPS} records"

        else
            echo "WARNING: File contains more than ${EXPECTED_TIME_STEPS} records:" >&2
            echo "  Records: $time_len" >&2
            echo "  File: $species_file" >&2
            echo "  It will not be deleted automatically." >&2
        fi
    done < <(
        find "$output_dir" \
            -maxdepth 1 \
            -type f \
            -name 'GEOSChem.SpeciesConc.[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_0000z.nc4' \
            -print0
    )

    if [[ "${#incomplete_dates[@]}" -eq 0 ]]; then
        echo "No incomplete SpeciesConc dates found."
        return 0
    fi

    # If SpeciesConc is incomplete for a date, remove all collections for
    # that date. This prevents MAPL from reopening a mixture of old and
    # newly created NetCDF files.
    while IFS= read -r date_to_remove; do
        [[ -n "$date_to_remove" ]] || continue

        if ! remove_output_date "$output_dir" "$date_to_remove"; then
            return 1
        fi
    done < <(
        printf '%s\n' "${!incomplete_dates[@]}" | sort
    )

    echo "Incomplete-output cleanup completed."
}


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

### Run directory
RUNDIR=$(pwd -P)

### Get current task ID
x=${SLURM_ARRAY_TASK_ID}

### Add zeros to the cluster ID
printf -v xstr "%04d" "$x"

### Shared error-status file
ERROR_STATUS_FILE="${RUNDIR}/.error_status_file.txt"

# This checks for the presence of the error status file. If present, this
# indicates that a prior Jacobian exited with an error, so this task will
# not run.
if [[ -f "$ERROR_STATUS_FILE" ]]; then
    echo "$ERROR_STATUS_FILE exists. Exiting."
    echo "Jacobian simulation ${xstr} exited without running."
    exit 1
fi

SIM_DIR="${RUNDIR}/{RunName}_${xstr}"
RUN_SCRIPT="./{RunName}_${xstr}.run"
OUTPUT_DIR="OutputDir"

if [[ ! -d "$SIM_DIR" ]]; then
    echo "ERROR: Simulation directory does not exist:" >&2
    echo "  $SIM_DIR" >&2
    exit 1
fi

cd "$SIM_DIR" || {
    echo "ERROR: Could not enter simulation directory:" >&2
    echo "  $SIM_DIR" >&2
    exit 1
}

if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "ERROR: Run script does not exist or is not executable:" >&2
    echo "  ${SIM_DIR}/{RunName}_${xstr}.run" >&2
    exit 1
fi


if {ReDoJacobian}; then

    # Remove incomplete output left by a cancelled or failed run.
    #
    # When SpeciesConc is incomplete for a date, all GEOSChem NetCDF
    # collections for that date are deleted so MAPL can recreate them
    # consistently.
    if ! remove_incomplete_output_dates "$OUTPUT_DIR"; then
        echo "ERROR: Failed while removing incomplete output files." >&2
        exit 1
    fi

    # Check the final expected daily SpeciesConc file.
    # The simulation end date is exclusive, so the final output date is
    # one day before EndDate.
    yyyymmdd={EndDate}
    last_date=$(date -d "${yyyymmdd} -1 day" +%Y%m%d)
    LastConcFile="${OUTPUT_DIR}/GEOSChem.SpeciesConc.${last_date}_0000z.nc4"

    if is_valid_nc "$LastConcFile"; then
        echo "Final SpeciesConc file is complete:"
        echo "  $LastConcFile"
        echo "Not re-running Jacobian simulation: ${xstr}"
        exit 0
    fi

    echo "Re-running Jacobian simulation: ${xstr}"
    "$RUN_SCRIPT"
    retVal=$?

else

    echo "Running Jacobian simulation: ${xstr}"
    "$RUN_SCRIPT"
    retVal=$?

fi


# ----------------------------------------------------------------------
# Check whether the Jacobian finished successfully
# ----------------------------------------------------------------------

if [[ "$retVal" -ne 0 ]]; then
    echo "Error Status: $retVal" > "$ERROR_STATUS_FILE"

    echo "Jacobian simulation ${xstr} exited with error code: $retVal"
    echo "Check the log file in the following directory:"
    echo "  ${RUNDIR}/{RunName}_${xstr}"

    exit "$retVal"
fi

echo "Finished Jacobian simulation: ${xstr}"

exit 0