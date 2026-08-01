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