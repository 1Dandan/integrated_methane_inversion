#!/usr/bin/env python3
"""Verify satellite overpass diagnostics for one target face.

Run this ONCE per face, before reprocessing, on overpass data that has come
back from S3. It answers a single question for every expected file: is this
file exactly what the current processing scripts would produce?

    existence -> openability -> expected variables -> no NaN

The StartDate-1 file (20241231) is mostly NaN by construction: overpass
sampling of local date L reads UTC L and L+1, and UTC L is outside the
simulation window for that one date.

Whether that counts as damage depends on whether the face has been processed,
so the overpass marker decides it:

  no marker    the face has not been processed, so the file is flagged and
               rebuilt along with everything else. process_run_day can build
               it from the next day alone, so no OutputDir is needed for it.
  marker       the overpass stage completed and wrote that file deliberately.
               Flagging it would delete a correct file and, since deleting
               overpass output retires the marker, reprocess the whole window
               -- and on a face that has since been pruned the OutputDir needed
               to rebuild it is gone, so the deletion cannot be undone.

--allow-nan-first-date forces the exemption on regardless, which is what
process_face_cycle.sh passes.

Because that turns on the marker rather than on a flag, a sweep across every
face does the right thing per face: unprocessed ones get their StartDate-1
file rebuilt, processed ones keep theirs.

After reprocessing, the overpass_complete marker is the statement of
completeness; files are written atomically (temp file + os.replace), so a
completed run cannot leave a partial file behind.

Paths that failed are printed to stdout, one per line, for
delete_files_from_list.sh. Everything else, including files that are simply
missing and will be created by reprocessing, goes to stderr.

Usage:
    check_overpass_complete.py CONFIG [N_ELEMENTS] [options] > bad_files.txt

Exit codes:
    0  every expected file is present and complete, or no overpass files exist
    1  usage or configuration error
    2  at least one file is missing or incomplete
"""

import argparse
import concurrent.futures as cf
import contextlib
import multiprocessing as mp
import os
import sys
import threading

# HDF5 file locking must be disabled BEFORE netCDF4 (and thus libhdf5) is
# imported. The POSIX locks HDF5 takes are unreliable on network filesystems
# and can block a read indefinitely. setdefault keeps an explicit environment
# override working.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import yaml
from netCDF4 import Dataset

REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.inversion_scripts.calculate_satellite_overpass_diagnostics import (  # noqa: E402
    build_date_list,
    get_keepvars,
)
from src.inversion_scripts.utils import (  # noqa: E402
    build_pert_simulations_dict,
    read_stage_marker,
)

# Grid metadata copied onto every diagnostic by attach_overpass_grid_metadata.
# Excluded from the NaN scan so a grid convention can never look like data
# corruption.
GRID_METADATA_VARS = frozenset({
    "lat",
    "lon",
    "lats",
    "lons",
    "corner_lats",
    "corner_lons",
})

# Variables the TROPOMI satellite-diagnostic operator reads out of the base
# run. Checked explicitly so a file that opens but lost a collection is not
# mistaken for a complete one.
REQUIRED_BASERUN_VARS = {
    "GEOSChem.SpeciesConc": {
        "SpeciesConcVV_CH4",
        "Met_AIRDEN",
        "Met_BXHEIGHT",
    },
    "GEOSChem.BaseSpeciesConc": {
        "SpeciesConcVV_CH4",
        "Met_AIRDEN",
        "Met_BXHEIGHT",
    },
    "GEOSChem.StateMetLevEdge": {
        "Met_PEDGE",
    },
}

MAX_VALUES_PER_READ = 10_000_000


def _isolation_context():
    """A start method whose children inherit none of this process's locks.

    The readers run in a thread pool and each one starts a child, so the naive
    choice -- fork -- is the wrong one. Only the forking thread exists in the
    child, so a lock another thread happened to hold at that instant is held
    forever by a thread that is not there. The child then deadlocks inside
    libhdf5 or the allocator, the parent waits out its timeout, and a perfectly
    good file is reported as a read that never finished.

    That failure is intermittent by nature: it depends on what the other
    threads were doing at the moment of the fork, so it lands on a different
    file every run and never reproduces when one file is checked on its own.

    forkserver forks from a separate, single-threaded helper, so there is
    nothing to inherit. spawn starts a fresh interpreter and is the fallback.
    """
    for method in ("forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except ValueError:
            continue

    return mp.get_context("fork")


FORK = _isolation_context()

# Serialize the fork() call itself so two forks are never concurrent.
_FORK_LOCK = threading.Lock()


def variable_contains_nan(variable):
    """Return True when a variable holds at least one NaN.

    The whole variable is read, in slices along the first dimension so a
    large field never has to fit in memory at once.
    """
    if variable.size == 0:
        return False

    if variable.ndim == 0:
        return bool(np.isnan(np.asarray(variable[...])).any())

    values_per_index = int(np.prod(variable.shape[1:])) or 1

    indices_per_read = max(1, MAX_VALUES_PER_READ // values_per_index)

    for start in range(0, variable.shape[0], indices_per_read):
        stop = min(start + indices_per_read, variable.shape[0])

        index = (slice(start, stop),) + (slice(None),) * (variable.ndim - 1)

        if np.isnan(np.asarray(variable[index])).any():
            return True

    return False


def inspect_file(file_path, expected_vars):
    """Inspect one overpass file. Returns (status, detail)."""
    try:
        with Dataset(file_path, "r") as dataset:
            # Return stored values rather than masked arrays, so a fill value
            # is never silently hidden.
            dataset.set_auto_mask(False)

            present = set(dataset.variables)

            missing = sorted(set(expected_vars) - present)

            if missing:
                return "MISSING_VARIABLE", ", ".join(missing)

            for name, variable in dataset.variables.items():
                if name in GRID_METADATA_VARS:
                    continue

                try:
                    dtype = np.dtype(variable.dtype)
                except TypeError:
                    continue

                if not np.issubdtype(dtype, np.floating):
                    continue

                if variable.size == 0:
                    return "EMPTY_VARIABLE", name

                if variable_contains_nan(variable):
                    return "NAN", name

            return "COMPLETE", None

    except FileNotFoundError:
        return "MISSING", None

    except Exception as error:
        return "READ_ERROR", str(error)


def _inspect_target(file_path, expected_vars, conn):
    """Run one inspection inside a dedicated child process."""
    try:
        result = inspect_file(file_path, expected_vars)
    except Exception as error:  # pragma: no cover - defensive
        result = ("READ_ERROR", repr(error))

    try:
        conn.send(result)
    finally:
        conn.close()


def inspect_isolated(file_path, expected_vars, timeout):
    """Inspect one file in a child process, surviving hangs and hard crashes.

    A corrupt file can hang or segfault inside libhdf5, which would take down
    a shared worker pool. Isolating each read means one bad file can never
    stall or kill the scan.
    """
    parent_conn, child_conn = FORK.Pipe(duplex=False)

    proc = FORK.Process(
        target=_inspect_target,
        args=(file_path, expected_vars, child_conn),
    )

    with _FORK_LOCK:
        proc.start()

    # The child owns the write end; the parent only reads.
    child_conn.close()

    result = None

    if parent_conn.poll(None if timeout <= 0 else timeout):
        try:
            result = parent_conn.recv()
        except EOFError:
            result = None

    parent_conn.close()
    proc.join(1)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "TIMEOUT", f"read exceeded {timeout:g}s"

    if result is None:
        return (
            "CRASH",
            f"child exited without a result (exitcode={proc.exitcode})",
        )

    return result


def derive_n_elements(config, run_dirs):
    """Derive the state vector element count the way setup.sh does."""
    state_vector_path = os.path.join(run_dirs, "StateVector.nc")

    if not os.path.isfile(state_vector_path):
        raise FileNotFoundError(
            f"Cannot derive n_elements: {state_vector_path} does not exist. "
            "Pass the count explicitly as the second argument."
        )

    with Dataset(state_vector_path, "r") as dataset:
        n_elements = int(np.nanmax(dataset.variables["StateVector"][:]))

    if config["OptimizeBCs"]:
        n_elements += 4

    if config["OptimizeOH"]:
        n_elements += 1 if config["isRegional"] else 2

    return n_elements


def build_expected_files(config, n_elements, run_dirs, date_list):
    """Return [(path, expected_vars)] for every file the face should hold.

    Mirrors the expected_outputs logic in process_run_day so the checker and
    the producer cannot drift apart.
    """
    run_name = config["RunName"]
    overpass_tag = config["OverpassTime"].replace(":", "")
    disable_run_0000 = config.get("DisableRun0000", False)

    jacobian_root = os.path.join(run_dirs, "jacobian_runs")

    if not os.path.isdir(jacobian_root):
        print(
            f"NOTE: no jacobian_runs directory: {jacobian_root}",
            file=sys.stderr,
            flush=True,
        )
        return []

    run_names = sorted(
        name
        for name in os.listdir(jacobian_root)
        if os.path.isdir(os.path.join(jacobian_root, name))
    )

    unexpected = [
        name
        for name in run_names
        if not name.startswith(f"{run_name}_")
    ]

    if unexpected:
        print(
            "WARNING: ignoring unrecognized directories under jacobian_runs: "
            + ", ".join(unexpected),
            file=sys.stderr,
            flush=True,
        )

    # Read from the directory names, the same way the producer does. A count
    # used as an exclusive range bound would skip the last run whenever the
    # directories start at _0001.
    start_run = 1 if disable_run_0000 else 0

    run_indices = sorted(
        run_i
        for run_i in (
            int(name.rsplit("_", 1)[1])
            for name in run_names
            if name.startswith(f"{run_name}_")
            and name.rsplit("_", 1)[1].isdigit()
        )
        if run_i >= start_run
    )

    pert_simulations_dict = build_pert_simulations_dict(config, n_elements)

    expected = []

    for run_i in run_indices:
        run_dir = os.path.join(
            jacobian_root,
            f"{run_name}_{run_i:04d}",
        )

        output_dir = os.path.join(run_dir, "OverpassDiagnostics")

        if not os.path.isdir(run_dir):
            print(
                f"WARNING: expected run directory is absent: {run_dir}",
                file=sys.stderr,
                flush=True,
            )

        do_sample_base_3d = (
            run_i == 0
            or (disable_run_0000 and run_i == 1)
        )

        if disable_run_0000 and run_i == 1:
            baserun_file_types = [
                "GEOSChem.BaseSpeciesConc",
                "GEOSChem.StateMetLevEdge",
            ]
        else:
            baserun_file_types = [
                "GEOSChem.SpeciesConc",
                "GEOSChem.StateMetLevEdge",
            ]

        sv_elems = pert_simulations_dict.get(f"{run_i:04d}", [])

        if run_i != 0:
            keepvars = get_keepvars(
                sv_elems,
                n_elements,
                config,
                baserun=(run_i == 1),
            )

            col_vars = [f"{var}_col" for var in keepvars]

            if not col_vars:
                print(
                    f"WARNING: run {run_i:04d} maps to no state vector "
                    "elements; its CH4col files carry no data variables",
                    file=sys.stderr,
                    flush=True,
                )

        for date_str in date_list:
            if do_sample_base_3d:
                for file_prefix in baserun_file_types:
                    expected.append((
                        os.path.join(
                            output_dir,
                            f"{file_prefix}.overpass."
                            f"{date_str}_{overpass_tag}.nc4",
                        ),
                        REQUIRED_BASERUN_VARS[file_prefix],
                    ))

            if run_i != 0:
                expected.append((
                    os.path.join(
                        output_dir,
                        f"GEOSChem.CH4col.overpass."
                        f"{date_str}_{overpass_tag}.nc4",
                    ),
                    col_vars,
                ))

    return expected


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Verify satellite overpass diagnostics for one target face. "
            "Incomplete files are printed to stdout for deletion."
        )
    )

    parser.add_argument(
        "config",
        help="Path to the face's IMI config yaml",
    )

    parser.add_argument(
        "n_elements",
        nargs="?",
        type=int,
        default=None,
        help=(
            "State vector element count. Derived from StateVector.nc when "
            "omitted, using the same rule as setup.sh"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel inspections; default 8",
    )

    parser.add_argument(
        "--allow-nan-first-date",
        action="store_true",
        help=(
            "Do not flag NaN in the StartDate-1 file. That date is only "
            "partly covered -- overpass sampling of local date L reads UTC L "
            "and L+1, and UTC L does not exist for it -- so NaN there is "
            "correct, not damage. Without this the file is flagged on every "
            "run, deleted, regenerated, and flagged again; and because "
            "deleting overpass output also retires the overpass marker, the "
            "whole window reprocesses each time. Set it for repeated or "
            "automated runs; leave it off for a one-off cleanup that is meant "
            "to rebuild that file."
        ),
    )

    parser.add_argument(
        "--flag-nan-first-date",
        action="store_true",
        help=(
            "Flag the StartDate-1 file for deletion even though its NaN is "
            "expected, and even when a marker would otherwise exempt it. Use "
            "for a clean rebuild rather than a resume: the file is deleted, "
            "the overpass marker goes with it, and the whole window is "
            "recomputed. The rebuilt file is partly NaN again -- that is "
            "correct -- so leaving this on makes every pass reprocess "
            "everything. Requires the OutputDir inputs to still be present, "
            "which they are not on a face that has already been pruned."
        ),
    )

    parser.add_argument(
        "--per-file-timeout",
        type=float,
        default=180.0,
        help=(
            "Seconds before a single file's read is abandoned as TIMEOUT. "
            "0 waits indefinitely. Default 180"
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Report progress every N files; 0 disables",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 1

    with open(args.config) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    run_dirs = os.path.join(
        os.path.expandvars(config["OutputPath"]),
        config["RunName"],
    )

    if not os.path.isdir(run_dirs):
        print(
            f"ERROR: run directory does not exist: {run_dirs}",
            file=sys.stderr,
        )
        return 1

    try:
        n_elements = (
            args.n_elements
            if args.n_elements is not None
            else derive_n_elements(config, run_dirs)
        )
    except (FileNotFoundError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # The authoritative expected date list, straight from the producer.
    #
    # OVERPASS_IGNORE_MARKER, because the producer and the checker want
    # different things from the same function. build_date_list normally drops
    # the dates an existing marker already covers, which is right when the
    # point is to avoid redoing work -- but this is verifying that work, and a
    # complete face would otherwise return no dates at all and check nothing.
    #
    # Output is redirected too: build_date_list logs to stdout, and stdout here
    # carries the paths to delete and nothing else. A stray line reaches
    # delete_files_from_list.sh, which rejects it and refuses the whole list.
    previous_ignore = os.environ.get("OVERPASS_IGNORE_MARKER")
    os.environ["OVERPASS_IGNORE_MARKER"] = "1"

    try:
        with contextlib.redirect_stdout(sys.stderr):
            _, _, date_list, shared_end_date = build_date_list(config)
    finally:
        if previous_ignore is None:
            del os.environ["OVERPASS_IGNORE_MARKER"]
        else:
            os.environ["OVERPASS_IGNORE_MARKER"] = previous_ignore

    if not date_list:
        print(
            f"ERROR: no local dates to check. Shared end date "
            f"{shared_end_date} leaves the window empty, which means the "
            f"simulations have not advanced past {config['StartDate']}.",
            file=sys.stderr,
        )
        return 1

    expected = build_expected_files(
        config,
        n_elements,
        run_dirs,
        date_list,
    )

    existing = [
        file_path
        for file_path, _ in expected
        if os.path.isfile(file_path)
    ]

    if not existing:
        print(
            "NOTE: no overpass .nc4 files to check; skipping.",
            file=sys.stderr,
        )
        return 0

    print(
        f"Face:               {config['RunName']}"
        f"\nRun directory:      {run_dirs}"
        f"\nState vector size:  {n_elements}"
        f"\nShared end date:    {shared_end_date} (exclusive)"
        f"\nLocal dates:        {len(date_list)}"
        f"  ({date_list[0]} .. {date_list[-1]})"
        f"\nExpected files:     {len(expected)}"
        "\nNOTE: the StartDate-1 file is expected to contain NaN and is "
        "reported as incomplete on purpose; delete and regenerate it.",
        file=sys.stderr,
        flush=True,
    )

    counts = {
        "COMPLETE": 0,
        "MISSING": 0,
        "MISSING_VARIABLE": 0,
        "EMPTY_VARIABLE": 0,
        "NAN": 0,
        "NAN_FIRST_DATE": 0,
        "READ_ERROR": 0,
        "TIMEOUT": 0,
        "CRASH": 0,
    }

    inspected = 0

    # StartDate-1: the one date whose NaN is expected rather than suspicious.
    first_date_tag = (
        f".{date_list[0]}_{config['OverpassTime'].replace(':', '')}."
        if date_list
        else None
    )

    # Whether to flag it turns on one question: has this face been processed?
    #
    # No marker means it has not, so flagging that file is how it gets built
    # fresh -- which is the point of a one-off sweep before processing.
    #
    # A marker means the overpass stage completed and wrote it deliberately.
    # Flagging it then would delete a correct file and, because deleting
    # overpass output retires the marker, reprocess the whole window. Worse on
    # a face that has since been pruned: the OutputDir needed to rebuild it is
    # gone by design, so the deletion is not recoverable.
    #
    # --allow-nan-first-date forces the exemption on regardless.
    # --flag-nan-first-date wins over both: a marker turns the exemption on by
    # itself, so without it a clean rebuild cannot be asked for.
    overpass_marker = read_stage_marker(
        run_dirs,
        "overpass",
        str(config["StartDate"]),
    )

    exempt_first_date = (
        args.allow_nan_first_date or overpass_marker is not None
    ) and not args.flag_nan_first_date

    if first_date_tag:
        if exempt_first_date:
            reason = (
                "--allow-nan-first-date"
                if args.allow_nan_first_date
                else f"overpass marker S{overpass_marker} present"
            )

            print(
                f"StartDate-1 ({date_list[0]}): NaN expected, not flagged"
                f"  [{reason}]",
                file=sys.stderr,
            )

        else:
            print(
                f"StartDate-1 ({date_list[0]}): no overpass marker, so it will"
                f" be flagged for rebuilding",
                file=sys.stderr,
            )

    inspected = 0

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}

        for file_path, expected_vars in expected:
            # A missing file needs no read, and cannot be deleted, so it is
            # resolved here rather than in a worker.
            if not os.path.isfile(file_path):
                counts["MISSING"] += 1

                print(
                    f"MISSING: {file_path}",
                    file=sys.stderr,
                    flush=True,
                )

                continue

            futures[
                pool.submit(
                    inspect_isolated,
                    file_path,
                    expected_vars,
                    args.per_file_timeout,
                )
            ] = file_path

        for future in cf.as_completed(futures):
            file_path = futures[future]
            status, detail = future.result()

            # NaN in the StartDate-1 file is how that date is supposed to look:
            # only the part covered by UTC StartDate can be filled. Counted
            # under its own name so it stays visible rather than being quietly
            # folded into COMPLETE.
            if (
                exempt_first_date
                and status == "NAN"
                and first_date_tag
                and first_date_tag in os.path.basename(file_path)
            ):
                status = "NAN_FIRST_DATE"

            counts[status] = counts.get(status, 0) + 1
            inspected += 1

            if status not in ("COMPLETE", "NAN_FIRST_DATE"):
                # stdout is the deletion list, and nothing else.
                print(file_path)
                sys.stdout.flush()

                print(
                    f"{status}"
                    + (f" [{detail}]" if detail else "")
                    + f": {file_path}",
                    file=sys.stderr,
                    flush=True,
                )

            if (
                args.progress_every > 0
                and inspected % args.progress_every == 0
            ):
                print(
                    f"Inspected {inspected}/{len(futures)}",
                    file=sys.stderr,
                    flush=True,
                )

    # Exactly the statuses the loop above prints, so the count and the list
    # cannot disagree.
    printed = sum(
        count
        for status, count in counts.items()
        if status not in ("COMPLETE", "NAN_FIRST_DATE")
    )

    # A missing file is not deleted, but it does mean work is outstanding.
    incomplete = printed - counts["MISSING"]

    print(
        "\nSummary:"
        f"\n  Expected files:        {len(expected)}"
        f"\n  Complete:              {counts['COMPLETE']}"
        f"\n  Missing (will be made) {counts['MISSING']}"
        f"\n  Missing variables:     {counts['MISSING_VARIABLE']}"
        f"\n  Empty variables:       {counts['EMPTY_VARIABLE']}"
        f"\n  Containing NaN:        {counts['NAN']}"
        f"\n  NaN, StartDate-1:      {counts['NAN_FIRST_DATE']} "
        "(expected, not flagged)"
        f"\n  Read errors:           {counts['READ_ERROR']}"
        f"\n  Read timeouts:         {counts['TIMEOUT']}"
        f"\n  Reader crashes:        {counts['CRASH']}"
        f"\n  Printed for deletion:  {incomplete}"
        f"\n  (MISSING is not printed: there is no file to delete)",
        file=sys.stderr,
        flush=True,
    )

    if incomplete or counts["MISSING"]:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())