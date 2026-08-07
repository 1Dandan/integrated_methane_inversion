#!/usr/bin/env python3
"""Check that inversion/data_converted pickles are readable and well formed.

These were not always written atomically. A file produced before
save_obj_atomic() was adopted, by a run that was interrupted while writing,
is truncated: it exists, it has a plausible size, and nothing notices until
something tries to load it.

That matters because prune_outputdir.py's fourth precondition tests only that
the pickle is *present*. A truncated one satisfies it, and the OutputDir the
inversion would need to rebuild it is then deleted. This closes that gap.

Two levels, both applied to every file:

  structure   pickletools.genops walks the opcode stream without building any
              object or running any code. A truncated file fails here -- the
              stream ends before STOP -- and it costs almost nothing.
  contents    the file is then unpickled and checked: a dict carrying obs_GC
              as an (n, 5) array, GC_index of matching length, and K, when
              present, with one row per observation.

With a manifest to compare against, coverage is checked too: every granule
recorded as written or cached must have a loadable pickle, and one recorded
as no_valid_obs must have none, because the operator found nothing to save.

Usage:
    check_data_converted.py CONFIG [options] > corrupt.txt

Exit codes:
    0  every pickle loaded and checked out
    1  usage or configuration error
    2  at least one pickle is corrupt, or the manifest disagrees
"""

import argparse
import concurrent.futures as cf
import json
import os
import pickle
import pickletools
import sys

import numpy as np
import yaml


def check_stream(path):
    """Walk the pickle opcodes without building anything. Returns a problem or None."""
    try:
        with open(path, "rb") as handle:
            for _ in pickletools.genops(handle):
                pass
    except Exception as error:
        # "pickle exhausted before seeing STOP" is what truncation looks like.
        return f"malformed pickle stream: {error}"

    return None


def check_contents(path, expect_obs_gc):
    """Unpickle and verify the shape of what comes back.

    What the operator writes depends on the configuration, so the only key
    always present is GC_index. obs_GC is written unless DisableRun0000, and
    K (or K_noEmis) only when the Jacobian is being built -- so their absence
    is checked against the config rather than assumed.

    Returns a list of problems.
    """
    try:
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
    except Exception as error:
        return [f"unpicklable: {type(error).__name__}: {error}"]

    if not isinstance(obj, dict):
        return [f"expected a dict, got {type(obj).__name__}"]

    problems = []

    if "GC_index" not in obj:
        return ["missing 'GC_index'"]

    index = np.asarray(obj["GC_index"])

    if index.ndim != 1:
        problems.append(f"GC_index has shape {index.shape}, expected 1-D")
        return problems

    n_obs = index.shape[0]

    if n_obs == 0:
        problems.append("GC_index is empty; the file records no gridcell")

    if expect_obs_gc and "obs_GC" not in obj:
        problems.append("missing 'obs_GC' (DisableRun0000 is false, so it is written)")

    if "obs_GC" in obj:
        obs = np.asarray(obj["obs_GC"])

        if obs.ndim != 2 or obs.shape[1] < 5:
            problems.append(f"obs_GC has shape {obs.shape}, expected (n, 5)")
        elif obs.shape[0] != n_obs:
            problems.append(
                f"obs_GC rows {obs.shape[0]} != GC_index length {n_obs}"
            )
        # Columns 0 and 1 are the TROPOMI and GEOS-Chem columns; all-NaN there
        # means the file carries nothing usable, whatever its length.
        elif obs.shape[0] and (
            np.all(np.isnan(obs[:, 0])) or np.all(np.isnan(obs[:, 1]))
        ):
            problems.append("obs_GC columns 0/1 are entirely NaN")

    # K is initialised to NaN and filled per observation, so NaN in it is
    # ordinary. Only its shape is worth asserting.
    for key in ("K", "K_noEmis"):
        if key in obj:
            jac = np.asarray(obj[key])

            if jac.ndim != 2 or jac.shape[0] != n_obs:
                problems.append(
                    f"{key} has shape {jac.shape}, expected ({n_obs}, n_elements)"
                )

    return problems


def inspect(path, expect_obs_gc):
    """Both levels. Returns (path, problems)."""
    if os.path.getsize(path) == 0:
        return path, ["zero length"]

    stream_problem = check_stream(path)

    if stream_problem:
        return path, [stream_problem]

    return path, check_contents(path, expect_obs_gc)


def check_against_manifest(converted_dir, manifest_path, present):
    """Compare what is on disk with what the manifest says should be."""
    problems = []

    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        return [f"unreadable manifest: {error}"]

    for entry in manifest.get("granules", []):
        granule = entry.get("granule", "")
        status = entry.get("status", "")
        name = f"{granule}_GCtoTROPOMI.pkl"

        if status in ("written", "cached"):
            if name not in present:
                problems.append(f"{status} in manifest but absent: {name}")

        elif status == "no_valid_obs":
            if name in present:
                problems.append(
                    f"no_valid_obs in manifest but a pickle exists: {name}"
                )

    return problems


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Check inversion/data_converted pickles. Corrupt paths go to "
            "stdout; the summary goes to stderr."
        )
    )
    parser.add_argument("config", help="Path to the face's IMI config yaml")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel readers; default 4. Each holds one pickle in memory.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help=(
            "Remove corrupt pickles so the next inversion regenerates them. "
            "Safe only while the OutputDir those granules need is still "
            "present -- after a prune it is not."
        ),
    )
    parser.add_argument(
        "--skip-manifest", action="store_true",
        help="Check the files only, not their agreement with the manifest",
    )
    parser.add_argument(
        "--progress-every", type=int, default=200,
        help="Report progress every N files; 0 disables",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.workers < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 1

    with open(args.config) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    run_dirs = os.path.join(
        os.path.expandvars(config["OutputPath"]), config["RunName"]
    )
    converted_dir = os.path.join(run_dirs, "inversion", "data_converted")

    if not os.path.isdir(converted_dir):
        print(f"ERROR: no data_converted directory: {converted_dir}",
              file=sys.stderr)
        return 1

    names = sorted(f for f in os.listdir(converted_dir) if f.endswith(".pkl"))

    # obs_GC is written only when a base run exists to compare against.
    expect_obs_gc = not config.get("DisableRun0000", False)

    print(
        f"Face:           {config['RunName']}"
        f"\ndata_converted: {converted_dir}"
        f"\nPickles:        {len(names)}"
        f"\nExpect obs_GC:  {expect_obs_gc}"
        f"  (DisableRun0000={config.get('DisableRun0000', False)})",
        file=sys.stderr, flush=True,
    )

    if not names:
        print("ERROR: no .pkl files to check", file=sys.stderr)
        return 1

    corrupt = []
    checked = 0

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(inspect, os.path.join(converted_dir, name), expect_obs_gc)
            for name in names
        ]

        for future in cf.as_completed(futures):
            path, problems = future.result()
            checked += 1

            if problems:
                corrupt.append(path)
                print(path)
                sys.stdout.flush()
                print(f"CORRUPT  {os.path.basename(path)}", file=sys.stderr)
                for problem in problems:
                    print(f"    {problem}", file=sys.stderr)

            if args.progress_every > 0 and checked % args.progress_every == 0:
                print(f"  checked {checked}/{len(names)}, corrupt {len(corrupt)}",
                      file=sys.stderr, flush=True)

    manifest_problems = []

    if not args.skip_manifest:
        manifest_path = os.path.join(
            run_dirs, "inversion", "data_converted_manifest.json"
        )

        if os.path.isfile(manifest_path):
            present = set(names) - {os.path.basename(p) for p in corrupt}
            manifest_problems = check_against_manifest(
                converted_dir, manifest_path, present
            )

            for problem in manifest_problems:
                print(f"MANIFEST  {problem}", file=sys.stderr)
        else:
            print(
                "NOTE: no data_converted_manifest.json; file checks only. "
                "Rerun the inversion to write one.",
                file=sys.stderr,
            )

    if args.delete and corrupt:
        for path in corrupt:
            os.remove(path)
        print(f"\nDeleted {len(corrupt)} corrupt pickle(s).", file=sys.stderr)

    print(
        f"\nSummary:"
        f"\n  checked            {checked}"
        f"\n  corrupt            {len(corrupt)}"
        f"\n  manifest problems  {len(manifest_problems)}",
        file=sys.stderr,
    )

    if corrupt and not args.delete:
        print(
            "\nCorrupt paths listed on stdout. Delete them and rerun the"
            "\ninversion to regenerate -- but only while the OutputDir those"
            "\ngranules need is still present. After a prune it is not.",
            file=sys.stderr,
        )

    return 2 if (corrupt or manifest_problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
