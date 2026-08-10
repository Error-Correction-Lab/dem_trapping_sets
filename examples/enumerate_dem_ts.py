#!/usr/bin/env python3
"""
Enumerate trapping sets in the circuit-level detector error model of a
bivariate-bicycle (BB) code.

Examples
--------
[[144,12,12]] code (default):
    python examples/enumerate_dem_ts.py \
        --output-dir results/bb144 \
        --amax 5 \
        --bmax 5

[[108,8,10]] code:
    python examples/enumerate_dem_ts.py \
        --code 108 \
        --output-dir results/bb108 \
        --amax 5 \
        --bmax 5

Use --etsl to include the optional ETS-with-leaves stage.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CPP_DIR = REPO_ROOT / "cpp" / "include"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(CPP_DIR))

import include_py.codes_q as codes
import include_py.build_decoding_matrices_layered as bdm
import dpl_search


PHYSICAL_ERROR_RATE = 1e-3
GIRTH = 4


# Each entry contains:
#   (l, m, A_x, A_y, B_x, B_y, default_rounds, label)
BB_CODES = {
    "72":  (6,  6,  [3], [1, 2], [1, 2], [3],  6,  "[[72,12,6]]"),
    "90":  (15, 3,  [9], [1, 2], [2, 7], [0], 10,  "[[90,8,10]]"),
    "108": (9,  6,  [3], [1, 2], [1, 2], [3], 10,  "[[108,8,10]]"),
    "144": (12, 6,  [3], [1, 2], [1, 2], [3], 12,  "[[144,12,12]]"),
    "288": (12, 12, [3], [2, 7], [1, 2], [3], 18,  "[[288,12,18]]"),
}

CODE_ALIASES = {
    "bb72": "72",
    "bb90": "90",
    "bb108": "108",
    "bb144": "144",
    "gross": "144",
    "bb288": "288",
    "two-gross": "288",
    "twogross": "288",
}


def normalize_code_name(code_name: str) -> str:
    key = code_name.strip().lower()
    key = CODE_ALIASES.get(key, key)

    if key not in BB_CODES:
        raise ValueError(
            "Unknown --code. Use one of: 72, 90, 108, 144/gross, "
            "288/two-gross."
        )

    return key


def build_code(code_name: str):
    """Return the selected BB code, circuit data, default rounds, and label."""

    key = normalize_code_name(code_name)
    l, m, Ax, Ay, Bx, By, default_rounds, label = BB_CODES[key]

    code, A_list, B_list = codes.create_bivariate_bicycle_codes(
        l, m, Ax, Ay, Bx, By
    )

    return code, A_list, B_list, default_rounds, label


def theorem2_bmax_by_a(
    amax: int,
    bmax: int,
    degrees: list[int],
    g: int,
) -> list[int]:
    """Compute the class-dependent bounds b_max^a used by Algorithm 1."""

    cutoff = amax + bmax
    eta_candidates = [d for d in degrees if d < cutoff]
    if not eta_candidates:
        raise ValueError(
            "Cannot construct bmax_by_a: no variable-node degree is "
            "strictly smaller than amax + bmax."
        )

    eta = max(eta_candidates)
    g2 = g // 2

    bounds = [0] * (amax + 1)
    bounds[amax] = bmax

    for a in range(amax - 1, g2 - 1, -1):
        y_candidates = [d for d in degrees if d <= a]
        if not y_candidates:
            raise ValueError(
                f"Cannot construct bmax_by_a at a={a}: "
                "no variable-node degree is <= a."
            )

        y = max(y_candidates)
        z_candidates = [d for d in degrees if a < d < cutoff]

        decrease = (
            max(y, 2 * a - min(z_candidates))
            if z_candidates
            else y
        )

        bounds[a] = min(
            bounds[a + 1] + decrease,
            a * (eta - 2),
        )

    return bounds


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' is not empty. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def build_detector_matrix(
    code_name: str,
    rounds: int | None,
) -> tuple[sp.csr_matrix, str, int]:
    """Construct the full circuit-level detector matrix."""

    code, A_list, B_list, default_rounds, label = build_code(code_name)
    num_rounds = default_rounds if rounds is None else rounds

    circuit = bdm.build_circuit(
        code,
        A_list,
        B_list,
        p=PHYSICAL_ERROR_RATE,
        num_repeat=num_rounds,
    )

    dem = circuit.detector_error_model()
    H, _, _ = bdm.dem_to_check_matrices(dem)

    if sp.issparse(H):
        H = H.astype(np.int8).tocsr()
    else:
        H = sp.csr_matrix(np.asarray(H, dtype=np.int8))

    return H, label, num_rounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate trapping sets in the circuit-level detector matrix "
            "of a bivariate-bicycle code."
        )
    )

    parser.add_argument(
        "--code",
        default="144",
        help=(
            "BB code: 72, 90, 108, 144/gross, or 288/two-gross "
            "(default: 144)."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=(
            "Number of syndrome-extraction rounds. If omitted, use the "
            "default associated with the selected code."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which the support CSV files are written.",
    )
    parser.add_argument(
        "--amax",
        type=int,
        default=5,
        help="Maximum trapping-set size a (default: 5).",
    )
    parser.add_argument(
        "--bmax",
        type=int,
        default=5,
        help="Maximum target value of b (default: 5).",
    )
    parser.add_argument(
        "--etsl",
        action="store_true",
        help="Also enumerate elementary trapping sets with leaves.",
    )
    parser.add_argument(
        "--singleton-etsl-seeds",
        action="store_true",
        help=(
            "Initialize the ETSL stage from singleton variable nodes as well "
            "as from the LETSs found by DPL. Requires --etsl."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing nonempty output directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress messages from the C++ enumerator.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.amax <= 0:
        raise ValueError("--amax must be positive")
    if args.bmax < 0:
        raise ValueError("--bmax must be nonnegative")
    if args.rounds is not None and args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.singleton_etsl_seeds and not args.etsl:
        raise ValueError("--singleton-etsl-seeds requires --etsl")

    prepare_output_dir(args.output_dir, args.overwrite)

    H, code_label, num_rounds = build_detector_matrix(
        args.code,
        args.rounds,
    )
    M, N = H.shape

    degrees = sorted(
        np.unique(np.asarray(H.sum(axis=0)).ravel().astype(int)).tolist()
    )

    bmax_by_a = theorem2_bmax_by_a(
        amax=args.amax,
        bmax=args.bmax,
        degrees=degrees,
        g=GIRTH,
    )

    print(f"Code: {code_label}")
    print(f"Syndrome-extraction rounds: {num_rounds}")
    print(f"Detector matrix: {M} x {N}")
    print(f"Variable-node degrees: {degrees}")
    print(f"a_max={args.amax}, b_max={args.bmax}, g={GIRTH}")
    print(f"bmax_by_a={bmax_by_a}")
    print(f"Output directory: {args.output_dir}")
    print(f"ETSL: {args.etsl}")

    start = time.perf_counter()

    result = dpl_search.enumerate(
        M,
        N,
        H.indptr.astype(np.int32, copy=False),
        H.indices.astype(np.int32, copy=False),
        g=GIRTH,
        amax=args.amax,
        bmax_by_a=bmax_by_a,
        output_dir=str(args.output_dir),
        allowed_variable_degrees=degrees,
        include_ets_with_leaves=args.etsl,
        include_singleton_etsl_seeds=args.singleton_etsl_seeds,
        verbose=not args.quiet,
    )

    elapsed = time.perf_counter() - start
    counts = np.asarray(result["counts"], dtype=np.uint64)

    print("\nEnumerated supports:")
    for a in range(args.amax + 1):
        for b in range(min(args.bmax + 1, counts.shape[0])):
            count = int(counts[b, a])
            if count:
                print(f"  ({a},{b}): {count}")

    print(f"\nWritten supports: {result['written_supports']}")
    print(f"Elapsed time: {elapsed:.3f} s")


if __name__ == "__main__":
    main()