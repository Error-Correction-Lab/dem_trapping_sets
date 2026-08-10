#!/usr/bin/env python3
"""
Run the irregular-code DPL/LETS search using the Algorithm-1 expansion table.

This script assumes that:
  1) the parallel external-memory DPL/ETSL header is installed; and
  2) the pybind module has been rebuilt with the parallel binder so that
     run_enum_irreg23 accepts bmax_by_a and the external_* arguments.

In Algorithm-1 mode Python does not pass path_m/lollipop_mc.  It only computes
and passes the Theorem-2 sequence bmax_by_a.  The C++ enumerator then computes
whether dot, pa_m, and lo_m^c are allowed for each parent class (a,b).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import stim  # noqa: F401  # imported because the local build stack uses stim objects


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
INCLUDE_DIR = ROOT / "include"

sys.path.insert(0, str(ROOT))         # needed for import include_py.codes_q
sys.path.insert(0, str(INCLUDE_DIR))  # needed for import dpl_search

import include_py.codes_q as codes
import include_py.build_decoding_matrices_layered as bdm
import dpl_search


# =============================================================================
# Parallel external-memory configuration
# =============================================================================
#
# The C++ enumerator keeps target support sizes sequential, but processes
# frontier-parent chunks concurrently within each target size. Candidate
# buffers are worker-local and the final external merge/deduplication remains
# global, so the output is independent of thread scheduling.
#
# Every setting can be overridden from the shell, for example:
#
#   DPL_NUM_THREADS=64 python run_dpl_search_parallel.py
#
# external_sort_run_records is the TOTAL in-memory candidate-record budget
# across all workers, not a per-thread budget.


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 0/1, false/true, no/yes, or off/on"
    )


CPU_COUNT = os.cpu_count() or 1

USE_EXTERNAL_MEMORY = env_bool("DPL_USE_EXTERNAL_MEMORY", True)

# Start below the 100-core machine limit. Benchmark 16, 32, and 64 before
# trying every core; external sorting and NVMe traffic normally limit scaling.
EXTERNAL_NUM_THREADS = env_int(
    "DPL_NUM_THREADS",
    min(32, CPU_COUNT),
    minimum=0,  # 0 asks C++ to use hardware_concurrency().
)

EXTERNAL_NUM_BUCKETS = env_int("DPL_NUM_BUCKETS", 512, minimum=1)
EXTERNAL_BATCH_SIZE = env_int("DPL_BATCH_SIZE", 4096, minimum=1)

# With the binder's AMAX=32 specialization, 4,000,000 pending records can use
# several hundred MB. The budget is shared across all workers.
EXTERNAL_SORT_RUN_RECORDS = env_int(
    "DPL_SORT_RUN_RECORDS",
    4_000_000,
    minimum=1,
)
EXTERNAL_MERGE_FAN_IN = env_int("DPL_MERGE_FAN_IN", 128, minimum=2)
EXTERNAL_PARENT_CHUNK_RECORDS = env_int(
    "DPL_PARENT_CHUNK_RECORDS",
    4096,
    minimum=1,
)
EXTERNAL_CYCLE_CACHE_MAX_BYTES = env_int(
    "DPL_CYCLE_CACHE_BYTES",
    512 * 1024 * 1024,
    minimum=0,
)
PRESERVE_EXTERNAL_WORK_DIR_ON_ERROR = env_bool(
    "DPL_PRESERVE_WORK_DIR_ON_ERROR",
    True,
)

if EXTERNAL_NUM_THREADS > CPU_COUNT:
    print(
        f"[warning] DPL_NUM_THREADS={EXTERNAL_NUM_THREADS} exceeds "
        f"os.cpu_count()={CPU_COUNT}; oversubscription may reduce performance."
    )


# =============================================================================
# Code choice
# =============================================================================

# [[72,12,6]]
# code, A_list, B_list = codes.create_bivariate_bicycle_codes(
#     6, 6, [3], [1, 2], [1, 2], [3]
# )
# num_rounds = 6

# [[90,8,10]]
# code, A_list, B_list = codes.create_bivariate_bicycle_codes(
#     15, 3, [9], [1, 2], [2, 7], [0]
# )
# num_rounds = 10

# [[108,8,10]]
# code, A_list, B_list = codes.create_bivariate_bicycle_codes(
#     9, 6, [3], [1, 2], [1, 2], [3]
# )
# num_rounds = 10

# [[144,12,12]]
code, A_list, B_list = codes.create_bivariate_bicycle_codes(
    12, 6, [3], [1, 2], [1, 2], [3]
)
num_rounds = 12

# [[288,12,18]]
# code, A_list, B_list = codes.create_bivariate_bicycle_codes(
#     12, 12, [3], [2, 7], [1, 2], [3]
# )
# num_rounds = 18


# =============================================================================
# Build circuit / DEM / detector matrix
# =============================================================================

real_circuit = bdm.build_circuit(
    code,
    A_list,
    B_list,
    p=1e-3,
    num_repeat=num_rounds,
)

dem = real_circuit.detector_error_model()
chk, obs_raw, PRIORS = bdm.dem_to_check_matrices(dem)

if sp.issparse(chk):
    chk = chk.astype(np.int8).tocsr()
else:
    chk = sp.csr_matrix(chk.astype(np.int8))

full_col_w = np.asarray(chk.sum(axis=0)).ravel().astype(int)

hook_cols = np.where(full_col_w > 3)[0]
data_and_ancilla_cols = np.where(full_col_w <= 3)[0]

print("Full chk shape:", chk.shape)
print(
    "Full column-degree distribution:",
    dict(zip(*np.unique(full_col_w, return_counts=True))),
)
print("num hook cols:", len(hook_cols))
print("num data/ancilla cols:", len(data_and_ancilla_cols))


# =============================================================================
# Matrix choice
# =============================================================================
#
# Old behavior:
#     RUN_FULL_DETECTOR = False
#     H = chk[:, data_and_ancilla_cols]
#
# Algorithm-1 irregular-degree behavior:
#     RUN_FULL_DETECTOR = True
#     H = chk
#
# Since the modified C++ search supports degrees 2,3,4,5,6, we can run on the
# full detector matrix.

RUN_FULL_DETECTOR = True

if RUN_FULL_DETECTOR:
    selected_cols = np.arange(chk.shape[1], dtype=np.int64)
    H = chk
    output_dir = "ets_supports_full_detector_algorithm1_full_parallel_ETSL"
else:
    selected_cols = data_and_ancilla_cols.astype(np.int64)
    H = chk[:, selected_cols].tocsr()
    output_dir = "ets_supports_chk_restricted_algorithm1_anchored_parallel_ETSL"

M, N = H.shape

col_deg = np.asarray(H.sum(axis=0)).ravel().astype(int)
print("Search H shape:", H.shape)
print(
    "Search column-degree distribution:",
    dict(zip(*np.unique(col_deg, return_counts=True))),
)

# This degree set is used both by the C++ enumerator and by Theorem 2.
allowed_degrees = [2, 3, 4, 5, 6]

bad = np.where(~np.isin(col_deg, allowed_degrees))[0]
if len(bad) > 0:
    raise RuntimeError(
        f"H has columns outside allowed degrees {allowed_degrees}. "
        f"First bad restricted indices: {bad[:20].tolist()}, "
        f"degrees: {col_deg[bad[:20]].tolist()}"
    )


# =============================================================================
# First-block / anchoring convention
# =============================================================================
#
# The binder expects limits in the column indexing of H, not necessarily in the
# original chk indexing.
#
# If H = chk, then the original and search indices coincide.
# If H = chk[:, selected_cols], then we must convert the original first-block
# cutoff into the number of selected columns that belong to the first block.
#
# IMPORTANT:
# Set N_COLS_FIRST_BLOCK_ORIGINAL if you know the exact number of original DEM
# columns in the first spatial/temporal block.
#
# If left as None, the script uses a conservative ordering-based approximation:
# it assumes repeated blocks are ordered consecutively and that all non-first
# blocks have approximately the same number of columns.

N_COLS_FIRST_BLOCK_ORIGINAL = 144 * 4 + 72 + 72*3 - 1

if N_COLS_FIRST_BLOCK_ORIGINAL is None:
    later_block_size_guess = chk.shape[1] // num_rounds
    N_COLS_FIRST_BLOCK_ORIGINAL = (
        chk.shape[1] - (num_rounds - 1) * later_block_size_guess
    )

    print(
        "[warning] N_COLS_FIRST_BLOCK_ORIGINAL was not set explicitly. "
        "Using ordering-based guess:"
    )
    print("  later_block_size_guess:", later_block_size_guess)
    print("  N_COLS_FIRST_BLOCK_ORIGINAL:", N_COLS_FIRST_BLOCK_ORIGINAL)

if N_COLS_FIRST_BLOCK_ORIGINAL <= 0:
    raise RuntimeError("N_COLS_FIRST_BLOCK_ORIGINAL must be positive.")

if N_COLS_FIRST_BLOCK_ORIGINAL > chk.shape[1]:
    raise RuntimeError(
        "N_COLS_FIRST_BLOCK_ORIGINAL exceeds number of columns in chk."
    )

if RUN_FULL_DETECTOR:
    first_block_limit_in_H = int(N_COLS_FIRST_BLOCK_ORIGINAL)
else:
    # selected_cols is sorted because it comes from np.where.
    first_block_limit_in_H = int(
        np.count_nonzero(selected_cols < N_COLS_FIRST_BLOCK_ORIGINAL)
    )

if first_block_limit_in_H <= 0:
    raise RuntimeError(
        "The selected H has no columns in the first block. "
        "Check selected_cols and N_COLS_FIRST_BLOCK_ORIGINAL."
    )

print("first_block_limit_in_H:", first_block_limit_in_H)
print(
    "first-block degree distribution in H:",
    dict(zip(*np.unique(col_deg[:first_block_limit_in_H], return_counts=True))),
)


# =============================================================================
# Theorem-2 bmax sequence and optional Algorithm-1 debug table
# =============================================================================

def theorem2_bmax_by_a(
    *,
    amax: int,
    bmax: int,
    degrees: Iterable[int],
    g: int,
) -> List[int]:
    """
    Compute the recursive Theorem-2 upper bounds bmax^a.

    The recursion is

        bmax^a = min{ bmax^{a+1} + max{y, 2a - z}, a(eta - 2) }

    where:
        y   = largest variable degree <= a,
        z   = smallest variable degree > a and < amax + bmax,
        eta = largest variable degree < amax + bmax.

    The returned list is indexed by a.  Entries below g/2 are left as 0.
    """
    if amax <= 0:
        raise ValueError("amax must be positive")
    if bmax < 0:
        raise ValueError("bmax must be nonnegative")
    if g <= 0 or g % 2 != 0:
        raise ValueError("g must be a positive even integer")
    if g // 2 > amax:
        raise ValueError("g/2 cannot exceed amax")

    degs = sorted({int(d) for d in degrees if int(d) > 0})
    if not degs:
        raise ValueError("degrees must contain at least one positive degree")

    cutoff = amax + bmax
    eta_candidates = [d for d in degs if d < cutoff]
    if not eta_candidates:
        raise ValueError("No variable degree is strictly smaller than amax + bmax")

    eta = max(eta_candidates)
    g2 = g // 2

    bseq = [0] * (amax + 1)
    bseq[amax] = int(bmax)

    for a in range(amax - 1, g2 - 1, -1):
        y_candidates = [d for d in degs if d <= a]
        z_candidates = [d for d in degs if a < d < cutoff]

        if not y_candidates:
            raise ValueError(
                f"Theorem-2 recursion undefined at a={a}: "
                "no variable degree is <= a."
            )

        y = max(y_candidates)

        # If z does not exist, the k>a dot-decrease case is absent, so only y
        # contributes to the largest possible b decrease.
        if z_candidates:
            z = min(z_candidates)
            max_dot_decrease = max(y, 2 * a - z)
        else:
            max_dot_decrease = y

        bseq[a] = min(
            bseq[a + 1] + max_dot_decrease,
            a * (eta - 2),
        )

    return bseq


def build_algorithm1_expansion_table_for_debug(
    *,
    amax: int,
    bmax_by_a: List[int],
    g: int,
) -> Dict[Tuple[int, int], List[Tuple]]:
    """
    Python mirror of Algorithm 1, used only for printing/debugging.

    The updated C++ enumerator computes this table implicitly from bmax_by_a;
    this table is not passed to the binder.
    """
    if len(bmax_by_a) <= amax:
        raise ValueError("bmax_by_a must have length at least amax + 1")
    if g <= 0 or g % 2 != 0:
        raise ValueError("g must be a positive even integer")

    g2 = g // 2
    table: Dict[Tuple[int, int], List[Tuple]] = {}

    for a in range(amax - 1, g2 - 1, -1):
        for b in range(1, bmax_by_a[a] + 1):
            ops: List[Tuple] = []

            # Algorithm 1 line 4: generic dot expansion for b >= 2.
            if b >= 2:
                ops.append(("dot",))

            m = 2
            while a + m <= amax:
                child_a = a + m

                # Algorithm 1 lines 8--10: path expansion pa_m.
                if 0 <= b - 2 <= bmax_by_a[child_a]:
                    ops.append(("pa", m))

                # Algorithm 1 lines 11--15: lollipop expansions lo_m^c.
                if m >= g2 and b - 1 <= bmax_by_a[child_a]:
                    for c in range(g2, m + 1):
                        ops.append(("lo", m, c))

                m += 1

            table[(a, b)] = ops

    return table


def format_expansion(op: Tuple) -> str:
    if op[0] == "dot":
        return "dot"
    if op[0] == "pa":
        return f"pa_{op[1]}"
    if op[0] == "lo":
        return f"lo_{op[1]}^{op[2]}"
    raise ValueError(f"Unknown expansion tuple: {op!r}")


def print_algorithm1_debug_table(
    *,
    amax: int,
    bmax_by_a: List[int],
    g: int,
) -> None:
    table = build_algorithm1_expansion_table_for_debug(
        amax=amax,
        bmax_by_a=bmax_by_a,
        g=g,
    )

    print("\nTheorem-2 bmax sequence:")
    for a in range(g // 2, amax + 1):
        print(f"  bmax^{a} = {bmax_by_a[a]}")

    print("\nAlgorithm-1 EX(a,b) table, Python debug copy only:")
    for a in range(g // 2, amax):
        print(f"\na = {a}, b <= {bmax_by_a[a]}")
        for b in range(1, bmax_by_a[a] + 1):
            ops = table[(a, b)]
            rhs = ", ".join(format_expansion(op) for op in ops) or "empty"
            print(f"  EX({a},{b}) = {rhs}")


# =============================================================================
# DPL / ETS parameters
# =============================================================================

# Search target.
g = 4
amax = 5
bmax = 5

# Algorithm 2 starts from simple cycles s_k for k = g/2, ..., amax.
Kmax = amax

bmax_by_a = theorem2_bmax_by_a(
    amax=amax,
    bmax=bmax,
    degrees=allowed_degrees,
    g=g,
)

# Scalar auxiliary bound still sizes some C++ arrays, so it must cover all
# intermediate bounds.
b_aux_max = max(bmax_by_a)

print("\nDPL / Algorithm-1 parameters:")
print("g:", g)
print("Kmax:", Kmax)
print("amax:", amax)
print("target bmax:", bmax)
print("bmax_by_a indexed by a:", bmax_by_a)
print("b_aux_max:", b_aux_max)
print("allowed_degrees:", allowed_degrees)

PRINT_DEBUG_EXPANSION_TABLE = True
if PRINT_DEBUG_EXPANSION_TABLE:
    print_algorithm1_debug_table(
        amax=amax,
        bmax_by_a=bmax_by_a,
        g=g,
    )


# =============================================================================
# Helpers
# =============================================================================

def clean_dir(dirname: str) -> None:
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname, exist_ok=True)


def load_supports_from_csv_dir(dirname: str) -> pd.DataFrame:
    """
    Loads streamed a_<a>_b_<b>.csv supports and returns a dataframe with:
        a, b, support_H, support_original, span_guess

    span_guess is computed only from the ordering-based block guess.
    """
    rows = []

    root = Path(dirname)
    if not root.exists():
        return pd.DataFrame()

    later_block_size_guess = chk.shape[1] // num_rounds

    def original_block_guess(col_original: int) -> int:
        if col_original < N_COLS_FIRST_BLOCK_ORIGINAL:
            return 0

        return 1 + (
            (col_original - N_COLS_FIRST_BLOCK_ORIGINAL)
            // later_block_size_guess
        )

    for path in sorted(root.glob("a_*_b_*.csv")):
        name = path.stem  # a_4_b_2
        parts = name.split("_")
        a = int(parts[1])
        b = int(parts[3])

        if path.stat().st_size == 0:
            continue

        df = pd.read_csv(path, header=None)

        for _, r in df.iterrows():
            support_H = [int(x) for x in r.dropna().tolist()]
            support_original = [int(selected_cols[x]) for x in support_H]

            blocks = [original_block_guess(x) for x in support_original]
            span_guess = max(blocks) - min(blocks) + 1

            rows.append({
                "a": a,
                "b": b,
                "support_H": support_H,
                "support_original": support_original,
                "span_guess": span_guess,
            })

    return pd.DataFrame(rows)


def ensure_parallel_binder() -> None:
    """Fail early if Python loaded an older, non-parallel extension."""
    doc = getattr(dpl_search.run_enum_irreg23, "__doc__", "") or ""
    required_keywords = (
        "bmax_by_a",
        "use_external_memory",
        "external_sort_run_records",
        "external_num_threads",
        "external_parent_chunk_records",
        "external_cycle_cache_max_bytes",
    )
    missing = [name for name in required_keywords if name not in doc]
    if missing:
        raise RuntimeError(
            "The loaded dpl_search.run_enum_irreg23 binding is missing: "
            + ", ".join(missing)
            + ". Rebuild the extension with ts_enum_binder_parallel.cpp "
              "and -pthread, then rerun this script."
        )


def run_search(include_ets_with_leaves: bool, output_dir: str):
    clean_dir(output_dir)
    ensure_parallel_binder()

    print("\nParallel external-memory configuration:")
    print("  os.cpu_count():", CPU_COUNT)
    print("  use_external_memory:", USE_EXTERNAL_MEMORY)
    print("  external_num_threads requested:", EXTERNAL_NUM_THREADS)
    print("  external_num_buckets:", EXTERNAL_NUM_BUCKETS)
    print("  external_batch_size:", EXTERNAL_BATCH_SIZE)
    print("  external_sort_run_records total:", EXTERNAL_SORT_RUN_RECORDS)
    print("  external_merge_fan_in:", EXTERNAL_MERGE_FAN_IN)
    print("  external_parent_chunk_records:", EXTERNAL_PARENT_CHUNK_RECORDS)
    print("  external_cycle_cache_max_bytes:", EXTERNAL_CYCLE_CACHE_MAX_BYTES)
    print(
        "  preserve_external_work_dir_on_error:",
        PRESERVE_EXTERNAL_WORK_DIR_ON_ERROR,
    )

    search_start = time.perf_counter()

    try:
        out = dpl_search.run_enum_irreg23(
            M,
            N,
            H.indptr.astype(np.int32, copy=False),
            H.indices.astype(np.int32, copy=False),

            K=Kmax,
            g=g,
            amax=amax,
            b_aux_max=b_aux_max,
            b_target_min=0,
            b_target_max=bmax,

            store_final_supports=False,

            # Algorithm-1 mode: leave the old global menus empty.
            # The C++ enumerator uses bmax_by_a to decide EX(a,b) internally.
            path_m=[],
            lollipop_mc=[],

            verbose=True,
            include_ets_with_leaves=include_ets_with_leaves,
            support_output_dir=output_dir,

            hash_max_load_factor=4.0,
            cache_cycles_for_lollipop=True,
            cache_lets_for_etsl=False,
            include_singleton_etsl_seeds=False,

            # 1) Seed only from first-block variables. This is the stronger
            #    memory-saving heuristic.
            seed_vn_limit=-1,

            # 2) Count/write/store only final supports containing at least one
            #    variable in the first block. This is the anchored-output filter.
            final_anchor_vn_limit=-1,

            # 3) Allow higher-degree detector columns.
            allowed_variable_degrees=allowed_degrees,

            # 4) Algorithm-1 argument.
            bmax_by_a=bmax_by_a,

            # 5) Parallel external-memory search. Target-a layers remain
            #    sequential; frontier-parent chunks within a layer run in
            #    parallel. Both DPL and dot_1^k candidates use bounded worker
            #    buffers followed by external merge/deduplication.
            use_external_memory=USE_EXTERNAL_MEMORY,
            external_num_buckets=EXTERNAL_NUM_BUCKETS,
            external_batch_size=EXTERNAL_BATCH_SIZE,
            external_sort_run_records=EXTERNAL_SORT_RUN_RECORDS,
            external_merge_fan_in=EXTERNAL_MERGE_FAN_IN,
            external_num_threads=EXTERNAL_NUM_THREADS,
            external_parent_chunk_records=EXTERNAL_PARENT_CHUNK_RECORDS,
            external_cycle_cache_max_bytes=EXTERNAL_CYCLE_CACHE_MAX_BYTES,
            preserve_external_work_dir_on_error=(
                PRESERVE_EXTERNAL_WORK_DIR_ON_ERROR
            ),
        )
    except TypeError as exc:
        raise RuntimeError(
            "run_enum_irreg23 rejected one or more Algorithm-1/parallel "
            "keywords. Rebuild dpl_search with the parallel binder and "
            "-pthread."
        ) from exc

    elapsed_seconds = time.perf_counter() - search_start
    out["python_elapsed_seconds"] = elapsed_seconds

    counts_raw = np.asarray(out["counts"], dtype=np.uint64)

    if counts_raw.ndim == 1:
        # Backward-compatible fallback. The updated binder returns a 2D table.
        counts = counts_raw.reshape((b_aux_max + 1, amax + 1))
    else:
        counts = counts_raw

    print("\nCounts over target b only:")
    print(pd.DataFrame(
        counts[:bmax + 1, :amax + 1],
        index=[f"b={b}" for b in range(min(bmax + 1, counts.shape[0]))],
        columns=[f"a={a}" for a in range(amax + 1)],
    ))

    print("\nRun statistics:")
    print("written_supports:", out.get("written_supports", None))
    print("support_output_dir:", out.get("support_output_dir", output_dir))
    print("generated_candidates:", out["generated_candidates"])
    print("accepted_intermediate:", out["accepted_intermediate"])
    print("duplicates:", out["duplicate_candidates"])
    print("rejected_by_b:", out["rejected_by_b"])
    print("rejected_non_elementary:", out["rejected_non_elementary"])
    print("elapsed_seconds:", f"{elapsed_seconds:.3f}")

    print("\nParallel parameter echo:")
    print("use_external_memory:", out.get("use_external_memory", None))
    print(
        "external_num_threads_requested:",
        out.get("external_num_threads_requested", EXTERNAL_NUM_THREADS),
    )
    print(
        "external_num_threads_effective:",
        out.get("external_num_threads_effective", None),
    )
    print(
        "parallel_candidate_generation_enabled:",
        out.get("parallel_candidate_generation_enabled", None),
    )
    print(
        "external_sort_run_records:",
        out.get("external_sort_run_records", EXTERNAL_SORT_RUN_RECORDS),
    )
    print(
        "external_parent_chunk_records:",
        out.get(
            "external_parent_chunk_records",
            EXTERNAL_PARENT_CHUNK_RECORDS,
        ),
    )
    print(
        "external_cycle_cache_max_bytes:",
        out.get(
            "external_cycle_cache_max_bytes",
            EXTERNAL_CYCLE_CACHE_MAX_BYTES,
        ),
    )

    print("\nAlgorithm/search parameter echo:")
    print("algorithm1_table_mode:", out.get("algorithm1_table_mode", None))
    print("bmax_by_a:", out.get("bmax_by_a", bmax_by_a))
    print("seed_vn_limit:", out.get("seed_vn_limit", first_block_limit_in_H))
    print("final_anchor_vn_limit:", out.get("final_anchor_vn_limit", first_block_limit_in_H))
    print("allowed_variable_degrees:", out.get("allowed_variable_degrees", allowed_degrees))

    return out, counts


# =============================================================================
# Run
# =============================================================================

# If include_ets_with_leaves=True, this includes dot_1^k leaf expansions.
# Algorithm 1 itself is the LETS/DPL expansion table; the ETS-with-leaves stage
# is the additional Section-VI/Algorithm-3-style layer in your C++ code.
out_ets, counts_ets = run_search(
    include_ets_with_leaves=True,
    output_dir=output_dir,
)


# =============================================================================
# Optional: load supports and estimate maximum observed span
# =============================================================================

supports_df = load_supports_from_csv_dir(output_dir)

if len(supports_df) > 0:
    print("\nSupport span summary using ordering-based block guess:")
    span_summary = (
        supports_df
        .groupby(["a", "b", "span_guess"])
        .size()
        .reset_index(name="count")
        .sort_values(["a", "b", "span_guess"])
    )
    print(span_summary)

    print("\nMaximum observed span by (a,b):")
    max_span = (
        supports_df
        .groupby(["a", "b"])["span_guess"]
        .max()
        .reset_index(name="max_span_guess")
        .sort_values(["a", "b"])
    )
    print(max_span)

    span_summary.to_csv(Path(output_dir) / "span_summary_guess.csv", index=False)
    max_span.to_csv(Path(output_dir) / "max_span_guess.csv", index=False)
    supports_df.to_pickle(Path(output_dir) / "supports_df.pkl")
else:
    print("\nNo streamed supports found; span summary skipped.")
