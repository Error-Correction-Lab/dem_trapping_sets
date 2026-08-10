// ts_enum_binder.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>

#include "tannergraph_parallelized_csr.hpp"
#include "dpl-search.hpp"

namespace py = pybind11;

namespace {

std::vector<int> array_to_int_vector(
    const py::array_t<int, py::array::c_style | py::array::forcecast>& arr,
    const std::string& name
) {
    py::buffer_info info = arr.request();

    if (info.ndim != 1) {
        throw std::runtime_error(name + " must be a 1D array");
    }

    const int* ptr = static_cast<const int*>(info.ptr);
    return std::vector<int>(ptr, ptr + info.shape[0]);
}

void validate_common_params(
    int K,
    int g,
    int amax,
    int b_aux_max,
    int b_target_min,
    int b_target_max,
    double hash_max_load_factor,
    int seed_vn_limit,
    int final_anchor_vn_limit,
    const std::vector<int>& allowed_variable_degrees,
    const std::vector<int>& bmax_by_a,
    bool use_external_memory,
    std::int64_t external_num_buckets,
    std::int64_t external_batch_size,
    std::int64_t external_sort_run_records,
    std::int64_t external_merge_fan_in,
    std::int64_t external_num_threads,
    std::int64_t external_parent_chunk_records,
    std::int64_t external_cycle_cache_max_bytes
) {
    if (K <= 0 || K > 255) {
        throw std::runtime_error("K must be in [1, 255]");
    }

    if (g <= 0 || g > 255 || (g % 2) != 0) {
        throw std::runtime_error("g must be a positive even integer in [2, 254]");
    }

    if (amax <= 0 || amax > 255) {
        throw std::runtime_error("amax must be in [1, 255]");
    }

    if (b_aux_max < 0 || b_aux_max > 65535) {
        throw std::runtime_error("b_aux_max must be in [0, 65535]");
    }

    if (b_target_min < 0 || b_target_min > 65535) {
        throw std::runtime_error("b_target_min must be in [0, 65535]");
    }

    if (b_target_max < 0 || b_target_max > 65535) {
        throw std::runtime_error("b_target_max must be in [0, 65535]");
    }

    if (b_target_min > b_target_max) {
        throw std::runtime_error("b_target_min cannot exceed b_target_max");
    }

    if (b_target_max > b_aux_max) {
        throw std::runtime_error("b_target_max cannot exceed b_aux_max");
    }

    if (!(hash_max_load_factor > 0.0)) {
        throw std::runtime_error("hash_max_load_factor must be positive");
    }

    if (seed_vn_limit < -1) {
        throw std::runtime_error("seed_vn_limit must be -1 or nonnegative");
    }

    if (final_anchor_vn_limit < -1) {
        throw std::runtime_error("final_anchor_vn_limit must be -1 or nonnegative");
    }

    for (int d : allowed_variable_degrees) {
        if (d <= 0 || d > 255) {
            throw std::runtime_error(
                "allowed_variable_degrees entries must be positive integers in [1, 255]"
            );
        }
    }

    if (external_num_buckets <= 0 ||
        external_num_buckets > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "external_num_buckets must be in [1, 2^32-1]"
        );
    }

    if (external_batch_size <= 0 ||
        external_batch_size > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "external_batch_size must be in [1, 2^32-1]"
        );
    }

    if (external_sort_run_records <= 0) {
        throw std::runtime_error(
            "external_sort_run_records must be positive"
        );
    }

    if (external_merge_fan_in < 2 ||
        external_merge_fan_in > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "external_merge_fan_in must be in [2, 2^32-1]"
        );
    }

    if (external_num_threads < 0 ||
        external_num_threads > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(
            "external_num_threads must be in [0, 2^32-1]; 0 selects hardware concurrency"
        );
    }

    if (external_parent_chunk_records <= 0) {
        throw std::runtime_error(
            "external_parent_chunk_records must be positive"
        );
    }

    if (external_cycle_cache_max_bytes < 0) {
        throw std::runtime_error(
            "external_cycle_cache_max_bytes must be nonnegative"
        );
    }

    if (!bmax_by_a.empty()) {
        if (static_cast<int>(bmax_by_a.size()) <= amax) {
            throw std::runtime_error(
                "bmax_by_a must have length at least amax + 1"
            );
        }

        if (K < amax) {
            throw std::runtime_error(
                "Algorithm-1 mode requires K >= amax"
            );
        }

        if (bmax_by_a[static_cast<size_t>(amax)] != b_target_max) {
            throw std::runtime_error(
                "bmax_by_a[amax] must equal b_target_max"
            );
        }

        const int g2 = g / 2;
        int max_bound = 0;

        for (int a = g2; a <= amax; ++a) {
            const int ba = bmax_by_a[static_cast<size_t>(a)];

            if (ba < 0 || ba > 65535) {
                throw std::runtime_error(
                    "bmax_by_a entries must be in [0, 65535]"
                );
            }

            if (ba > max_bound) {
                max_bound = ba;
            }
        }

        if (b_aux_max < max_bound) {
            throw std::runtime_error(
                "b_aux_max must be at least max_a bmax_by_a[a]"
            );
        }
    }
}

template <size_t AMAX>
py::dict run_enum_impl(
    int M,
    int N,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& indptr_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& indices_arr,
    int K,
    int g,
    int amax,
    int b_aux_max,
    int b_target_min,
    int b_target_max,
    bool store_final_supports,
    const std::vector<int>& path_m,
    const std::vector<std::pair<int, int>>& lollipop_mc,
    bool verbose,
    bool include_ets_with_leaves,
    const std::string& support_output_dir,
    double hash_max_load_factor,
    bool cache_cycles_for_lollipop,
    bool cache_lets_for_etsl,
    bool include_singleton_etsl_seeds,
    int seed_vn_limit,
    int final_anchor_vn_limit,
    const std::vector<int>& allowed_variable_degrees,
    const std::vector<int>& bmax_by_a,
    bool use_external_memory,
    std::int64_t external_num_buckets,
    std::int64_t external_batch_size,
    std::int64_t external_sort_run_records,
    std::int64_t external_merge_fan_in,
    std::int64_t external_num_threads,
    std::int64_t external_parent_chunk_records,
    std::int64_t external_cycle_cache_max_bytes,
    bool preserve_external_work_dir_on_error
) {
    validate_common_params(
        K,
        g,
        amax,
        b_aux_max,
        b_target_min,
        b_target_max,
        hash_max_load_factor,
        seed_vn_limit,
        final_anchor_vn_limit,
        allowed_variable_degrees,
        bmax_by_a,
        use_external_memory,
        external_num_buckets,
        external_batch_size,
        external_sort_run_records,
        external_merge_fan_in,
        external_num_threads,
        external_parent_chunk_records,
        external_cycle_cache_max_bytes
    );

    if (amax > static_cast<int>(AMAX)) {
        throw std::runtime_error("amax exceeds compiled AMAX");
    }

    if (seed_vn_limit > N) {
        throw std::runtime_error("seed_vn_limit cannot exceed N");
    }

    if (final_anchor_vn_limit > N) {
        throw std::runtime_error("final_anchor_vn_limit cannot exceed N");
    }

    if (!cache_cycles_for_lollipop && !lollipop_mc.empty()) {
        throw std::runtime_error(
            "cache_cycles_for_lollipop=false is incompatible with nonempty lollipop_mc"
        );
    }

    std::vector<int> indptr = array_to_int_vector(indptr_arr, "indptr");
    std::vector<int> indices = array_to_int_vector(indices_arr, "indices");

    TannerTopology T(M, N, indptr, indices);

    ts_irreg23::ExpansionMenu menu;

    for (int x : path_m) {
        if (x < 0 || x > 255) {
            throw std::runtime_error("path_m contains an invalid value");
        }

        menu.path_m.push_back(static_cast<uint8_t>(x));
    }

    for (const auto& mc : lollipop_mc) {
        const int m_path = mc.first;
        const int c_cycle = mc.second;

        if (m_path < 0 || m_path > 255 || c_cycle < 0 || c_cycle > 255) {
            throw std::runtime_error("lollipop_mc contains an invalid value");
        }

        menu.lollipop_mc.emplace_back(
            static_cast<uint8_t>(m_path),
            static_cast<uint8_t>(c_cycle)
        );
    }

    ts_irreg23::EnumeratorParams params;
    params.K = static_cast<uint8_t>(K);
    params.g = static_cast<uint8_t>(g);
    params.amax = static_cast<uint8_t>(amax);
    params.b_aux_max = static_cast<uint16_t>(b_aux_max);
    params.b_target_min = static_cast<uint16_t>(b_target_min);
    params.b_target_max = static_cast<uint16_t>(b_target_max);

    params.require_elementary = true;
    params.include_ets_with_leaves = include_ets_with_leaves;
    params.include_singleton_etsl_seeds = include_singleton_etsl_seeds;
    params.store_final_supports = store_final_supports;
    params.support_output_dir = support_output_dir;
    params.verbose = verbose;

    // Memory-control parameters added in the memory-efficient enumerator.
    params.hash_max_load_factor = static_cast<float>(hash_max_load_factor);
    params.cache_cycles_for_lollipop = cache_cycles_for_lollipop;
    params.cache_lets_for_etsl = cache_lets_for_etsl;

    // External-memory and parallel candidate-generation controls.
    params.use_external_memory = use_external_memory;
    params.external_num_buckets = static_cast<std::uint32_t>(external_num_buckets);
    params.external_batch_size = static_cast<std::uint32_t>(external_batch_size);
    params.external_sort_run_records =
        static_cast<std::uint64_t>(external_sort_run_records);
    params.external_merge_fan_in =
        static_cast<std::uint32_t>(external_merge_fan_in);
    params.external_num_threads =
        static_cast<std::uint32_t>(external_num_threads);
    params.external_parent_chunk_records =
        static_cast<std::uint64_t>(external_parent_chunk_records);
    params.external_cycle_cache_max_bytes =
        static_cast<std::uint64_t>(external_cycle_cache_max_bytes);
    params.preserve_external_work_dir_on_error =
        preserve_external_work_dir_on_error;

    // Anchored / irregular-degree search controls.
    params.seed_vn_limit = seed_vn_limit;
    params.final_anchor_vn_limit = final_anchor_vn_limit;
    params.allowed_variable_degrees.clear();
    params.allowed_variable_degrees.reserve(allowed_variable_degrees.size());

    for (int d : allowed_variable_degrees) {
        params.allowed_variable_degrees.push_back(static_cast<uint8_t>(d));
    }

    // Algorithm-1 expansion-table mode.
    // If this vector is nonempty, the C++ enumerator uses the implicit
    // class-dependent EX(a,b) table and enforces b <= bmax_by_a[a]
    // for every intermediate LETS candidate.
    params.bmax_by_a.clear();
    params.bmax_by_a.reserve(bmax_by_a.size());

    for (int ba : bmax_by_a) {
        params.bmax_by_a.push_back(static_cast<uint16_t>(ba));
    }

    ts_irreg23::TS_enumerator_irreg23<AMAX> enumerator(T, menu, params);

    ts_irreg23::EnumeratorResult<AMAX> result;
    {
        py::gil_scoped_release release;
        result = enumerator.run();
    }

    py::dict out;

    out["generated_candidates"] = py::int_(result.generated_candidates);
    out["accepted_intermediate"] = py::int_(result.accepted_intermediate);
    out["rejected_by_size"] = py::int_(result.rejected_by_size);
    out["rejected_by_b"] = py::int_(result.rejected_by_b);
    out["rejected_non_elementary"] = py::int_(result.rejected_non_elementary);
    out["duplicate_candidates"] = py::int_(result.duplicate_candidates);
    out["written_supports"] = py::int_(result.written_supports);

    out["include_ets_with_leaves"] = py::bool_(include_ets_with_leaves);
    out["support_output_dir"] = py::str(support_output_dir);

    out["hash_max_load_factor"] = py::float_(hash_max_load_factor);
    out["cache_cycles_for_lollipop"] = py::bool_(cache_cycles_for_lollipop);
    out["cache_lets_for_etsl"] = py::bool_(cache_lets_for_etsl);

    std::uint32_t effective_external_num_threads =
        static_cast<std::uint32_t>(external_num_threads);
    if (effective_external_num_threads == 0) {
        effective_external_num_threads = std::thread::hardware_concurrency();
        if (effective_external_num_threads == 0) {
            effective_external_num_threads = 1;
        }
    }

    out["use_external_memory"] = py::bool_(use_external_memory);
    out["external_num_buckets"] = py::int_(external_num_buckets);
    out["external_batch_size"] = py::int_(external_batch_size);
    out["external_sort_run_records"] = py::int_(external_sort_run_records);
    out["external_merge_fan_in"] = py::int_(external_merge_fan_in);
    out["external_num_threads_requested"] = py::int_(external_num_threads);
    out["external_num_threads_effective"] =
        py::int_(effective_external_num_threads);
    out["external_parent_chunk_records"] =
        py::int_(external_parent_chunk_records);
    out["external_cycle_cache_max_bytes"] =
        py::int_(external_cycle_cache_max_bytes);
    out["preserve_external_work_dir_on_error"] =
        py::bool_(preserve_external_work_dir_on_error);
    out["parallel_candidate_generation_enabled"] = py::bool_(
    use_external_memory &&
    effective_external_num_threads > 1
    );

    out["seed_vn_limit"] = py::int_(seed_vn_limit);
    out["final_anchor_vn_limit"] = py::int_(final_anchor_vn_limit);
    out["allowed_variable_degrees"] = py::cast(allowed_variable_degrees);
    out["bmax_by_a"] = py::cast(bmax_by_a);
    out["algorithm1_table_mode"] = py::bool_(!bmax_by_a.empty());

    py::list supports;
    for (const auto& S : result.final_supports) {
        py::list row;

        for (uint8_t i = 0; i < S.len; ++i) {
            row.append(S.v[i]);
        }

        supports.append(row);
    }

    out["final_supports"] = supports;

    py::array_t<uint64_t> counts_arr({b_target_max + 1, amax + 1});
    auto counts = counts_arr.mutable_unchecked<2>();

    for (int b = 0; b <= b_target_max; ++b) {
        for (int a = 0; a <= amax; ++a) {
            uint64_t val = 0;

            if (b <= b_aux_max) {
                const size_t idx =
                    static_cast<size_t>(b) * (AMAX + 1) +
                    static_cast<size_t>(a);

                val = result.counts[idx];
            }

            counts(b, a) = val;
        }
    }

    out["counts"] = counts_arr;

    return out;
}

py::dict run_enum_irreg23(
    int M,
    int N,
    py::array_t<int, py::array::c_style | py::array::forcecast> indptr,
    py::array_t<int, py::array::c_style | py::array::forcecast> indices,
    int K,
    int g,
    int amax,
    int b_aux_max,
    int b_target_min,
    int b_target_max,
    bool store_final_supports,
    const std::vector<int>& path_m,
    const std::vector<std::pair<int, int>>& lollipop_mc,
    bool verbose,
    bool include_ets_with_leaves,
    const std::string& support_output_dir,
    double hash_max_load_factor,
    bool cache_cycles_for_lollipop,
    bool cache_lets_for_etsl,
    bool include_singleton_etsl_seeds,
    int seed_vn_limit,
    int final_anchor_vn_limit,
    const std::vector<int>& allowed_variable_degrees,
    const std::vector<int>& bmax_by_a,
    bool use_external_memory,
    std::int64_t external_num_buckets,
    std::int64_t external_batch_size,
    std::int64_t external_sort_run_records,
    std::int64_t external_merge_fan_in,
    std::int64_t external_num_threads,
    std::int64_t external_parent_chunk_records,
    std::int64_t external_cycle_cache_max_bytes,
    bool preserve_external_work_dir_on_error
) {
    if (amax <= 32) {
        return run_enum_impl<32>(
            M,
            N,
            indptr,
            indices,
            K,
            g,
            amax,
            b_aux_max,
            b_target_min,
            b_target_max,
            store_final_supports,
            path_m,
            lollipop_mc,
            verbose,
            include_ets_with_leaves,
            support_output_dir,
            hash_max_load_factor,
            cache_cycles_for_lollipop,
            cache_lets_for_etsl,
            include_singleton_etsl_seeds,
            seed_vn_limit,
            final_anchor_vn_limit,
            allowed_variable_degrees,
            bmax_by_a,
            use_external_memory,
            external_num_buckets,
            external_batch_size,
            external_sort_run_records,
            external_merge_fan_in,
            external_num_threads,
            external_parent_chunk_records,
            external_cycle_cache_max_bytes,
            preserve_external_work_dir_on_error
        );
    }

    if (amax <= 64) {
        return run_enum_impl<64>(
            M,
            N,
            indptr,
            indices,
            K,
            g,
            amax,
            b_aux_max,
            b_target_min,
            b_target_max,
            store_final_supports,
            path_m,
            lollipop_mc,
            verbose,
            include_ets_with_leaves,
            support_output_dir,
            hash_max_load_factor,
            cache_cycles_for_lollipop,
            cache_lets_for_etsl,
            include_singleton_etsl_seeds,
            seed_vn_limit,
            final_anchor_vn_limit,
            allowed_variable_degrees,
            bmax_by_a,
            use_external_memory,
            external_num_buckets,
            external_batch_size,
            external_sort_run_records,
            external_merge_fan_in,
            external_num_threads,
            external_parent_chunk_records,
            external_cycle_cache_max_bytes,
            preserve_external_work_dir_on_error
        );
    }

    throw std::runtime_error(
        "amax too large for this test binder; compile a larger AMAX if needed"
    );
}

} // namespace

PYBIND11_MODULE(dpl_search, m) {
    m.doc() = "Parallel external-memory irregular-degree LETS/ETSL trapping-set enumerator";

    m.def(
        "run_enum_irreg23",
        &run_enum_irreg23,
        py::arg("M"),
        py::arg("N"),
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("K"),
        py::arg("g"),
        py::arg("amax"),
        py::arg("b_aux_max"),
        py::arg("b_target_min") = 0,
        py::arg("b_target_max") = 0,

        // Safer default for large searches. Use CSV streaming instead.
        py::arg("store_final_supports") = false,

        py::arg("path_m") = std::vector<int>{},
        py::arg("lollipop_mc") = std::vector<std::pair<int, int>>{},
        py::arg("verbose") = false,

        // Kept near the end to preserve old positional-call compatibility.
        py::arg("include_ets_with_leaves") = false,

        // If non-empty, stream final supports into a directory of
        // a_<a>_b_<b>.csv files.
        py::arg("support_output_dir") = std::string{},

        // Memory-control knobs.
        py::arg("hash_max_load_factor") = 2.0,
        py::arg("cache_cycles_for_lollipop") = true,
        py::arg("cache_lets_for_etsl") = false,
        py::arg("include_singleton_etsl_seeds") = true,

        // Anchored / irregular-degree knobs.
        // -1 means disabled for the two VN-limit parameters.
        // An empty degree list means the C++ enumerator allows every positive degree.
        py::arg("seed_vn_limit") = -1,
        py::arg("final_anchor_vn_limit") = -1,
        py::arg("allowed_variable_degrees") = std::vector<int>{2, 3},

        // Optional Algorithm-1 expansion-table mode.
        // Indexed by support size a. Empty vector preserves the old
        // global path_m / lollipop_mc behavior.
        py::arg("bmax_by_a") = std::vector<int>{},

        // External-memory and parallel candidate-generation knobs.
        // These are appended to preserve all existing positional calls.
        py::arg("use_external_memory") = true,
        py::arg("external_num_buckets") = std::int64_t{512},
        py::arg("external_batch_size") = std::int64_t{4096},

        // Total candidate-buffer record budget across all workers.
        py::arg("external_sort_run_records") = std::int64_t{1000000},
        py::arg("external_merge_fan_in") = std::int64_t{128},

        // 0 selects std::thread::hardware_concurrency(); 1 is serial.
        py::arg("external_num_threads") = std::int64_t{1},
        py::arg("external_parent_chunk_records") = std::int64_t{4096},

        // Shared read-only cycle-cache limit. Set to 0 to disable.
        py::arg("external_cycle_cache_max_bytes") =
            std::int64_t{512ll * 1024ll * 1024ll},

        py::arg("preserve_external_work_dir_on_error") = true
    );
}
