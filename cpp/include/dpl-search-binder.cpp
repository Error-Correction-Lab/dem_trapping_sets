// dpl-search-binder.cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "tannergraph_parallelized_csr.hpp"
#include "dpl-search.hpp"

namespace py = pybind11;

namespace {

constexpr std::size_t COMPILED_AMAX = 32;

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

void validate_inputs(
    int M,
    int N,
    int g,
    int amax,
    const std::vector<int>& bmax_by_a,
    const std::string& output_dir,
    const std::vector<int>& allowed_variable_degrees
) {
    if (M <= 0 || N <= 0) {
        throw std::runtime_error("M and N must be positive");
    }

    if (g <= 0 || g > 254 || (g % 2) != 0) {
        throw std::runtime_error(
            "g must be a positive even integer in [2, 254]"
        );
    }

    if (amax <= 0 || amax > static_cast<int>(COMPILED_AMAX)) {
        throw std::runtime_error(
            "amax must be in [1, " + std::to_string(COMPILED_AMAX) + "]"
        );
    }

    if (g / 2 > amax) {
        throw std::runtime_error("g/2 cannot exceed amax");
    }

    if (static_cast<int>(bmax_by_a.size()) <= amax) {
        throw std::runtime_error(
            "bmax_by_a must have length at least amax + 1"
        );
    }

    for (int a = g / 2; a <= amax; ++a) {
        const int b = bmax_by_a[static_cast<std::size_t>(a)];
        if (b < 0 || b > 65535) {
            throw std::runtime_error(
                "bmax_by_a entries must be in [0, 65535]"
            );
        }
    }

    if (output_dir.empty()) {
        throw std::runtime_error("output_dir cannot be empty");
    }

    for (int d : allowed_variable_degrees) {
        if (d <= 0 || d > 255) {
            throw std::runtime_error(
                "allowed_variable_degrees entries must be in [1, 255]"
            );
        }
    }
}

py::dict enumerate_impl(
    int M,
    int N,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& indptr_arr,
    const py::array_t<int, py::array::c_style | py::array::forcecast>& indices_arr,
    int g,
    int amax,
    const std::vector<int>& bmax_by_a,
    const std::string& output_dir,
    const std::vector<int>& allowed_variable_degrees,
    bool include_ets_with_leaves,
    bool include_singleton_etsl_seeds,
    bool verbose
) {
    validate_inputs(
        M,
        N,
        g,
        amax,
        bmax_by_a,
        output_dir,
        allowed_variable_degrees
    );

    std::vector<int> indptr = array_to_int_vector(indptr_arr, "indptr");
    std::vector<int> indices = array_to_int_vector(indices_arr, "indices");

    TannerTopology topology(M, N, indptr, indices);

    const int b_target_max =
        bmax_by_a[static_cast<std::size_t>(amax)];

    int b_aux_max = 0;
    for (int a = g / 2; a <= amax; ++a) {
        b_aux_max = std::max(
            b_aux_max,
            bmax_by_a[static_cast<std::size_t>(a)]
        );
    }

    ts_irreg23::EnumeratorParams params;
    params.g = static_cast<std::uint8_t>(g);
    params.amax = static_cast<std::uint8_t>(amax);
    params.b_aux_max = static_cast<std::uint16_t>(b_aux_max);
    params.b_target_max = static_cast<std::uint16_t>(b_target_max);
    params.include_ets_with_leaves = include_ets_with_leaves;
    params.include_singleton_etsl_seeds = include_singleton_etsl_seeds;
    params.support_output_dir = output_dir;
    params.verbose = verbose;

    params.bmax_by_a.reserve(bmax_by_a.size());
    for (int b : bmax_by_a) {
        if (b < 0 || b > 65535) {
            throw std::runtime_error(
                "bmax_by_a entries must be in [0, 65535]"
            );
        }
        params.bmax_by_a.push_back(static_cast<std::uint16_t>(b));
    }

    params.allowed_variable_degrees.reserve(
        allowed_variable_degrees.size()
    );
    for (int d : allowed_variable_degrees) {
        params.allowed_variable_degrees.push_back(
            static_cast<std::uint8_t>(d)
        );
    }

    ts_irreg23::TS_enumerator_irreg23<COMPILED_AMAX> enumerator(
        topology,
        std::move(params)
    );

    ts_irreg23::EnumeratorResult<COMPILED_AMAX> result;
    {
        py::gil_scoped_release release;
        result = enumerator.run();
    }

    py::array_t<std::uint64_t> counts_arr(
        {b_target_max + 1, amax + 1}
    );
    auto counts = counts_arr.mutable_unchecked<2>();

    for (int b = 0; b <= b_target_max; ++b) {
        for (int a = 0; a <= amax; ++a) {
            std::uint64_t value = 0;

            if (b <= b_aux_max) {
                const std::size_t index =
                    static_cast<std::size_t>(b) *
                    (COMPILED_AMAX + 1) +
                    static_cast<std::size_t>(a);
                value = result.counts[index];
            }

            counts(b, a) = value;
        }
    }

    py::dict out;
    out["counts"] = counts_arr;
    out["written_supports"] = py::int_(result.written_supports);
    out["generated_candidates"] = py::int_(result.generated_candidates);
    out["accepted_intermediate"] = py::int_(result.accepted_intermediate);
    out["duplicate_candidates"] = py::int_(result.duplicate_candidates);
    out["rejected_by_size"] = py::int_(result.rejected_by_size);
    out["rejected_by_b"] = py::int_(result.rejected_by_b);
    out["rejected_non_elementary"] =
        py::int_(result.rejected_non_elementary);

    return out;
}

} // namespace

PYBIND11_MODULE(dpl_search, m) {
    m.doc() =
        "External-memory DPL trapping-set enumerator for irregular Tanner graphs";

    m.def(
        "enumerate",
        &enumerate_impl,
        py::arg("M"),
        py::arg("N"),
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("g"),
        py::arg("amax"),
        py::arg("bmax_by_a"),
        py::arg("output_dir"),
        py::arg("allowed_variable_degrees") = std::vector<int>{},
        py::arg("include_ets_with_leaves") = false,
        py::arg("include_singleton_etsl_seeds") = false,
        py::arg("verbose") = false
    );
}