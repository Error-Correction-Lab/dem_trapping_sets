#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <queue>
#include <sstream>
#include <system_error>

#include "tannergraph_parallelized_csr.hpp"

namespace ts_irreg23 {

using Index = int;

// ============================================================================
// Compact support
// ============================================================================

template <size_t AMAX>
struct Support {
    static_assert(AMAX <= 255, "Support length is stored in uint8_t.");

    uint8_t len = 0;
    std::array<Index, AMAX> v{};

    bool operator==(const Support& other) const noexcept {
        if (len != other.len) return false;
        for (uint8_t i = 0; i < len; ++i) {
            if (v[i] != other.v[i]) return false;
        }
        return true;
    }

    bool contains(Index x) const noexcept {
        return std::binary_search(v.begin(), v.begin() + len, x);
    }
};

template <size_t AMAX>
struct SupportHash {
    size_t operator()(const Support<AMAX>& s) const noexcept {
        uint64_t h = 1469598103934665603ull;

        for (uint8_t i = 0; i < s.len; ++i) {
            uint64_t x = static_cast<uint64_t>(s.v[i]);
            h ^= x + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            h *= 1099511628211ull;
        }

        h ^= static_cast<uint64_t>(s.len);
        return static_cast<size_t>(h);
    }
};

template <size_t AMAX>
inline Support<AMAX> make_support_sorted_unique(std::vector<Index>& tmp) {
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

    if (tmp.size() > AMAX) {
        throw std::runtime_error("make_support_sorted_unique: support exceeds AMAX");
    }

    Support<AMAX> s;
    s.len = static_cast<uint8_t>(tmp.size());

    for (size_t i = 0; i < tmp.size(); ++i) {
        s.v[i] = tmp[i];
    }

    return s;
}

template <size_t AMAX>
inline Support<AMAX> insert_one_sorted(const Support<AMAX>& S, Index x) {
    if (S.len >= AMAX) {
        throw std::runtime_error("insert_one_sorted: support exceeds AMAX");
    }

    if (S.contains(x)) {
        throw std::runtime_error("insert_one_sorted: duplicate variable");
    }

    Support<AMAX> out;
    out.len = static_cast<uint8_t>(S.len + 1);

    uint8_t i = 0;
    uint8_t j = 0;
    bool inserted = false;

    while (i < S.len) {
        if (!inserted && x < S.v[i]) {
            out.v[j++] = x;
            inserted = true;
        }

        out.v[j++] = S.v[i++];
    }

    if (!inserted) {
        out.v[j++] = x;
    }

    return out;
}

// ============================================================================
// Induced subgraph statistics
// ============================================================================

struct StructureStats {
    uint16_t b = 0;
    bool elementary = true;
};

struct EvalScratch {
    std::vector<uint8_t> deg;
    std::vector<Index> touched;

    explicit EvalScratch(int M)
        : deg(static_cast<size_t>(M), 0)
    {
        touched.reserve(256);
    }

    void reset() {
        for (Index c : touched) {
            deg[static_cast<size_t>(c)] = 0;
        }
        touched.clear();
    }
};

template <size_t AMAX>
inline StructureStats induced_stats(
    const TannerTopology& T,
    const Support<AMAX>& S,
    EvalScratch& scratch
) {
    scratch.reset();

    for (uint8_t i = 0; i < S.len; ++i) {
        const int v = S.v[i];

        for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
            const int c = T.vn_cn[p];
            uint8_t& d = scratch.deg[static_cast<size_t>(c)];

            if (d == 0) {
                scratch.touched.push_back(c);
            }

            ++d;
        }
    }

    StructureStats st;

    for (Index c : scratch.touched) {
        const uint8_t d = scratch.deg[static_cast<size_t>(c)];

        if (d > 2) {
            st.elementary = false;
        }

        if (d & 1u) {
            ++st.b;
        }
    }

    return st;
}

// ============================================================================
// Parameters / result
// ============================================================================

struct EnumeratorParams {
    uint8_t g = 0;          // Tanner girth, even
    uint8_t amax = 0;       // maximum support size
    uint16_t b_aux_max = 0; // max_a bmax_by_a[a], derived by the Python binder
    uint16_t b_target_max = 0;

    // Algorithm-1 class-dependent bounds, indexed by support size a.
    std::vector<uint16_t> bmax_by_a;

    // Optional ETS-with-leaves stage using dot_1^k expansions.
    bool include_ets_with_leaves = false;

    // If true, initialize the ETSL stage from singleton variable nodes in
    // addition to the LETSs found by DPL.
    bool include_singleton_etsl_seeds = false;

    // Allowed variable-node degrees. An empty vector allows every positive degree.
    std::vector<uint8_t> allowed_variable_degrees;

    // Final supports are streamed to a_<a>_b_<b>.csv files in this directory.
    std::string support_output_dir;

    bool verbose = false;
};

template <size_t AMAX>
struct EnumeratorResult {
    // Flattened count table: counts[b * (AMAX + 1) + a].
    std::vector<uint64_t> counts;

    uint64_t generated_candidates = 0;
    uint64_t accepted_intermediate = 0;
    uint64_t rejected_by_size = 0;
    uint64_t rejected_by_b = 0;
    uint64_t rejected_non_elementary = 0;
    uint64_t duplicate_candidates = 0;
    uint64_t written_supports = 0;
};

// ============================================================================
// Outer wrapper
// ============================================================================

template <size_t AMAX>
class TS_enumerator_irreg23 {
public:
    using SupportT = Support<AMAX>;

    TS_enumerator_irreg23(
        const TannerTopology& topo,
        EnumeratorParams params
    )
        : T(topo),
          params_(std::move(params)),
          scratch_(topo.M),
          parent_scratch_(topo.M),
          vn_mark_(static_cast<size_t>(topo.N), 0),
          var_on_path_(static_cast<size_t>(topo.N), 0),
          check_on_path_(static_cast<size_t>(topo.M), 0),
          check_on_cycle_(static_cast<size_t>(topo.M), 0)
    {
        validate_params();
        validate_allowed_variable_degrees();
        init_support_output();
    }

    EnumeratorResult<AMAX> run() {
        return run_external_memory();
    }

private:
    const TannerTopology& T;
    EnumeratorParams params_;

    EvalScratch scratch_;
    EvalScratch parent_scratch_;

    std::vector<uint32_t> vn_mark_;
    uint32_t vn_stamp_ = 1;

    std::vector<uint8_t> var_on_path_;
    std::vector<uint8_t> check_on_path_;
    std::vector<uint8_t> check_on_cycle_;

    std::vector<std::vector<std::unique_ptr<std::ofstream>>> support_streams_;


    static constexpr uint8_t CAND_FLAG_SEED_CYCLE = 1u;
    static constexpr uint8_t CAND_FLAG_ETSL_DPL_SEED = 2u;
    static constexpr uint8_t CAND_FLAG_ETSL_DOT1 = 4u;
    static constexpr uint8_t CAND_FLAG_ETSL_SINGLETON = 8u;

    struct DiskCandidate {
        SupportT S;
        uint8_t flags = 0;
    };

    struct PendingRecord {
        uint32_t bucket = 0;
        DiskCandidate cand;
    };

    static bool support_equal(const SupportT& a, const SupportT& b) {
        return a == b;
    }

    static bool candidate_less(const DiskCandidate& a, const DiskCandidate& b) {
        if (a.S.len != b.S.len) return a.S.len < b.S.len;

        for (uint8_t i = 0; i < a.S.len; ++i) {
            if (a.S.v[i] != b.S.v[i]) {
                return a.S.v[i] < b.S.v[i];
            }
        }

        return false;
    }

    static bool pending_record_less(
        const PendingRecord& a,
        const PendingRecord& b
    ) {
        if (a.bucket != b.bucket) {
            return a.bucket < b.bucket;
        }
        return candidate_less(a.cand, b.cand);
    }

    static bool pending_record_equal(
        const PendingRecord& a,
        const PendingRecord& b
    ) {
        return a.bucket == b.bucket && support_equal(a.cand.S, b.cand.S);
    }

    std::string io_failure_message(
        const std::string& context,
        const std::filesystem::path& path,
        int error_number
    ) const {
        std::ostringstream oss;
        oss << context << " | path=" << path.string();

        if (error_number != 0) {
            oss << " | errno=" << error_number
                << " (" << std::strerror(error_number) << ")";
        } else {
            oss << " | errno=0 (stream failure without errno)";
        }

        std::error_code ec;
        const auto parent = path.has_parent_path()
            ? path.parent_path()
            : std::filesystem::current_path(ec);

        if (!ec) {
            const auto info = std::filesystem::space(parent, ec);
            if (!ec) {
                oss << " | space_available=" << info.available
                    << " | space_capacity=" << info.capacity;
            }
        }

        return oss.str();
    }

    std::filesystem::path make_external_work_dir() const {
        std::filesystem::path base;

        if (!params_.support_output_dir.empty()) {
            base = std::filesystem::path(params_.support_output_dir);
            std::filesystem::create_directories(base);
        } else {
            base = std::filesystem::temp_directory_path();
        }

        const auto now = std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count();

        const std::string dirname =
            std::string(".dpl_tmp_") + std::to_string(static_cast<long long>(now)) +
            std::string("_") + std::to_string(reinterpret_cast<std::uintptr_t>(this));

        std::filesystem::path work_dir = base / dirname;
        std::filesystem::create_directories(work_dir);
        std::filesystem::create_directories(work_dir / "pending");
        std::filesystem::create_directories(work_dir / "frontier");
        std::filesystem::create_directories(work_dir / "cycles");
        std::filesystem::create_directories(work_dir / "etsl_pending");
        std::filesystem::create_directories(work_dir / "etsl_frontier");

        return work_dir;
    }

    std::filesystem::path pending_run_path(
        const std::filesystem::path& work_dir,
        uint8_t a,
        uint32_t pass,
        uint64_t run_index
    ) const {
        return work_dir / "pending" /
            (std::string("a_") + std::to_string(static_cast<int>(a)) +
             std::string("_pass_") + std::to_string(pass) +
             std::string("_run_") + std::to_string(run_index) +
             std::string(".bin"));
    }

    std::filesystem::path etsl_pending_run_path(
        const std::filesystem::path& work_dir,
        uint8_t a,
        uint32_t pass,
        uint64_t run_index
    ) const {
        return work_dir / "etsl_pending" /
            (std::string("a_") + std::to_string(static_cast<int>(a)) +
             std::string("_pass_") + std::to_string(pass) +
             std::string("_run_") + std::to_string(run_index) +
             std::string(".bin"));
    }

    std::filesystem::path etsl_frontier_file_path(
        const std::filesystem::path& work_dir,
        uint8_t a,
        uint16_t b
    ) const {
        return work_dir / "etsl_frontier" /
            (std::string("a_") + std::to_string(static_cast<int>(a)) +
             std::string("_b_") + std::to_string(static_cast<int>(b)) +
             std::string(".bin"));
    }

    std::filesystem::path frontier_file_path(
        const std::filesystem::path& work_dir,
        uint8_t a,
        uint16_t b
    ) const {
        return work_dir / "frontier" /
            (std::string("a_") + std::to_string(static_cast<int>(a)) +
             std::string("_b_") + std::to_string(static_cast<int>(b)) +
             std::string(".bin"));
    }

    std::filesystem::path cycle_file_path(
        const std::filesystem::path& work_dir,
        uint8_t c
    ) const {
        return work_dir / "cycles" /
            (std::string("c_") + std::to_string(static_cast<int>(c)) +
             std::string(".bin"));
    }

    void write_support_binary(std::ostream& os, const SupportT& S) const {
        os.write(reinterpret_cast<const char*>(&S.len), sizeof(S.len));

        for (size_t i = 0; i < AMAX; ++i) {
            const int32_t x = (i < S.len) ? static_cast<int32_t>(S.v[i]) : int32_t{0};
            os.write(reinterpret_cast<const char*>(&x), sizeof(x));
        }

        if (!os) {
            throw std::runtime_error("write_support_binary: write failed");
        }
    }

    bool read_support_binary(std::istream& is, SupportT& S) const {
        uint8_t len = 0;
        is.read(reinterpret_cast<char*>(&len), sizeof(len));

        if (!is) {
            return false;
        }

        if (len > AMAX) {
            throw std::runtime_error("read_support_binary: support length exceeds AMAX");
        }

        S.len = len;

        for (size_t i = 0; i < AMAX; ++i) {
            int32_t x = 0;
            is.read(reinterpret_cast<char*>(&x), sizeof(x));

            if (!is) {
                throw std::runtime_error("read_support_binary: truncated record");
            }

            if (i < S.len) {
                S.v[i] = static_cast<Index>(x);
            }
        }

        return true;
    }

    void write_candidate_binary(
        std::ostream& os,
        const SupportT& S,
        uint8_t flags
    ) const {
        os.write(reinterpret_cast<const char*>(&S.len), sizeof(S.len));
        os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));

        for (size_t i = 0; i < AMAX; ++i) {
            const int32_t x = (i < S.len) ? static_cast<int32_t>(S.v[i]) : int32_t{0};
            os.write(reinterpret_cast<const char*>(&x), sizeof(x));
        }

        if (!os) {
            throw std::runtime_error("write_candidate_binary: write failed");
        }
    }

    bool read_candidate_binary(std::istream& is, DiskCandidate& cand) const {
        uint8_t len = 0;
        uint8_t flags = 0;

        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!is) {
            return false;
        }

        is.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        if (!is) {
            throw std::runtime_error("read_candidate_binary: truncated candidate flags");
        }

        if (len > AMAX) {
            throw std::runtime_error("read_candidate_binary: support length exceeds AMAX");
        }

        cand.S.len = len;
        cand.flags = flags;

        for (size_t i = 0; i < AMAX; ++i) {
            int32_t x = 0;
            is.read(reinterpret_cast<char*>(&x), sizeof(x));

            if (!is) {
                throw std::runtime_error("read_candidate_binary: truncated candidate record");
            }

            if (i < cand.S.len) {
                cand.S.v[i] = static_cast<Index>(x);
            }
        }

        return true;
    }

    void write_pending_record_binary(
        std::ostream& os,
        const PendingRecord& rec,
        const std::filesystem::path& path
    ) const {
        errno = 0;
        os.write(
            reinterpret_cast<const char*>(&rec.bucket),
            sizeof(rec.bucket)
        );
        os.write(
            reinterpret_cast<const char*>(&rec.cand.S.len),
            sizeof(rec.cand.S.len)
        );
        os.write(
            reinterpret_cast<const char*>(&rec.cand.flags),
            sizeof(rec.cand.flags)
        );

        for (size_t i = 0; i < AMAX; ++i) {
            const int32_t x = (i < rec.cand.S.len)
                ? static_cast<int32_t>(rec.cand.S.v[i])
                : int32_t{0};
            os.write(reinterpret_cast<const char*>(&x), sizeof(x));
        }

        if (!os) {
            const int saved_errno = errno;
            throw std::runtime_error(
                io_failure_message(
                    "write_pending_record_binary: write failed",
                    path,
                    saved_errno
                )
            );
        }
    }

    bool read_pending_record_binary(
        std::istream& is,
        PendingRecord& rec,
        const std::filesystem::path& path
    ) const {
        uint32_t bucket = 0;
        uint8_t len = 0;
        uint8_t flags = 0;

        is.read(reinterpret_cast<char*>(&bucket), sizeof(bucket));
        if (!is) {
            if (is.eof() && is.gcount() == 0) {
                return false;
            }
            throw std::runtime_error(
                "read_pending_record_binary: truncated bucket in " + path.string()
            );
        }

        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!is) {
            throw std::runtime_error(
                "read_pending_record_binary: truncated length in " + path.string()
            );
        }

        is.read(reinterpret_cast<char*>(&flags), sizeof(flags));
        if (!is) {
            throw std::runtime_error(
                "read_pending_record_binary: truncated flags in " + path.string()
            );
        }

        if (len > AMAX) {
            throw std::runtime_error(
                "read_pending_record_binary: support length exceeds AMAX in " +
                path.string()
            );
        }

        rec.bucket = bucket;
        rec.cand.S.len = len;
        rec.cand.flags = flags;

        for (size_t i = 0; i < AMAX; ++i) {
            int32_t x = 0;
            is.read(reinterpret_cast<char*>(&x), sizeof(x));

            if (!is) {
                throw std::runtime_error(
                    "read_pending_record_binary: truncated record in " + path.string()
                );
            }

            if (i < rec.cand.S.len) {
                rec.cand.S.v[i] = static_cast<Index>(x);
            }
        }

        return true;
    }

    template <typename Emit>
    void stream_support_file(
        const std::filesystem::path& path,
        Emit&& emit
    ) const {
        if (!std::filesystem::exists(path)) {
            return;
        }

        std::ifstream is(path, std::ios::in | std::ios::binary);

        if (!is) {
            throw std::runtime_error("stream_support_file: failed to open " + path.string());
        }

        SupportT S;
        while (read_support_binary(is, S)) {
            emit(S);
        }
    }

    template <typename EmitBatch>
    void stream_support_file_batches(
        const std::filesystem::path& path,
        size_t batch_size,
        EmitBatch&& emit_batch
    ) const {
        if (!std::filesystem::exists(path)) {
            return;
        }

        if (batch_size == 0) {
            batch_size = 1;
        }

        std::ifstream is(path, std::ios::in | std::ios::binary);

        if (!is) {
            throw std::runtime_error("stream_support_file_batches: failed to open " + path.string());
        }

        std::vector<SupportT> batch;
        batch.reserve(batch_size);

        SupportT S;
        while (read_support_binary(is, S)) {
            batch.push_back(S);

            if (batch.size() >= batch_size) {
                emit_batch(batch);
                batch.clear();
            }
        }

        if (!batch.empty()) {
            emit_batch(batch);
        }
    }

    EnumeratorResult<AMAX> run_external_memory() {
        // Fixed external-memory settings. They affect performance and memory use,
        // but not the enumerated trapping sets.
        static constexpr uint32_t NUM_BUCKETS = 512;
        static constexpr size_t CYCLE_BATCH_SIZE = 4096;
        static constexpr size_t SORT_RUN_RECORDS = 1'000'000;
        static constexpr size_t MERGE_FAN_IN = 128;

        EnumeratorResult<AMAX> result;
        result.counts.assign(
            static_cast<size_t>(params_.b_aux_max + 1) * (AMAX + 1),
            0
        );

        const uint8_t k0 = static_cast<uint8_t>(params_.g / 2);
        const uint8_t maxk = params_.amax;
        const uint32_t num_buckets = NUM_BUCKETS;
        const size_t cycle_batch_size = CYCLE_BATCH_SIZE;
        const size_t sort_run_records = SORT_RUN_RECORDS;
        const size_t merge_fan_in = MERGE_FAN_IN;

        const std::filesystem::path work_dir = make_external_work_dir();

        // In ETSL mode, every accepted LETS is also inserted into the disk-backed
        // ETSL frontier before the corresponding dot_1^k layer is processed.
        std::function<void(const SupportT&, uint16_t)> add_lets_to_etsl_layer =
            [](const SupportT&, uint16_t) {};

        try {
            if (params_.verbose) {
                std::cout << "[DPL] start search"
                          << " | g=" << static_cast<int>(params_.g)
                          << " | amax=" << static_cast<int>(params_.amax)
                          << " | bmax=" << params_.b_target_max
                          << " | ETSL="
                          << (params_.include_ets_with_leaves ? "yes" : "no")
                          << " | singleton_ETSL_seeds="
                          << (params_.include_singleton_etsl_seeds ? "yes" : "no")
                          << " | work_dir=" << work_dir.string()
                          << std::endl;
            }

            std::vector<std::vector<std::unique_ptr<std::ofstream>>> frontier_streams(AMAX + 1);
            for (auto& row : frontier_streams) {
                row.resize(static_cast<size_t>(params_.b_aux_max + 1));
            }
            std::vector<std::unique_ptr<std::ofstream>> cycle_streams(AMAX + 1);

            auto open_append_stream = [&]( 
                std::unique_ptr<std::ofstream>& ptr,
                const std::filesystem::path& path
            ) -> std::ofstream& {
                if (!ptr) {
                    std::error_code ec;
                    std::filesystem::create_directories(path.parent_path(), ec);
                    if (ec) {
                        throw std::runtime_error(
                            "open_append_stream: create_directories failed for " +
                            path.parent_path().string() + ": " + ec.message()
                        );
                    }

                    errno = 0;
                    ptr = std::make_unique<std::ofstream>(
                        path,
                        std::ios::out | std::ios::binary | std::ios::app
                    );

                    if (!(*ptr)) {
                        const int saved_errno = errno;
                        ptr.reset();
                        throw std::runtime_error(
                            io_failure_message(
                                "open_append_stream: failed to open",
                                path,
                                saved_errno
                            )
                        );
                    }
                }

                return *ptr;
            };

            auto close_stream_checked = [&]( 
                std::unique_ptr<std::ofstream>& ptr,
                const std::filesystem::path& path,
                const std::string& context
            ) {
                if (!ptr) {
                    return;
                }

                errno = 0;
                ptr->flush();
                if (!(*ptr)) {
                    const int saved_errno = errno;
                    ptr.reset();
                    throw std::runtime_error(
                        io_failure_message(context + ": flush failed", path, saved_errno)
                    );
                }

                errno = 0;
                ptr->close();
                if (!(*ptr)) {
                    const int saved_errno = errno;
                    ptr.reset();
                    throw std::runtime_error(
                        io_failure_message(context + ": close failed", path, saved_errno)
                    );
                }

                ptr.reset();
            };

            auto close_target_streams = [&](uint8_t a) {
                if (a > AMAX) {
                    return;
                }

                if (cycle_streams[a]) {
                    close_stream_checked(
                        cycle_streams[a],
                        cycle_file_path(work_dir, a),
                        "cycle stream"
                    );
                }

                for (uint16_t b = 0; b <= params_.b_aux_max; ++b) {
                    if (frontier_streams[a][b]) {
                        close_stream_checked(
                            frontier_streams[a][b],
                            frontier_file_path(work_dir, a, b),
                            "frontier stream"
                        );
                    }
                }
            };

            auto append_frontier = [&](const SupportT& S, uint16_t b) {
                if (S.len == 0 || S.len > AMAX || b > params_.b_aux_max) {
                    return;
                }

                const auto path = frontier_file_path(work_dir, S.len, b);
                std::ofstream& os = open_append_stream(
                    frontier_streams[S.len][b],
                    path
                );

                try {
                    write_support_binary(os, S);
                } catch (...) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message(
                            "append_frontier: write failed",
                            path,
                            saved_errno
                        )
                    );
                }
            };

            auto append_cycle = [&](const SupportT& S) {
                if (S.len == 0 || S.len > AMAX) {
                    return;
                }

                const auto path = cycle_file_path(work_dir, S.len);
                std::ofstream& os = open_append_stream(
                    cycle_streams[S.len],
                    path
                );

                try {
                    write_support_binary(os, S);
                } catch (...) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message(
                            "append_cycle: write failed",
                            path,
                            saved_errno
                        )
                    );
                }
            };

            auto write_run_file = [&]( 
                const std::filesystem::path& path,
                const std::vector<PendingRecord>& records,
                size_t count
            ) {
                std::error_code ec;
                std::filesystem::create_directories(path.parent_path(), ec);
                if (ec) {
                    throw std::runtime_error(
                        "write_run_file: create_directories failed for " +
                        path.parent_path().string() + ": " + ec.message()
                    );
                }

                errno = 0;
                std::ofstream os(path, std::ios::out | std::ios::binary | std::ios::trunc);
                if (!os) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message("write_run_file: open failed", path, saved_errno)
                    );
                }

                for (size_t i = 0; i < count; ++i) {
                    write_pending_record_binary(os, records[i], path);
                }

                errno = 0;
                os.flush();
                if (!os) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message("write_run_file: flush failed", path, saved_errno)
                    );
                }

                errno = 0;
                os.close();
                if (!os) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message("write_run_file: close failed", path, saved_errno)
                    );
                }
            };

            auto merge_run_group = [&]( 
                const std::vector<std::filesystem::path>& inputs,
                const std::function<void(const PendingRecord&)>& emit
            ) {
                struct ReaderState {
                    std::filesystem::path path;
                    std::unique_ptr<std::ifstream> stream;
                };

                struct HeapItem {
                    PendingRecord rec;
                    size_t reader = 0;
                };

                auto heap_greater = [](const HeapItem& lhs, const HeapItem& rhs) {
                    return pending_record_less(rhs.rec, lhs.rec);
                };

                std::vector<ReaderState> readers;
                readers.reserve(inputs.size());

                std::priority_queue<
                    HeapItem,
                    std::vector<HeapItem>,
                    decltype(heap_greater)
                > heap(heap_greater);

                for (const auto& path : inputs) {
                    errno = 0;
                    auto stream = std::make_unique<std::ifstream>(
                        path,
                        std::ios::in | std::ios::binary
                    );

                    if (!(*stream)) {
                        const int saved_errno = errno;
                        throw std::runtime_error(
                            io_failure_message(
                                "merge_run_group: failed to open input",
                                path,
                                saved_errno
                            )
                        );
                    }

                    const size_t reader_index = readers.size();
                    readers.push_back(ReaderState{path, std::move(stream)});

                    PendingRecord first;
                    if (read_pending_record_binary(
                            *readers.back().stream,
                            first,
                            readers.back().path
                        )) {
                        heap.push(HeapItem{first, reader_index});
                    }
                }

                bool have_merged = false;
                PendingRecord merged;

                while (!heap.empty()) {
                    HeapItem item = heap.top();
                    heap.pop();

                    if (!have_merged) {
                        merged = item.rec;
                        have_merged = true;
                    } else if (pending_record_equal(merged, item.rec)) {
                        merged.cand.flags |= item.rec.cand.flags;
                    } else {
                        emit(merged);
                        merged = item.rec;
                    }

                    ReaderState& reader = readers[item.reader];
                    PendingRecord next;
                    if (read_pending_record_binary(
                            *reader.stream,
                            next,
                            reader.path
                        )) {
                        heap.push(HeapItem{next, item.reader});
                    }
                }

                if (have_merged) {
                    emit(merged);
                }
            };

            auto process_pending_for_target = [&]( 
                uint8_t target_a,
                std::vector<std::filesystem::path> run_paths,
                uint64_t raw_records
            ) {
                uint32_t pass = 1;
                uint64_t next_run_index = 0;

                while (run_paths.size() > merge_fan_in) {
                    std::vector<std::filesystem::path> next_paths;
                    next_paths.reserve(
                        (run_paths.size() + merge_fan_in - 1) / merge_fan_in
                    );

                    for (size_t begin = 0; begin < run_paths.size(); begin += merge_fan_in) {
                        const size_t end = std::min(
                            run_paths.size(),
                            begin + merge_fan_in
                        );

                        std::vector<std::filesystem::path> group(
                            run_paths.begin() + static_cast<std::ptrdiff_t>(begin),
                            run_paths.begin() + static_cast<std::ptrdiff_t>(end)
                        );

                        const auto output_path = pending_run_path(
                            work_dir,
                            target_a,
                            pass,
                            next_run_index++
                        );

                        errno = 0;
                        std::ofstream os(
                            output_path,
                            std::ios::out | std::ios::binary | std::ios::trunc
                        );
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_pending_for_target: merge output open failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        merge_run_group(group, [&](const PendingRecord& rec) {
                            write_pending_record_binary(os, rec, output_path);
                        });

                        errno = 0;
                        os.flush();
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_pending_for_target: merge output flush failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        errno = 0;
                        os.close();
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_pending_for_target: merge output close failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        for (const auto& input_path : group) {
                            std::error_code remove_ec;
                            std::filesystem::remove(input_path, remove_ec);
                            if (remove_ec) {
                                throw std::runtime_error(
                                    "process_pending_for_target: failed to remove " +
                                    input_path.string() + ": " + remove_ec.message()
                                );
                            }
                        }

                        next_paths.push_back(output_path);
                    }

                    run_paths.swap(next_paths);
                    ++pass;
                    next_run_index = 0;
                }

                uint64_t unique_records = 0;

                if (!run_paths.empty()) {
                    merge_run_group(run_paths, [&](const PendingRecord& rec) {
                        ++unique_records;

                        const SupportT& S = rec.cand.S;
                        const uint8_t flags = rec.cand.flags;
                        const StructureStats st = induced_stats(T, S, scratch_);

                        if (!st.elementary) {
                            ++result.rejected_non_elementary;
                            return;
                        }

                        if (st.b > b_bound_for_size(S.len)) {
                            ++result.rejected_by_b;
                            return;
                        }

                        ++result.accepted_intermediate;

                        // The merged record is now a unique accepted LETS.
                        // Insert it into the online ETSL layer before that
                        // support size is processed by dot_1^k.
                        add_lets_to_etsl_layer(S, st.b);

                        if ((flags & CAND_FLAG_SEED_CYCLE) != 0) {
                            append_cycle(S);
                        }

                        const bool is_target = st.b <= params_.b_target_max;

                        if (is_target) {
                            increment_count(result, st.b, S.len);
                            write_support_if_requested(S, st.b, result);
                        }

                        if (st.b != 0 && S.len < params_.amax) {
                            append_frontier(S, st.b);
                        }
                    });
                }

                if (raw_records < unique_records) {
                    throw std::runtime_error(
                        "process_pending_for_target: unique record count exceeds raw count"
                    );
                }
                result.duplicate_candidates += raw_records - unique_records;

                for (const auto& path : run_paths) {
                    std::error_code remove_ec;
                    std::filesystem::remove(path, remove_ec);
                    if (remove_ec) {
                        throw std::runtime_error(
                            "process_pending_for_target: failed to remove " +
                            path.string() + ": " + remove_ec.message()
                        );
                    }
                }

                close_target_streams(target_a);

                if (params_.verbose) {
                    std::cout << "[DPL-ext] a=" << static_cast<int>(target_a)
                              << " merge/dedup/accept done"
                              << " | raw=" << raw_records
                              << " | unique=" << unique_records
                              << " | accepted_total=" << result.accepted_intermediate
                              << " | duplicates_total=" << result.duplicate_candidates
                              << std::endl;
                }
            };

            std::vector<PendingRecord> pending_buffer;
            pending_buffer.reserve(sort_run_records);
            std::vector<std::filesystem::path> pending_runs;
            uint64_t pending_raw_records = 0;
            uint64_t pending_run_index = 0;
            uint8_t pending_target_a = 0;

            auto flush_pending_run = [&]() {
                if (pending_buffer.empty()) {
                    return;
                }

                std::sort(
                    pending_buffer.begin(),
                    pending_buffer.end(),
                    pending_record_less
                );

                size_t out = 0;
                size_t i = 0;
                while (i < pending_buffer.size()) {
                    PendingRecord merged = pending_buffer[i];
                    size_t j = i + 1;

                    while (
                        j < pending_buffer.size() &&
                        pending_record_equal(merged, pending_buffer[j])
                    ) {
                        merged.cand.flags |= pending_buffer[j].cand.flags;
                        ++j;
                    }

                    pending_buffer[out++] = merged;
                    i = j;
                }

                const auto path = pending_run_path(
                    work_dir,
                    pending_target_a,
                    0,
                    pending_run_index++
                );
                write_run_file(path, pending_buffer, out);
                pending_runs.push_back(path);
                pending_buffer.clear();
            };

            auto begin_pending_target = [&](uint8_t target_a) {
                if (!pending_buffer.empty() || !pending_runs.empty()) {
                    throw std::runtime_error(
                        "begin_pending_target: previous target was not finalized"
                    );
                }

                pending_target_a = target_a;
                pending_raw_records = 0;
                pending_run_index = 0;
            };

            // Used for serially generated cycle seeds. Parent-generated
            // candidates are appended to the same bounded external-sort buffer.
            auto append_pending = [&](const SupportT& S, uint8_t flags) {
                if (S.len == 0 || S.len > params_.amax) {
                    ++result.rejected_by_size;
                    return;
                }

                ++result.generated_candidates;
                ++pending_raw_records;

                PendingRecord rec;
                rec.bucket = static_cast<uint32_t>(
                    SupportHash<AMAX>{}(S) % num_buckets
                );
                rec.cand.S = S;
                rec.cand.flags = flags;
                pending_buffer.push_back(rec);

                if (pending_buffer.size() >= sort_run_records) {
                    flush_pending_run();
                }
            };


            auto generate_parent_candidates = [&](uint8_t target_a) {
                for (uint8_t parent_a = k0; parent_a < target_a; ++parent_a) {
                    const uint8_t m_needed =
                        static_cast<uint8_t>(target_a - parent_a);
                    const uint16_t bmax_parent = b_bound_for_size(parent_a);

                    for (uint16_t parent_b = 1; parent_b <= bmax_parent; ++parent_b) {
                        const auto path =
                            frontier_file_path(work_dir, parent_a, parent_b);

                        stream_support_file(path, [&](const SupportT& parent) {
                            const StructureStats parent_st =
                                induced_stats(T, parent, parent_scratch_);

                            if (!parent_st.elementary || parent_st.b != parent_b) {
                                return;
                            }

                            if (parent_b > b_bound_for_size(parent_a)) {
                                return;
                            }

                            if (
                                m_needed == 1 &&
                                ex_allows_dot(parent_a, parent_b)
                            ) {
                                expand_dot(parent, [&](const SupportT& child) {
                                    append_pending(child, 0u);
                                });
                            }

                            if (ex_allows_path(parent_a, parent_b, m_needed)) {
                                expand_path(
                                    parent,
                                    m_needed,
                                    [&](const SupportT& child) {
                                        append_pending(child, 0u);
                                    }
                                );
                            }

                            if (ex_allows_lollipop(parent_a, parent_b, m_needed)) {
                                const uint8_t g2 =
                                    static_cast<uint8_t>(params_.g / 2);

                                for (uint8_t c = g2; c <= m_needed; ++c) {
                                    const auto cycle_path =
                                        cycle_file_path(work_dir, c);

                                    stream_support_file_batches(
                                        cycle_path,
                                        cycle_batch_size,
                                        [&](const std::vector<SupportT>& cycles) {
                                            expand_lollipop(
                                                parent,
                                                m_needed,
                                                c,
                                                cycles,
                                                [&](const SupportT& child) {
                                                    append_pending(child, 0u);
                                                }
                                            );
                                        }
                                    );
                                }
                            }
                        });
                    }
                }
            };

            // --------------------------------------------------------------
            // Fully external-memory ETS-with-leaves (dot_1^k) stage.
            //
            // The previous implementation kept each ETSL frontier and an
            // in-memory deduplication table. Here, LETS seeds and dot_1^k children
            // are accumulated as bounded sorted runs, merged/deduplicated on
            // disk, and written to disk-backed (a,b) frontier files.
            // --------------------------------------------------------------
            std::vector<PendingRecord> etsl_pending_buffer;
            std::vector<std::filesystem::path> etsl_pending_runs;
            uint64_t etsl_pending_raw_records = 0;
            uint64_t etsl_pending_run_index = 0;
            uint8_t etsl_pending_target_a = 0;
            bool etsl_pending_active = false;

            auto begin_etsl_pending_target = [&](uint8_t target_a) {
                if (!params_.include_ets_with_leaves) {
                    return;
                }

                if (etsl_pending_active) {
                    if (etsl_pending_target_a != target_a) {
                        throw std::runtime_error(
                            "begin_etsl_pending_target: another ETSL target is active"
                        );
                    }
                    return;
                }

                if (!etsl_pending_buffer.empty() || !etsl_pending_runs.empty()) {
                    throw std::runtime_error(
                        "begin_etsl_pending_target: stale ETSL pending state"
                    );
                }

                etsl_pending_target_a = target_a;
                etsl_pending_raw_records = 0;
                etsl_pending_run_index = 0;
                etsl_pending_active = true;
            };

            auto flush_etsl_pending_run = [&]() {
                if (etsl_pending_buffer.empty()) {
                    return;
                }

                std::sort(
                    etsl_pending_buffer.begin(),
                    etsl_pending_buffer.end(),
                    pending_record_less
                );

                size_t out = 0;
                size_t i = 0;
                while (i < etsl_pending_buffer.size()) {
                    PendingRecord merged = etsl_pending_buffer[i];
                    size_t j = i + 1;

                    while (
                        j < etsl_pending_buffer.size() &&
                        pending_record_equal(merged, etsl_pending_buffer[j])
                    ) {
                        merged.cand.flags |= etsl_pending_buffer[j].cand.flags;
                        ++j;
                    }

                    etsl_pending_buffer[out++] = merged;
                    i = j;
                }

                const auto path = etsl_pending_run_path(
                    work_dir,
                    etsl_pending_target_a,
                    0,
                    etsl_pending_run_index++
                );
                write_run_file(path, etsl_pending_buffer, out);
                etsl_pending_runs.push_back(path);

                // Release the capacity so the ETSL buffer does not retain
                // unnecessary memory between layers.
                std::vector<PendingRecord>().swap(etsl_pending_buffer);
            };

            auto append_etsl_pending = [&](
                const SupportT& S,
                uint8_t flags,
                bool count_as_generated
            ) {
                if (!params_.include_ets_with_leaves) {
                    return;
                }

                if (S.len == 0 || S.len > params_.amax) {
                    if (count_as_generated) {
                        ++result.rejected_by_size;
                    }
                    return;
                }

                begin_etsl_pending_target(S.len);

                if (count_as_generated) {
                    ++result.generated_candidates;
                }
                ++etsl_pending_raw_records;

                PendingRecord rec;
                rec.bucket = static_cast<uint32_t>(
                    SupportHash<AMAX>{}(S) % num_buckets
                );
                rec.cand.S = S;
                rec.cand.flags = flags;
                etsl_pending_buffer.push_back(rec);

                if (etsl_pending_buffer.size() >= sort_run_records) {
                    flush_etsl_pending_run();
                }
            };

            add_lets_to_etsl_layer = [&](const SupportT& S, uint16_t b) {
                if (
                    !params_.include_ets_with_leaves ||
                    b == 0 ||
                    b > params_.b_target_max
                ) {
                    return;
                }

                append_etsl_pending(
                    S,
                    CAND_FLAG_ETSL_DPL_SEED,
                    false
                );
            };

            auto process_etsl_pending_for_target = [&](uint8_t target_a) {
                if (!params_.include_ets_with_leaves) {
                    return;
                }

                begin_etsl_pending_target(target_a);
                flush_etsl_pending_run();

                std::vector<std::filesystem::path> run_paths =
                    std::move(etsl_pending_runs);
                const uint64_t raw_records = etsl_pending_raw_records;

                etsl_pending_runs.clear();
                etsl_pending_raw_records = 0;
                etsl_pending_run_index = 0;
                etsl_pending_active = false;

                uint32_t pass = 1;
                uint64_t next_run_index = 0;

                while (run_paths.size() > merge_fan_in) {
                    std::vector<std::filesystem::path> next_paths;
                    next_paths.reserve(
                        (run_paths.size() + merge_fan_in - 1) / merge_fan_in
                    );

                    for (size_t begin = 0; begin < run_paths.size(); begin += merge_fan_in) {
                        const size_t end = std::min(
                            run_paths.size(),
                            begin + merge_fan_in
                        );

                        std::vector<std::filesystem::path> group(
                            run_paths.begin() + static_cast<std::ptrdiff_t>(begin),
                            run_paths.begin() + static_cast<std::ptrdiff_t>(end)
                        );

                        const auto output_path = etsl_pending_run_path(
                            work_dir,
                            target_a,
                            pass,
                            next_run_index++
                        );

                        errno = 0;
                        std::ofstream os(
                            output_path,
                            std::ios::out | std::ios::binary | std::ios::trunc
                        );
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_etsl_pending_for_target: merge output open failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        merge_run_group(group, [&](const PendingRecord& rec) {
                            write_pending_record_binary(os, rec, output_path);
                        });

                        errno = 0;
                        os.flush();
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_etsl_pending_for_target: merge output flush failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        errno = 0;
                        os.close();
                        if (!os) {
                            const int saved_errno = errno;
                            throw std::runtime_error(
                                io_failure_message(
                                    "process_etsl_pending_for_target: merge output close failed",
                                    output_path,
                                    saved_errno
                                )
                            );
                        }

                        for (const auto& input_path : group) {
                            std::error_code remove_ec;
                            std::filesystem::remove(input_path, remove_ec);
                            if (remove_ec) {
                                throw std::runtime_error(
                                    "process_etsl_pending_for_target: failed to remove " +
                                    input_path.string() + ": " + remove_ec.message()
                                );
                            }
                        }

                        next_paths.push_back(output_path);
                    }

                    run_paths.swap(next_paths);
                    ++pass;
                    next_run_index = 0;
                }

                std::vector<std::unique_ptr<std::ofstream>> frontier_outputs(
                    static_cast<size_t>(params_.b_target_max + 1)
                );

                auto append_etsl_frontier = [&](const SupportT& S, uint16_t b) {
                    if (
                        S.len >= params_.amax ||
                        b == 0 ||
                        b > params_.b_target_max
                    ) {
                        return;
                    }

                    auto& ptr = frontier_outputs[b];
                    const auto path = etsl_frontier_file_path(work_dir, S.len, b);
                    std::ofstream& os = open_append_stream(ptr, path);
                    write_support_binary(os, S);
                };

                uint64_t unique_records = 0;
                uint64_t accepted_etsl_only = 0;

                if (!run_paths.empty()) {
                    merge_run_group(run_paths, [&](const PendingRecord& rec) {
                        ++unique_records;

                        const SupportT& S = rec.cand.S;
                        const uint8_t flags = rec.cand.flags;
                        const StructureStats st = induced_stats(T, S, scratch_);

                        if (!st.elementary) {
                            ++result.rejected_non_elementary;
                            return;
                        }

                        if (st.b > params_.b_target_max) {
                            ++result.rejected_by_b;
                            return;
                        }

                        const bool already_counted_by_dpl =
                            (flags & CAND_FLAG_ETSL_DPL_SEED) != 0;
                        const bool generated_by_dot1 =
                            (flags & CAND_FLAG_ETSL_DOT1) != 0;
                        const bool singleton_seed =
                            (flags & CAND_FLAG_ETSL_SINGLETON) != 0;

                        if (!already_counted_by_dpl && generated_by_dot1) {
                            ++result.accepted_intermediate;
                            ++accepted_etsl_only;
                        }

                        if (!already_counted_by_dpl && (generated_by_dot1 || singleton_seed)) {
                            add_final_if_target(S, st.b, result);
                        }

                        append_etsl_frontier(S, st.b);
                    });
                }

                for (uint16_t b = 0; b <= params_.b_target_max; ++b) {
                    if (frontier_outputs[b]) {
                        close_stream_checked(
                            frontier_outputs[b],
                            etsl_frontier_file_path(work_dir, target_a, b),
                            "ETSL frontier stream"
                        );
                    }
                }

                if (raw_records < unique_records) {
                    throw std::runtime_error(
                        "process_etsl_pending_for_target: unique count exceeds raw count"
                    );
                }
                result.duplicate_candidates += raw_records - unique_records;

                for (const auto& path : run_paths) {
                    std::error_code remove_ec;
                    std::filesystem::remove(path, remove_ec);
                    if (remove_ec) {
                        throw std::runtime_error(
                            "process_etsl_pending_for_target: failed to remove " +
                            path.string() + ": " + remove_ec.message()
                        );
                    }
                }

                if (params_.verbose) {
                    std::cout << "[ETSL-ext] a=" << static_cast<int>(target_a)
                              << " merge/dedup/accept done"
                              << " | raw=" << raw_records
                              << " | unique=" << unique_records
                              << " | accepted_etsl_only=" << accepted_etsl_only
                              << std::endl;
                }
            };


            auto process_external_etsl_layer = [&](uint8_t a_layer) {
                if (
                    !params_.include_ets_with_leaves ||
                    a_layer == 0 ||
                    a_layer >= params_.amax
                ) {
                    return;
                }

                const uint8_t target_a = static_cast<uint8_t>(a_layer + 1);
                begin_etsl_pending_target(target_a);

                uint64_t parents_total = 0;
                const uint64_t gen_before = result.generated_candidates;

                for (uint16_t b = 1; b <= params_.b_target_max; ++b) {
                    const auto path =
                        etsl_frontier_file_path(work_dir, a_layer, b);

                    stream_support_file(path, [&](const SupportT& parent) {
                        ++parents_total;

                        expand_dot1(
                            parent,
                            b,
                            [&](const SupportT& child, uint16_t) {
                                append_etsl_pending(
                                    child,
                                    CAND_FLAG_ETSL_DOT1,
                                    true
                                );
                            }
                        );
                    });

                    std::error_code ec;
                    std::filesystem::remove(path, ec);
                    if (ec) {
                        throw std::runtime_error(
                            "process_external_etsl_layer: failed to remove " +
                            path.string() + ": " + ec.message()
                        );
                    }
                }

                if (params_.verbose) {
                    std::cout << "[ETSL] a=" << static_cast<int>(a_layer)
                              << " dot_1^k generation"
                              << " | parents=" << parents_total
                              << " | generated="
                              << (result.generated_candidates - gen_before)
                              << std::endl;
                }
            };

            auto initialize_external_etsl_singletons = [&]() {
                if (
                    !params_.include_ets_with_leaves ||
                    !params_.include_singleton_etsl_seeds
                ) {
                    return;
                }

                begin_etsl_pending_target(1);
                for (int v = 0; v < T.N; ++v) {
                    if (!variable_degree_allowed(v)) {
                        continue;
                    }

                    const int dv = T.variable_degree(v);
                    if (static_cast<uint16_t>(dv) > params_.b_target_max) {
                        continue;
                    }

                    SupportT S;
                    S.len = 1;
                    S.v[0] = v;
                    append_etsl_pending(
                        S,
                        CAND_FLAG_ETSL_SINGLETON,
                        false
                    );
                }
            };

            initialize_external_etsl_singletons();

            // Complete singleton-seeded ETSL layers below the minimum
            // cycle size. The pending size-k0 candidates remain unmerged so
            // they can be deduplicated together with the first DPL LETS seeds.
            if (params_.include_ets_with_leaves && k0 > 1) {
                if (params_.include_singleton_etsl_seeds) {
                    process_etsl_pending_for_target(1);
                }

                for (uint8_t a_layer = 1; a_layer < k0; ++a_layer) {
                    process_external_etsl_layer(a_layer);
                    if (static_cast<uint8_t>(a_layer + 1) < k0) {
                        process_etsl_pending_for_target(
                            static_cast<uint8_t>(a_layer + 1)
                        );
                    }
                }
            }

            for (uint8_t target_a = k0; target_a <= params_.amax; ++target_a) {
                begin_pending_target(target_a);
                if (params_.include_ets_with_leaves) {
                    begin_etsl_pending_target(target_a);
                }

                if (params_.verbose) {
                    std::cout << "[DPL] target a=" << static_cast<int>(target_a)
                              << " candidate generation..." << std::flush;
                }

                const uint64_t gen_before = result.generated_candidates;
                const uint64_t acc_before = result.accepted_intermediate;
                const uint64_t dup_before = result.duplicate_candidates;
                const uint64_t rb_before = result.rejected_by_b;
                const uint64_t re_before = result.rejected_non_elementary;

                if (target_a >= k0 && target_a <= maxk) {
                    enumerate_cycles(target_a, [&](const SupportT& cyc) {
                        append_pending(cyc, CAND_FLAG_SEED_CYCLE);
                    });
                }

                generate_parent_candidates(target_a);
                flush_pending_run();

                if (params_.verbose) {
                    std::cout << " done"
                              << " | generated="
                              << (result.generated_candidates - gen_before)
                              << " | sorted_runs=" << pending_runs.size()
                              << std::endl;
                }

                process_pending_for_target(
                    target_a,
                    std::move(pending_runs),
                    pending_raw_records
                );
                pending_runs.clear();
                pending_buffer.clear();
                pending_raw_records = 0;

                // Merge/deduplicate the union of dot_1^k children from the
                // preceding layer and all DPL LETS seeds of this size. The
                // resulting ETSL frontier remains on disk and is streamed to
                // generate the next layer.
                if (params_.include_ets_with_leaves) {
                    process_etsl_pending_for_target(target_a);
                    process_external_etsl_layer(target_a);
                }

                if (params_.verbose) {
                    std::cout << "[DPL] target a=" << static_cast<int>(target_a)
                              << " layer stats"
                              << " | accepted=" << (result.accepted_intermediate - acc_before)
                              << " | duplicates=" << (result.duplicate_candidates - dup_before)
                              << " | rejected_b=" << (result.rejected_by_b - rb_before)
                              << " | rejected_non_ETS=" << (result.rejected_non_elementary - re_before)
                              << std::endl;
                }
            }

            for (uint8_t a = 0; a <= AMAX; ++a) {
                close_target_streams(a);
            }

            flush_support_output_streams();
            std::filesystem::remove_all(work_dir);

            if (params_.verbose) {
                std::cout << "[DPL] search completed"
                          << " | generated=" << result.generated_candidates
                          << " | accepted_intermediate=" << result.accepted_intermediate
                          << " | duplicates=" << result.duplicate_candidates
                          << " | rejected_b=" << result.rejected_by_b
                          << " | rejected_non_ETS=" << result.rejected_non_elementary
                          << " | written_supports=" << result.written_supports
                          << std::endl;
            }

            return result;
        } catch (const std::exception& e) {
            std::cerr << "[DPL] search failed: " << e.what() << '\n'
                      << "[DPL] work directory: " << work_dir.string()
                      << std::endl;
            throw;
        } catch (...) {
            std::cerr << "[DPL] search failed with a non-standard exception\n"
                      << "[DPL] work directory: " << work_dir.string()
                      << std::endl;
            throw;
        }
    }

    uint16_t b_bound_for_size(uint8_t a) const {
        if (static_cast<size_t>(a) >= params_.bmax_by_a.size()) {
            return 0;
        }
        return params_.bmax_by_a[static_cast<size_t>(a)];
    }

    bool ex_allows_dot(uint8_t a, uint16_t b) const {
        // Algorithm 1, line 4.
        return
            a + 1 <= params_.amax &&
            b >= 2 &&
            b <= b_bound_for_size(a);
    }

    bool ex_allows_path(uint8_t a, uint16_t b, uint8_t m) const {
        // Algorithm 1, lines 7--10.
        if (m < 2) return false;
        if (static_cast<int>(a) + static_cast<int>(m) > params_.amax) return false;
        if (b < 2) return false;

        const uint8_t child_a = static_cast<uint8_t>(a + m);
        return static_cast<uint16_t>(b - 2) <= b_bound_for_size(child_a);
    }

    bool ex_allows_lollipop(uint8_t a, uint16_t b, uint8_t m) const {
        // Algorithm 1, lines 11--15.
        const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);

        if (m < g2) return false;
        if (static_cast<int>(a) + static_cast<int>(m) > params_.amax) return false;
        if (b < 1) return false;

        const uint8_t child_a = static_cast<uint8_t>(a + m);
        return static_cast<uint16_t>(b - 1) <= b_bound_for_size(child_a);
    }

    void init_support_output() {
        if (params_.support_output_dir.empty()) {
            return;
        }

        std::filesystem::create_directories(params_.support_output_dir);

        support_streams_.resize(AMAX + 1);
        for (auto& row : support_streams_) {
            row.resize(static_cast<size_t>(params_.b_target_max + 1));
        }
    }

    std::ofstream& support_stream(uint8_t a, uint16_t b) {
        if (a > AMAX || b > params_.b_target_max) {
            throw std::runtime_error("support_stream: invalid (a,b) index");
        }

        auto& ptr = support_streams_[a][b];

        if (!ptr) {
            const std::string filename =
                std::string("a_") + std::to_string(static_cast<int>(a)) +
                std::string("_b_") + std::to_string(static_cast<int>(b)) +
                std::string(".csv");

            const std::filesystem::path path =
                std::filesystem::path(params_.support_output_dir) / filename;

            ptr = std::make_unique<std::ofstream>(path, std::ios::out | std::ios::trunc);

            if (!(*ptr)) {
                throw std::runtime_error("support_stream: failed to open " + path.string());
            }
        }

        return *ptr;
    }

    void write_support_if_requested(
        const SupportT& S,
        uint16_t b,
        EnumeratorResult<AMAX>& result
    ) {
        if (params_.support_output_dir.empty()) {
            return;
        }

        std::ofstream& os = support_stream(S.len, b);

        for (uint8_t i = 0; i < S.len; ++i) {
            if (i != 0) {
                os << ',';
            }
            os << S.v[i];
        }

        os << '\n';

        if (!os) {
            throw std::runtime_error("write_support_if_requested: write failed");
        }

        ++result.written_supports;
    }

    void flush_support_output_streams() {
        for (auto& row : support_streams_) {
            for (auto& ptr : row) {
                if (ptr) {
                    ptr->flush();
                }
            }
        }
    }

    void validate_params() const {
        if (params_.amax == 0 || params_.amax > AMAX) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: amax must be in [1, AMAX]"
            );
        }

        if (params_.g == 0 || (params_.g % 2) != 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: g must be positive and even"
            );
        }

        if (params_.g / 2 > params_.amax) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: g/2 exceeds amax"
            );
        }

        if (T.M <= 0 || T.N <= 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: empty Tanner topology"
            );
        }

        if (params_.support_output_dir.empty()) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: support_output_dir cannot be empty"
            );
        }

        if (params_.bmax_by_a.size() <= static_cast<size_t>(params_.amax)) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: bmax_by_a must have length at least amax + 1"
            );
        }

        if (params_.bmax_by_a[params_.amax] != params_.b_target_max) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: bmax_by_a[amax] must equal b_target_max"
            );
        }

        uint16_t max_bound = 0;
        const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);
        for (uint8_t a = g2; a <= params_.amax; ++a) {
            max_bound = std::max(max_bound, params_.bmax_by_a[a]);
        }

        if (params_.b_aux_max != max_bound) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: b_aux_max must equal max_a bmax_by_a[a]"
            );
        }
    }

    void mark_next_stamp() {
        ++vn_stamp_;

        if (vn_stamp_ == 0) {
            std::fill(vn_mark_.begin(), vn_mark_.end(), 0);
            vn_stamp_ = 1;
        }
    }

    template <size_t AMAX_LOCAL>
    void build_check_degrees_for_parent(
        const Support<AMAX_LOCAL>& S,
        EvalScratch& scratch
    ) const {
        scratch.reset();

        for (uint8_t i = 0; i < S.len; ++i) {
            const int v = S.v[i];

            for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                const int c = T.vn_cn[p];
                uint8_t& d = scratch.deg[static_cast<size_t>(c)];

                if (d == 0) {
                    scratch.touched.push_back(c);
                }

                ++d;
            }
        }
    }

    void validate_allowed_variable_degrees() const {
        for (uint8_t d : params_.allowed_variable_degrees) {
            if (d == 0) {
                throw std::runtime_error(
                    "TS_enumerator_irreg23: allowed_variable_degrees cannot contain 0"
                );
            }
        }
    }

    bool variable_degree_allowed(int v) const {
        const int dv = T.variable_degree(v);

        if (dv <= 0) {
            return false;
        }

        if (params_.allowed_variable_degrees.empty()) {
            return true;
        }

        if (dv > 255) {
            return false;
        }

        const uint8_t d8 = static_cast<uint8_t>(dv);
        return std::find(
            params_.allowed_variable_degrees.begin(),
            params_.allowed_variable_degrees.end(),
            d8
        ) != params_.allowed_variable_degrees.end();
    }

    void increment_count(
        EnumeratorResult<AMAX>& result,
        uint16_t b,
        uint8_t a
    ) const {
        if (b > params_.b_aux_max) return;
        if (a > AMAX) return;

        result.counts[static_cast<size_t>(b) * (AMAX + 1) + a]++;
    }

    void add_final_if_target(
        const SupportT& S,
        uint16_t b,
        EnumeratorResult<AMAX>& result
    ) {
        if (b > params_.b_target_max) {
            return;
        }

        increment_count(result, b, S.len);
        write_support_if_requested(S, b, result);
    }

    template <class Emit>
    void expand_dot1(const SupportT& parent, uint16_t parent_b, Emit&& emit) {
        if (parent.len + 1 > params_.amax) {
            return;
        }

        if (parent_b == 0 || parent_b > params_.b_target_max) {
            return;
        }

        build_check_degrees_for_parent(parent, parent_scratch_);

        mark_next_stamp();

        std::vector<int> candidates;
        candidates.reserve(64);

        // dot_1^k candidates are generated from exactly one unsatisfied
        // parent check. They must touch no satisfied parent checks.
        for (int c : parent_scratch_.touched) {
            const uint8_t d = parent_scratch_.deg[static_cast<size_t>(c)];

            if ((d & 1u) == 0) {
                continue;
            }

            for (int p = T.cn_ptr[c]; p < T.cn_ptr[c + 1]; ++p) {
                const int v = T.cn_vn[p];

                if (parent.contains(v)) {
                    continue;
                }

                if (!variable_degree_allowed(v)) {
                    continue;
                }

                const int dv = T.variable_degree(v);

                const uint16_t child_b = static_cast<uint16_t>(
                    parent_b + static_cast<uint16_t>(dv) - 2u
                );

                if (child_b > params_.b_target_max) {
                    continue;
                }

                uint32_t& mark = vn_mark_[static_cast<size_t>(v)];

                if (mark == vn_stamp_) {
                    continue;
                }

                mark = vn_stamp_;
                candidates.push_back(v);
            }
        }

        for (int v : candidates) {
            int n_unsat = 0;
            bool touches_satisfied = false;

            for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                const int c = T.vn_cn[p];
                const uint8_t d = parent_scratch_.deg[static_cast<size_t>(c)];

                if (d == 0) {
                    continue;
                }

                if (d & 1u) {
                    ++n_unsat;
                } else {
                    touches_satisfied = true;
                    break;
                }
            }

            if (touches_satisfied || n_unsat != 1) {
                continue;
            }

            SupportT child = insert_one_sorted(parent, v);
            const uint16_t child_b = static_cast<uint16_t>(
                parent_b + static_cast<uint16_t>(T.variable_degree(v)) - 2u
            );

            emit(child, child_b);
        }
    }

    // ------------------------------------------------------------------------
    // Expansion hooks.
    //
    // These are intentionally callback-based. They must emit candidates one by
    // one, instead of returning huge vectors.
    // ------------------------------------------------------------------------

    template <class Emit>
    void enumerate_cycles(uint8_t cycle_size_vns, Emit&& emit) {
        const int k = static_cast<int>(cycle_size_vns);

        if (k < 2) {
            return;
        }

        if (cycle_size_vns > params_.amax || cycle_size_vns > AMAX) {
            return;
        }

        /*
            We enumerate Tanner cycles of length 2k, represented by their k
            variable nodes.

            Canonicalization:
            - the start variable s must be the smallest VN in the cycle;
            - at closure, require path_v[1] < path_v[k-1] to remove the
                two orientations.

            Duplicate VN supports are handled by the global layer-level
            deduplication in accept_candidate(). Avoiding a second local
            in-memory hash table here saves temporary memory during
            high-k cycle enumeration.
        */

        std::vector<int> path_v(static_cast<size_t>(k), -1);
        std::vector<int> path_c(static_cast<size_t>(k), -1);

        auto emit_cycle = [&]() {
            if (k >= 3) {
                if (!(path_v[1] < path_v[k - 1])) {
                    return;
                }
            }

            SupportT cyc;
            cyc.len = cycle_size_vns;

            for (int i = 0; i < k; ++i) {
                cyc.v[static_cast<size_t>(i)] = path_v[static_cast<size_t>(i)];
            }

            std::sort(cyc.v.begin(), cyc.v.begin() + cyc.len);
            emit(cyc);
        };

        std::function<void(int, int, int)> dfs =
            [&](int start, int current, int depth) {
                /*
                    depth = number of variable-to-variable steps already taken.
                    path_v[0], ..., path_v[depth] are valid.

                    If depth == k - 1, we have k distinct variable nodes and now
                    need one unused check that closes current back to start.
                */
                if (depth == k - 1) {
                    for (int pe = T.vn_ptr[current]; pe < T.vn_ptr[current + 1]; ++pe) {
                        const int c = T.vn_cn[pe];

                        if (check_on_path_[static_cast<size_t>(c)]) {
                            continue;
                        }

                        bool closes_to_start = false;

                        for (int q = T.cn_ptr[c]; q < T.cn_ptr[c + 1]; ++q) {
                            if (T.cn_vn[q] == start) {
                                closes_to_start = true;
                                break;
                            }
                        }

                        if (!closes_to_start) {
                            continue;
                        }

                        path_c[static_cast<size_t>(depth)] = c;
                        emit_cycle();
                        path_c[static_cast<size_t>(depth)] = -1;
                    }

                    return;
                }

                for (int pe = T.vn_ptr[current]; pe < T.vn_ptr[current + 1]; ++pe) {
                    const int c = T.vn_cn[pe];

                    if (check_on_path_[static_cast<size_t>(c)]) {
                        continue;
                    }

                    check_on_path_[static_cast<size_t>(c)] = 1;
                    path_c[static_cast<size_t>(depth)] = c;

                    for (int q = T.cn_ptr[c]; q < T.cn_ptr[c + 1]; ++q) {
                        const int next_v = T.cn_vn[q];

                        if (next_v == current) {
                            continue;
                        }

                        /*
                            Enforce start as the minimum VN in the cycle.
                            Returning to start is allowed only in the closing step,
                            handled above.
                        */
                        if (next_v <= start) {
                            continue;
                        }

                        if (!variable_degree_allowed(next_v)) {
                            continue;
                        }

                        if (var_on_path_[static_cast<size_t>(next_v)]) {
                            continue;
                        }

                        path_v[static_cast<size_t>(depth + 1)] = next_v;
                        var_on_path_[static_cast<size_t>(next_v)] = 1;

                        dfs(start, next_v, depth + 1);

                        var_on_path_[static_cast<size_t>(next_v)] = 0;
                        path_v[static_cast<size_t>(depth + 1)] = -1;
                    }

                    path_c[static_cast<size_t>(depth)] = -1;
                    check_on_path_[static_cast<size_t>(c)] = 0;
                }
            };

        for (int start = 0; start < T.N; ++start) {

            if (!variable_degree_allowed(start)) {
                continue;
            }

            if (T.variable_degree(start) < 2) {
                continue;
            }

            path_v[0] = start;
            var_on_path_[static_cast<size_t>(start)] = 1;

            dfs(start, start, 0);

            var_on_path_[static_cast<size_t>(start)] = 0;
            path_v[0] = -1;
        }
    }

    template <class Emit>
    void expand_dot(const SupportT& parent, Emit&& emit) {
        if (parent.len + 1 > params_.amax) {
            return;
        }

        build_check_degrees_for_parent(parent, parent_scratch_);

        mark_next_stamp();

        std::vector<int> candidates;
        candidates.reserve(64);

        /*
            A DOT child is obtained by adding one VN v outside S such that:
            - v touches at least two unsatisfied checks of G(S);
            - v touches no satisfied check of G(S).

            We generate candidates from the unsatisfied-check neighborhood only,
            instead of scanning all variables.
        */
        for (int c : parent_scratch_.touched) {
            const uint8_t d = parent_scratch_.deg[static_cast<size_t>(c)];

            if ((d & 1u) == 0) {
                continue;
            }

            for (int p = T.cn_ptr[c]; p < T.cn_ptr[c + 1]; ++p) {
                const int v = T.cn_vn[p];

                if (parent.contains(v)) {
                    continue;
                }

                if (!variable_degree_allowed(v)) {
                    continue;
                }

                uint32_t& mark = vn_mark_[static_cast<size_t>(v)];

                if (mark == vn_stamp_) {
                    continue;
                }

                mark = vn_stamp_;
                candidates.push_back(v);
            }
        }

        for (int v : candidates) {
            int n_unsat = 0;
            bool touches_satisfied = false;

            for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                const int c = T.vn_cn[p];
                const uint8_t d = parent_scratch_.deg[static_cast<size_t>(c)];

                if (d == 0) {
                    continue;
                }

                if (d & 1u) {
                    ++n_unsat;
                } else {
                    touches_satisfied = true;
                    break;
                }
            }

            if (touches_satisfied) {
                continue;
            }

            if (n_unsat < 2) {
                continue;
            }

            SupportT child = insert_one_sorted(parent, v);
            emit(child);
        }
    }

    template <class Emit>
    void expand_path(const SupportT& parent, uint8_t m, Emit&& emit) {
        /*
            Path expansion pa_m.

            We add m new VNs through a Tanner path

                c_start - x_1 - c_1 - x_2 - ... - c_{m-1} - x_m - c_end

            where:
            - c_start and c_end are distinct unsatisfied checks of G(parent);
            - x_i are new VNs outside parent;
            - intermediate checks c_1,...,c_{m-1} are outside G(parent);
            - no explicit path VN/check is repeated;
            - new VNs have degree 2 or 3.

            This is the direct Tanner-graph version of the MATLAB path expansion.
            It emits candidates one at a time and relies on accept_candidate() for
            exact b filtering, ETS filtering, and deduplication.
        */

        if (m < 2) {
            // pa_m in the dpl LETS search is normally used with m >= 2.
            // m = 1 is essentially a dot_2-type operation and would duplicate DOT.
            return;
        }

        if (parent.len + m > params_.amax) {
            return;
        }

        build_check_degrees_for_parent(parent, parent_scratch_);

        std::vector<int> unsat_checks;
        unsat_checks.reserve(parent_scratch_.touched.size());

        for (int c : parent_scratch_.touched) {
            const uint8_t d = parent_scratch_.deg[static_cast<size_t>(c)];
            if (d & 1u) {
                unsat_checks.push_back(c);
            }
        }

        if (unsat_checks.size() < 2) {
            return;
        }

        std::vector<int> path_vars(static_cast<size_t>(m), -1);
        std::vector<int> path_checks(static_cast<size_t>(m + 1), -1);

        auto active_deg = [&](int c) -> uint8_t {
            return parent_scratch_.deg[static_cast<size_t>(c)];
        };

        auto variable_has_forbidden_parent_check =
            [&](int v, int allowed_c1, int allowed_c2) -> bool {
                for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                    const int c = T.vn_cn[p];

                    if (c == allowed_c1 || c == allowed_c2) {
                        continue;
                    }

                    if (active_deg(c) > 0) {
                        return true;
                    }
                }

                return false;
            };

        auto emit_current_candidate = [&]() {
            std::vector<int> tmp;
            tmp.reserve(static_cast<size_t>(parent.len + m));

            for (uint8_t i = 0; i < parent.len; ++i) {
                tmp.push_back(parent.v[i]);
            }

            for (uint8_t i = 0; i < m; ++i) {
                tmp.push_back(path_vars[i]);
            }

            SupportT child = make_support_sorted_unique<AMAX>(tmp);

            // Safety: if the unique support is smaller than parent + m,
            // then some repeated/new-parent VN leaked in. Do not emit.
            if (child.len != parent.len + m) {
                return;
            }

            emit(child);
        };

        std::function<void(int, int)> dfs_from_check =
            [&](int current_check, int depth) {
                /*
                    current_check is path_checks[depth].
                    We now choose path_vars[depth].

                    If depth == m-1, then after choosing the last new variable,
                    we close to an unsatisfied end check of the parent.
                */

                for (int q = T.cn_ptr[current_check]; q < T.cn_ptr[current_check + 1]; ++q) {
                    const int v = T.cn_vn[q];

                    if (parent.contains(v)) {
                        continue;
                    }

                    if (var_on_path_[static_cast<size_t>(v)]) {
                        continue;
                    }

                    if (!variable_degree_allowed(v)) {
                        continue;
                    }

                    if (depth == static_cast<int>(m) - 1) {
                        /*
                            Final new variable. It must connect to a distinct
                            unsatisfied parent check c_end. It may touch the current
                            path check and c_end, but no other check in G(parent).
                        */
                        for (int pe = T.vn_ptr[v]; pe < T.vn_ptr[v + 1]; ++pe) {
                            const int c_end = T.vn_cn[pe];

                            if (c_end == current_check) {
                                continue;
                            }

                            if (check_on_path_[static_cast<size_t>(c_end)]) {
                                continue;
                            }

                            const uint8_t d_end = active_deg(c_end);

                            if (d_end == 0) {
                                continue;
                            }

                            if ((d_end & 1u) == 0) {
                                continue; // final endpoint must be unsatisfied
                            }

                            if (variable_has_forbidden_parent_check(v, current_check, c_end)) {
                                continue;
                            }

                            path_vars[static_cast<size_t>(depth)] = v;
                            path_checks[static_cast<size_t>(depth + 1)] = c_end;
                            var_on_path_[static_cast<size_t>(v)] = 1;

                            emit_current_candidate();

                            var_on_path_[static_cast<size_t>(v)] = 0;
                            path_checks[static_cast<size_t>(depth + 1)] = -1;
                            path_vars[static_cast<size_t>(depth)] = -1;
                        }

                        continue;
                    }

                    /*
                        Non-final new variable. It may touch current_check, but it
                        must not touch any other check already in G(parent).
                    */
                    if (variable_has_forbidden_parent_check(v, current_check, -1)) {
                        continue;
                    }

                    path_vars[static_cast<size_t>(depth)] = v;
                    var_on_path_[static_cast<size_t>(v)] = 1;

                    for (int pe = T.vn_ptr[v]; pe < T.vn_ptr[v + 1]; ++pe) {
                        const int c_next = T.vn_cn[pe];

                        if (c_next == current_check) {
                            continue;
                        }

                        if (active_deg(c_next) > 0) {
                            continue; // intermediate checks must be outside G(parent)
                        }

                        if (check_on_path_[static_cast<size_t>(c_next)]) {
                            continue;
                        }

                        path_checks[static_cast<size_t>(depth + 1)] = c_next;
                        check_on_path_[static_cast<size_t>(c_next)] = 1;

                        dfs_from_check(c_next, depth + 1);

                        check_on_path_[static_cast<size_t>(c_next)] = 0;
                        path_checks[static_cast<size_t>(depth + 1)] = -1;
                    }

                    var_on_path_[static_cast<size_t>(v)] = 0;
                    path_vars[static_cast<size_t>(depth)] = -1;
                }
            };

        for (int c_start : unsat_checks) {
            path_checks[0] = c_start;
            check_on_path_[static_cast<size_t>(c_start)] = 1;

            dfs_from_check(c_start, 0);

            check_on_path_[static_cast<size_t>(c_start)] = 0;
            path_checks[0] = -1;
        }
    }

    template <class Emit>
    void expand_lollipop(
        const SupportT& parent,
        uint8_t m,
        uint8_t c,
        const std::vector<SupportT>& cycles_c,
        Emit&& emit
    ) {
        /*
            Lollipop expansion lo^c_m.

            Parameters:
                m = total number of new VNs added
                c = cycle size in VNs
                d = (m + 1) - c

            Therefore:
                m = c + d - 1

            d == 1:
                Parent is attached directly to a detached c-cycle through exactly
                one common check, and that common check must be unsatisfied in
                the parent.

            d > 1:
                A stem with d-1 new VNs leaves the parent through an unsatisfied
                check and reaches a detached c-cycle through its last check.

            Candidate validation is still delegated to accept_candidate(), which
            checks exact b, elementary condition, and deduplication.
        */

        if (c == 0) return;
        if (m + 1 < c) return;

        const int d = static_cast<int>(m) + 1 - static_cast<int>(c);
        if (d < 1) return;

        if (parent.len + m > params_.amax) {
            return;
        }

        build_check_degrees_for_parent(parent, parent_scratch_);

        auto active_deg = [&](int chk) -> uint8_t {
            return parent_scratch_.deg[static_cast<size_t>(chk)];
        };

        auto collect_cycle_neighbor_checks =
            [&](const SupportT& C, std::vector<int>& out) {
                out.clear();

                for (uint8_t i = 0; i < C.len; ++i) {
                    const int v = C.v[i];

                    for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                        out.push_back(T.vn_cn[p]);
                    }
                }

                std::sort(out.begin(), out.end());
                out.erase(std::unique(out.begin(), out.end()), out.end());
            };

        auto support_disjoint = [&](const SupportT& A, const SupportT& B) -> bool {
            uint8_t i = 0;
            uint8_t j = 0;

            while (i < A.len && j < B.len) {
                if (A.v[i] == B.v[j]) return false;
                if (A.v[i] < B.v[j]) ++i;
                else ++j;
            }

            return true;
        };

        auto all_vn_degrees_allowed = [&](const SupportT& S) -> bool {
            for (uint8_t i = 0; i < S.len; ++i) {
                if (!variable_degree_allowed(S.v[i])) return false;
            }
            return true;
        };

        auto variable_has_forbidden_parent_check =
            [&](int v, int allowed_parent_check) -> bool {
                for (int p = T.vn_ptr[v]; p < T.vn_ptr[v + 1]; ++p) {
                    const int chk = T.vn_cn[p];

                    if (chk == allowed_parent_check) {
                        continue;
                    }

                    if (active_deg(chk) > 0) {
                        return true;
                    }
                }

                return false;
            };

        auto emit_union_candidate =
            [&](const SupportT& C, const std::vector<int>& stem_vars) {
                std::vector<int> tmp;
                tmp.reserve(static_cast<size_t>(parent.len + C.len + stem_vars.size()));

                for (uint8_t i = 0; i < parent.len; ++i) {
                    tmp.push_back(parent.v[i]);
                }

                for (uint8_t i = 0; i < C.len; ++i) {
                    tmp.push_back(C.v[i]);
                }

                for (int v : stem_vars) {
                    tmp.push_back(v);
                }

                SupportT child = make_support_sorted_unique<AMAX>(tmp);

                if (child.len != parent.len + m) {
                    return;
                }

                emit(child);
            };

        std::vector<int> cycle_checks;
        cycle_checks.reserve(32);

        // ---------------------------------------------------------------------
        // Case d == 1: direct attachment.
        // ---------------------------------------------------------------------
        if (d == 1) {
            for (const SupportT& C : cycles_c) {
                if (C.len != c) continue;
                if (!support_disjoint(parent, C)) continue;
                if (!all_vn_degrees_allowed(C)) continue;

                collect_cycle_neighbor_checks(C, cycle_checks);

                int common_count = 0;
                int common_check = -1;

                for (int chk : cycle_checks) {
                    if (active_deg(chk) > 0) {
                        ++common_count;
                        common_check = chk;
                    }
                }

                if (common_count != 1) {
                    continue;
                }

                if ((active_deg(common_check) & 1u) == 0) {
                    continue;
                }

                std::vector<int> no_stem;
                emit_union_candidate(C, no_stem);
            }

            return;
        }

        // ---------------------------------------------------------------------
        // Case d > 1: stem plus cycle.
        // ---------------------------------------------------------------------

        const int stem_var_count = d - 1;
        std::vector<int> stem_vars(static_cast<size_t>(stem_var_count), -1);
        std::vector<int> stem_checks(static_cast<size_t>(d), -1);

        for (const SupportT& C : cycles_c) {
            if (C.len != c) continue;
            if (!support_disjoint(parent, C)) continue;
            if (!all_vn_degrees_allowed(C)) continue;

            collect_cycle_neighbor_checks(C, cycle_checks);

            /*
                For d > 1, the cycle must not touch G(parent) directly.
                The only contact is through the final stem check.
            */
            bool cycle_touches_parent = false;
            for (int chk : cycle_checks) {
                if (active_deg(chk) > 0) {
                    cycle_touches_parent = true;
                    break;
                }
            }

            if (cycle_touches_parent) {
                continue;
            }

            for (int chk : cycle_checks) {
                check_on_cycle_[static_cast<size_t>(chk)] = 1;
            }

            std::function<void(int, int)> dfs_from_var =
                [&](int current_var, int depth_check) {
                    /*
                        Choose stem_checks[depth_check] adjacent to current_var.

                        depth_check == 0:
                            check must be unsatisfied in parent.

                        0 < depth_check < d-1:
                            check must be outside G(parent) and outside the cycle.

                        depth_check == d-1:
                            check must be outside G(parent) and adjacent to the
                            cycle. At that point the lollipop is complete.
                    */

                    for (int p = T.vn_ptr[current_var]; p < T.vn_ptr[current_var + 1]; ++p) {
                        const int chk = T.vn_cn[p];

                        if (check_on_path_[static_cast<size_t>(chk)]) {
                            continue;
                        }

                        const uint8_t pd = active_deg(chk);

                        if (depth_check == 0) {
                            if (pd == 0) continue;
                            if ((pd & 1u) == 0) continue;
                        } else {
                            if (pd > 0) continue;
                        }

                        if (depth_check < d - 1) {
                            if (check_on_cycle_[static_cast<size_t>(chk)]) {
                                continue;
                            }
                        } else {
                            if (!check_on_cycle_[static_cast<size_t>(chk)]) {
                                continue;
                            }
                        }

                        stem_checks[static_cast<size_t>(depth_check)] = chk;
                        check_on_path_[static_cast<size_t>(chk)] = 1;

                        if (depth_check == d - 1) {
                            emit_union_candidate(C, stem_vars);

                            check_on_path_[static_cast<size_t>(chk)] = 0;
                            stem_checks[static_cast<size_t>(depth_check)] = -1;
                            continue;
                        }

                        for (int q = T.cn_ptr[chk]; q < T.cn_ptr[chk + 1]; ++q) {
                            const int next_v = T.cn_vn[q];

                            if (next_v == current_var) {
                                continue;
                            }

                            if (parent.contains(next_v)) {
                                continue;
                            }

                            if (C.contains(next_v)) {
                                continue;
                            }

                            if (var_on_path_[static_cast<size_t>(next_v)]) {
                                continue;
                            }

                            if (!variable_degree_allowed(next_v)) {
                                continue;
                            }

                            const int allowed_parent_check = (pd > 0) ? chk : -1;

                            if (variable_has_forbidden_parent_check(next_v, allowed_parent_check)) {
                                continue;
                            }

                            const int stem_var_idx = depth_check;

                            stem_vars[static_cast<size_t>(stem_var_idx)] = next_v;
                            var_on_path_[static_cast<size_t>(next_v)] = 1;

                            dfs_from_var(next_v, depth_check + 1);

                            var_on_path_[static_cast<size_t>(next_v)] = 0;
                            stem_vars[static_cast<size_t>(stem_var_idx)] = -1;
                        }

                        check_on_path_[static_cast<size_t>(chk)] = 0;
                        stem_checks[static_cast<size_t>(depth_check)] = -1;
                    }
                };

            for (uint8_t i = 0; i < parent.len; ++i) {
                const int root_v = parent.v[i];
                dfs_from_var(root_v, 0);
            }

            for (int chk : cycle_checks) {
                check_on_cycle_[static_cast<size_t>(chk)] = 0;
            }
        }
    }
};

} // namespace ts_irreg23