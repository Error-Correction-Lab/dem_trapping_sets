#pragma once

#include <iostream>
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
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
#include <limits>
#include <queue>
#include <sstream>
#include <system_error>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>

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

struct ExpansionMenu {
    // MATLAB TOTEX{1,1}: path m values.
    std::vector<uint8_t> path_m;

    // MATLAB TOTEX{1,2}: rows [m, c] for lollipop.
    std::vector<std::pair<uint8_t, uint8_t>> lollipop_mc;
};

struct EnumeratorParams {
    uint8_t K = 0;          // maximum seed-cycle size, in variable nodes
    uint8_t g = 0;          // Tanner girth, even
    uint8_t amax = 0;       // maximum support size
    uint16_t b_aux_max = 0; // auxiliary b bound for intermediate structures

    // Optional Algorithm-1 class-dependent auxiliary bounds.
    // Indexed by support size a: bmax_by_a[a] = b^a_max.
    // If empty, the enumerator uses the old scalar b_aux_max and global
    // path/lollipop menus. If nonempty, DPL expansions are restricted by the
    // implicit expansion table EX(a,b) of Algorithm 1.
    std::vector<uint16_t> bmax_by_a;

    uint16_t b_target_min = 0;
    uint16_t b_target_max = 0;

    bool require_elementary = true;

    // If true, after the LETS/DPL stage, run the Section-VI ETSL stage.
    // This adds all ETSs whose normal graph has leaves via dot_1^k expansions.
    bool include_ets_with_leaves = false;

    // If true, the ETSL dot_1^k stage is initialized from all single VNs.
    // This gives the full ETSL2 singleton-seeded enumeration, but can be huge.
    // If false, dot_1^k starts only from DPL/LETS seeds.
    bool include_singleton_etsl_seeds = true;

    // If nonnegative, seed enumeration is restricted to canonical seed VNs
    // with index v < seed_vn_limit. For cycle seeds this means the canonical
    // cycle start variable. For singleton ETSL seeds this means the singleton VN.
    // Use this when variable indices are block ordered and you only want seeds
    // starting in the first spatial block. The default -1 disables the filter.
    int seed_vn_limit = -1;

    // If nonnegative, final target supports are counted/written/stored only
    // if they contain at least one VN with index v < final_anchor_vn_limit.
    // Because supports are sorted, this is equivalent to S.v[0] < limit.
    // Use this to keep only supports touching the first spatial block.
    // The default -1 disables the filter.
    int final_anchor_vn_limit = -1;

    // Allowed VN degrees for seed/expansion variables. The old implementation
    // was hard-coded to degrees {2,3}. To run on your detector, set this to
    // {2,3,4,5,6}. Use an empty vector to allow every positive VN degree.
    std::vector<uint8_t> allowed_variable_degrees = {2, 3};

    bool store_final_supports = false;

    // Memory knob for the online deduplication hash tables. Larger values use
    // fewer buckets and therefore less memory, at the cost of slower lookup.
    // Values in [1.5, 2.5] are usually reasonable for large enumerations.
    float hash_max_load_factor = 2.0f;

    // If true, seed cycles are cached for later lollipop expansions. This is
    // needed only when menu.lollipop_mc is non-empty. If false, lollipop
    // expansions are skipped and no separate cycle cache is kept.
    bool cache_cycles_for_lollipop = true;

    // Keep LETS seeds for the optional ETS-with-leaves stage only when needed.
    // This should remain false unless include_ets_with_leaves is true.
    bool cache_lets_for_etsl = false;

    // If non-empty, final target supports are streamed to disk instead of
    // only being optionally retained in result.final_supports. One CSV file
    // is created per (a,b) class: a_<a>_b_<b>.csv.
    std::string support_output_dir;

    bool verbose = false;

    // External-memory DPL mode. This keeps intermediate candidates/frontiers
    // in temporary binary files and removes those files at the end of run().
    // Final supports are still written only through support_output_dir CSVs.
    bool use_external_memory = true;

    // Number of hash buckets retained in each on-disk sort key. This keeps
    // records with the same support in the same ordered partition.
    uint32_t external_num_buckets = 512;

    // Batch size used when streaming cached cycles into lollipop expansions.
    uint32_t external_batch_size = 4096;

    // Total raw-candidate buffer budget across all external-memory workers.
    // It is divided by external_num_threads. Increasing the thread count
    // without increasing this value reduces each worker's run size and creates
    // more temporary files. The default uses roughly a few tens of MB for
    // typical AMAX values.
    uint64_t external_sort_run_records = 1000000;

    // Maximum number of sorted run files opened by one merge pass. Larger
    // values reduce the number of merge passes but consume more file handles.
    uint32_t external_merge_fan_in = 128;

    // Number of worker threads used for external-memory candidate generation.
    // A value of 0 selects std::thread::hardware_concurrency(). The target-a
    // layers and the final global merge remain sequential.
    uint32_t external_num_threads = 1;

    // Number of fixed-size parent records assigned to one worker task. Larger
    // values reduce scheduling overhead; smaller values improve load balance.
    uint64_t external_parent_chunk_records = 4096;

    // Maximum total size of cycle files loaded into a shared read-only cache
    // for one target-a layer. If the required files exceed this limit, workers
    // fall back to streaming the cycle files. Set to 0 to disable the cache.
    uint64_t external_cycle_cache_max_bytes = 512ull * 1024ull * 1024ull;

    // Preserve the temporary external-sort directory when an exception is
    // raised. This makes the exact failing run and its size inspectable.
    bool preserve_external_work_dir_on_error = true;
};

template <size_t AMAX>
struct EnumeratorResult {
    // Flattened count table: counts[b * (AMAX + 1) + a].
    std::vector<uint64_t> counts;

    // Optional. For very large enumerations, leave store_final_supports=false.
    std::vector<Support<AMAX>> final_supports;

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
    using SupportSet = std::unordered_set<SupportT, SupportHash<AMAX>>;
    using SupportsByAB = std::vector<std::vector<std::vector<SupportT>>>;
    using SeenByAB = std::vector<std::vector<SupportSet>>;

    TS_enumerator_irreg23(
        const TannerTopology& topo,
        ExpansionMenu menu,
        EnumeratorParams params
    )
        : T(topo),
        menu_(std::move(menu)),
        params_(params),
        scratch_(topo.M),
        parent_scratch_(topo.M),
        vn_mark_(static_cast<size_t>(topo.N), 0),
        var_on_path_(static_cast<size_t>(topo.N), 0),
        check_on_path_(static_cast<size_t>(topo.M), 0),
        check_on_cycle_(static_cast<size_t>(topo.M), 0)
    {
        // Online ETSL no longer needs a global LETS cache. Keep the user/binder
        // value of cache_lets_for_etsl unchanged; it is retained only for
        // backward compatibility and diagnostics.
        if (!use_algorithm1_table() && menu_.lollipop_mc.empty()) {
            params_.cache_cycles_for_lollipop = false;
        }

        validate_params();
        validate_allowed_variable_degrees();
        init_support_output();
    }

    EnumeratorResult<AMAX> run() {
        if (params_.use_external_memory) {
            return run_external_memory();
        }

        EnumeratorResult<AMAX> result;

        result.counts.assign(
            static_cast<size_t>(params_.b_aux_max + 1) * (AMAX + 1),
            0
        );

        const uint8_t k0 = static_cast<uint8_t>(params_.g / 2);
        const uint8_t maxk = std::min(params_.K, params_.amax);

        std::vector<std::vector<SupportT>> cycles_by_size(AMAX + 1);
        std::vector<std::vector<SupportT>> frontier(AMAX + 1);
        std::vector<SupportSet> seen_by_size(AMAX + 1);

        for (auto& seen : seen_by_size) {
            seen.max_load_factor(params_.hash_max_load_factor);
        }

        const bool need_lollipop_cache =
            use_algorithm1_table()
                ? algorithm1_table_has_lollipop()
                : !menu_.lollipop_mc.empty();

        const bool keep_cycle_cache =
            params_.cache_cycles_for_lollipop && need_lollipop_cache;

        if (!keep_cycle_cache && need_lollipop_cache) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: lollipop expansions are enabled but cycle cache is disabled"
            );
        }

        // ------------------------------------------------------------------
        // Online ETSL state.
        //
        // Old behavior:
        //   cache all LETS seeds in lets_by_ab, complete the full DPL search,
        //   then run dot_1^k over all sizes.
        //
        // New behavior:
        //   insert DPL LETS seeds directly into an online ETSL frontier and
        //   process/clear each ETSL layer as soon as the corresponding DPL
        //   size layer is complete.
        // ------------------------------------------------------------------
        SupportsByAB etsl_by_ab;
        SeenByAB seen_etsl_by_ab;

        std::function<void(const SupportT&, uint16_t)> etsl_seed_callback;
        std::function<void(const SupportT&, uint16_t)>* etsl_seed_callback_ptr = nullptr;

        if (params_.include_ets_with_leaves) {
            const uint16_t bmax = params_.b_target_max;

            etsl_by_ab.assign(
                AMAX + 1,
                std::vector<std::vector<SupportT>>(static_cast<size_t>(bmax + 1))
            );

            seen_etsl_by_ab.assign(
                AMAX + 1,
                std::vector<SupportSet>(static_cast<size_t>(bmax + 1))
            );

            for (auto& row : seen_etsl_by_ab) {
                for (auto& seen : row) {
                    seen.max_load_factor(params_.hash_max_load_factor);
                }
            }

            // DPL-generated accepted structures are LETSs. Only b >= 1 LETSs
            // can seed dot_1^k, since b = 0 has no unsatisfied check to attach to.
            etsl_seed_callback = [&](const SupportT& S, uint16_t b) {
                if (b == 0 || b > params_.b_target_max) {
                    return;
                }

                add_temp_etsl(S, b, etsl_by_ab, seen_etsl_by_ab);
            };

            etsl_seed_callback_ptr = &etsl_seed_callback;

            if (params_.verbose) {
                std::cout << "[ETSL] online mode enabled"
                          << " | bmax=" << params_.b_target_max
                          << std::endl;
            }

            // Algorithm 3 initialization for ETSL2: single variable nodes.
            // In a degree-2/3 code, a single VN is an elementary (1, deg(v)) TS.
            if (params_.include_singleton_etsl_seeds) {
                for (int v = 0; v < T.N; ++v) {
                    if (!seed_variable_allowed(v)) {
                        continue;
                    }

                    const int dv = T.variable_degree(v);

                    if (static_cast<uint16_t>(dv) > params_.b_target_max) {
                        continue;
                    }

                    SupportT S;
                    S.len = 1;
                    S.v[0] = v;

                    const uint16_t b = static_cast<uint16_t>(dv);

                    if (add_temp_etsl(S, b, etsl_by_ab, seen_etsl_by_ab)) {
                        add_final_if_target(S, b, result);
                    }
                }
            }
        }

        auto clear_etsl_layer = [&](uint8_t a_layer) {
            if (!params_.include_ets_with_leaves) {
                return;
            }

            for (uint16_t b = 0; b <= params_.b_target_max; ++b) {
                etsl_by_ab[a_layer][b].clear();
                etsl_by_ab[a_layer][b].shrink_to_fit();

                seen_etsl_by_ab[a_layer][b].clear();
                seen_etsl_by_ab[a_layer][b].rehash(0);
            }
        };

        auto process_etsl_layer = [&](uint8_t a_layer) {
            if (!params_.include_ets_with_leaves) {
                return;
            }

            if (a_layer == 0 || a_layer > params_.amax) {
                return;
            }

            if (params_.verbose) {
                size_t parents_total = 0;
                for (uint16_t b = 1; b <= params_.b_target_max; ++b) {
                    parents_total += etsl_by_ab[a_layer][b].size();
                }

                std::cout << "[ETSL] a=" << static_cast<int>(a_layer)
                          << " dot_1^k layer in progress"
                          << " | parents=" << parents_total
                          << "..."
                          << std::flush;
            }

            const uint64_t gen_before = result.generated_candidates;
            const uint64_t acc_before = result.accepted_intermediate;
            const uint64_t dup_before = result.duplicate_candidates;
            const uint64_t rb_before  = result.rejected_by_b;
            const uint64_t re_before  = result.rejected_non_elementary;

            if (a_layer < params_.amax) {
                for (uint16_t b = 1; b <= params_.b_target_max; ++b) {
                    auto& parents = etsl_by_ab[a_layer][b];

                    for (size_t pos = 0; pos < parents.size(); ++pos) {
                        const SupportT& parent = parents[pos];

                        expand_dot1(parent, b, [&](const SupportT& child, uint16_t child_b) {
                            ++result.generated_candidates;

                            const StructureStats st = induced_stats(T, child, scratch_);

                            if (params_.require_elementary && !st.elementary) {
                                ++result.rejected_non_elementary;
                                return;
                            }

                            if (st.b != child_b || st.b > params_.b_target_max) {
                                ++result.rejected_by_b;
                                return;
                            }

                            if (!add_temp_etsl(child, st.b, etsl_by_ab, seen_etsl_by_ab)) {
                                ++result.duplicate_candidates;
                                return;
                            }

                            ++result.accepted_intermediate;
                            add_final_if_target(child, st.b, result);
                        });
                    }
                }
            }

            clear_etsl_layer(a_layer);

            if (params_.verbose) {
                std::cout << " done"
                          << " | generated=" << (result.generated_candidates - gen_before)
                          << " | accepted=" << (result.accepted_intermediate - acc_before)
                          << " | duplicates=" << (result.duplicate_candidates - dup_before)
                          << " | rejected_b=" << (result.rejected_by_b - rb_before)
                          << " | rejected_non_ETS=" << (result.rejected_non_elementary - re_before)
                          << std::endl;
            }
        };

        if (params_.verbose) {
            std::cout << "[DPL] start search"
                    << " | g=" << static_cast<int>(params_.g)
                    << " | k0=" << static_cast<int>(k0)
                    << " | K=" << static_cast<int>(maxk)
                    << " | amax=" << static_cast<int>(params_.amax)
                    << " | b_aux_max=" << params_.b_aux_max
                    << " | algorithm1_table=" << (use_algorithm1_table() ? "yes" : "no")
                    << " | lollipop=" << (keep_cycle_cache ? "yes" : "no")
                    << " | online_etsl=" << (params_.include_ets_with_leaves ? "yes" : "no")
                    << " | seed_vn_limit=" << params_.seed_vn_limit
                    << " | final_anchor_vn_limit=" << params_.final_anchor_vn_limit
                    << std::endl;
        }

        // Before the first DPL size, complete any singleton-seeded ETSL
        // layers that are smaller than the minimum cycle size k0. This matters
        // for Tanner graphs with girth greater than four.
        if (params_.include_ets_with_leaves) {
            for (uint8_t a_layer = 1; a_layer < k0; ++a_layer) {
                process_etsl_layer(a_layer);
            }
        }

        // ------------------------------------------------------------------
        // Target-size staged DPL expansion.
        //
        // The previous implementation expanded every parent into all future
        // support sizes in one pass. For a=2, that immediately populated
        // frontier[4], ..., frontier[10] and their hash tables. Here we process
        // exactly one target support size at a time.
        // ------------------------------------------------------------------
        for (uint8_t target_a = k0; target_a <= params_.amax; ++target_a) {
            if (params_.verbose) {
                size_t parents_scanned = 0;
                for (uint8_t parent_a = k0; parent_a < target_a; ++parent_a) {
                    parents_scanned += frontier[parent_a].size();
                }

                std::cout << "[DPL] target a=" << static_cast<int>(target_a)
                          << " staged expansion in progress"
                          << " | existing=" << frontier[target_a].size()
                          << " | parents_scanned=" << parents_scanned
                          << "..."
                          << std::flush;
            }

            const uint64_t gen_before = result.generated_candidates;
            const uint64_t acc_before = result.accepted_intermediate;
            const uint64_t dup_before = result.duplicate_candidates;
            const uint64_t rb_before  = result.rejected_by_b;
            const uint64_t re_before  = result.rejected_non_elementary;

            // Generate cycle seeds of this exact size in the same target-a
            // layer as all DPL expansions that produce this size.
            if (target_a <= maxk) {
                enumerate_cycles(target_a, [&](const SupportT& cyc) {
                    const bool accepted = accept_candidate(
                        cyc,
                        frontier,
                        seen_by_size,
                        result,
                        etsl_seed_callback_ptr
                    );

                    // Lollipop expansions only need accepted, deduplicated
                    // cycle seeds from already completed smaller layers.
                    if (keep_cycle_cache && accepted) {
                        cycles_by_size[target_a].push_back(cyc);
                    }
                });
            }

            for (uint8_t parent_a = k0; parent_a < target_a; ++parent_a) {
                auto& parents = frontier[parent_a];

                if (parents.empty()) {
                    continue;
                }

                const uint8_t m_needed =
                    static_cast<uint8_t>(target_a - parent_a);

                for (size_t pos = 0; pos < parents.size(); ++pos) {
                    const SupportT& parent = parents[pos];

                    const StructureStats parent_st = induced_stats(T, parent, scratch_);

                    if (params_.require_elementary && !parent_st.elementary) {
                        continue;
                    }

                    if (parent_st.b > b_bound_for_size(parent_a)) {
                        continue;
                    }

                    const uint16_t parent_b = parent_st.b;

                    // b=0 ETSs are valid outputs but terminal under elementary DPL
                    // expansions. They have no unsatisfied degree-1 check where a
                    // connected elementary child can attach.
                    if (parent_b == 0) {
                        continue;
                    }

                    if (use_algorithm1_table()) {
                        // --------------------------------------------------
                        // Exact Algorithm-1 expansion-table mode.
                        // Allowed expansions depend on the parent class
                        // (parent_a, parent_b).
                        // --------------------------------------------------

                        // DOT contributes exactly one new VN.
                        if (m_needed == 1 && ex_allows_dot(parent_a, parent_b)) {
                            expand_dot(parent, [&](const SupportT& child) {
                                accept_candidate(
                                    child,
                                    frontier,
                                    seen_by_size,
                                    result,
                                    etsl_seed_callback_ptr
                                );
                            });
                        }

                        // Path expansion pa_m contributes exactly m new VNs.
                        if (ex_allows_path(parent_a, parent_b, m_needed)) {
                            expand_path(parent, m_needed, [&](const SupportT& child) {
                                accept_candidate(
                                    child,
                                    frontier,
                                    seen_by_size,
                                    result,
                                    etsl_seed_callback_ptr
                                );
                            });
                        }

                        // Lollipop lo_m^c contributes exactly m new VNs.
                        if (ex_allows_lollipop(parent_a, parent_b, m_needed)) {
                            const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);

                            for (uint8_t c = g2; c <= m_needed; ++c) {
                                if (!keep_cycle_cache) continue;
                                if (c > AMAX) continue;
                                if (cycles_by_size[c].empty()) continue;

                                expand_lollipop(
                                    parent,
                                    m_needed,
                                    c,
                                    cycles_by_size[c],
                                    [&](const SupportT& child) {
                                        accept_candidate(
                                            child,
                                            frontier,
                                            seen_by_size,
                                            result,
                                            etsl_seed_callback_ptr
                                        );
                                    }
                                );
                            }
                        }
                    } else {
                        // --------------------------------------------------
                        // Backward-compatible global-menu mode.
                        // --------------------------------------------------

                        // DOT contributes exactly one new VN.
                        if (m_needed == 1) {
                            expand_dot(parent, [&](const SupportT& child) {
                                accept_candidate(
                                    child,
                                    frontier,
                                    seen_by_size,
                                    result,
                                    etsl_seed_callback_ptr
                                );
                            });
                        }

                        // Path expansion pa_m contributes exactly m new VNs.
                        for (uint8_t m : menu_.path_m) {
                            if (m != m_needed) {
                                continue;
                            }

                            expand_path(parent, m, [&](const SupportT& child) {
                                accept_candidate(
                                    child,
                                    frontier,
                                    seen_by_size,
                                    result,
                                    etsl_seed_callback_ptr
                                );
                            });
                        }

                        // Lollipop lo_m^c contributes exactly m new VNs.
                        for (const auto& mc : menu_.lollipop_mc) {
                            const uint8_t m = mc.first;
                            const uint8_t c = mc.second;

                            if (!keep_cycle_cache) continue;
                            if (m != m_needed) continue;
                            if (c > AMAX) continue;
                            if (cycles_by_size[c].empty()) continue;

                            expand_lollipop(
                                parent,
                                m,
                                c,
                                cycles_by_size[c],
                                [&](const SupportT& child) {
                                    accept_candidate(
                                        child,
                                        frontier,
                                        seen_by_size,
                                        result,
                                        etsl_seed_callback_ptr
                                    );
                                }
                            );
                        }
                    }
                }
            }

            if (params_.verbose) {
                std::cout << " done"
                        << " | generated=" << (result.generated_candidates - gen_before)
                        << " | accepted=" << (result.accepted_intermediate - acc_before)
                        << " | duplicates=" << (result.duplicate_candidates - dup_before)
                        << " | rejected_b=" << (result.rejected_by_b - rb_before)
                        << " | rejected_non_ETS=" << (result.rejected_non_elementary - re_before)
                        << " | total_a=" << frontier[target_a].size()
                        << std::endl;
            }

            // The DPL layer target_a is now complete. Therefore the corresponding
            // ETSL layer has also received all DPL LETS seeds of this size.
            process_etsl_layer(target_a);

            // No later staged expansion can produce support size target_a again.
            // Keep frontier[target_a] for use as parents in larger target layers,
            // but release its deduplication table.
            seen_by_size[target_a].clear();
            seen_by_size[target_a].rehash(0);
        }

        for (auto& layer : frontier) {
            layer.clear();
            layer.shrink_to_fit();
        }

        for (auto& seen : seen_by_size) {
            seen.clear();
            seen.rehash(0);
        }

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

        flush_support_output_streams();
        return result;
    }

private:
    const TannerTopology& T;
    ExpansionMenu menu_;
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
        if (params_.external_num_buckets == 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_num_buckets must be positive"
            );
        }

        if (params_.external_batch_size == 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_batch_size must be positive"
            );
        }

        if (params_.external_sort_run_records == 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_sort_run_records must be positive"
            );
        }

        if (params_.external_sort_run_records >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_sort_run_records exceeds size_t"
            );
        }

        if (params_.external_merge_fan_in < 2) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_merge_fan_in must be at least 2"
            );
        }

        if (params_.external_parent_chunk_records == 0) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: external_parent_chunk_records must be positive"
            );
        }

        EnumeratorResult<AMAX> result;
        result.counts.assign(
            static_cast<size_t>(params_.b_aux_max + 1) * (AMAX + 1),
            0
        );

        const uint8_t k0 = static_cast<uint8_t>(params_.g / 2);
        const uint8_t maxk = std::min(params_.K, params_.amax);
        const uint32_t num_buckets = params_.external_num_buckets;
        const size_t cycle_batch_size = params_.external_batch_size;
        const size_t sort_run_records =
            static_cast<size_t>(params_.external_sort_run_records);
        const size_t merge_fan_in =
            static_cast<size_t>(params_.external_merge_fan_in);

        uint32_t configured_threads = params_.external_num_threads;
        if (configured_threads == 0) {
            configured_threads = std::thread::hardware_concurrency();
            if (configured_threads == 0) {
                configured_threads = 1;
            }
        }

        // external_sort_run_records remains a TOTAL memory budget. Dividing it
        // among workers prevents candidate-buffer RAM from growing linearly
        // with the number of threads.
        const size_t sort_run_records_per_worker = std::max<size_t>(
            1,
            sort_run_records / static_cast<size_t>(configured_threads)
        );

        const size_t parent_chunk_records = static_cast<size_t>(
            std::min<uint64_t>(
                params_.external_parent_chunk_records,
                static_cast<uint64_t>(std::numeric_limits<size_t>::max())
            )
        );

        const std::filesystem::path work_dir = make_external_work_dir();

        // In external-memory mode the ETS-with-leaves stage is also
        // disk-backed. This callback is assigned after the external run/merge
        // helpers have been constructed and is invoked for every accepted LETS.
        std::function<void(const SupportT&, uint16_t)> add_lets_to_etsl_layer =
            [](const SupportT&, uint16_t) {};

        try {
            if (params_.verbose) {
                std::cout << "[DPL-ext] start search"
                          << " | g=" << static_cast<int>(params_.g)
                          << " | k0=" << static_cast<int>(k0)
                          << " | K=" << static_cast<int>(maxk)
                          << " | amax=" << static_cast<int>(params_.amax)
                          << " | b_aux_max=" << params_.b_aux_max
                          << " | algorithm1_table=" << (use_algorithm1_table() ? "yes" : "no")
                          << " | online_etsl="
                          << (params_.include_ets_with_leaves ? "yes" : "no")
                          << " | singleton_etsl_seeds="
                          << (params_.include_singleton_etsl_seeds ? "yes" : "no")
                          << " | buckets=" << num_buckets
                          << " | cycle_batch_size=" << cycle_batch_size
                          << " | sort_run_records_total=" << sort_run_records
                          << " | sort_buffer_bytes_total="
                          << (static_cast<uint64_t>(sort_run_records) * sizeof(PendingRecord))
                          << " | sort_run_records_per_worker=" << sort_run_records_per_worker
                          << " | configured_threads=" << configured_threads
                          << " | parent_chunk_records=" << parent_chunk_records
                          << " | cycle_cache_max_bytes=" << params_.external_cycle_cache_max_bytes
                          << " | merge_fan_in=" << merge_fan_in
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

                        if (params_.require_elementary && !st.elementary) {
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

                        const bool is_target =
                            (st.b >= params_.b_target_min) &&
                            (st.b <= params_.b_target_max);

                        if (is_target && final_support_allowed(S)) {
                            increment_count(result, st.b, S.len);
                            write_support_if_requested(S, st.b, result);

                            if (params_.store_final_supports) {
                                result.final_supports.push_back(S);
                            }
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
            pending_buffer.reserve(sort_run_records_per_worker);
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
            // candidates use worker-local buffers below.
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

                if (pending_buffer.size() >= sort_run_records_per_worker) {
                    flush_pending_run();
                }
            };

            struct ParentChunkTask {
                std::filesystem::path path;
                uint8_t parent_a = 0;
                uint16_t parent_b = 0;
                uint64_t first_record = 0;
                uint64_t record_count = 0;
            };

            constexpr uint64_t support_record_bytes =
                sizeof(uint8_t) + AMAX * sizeof(int32_t);

            auto make_parent_tasks = [&](uint8_t target_a) {
                std::vector<ParentChunkTask> tasks;

                for (uint8_t parent_a = k0; parent_a < target_a; ++parent_a) {
                    const uint16_t bmax_parent = b_bound_for_size(parent_a);

                    for (uint16_t parent_b = 1; parent_b <= bmax_parent; ++parent_b) {
                        const std::filesystem::path fpath =
                            frontier_file_path(work_dir, parent_a, parent_b);

                        std::error_code ec;
                        if (!std::filesystem::exists(fpath, ec)) {
                            if (ec) {
                                throw std::runtime_error(
                                    "make_parent_tasks: exists failed for " +
                                    fpath.string() + ": " + ec.message()
                                );
                            }
                            continue;
                        }

                        const uint64_t bytes = std::filesystem::file_size(fpath, ec);
                        if (ec) {
                            throw std::runtime_error(
                                "make_parent_tasks: file_size failed for " +
                                fpath.string() + ": " + ec.message()
                            );
                        }

                        if (bytes % support_record_bytes != 0) {
                            throw std::runtime_error(
                                "make_parent_tasks: malformed frontier file " +
                                fpath.string()
                            );
                        }

                        const uint64_t records = bytes / support_record_bytes;
                        for (uint64_t first = 0; first < records; ) {
                            const uint64_t count = std::min<uint64_t>(
                                records - first,
                                static_cast<uint64_t>(parent_chunk_records)
                            );

                            tasks.push_back(ParentChunkTask{
                                fpath,
                                parent_a,
                                parent_b,
                                first,
                                count
                            });
                            first += count;
                        }
                    }
                }

                return tasks;
            };

            auto read_parent_chunk = [&](const ParentChunkTask& task, auto&& emit) {
                errno = 0;
                std::ifstream is(task.path, std::ios::in | std::ios::binary);
                if (!is) {
                    const int saved_errno = errno;
                    throw std::runtime_error(
                        io_failure_message(
                            "read_parent_chunk: failed to open",
                            task.path,
                            saved_errno
                        )
                    );
                }

                const uint64_t byte_offset = task.first_record * support_record_bytes;
                if (byte_offset > static_cast<uint64_t>(
                        std::numeric_limits<std::streamoff>::max())) {
                    throw std::runtime_error(
                        "read_parent_chunk: byte offset exceeds streamoff"
                    );
                }

                is.seekg(static_cast<std::streamoff>(byte_offset), std::ios::beg);
                if (!is) {
                    throw std::runtime_error(
                        "read_parent_chunk: seek failed for " + task.path.string()
                    );
                }

                SupportT parent;
                for (uint64_t i = 0; i < task.record_count; ++i) {
                    if (!read_support_binary(is, parent)) {
                        throw std::runtime_error(
                            "read_parent_chunk: unexpected EOF in " + task.path.string()
                        );
                    }
                    emit(parent);
                }
            };

            struct CycleCache {
                std::vector<std::vector<SupportT>> by_size;
                std::vector<uint8_t> loaded;
                uint64_t bytes = 0;
                bool enabled = false;

                CycleCache()
                    : by_size(AMAX + 1), loaded(AMAX + 1, 0)
                {}
            };

            auto build_cycle_cache = [&](uint8_t target_a) {
                CycleCache cache;

                if (params_.external_cycle_cache_max_bytes == 0) {
                    return cache;
                }

                std::vector<uint8_t> needed(AMAX + 1, 0);
                const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);

                for (uint8_t parent_a = k0; parent_a < target_a; ++parent_a) {
                    const uint8_t m_needed =
                        static_cast<uint8_t>(target_a - parent_a);
                    const uint16_t bmax_parent = b_bound_for_size(parent_a);

                    for (uint16_t parent_b = 1; parent_b <= bmax_parent; ++parent_b) {
                        if (use_algorithm1_table()) {
                            if (ex_allows_lollipop(parent_a, parent_b, m_needed)) {
                                for (uint8_t c = g2; c <= m_needed && c <= AMAX; ++c) {
                                    needed[c] = 1;
                                }
                            }
                        } else {
                            for (const auto& mc : menu_.lollipop_mc) {
                                if (mc.first == m_needed && mc.second <= AMAX) {
                                    needed[mc.second] = 1;
                                }
                            }
                        }
                    }
                }

                uint64_t required_bytes = 0;
                for (uint8_t c = 0; c <= AMAX; ++c) {
                    if (!needed[c]) {
                        continue;
                    }

                    const auto path = cycle_file_path(work_dir, c);
                    std::error_code ec;
                    if (!std::filesystem::exists(path, ec)) {
                        if (ec) {
                            throw std::runtime_error(
                                "build_cycle_cache: exists failed for " +
                                path.string() + ": " + ec.message()
                            );
                        }
                        continue;
                    }

                    const uint64_t bytes = std::filesystem::file_size(path, ec);
                    if (ec) {
                        throw std::runtime_error(
                            "build_cycle_cache: file_size failed for " +
                            path.string() + ": " + ec.message()
                        );
                    }

                    if (bytes % support_record_bytes != 0) {
                        throw std::runtime_error(
                            "build_cycle_cache: malformed cycle file " + path.string()
                        );
                    }

                    const uint64_t records = bytes / support_record_bytes;
                    if (
                        records >
                        std::numeric_limits<uint64_t>::max() / sizeof(SupportT)
                    ) {
                        return cache;
                    }
                    const uint64_t memory_bytes = records * sizeof(SupportT);

                    if (
                        memory_bytes > params_.external_cycle_cache_max_bytes ||
                        required_bytes >
                            params_.external_cycle_cache_max_bytes - memory_bytes
                    ) {
                        return cache;
                    }
                    required_bytes += memory_bytes;
                }

                if (required_bytes > params_.external_cycle_cache_max_bytes) {
                    return cache;
                }

                for (uint8_t c = 0; c <= AMAX; ++c) {
                    if (!needed[c]) {
                        continue;
                    }

                    const auto path = cycle_file_path(work_dir, c);
                    if (!std::filesystem::exists(path)) {
                        continue;
                    }

                    const uint64_t bytes = std::filesystem::file_size(path);
                    if (bytes % support_record_bytes != 0) {
                        throw std::runtime_error(
                            "build_cycle_cache: malformed cycle file " + path.string()
                        );
                    }

                    cache.by_size[c].reserve(
                        static_cast<size_t>(bytes / support_record_bytes)
                    );
                    stream_support_file(path, [&](const SupportT& S) {
                        cache.by_size[c].push_back(S);
                    });
                    cache.loaded[c] = 1;
                }

                cache.bytes = required_bytes;
                cache.enabled = true;
                return cache;
            };

            struct WorkerTotals {
                uint64_t generated = 0;
                uint64_t raw_records = 0;
                uint64_t rejected_by_size = 0;
                std::vector<std::filesystem::path> run_paths;
            };

            auto generate_parent_candidates_parallel = [&](
                uint8_t target_a,
                const std::vector<ParentChunkTask>& tasks,
                const CycleCache& cycle_cache
            ) {
                if (tasks.empty()) {
                    return;
                }

                const uint32_t num_workers = static_cast<uint32_t>(
                    std::min<size_t>(configured_threads, tasks.size())
                );

                std::atomic<size_t> next_task{0};
                std::atomic<uint64_t> next_run_index{pending_run_index};
                std::atomic<bool> stop{false};
                std::mutex error_mutex;
                std::exception_ptr first_error;

                std::vector<WorkerTotals> totals(num_workers);
                std::vector<std::thread> workers;
                workers.reserve(num_workers);

                for (uint32_t worker_id = 0; worker_id < num_workers; ++worker_id) {
                    workers.emplace_back([&, worker_id]() {
                        try {
                            EnumeratorParams worker_params = params_;
                            worker_params.support_output_dir.clear();
                            worker_params.store_final_supports = false;
                            worker_params.verbose = false;
                            worker_params.use_external_memory = false;
                            worker_params.include_ets_with_leaves = false;
                            worker_params.external_num_threads = 1;

                            TS_enumerator_irreg23<AMAX> worker(
                                T,
                                menu_,
                                std::move(worker_params)
                            );

                            WorkerTotals& wt = totals[worker_id];
                            std::vector<PendingRecord> local_buffer;
                            local_buffer.reserve(sort_run_records_per_worker);

                            auto flush_local_run = [&]() {
                                if (local_buffer.empty()) {
                                    return;
                                }

                                std::sort(
                                    local_buffer.begin(),
                                    local_buffer.end(),
                                    pending_record_less
                                );

                                size_t out = 0;
                                size_t i = 0;
                                while (i < local_buffer.size()) {
                                    PendingRecord merged = local_buffer[i];
                                    size_t j = i + 1;

                                    while (
                                        j < local_buffer.size() &&
                                        pending_record_equal(merged, local_buffer[j])
                                    ) {
                                        merged.cand.flags |= local_buffer[j].cand.flags;
                                        ++j;
                                    }

                                    local_buffer[out++] = merged;
                                    i = j;
                                }

                                const uint64_t run_index =
                                    next_run_index.fetch_add(1, std::memory_order_relaxed);
                                const auto path = pending_run_path(
                                    work_dir,
                                    target_a,
                                    0,
                                    run_index
                                );
                                write_run_file(path, local_buffer, out);
                                wt.run_paths.push_back(path);
                                local_buffer.clear();
                            };

                            auto append_local = [&](const SupportT& S, uint8_t flags) {
                                if (S.len == 0 || S.len > params_.amax) {
                                    ++wt.rejected_by_size;
                                    return;
                                }

                                ++wt.generated;
                                ++wt.raw_records;

                                PendingRecord rec;
                                rec.bucket = static_cast<uint32_t>(
                                    SupportHash<AMAX>{}(S) % num_buckets
                                );
                                rec.cand.S = S;
                                rec.cand.flags = flags;
                                local_buffer.push_back(rec);

                                if (local_buffer.size() >= sort_run_records_per_worker) {
                                    flush_local_run();
                                }
                            };

                            auto expand_parent_to_target_local = [&](
                                const SupportT& parent,
                                uint8_t parent_a,
                                uint16_t parent_b
                            ) {
                                const uint8_t m_needed =
                                    static_cast<uint8_t>(target_a - parent_a);

                                const StructureStats parent_st = induced_stats(
                                    T,
                                    parent,
                                    worker.parent_scratch_
                                );

                                if (params_.require_elementary && !parent_st.elementary) {
                                    return;
                                }

                                if (parent_st.b != parent_b) {
                                    return;
                                }

                                if (
                                    parent_b == 0 ||
                                    parent_b > worker.b_bound_for_size(parent_a)
                                ) {
                                    return;
                                }

                                auto run_lollipop = [&](
                                    uint8_t m,
                                    uint8_t c
                                ) {
                                    if (
                                        c <= AMAX &&
                                        cycle_cache.enabled &&
                                        cycle_cache.loaded[c]
                                    ) {
                                        worker.expand_lollipop(
                                            parent,
                                            m,
                                            c,
                                            cycle_cache.by_size[c],
                                            [&](const SupportT& child) {
                                                append_local(child, 0u);
                                            }
                                        );
                                        return;
                                    }

                                    const auto cpath = cycle_file_path(work_dir, c);
                                    worker.stream_support_file_batches(
                                        cpath,
                                        cycle_batch_size,
                                        [&](const std::vector<SupportT>& cycles_batch) {
                                            worker.expand_lollipop(
                                                parent,
                                                m,
                                                c,
                                                cycles_batch,
                                                [&](const SupportT& child) {
                                                    append_local(child, 0u);
                                                }
                                            );
                                        }
                                    );
                                };

                                if (worker.use_algorithm1_table()) {
                                    if (
                                        m_needed == 1 &&
                                        worker.ex_allows_dot(parent_a, parent_b)
                                    ) {
                                        worker.expand_dot(parent, [&](const SupportT& child) {
                                            append_local(child, 0u);
                                        });
                                    }

                                    if (worker.ex_allows_path(parent_a, parent_b, m_needed)) {
                                        worker.expand_path(
                                            parent,
                                            m_needed,
                                            [&](const SupportT& child) {
                                                append_local(child, 0u);
                                            }
                                        );
                                    }

                                    if (
                                        worker.ex_allows_lollipop(
                                            parent_a,
                                            parent_b,
                                            m_needed
                                        )
                                    ) {
                                        const uint8_t g2 =
                                            static_cast<uint8_t>(params_.g / 2);
                                        for (uint8_t c = g2; c <= m_needed; ++c) {
                                            run_lollipop(m_needed, c);
                                        }
                                    }
                                } else {
                                    if (m_needed == 1) {
                                        worker.expand_dot(parent, [&](const SupportT& child) {
                                            append_local(child, 0u);
                                        });
                                    }

                                    for (uint8_t m : menu_.path_m) {
                                        if (m == m_needed) {
                                            worker.expand_path(
                                                parent,
                                                m,
                                                [&](const SupportT& child) {
                                                    append_local(child, 0u);
                                                }
                                            );
                                        }
                                    }

                                    for (const auto& mc : menu_.lollipop_mc) {
                                        if (mc.first == m_needed) {
                                            run_lollipop(mc.first, mc.second);
                                        }
                                    }
                                }
                            };

                            while (!stop.load(std::memory_order_relaxed)) {
                                const size_t task_index =
                                    next_task.fetch_add(1, std::memory_order_relaxed);
                                if (task_index >= tasks.size()) {
                                    break;
                                }

                                const ParentChunkTask& task = tasks[task_index];
                                read_parent_chunk(task, [&](const SupportT& parent) {
                                    expand_parent_to_target_local(
                                        parent,
                                        task.parent_a,
                                        task.parent_b
                                    );
                                });
                            }

                            flush_local_run();
                        } catch (...) {
                            stop.store(true, std::memory_order_relaxed);
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (!first_error) {
                                first_error = std::current_exception();
                            }
                        }
                    });
                }

                for (auto& worker : workers) {
                    worker.join();
                }

                if (first_error) {
                    std::rethrow_exception(first_error);
                }

                pending_run_index = next_run_index.load(std::memory_order_relaxed);

                for (auto& wt : totals) {
                    result.generated_candidates += wt.generated;
                    result.rejected_by_size += wt.rejected_by_size;
                    pending_raw_records += wt.raw_records;

                    pending_runs.insert(
                        pending_runs.end(),
                        std::make_move_iterator(wt.run_paths.begin()),
                        std::make_move_iterator(wt.run_paths.end())
                    );
                }
            };

            // --------------------------------------------------------------
            // Fully external-memory ETS-with-leaves (dot_1^k) stage.
            //
            // The previous implementation kept each ETSL frontier and an
            // unordered_set copy in RAM. Here, LETS seeds and dot_1^k children
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

                // Release the capacity so the ETSL serial buffer does not
                // remain resident while DPL worker buffers are active.
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

                if (etsl_pending_buffer.size() >= sort_run_records_per_worker) {
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

                        if (params_.require_elementary && !st.elementary) {
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

            auto make_etsl_parent_tasks = [&](uint8_t a_layer) {
                std::vector<ParentChunkTask> tasks;

                for (uint16_t b = 1; b <= params_.b_target_max; ++b) {
                    const auto path = etsl_frontier_file_path(work_dir, a_layer, b);
                    std::error_code ec;
                    if (!std::filesystem::exists(path, ec)) {
                        if (ec) {
                            throw std::runtime_error(
                                "make_etsl_parent_tasks: exists failed for " +
                                path.string() + ": " + ec.message()
                            );
                        }
                        continue;
                    }

                    const uint64_t bytes = std::filesystem::file_size(path, ec);
                    if (ec) {
                        throw std::runtime_error(
                            "make_etsl_parent_tasks: file_size failed for " +
                            path.string() + ": " + ec.message()
                        );
                    }
                    if (bytes % support_record_bytes != 0) {
                        throw std::runtime_error(
                            "make_etsl_parent_tasks: malformed frontier file " +
                            path.string()
                        );
                    }

                    const uint64_t records = bytes / support_record_bytes;
                    for (uint64_t first = 0; first < records; ) {
                        const uint64_t count = std::min<uint64_t>(
                            records - first,
                            static_cast<uint64_t>(parent_chunk_records)
                        );
                        tasks.push_back(ParentChunkTask{
                            path,
                            a_layer,
                            b,
                            first,
                            count
                        });
                        first += count;
                    }
                }

                return tasks;
            };

            auto generate_etsl_dot1_candidates_parallel = [&](
                uint8_t a_layer,
                const std::vector<ParentChunkTask>& tasks
            ) {
                if (!params_.include_ets_with_leaves || a_layer >= params_.amax) {
                    return;
                }

                const uint8_t target_a = static_cast<uint8_t>(a_layer + 1);
                begin_etsl_pending_target(target_a);

                if (tasks.empty()) {
                    return;
                }

                // Serial ETSL candidates should not coexist with worker buffers.
                flush_etsl_pending_run();

                const uint32_t num_workers = static_cast<uint32_t>(
                    std::min<size_t>(configured_threads, tasks.size())
                );

                std::atomic<size_t> next_task{0};
                std::atomic<uint64_t> next_run_index{etsl_pending_run_index};
                std::atomic<bool> stop{false};
                std::mutex error_mutex;
                std::exception_ptr first_error;

                std::vector<WorkerTotals> totals(num_workers);
                std::vector<std::thread> workers;
                workers.reserve(num_workers);

                for (uint32_t worker_id = 0; worker_id < num_workers; ++worker_id) {
                    workers.emplace_back([&, worker_id]() {
                        try {
                            EnumeratorParams worker_params = params_;
                            worker_params.support_output_dir.clear();
                            worker_params.store_final_supports = false;
                            worker_params.verbose = false;
                            worker_params.use_external_memory = false;
                            worker_params.include_ets_with_leaves = false;
                            worker_params.external_num_threads = 1;

                            TS_enumerator_irreg23<AMAX> worker(
                                T,
                                menu_,
                                std::move(worker_params)
                            );

                            WorkerTotals& wt = totals[worker_id];
                            std::vector<PendingRecord> local_buffer;
                            local_buffer.reserve(sort_run_records_per_worker);

                            auto flush_local_run = [&]() {
                                if (local_buffer.empty()) {
                                    return;
                                }

                                std::sort(
                                    local_buffer.begin(),
                                    local_buffer.end(),
                                    pending_record_less
                                );

                                size_t out = 0;
                                size_t i = 0;
                                while (i < local_buffer.size()) {
                                    PendingRecord merged = local_buffer[i];
                                    size_t j = i + 1;
                                    while (
                                        j < local_buffer.size() &&
                                        pending_record_equal(merged, local_buffer[j])
                                    ) {
                                        merged.cand.flags |= local_buffer[j].cand.flags;
                                        ++j;
                                    }
                                    local_buffer[out++] = merged;
                                    i = j;
                                }

                                const uint64_t run_index =
                                    next_run_index.fetch_add(1, std::memory_order_relaxed);
                                const auto path = etsl_pending_run_path(
                                    work_dir,
                                    target_a,
                                    0,
                                    run_index
                                );
                                write_run_file(path, local_buffer, out);
                                wt.run_paths.push_back(path);
                                local_buffer.clear();
                            };

                            auto append_local = [&](const SupportT& S) {
                                if (S.len == 0 || S.len > params_.amax) {
                                    ++wt.rejected_by_size;
                                    return;
                                }

                                ++wt.generated;
                                ++wt.raw_records;

                                PendingRecord rec;
                                rec.bucket = static_cast<uint32_t>(
                                    SupportHash<AMAX>{}(S) % num_buckets
                                );
                                rec.cand.S = S;
                                rec.cand.flags = CAND_FLAG_ETSL_DOT1;
                                local_buffer.push_back(rec);

                                if (local_buffer.size() >= sort_run_records_per_worker) {
                                    flush_local_run();
                                }
                            };

                            while (!stop.load(std::memory_order_relaxed)) {
                                const size_t task_index =
                                    next_task.fetch_add(1, std::memory_order_relaxed);
                                if (task_index >= tasks.size()) {
                                    break;
                                }

                                const ParentChunkTask& task = tasks[task_index];
                                read_parent_chunk(task, [&](const SupportT& parent) {
                                    worker.expand_dot1(
                                        parent,
                                        task.parent_b,
                                        [&](const SupportT& child, uint16_t) {
                                            append_local(child);
                                        }
                                    );
                                });
                            }

                            flush_local_run();
                        } catch (...) {
                            stop.store(true, std::memory_order_relaxed);
                            std::lock_guard<std::mutex> lock(error_mutex);
                            if (!first_error) {
                                first_error = std::current_exception();
                            }
                        }
                    });
                }

                for (auto& worker : workers) {
                    worker.join();
                }

                if (first_error) {
                    std::rethrow_exception(first_error);
                }

                etsl_pending_run_index =
                    next_run_index.load(std::memory_order_relaxed);

                for (auto& wt : totals) {
                    result.generated_candidates += wt.generated;
                    result.rejected_by_size += wt.rejected_by_size;
                    etsl_pending_raw_records += wt.raw_records;
                    etsl_pending_runs.insert(
                        etsl_pending_runs.end(),
                        std::make_move_iterator(wt.run_paths.begin()),
                        std::make_move_iterator(wt.run_paths.end())
                    );
                }
            };

            auto process_external_etsl_layer = [&](uint8_t a_layer) {
                if (
                    !params_.include_ets_with_leaves ||
                    a_layer == 0 ||
                    a_layer > params_.amax
                ) {
                    return;
                }

                const auto tasks = make_etsl_parent_tasks(a_layer);
                uint64_t parents_total = 0;
                for (const auto& task : tasks) {
                    parents_total += task.record_count;
                }

                if (params_.verbose) {
                    const uint32_t active_workers = tasks.empty()
                        ? 0u
                        : static_cast<uint32_t>(
                            std::min<size_t>(configured_threads, tasks.size())
                        );
                    std::cout << "[ETSL-ext] a=" << static_cast<int>(a_layer)
                              << " dot_1^k generation"
                              << " | parents=" << parents_total
                              << " | tasks=" << tasks.size()
                              << " | workers=" << active_workers
                              << "..." << std::flush;
                }

                const uint64_t gen_before = result.generated_candidates;
                generate_etsl_dot1_candidates_parallel(a_layer, tasks);

                for (uint16_t b = 1; b <= params_.b_target_max; ++b) {
                    const auto path = etsl_frontier_file_path(work_dir, a_layer, b);
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
                    std::cout << " done"
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
                    if (!seed_variable_allowed(v)) {
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
                    std::cout << "[DPL-ext] target a=" << static_cast<int>(target_a)
                              << " candidate generation..." << std::flush;
                }

                const uint64_t gen_before = result.generated_candidates;
                const uint64_t acc_before = result.accepted_intermediate;
                const uint64_t dup_before = result.duplicate_candidates;
                const uint64_t rb_before = result.rejected_by_b;
                const uint64_t re_before = result.rejected_non_elementary;

                // Cycle enumeration is retained as a serial phase because it is
                // usually much smaller than expansion of the existing frontier.
                // Its output is still merged with all worker-generated runs.
                if (target_a >= k0 && target_a <= maxk) {
                    enumerate_cycles(target_a, [&](const SupportT& cyc) {
                        append_pending(cyc, CAND_FLAG_SEED_CYCLE);
                    });
                }
                flush_pending_run();

                {
                    const CycleCache cycle_cache = build_cycle_cache(target_a);
                    const std::vector<ParentChunkTask> parent_tasks =
                        make_parent_tasks(target_a);

                    if (params_.verbose) {
                        const uint32_t active_workers = parent_tasks.empty()
                            ? 0u
                            : static_cast<uint32_t>(
                                std::min<size_t>(configured_threads, parent_tasks.size())
                            );
                        std::cout << " | parent_tasks=" << parent_tasks.size()
                                  << " | workers=" << active_workers
                                  << " | cycle_cache="
                                  << (cycle_cache.enabled ? "memory" : "stream")
                                  << " | cycle_cache_bytes=" << cycle_cache.bytes
                                  << std::flush;
                    }

                    generate_parent_candidates_parallel(
                        target_a,
                        parent_tasks,
                        cycle_cache
                    );
                } // release the optional cycle cache before ETSL processing

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
                    std::cout << "[DPL-ext] target a=" << static_cast<int>(target_a)
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
                std::cout << "[DPL-ext] search completed"
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
            std::cerr << "[DPL-ext] search failed: " << e.what() << '\n'
                      << "[DPL-ext] work directory: " << work_dir.string()
                      << std::endl;

            if (!params_.preserve_external_work_dir_on_error) {
                std::error_code ec;
                std::filesystem::remove_all(work_dir, ec);
            }
            throw;
        } catch (...) {
            std::cerr << "[DPL-ext] search failed with a non-standard exception\n"
                      << "[DPL-ext] work directory: " << work_dir.string()
                      << std::endl;

            if (!params_.preserve_external_work_dir_on_error) {
                std::error_code ec;
                std::filesystem::remove_all(work_dir, ec);
            }
            throw;
        }
    }

    bool use_algorithm1_table() const {
        return !params_.bmax_by_a.empty();
    }

    uint16_t b_bound_for_size(uint8_t a) const {
        if (!use_algorithm1_table()) {
            return params_.b_aux_max;
        }

        if (static_cast<size_t>(a) >= params_.bmax_by_a.size()) {
            return 0;
        }

        return params_.bmax_by_a[static_cast<size_t>(a)];
    }

    bool ex_allows_dot(uint8_t a, uint16_t b) const {
        // Algorithm 1, line 4: dot is listed for b = 2,...,b^a_max.
        if (a + 1 > params_.amax) return false;
        if (b < 2) return false;
        if (b > b_bound_for_size(a)) return false;
        return true;
    }

    bool ex_allows_path(uint8_t a, uint16_t b, uint8_t m) const {
        // Algorithm 1, lines 7--10:
        // pa_m is allowed if a+m <= amax and 0 <= b-2 <= b^{a+m}_max.
        if (m < 2) return false;
        if (static_cast<int>(a) + static_cast<int>(m) > params_.amax) return false;
        if (b < 2) return false;

        const uint8_t child_a = static_cast<uint8_t>(a + m);
        const uint16_t child_lower_b = static_cast<uint16_t>(b - 2);

        return child_lower_b <= b_bound_for_size(child_a);
    }

    bool ex_allows_lollipop(uint8_t a, uint16_t b, uint8_t m) const {
        // Algorithm 1, lines 11--15:
        // lo_m^c is allowed if m >= g/2 and b-1 <= b^{a+m}_max.
        const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);

        if (m < g2) return false;
        if (static_cast<int>(a) + static_cast<int>(m) > params_.amax) return false;
        if (b < 1) return false;

        const uint8_t child_a = static_cast<uint8_t>(a + m);
        const uint16_t child_lower_b = static_cast<uint16_t>(b - 1);

        return child_lower_b <= b_bound_for_size(child_a);
    }

    bool algorithm1_table_has_lollipop() const {
        if (!use_algorithm1_table()) {
            return !menu_.lollipop_mc.empty();
        }

        const uint8_t g2 = static_cast<uint8_t>(params_.g / 2);

        for (uint8_t a = g2; a < params_.amax; ++a) {
            const uint16_t ba = b_bound_for_size(a);

            for (uint16_t b = 1; b <= ba; ++b) {
                for (uint8_t m = g2;
                     static_cast<int>(a) + static_cast<int>(m) <= params_.amax;
                     ++m) {
                    if (ex_allows_lollipop(a, b, m)) {
                        return true;
                    }
                }
            }
        }

        return false;
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
        if (params_.amax == 0) {
            throw std::runtime_error("TS_enumerator_irreg23: amax must be positive");
        }

        if (params_.amax > AMAX) {
            throw std::runtime_error("TS_enumerator_irreg23: params.amax exceeds template AMAX");
        }

        if (params_.K == 0) {
            throw std::runtime_error("TS_enumerator_irreg23: K must be positive");
        }

        if (params_.g == 0 || (params_.g % 2) != 0) {
            throw std::runtime_error("TS_enumerator_irreg23: g must be positive and even");
        }

        if (params_.g / 2 > params_.amax) {
            throw std::runtime_error("TS_enumerator_irreg23: g/2 exceeds amax");
        }

        if (T.M <= 0 || T.N <= 0) {
            throw std::runtime_error("TS_enumerator_irreg23: empty Tanner topology");
        }

        if (params_.b_target_min > params_.b_target_max) {
            throw std::runtime_error("TS_enumerator_irreg23: invalid target b range");
        }

        if (params_.b_target_max > params_.b_aux_max) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: b_target_max cannot exceed b_aux_max"
            );
        }

        if (!params_.bmax_by_a.empty()) {
            if (params_.bmax_by_a.size() <= static_cast<size_t>(params_.amax)) {
                throw std::runtime_error(
                    "TS_enumerator_irreg23: bmax_by_a must have length at least amax + 1"
                );
            }

            if (params_.K < params_.amax) {
                throw std::runtime_error(
                    "TS_enumerator_irreg23: Algorithm-1 mode requires K >= amax"
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

            if (params_.b_aux_max < max_bound) {
                throw std::runtime_error(
                    "TS_enumerator_irreg23: b_aux_max must be at least max_a bmax_by_a[a]"
                );
            }
        }

        if (params_.hash_max_load_factor <= 0.25f) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: hash_max_load_factor is too small"
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

    static bool support_less(const SupportT& a, const SupportT& b) {
        if (a.len != b.len) return a.len < b.len;

        for (uint8_t i = 0; i < a.len; ++i) {
            if (a.v[i] != b.v[i]) {
                return a.v[i] < b.v[i];
            }
        }

        return false;
    }

    void validate_allowed_variable_degrees() const {
        if (params_.seed_vn_limit < -1) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: seed_vn_limit must be -1 or nonnegative"
            );
        }

        if (params_.final_anchor_vn_limit < -1) {
            throw std::runtime_error(
                "TS_enumerator_irreg23: final_anchor_vn_limit must be -1 or nonnegative"
            );
        }

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

    bool seed_variable_allowed(int v) const {
        if (params_.seed_vn_limit >= 0 && v >= params_.seed_vn_limit) {
            return false;
        }

        return variable_degree_allowed(v);
    }

    bool final_support_allowed(const SupportT& S) const {
        if (params_.final_anchor_vn_limit < 0) {
            return true;
        }

        return S.len > 0 && S.v[0] < params_.final_anchor_vn_limit;
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

    bool accept_candidate(
        const SupportT& cand,
        std::vector<std::vector<SupportT>>& frontier,
        std::vector<SupportSet>& seen_by_size,
        EnumeratorResult<AMAX>& result,
        std::function<void(const SupportT&, uint16_t)>* etsl_seed_callback = nullptr
    ) {
        ++result.generated_candidates;

        if (cand.len == 0 || cand.len > params_.amax) {
            ++result.rejected_by_size;
            return false;
        }

        const StructureStats st = induced_stats(T, cand, scratch_);

        if (params_.require_elementary && !st.elementary) {
            ++result.rejected_non_elementary;
            return false;
        }

        if (st.b > b_bound_for_size(cand.len)) {
            ++result.rejected_by_b;
            return false;
        }

        auto& seen = seen_by_size[cand.len];
        const auto inserted = seen.insert(cand);

        if (!inserted.second) {
            ++result.duplicate_candidates;
            return false;
        }

        // b=0 structures are valid outputs but terminal under elementary DPL
        // expansions, so do not retain them as future parents.
        if (st.b != 0) {
            frontier[cand.len].push_back(cand);
        }
        ++result.accepted_intermediate;

        // DPL-generated accepted structures are LETSs. In online ETSL mode,
        // pass each accepted LETS seed to the ETSL layer cache immediately.
        // The callback performs its own deduplication against dot_1^k-generated
        // ETSL structures.
        if (etsl_seed_callback != nullptr && st.b <= params_.b_target_max) {
            (*etsl_seed_callback)(cand, st.b);
        }

        const bool is_target =
            (st.b >= params_.b_target_min) &&
            (st.b <= params_.b_target_max);

        if (is_target && final_support_allowed(cand)) {
            increment_count(result, st.b, cand.len);
            write_support_if_requested(cand, st.b, result);

            if (params_.store_final_supports) {
                result.final_supports.push_back(cand);
            }
        }

        return true;
    }

    void add_final_if_target(
        const SupportT& S,
        uint16_t b,
        EnumeratorResult<AMAX>& result
    ) {
        if (b < params_.b_target_min || b > params_.b_target_max) {
            return;
        }

        if (!final_support_allowed(S)) {
            return;
        }

        increment_count(result, b, S.len);
        write_support_if_requested(S, b, result);

        if (params_.store_final_supports) {
            result.final_supports.push_back(S);
        }
    }

    bool add_temp_etsl(
        const SupportT& S,
        uint16_t b,
        SupportsByAB& temp_by_ab,
        SeenByAB& seen_temp_by_ab
    ) {
        if (S.len == 0 || S.len > params_.amax) {
            return false;
        }

        if (b > params_.b_target_max) {
            return false;
        }

        auto& seen = seen_temp_by_ab[S.len][b];
        const auto inserted = seen.insert(S);

        if (!inserted.second) {
            return false;
        }

        temp_by_ab[S.len][b].push_back(S);
        return true;
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

    void run_dot1_etsl_search(
        const SupportsByAB& lets_by_ab,
        EnumeratorResult<AMAX>& result
    ) {
        const uint16_t bmax = params_.b_target_max;

        if (bmax == 0) {
            return;
        }

        SupportsByAB temp_by_ab(
            AMAX + 1,
            std::vector<std::vector<SupportT>>(static_cast<size_t>(bmax + 1))
        );

        SeenByAB seen_temp_by_ab(
            AMAX + 1,
            std::vector<SupportSet>(static_cast<size_t>(bmax + 1))
        );

        for (auto& row : seen_temp_by_ab) {
            for (auto& seen : row) {
                seen.max_load_factor(params_.hash_max_load_factor);
            }
        }

        if (params_.verbose) {
            std::cout << "[ETSL] dot_1^k search in progress..." << std::flush;
        }

        // Algorithm 3 initialization for ETSL2: single variable nodes.
        // In a degree-2/3 code, a single VN is an elementary (1, deg(v)) TS.
        // Algorithm 3 initialization for ETSL2: single variable nodes.
        // In a degree-2/3 code, a single VN is an elementary (1, deg(v)) TS.
        // This can be extremely expensive because it grows many tree-like ETSs.
        if (params_.include_singleton_etsl_seeds) {
            for (int v = 0; v < T.N; ++v) {
                if (!seed_variable_allowed(v)) {
                    continue;
                }

                const int dv = T.variable_degree(v);

                if (static_cast<uint16_t>(dv) > bmax) {
                    continue;
                }

                SupportT S;
                S.len = 1;
                S.v[0] = v;

                const uint16_t b = static_cast<uint16_t>(dv);

                if (add_temp_etsl(S, b, temp_by_ab, seen_temp_by_ab)) {
                    add_final_if_target(S, b, result);
                }
            }
        }
        // Algorithm 3 initialization for ETSL1: all LETS seeds with b >= 1.
        // b = 0 LETSs cannot be expanded by dot_1^k because there is no
        // unsatisfied parent check to attach to.
        for (uint8_t a = 1; a <= params_.amax; ++a) {
            for (uint16_t b = 1; b <= bmax; ++b) {
                for (const SupportT& S : lets_by_ab[a][b]) {
                    add_temp_etsl(S, b, temp_by_ab, seen_temp_by_ab);
                }
            }
        }

        uint64_t generated_before = result.generated_candidates;
        uint64_t accepted_before = result.accepted_intermediate;

        for (uint8_t a = 1; a < params_.amax; ++a) {
            for (uint16_t b = 1; b <= bmax; ++b) {
                auto& parents = temp_by_ab[a][b];

                for (size_t pos = 0; pos < parents.size(); ++pos) {
                    const SupportT& parent = parents[pos];

                    expand_dot1(parent, b, [&](const SupportT& child, uint16_t child_b) {
                        ++result.generated_candidates;

                        const StructureStats st = induced_stats(T, child, scratch_);

                        if (params_.require_elementary && !st.elementary) {
                            ++result.rejected_non_elementary;
                            return;
                        }

                        if (st.b != child_b || st.b > bmax) {
                            ++result.rejected_by_b;
                            return;
                        }

                        if (!add_temp_etsl(child, st.b, temp_by_ab, seen_temp_by_ab)) {
                            ++result.duplicate_candidates;
                            return;
                        }

                        ++result.accepted_intermediate;
                        add_final_if_target(child, st.b, result);
                    });
                }
            }
        }

        if (params_.verbose) {
            std::cout << " done"
                    << " | generated=" << (result.generated_candidates - generated_before)
                    << " | accepted=" << (result.accepted_intermediate - accepted_before)
                    << std::endl;
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
            unordered_set here saves a large temporary hash table during
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
            if (params_.seed_vn_limit >= 0 && start >= params_.seed_vn_limit) {
                break;
            }

            if (!seed_variable_allowed(start)) {
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