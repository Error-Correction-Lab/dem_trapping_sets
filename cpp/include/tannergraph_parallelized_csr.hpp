#pragma once

#include <vector>
#include <stdexcept>

struct TannerTopology {
    int M = 0;
    int N = 0;
    int E = 0;

    std::vector<int> cn_ptr;
    std::vector<int> cn_vn;

    std::vector<int> vn_ptr;
    std::vector<int> vn_cn;
    std::vector<int> vn_edge;

    TannerTopology() = default;

    TannerTopology(
        int M_in,
        int N_in,
        const std::vector<int>& indptr,
        const std::vector<int>& indices
    ) {
        build_from_csr_rows(M_in, N_in, indptr, indices);
    }

    void clear() {
        M = N = E = 0;
        cn_ptr.clear();
        cn_vn.clear();
        vn_ptr.clear();
        vn_cn.clear();
        vn_edge.clear();
    }

    inline int check_degree(int c) const {
        return cn_ptr[c + 1] - cn_ptr[c];
    }

    inline int variable_degree(int v) const {
        return vn_ptr[v + 1] - vn_ptr[v];
    }

    void build_from_csr_rows(
        int M_in,
        int N_in,
        const std::vector<int>& indptr,
        const std::vector<int>& indices
    ) {
        clear();

        if (M_in < 0 || N_in < 0) {
            throw std::runtime_error("TannerTopology::build_from_csr_rows: negative dimensions");
        }

        if ((int)indptr.size() != M_in + 1) {
            throw std::runtime_error("TannerTopology::build_from_csr_rows: bad indptr size");
        }

        if (indptr.empty() || indptr[0] != 0) {
            throw std::runtime_error("TannerTopology::build_from_csr_rows: indptr[0] must be zero");
        }

        for (int c = 0; c < M_in; ++c) {
            if (indptr[c] > indptr[c + 1]) {
                throw std::runtime_error("TannerTopology::build_from_csr_rows: indptr must be nondecreasing");
            }
        }

        if (indptr[M_in] < 0) {
            throw std::runtime_error("TannerTopology::build_from_csr_rows: negative edge count");
        }

        M = M_in;
        N = N_in;
        E = indptr[M];

        cn_ptr = indptr;
        cn_vn  = indices;

        if ((int)cn_vn.size() != E) {
            throw std::runtime_error("TannerTopology::build_from_csr_rows: bad indices size");
        }

        std::vector<int> vn_deg(N, 0);

        for (int c = 0; c < M; ++c) {
            for (int p = cn_ptr[c]; p < cn_ptr[c + 1]; ++p) {
                const int v = cn_vn[p];

                if (v < 0 || v >= N) {
                    throw std::runtime_error("TannerTopology::build_from_csr_rows: variable index out of range");
                }

                ++vn_deg[v];
            }
        }

        vn_ptr.assign(N + 1, 0);

        for (int v = 0; v < N; ++v) {
            vn_ptr[v + 1] = vn_ptr[v] + vn_deg[v];
        }

        vn_cn.assign(E, 0);
        vn_edge.assign(E, 0);

        std::vector<int> vn_write = vn_ptr;

        for (int c = 0; c < M; ++c) {
            for (int p = cn_ptr[c]; p < cn_ptr[c + 1]; ++p) {
                const int v = cn_vn[p];

                const int q = vn_write[v]++;
                vn_cn[q] = c;
                vn_edge[q] = p;
            }
        }
    }
};