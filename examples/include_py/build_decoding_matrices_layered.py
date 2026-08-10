import numpy as np
import stim
import os
import scipy.sparse as sp
import scipy.io as sio  # Import scipy.io
import itertools
from collections import defaultdict
from typing import Dict, List, FrozenSet
import pickle
# ==============================================================================
# C++ bridge helpers
# ==============================================================================

def as_bool_csr(mat) -> sp.csr_matrix:
    if not sp.isspmatrix(mat):
        mat = sp.csr_matrix(mat)
    elif not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    
    # 1. Sum duplicates FIRST to merge edges like 1+1=2
    mat.sum_duplicates() 
    
    mat = mat.astype(np.uint8)
    mat.data &= 1
    
    # 2. Eliminate actual zeros (even those that became 0 modulo 2)
    mat.eliminate_zeros()
    
    return mat.astype(np.int32)




# ==============================================================================
# 1) Separate Data/Ancilla vs Hooks (unchanged logic)
# ==============================================================================

def separate_data_and_hooks(
    C,
    Hz: np.ndarray,
    num_rounds: int,
    num_detectors_per_round: int,
    subset_weight: int = 3
):
    if sp.issparse(C):
        C = C.toarray()
    C = (C.astype(np.uint8) & 1)
    Hz = (Hz.astype(np.uint8) & 1)

    D, K = C.shape
    _, n = Hz.shape

    used = np.zeros(K, dtype=bool)

    data_indices_ordered = []
    hook_indices_ordered = []
    data_info = {}
    hook_info = {}

    col_weights = C.sum(axis=0)
    C_bool = C.astype(bool)

    for t in range(num_rounds):
        row_start = t * num_detectors_per_round
        row_mid   = (t + 1) * num_detectors_per_round
        row_end   = (t + 2) * num_detectors_per_round

        if row_start >= D:
            break
        row_end = min(row_end, D)

        top_rows = np.arange(row_start, min(row_mid, D))
        bot_rows = np.arange(row_mid, row_end) if row_mid < row_end else np.array([], dtype=int)

        round_mask = np.zeros(D, dtype=bool)
        round_mask[top_rows] = True
        if bot_rows.size > 0:
            round_mask[bot_rows] = True

        cand_w2_all = np.flatnonzero((col_weights == 2) & ~used)
        cand_w3_all = np.flatnonzero((col_weights == subset_weight) & ~used)

        for j in range(n):
            check_indices = np.flatnonzero(Hz[:, j])

            qubit_mask = np.zeros(D, dtype=bool)
            valid_top = check_indices[check_indices < top_rows.size]
            qubit_mask[top_rows[valid_top]] = True
            if bot_rows.size > 0:
                valid_bot = check_indices[check_indices < bot_rows.size]
                qubit_mask[bot_rows[valid_bot]] = True

            # Ancilla (w=2): pick 1
            if cand_w2_all.size > 0:
                cols = cand_w2_all[~used[cand_w2_all]]
                if cols.size > 0:
                    cols_mat = C_bool[:, cols]
                    is_contained = (np.logical_and(cols_mat, ~qubit_mask[:, None]).sum(axis=0) == 0)
                    valid = cols[is_contained]
                    if valid.size > 0:
                        best = int(valid[0])
                        used[best] = True
                        new_idx = len(data_indices_ordered)
                        data_indices_ordered.append(best)
                        data_info[new_idx] = {"round": t, "hz_idx": j, "type": "Ancilla", "orig_idx": best}

            # Data (w=subset_weight): take all
            if cand_w3_all.size > 0:
                cols = cand_w3_all[~used[cand_w3_all]]
                if cols.size > 0:
                    cols_mat = C_bool[:, cols]
                    is_contained = (np.logical_and(cols_mat, ~qubit_mask[:, None]).sum(axis=0) == 0)
                    valid = cols[is_contained]
                    if valid.size > 0:
                        used[valid] = True
                        for idx in valid.tolist():
                            idx = int(idx)
                            new_idx = len(data_indices_ordered)
                            data_indices_ordered.append(idx)
                            data_info[new_idx] = {"round": t, "hz_idx": j, "type": "Data", "orig_idx": idx}

        # Hooks: unused fully inside the round window
        candidates = np.flatnonzero(~used)
        if candidates.size > 0:
            cand_cols = C_bool[:, candidates]
            outside = np.logical_and(cand_cols, ~round_mask[:, None])
            is_inside = (outside.sum(axis=0) == 0)
            valid_hooks = candidates[is_inside]
            if valid_hooks.size > 0:
                used[valid_hooks] = True
                for idx in valid_hooks.tolist():
                    idx = int(idx)
                    new_idx = len(hook_indices_ordered)
                    hook_indices_ordered.append(idx)
                    hook_info[new_idx] = {"round": t, "orig_idx": idx}

    C_data  = C[:, data_indices_ordered] if data_indices_ordered else np.zeros((D, 0), dtype=np.uint8)
    C_hooks = C[:, hook_indices_ordered] if hook_indices_ordered else np.zeros((D, 0), dtype=np.uint8)

    return C_data, C_hooks, data_info, hook_info


# ==============================================================================
# 2) Decompose hooks (unchanged logic)
# ==============================================================================

def decompose_hooks(
    C_data: np.ndarray,
    C_hooks: np.ndarray,
    Hx: np.ndarray,
    data_info: dict,
    hook_info: dict,
    num_detectors_per_round: int
):
    C_data  = (C_data.astype(np.uint8)  & 1)
    C_hooks = (C_hooks.astype(np.uint8) & 1)
    Hx      = (Hx.astype(np.uint8)      & 1)

    D, N_hooks = C_hooks.shape
    m_x, _ = Hx.shape

    data_lookup = defaultdict(lambda: defaultdict(list))
    for col_idx, info in data_info.items():
        if info["type"] == "Data":
            data_lookup[info["round"]][info["hz_idx"]].append(col_idx)

    solution_cache = defaultdict(lambda: defaultdict(list))
    unique_rounds = sorted(data_lookup.keys())

    for r_idx in unique_rounds:
        row_start = r_idx * num_detectors_per_round
        row_end   = min((r_idx + 2) * num_detectors_per_round, D)

        for row_hx in range(m_x):
            support_qubits = np.flatnonzero(Hx[row_hx, :])

            candidates = []
            for q in support_qubits:
                candidates.extend(data_lookup[r_idx][q])

            prev_r = r_idx - 1
            if prev_r in data_lookup:
                for q in support_qubits:
                    candidates.extend(data_lookup[prev_r][q])

            candidates = sorted(set(candidates))
            if len(candidates) < 2:
                continue

            cand_mat = C_data[row_start:row_end, candidates]

            for r_comb in (2, 3):
                if len(candidates) < r_comb:
                    continue
                for idx_tuple in itertools.combinations(range(len(candidates)), r_comb):
                    sub_cols = cand_mat[:, idx_tuple]
                    sum_vec = np.bitwise_xor.reduce(sub_cols, axis=1)
                    key = sum_vec.tobytes()
                    global_indices = [candidates[x] for x in idx_tuple]
                    solution_cache[r_idx][key].append(global_indices)

    solutions = {}
    for h_col in range(N_hooks):
        meta = hook_info.get(h_col)
        if meta is None:
            continue
        r_idx = meta["round"]
        row_start = r_idx * num_detectors_per_round
        row_end   = min((r_idx + 2) * num_detectors_per_round, D)

        target_vec = C_hooks[row_start:row_end, h_col]
        if not np.any(target_vec):
            solutions[h_col] = []
            continue

        key = target_vec.tobytes()
        candidates = solution_cache[r_idx].get(key)
        if not candidates:
            raise ValueError(f"No decomposition found for hook col={h_col} at round={r_idx}.")

        solutions[h_col] = candidates[0]

    return solutions


# ==============================================================================
# 3) Noise Tanner graph: T = [I | I | S]
# ==============================================================================

def build_noise_tanner_graph(N_data: int, solutions: dict):
    I = sp.eye(N_data, dtype=np.uint8, format="csc")
    base = sp.hstack([I, I], format="csc")

    num_hooks = len(solutions)
    if num_hooks == 0:
        return base.tocsr()

    rows, cols, data = [], [], []
    for s_col, hook_idx in enumerate(sorted(solutions.keys())):
        for r in solutions[hook_idx]:
            rows.append(int(r))
            cols.append(int(s_col))
            data.append(1)

    S = sp.csc_matrix((data, (rows, cols)), shape=(N_data, num_hooks), dtype=np.uint8)
    return sp.hstack([base, S], format="csr")


# ==============================================================================
# 4) Align LLRs to Noisegraph variable order (unchanged)
# ==============================================================================

def align_llrs_for_noise_graph(original_llrs: np.ndarray, data_info: dict, hook_info: dict, export_mat: bool = True) -> np.ndarray:
    N = len(data_info)
    M = len(hook_info)
    total = 2 * N + M

    data_orig = [data_info[i]["orig_idx"] for i in range(N)]
    hook_orig = [hook_info[i]["orig_idx"] for i in range(M)]

    ext = np.zeros(total, dtype=np.float64)
    ext[N:2*N] = original_llrs[data_orig]
    ext[2*N:2*N+M] = original_llrs[hook_orig]
    if export_mat:
        sio.savemat("decoder_data/LLRS.mat", {"LLRS": ext})
    return np.ascontiguousarray(ext, dtype=np.float64)


# ==============================================================================
# 5) Align OBS to Fullgraph columns (this is the key change)
# ==============================================================================

def align_obs_for_fullgraph(obs_raw, data_info: dict):
    """
    obs_raw: (n_obs x K_raw) where columns match C_raw/prior ordering.
    returns obs_full: (n_obs x N_data) aligned to C_data/Fullgraph columns.
    """
    N = len(data_info)
    data_orig = [data_info[i]["orig_idx"] for i in range(N)]

    if sp.issparse(obs_raw):
        # keep sparse, deterministic
        return obs_raw[:, data_orig].tocsc()
    else:
        return np.ascontiguousarray(obs_raw[:, data_orig], dtype=np.uint8)


# ==============================================================================
# 2. Circuit & DEM Logic
# ==============================================================================

def build_circuit(code, A_list, B_list, p, num_repeat, z_basis=True, use_both=False, HZH=False):
    n = code.N
    a1, a2, a3 = A_list
    b1, b2, b3 = B_list

    def nnz(m):
        a, b = m.nonzero()
        return b[np.argsort(a)]

    A1, A2, A3 = nnz(a1), nnz(a2), nnz(a3)
    B1, B2, B3 = nnz(b1), nnz(b2), nnz(b3)
    A1_T, A2_T, A3_T = nnz(a1.T), nnz(a2.T), nnz(a3.T)
    B1_T, B2_T, B3_T = nnz(b1.T), nnz(b2.T), nnz(b3.T)

    # Offsets
    X_chk, L_dat, R_dat, Z_chk = 0, n//2, n, 3*n//2

    # Circuit construction helpers
    detector_circuit = stim.Circuit()
    for i in range(n//2):
        detector_circuit.append("DETECTOR", [stim.target_rec(-n//2 + i)])

    detector_repeat_circuit = stim.Circuit()
    for i in range(n//2):
        detector_repeat_circuit.append("DETECTOR", [stim.target_rec(-n//2 + i), stim.target_rec(-n - n//2 + i)])

    def append_blocks(circ, repeat=False):
        # Round 1
        if repeat:
            for i in range(n//2):
                circ.append("X_ERROR", Z_chk + i, p)
                if HZH:
                    circ.append("X_ERROR", X_chk + i, p)
                    circ.append("H", [X_chk + i])
                    circ.append("DEPOLARIZE1", X_chk + i, p)
                else:
                    circ.append("Z_ERROR", X_chk + i, p)
                circ.append("DEPOLARIZE1", R_dat + i, p)
        else:
            for i in range(n//2):
                circ.append("H", [X_chk + i])
                if HZH: circ.append("DEPOLARIZE1", X_chk + i, p)

        for i in range(n//2):
            circ.append("CNOT", [R_dat + A1_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [R_dat + A1_T[i], Z_chk + i], p)
            circ.append("DEPOLARIZE1", L_dat + i, p)
        circ.append("TICK")

        # Round 2-6 (Standard CNOT schedules)
        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, L_dat + A2[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, L_dat + A2[i]], p)
            circ.append("CNOT", [R_dat + A3_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [R_dat + A3_T[i], Z_chk + i], p)
        circ.append("TICK")

        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, R_dat + B2[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, R_dat + B2[i]], p)
            circ.append("CNOT", [L_dat + B1_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [L_dat + B1_T[i], Z_chk + i], p)
        circ.append("TICK")

        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, R_dat + B1[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, R_dat + B1[i]], p)
            circ.append("CNOT", [L_dat + B2_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [L_dat + B2_T[i], Z_chk + i], p)
        circ.append("TICK")

        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, R_dat + B3[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, R_dat + B3[i]], p)
            circ.append("CNOT", [L_dat + B3_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [L_dat + B3_T[i], Z_chk + i], p)
        circ.append("TICK")

        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, L_dat + A1[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, L_dat + A1[i]], p)
            circ.append("CNOT", [R_dat + A2_T[i], Z_chk + i])
            circ.append("DEPOLARIZE2", [R_dat + A2_T[i], Z_chk + i], p)
        circ.append("TICK")

        # Round 7
        for i in range(n//2):
            circ.append("CNOT", [X_chk + i, L_dat + A3[i]])
            circ.append("DEPOLARIZE2", [X_chk + i, L_dat + A3[i]], p)
            circ.append("X_ERROR", Z_chk + i, p)
            circ.append("MR", [Z_chk + i])

        if z_basis:
            circ += detector_repeat_circuit if repeat else detector_circuit
        elif use_both and repeat:
            circ += detector_repeat_circuit
        circ.append("TICK")

        # Round 8
        for i in range(n//2):
            if HZH:
                circ.append("H", [X_chk + i])
                circ.append("DEPOLARIZE1", X_chk + i, p)
                circ.append("X_ERROR", X_chk + i, p)
                circ.append("MR", [X_chk + i])
            else:
                circ.append("Z_ERROR", X_chk + i, p)
                circ.append("MRX", [X_chk + i])
        
        if not z_basis:
            circ += detector_repeat_circuit if repeat else detector_circuit
        elif use_both and repeat:
            circ += detector_repeat_circuit
        circ.append("TICK")

    # Build Full Circuit
    circuit = stim.Circuit()
    for i in range(n//2):
        circuit.append("R", X_chk + i)
        circuit.append("R", Z_chk + i)
        circuit.append("X_ERROR", X_chk + i, p)
        circuit.append("X_ERROR", Z_chk + i, p)
    for i in range(n):
        circuit.append("R" if z_basis else "RX", L_dat + i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", L_dat + i, p)
    
    circuit.append("TICK")
    append_blocks(circuit, repeat=False)

    rep_circuit = stim.Circuit()
    append_blocks(rep_circuit, repeat=True)
    circuit += (num_repeat-1) * rep_circuit

    for i in range(n):
        circuit.append("M" if z_basis else "MX", L_dat + i)

    # Stabilizers
    pcm = code.hz if z_basis else code.hx
    for i, s in enumerate(pcm):
        nnz_idx = np.nonzero(s)[0]
        targets = [stim.target_rec(-n + ind) for ind in nnz_idx]
        targets.append(stim.target_rec(-n - n + i if z_basis else -n - n//2 + i))
        circuit.append("DETECTOR", targets)
    
    # Logical Operators
    logical_pcm = code.lz if z_basis else code.lx
    for i, l in enumerate(logical_pcm):
        nnz_idx = np.nonzero(l)[0]
        targets = [stim.target_rec(-n + ind) for ind in nnz_idx]
        circuit.append("OBSERVABLE_INCLUDE", targets, i)

    return circuit

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):
    DL_ids: Dict[str, int] = {}
    L_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}

    def handle_error(prob: float, detectors: List[int], observables: List[int]):
        dets, obs = frozenset(detectors), frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])
        
        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0
        
        hid = DL_ids[key]
        L_map[hid] = obs
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets, frames = [], []
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
            
    check_matrix = dict_to_csc_matrix(
        {v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] for k, v in DL_ids.items()},
        shape=(dem.num_detectors, len(DL_ids))
    )
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors

def dict_to_csc_matrix(elements_dict, shape) -> sp.csc_matrix:
    """Constructs a csc_matrix from a dictionary of col -> [rows]."""
    nnz = sum(len(v) for v in elements_dict.values())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    
    i = 0
    for col, rows in elements_dict.items():
        for row in rows:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return sp.csc_matrix((data, (row_ind, col_ind)), shape=shape)
