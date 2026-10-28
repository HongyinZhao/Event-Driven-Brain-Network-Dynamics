# This code calculates associations between intrinsic COC and neonatal age.
# Partial correlations test GA controlling for PNA and PNA controlling for GA.
# Results are calculated at global, network and edge levels with FDR correction.
# Set the data paths and numerical analysis parameters below before running.

import os
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, t
from statsmodels.stats.multitest import fdrcorrection
import multiprocessing as mp
import threading
from tqdm import tqdm

# %% Analysis parameters and file paths
BASE_DIR = r""
PMA_PATH = r""
NET_LABEL_PATH = r""

EVENTS = ["emergence", "disappearance", "expansion", "contraction"]
NUM_SUBJECTS = []      # Number of participants
MATRIX_SIZE = []       # Number of cortical vertices
DIVISOR = []           # Scaling divisor for input COC matrices
NUM_WORKERS = []       # Parallel processes

COMMUNITY_NAMES = ["VIS", "SM", "DA", "VA", "LIM", "FPN", "DMN"]
NUM_NETWORKS = len(COMMUNITY_NAMES)

OUTPUT_DIR = ""           # Folder for the results


# %% Loading of MAT data and calculation of partial Pearson correlation
def load_mat_variable(filepath):
    data = sio.loadmat(filepath)
    for key in data:
        if not key.startswith('__'):
            return data[key]
    raise ValueError(f"no valid data found")


def calc_partial_r(x, y, z):
    """Pearson partial correlation between x and y after controlling for z."""
    r_xy, _ = pearsonr(x, y)
    r_xz, _ = pearsonr(x, z)
    r_yz, _ = pearsonr(y, z)

    denom = np.sqrt(1 - r_xz ** 2) * np.sqrt(1 - r_yz ** 2)
    if denom < 1e-12:
        return 0.0, 1.0

    pr = (r_xy - r_xz * r_yz) / denom
    pr = np.clip(pr, -1.0, 1.0)

    df = len(x) - 3
    if abs(pr) == 1.0:
        return pr, 0.0

    t_val = pr * np.sqrt(df / (1.0 - pr ** 2))
    p_val = t.sf(np.abs(t_val), df=df) * 2
    return pr, p_val

W_IU = None
W_NET_NODES = None
W_GA = None
W_PNA = None
W_NUM_EDGES = None


# %% Initialization of parallel workers
def init_worker(matrix_size, net_nodes, ga_values, pna_values):
    global W_IU, W_NET_NODES, W_GA, W_PNA, W_NUM_EDGES
    W_IU = np.triu_indices(matrix_size, k=1)
    W_NUM_EDGES = len(W_IU[0])
    W_NET_NODES = net_nodes
    W_GA = ga_values
    W_PNA = pna_values


# %% Calculation of subject-level global and network COC
def process_chunk(args):
    event, subj_indices, base_dir, divisor, queue = args

    sum_X = np.zeros(W_NUM_EDGES, dtype=np.float64)
    sum_X2 = np.zeros(W_NUM_EDGES, dtype=np.float64)
    sum_X_GA = np.zeros(W_NUM_EDGES, dtype=np.float64)
    sum_X_PNA = np.zeros(W_NUM_EDGES, dtype=np.float64)

    chunk_results = []

    for subj_idx in subj_indices:
        subject_id = subj_idx + 1
        file_path = os.path.join(base_dir, f"subject_{subject_id}", f"{event}_COC.mat")

        try:
            mat = load_mat_variable(file_path).astype(np.float32) / divisor
            edge_vals = mat[W_IU]
            global_mean = float(np.mean(edge_vals))

            # Exclude zero or near-zero COC matrices.
            if np.isclose(global_mean, 0.0) or global_mean == 0:
                is_valid = False
                net_mean = np.zeros((NUM_NETWORKS, NUM_NETWORKS), dtype=np.float32)
                chunk_results.append((subj_idx, global_mean, net_mean, is_valid))
            else:
                is_valid = True
                sum_X += edge_vals
                sum_X2 += (edge_vals ** 2)
                sum_X_GA += (edge_vals * W_GA[subj_idx])
                sum_X_PNA += (edge_vals * W_PNA[subj_idx])

                # Within-network pairs exclude the diagonal; between-network pairs use all edges.
                net_mean = np.zeros((NUM_NETWORKS, NUM_NETWORKS), dtype=np.float32)
                for i in range(1, NUM_NETWORKS + 1):
                    for j in range(i, NUM_NETWORKS + 1):
                        nodes_i = W_NET_NODES[i]
                        nodes_j = W_NET_NODES[j]

                        if i == j:
                            block = mat[np.ix_(nodes_i, nodes_i)]
                            block_iu = np.triu_indices(len(nodes_i), k=1)
                            val = np.mean(block[block_iu]) if len(block_iu[0]) > 0 else 0
                        else:
                            block = mat[np.ix_(nodes_i, nodes_j)]
                            val = np.mean(block) if block.size > 0 else 0

                        net_mean[i - 1, j - 1] = val

                chunk_results.append((subj_idx, global_mean, net_mean, is_valid))

        except Exception:
            pass

        if queue is not None:
            queue.put(1)

    return sum_X, sum_X2, sum_X_GA, sum_X_PNA, chunk_results


# %% Monitoring of subject processing
def progress_listener(queue, total):
    pbar = tqdm(total=total, desc="processing", unit="subject")
    for _ in range(total):
        queue.get()
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    mp.set_start_method('spawn', force=True)

    age_data = sio.loadmat(PMA_PATH)
    ga_values = age_data['TermGA'].flatten()
    pma_values = age_data['TermPMA'].flatten()
    pna_values = pma_values - ga_values

    network_labels = load_mat_variable(NET_LABEL_PATH).flatten()
    net_nodes = {i: np.where(network_labels == i)[0] for i in range(1, NUM_NETWORKS + 1)}

    tmp_iu = np.triu_indices(MATRIX_SIZE, k=1)
    NUM_EDGES = len(tmp_iu[0])
    del tmp_iu

    all_indices = list(range(NUM_SUBJECTS))
    chunks = [all_indices[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    for event in EVENTS:

        manager = mp.Manager()
        progress_queue = manager.Queue()
        watcher = threading.Thread(target=progress_listener, args=(progress_queue, NUM_SUBJECTS))
        watcher.start()

        pool = mp.Pool(
            processes=NUM_WORKERS,
            initializer=init_worker,
            initargs=(MATRIX_SIZE, net_nodes, ga_values, pna_values)
        )

        tasks = [(event, chunk, BASE_DIR, DIVISOR, progress_queue) for chunk in chunks]
        results = pool.map(process_chunk, tasks)

        pool.close()
        pool.join()
        watcher.join()

        total_sum_X = np.zeros(NUM_EDGES, dtype=np.float64)
        total_sum_X2 = np.zeros(NUM_EDGES, dtype=np.float64)
        total_sum_X_GA = np.zeros(NUM_EDGES, dtype=np.float64)
        total_sum_X_PNA = np.zeros(NUM_EDGES, dtype=np.float64)

        global_means = np.zeros(NUM_SUBJECTS)
        net_means = np.zeros((NUM_SUBJECTS, NUM_NETWORKS, NUM_NETWORKS))
        valid_mask = np.zeros(NUM_SUBJECTS, dtype=bool)

        for (local_X, local_X2, local_X_GA, local_X_PNA, chunk_res) in results:
            total_sum_X += local_X
            total_sum_X2 += local_X2
            total_sum_X_GA += local_X_GA
            total_sum_X_PNA += local_X_PNA

            for (subj_idx, g_mean, n_mean, is_valid) in chunk_res:
                global_means[subj_idx] = g_mean
                net_means[subj_idx] = n_mean
                valid_mask[subj_idx] = is_valid

        del results

        N_valid = int(np.sum(valid_mask))

        if N_valid < 4:
            continue

        # Reuse the same valid-subject mask at every analysis level.
        valid_ga = ga_values[valid_mask]
        valid_pna = pna_values[valid_mask]
        valid_global_means = global_means[valid_mask]
        valid_net_means = net_means[valid_mask]

        # Global COC: GA controlling for PNA, and PNA controlling for GA.
        pr_GA_glob, p_GA_glob = calc_partial_r(valid_global_means, valid_ga, valid_pna)
        pr_PNA_glob, p_PNA_glob = calc_partial_r(valid_global_means, valid_pna, valid_ga)

        sio.savemat(os.path.join(OUTPUT_DIR, f"{event}_Global_Results.mat"), {
            'PR_GA': pr_GA_glob, 'P_GA': p_GA_glob,
            'PR_PNA': pr_PNA_glob, 'P_PNA': p_PNA_glob,
            'Global_COC_Means': global_means.reshape(-1, 1),
            'Valid_Mask': valid_mask.reshape(-1, 1),
            'N_valid': N_valid
        })

        PR_GA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))
        P_uncorr_GA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))
        p_vals_GA_net = []

        PR_PNA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))
        P_uncorr_PNA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))
        # Use unique network pairs (including within-network comparisons).
        p_vals_PNA_net = []

        for i in range(NUM_NETWORKS):
            for j in range(i, NUM_NETWORKS):
                pr_ga, p_ga = calc_partial_r(valid_net_means[:, i, j], valid_ga, valid_pna)
                pr_pna, p_pna = calc_partial_r(valid_net_means[:, i, j], valid_pna, valid_ga)

                PR_GA_net[i, j] = pr_ga
                P_uncorr_GA_net[i, j] = p_ga
                p_vals_GA_net.append(p_ga)

                PR_PNA_net[i, j] = pr_pna
                P_uncorr_PNA_net[i, j] = p_pna
                p_vals_PNA_net.append(p_pna)

        _, p_vals_fdr_GA = fdrcorrection(p_vals_GA_net, method='indep')
        _, p_vals_fdr_PNA = fdrcorrection(p_vals_PNA_net, method='indep')

        P_fdr_GA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))
        P_fdr_PNA_net = np.zeros((NUM_NETWORKS, NUM_NETWORKS))

        idx = 0
        for i in range(NUM_NETWORKS):
            for j in range(i, NUM_NETWORKS):
                P_fdr_GA_net[i, j] = p_vals_fdr_GA[idx]
                P_fdr_PNA_net[i, j] = p_vals_fdr_PNA[idx]
                idx += 1

        def symmetrize(matrix):
            return matrix + matrix.T - np.diag(matrix.diagonal())

        sio.savemat(os.path.join(OUTPUT_DIR, f"{event}_Network_Results.mat"), {
            'PR_GA_net': symmetrize(PR_GA_net), 'P_uncorr_GA_net': symmetrize(P_uncorr_GA_net),
            'P_FDR_GA_net': symmetrize(P_fdr_GA_net),
            'PR_PNA_net': symmetrize(PR_PNA_net), 'P_uncorr_PNA_net': symmetrize(P_uncorr_PNA_net),
            'P_FDR_PNA_net': symmetrize(P_fdr_PNA_net),
            'Community_Names': COMMUNITY_NAMES, 'N_valid': N_valid
        })

        N_float = float(N_valid)

        sum_GA, sum_PNA = np.sum(valid_ga), np.sum(valid_pna)
        SS_GA = np.sum((valid_ga - np.mean(valid_ga)) ** 2)
        SS_PNA = np.sum((valid_pna - np.mean(valid_pna)) ** 2)

        r_GA_PNA, _ = pearsonr(valid_ga, valid_pna)
        denom_GA_PNA = np.sqrt(1 - r_GA_PNA ** 2)

        # Calculate edge-wise Pearson correlations from accumulated statistics.
        SS_X = total_sum_X2 - (total_sum_X ** 2) / N_float
        SP_X_GA = total_sum_X_GA - (total_sum_X * sum_GA) / N_float
        SP_X_PNA = total_sum_X_PNA - (total_sum_X * sum_PNA) / N_float

        edge_valid_mask = (SS_X > 1e-12)

        R_X_GA = np.zeros(NUM_EDGES, dtype=np.float32)
        R_X_PNA = np.zeros(NUM_EDGES, dtype=np.float32)

        R_X_GA[edge_valid_mask] = SP_X_GA[edge_valid_mask] / (np.sqrt(SS_X[edge_valid_mask] * SS_GA) + 1e-12)
        R_X_PNA[edge_valid_mask] = SP_X_PNA[edge_valid_mask] / (np.sqrt(SS_X[edge_valid_mask] * SS_PNA) + 1e-12)

        R_X_GA = np.clip(R_X_GA, -1.0, 1.0)
        R_X_PNA = np.clip(R_X_PNA, -1.0, 1.0)

        PR_GA = np.zeros(NUM_EDGES, dtype=np.float32)
        PR_PNA = np.zeros(NUM_EDGES, dtype=np.float32)

        valid_pr_GA_mask = edge_valid_mask & (np.abs(R_X_PNA) < 1.0)
        PR_GA[valid_pr_GA_mask] = (R_X_GA[valid_pr_GA_mask] - R_X_PNA[valid_pr_GA_mask] * r_GA_PNA) / (
                    np.sqrt(1 - R_X_PNA[valid_pr_GA_mask] ** 2) * denom_GA_PNA + 1e-12)

        valid_pr_PNA_mask = edge_valid_mask & (np.abs(R_X_GA) < 1.0)
        PR_PNA[valid_pr_PNA_mask] = (R_X_PNA[valid_pr_PNA_mask] - R_X_GA[valid_pr_PNA_mask] * r_GA_PNA) / (
                    np.sqrt(1 - R_X_GA[valid_pr_PNA_mask] ** 2) * denom_GA_PNA + 1e-12)

        PR_GA = np.clip(PR_GA, -1.0, 1.0)
        PR_PNA = np.clip(PR_PNA, -1.0, 1.0)

        df_edge = N_float - 3
        P_uncorr_GA = np.ones(NUM_EDGES, dtype=np.float32)
        P_uncorr_PNA = np.ones(NUM_EDGES, dtype=np.float32)

        valid_t_GA_mask = valid_pr_GA_mask & (np.abs(PR_GA) < 1.0)
        T_stat_GA = PR_GA[valid_t_GA_mask] * np.sqrt(df_edge / (1.0 - PR_GA[valid_t_GA_mask] ** 2))
        P_uncorr_GA[valid_t_GA_mask] = t.sf(np.abs(T_stat_GA), df=df_edge) * 2

        valid_t_PNA_mask = valid_pr_PNA_mask & (np.abs(PR_PNA) < 1.0)
        T_stat_PNA = PR_PNA[valid_t_PNA_mask] * np.sqrt(df_edge / (1.0 - PR_PNA[valid_t_PNA_mask] ** 2))
        P_uncorr_PNA[valid_t_PNA_mask] = t.sf(np.abs(T_stat_PNA), df=df_edge) * 2

        # Apply FDR correction to the edge-wise p-values.
        _, P_fdr_GA = fdrcorrection(P_uncorr_GA, method='indep')
        _, P_fdr_PNA = fdrcorrection(P_uncorr_PNA, method='indep')

        def reconstruct_matrix(vector_data, diag_val=0):
            mat = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
            iu = np.triu_indices(MATRIX_SIZE, k=1)
            mat[iu] = vector_data
            mat = mat + mat.T
            np.fill_diagonal(mat, diag_val)
            return mat

        sio.savemat(os.path.join(OUTPUT_DIR, f"{event}_Edge_Results.mat"), {
            'PR_GA_matrix': reconstruct_matrix(PR_GA, diag_val=0),
            'P_uncorr_GA': reconstruct_matrix(P_uncorr_GA, diag_val=1),
            'P_FDR_GA': reconstruct_matrix(P_fdr_GA, diag_val=1),

            'PR_PNA_matrix': reconstruct_matrix(PR_PNA, diag_val=0),
            'P_uncorr_PNA': reconstruct_matrix(P_uncorr_PNA, diag_val=1),
            'P_FDR_PNA': reconstruct_matrix(P_fdr_PNA, diag_val=1),

            'N_valid': N_valid
        })

    print("COC age-association analysis complete.")
