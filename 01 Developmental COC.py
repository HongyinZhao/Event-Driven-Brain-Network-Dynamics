import os
import gc
import random
import multiprocessing
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import savemat
import infomap
from tqdm import tqdm

# Analysis parameter
NUM_NODES = 8589
EVENT_TYPES = ['emergence', 'disappearance', 'expansion', 'contraction']

# 1. FC calculation and Fisher-Z transformation
def calculate_pearson_fc(bold_signal):
    X_mean = bold_signal.mean(axis=1, keepdims=True)
    X_centered = bold_signal - X_mean
    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    X_norm[X_norm == 0] = 1e-10
    X_normalized = X_centered / X_norm
    window_r = np.dot(X_normalized, X_normalized.T)
    return window_r

def fisher_z_transform(window_r):
    window_r = np.clip(window_r, -0.9999, 0.9999)
    fisher_z_matrix = np.arctanh(window_r)
    return fisher_z_matrix


# 2. Community Detection
def detect_communities_in_window(fisher_z_matrix, threshold=[]):
    n_nodes = fisher_z_matrix.shape[0]
    work_matrix = fisher_z_matrix.copy()
    np.fill_diagonal(work_matrix, 0)

    rows, cols = np.triu_indices_from(work_matrix, k=1)
    weights = work_matrix[rows, cols]

    valid_mask = (weights > threshold) & (~np.isnan(weights))
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    valid_weights = weights[valid_mask]

    if len(valid_weights) == 0:
        return {node: 0 for node in range(n_nodes)}

    im = infomap.Infomap("--two-level --silent")
    for r, c, w in zip(valid_rows, valid_cols, valid_weights):
        im.add_link(int(r), int(c), float(w))
    im.run()

    communities = {node: 0 for node in range(n_nodes)}
    for node in im.nodes:
        if node.node_id < n_nodes:
            communities[node.node_id] = node.module_id

    return communities

def process_subject_task(args):
    subject_id, bold_dir, threshold = args
    file_path = os.path.join(bold_dir, f"sub_BOLD_{subject_id}.mat")

    try:

        mat_data = sio.loadmat(file_path)
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]
        if not var_names:
            return subject_id, None

        bold_signal = mat_data[var_names[0]]

        window_r = calculate_pearson_fc(bold_signal)
        fisher_z_matrix = fisher_z_transform(window_r)

        communities = detect_communities_in_window(fisher_z_matrix, threshold)


        del window_r, fisher_z_matrix, bold_signal

        return subject_id, communities

    except Exception as e:
        print()
        return subject_id, None


# 3. Community Evolution Events Detection
def calculate_similarity(nodes_set1, nodes_set2):
    """Calculating Jaccard Similarity"""
    if len(nodes_set1) == 0 and len(nodes_set2) == 0:
        return 1.0
    intersection = len(nodes_set1.intersection(nodes_set2))
    union = len(nodes_set1.union(nodes_set2))
    return intersection / union if union > 0 else 0.0


def group_nodes_by_community(communities):
    groups = defaultdict(list)
    for node, comm_id in communities.items():
        groups[comm_id].append(node)
    return dict(groups)


def detect_evolution_events(prev_communities, curr_communities, window_idx,
                            sim_threshold=[], expansion_th=[], contraction_th=[]):

    events = []
    if not prev_communities or not curr_communities:
        return events

    prev_groups = group_nodes_by_community(prev_communities)
    curr_groups = group_nodes_by_community(curr_communities)

    # 1. Emergence
    for curr_id, curr_nodes in curr_groups.items():
        curr_nodes_set = set(curr_nodes)
        best_similarity = 0
        for prev_id, prev_nodes in prev_groups.items():
            similarity = calculate_similarity(curr_nodes_set, set(prev_nodes))
            if similarity > best_similarity:
                best_similarity = similarity

        if best_similarity < sim_threshold:
            events.append({
                'Window_Number': window_idx + 1,
                'Event_Type': 'emergence',
                'Prev_community_ID': None,
                'Curr_community_ID': curr_id,
                'Similarity': best_similarity,
                'Node_Count': len(curr_nodes)
            })

    # 2. Disappearance
    for prev_id, prev_nodes in prev_groups.items():
        prev_nodes_set = set(prev_nodes)
        best_similarity = 0
        best_match_id = None
        best_match_nodes = None

        for curr_id, curr_nodes in curr_groups.items():
            similarity = calculate_similarity(prev_nodes_set, set(curr_nodes))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = curr_id
                best_match_nodes = curr_nodes

        if best_similarity < sim_threshold:
            events.append({
                'Window_Number': window_idx + 1,
                'Event_Type': 'disappearance',
                'Prev_community_ID': prev_id,
                'Curr_community_ID': None,
                'Similarity': best_similarity,
                'Node_Count': len(prev_nodes)
            })
        elif best_match_nodes is not None:
            prev_size = len(prev_nodes)
            curr_size = len(best_match_nodes)
            ratio = curr_size / prev_size if prev_size > 0 else 1
            # 3. Expansion
            if ratio > expansion_th:
                events.append({
                    'Window_Number': window_idx + 1,
                    'Event_Type': 'expansion',
                    'Prev_community_ID': prev_id,
                    'Curr_community_ID': best_match_id,
                    'Prev_Node_Count': prev_size,
                    'Curr_Node_Count': curr_size,
                    'Expansion_Ratio': f"{ratio:.2%}",
                    'Similarity': best_similarity
                })
            # 4. Contraction
            elif ratio < contraction_th:
                events.append({
                    'Window_Number': window_idx + 1,
                    'Event_Type': 'contraction',
                    'Prev_community_ID': prev_id,
                    'Curr_community_ID': best_match_id,
                    'Prev_Node_Count': prev_size,
                    'Curr_Node_Count': curr_size,
                    'Contraction_Ratio': f"{ratio:.2%}",
                    'Similarity': best_similarity
                })

    return events


def track_evolution_timeline(timeline_iter, sim_threshold, expansion_th, contraction_th):

    all_evolution_events = []
    community_details = []
    comm_dict = {}

    prev_communities = None

    for snapshot_data in timeline_iter:
        window_idx = snapshot_data[0]
        win_num = window_idx + 1
        communities = snapshot_data[3]

        if communities is None:
            continue


        community_groups = defaultdict(list)
        for node, comm_id in communities.items():
            community_groups[comm_id].append(node)

        for comm_id, nodes in community_groups.items():
            sorted_nodes = sorted(nodes)
            community_details.append({
                'Window_Number': win_num,
                'community_ID': comm_id,
                'community_Size': len(sorted_nodes),
                'Brain_Region_Nodes': ','.join(map(str, sorted_nodes))
            })
            comm_dict[(int(comm_id), win_num)] = np.array(sorted_nodes, dtype=np.int32)

        if prev_communities is not None:
            evolution_events = detect_evolution_events(
                prev_communities, communities, window_idx,
                sim_threshold, expansion_th, contraction_th
            )
            all_evolution_events.extend(evolution_events)

        prev_communities = communities

    return all_evolution_events, community_details, comm_dict


# 4. Event-specific COC Calculation
def calculate_cooccurrence_matrices_in_memory(all_evolution_events, comm_dict, n_windows):

    num_transitions = max(1, n_windows - 1)

    matrices = {
        et: np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
        for et in EVENT_TYPES
    }

    for event in all_evolution_events:
        ev_type = str(event['Event_Type']).lower()
        if ev_type not in matrices:
            continue

        win_num = int(event['Window_Number'])
        prev_id = event['Prev_community_ID']
        curr_id = event['Curr_community_ID']

        prev_id = int(prev_id) if prev_id is not None and pd.notna(prev_id) else -1
        curr_id = int(curr_id) if curr_id is not None and pd.notna(curr_id) else -1

        if ev_type == 'emergence' and curr_id != -1:
            nodes = comm_dict.get((curr_id, win_num), np.array([], dtype=np.int32))
            if len(nodes) >= 2:
                nodes = nodes[(nodes >= 0) & (nodes < NUM_NODES)]
                matrices['emergence'][np.ix_(nodes, nodes)] += 1.0

        elif ev_type == 'disappearance' and prev_id != -1:
            nodes = comm_dict.get((prev_id, win_num - 1), np.array([], dtype=np.int32))
            if len(nodes) >= 2:
                nodes = nodes[(nodes >= 0) & (nodes < NUM_NODES)]
                matrices['disappearance'][np.ix_(nodes, nodes)] += 1.0

        elif ev_type in ['expansion', 'contraction'] and prev_id != -1 and curr_id != -1:
            prev_nodes = comm_dict.get((prev_id, win_num - 1), np.array([], dtype=np.int32))
            curr_nodes = comm_dict.get((curr_id, win_num), np.array([], dtype=np.int32))

            if len(prev_nodes) > 0 and len(curr_nodes) > 0:
                common_nodes = np.intersect1d(prev_nodes, curr_nodes)
                common_nodes = common_nodes[(common_nodes >= 0) & (common_nodes < NUM_NODES)]

                if len(common_nodes) > 0:
                    if ev_type == 'expansion':
                        new_nodes = np.setdiff1d(curr_nodes, prev_nodes)
                        new_nodes = new_nodes[(new_nodes >= 0) & (new_nodes < NUM_NODES)]
                        if len(new_nodes) > 0:
                            matrices['expansion'][np.ix_(new_nodes, common_nodes)] += 1.0
                            matrices['expansion'][np.ix_(common_nodes, new_nodes)] += 1.0

                    elif ev_type == 'contraction':
                        removed_nodes = np.setdiff1d(prev_nodes, curr_nodes)
                        removed_nodes = removed_nodes[(removed_nodes >= 0) & (removed_nodes < NUM_NODES)]
                        if len(removed_nodes) > 0:
                            matrices['contraction'][np.ix_(removed_nodes, common_nodes)] += 1.0
                            matrices['contraction'][np.ix_(common_nodes, removed_nodes)] += 1.0

    np.fill_diagonal(matrices['emergence'], 0.0)
    np.fill_diagonal(matrices['disappearance'], 0.0)

    for key in matrices:
        matrices[key] /= float(num_transitions)

    return matrices


# 5. Monte Carlo Analysis
def run_pipeline(subject_pma_list, bold_dir, base_output_dir):
    threshold = []

    task_args = [(subject_id, bold_dir, threshold) for subject_id, _ in subject_pma_list]

    precomputed_results = {}
    with multiprocessing.Pool(processes=20) as pool:
        for subj_id, res_comm in tqdm(pool.imap_unordered(process_subject_task, task_args),
                                      total=len(task_args), desc="the progress of subject's community calculation"):
            precomputed_results[subj_id] = res_comm


    # group by PMA, and select one subject from each unique PMA group each time
    pma_groups = defaultdict(list)
    for subj, pma in subject_pma_list:
        pma_groups[pma].append(subj)

    sorted_unique_pmas = sorted(pma_groups.keys())
    n_windows = len(sorted_unique_pmas)

    num_iterations = []

    timelines_50_runs = []
    for run_idx in range(1, num_iterations + 1):
        current_timeline = []
        for window_idx, pma in enumerate(sorted_unique_pmas):
            subjs_in_this_age = pma_groups[pma]
            chosen_subj = random.choice(subjs_in_this_age)
            res_comm = precomputed_results[chosen_subj]
            current_timeline.append((window_idx, chosen_subj, pma, res_comm))
        timelines_50_runs.append(current_timeline)

    configs = [
        {'name': 'sim0.4_sz1.02_0.98', 'sim_th': [], 'expansion_th': [], 'contraction_th': []},
    ]


    for config in configs:
        print(
            f"\n>>> 运行参数组 [{config['name']}]: Sim={config['sim_th']}, "
            f"Expansion={config['expansion_th']}, Contraction={config['contraction_th']}"
        )

        current_output_dir = f"{base_output_dir}_{config['name']}"
        os.makedirs(current_output_dir, exist_ok=True)

        accumulated_matrices = {
            et: np.zeros((NUM_NODES, NUM_NODES), dtype=np.float64)
            for et in EVENT_TYPES
        }

        for i, current_timeline in enumerate(tqdm(timelines_50_runs, desc="Monte Carlo simulation progress")):
            run_id = i + 1
            run_out_dir = os.path.join(current_output_dir, f"run_{run_id}")
            os.makedirs(run_out_dir, exist_ok=True)

            # 1. track the evolution events and community details
            all_evolution_events, community_details, comm_dict = track_evolution_timeline(
                current_timeline,
                sim_threshold=config['sim_th'],
                expansion_th=config['expansion_th'],
                contraction_th=config['contraction_th']
            )

            # 2. save the details of the current round of events and communities
            if all_evolution_events:
                events_df = pd.DataFrame(all_evolution_events)
                events_df.to_csv(
                    os.path.join(run_out_dir, f"evolution_events_{run_id}.csv"),
                    index=False, encoding='utf-8-sig'
                )

            if community_details:
                sub_df = pd.DataFrame(community_details)
                sub_df.to_csv(
                    os.path.join(run_out_dir, f"subcommunity_details_{run_id}.csv"),
                    index=False, encoding='utf-8-sig'
                )

            # 3. Calculating COC and normalize it
            matrices = calculate_cooccurrence_matrices_in_memory(
                all_evolution_events, comm_dict, n_windows=n_windows
            )


            for key in EVENT_TYPES:
                accumulated_matrices[key] += matrices[key]

            del matrices, comm_dict, all_evolution_events, community_details
            gc.collect()

        # average Monte Carlo results and save
            avg_matrix = (accumulated_matrices[key] / float(num_iterations)).astype(np.float32)
            matrix_file = os.path.join(current_output_dir, f"average_{key}_COC.mat")
            savemat(matrix_file, {'matrix': avg_matrix}, do_compression=True)
            del avg_matrix

        del accumulated_matrices
        gc.collect()

def main_population(bold_dir, pma_file, base_output_dir):

    if not os.path.exists(pma_file):
        return

    pma_data = sio.loadmat(pma_file)
    pma_values = pma_data[''].flatten()

    subject_pma_list = [(i + 1, pma_values[i]) for i in range(len(pma_values))]

    run_pipeline(subject_pma_list, bold_dir, base_output_dir)


if __name__ == "__main__":
    bold_data_dir = r""
    pma_mat_file = r""
    base_output_dir = r""

    if os.path.exists(bold_data_dir):
        main_population(bold_data_dir, pma_mat_file, base_output_dir)