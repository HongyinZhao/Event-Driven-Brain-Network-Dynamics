# This code calculates intrinsic event-specific co-occurrence connectivity (COC).
# Individual BOLD time series are divided into non-overlapping windows.
# The analysis includes FC, Infomap communities, evolution events and subject/group COC.
# Set the data paths and numerical analysis parameters below before running.

import os
import gc
import glob
import re
import multiprocessing
import concurrent.futures
from collections import defaultdict
import numpy as np
import pandas as pd
import infomap
import scipy.io as sio
from scipy.io import loadmat, savemat
from tqdm import tqdm

# %% Analysis parameters and file paths
BOLD_DATA_DIR = r""     # Folder containing sub_BOLD_<subject_id>.mat
OUTPUT_DIR = r""        # Folder for individual COC output
POPULATION_AVG_DIR = os.path.join(OUTPUT_DIR, "Population_Average_Matrices")

NUM_NODES = []           # Number of cortical vertices
TOTAL_SUBJECTS = []     # Fallback subject count when files are not discovered
WINDOW_SIZE = []         # Volumes per non-overlapping window
CORR_THRESHOLD = []      # Fisher-z threshold for Infomap
SIMILARITY_THRESHOLD = []   # Jaccard similarity threshold
EXPANSION_THRESHOLD = []   # Community-size ratio for expansion
CONTRACTION_THRESHOLD = [] # Community-size ratio for contraction
MAX_WORKERS = []         # Parallel processes for individual analysis
MAX_AVERAGE_WORKERS = [] # Maximum processes for group averaging

EVENT_TYPES = ['emergence', 'disappearance', 'expansion', 'contraction']


# %% Calculation of window-wise Pearson FC and Fisher-z transformation
def calculate_pearson_fc(window_data):
    X_mean = window_data.mean(axis=1, keepdims=True)
    X_centered = window_data - X_mean
    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    X_norm[X_norm == 0] = 1e-10
    X_normalized = X_centered / X_norm
    window_r = np.dot(X_normalized, X_normalized.T)
    return window_r


def fisher_z_transform(window_r):
    window_r = np.clip(window_r, -0.9999, 0.9999)
    fisher_z_matrix = np.arctanh(window_r)
    return fisher_z_matrix


# %% Detection of Infomap communities
def detect_communities_in_window(fisher_z_matrix, threshold):
    n_nodes = fisher_z_matrix.shape[0]
    work_matrix = fisher_z_matrix.copy()
    np.fill_diagonal(work_matrix, 0)

    rows, cols = np.triu_indices_from(work_matrix, k=1)
    weights = work_matrix[rows, cols]

    # Retain positive FC values above the selected threshold.
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


# %% Identification of community evolution events
def calculate_similarity(nodes_set1, nodes_set2):
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


def detect_evolution_events(prev_communities, curr_communities, window_idx, threshold):
    events = []
    if not prev_communities or not curr_communities:
        return events

    prev_groups = group_nodes_by_community(prev_communities)
    curr_groups = group_nodes_by_community(curr_communities)

    # A current community is emergent if it lacks a strong prior match.
    for curr_id, curr_nodes in curr_groups.items():
        curr_nodes_set = set(curr_nodes)
        best_similarity = 0
        for prev_id, prev_nodes in prev_groups.items():
            similarity = calculate_similarity(curr_nodes_set, set(prev_nodes))
            if similarity > best_similarity:
                best_similarity = similarity

        if best_similarity < threshold:
            events.append({
                'Window_Number': window_idx + 1,
                'Event_Type': 'emergence',
                'Prev_community_ID': None,
                'Curr_community_ID': curr_id,
                'Similarity': best_similarity,
                'Node_Count': len(curr_nodes)
            })

    # A prior community disappears when no current match qualifies.
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

        if best_similarity < threshold:
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

            if ratio > EXPANSION_THRESHOLD:
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

            elif ratio < CONTRACTION_THRESHOLD:
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


# %% Calculation of individual event-specific COC matrices
def build_and_save_cooccurrence_matrices(all_evolution_events, comm_dict, subject_out_dir, n_windows, num_nodes):

    num_transitions = max(1, n_windows - 1)

    for event_type in EVENT_TYPES:
        matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for ev in all_evolution_events:
            if str(ev['Event_Type']).lower() != event_type:
                continue

            win_num = int(ev['Window_Number'])
            prev_id = ev['Prev_community_ID']
            curr_id = ev['Curr_community_ID']

            prev_id = int(prev_id) if prev_id is not None and pd.notna(prev_id) else -1
            curr_id = int(curr_id) if curr_id is not None and pd.notna(curr_id) else -1

            if event_type == 'emergence':
                if curr_id == -1:
                    continue
                nodes = comm_dict.get((curr_id, win_num), np.array([], dtype=np.int32))
                if len(nodes) >= 2:
                    nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
                    matrix[np.ix_(nodes, nodes)] += 1.0

            elif event_type == 'disappearance':
                if prev_id == -1:
                    continue
                nodes = comm_dict.get((prev_id, win_num - 1), np.array([], dtype=np.int32))
                if len(nodes) >= 2:
                    nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
                    matrix[np.ix_(nodes, nodes)] += 1.0

            # Count links between changing and persisting community members.
            elif event_type in ['expansion', 'contraction']:
                if prev_id == -1 or curr_id == -1:
                    continue
                prev_nodes = comm_dict.get((prev_id, win_num - 1), np.array([], dtype=np.int32))
                curr_nodes = comm_dict.get((curr_id, win_num), np.array([], dtype=np.int32))

                if len(prev_nodes) > 0 and len(curr_nodes) > 0:
                    common_nodes = np.intersect1d(prev_nodes, curr_nodes)
                    common_nodes = common_nodes[(common_nodes >= 0) & (common_nodes < num_nodes)]

                    if len(common_nodes) > 0:
                        if event_type == 'expansion':
                            target_nodes = np.setdiff1d(curr_nodes, prev_nodes)
                        else:
                            target_nodes = np.setdiff1d(prev_nodes, curr_nodes)

                        target_nodes = target_nodes[(target_nodes >= 0) & (target_nodes < num_nodes)]
                        if len(target_nodes) > 0:
                            matrix[np.ix_(target_nodes, common_nodes)] += 1.0
                            matrix[np.ix_(common_nodes, target_nodes)] += 1.0

        if event_type in ['emergence', 'disappearance']:
            np.fill_diagonal(matrix, 0.0)

        matrix /= float(num_transitions)

        matrix_file = os.path.join(subject_out_dir, f"{event_type}_COC.mat")
        savemat(matrix_file, {'matrix': matrix}, do_compression=True)

        del matrix
        gc.collect()


# %% Processing of individual BOLD time series
def process_individual_subject(args):
    subject_id, bold_dir, output_dir, threshold, num_nodes = args
    file_path = os.path.join(bold_dir, f"sub_BOLD_{subject_id}.mat")
    subject_out_dir = os.path.join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_out_dir, exist_ok=True)

    if not os.path.exists(file_path):
        return subject_id, False, None

    try:
        mat_data = sio.loadmat(file_path)
        var_names = [k for k in mat_data.keys() if not k.startswith('__')]
        if not var_names:
            return subject_id, False, None

        bold_signal = mat_data[var_names[0]]
        _, n_time = bold_signal.shape
        # Discard any trailing volumes that do not form a complete window.
        n_windows = n_time // WINDOW_SIZE

        if n_windows < 2:
            return subject_id, False, None

        subject_timeline_results = []

        for w in range(n_windows):
            start_idx = w * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            window_data = bold_signal[:, start_idx:end_idx]

            window_r = calculate_pearson_fc(window_data)
            fisher_z_matrix = fisher_z_transform(window_r)
            del window_r

            comms = detect_communities_in_window(fisher_z_matrix, threshold)
            del fisher_z_matrix
            subject_timeline_results.append(comms)

        del bold_signal

        all_evolution_events = []
        community_details = []
        comm_dict = {}

        prev_communities = None
        for window_idx, communities in enumerate(subject_timeline_results):
            win_num = window_idx + 1

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
                events = detect_evolution_events(
                    prev_communities, communities, window_idx, threshold=SIMILARITY_THRESHOLD
                )
                all_evolution_events.extend(events)

            prev_communities = communities

        if all_evolution_events:
            events_df = pd.DataFrame(all_evolution_events)
            events_df.to_csv(os.path.join(subject_out_dir, f"evolution_events_{subject_id}.csv"),
                             index=False, encoding='utf-8-sig')

        if community_details:
            sub_df = pd.DataFrame(community_details)
            sub_df.to_csv(os.path.join(subject_out_dir, f"subcommunity_details_{subject_id}.csv"),
                          index=False, encoding='utf-8-sig')

        build_and_save_cooccurrence_matrices(
            all_evolution_events, comm_dict, subject_out_dir, n_windows=n_windows, num_nodes=num_nodes
        )

        return subject_id, True, "successful"

    except Exception as e:
        return subject_id, False, None


# %% Calculation of group-averaged intrinsic COC
def process_chunk_average(args):
    event_type, chunk_ids, base_dir, num_nodes = args
    chunk_sum_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    valid_count = 0

    for person_id in chunk_ids:
        person_dir = os.path.join(base_dir, f"subject_{person_id}")
        mat_file = os.path.join(person_dir, f"{event_type}_COC.mat")

        if os.path.exists(mat_file):
            try:
                mat_data = loadmat(mat_file)
                chunk_sum_matrix += mat_data['matrix']
                valid_count += 1
                del mat_data
            except Exception:
                pass

    return chunk_sum_matrix, valid_count


def compute_population_average_matrices(base_dir, output_avg_dir, subject_ids, num_nodes, max_workers):
    total_subjects = len(subject_ids)
    merge_workers = min(max_workers, MAX_AVERAGE_WORKERS)

    os.makedirs(output_avg_dir, exist_ok=True)

    chunk_size = (total_subjects + merge_workers - 1) // merge_workers
    chunks = [subject_ids[i:i + chunk_size] for i in range(0, total_subjects, chunk_size)]

    for event_type in EVENT_TYPES:

        total_sum_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        total_valid_count = 0

        args_list = [(event_type, chunk, base_dir, num_nodes) for chunk in chunks]

        with concurrent.futures.ProcessPoolExecutor(max_workers=merge_workers) as executor:
            futures = [executor.submit(process_chunk_average, arg) for arg in args_list]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Merging {event_type}"):
                chunk_sum, chunk_valid = future.result()
                total_sum_matrix += chunk_sum
                total_valid_count += chunk_valid
                del chunk_sum

        # Average only the matrices successfully loaded for this event.
        if total_valid_count > 0:
            avg_matrix = total_sum_matrix / total_valid_count
            output_file = os.path.join(output_avg_dir, f"average_{event_type}_COC.mat")
            savemat(output_file, {'matrix': avg_matrix.astype(np.float32)}, do_compression=True)

            print(f"{event_type}: average saved for {total_valid_count} subjects")
            del avg_matrix

        del total_sum_matrix
        gc.collect()


# %% Loading of subjects and execution
def get_subject_ids(bold_dir):
    mat_files = glob.glob(os.path.join(bold_dir, "sub_BOLD_*.mat"))
    subject_ids = []
    for f in mat_files:
        m = re.search(r"sub_BOLD_(\d+)\.mat", os.path.basename(f))
        if m:
            subject_ids.append(int(m.group(1)))
    if subject_ids:
        return sorted(subject_ids)
    return list(range(1, TOTAL_SUBJECTS + 1))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subject_ids = get_subject_ids(BOLD_DATA_DIR)
    total_subjects = len(subject_ids)
    print(f"Subjects found: {total_subjects}")

    task_args = [
        (subject_id, BOLD_DATA_DIR, OUTPUT_DIR, CORR_THRESHOLD, NUM_NODES)
        for subject_id in subject_ids
    ]

    print(f"Computing individual COC for {total_subjects} subjects...")
    success_count = 0
    skipped_count = 0

    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        for subj_id, status, msg in tqdm(pool.imap_unordered(process_individual_subject, task_args),
                                         total=len(task_args), desc="handling individual data"):
            if status:
                success_count += 1
            else:
                skipped_count += 1

    print(f"Completed: {success_count}/{total_subjects}; skipped: {skipped_count}")

    compute_population_average_matrices(
        base_dir=OUTPUT_DIR,
        output_avg_dir=POPULATION_AVG_DIR,
        subject_ids=subject_ids,
        num_nodes=NUM_NODES,
        max_workers=MAX_WORKERS
    )

    print("Intrinsic COC analysis complete.")


if __name__ == "__main__":
    main()
