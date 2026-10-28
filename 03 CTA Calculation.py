# This code calculates cross-timescale alignment (CTA) of the neonatal connectome.
# It compares individual intrinsic COC with developmental population-level COC.
# Global CTA uses upper-triangular connections; vertex CTA uses corresponding rows.
# Set the data paths and numerical analysis parameters below before running.

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm

# %% Analysis parameters and file paths
DEVELOPMENTAL_COC_DIR = ""  # Developmental COC matrices
DEVELOPMENTAL_FILE_PREFIX = ""  # Prefix of the developmental COC filenames
INTRINSIC_COC_DIR = ""      # Subject-level intrinsic COC matrices
OUTPUT_DIR = ""             # Folder for CTA results
EVENT_TYPES = ["emergence", "disappearance", "expansion", "contraction"]
NUM_SUBJECTS = []      # Number of participants
MAX_WORKERS = []       # Parallel processes


# %% Loading and preparation of developmental COC
def initialize_worker(pop_file, worker_ind_dir, worker_out_dir):
    global ind_dir, out_dir
    ind_dir = worker_ind_dir
    out_dir = worker_out_dir

    global pop_centered, pop_row_sum_sq, pop_row_names, pop_row_means
    global pop_upper_sum, pop_upper_sum_sq, n_upper_edges

    mat_data = sio.loadmat(pop_file)
    matrix_key = next(k for k in mat_data if not k.startswith("__"))
    pop_matrix = np.asarray(mat_data[matrix_key], dtype=np.float64)

    if pop_matrix.ndim != 2 or pop_matrix.shape[0] != pop_matrix.shape[1]:
        raise ValueError(f"Population COC must be a square matrix: {pop_file}")

    n = pop_matrix.shape[0]
    n_upper_edges = n * (n - 1) // 2
    pop_upper_sum = 0.0
    pop_upper_sum_sq = 0.0

    # Use the strict upper triangle without diagonal elements.
    for i in range(n - 1):
        upper_row = pop_matrix[i, i + 1:]
        pop_upper_sum += np.sum(upper_row)
        pop_upper_sum_sq += np.dot(upper_row, upper_row)

    # Cache centered reference rows for vertex-level Pearson CTA.
    pop_row_names = [f"Row_{i + 1}" for i in range(n)]
    pop_row_means = np.mean(pop_matrix, axis=1)
    pop_matrix -= pop_row_means[:, None]
    pop_centered = pop_matrix
    pop_row_sum_sq = np.einsum("ij,ij->i", pop_centered, pop_centered)


# %% Calculation of global CTA
def calculate_matrix_pearson(ind_matrix):
    n = ind_matrix.shape[0]
    ind_upper_sum = 0.0
    ind_upper_sum_sq = 0.0
    cross_upper_sum = 0.0

    # Compute upper-triangle correlation without materializing its indices.
    for i in range(n - 1):
        upper_row = ind_matrix[i, i + 1:]
        row_sum = np.sum(upper_row)
        ind_upper_sum += row_sum
        ind_upper_sum_sq += np.dot(upper_row, upper_row)
        cross_upper_sum += (
            np.dot(pop_centered[i, i + 1:], upper_row)
            + pop_row_means[i] * row_sum
        )

    pop_ss = pop_upper_sum_sq - pop_upper_sum ** 2 / n_upper_edges
    ind_ss = ind_upper_sum_sq - ind_upper_sum ** 2 / n_upper_edges
    if pop_ss <= 0 or ind_ss <= 0:
        return np.nan

    covariance = cross_upper_sum - pop_upper_sum * ind_upper_sum / n_upper_edges
    return float(np.clip(covariance / np.sqrt(pop_ss * ind_ss), -1.0, 1.0))


# %% Calculation of individual global and vertex-level CTA
def process_person(person_id, m_type):
    ind_file = os.path.join(ind_dir, f"subject_{person_id}", f"{DEVELOPMENTAL_FILE_PREFIX}{m_type}_COC.mat")
    if not os.path.exists(ind_file):
        return person_id, np.nan, "unavailable"

    try:
        mat_data = sio.loadmat(ind_file)
        matrix_key = next(k for k in mat_data if not k.startswith("__"))
        ind_matrix = np.asarray(mat_data[matrix_key], dtype=np.float64)
        if ind_matrix.shape != pop_centered.shape:
            raise ValueError(f"Matrix shape mismatch: {ind_matrix.shape} vs {pop_centered.shape}")

        matrix_r = calculate_matrix_pearson(ind_matrix)

        # Row-wise CTA correlates the same vertex in both COC matrices.
        ind_matrix -= np.mean(ind_matrix, axis=1, keepdims=True)
        numerator = np.einsum("ij,ij->i", pop_centered, ind_matrix)
        ind_row_sum_sq = np.einsum("ij,ij->i", ind_matrix, ind_matrix)
        denominator = np.sqrt(pop_row_sum_sq * ind_row_sum_sq)
        with np.errstate(divide="ignore", invalid="ignore"):
            vertex_r = numerator / denominator
        vertex_r[denominator == 0] = np.nan

        person_out_dir = os.path.join(out_dir, f"subject_{person_id}")
        os.makedirs(person_out_dir, exist_ok=True)
        out_file = os.path.join(person_out_dir, f"{m_type}_vertex_level_CTA.csv")
        pd.DataFrame({
            "Row_Name": pop_row_names,
            f"{m_type}_r": vertex_r,
        }).to_csv(out_file, index=False)

        return person_id, matrix_r, None
    except Exception:
        return person_id, np.nan, "unavailable"


# %% Calculation and saving of CTA for all subjects
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    matrix_results = {m_type: {} for m_type in EVENT_TYPES}

    for m_type in EVENT_TYPES:
        pop_file = os.path.join(DEVELOPMENTAL_COC_DIR, f"{DEVELOPMENTAL_FILE_PREFIX}{m_type}_COC.mat")
        success_count = 0
        with ProcessPoolExecutor(
            max_workers=MAX_WORKERS,
            initializer=initialize_worker,
            initargs=(pop_file, INTRINSIC_COC_DIR, OUTPUT_DIR),
        ) as executor:
            futures = {
                executor.submit(process_person, pid, m_type): pid
                for pid in range(1, NUM_SUBJECTS + 1)
            }
            for future in tqdm(as_completed(futures), total=NUM_SUBJECTS, desc=m_type):
                person_id, matrix_r, error = future.result()
                if error is None:
                    matrix_results[m_type][person_id] = matrix_r
                    success_count += 1
        print(f"{m_type}: {success_count}/{NUM_SUBJECTS} subjects completed")

    # Save subject results in their original order.
    summary = pd.DataFrame({"Subject_ID": range(1, NUM_SUBJECTS + 1)})
    for m_type in EVENT_TYPES:
        summary[f"{m_type}_r"] = [
            matrix_results[m_type].get(pid, np.nan)
            for pid in range(1, NUM_SUBJECTS + 1)
        ]
    summary.to_csv(os.path.join(OUTPUT_DIR, "matrix_level_CTA.csv"), index=False)


if __name__ == "__main__":
    main()
