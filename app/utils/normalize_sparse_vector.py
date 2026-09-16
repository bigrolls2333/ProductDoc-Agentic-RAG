"""Sparse-vector normalization helper."""

import numpy as np

def normalize_sparse_vector(sparse_vec):
    """Return an L2-normalized sparse vector while preserving its keys."""
    if not sparse_vec:
        return sparse_vec

    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:
        return sparse_vec

    normalized_values = values / l2_norm
    return dict(zip(sparse_vec.keys(), normalized_values))
