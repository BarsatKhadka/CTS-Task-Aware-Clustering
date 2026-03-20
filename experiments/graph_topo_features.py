"""
Extract graph topology features from processed_graphs/*.pt files.
These are STRUCTURE-level features (not node features which are z-scored).

Features:
- num_nodes, num_wire_edges, num_skip_edges (counts)
- Wire graph: avg_degree, max_degree, density, clustering proxy
- Skip graph: connectivity, path length proxies
- Per-FF area statistics from raw_areas (NOT z-scored)
"""

import torch
import os
import numpy as np
import pickle
import pandas as pd

BASE = '/home/rain/CTS-Task-Aware-Clustering'
GRAPH_DIR = f'{BASE}/processed_graphs'
GRAPH_FEAT_CACHE = f'{BASE}/absolute_v4_graph_cache.pkl'


def extract_graph_features(pt_path):
    """Extract structural + raw features from a .pt graph file."""
    try:
        data = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception:
        return None

    num_nodes = int(data.get('num_nodes', data['X'].shape[0]))
    X = data['X'].numpy() if isinstance(data['X'], torch.Tensor) else data['X']

    def get_csr_degrees(A):
        """Extract row degrees from either scipy CSR or torch sparse CSR tensor."""
        if A is None:
            return None, 0
        if isinstance(A, torch.Tensor):
            # Torch sparse CSR: crow_indices, col_indices, values
            crow = A.crow_indices().numpy()
            nnz = int(A._nnz())
            # Row degrees = diff of crow_indices
            degrees = np.diff(crow).astype(float)
            return degrees, nnz
        else:
            import scipy.sparse as sp
            if not hasattr(A, 'nnz'):
                try:
                    A = sp.csr_matrix(A)
                except Exception:
                    return None, 0
            degrees = np.array(A.sum(axis=1)).flatten()
            return degrees, A.nnz

    # Wire adjacency
    A_wire = data.get('A_wire_csr')
    degrees, n_wire_edges = get_csr_degrees(A_wire)
    if degrees is not None and len(degrees) > 0:
        avg_degree = degrees.mean()
        max_degree = float(degrees.max())
        std_degree = degrees.std()
        frac_high_degree = (degrees >= 10).mean()
    else:
        avg_degree = max_degree = std_degree = frac_high_degree = 0.0
        n_wire_edges = 0

    # Skip adjacency
    A_skip = data.get('A_skip_csr')
    skip_degrees, n_skip_edges = get_csr_degrees(A_skip)
    if skip_degrees is not None and len(skip_degrees) > 0:
        avg_skip_degree = skip_degrees.mean()
        max_skip_degree = float(skip_degrees.max())
        frac_connected = (skip_degrees > 0).mean()
    else:
        avg_skip_degree = max_skip_degree = frac_connected = 0.0
        n_skip_edges = 0

    # Raw areas (NOT z-scored)
    raw_areas = data.get('raw_areas')
    if raw_areas is not None:
        if isinstance(raw_areas, torch.Tensor):
            raw_areas = raw_areas.numpy()
        total_area = float(raw_areas.sum())
        mean_area = float(raw_areas.mean())
        std_area = float(raw_areas.std())
        max_area = float(raw_areas.max())
    else:
        total_area = mean_area = std_area = max_area = 0.0

    # Node feature STATISTICS (z-scored but relative structure is preserved)
    # X[:,10] = is_sequential, X[:,11] = is_buffer (binary, NOT z-scored)
    is_seq = X[:, 10]   # binary: 1 = FF
    is_buf = X[:, 11]   # binary: 1 = buffer
    n_seq = is_seq.sum()
    n_buf_nodes = is_buf.sum()
    frac_seq = n_seq / (num_nodes + 1)
    frac_buf_nodes = n_buf_nodes / (num_nodes + 1)

    # Toggle feature (z-scored but distributional shape matters)
    toggle_z = X[:, 12]  # z-scored toggle_count
    toggle_p90 = np.percentile(toggle_z, 90)
    toggle_std = toggle_z.std()
    toggle_max = toggle_z.max()

    # Capacitance (z-scored) - distributional shape
    cap_z = X[:, 8]  # total_pin_cap*1000 z-scored
    cap_mean = cap_z.mean()
    cap_p90 = np.percentile(cap_z, 90)
    cap_max = cap_z.max()

    return {
        'num_nodes': num_nodes,
        'n_wire_edges': n_wire_edges,
        'n_skip_edges': n_skip_edges,
        'avg_wire_degree': avg_degree,
        'max_wire_degree': max_degree,
        'std_wire_degree': std_degree,
        'frac_high_degree': frac_high_degree,
        'avg_skip_degree': avg_skip_degree,
        'max_skip_degree': max_skip_degree,
        'frac_skip_connected': frac_connected,
        'wire_density': n_wire_edges / (num_nodes * num_nodes + 1),
        'skip_density': n_skip_edges / (num_nodes * num_nodes + 1),
        'skip_wire_ratio': n_skip_edges / (n_wire_edges + 1),
        'total_area': total_area,
        'mean_area': mean_area,
        'std_area': std_area,
        'max_area': max_area,
        'n_seq': n_seq,
        'frac_seq': frac_seq,
        'frac_buf_nodes': frac_buf_nodes,
        'toggle_p90_z': toggle_p90,
        'toggle_std_z': toggle_std,
        'toggle_max_z': toggle_max,
        'cap_mean_z': cap_mean,
        'cap_p90_z': cap_p90,
        'cap_max_z': cap_max,
        # Log-scaled counts
        'log_num_nodes': np.log1p(num_nodes),
        'log_n_wire_edges': np.log1p(n_wire_edges),
        'log_n_skip_edges': np.log1p(n_skip_edges),
        'log_total_area': np.log1p(total_area),
    }


def build_graph_cache(df):
    if os.path.exists(GRAPH_FEAT_CACHE):
        with open(GRAPH_FEAT_CACHE, 'rb') as f:
            cache = pickle.load(f)
        print(f"Loaded graph cache: {len(cache)} entries")
        return cache

    unique_pids = df['placement_id'].unique()
    cache = {}
    missing = 0

    print(f"Building graph feature cache for {len(unique_pids)} placements...")
    for i, pid in enumerate(unique_pids):
        pt_path = os.path.join(GRAPH_DIR, f'{pid}.pt')
        if not os.path.exists(pt_path):
            missing += 1
            continue
        feats = extract_graph_features(pt_path)
        if feats:
            cache[pid] = feats
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(unique_pids)}, found={len(cache)}, missing={missing}")

    with open(GRAPH_FEAT_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Graph cache saved: {len(cache)} entries, {missing} missing PT files")
    return cache


if __name__ == '__main__':
    df = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
    cache = build_graph_cache(df)

    # Print sample features
    pid = list(cache.keys())[0]
    print(f"\nSample features for {pid}:")
    for k, v in cache[pid].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
