import numpy as np 
import torch
import scipy.sparse as sp
import torch.nn.functional as F




def normalize_features(nodes, die_x_min, die_y_min, die_x_max, die_y_max):
    die_w = die_x_max - die_x_min
    die_h = die_y_max - die_y_min
    
    cell_ids = []
    numeric = []
    
    for n in nodes:
        row = [
            (n['x'] - die_x_min) / die_w,
            (n['y'] - die_y_min) / die_h,
            n['dist_to_boundaries'][0] / die_w,
            n['dist_to_boundaries'][1] / die_w,
            n['dist_to_boundaries'][2] / die_h,
            n['dist_to_boundaries'][3] / die_h,
            np.log1p(n['cell_area']),
            n['avg_pin_cap'] * 1000,
            n['total_pin_cap'] * 1000,
            np.log2(max(n['drive_strength'], 1)),
            float(n['is_sequential']),
            float(n['is_buffer']),
            n['toggle_count'],
            n['sum_toggle_count'],
            n['signal_prob'],
            float(n['non_zero_count']),
            np.log1p(n['fan_in']),
            np.log1p(n['fan_out']),
        ]
        numeric.append(row)
        cell_ids.append(n['cell_type_id'])
    
    features = torch.tensor(numeric, dtype=torch.float32)
    cell_type_ids = torch.tensor(cell_ids, dtype=torch.long)
    
    # Standardize all columns except binary flags (indices 10, 11)
    binary_cols = {10, 11}
    norm_stats = {}
    
    for col in range(features.shape[1]):
        if col in binary_cols:
            continue
        mean = features[:, col].mean()
        std = features[:, col].std()
        if std > 1e-8:
            features[:, col] = (features[:, col] - mean) / std
            norm_stats[col] = (mean.item(), std.item())
    
    return features, cell_type_ids, norm_stats



def build_X_hop_mask(n_nodes, undirected_edges, hop_mask_len=3):
    """
    Creates the Omega mask. 
    If Omega[i, j] = 1, Node i is allowed to 'use' Node j for reconstruction.
    """
    # 1. Slice out the Source (rows) and Destination (cols) nodes
    u_rows = undirected_edges[:, 0]
    u_cols = undirected_edges[:, 1]
    
    # 2. Build the initial 1-hop Sparse Adjacency Matrix (A)
    # Using bool to save memory (we only care IF a wire exists)
    A = sp.csr_matrix((np.ones(len(u_rows), dtype=bool), (u_rows, u_cols)), 
                      shape=(n_nodes, n_nodes))
    
    # 3. Matrix Power Expansion (Find 2-hop and 3-hop neighbors)
    omega = A.copy()
    temp = A.copy()
    
    for i in range(2, hop_mask_len + 1):
        temp = temp.dot(A)
        omega = omega + temp
        print(f"  -> Hop {i} expansion complete.")

    # 4) diag(P) = 0
    # A node cannot reconstruct itself
    omega.setdiag(0)
    omega.eliminate_zeros()
    
    # 5. Convert back to coordinates for PyTorch
    omega_coo = omega.tocoo()
    
    print(f"Mask created! {omega_coo.nnz:,} total connections allowed.")
    return omega_coo.row, omega_coo.col


def get_compressed_graph(X, C, A_skip_csr, A_wire_csr):
    """
    Consolidated function to build the compressed graph with Section 6 traits.
    
    Args:
        X: [N, d] Raw node features (Coordinates at 0:2, Cell Area at 2).
        C: [N, k] Soft assignment matrix[cite: 104, 109].
        A_skip_csr: [N, N] Sparse timing highways[cite: 140, 141].
        A_wire_csr: [N, N] Sparse physical routing grid[cite: 174].
    """
    # 1. Column Normalization for Weighted Averaging
    # collapsible Node dimension N into Cluster dimension K [cite: 171, 175]
    cluster_sums = C.sum(dim=0, keepdim=True) + 1e-8
    C_norm = C / cluster_sums
    
    # 2. Base Supernode Features (Section 11.1) 
    X_tilde_base = torch.matmul(C_norm.t(), X) 

    # 3. Cluster physical traits (Section 6) [cite: 54]
    # Extract raw data from indices: 0:2=Coords, 2=Cell Area [cite: 27, 30]
    coords = X[:, 0:2]    
    cell_areas = X[:, 2:3]

    # Cluster 'Count': Weighted sum of cell areas [cite: 56]
    counts = torch.matmul(C.t(), cell_areas) 

    # Cluster 'Spread': Geometric standard deviation (Weighted Variance) [cite: 56]
    centroids = torch.matmul(C_norm.t(), coords) 
    node_sq = torch.sum(coords**2, dim=1, keepdim=True)
    exp_sq = torch.matmul(C_norm.t(), node_sq) 
    mu_sq = torch.sum(centroids**2, dim=1, keepdim=True)
    spreads = torch.sqrt(F.relu(exp_sq - mu_sq) + 1e-8)

    # 4. Final X_tilde: Aggregated Features + Section 6 additions [cite: 171, 172]
    X_tilde = torch.cat([X_tilde_base, counts, spreads], dim=-1)

    # 5. Compressed Adjacencies (Section 11.2, 11.3)
    # Timing Highways [cite: 179]
    inter_skip = torch.sparse.mm(A_skip_csr, C)
    A_tilde_skip = torch.matmul(C.t(), inter_skip)
    
    # Physical Routing [cite: 175]
    inter_wire = torch.sparse.mm(A_wire_csr, C)
    A_tilde_wire = torch.matmul(C.t(), inter_wire)
    
    return X_tilde, A_tilde_skip, A_tilde_wire


def relative_masking(A_dense, threshold=0.10):
    with torch.no_grad():
        # 1. Clone and zero-out the diagonal to find true EXTERNAL peak
        A_no_diag = A_dense.clone()
        A_no_diag.fill_diagonal_(0.0)
        
        # 2. Find the peak OUTGOING connection
        max_vals, _ = A_no_diag.max(dim=1, keepdim=True)
        row_thresholds = max_vals * threshold
        
        # 3. Create mask (keep external wires above threshold)
        mask = (A_dense >= row_thresholds) & (A_dense > 1e-9)
        
        # 4. Force-keep the self-loops (the diagonal)
        eye = torch.eye(A_dense.size(0), device=A_dense.device, dtype=torch.bool)
        mask = mask | eye
        
    # 5. Apply the mask (Preserves gradients on surviving weights)
    A_sparse = A_dense * mask.float()
    
    # 6. Extract into PyTorch Geometric format immediately
    row_idx, col_idx = (A_sparse > 0).nonzero(as_tuple=True)
    edge_index = torch.stack([row_idx, col_idx], dim=0)  # Shape: [2, num_edges]
    edge_weight = A_sparse[row_idx, col_idx]             # Shape: [num_edges]
    
    return edge_index, edge_weight