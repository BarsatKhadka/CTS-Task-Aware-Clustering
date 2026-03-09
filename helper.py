import numpy as np 
import torch
import scipy.sparse as sp




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

#compressed graph
def get_compressed_graph(X, C, A_skip_csr, A_wire_csr):
    """
    X: [N, feature_dim] 
    C: [N, current_k] (Soft assignments)
    A_skip_csr: [N, N] sparse matrix (Virtual timing highways)
    A_wire_csr: [N, N] sparse matrix (1-hop physical routing grid)
    """
    with torch.no_grad():
        pass # Optional hard assignment hook

    # 1. Supernode Features (X_tilde) -> [current_k, feature_dim]
    C_norm = C / (C.sum(dim=0, keepdim=True) + 1e-8)
    X_tilde = torch.matmul(C_norm.t(), X)
    
    # 2. Compressed Skip-Connection Adjacency (A_tilde_skip) -> [current_k, current_k]
    # As defined in Section 11.3: A_tilde_skip = C^T * A_skip * C
    inter_skip = torch.sparse.mm(A_skip_csr, C)
    A_tilde_skip = torch.matmul(C.t(), inter_skip)
    
    # 3. Compressed Physical Routing Adjacency (A_tilde_wire) -> [current_k, current_k]
    # As defined in Section 11.2: A_tilde_wire = C^T * A_wire * C
    inter_wire = torch.sparse.mm(A_wire_csr, C)
    A_tilde_wire = torch.matmul(C.t(), inter_wire)
    
    return X_tilde, A_tilde_skip, A_tilde_wire