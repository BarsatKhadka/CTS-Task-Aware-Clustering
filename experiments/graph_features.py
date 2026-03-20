"""
Feature extraction from processed_graphs/*.pt files.

Key insight: Per-FF features from the skip (timing path) graph are critical
for skew prediction. The FFs with highest skip-graph degree are on critical
paths and determine worst-case clock skew.

Feature groups:
  [0-17]  Skip graph: degree stats, top-FF positions
  [18-23] Wire graph: degree stats
  [24-31] Node feature aggregations: position, toggle, cap
  [32-39] Spatial aggregates
  [40-59] Physics interactions: skip × wire, position × toggle

Total: 60 features (per placement)

PERFORMANCE NOTE: Uses sparse-native operations to avoid dense materialization.
A_skip_csr is [N,N] sparse — never call .to_dense() on it.
"""

import numpy as np
import torch
from pathlib import Path

GRAPH_DIR = Path(__file__).parent / "processed_graphs"


def _gini(arr):
    arr = np.sort(arr.flatten())
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def _sparse_row_degrees(csr_tensor):
    """Compute row degrees (out-degree) from a PyTorch sparse CSR tensor without dense conversion."""
    crow = csr_tensor.crow_indices().numpy()  # [n+1]
    deg = np.diff(crow).astype(np.float32)    # [n] out-degree per row
    return deg


def _sparse_col_degrees(csr_tensor):
    """Compute column degrees (in-degree) from PyTorch sparse CSR tensor."""
    col_idx = csr_tensor.col_indices().numpy()  # [nnz]
    n = csr_tensor.shape[0]
    deg = np.bincount(col_idx, minlength=n).astype(np.float32)
    return deg


def extract_graph_features(placement_id: str) -> np.ndarray:
    """
    Extract 60 physics-informed features from a processed graph PT file.
    Uses sparse-native degree computation — no dense materialization.

    Returns zeros if the PT file doesn't exist.
    """
    pt_path = GRAPH_DIR / f"{placement_id}.pt"
    if not pt_path.exists():
        return np.zeros(60, dtype=np.float32)

    try:
        data = torch.load(str(pt_path), weights_only=False)
    except Exception:
        return np.zeros(60, dtype=np.float32)

    X = data['X'].numpy()           # [N, 18] node features
    n_nodes = X.shape[0]

    # Node feature columns:
    # 0:x_norm, 1:y_norm, 2-5:dist_boundaries, 6:log1p(area)
    # 7:avg_pin_cap*1000, 8:total_pin_cap*1000, 9:log2(drive)
    # 10:is_sequential, 11:is_buffer, 12:toggle, 13:sum_toggle
    # 14:signal_prob, 15:non_zero, 16:log1p(fan_in), 17:log1p(fan_out)

    x_pos = X[:, 0]
    y_pos = X[:, 1]
    log_area = X[:, 6]
    avg_cap = X[:, 7]
    total_cap = X[:, 8]
    drive = X[:, 9]
    is_ff = X[:, 10].astype(bool)
    is_buf = X[:, 11].astype(bool)
    toggle = X[:, 12]
    sum_toggle = X[:, 13]
    fan_in = X[:, 16]
    fan_out = X[:, 17]

    ff_idx = np.where(is_ff)[0]
    n_ff = len(ff_idx)

    # ── Skip graph features (sparse-native) ──────────────────────────────────
    A_skip = data.get('A_skip_csr', None)
    if A_skip is not None:
        skip_deg_out = _sparse_row_degrees(A_skip)
        skip_deg_in = _sparse_col_degrees(A_skip)
        skip_deg = skip_deg_out + skip_deg_in
    else:
        skip_deg = np.zeros(n_nodes)

    # Skip degree for FFs specifically
    if n_ff > 0:
        ff_skip_deg = skip_deg[ff_idx]
        ff_x = x_pos[ff_idx]
        ff_y = y_pos[ff_idx]

        # Sort FFs by skip degree (descending)
        sort_idx = np.argsort(ff_skip_deg)[::-1]
        ff_skip_sorted = ff_skip_deg[sort_idx]
        ff_x_sorted = ff_x[sort_idx]
        ff_y_sorted = ff_y[sort_idx]

        # Top critical FFs (top 5%, 10%, 25%)
        k5 = max(1, n_ff // 20)
        k10 = max(1, n_ff // 10)
        k25 = max(1, n_ff // 4)

        top5_x = ff_x_sorted[:k5]
        top5_y = ff_y_sorted[:k5]
        top10_x = ff_x_sorted[:k10]
        top10_y = ff_y_sorted[:k10]

        # Spatial spread of critical FFs (high spread → high skew)
        top5_hpwl = (top5_x.max() - top5_x.min()) + (top5_y.max() - top5_y.min()) if k5 > 1 else 0
        top10_hpwl = (top10_x.max() - top10_x.min()) + (top10_y.max() - top10_y.min()) if k10 > 1 else 0

        # Mean distance of critical FFs from centroid (skew proxy)
        ff_cx, ff_cy = ff_x.mean(), ff_y.mean()
        ff_dists = np.sqrt((ff_x - ff_cx)**2 + (ff_y - ff_cy)**2)

        top5_mean_dist = np.sqrt((top5_x - ff_cx)**2 + (top5_y - ff_cy)**2).mean() if k5 > 0 else 0
        top10_mean_dist = np.sqrt((top10_x - ff_cx)**2 + (top10_y - ff_cy)**2).mean() if k10 > 0 else 0

        # Critical FF coverage (what fraction of die area do they span)
        all_hpwl = (ff_x.max() - ff_x.min()) + (ff_y.max() - ff_y.min())
        crit_coverage = top5_hpwl / max(all_hpwl, 1e-6)

        # Skip degree stats
        skip_max = ff_skip_sorted[0] if len(ff_skip_sorted) > 0 else 0
        skip_p90 = np.percentile(ff_skip_sorted, 90) if len(ff_skip_sorted) > 0 else 0
        skip_mean = ff_skip_sorted.mean() if len(ff_skip_sorted) > 0 else 0
        skip_std = ff_skip_sorted.std() if len(ff_skip_sorted) > 0 else 0
        skip_gini = _gini(ff_skip_sorted)

        # Toggle-weighted skip degree (critical paths that are also active)
        ff_toggle = toggle[ff_idx]
        toggle_skip = ff_toggle * ff_skip_deg
        ts_mean = toggle_skip.mean() if len(toggle_skip) > 0 else 0
        ts_max = toggle_skip.max() if len(toggle_skip) > 0 else 0
        ts_p90 = np.percentile(toggle_skip, 90) if len(toggle_skip) > 0 else 0

        # Asymmetry between launch/capture FFs (affects skew)
        top_half_mask = ff_skip_sorted > skip_p90
        if top_half_mask.sum() >= 2:
            top_x = ff_x_sorted[top_half_mask[:len(ff_x_sorted)]]
            top_y = ff_y_sorted[top_half_mask[:len(ff_y_sorted)]]
            lc_asym = ((top_x.max() - top_x.min()) - (top_x.mean() - 0.5)**2)**0.5 if len(top_x) > 1 else 0
        else:
            lc_asym = 0
    else:
        top5_hpwl = top10_hpwl = 0
        top5_mean_dist = top10_mean_dist = 0
        crit_coverage = 0
        skip_max = skip_p90 = skip_mean = skip_std = skip_gini = 0
        ts_mean = ts_max = ts_p90 = 0
        lc_asym = 0
        all_hpwl = 0
        ff_dists = np.zeros(1)

    # ── Wire graph features (sparse-native) ──────────────────────────────────
    A_wire = data.get('A_wire_csr', None)
    if A_wire is not None:
        wire_deg = _sparse_row_degrees(A_wire)
    else:
        wire_deg = np.zeros(n_nodes)

    ff_wire_deg = wire_deg[ff_idx] if n_ff > 0 else np.zeros(1)
    wire_mean = ff_wire_deg.mean()
    wire_max = ff_wire_deg.max() if len(ff_wire_deg) > 0 else 0
    wire_gini = _gini(ff_wire_deg)

    # ── Node feature aggregations ────────────────────────────────────────────
    if n_ff > 0:
        ff_toggle_vals = toggle[ff_idx]
        ff_sum_toggle = sum_toggle[ff_idx]
        ff_cap = total_cap[ff_idx]

        toggle_mean = ff_toggle_vals.mean()
        toggle_std = ff_toggle_vals.std()
        toggle_max = ff_toggle_vals.max()
        toggle_gini = _gini(ff_toggle_vals[ff_toggle_vals > 0]) if (ff_toggle_vals > 0).sum() > 1 else 0

        sum_toggle_total = float(ff_sum_toggle.sum())
        cap_mean = ff_cap.mean()
        cap_std = ff_cap.std()

        # Spatial concentration of toggles
        ff_cx, ff_cy = x_pos[ff_idx].mean(), y_pos[ff_idx].mean()
        ff_dists = np.sqrt((x_pos[ff_idx] - ff_cx)**2 + (y_pos[ff_idx] - ff_cy)**2)

        # Activity-distance correlation
        dist_toggle_corr = np.corrcoef(ff_dists, ff_toggle_vals)[0, 1] if len(ff_dists) > 2 else 0
        dist_toggle_corr = 0 if np.isnan(dist_toggle_corr) else dist_toggle_corr

        # Cap × drive product
        ff_drive = drive[ff_idx]
        cap_drive = (ff_cap * ff_drive).mean()

        # FF density at different scales
        p25_dist = np.percentile(ff_dists, 25)
        p75_dist = np.percentile(ff_dists, 75)
        p90_dist = np.percentile(ff_dists, 90)
        max_dist = ff_dists.max()

        # HPWL of all FFs
        ff_x_all = x_pos[ff_idx]
        ff_y_all = y_pos[ff_idx]
        ff_hpwl = (ff_x_all.max() - ff_x_all.min()) + (ff_y_all.max() - ff_y_all.min())
    else:
        toggle_mean = toggle_std = toggle_max = toggle_gini = 0
        sum_toggle_total = cap_mean = cap_std = 0
        dist_toggle_corr = cap_drive = 0
        p25_dist = p75_dist = p90_dist = max_dist = 0
        ff_hpwl = 0
        ff_dists = np.zeros(1)

    n_buf = is_buf.sum()
    buf_frac = n_buf / max(n_nodes, 1)

    # ── Feature vector assembly ──────────────────────────────────────────────
    feats = np.array([
        # Skip graph: 18 features [0-17]
        np.log1p(max(skip_max, 0)),      # 0 max skip degree
        np.log1p(max(skip_p90, 0)),      # 1 p90 skip degree
        skip_mean,                       # 2 mean skip degree
        skip_std,                        # 3 std skip degree
        skip_gini,                       # 4 gini of skip degrees
        top5_hpwl,                       # 5 HPWL of top-5% critical FFs
        top10_hpwl,                      # 6 HPWL of top-10% critical FFs
        top5_mean_dist,                  # 7 mean dist of top-5% from centroid
        top10_mean_dist,                 # 8 mean dist of top-10% from centroid
        crit_coverage,                   # 9 spatial coverage of critical FFs
        np.log1p(max(ts_mean, 0)),       # 10 toggle × skip mean
        np.log1p(max(ts_max, 0)),        # 11 toggle × skip max
        ts_p90,                          # 12 toggle × skip p90
        lc_asym,                         # 13 launch-capture asymmetry proxy
        top5_hpwl / max(ff_hpwl, 1e-6), # 14 critical FF HPWL fraction
        p90_dist,                        # 15 p90 FF distance from centroid
        max_dist,                        # 16 max FF distance from centroid
        (p90_dist - p25_dist) / max(p25_dist, 1e-6),  # 17 tail-to-core ratio

        # Wire graph: 6 features [18-23]
        wire_mean,                       # 18 mean wire degree
        np.log1p(max(wire_max, 0)),      # 19 max wire degree
        wire_gini,                       # 20 wire degree gini
        np.log1p(n_ff),                  # 21 log(n_ff)
        buf_frac,                        # 22 buffer fraction
        np.log1p(n_nodes),               # 23 log(n_nodes)

        # Activity features: 8 features [24-31]
        toggle_mean,                     # 24
        toggle_std,                      # 25
        np.log1p(max(toggle_max, 0)),    # 26
        toggle_gini,                     # 27
        np.log1p(max(sum_toggle_total, 0)),  # 28 total activity
        cap_mean,                        # 29
        cap_std,                         # 30
        dist_toggle_corr,                # 31 activity-distance correlation

        # Spatial aggregates: 8 features [32-39]
        ff_hpwl,                         # 32 FF HPWL
        p25_dist, p75_dist, p90_dist, max_dist,  # 33-36
        cap_drive,                       # 37 cap × drive product
        np.log1p(n_ff) * ff_hpwl,       # 38 n_ff × HPWL (size-aware WL proxy)
        (p90_dist - p25_dist),           # 39 interquartile distance spread

        # Physics interactions: 20 features [40-59]
        skip_max * p90_dist,             # 40 max skip path × worst case distance
        skip_gini * ff_hpwl,             # 41 path imbalance × die size
        toggle_mean * cap_mean,          # 42 activity × cap (power proxy)
        np.log1p(max(sum_toggle_total, 0)) * cap_mean,  # 43 total power proxy
        top5_hpwl * skip_std,            # 44 critical path spread × degree variance
        ff_hpwl * cap_mean,              # 45 WL proxy × cap
        toggle_gini * p90_dist,          # 46 activity inequality × spatial spread
        wire_mean * toggle_mean,         # 47 connectivity × activity
        np.log1p(n_ff) * p90_dist,      # 48 n_ff × spread
        crit_coverage * skip_gini,       # 49 coverage × imbalance
        top10_hpwl * (1 + skip_mean),   # 50 critical spread × activity
        max_dist * skip_max,             # 51 worst case position × critical path
        p90_dist / max(ff_hpwl, 1e-6),  # 52 relative tail distance
        toggle_std / max(toggle_mean, 1e-6),  # 53 activity CV
        buf_frac * cap_mean,             # 54 buffer density × cap
        wire_gini * ff_hpwl,             # 55 wire imbalance × die size
        ts_max / max(np.log1p(max(sum_toggle_total, 0)), 1e-6),  # 56 critical toggle fraction
        np.log1p(max(skip_max, 0)) * toggle_mean,  # 57 critical path × activity
        (top5_hpwl - top10_hpwl) / max(top10_hpwl, 1e-6),  # 58 critical concentration
        p75_dist * skip_mean,            # 59 bulk spread × avg path length
    ], dtype=np.float32)

    # Replace NaN/inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    return feats


if __name__ == "__main__":
    # Test single file
    pt_files = list(GRAPH_DIR.glob("*.pt"))
    print(f"Found {len(pt_files)} PT files")
    feats = extract_graph_features(pt_files[0].stem)
    print(f"Feature shape: {feats.shape}")
    print(f"First 10 feats: {feats[:10]}")
    print(f"NaN: {np.isnan(feats).sum()}, Inf: {np.isinf(feats).sum()}")
