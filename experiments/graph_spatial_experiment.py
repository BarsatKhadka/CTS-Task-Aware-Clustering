"""
Graph/DEF spatial feature extraction experiment.

Extracts design-invariant spatial features from processed_graphs/*.pt files
(z-scored FF positions, skip path distances, capacitance distributions)
and tests if they improve LODO MAE for power, WL, and skew.

Also tests the X38 approach (clock intermediates) for context.

Key insight: XY coords in .pt files are z-scored within each design,
so spatial statistics are scale-invariant across designs.
"""
import pickle, glob, warnings, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract spatial features per placement from processed_graphs/*.pt
# ─────────────────────────────────────────────────────────────────────────────

def extract_graph_spatial_features(pt_path):
    """
    Extract design-invariant spatial features from a single .pt graph file.
    All spatial features use z-scored coordinates → comparable across designs.

    Returns: dict of scalar features for this placement
    """
    d = torch.load(pt_path, weights_only=False)
    X = d['X'].numpy()   # [N, 18]

    # Column mapping:
    # 0:x_norm  1:y_norm  2-5:dist_boundaries  6:log1p(area_z)
    # 7:avg_pin_cap_z  8:total_pin_cap_z  9:log2(drive)_z
    # 10:is_sequential  11:is_buffer  12:toggle_z  13:sum_toggle_z
    # 14:signal_prob_z  15:non_zero_z  16:log1p(fan_in)_z  17:log1p(fan_out)_z

    xy        = X[:, :2]           # z-scored coordinates
    area_z    = X[:, 6]            # log1p(area) z-scored
    cap_z     = X[:, 8]            # total_pin_cap z-scored
    drive_z   = X[:, 9]            # drive strength z-scored
    is_seq    = X[:, 10] > 0.5     # bool mask
    is_buf    = X[:, 11] > 0.5
    toggle_z  = X[:, 12]           # toggle z-scored
    fan_out_z = X[:, 17]           # fan_out z-scored

    ff_xy   = xy[is_seq]
    ff_cap  = cap_z[is_seq]
    ff_tog  = toggle_z[is_seq]
    ff_drv  = drive_z[is_seq]

    n_ff  = is_seq.sum()
    n_buf = is_buf.sum()
    n_all = len(X)

    feats = {}

    # ── FF spatial statistics ──────────────────────────────────────────────
    if n_ff < 4:
        return None   # degenerate

    feats['ff_hpwl_z']    = float((ff_xy[:,0].max() - ff_xy[:,0].min()) +
                                   (ff_xy[:,1].max() - ff_xy[:,1].min()))
    feats['ff_std_x']     = float(ff_xy[:,0].std())
    feats['ff_std_y']     = float(ff_xy[:,1].std())
    feats['ff_aspect_z']  = float((ff_xy[:,0].max() - ff_xy[:,0].min()) /
                                  max(ff_xy[:,1].max() - ff_xy[:,1].min(), 1e-4))

    # Centroid offset (asymmetry): how far the FF centroid is from 0
    cx, cy = ff_xy[:,0].mean(), ff_xy[:,1].mean()
    feats['ff_centroid_r'] = float(np.sqrt(cx**2 + cy**2))

    # ── kNN distances in normalized space ──────────────────────────────────
    tree = cKDTree(ff_xy)
    dists4, _ = tree.query(ff_xy, k=min(5, n_ff))   # k=1 is self
    if dists4.shape[1] > 1:
        knn = dists4[:, 1:]
        feats['knn4_mean']  = float(knn.mean())
        feats['knn4_std']   = float(knn.std())
        feats['knn4_p90']   = float(np.percentile(knn, 90))
        feats['knn4_max']   = float(knn.max())
    else:
        for k in ['knn4_mean','knn4_std','knn4_p90','knn4_max']: feats[k] = 0.0

    # ── Toggle-weighted centroid (activity asymmetry) ─────────────────────
    tog_pos = ff_tog - ff_tog.min() + 1e-4   # shift to positive
    tog_w   = tog_pos / tog_pos.sum()
    tc_x    = float((tog_w * ff_xy[:,0]).sum())
    tc_y    = float((tog_w * ff_xy[:,1]).sum())
    feats['toggle_centroid_r']  = float(np.sqrt(tc_x**2 + tc_y**2))
    feats['toggle_cx_offset']   = float(abs(tc_x - cx))
    feats['toggle_cy_offset']   = float(abs(tc_y - cy))

    # ── Capacitance-weighted HPWL (routing cap proxy) ────────────────────
    # Weighted centroid for capacitance
    cap_pos = ff_cap - ff_cap.min() + 1e-4
    cap_w   = cap_pos / cap_pos.sum()
    cc_x    = float((cap_w * ff_xy[:,0]).sum())
    cc_y    = float((cap_w * ff_xy[:,1]).sum())
    feats['cap_centroid_r']     = float(np.sqrt(cc_x**2 + cc_y**2))

    # ── Drive strength distribution ──────────────────────────────────────
    feats['drive_std']    = float(ff_drv.std())
    feats['drive_max']    = float(ff_drv.max())
    feats['drive_p90']    = float(np.percentile(ff_drv, 90))

    # ── Node counts (normalized to be design-invariant) ──────────────────
    feats['buf_to_ff_ratio'] = float(n_buf / max(n_ff, 1))
    feats['ff_fraction']     = float(n_ff / max(n_all, 1))

    # ── Grid entropy (FF spatial distribution uniformity) ─────────────────
    nx, ny = 8, 8
    x_bins = np.linspace(ff_xy[:,0].min()-0.01, ff_xy[:,0].max()+0.01, nx+1)
    y_bins = np.linspace(ff_xy[:,1].min()-0.01, ff_xy[:,1].max()+0.01, ny+1)
    H, _, _ = np.histogram2d(ff_xy[:,0], ff_xy[:,1], bins=[x_bins, y_bins])
    p = H.ravel() / H.sum()
    p = p[p > 0]
    feats['grid_entropy'] = float(-np.sum(p * np.log(p + 1e-12)) / np.log(nx*ny))

    # ── Skip-path (timing path) distance statistics ───────────────────────
    A_skip = d['A_skip_csr']
    A_np   = A_skip.to_dense().numpy()
    rows, cols = np.where(A_np > 0)

    if len(rows) > 0:
        skip_dx = xy[rows, 0] - xy[cols, 0]
        skip_dy = xy[rows, 1] - xy[cols, 1]
        skip_d  = np.sqrt(skip_dx**2 + skip_dy**2)
        feats['skip_max']   = float(skip_d.max())
        feats['skip_p90']   = float(np.percentile(skip_d, 90))
        feats['skip_p75']   = float(np.percentile(skip_d, 75))
        feats['skip_mean']  = float(skip_d.mean())
        feats['skip_std']   = float(skip_d.std())
        feats['skip_count'] = float(len(rows)) / max(n_ff, 1)  # normalized

        # Directional asymmetry: |mean_dx| / std_dx (launch-capture imbalance)
        feats['skip_dx_bias'] = float(abs(skip_dx.mean()) / (skip_dx.std() + 1e-4))
        feats['skip_dy_bias'] = float(abs(skip_dy.mean()) / (skip_dy.std() + 1e-4))
    else:
        for k in ['skip_max','skip_p90','skip_p75','skip_mean','skip_std',
                  'skip_count','skip_dx_bias','skip_dy_bias']: feats[k] = 0.0

    # ── 2-hop wire distances (physical proximity graph) ───────────────────
    A_wire = d['A_wire_csr']
    A_wire_np = A_wire.to_dense().numpy()
    wr, wc = np.where(A_wire_np > 0)
    if len(wr) > 0:
        wire_d = np.sqrt((xy[wr,0]-xy[wc,0])**2 + (xy[wr,1]-xy[wc,1])**2)
        feats['wire_mean']  = float(wire_d.mean())
        feats['wire_p90']   = float(np.percentile(wire_d, 90))
        feats['wire_max']   = float(wire_d.max())
    else:
        for k in ['wire_mean','wire_p90','wire_max']: feats[k] = 0.0

    # ── Area features (log-scale, z-scored → design-invariant) ───────────
    feats['area_z_std']   = float(area_z.std())
    feats['area_z_max']   = float(area_z.max())
    feats['area_z_range'] = float(area_z.max() - area_z.min())

    # Placement ID (stem of filename)
    feats['_placement_id'] = Path(pt_path).stem

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build feature cache from all graphs
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_FEAT_CACHE = 'graph_spatial_cache.pkl'

def build_or_load_graph_cache():
    try:
        with open(GRAPH_FEAT_CACHE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    pts = sorted(glob.glob('processed_graphs/*.pt'))
    print(f"Extracting spatial features from {len(pts)} graphs...")
    records = []
    for i, p in enumerate(pts):
        feats = extract_graph_spatial_features(p)
        if feats is not None:
            records.append(feats)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(pts)} done")

    df_graph = pd.DataFrame(records)
    print(f"Graph features: {df_graph.shape}")

    with open(GRAPH_FEAT_CACHE, 'wb') as f:
        pickle.dump(df_graph, f)
    print("Saved graph_spatial_cache.pkl")
    return df_graph


# ─────────────────────────────────────────────────────────────────────────────
# 3. Design-ID leakage test
# ─────────────────────────────────────────────────────────────────────────────

def leakage_test(feat_cols, df_combined):
    """Test what fraction of design ID can be predicted from feature subset."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y  = le.fit_transform(df_combined['design_name'])
    X  = df_combined[feat_cols].fillna(0).values

    sc  = StandardScaler()
    Xsc = sc.fit_transform(X)

    # LOO for design ID
    designs = df_combined['design_name'].unique()
    correct = 0; total = 0
    for des in designs:
        mask = (df_combined['design_name'] == des).values
        clf  = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(Xsc[~mask], y[~mask])
        preds = clf.predict(Xsc[mask])
        correct += (preds == y[mask]).sum()
        total   += mask.sum()
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build combined feature matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_features_combined(df_csv, df_graph):
    """
    Merge graph spatial features with CSV features.

    Returns X (per-run features), Y (z-score targets), df (metadata)
    """
    # Graph features are per-placement; CSV has 10 runs per placement
    # placement_id in CSV matches stem of pt file
    df_graph2 = df_graph.copy()
    df_graph2 = df_graph2.rename(columns={'_placement_id': 'placement_id'})

    # Merge: each run gets the spatial features of its placement
    df_merged = df_csv.merge(df_graph2, on='placement_id', how='left')
    print(f"Merged: {len(df_merged)} rows, {df_merged['design_name'].value_counts().to_dict()}")

    # Graph spatial feature columns
    graph_cols = [c for c in df_graph2.columns if c != 'placement_id']

    # ── CTS knob features (raw knobs per run) ─────────────────────────────
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']

    # Per-placement z-scored knobs
    Xkz = np.zeros((len(df_merged), 4), dtype=np.float32)
    for j, c in enumerate(knob_cols):
        vals = df_merged[c].values.astype(np.float64)
        for pid in df_merged['placement_id'].unique():
            mask = (df_merged['placement_id'] == pid).values
            mu, sig = vals[mask].mean(), vals[mask].std()
            if sig < 1e-9: sig = 1.0
            Xkz[mask, j] = ((vals[mask] - mu) / sig).astype(np.float32)

    # Per-placement rank-normalized knobs [0,1]
    Xkr = np.zeros((len(df_merged), 4), dtype=np.float32)
    for j, c in enumerate(knob_cols):
        vals = df_merged[c].values.astype(np.float64)
        for pid in df_merged['placement_id'].unique():
            mask = (df_merged['placement_id'] == pid).values
            v = vals[mask]
            r = (v - v.min()) / max(v.max() - v.min(), 1e-9)
            Xkr[mask, j] = r.astype(np.float32)

    # Centered knobs (global mean subtracted)
    raw_max = np.array([df_merged[c].max() for c in knob_cols], dtype=np.float32)
    raw_min = np.array([df_merged[c].min() for c in knob_cols], dtype=np.float32)
    Xkc = ((df_merged[knob_cols].values - df_merged[knob_cols].values.mean(0)) /
           np.maximum(df_merged[knob_cols].values.std(0), 1e-9)).astype(np.float32)

    # Per-placement knob range and mean
    Xrange = np.zeros((len(df_merged), 4), dtype=np.float32)
    Xmean  = np.zeros((len(df_merged), 4), dtype=np.float32)
    for j, c in enumerate(knob_cols):
        vals = df_merged[c].values.astype(np.float64)
        for pid in df_merged['placement_id'].unique():
            mask = (df_merged['placement_id'] == pid).values
            v = vals[mask]
            Xrange[mask, j] = v.max() - v.min()
            Xmean[mask, j]  = v.mean()

    # ── Placement metadata ──────────────────────────────────────────────
    place_cols = ['core_util', 'density', 'aspect_ratio']
    Xplace = df_merged[place_cols].values.astype(np.float32)

    # z-score placement features
    Xplace_norm = (Xplace - Xplace.mean(0)) / np.maximum(Xplace.std(0), 1e-9)

    # ── Knob × geometry interactions ────────────────────────────────────
    cd = df_merged['cts_cluster_dia'].values.astype(np.float32)
    mw = df_merged['cts_max_wire'].values.astype(np.float32)
    cs = df_merged['cts_cluster_size'].values.astype(np.float32)
    bd = df_merged['cts_buf_dist'].values.astype(np.float32)
    util   = Xplace[:, 0]
    dens   = Xplace[:, 1]
    aspect = Xplace[:, 2]

    Xkp = np.column_stack([
        cd * util / 100,
        mw * dens,
        cd / np.maximum(dens, 0.01),
        cd * aspect,
        Xkr[:, 3] * (util / 100),   # rank(cd) × util
        Xkr[:, 2] * (util / 100),   # rank(cs) × util
    ]).astype(np.float32)

    # ── Graph spatial features ─────────────────────────────────────────
    Xgraph = df_merged[graph_cols].fillna(0).values.astype(np.float32)
    # z-score graph features
    Xgraph_z = (Xgraph - Xgraph.mean(0)) / np.maximum(Xgraph.std(0), 1e-9)

    # ── Knob × graph spatial interactions ──────────────────────────────
    hpwl    = Xgraph_z[:, graph_cols.index('ff_hpwl_z')]
    skip_mx = Xgraph_z[:, graph_cols.index('skip_max')]
    knn4    = Xgraph_z[:, graph_cols.index('knn4_mean')]

    Xkg = np.column_stack([
        cd * hpwl,           # cluster_dia × HPWL
        Xkr[:, 3] * hpwl,   # rank(cd) × HPWL
        mw / np.maximum(cd + 1e-4, 1e-4) * hpwl,  # max_wire/cd × HPWL
        cd * knn4,           # cluster_dia × local_density
        Xkr[:, 3] * skip_mx, # rank(cd) × skip_max
        mw * skip_mx,        # max_wire × skip_max
    ]).astype(np.float32)

    # ── Assemble X variants ─────────────────────────────────────────────
    X29  = np.hstack([Xkz, Xkr, Xkc, Xplace_norm,
                      Xrange/np.maximum(raw_max, 1e-9),
                      Xmean/np.maximum(raw_max, 1e-9), Xkp])

    Xall = np.hstack([X29, Xgraph_z, Xkg])

    # ── Per-placement z-score targets ──────────────────────────────────
    target_cols = ['skew_setup', 'power_total', 'wirelength']
    Y = np.zeros((len(df_merged), 3), dtype=np.float32)
    for j, c in enumerate(target_cols):
        vals = df_merged[c].values.astype(np.float64)
        for pid in df_merged['placement_id'].unique():
            mask = (df_merged['placement_id'] == pid).values
            mu, sig = vals[mask].mean(), vals[mask].std()
            if sig < 1e-9: sig = 1.0
            Y[mask, j] = ((vals[mask] - mu) / sig).astype(np.float32)

    return X29, Xall, Y, df_merged, graph_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. LODO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def lodo_eval(X, Y, df, model_fn, label=""):
    designs = df['design_name'].unique()
    results = {'sk': [], 'pw': [], 'wl': []}

    for held in sorted(designs):
        tr = (df['design_name'] != held).values
        te = ~tr

        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])

        maes = []
        for j, name in enumerate(['sk', 'pw', 'wl']):
            m = model_fn(name)
            m.fit(Xtr, Y[tr, j])
            yp  = m.predict(Xte)
            mae = mean_absolute_error(Y[te, j], yp)
            results[name].append(mae)
            maes.append(mae)

        status = ' '.join([f"sk={maes[0]:.4f} pw={maes[1]:.4f} wl={maes[2]:.4f}"])
        print(f"  held={held}: {status}")

    print(f"\n  [{label}] LODO means:")
    for t in ['sk', 'pw', 'wl']:
        m = np.mean(results[t])
        ok = 'PASS✓' if m < 0.10 else 'FAIL '
        print(f"    {ok} {t}: {m:.4f}  per-fold={[f'{v:.4f}' for v in results[t]]}")

    return {t: np.mean(v) for t, v in results.items()}


def make_lgb(name):
    params = {
        'sk': dict(n_estimators=500, learning_rate=0.03, num_leaves=63,
                   min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=0.5, n_jobs=4, verbose=-1),
        'pw': dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                   min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.5, reg_lambda=1.0, n_jobs=4, verbose=-1),
        'wl': dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                   min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=0.5, n_jobs=4, verbose=-1),
    }
    return lgb.LGBMRegressor(**params[name])

def make_xgb(name):
    params = {
        'sk': dict(n_estimators=1000, learning_rate=0.03, max_depth=5,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=0),
        'pw': dict(n_estimators=500,  learning_rate=0.03, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.5, reg_lambda=2.0, n_jobs=4, verbosity=0),
        'wl': dict(n_estimators=1000, learning_rate=0.03, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=0),
    }
    return xgb.XGBRegressor(**params[name])


# ─────────────────────────────────────────────────────────────────────────────
# 6. Skew-specific features (targeting the 0.10 goal)
# ─────────────────────────────────────────────────────────────────────────────

def build_skew_features(df_csv, df_graph):
    """
    Focused feature set for skew prediction.
    Physics: skew = max(clock_delay) - min(clock_delay)
    Key: skip_max (longest timing path in normalized space) × buf_dist interaction
    """
    df_graph2 = df_graph.rename(columns={'_placement_id': 'placement_id'})
    df_merged = df_csv.merge(df_graph2, on='placement_id', how='left')

    graph_cols = [c for c in df_graph2.columns if c != 'placement_id']
    knob_cols  = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']

    cd = df_merged['cts_cluster_dia'].values.astype(np.float32)
    mw = df_merged['cts_max_wire'].values.astype(np.float32)
    cs = df_merged['cts_cluster_size'].values.astype(np.float32)
    bd = df_merged['cts_buf_dist'].values.astype(np.float32)

    # Per-placement z-scored knobs
    Xkz = np.zeros((len(df_merged), 4), dtype=np.float32)
    Xkr = np.zeros((len(df_merged), 4), dtype=np.float32)
    for j, c in enumerate(knob_cols):
        vals = df_merged[c].values.astype(np.float64)
        for pid in df_merged['placement_id'].unique():
            mask = (df_merged['placement_id'] == pid).values
            mu, sig = vals[mask].mean(), vals[mask].std()
            if sig < 1e-9: sig = 1.0
            Xkz[mask, j] = ((vals[mask] - mu) / sig).astype(np.float32)
            v = vals[mask]
            r = (v - v.min()) / max(v.max()-v.min(), 1e-9)
            Xkr[mask, j] = r.astype(np.float32)

    # Graph spatial features (z-scored)
    Xgraph = df_merged[graph_cols].fillna(0).values.astype(np.float32)
    Xgraph_z = (Xgraph - Xgraph.mean(0)) / np.maximum(Xgraph.std(0), 1e-9)

    skip_max = Xgraph_z[:, graph_cols.index('skip_max')]
    skip_p90 = Xgraph_z[:, graph_cols.index('skip_p90')]
    skip_std = Xgraph_z[:, graph_cols.index('skip_std')]
    hpwl     = Xgraph_z[:, graph_cols.index('ff_hpwl_z')]
    knn4     = Xgraph_z[:, graph_cols.index('knn4_mean')]
    entropy  = Xgraph_z[:, graph_cols.index('grid_entropy')]
    skip_cnt = Xgraph_z[:, graph_cols.index('skip_count')]
    dx_bias  = Xgraph_z[:, graph_cols.index('skip_dx_bias')]
    dy_bias  = Xgraph_z[:, graph_cols.index('skip_dy_bias')]

    # Physics-informed skew interaction features:
    # skew ≈ f(max_wire_path_imbalance × skip_max / buf_dist)
    # cluster_dia × spatial_imbalance → harder to balance
    Xskew_interactions = np.column_stack([
        # Elmore delay model: skew ∝ (max_path - min_path) / buf_stages
        Xkr[:, 1] * skip_max,       # rank(buf_dist) × skip_max
        Xkz[:, 0] * skip_max,       # z(max_wire) × skip_max
        Xkz[:, 3] * skip_max,       # z(cluster_dia) × skip_max
        Xkz[:, 3] * skip_std,       # z(cluster_dia) × skip_std

        # Spatial imbalance
        Xkz[:, 3] * dx_bias,        # cluster_dia × x-directional bias
        Xkz[:, 3] * dy_bias,        # cluster_dia × y-directional bias
        Xkr[:, 1] * skip_p90,       # rank(buf_dist) × p90 skip
        Xkr[:, 1] * hpwl,           # rank(buf_dist) × HPWL

        # Clustering quality
        Xkz[:, 3] * knn4,           # cluster_dia × local density
        Xkr[:, 2] * skip_cnt,       # rank(cluster_size) × skip density

        # Second-order
        skip_max * skip_std,         # max × variance (tail heaviness)
        hpwl * entropy,              # layout regularity × extent
        dx_bias * dy_bias,           # 2D directional imbalance
    ]).astype(np.float32)

    Xsk = np.hstack([Xkz, Xkr, Xgraph_z, Xskew_interactions])

    # Targets
    Y_sk = np.zeros(len(df_merged), dtype=np.float32)
    vals = df_merged['skew_setup'].values.astype(np.float64)
    for pid in df_merged['placement_id'].unique():
        mask = (df_merged['placement_id'] == pid).values
        mu, sig = vals[mask].mean(), vals[mask].std()
        if sig < 1e-9: sig = 1.0
        Y_sk[mask] = ((vals[mask] - mu) / sig).astype(np.float32)

    return Xsk, Y_sk, df_merged


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("GRAPH SPATIAL FEATURES EXPERIMENT")
    print("="*70)

    # Load data
    df_graph = build_or_load_graph_cache()
    print(f"\nGraph features shape: {df_graph.shape}")
    print(f"Graph feature columns: {[c for c in df_graph.columns if not c.startswith('_')]}")

    df_csv = pd.read_csv('dataset_with_def/unified_manifest_normalized.csv')
    print(f"\nCSV shape: {df_csv.shape}")

    # Match placement IDs
    graph_pids = set(df_graph['_placement_id'].values)
    csv_pids   = set(df_csv['placement_id'].values)
    print(f"Graph placements: {len(graph_pids)}, CSV placements: {len(csv_pids)}")
    print(f"Overlap: {len(graph_pids & csv_pids)}")

    # Filter CSV to matched placements
    df_csv_m = df_csv[df_csv['placement_id'].isin(graph_pids)].copy().reset_index(drop=True)
    print(f"CSV after filter: {len(df_csv_m)}, designs: {df_csv_m['design_name'].value_counts().to_dict()}")

    # ── Build features ─────────────────────────────────────────────────────
    print("\nBuilding combined features...")
    X29, Xall, Y, df_merged, graph_cols = build_features_combined(df_csv_m, df_graph)

    print(f"\nX29 shape: {X29.shape}")
    print(f"Xall shape: {Xall.shape}")
    print(f"Y shape: {Y.shape}")

    # ── Design ID leakage test ─────────────────────────────────────────────
    graph_feat_names = [c for c in df_graph.columns if not c.startswith('_')]
    print(f"\nDesign ID leakage test for graph spatial features...")
    df_for_leak = df_merged.copy()
    for c in graph_feat_names:
        if c not in df_for_leak.columns:
            df_for_leak[c] = 0.0

    leakage = leakage_test(graph_feat_names, df_for_leak)
    print(f"  Graph spatial features: {leakage:.1%} design ID accuracy")
    print(f"  (Benchmark: CTS knobs=35%, DEF-raw-HPWL=100%)")
    print(f"  {'SAFE ✓' if leakage < 0.5 else 'WARNING: leakage!'}")

    # ── Experiment 1: X29 baseline ────────────────────────────────────────
    print("\n" + "─"*70)
    print("EXP 1: X29 baseline (no graph features)")
    print("─"*70)
    r1 = lodo_eval(X29, Y, df_merged, make_lgb, "X29+LGB")

    # ── Experiment 2: X29 + graph spatial (LGB) ───────────────────────────
    print("\n" + "─"*70)
    print("EXP 2: X29 + graph spatial features (LGB)")
    print("─"*70)
    r2 = lodo_eval(Xall, Y, df_merged, make_lgb, "Xall+LGB")

    # ── Experiment 3: X29 + graph spatial (XGB) ───────────────────────────
    print("\n" + "─"*70)
    print("EXP 3: X29 + graph spatial features (XGB)")
    print("─"*70)
    r3 = lodo_eval(Xall, Y, df_merged, make_xgb, "Xall+XGB")

    # ── Experiment 4: Mixed best models ───────────────────────────────────
    print("\n" + "─"*70)
    print("EXP 4: Mixed (XGB for sk/wl, LGB for pw)")
    print("─"*70)
    def make_mixed(name):
        return make_xgb(name) if name in ['sk', 'wl'] else make_lgb(name)
    r4 = lodo_eval(Xall, Y, df_merged, make_mixed, "Xall+Mixed")

    # ── Experiment 5: Skew-focused features ───────────────────────────────
    print("\n" + "─"*70)
    print("EXP 5: Skew-focused physics features")
    print("─"*70)
    Xsk, Y_sk, df_sk = build_skew_features(df_csv_m, df_graph)

    designs_sk = df_sk['design_name'].unique()
    sk_results = []
    for held in sorted(designs_sk):
        tr = (df_sk['design_name'] != held).values
        te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xsk[tr])
        Xte = sc.transform(Xsk[te])
        m = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=5,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=0)
        m.fit(Xtr, Y_sk[tr])
        yp  = m.predict(Xte)
        mae = mean_absolute_error(Y_sk[te], yp)
        sk_results.append(mae)
        print(f"  held={held}: sk MAE={mae:.4f}")

    sk_mean = np.mean(sk_results)
    print(f"\n  Skew LODO mean: {sk_mean:.4f}  {'PASS✓' if sk_mean < 0.10 else 'FAIL'}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Skew':>8} {'Power':>8} {'WL':>8}")
    print(f"{'─'*25} {'─'*8} {'─'*8} {'─'*8}")
    for label, r in [("X29+LGB (baseline)", r1), ("Xall+LGB", r2),
                     ("Xall+XGB", r3), ("Xall+Mixed", r4)]:
        sk_s = f"{'✓' if r['sk']<0.10 else '✗'}{r['sk']:.4f}"
        pw_s = f"{'✓' if r['pw']<0.10 else '✗'}{r['pw']:.4f}"
        wl_s = f"{'✓' if r['wl']<0.10 else '✗'}{r['wl']:.4f}"
        print(f"{label:<25} {sk_s:>8} {pw_s:>8} {wl_s:>8}")
    print(f"{'Skew-focus XGB':<25} {f'{'✓' if sk_mean<0.10 else '✗'}{sk_mean:.4f}':>8} {'—':>8} {'—':>8}")

    print(f"\nPrevious best: sk=0.2369 pw=0.0656 wl=0.0858")

    # ── Feature importance for best model ─────────────────────────────────
    print("\n" + "─"*70)
    print("TOP GRAPH FEATURES for Power and WL (XGB, full fold):")
    print("─"*70)
    sc = StandardScaler()
    Xsc = sc.fit_transform(Xall)
    n29 = X29.shape[1]
    feat_names = ([f"x29_{i}" for i in range(n29)] +
                  graph_cols +
                  ["cd×hpwl","rank_cd×hpwl","mw/cd×hpwl","cd×knn4","rank_cd×skip","mw×skip"])

    for task_name, j in [('Power', 1), ('WL', 2)]:
        m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=0)
        m.fit(Xsc, Y[:, j])
        imp = m.feature_importances_
        top = np.argsort(imp)[::-1][:10]
        print(f"\n  {task_name} top-10:")
        for i in top:
            nm = feat_names[i] if i < len(feat_names) else f"feat_{i}"
            print(f"    [{i:3d}] {nm}: {imp[i]:.4f}")


if __name__ == '__main__':
    main()
