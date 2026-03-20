"""
skew_knn_exp.py — FF kNN density × CTS knob interactions for skew

Key physics: skew depends on whether cluster_dia covers the typical FF-to-FF
spacing. If cluster_dia >> knn_mean_dist → clusters are dense → balanced tree → low skew.
If cluster_dia << knn_mean_dist → FFs isolated → hard to balance → high skew.

Novel: kNN distances (per placement, from ff_positions) × CTS knobs
"""

import pickle, time, warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Skew kNN Density × Knob Interaction Experiment")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df_cache = cache['X'], cache['Y'], cache['df']

with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_cache = pickle.load(f)

pids    = df_cache['placement_id'].values
designs = df_cache['design_name'].values
y_sk_z  = Y_cache[:, 0]
y_pw_z  = Y_cache[:, 1]
y_wl_z  = Y_cache[:, 2]
n = len(pids)

def rank_within(vals):
    return np.argsort(np.argsort(vals)).astype(float) / max(len(vals)-1, 1)

def rank_mae(pred, y_true, pids_):
    pr = np.zeros(len(pids_)); tr = np.zeros(len(pids_))
    for pid in np.unique(pids_):
        m = pids_ == pid; idxs = np.where(m)[0]
        if len(idxs) > 1:
            pr[idxs] = rank_within(pred[idxs])
            tr[idxs] = rank_within(y_true[idxs])
        else:
            pr[idxs] = tr[idxs] = 0.5
    return mean_absolute_error(tr, pr)

def lodo_single(X, y, label, cls=LGBMRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs))
    maes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        maes.append(rank_mae(pred, y[te], pids[te]))
    mean_mae = np.mean(maes)
    s = 'PASS' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f} {s}")
    return mean_mae, maes

# ── Build kNN density features per placement ─────────────────────────
print(f"{T()} Computing kNN density features per placement...")
knn_feats = {}  # pid → 20-dim array
for pid in np.unique(pids):
    pos_info = ff_cache.get(pid)
    if pos_info is None or pos_info.get('ff_norm') is None:
        knn_feats[pid] = np.zeros(20, np.float32)
        continue
    ff_xy = pos_info['ff_norm']  # [N, 2], normalized [0,1]
    if len(ff_xy) < 4:
        knn_feats[pid] = np.zeros(20, np.float32)
        continue
    tree = cKDTree(ff_xy)
    # kNN distances (k=1 is self, so k=2..5 for k-NN 1..4)
    dists, _ = tree.query(ff_xy, k=min(9, len(ff_xy)))  # k=2..5: 1-NN, 2-NN, 3-NN, 4-NN
    knn1 = dists[:, 1] if dists.shape[1] > 1 else np.zeros(len(ff_xy))
    knn4 = dists[:, min(4, dists.shape[1]-1)]
    knn8 = dists[:, min(8, dists.shape[1]-1)]
    # Distance from centroid
    centroid = ff_xy.mean(axis=0)
    cdist = np.linalg.norm(ff_xy - centroid, axis=1)
    # Features capturing tail statistics (outlier FFs)
    feats = np.array([
        knn1.mean(), knn1.std(), np.percentile(knn1, 90), np.percentile(knn1, 95),
        np.percentile(knn1, 99), knn1.max(),
        knn4.mean(), knn4.std(), np.percentile(knn4, 90),
        knn8.mean(),
        cdist.mean(), cdist.std(), np.percentile(cdist, 50),
        np.percentile(cdist, 90), np.percentile(cdist, 95), np.percentile(cdist, 99),
        cdist.max(),
        np.percentile(cdist, 90) / (cdist.mean() + 1e-8),  # tail ratio
        np.percentile(cdist, 99) / (cdist.mean() + 1e-8),  # outlier ratio
        cdist.std() / (cdist.mean() + 1e-8),  # Gini-like CV
    ], dtype=np.float32)
    knn_feats[pid] = feats

# Map per-placement features to per-run array
Xknn_raw = np.array([knn_feats.get(pid, np.zeros(20)) for pid in pids], np.float32)

# Normalize globally
knn_std = Xknn_raw.std(axis=0); knn_std[knn_std < 1e-9] = 1.0
Xknn_n = Xknn_raw / knn_std

print(f"{T()} kNN feats: {Xknn_n.shape}")
print(f"  knn1_mean range: [{Xknn_raw[:,0].min():.4f}, {Xknn_raw[:,0].max():.4f}]")
print(f"  centroid_p99 range: [{Xknn_raw[:,15].min():.4f}, {Xknn_raw[:,15].max():.4f}]")

# ── X29 base features ────────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
place_cols = ['core_util', 'density', 'aspect_ratio']
Xraw = df_cache[knob_cols].values.astype(np.float32)
Xkz  = X_cache[:, 72:76]
raw_max = Xraw.max(axis=0) + 1e-6
Xrank = np.zeros((n, 4), np.float32)
Xcent = np.zeros((n, 4), np.float32)
Xrange = np.zeros((n, 4), np.float32)
Xmean_ = np.zeros((n, 4), np.float32)
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(4):
        v = Xraw[idxs, j]
        Xrank[idxs, j] = rank_within(v)
        Xcent[idxs, j] = (v - v.mean()) / raw_max[j]
        Xrange[idxs, j] = v.std() / raw_max[j]
        Xmean_[idxs, j] = v.mean() / raw_max[j]
Xplc = df_cache[place_cols].values.astype(np.float32)
Xplc_n = Xplc / (Xplc.max(axis=0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
util = Xplc[:, 0]/100.0; density = Xplc[:, 1]; aspect = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*density, cd/(density+0.01),
                           cd*aspect, Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrange, Xmean_])

# ── Novel: cluster_dia / kNN_distance interaction ─────────────────────
# This directly answers: "can cluster_dia group nearby FFs efficiently?"
# All distances are in [0,1] die-fraction units
# cluster_dia is in µm, die ~400-1500µm → cd/1000 ≈ die-fraction

knn1_mean = Xknn_raw[:, 0]  # mean nearest-neighbor distance [0,1 die units]
knn1_p90  = Xknn_raw[:, 2]  # p90 kNN-1
knn1_p99  = Xknn_raw[:, 4]  # p99 kNN-1 (outlier FF spacing)
knn4_mean = Xknn_raw[:, 6]  # mean kNN-4 distance
cdist_p90 = Xknn_raw[:, 13]  # p90 distance from centroid
cdist_p99 = Xknn_raw[:, 15]  # p99 distance from centroid
tail_ratio = Xknn_raw[:, 17]  # p90/mean centroid distance
outlier_ratio = Xknn_raw[:, 18]  # p99/mean centroid distance

# cd in die-fraction units (approximate, cd in µm, typical die ~800µm)
# but die varies, so we use the RATIO directly in raw units
# cd / (knn1_mean * die_scale) — since die_scale unknown, use cd_rank as proxy
# Key insight: rank(cd) vs knn tells relative coverage

# Direct physics interactions (all using raw kNN distances)
raw_phys = np.column_stack([
    # cluster_dia / knn1_mean: directly captures cluster coverage efficiency
    cd / (knn1_mean * 1000 + 1e-4),  # if die≈1000µm
    cd / (knn1_p90 * 1000 + 1e-4),   # covers 90th pctile FF neighbors?
    cd / (knn1_p99 * 1000 + 1e-4),   # covers outlier FFs?
    cd / (knn4_mean * 1000 + 1e-4),  # cluster vs 4-NN spread
    # buf_dist / knn1_mean: buffer frequency vs FF density
    bd / (knn1_mean * 1000 + 1e-4),
    # cluster_dia / centroid_p99: can cluster group outlier FFs?
    cd / (cdist_p99 * 1000 + 1e-4),
    cd / (cdist_p90 * 1000 + 1e-4),
    # outlier_ratio: inherent tail risk of placement (placement-level)
    outlier_ratio,
    tail_ratio,
    # rank(cd) × outlier_ratio: interaction
    Xrank[:, 3] * outlier_ratio,
    Xrank[:, 3] * tail_ratio,
    # knn_cv (placement-level density uniformity)
    Xknn_raw[:, 19],  # knn1 CV
])

# Fix inf/nan
for c in range(raw_phys.shape[1]):
    bad = ~np.isfinite(raw_phys[:, c])
    if bad.any():
        raw_phys[bad, c] = np.nanmedian(raw_phys[~bad, c]) if (~bad).any() else 0.0
    raw_phys[:, c] = np.clip(raw_phys[:, c], -1e6, 1e6)

# Per-placement rank and centered
phys_rank = np.zeros_like(raw_phys)
phys_cent = np.zeros_like(raw_phys)
g_rp = np.abs(raw_phys).max(axis=0) + 1e-9
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(raw_phys.shape[1]):
        v = raw_phys[idxs, j]
        phys_rank[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
        phys_cent[idxs, j] = (v - v.mean()) / g_rp[j]

Xphys = np.hstack([raw_phys / g_rp, phys_rank, phys_cent])  # [n, 36]

# Final feature sets
X49 = np.hstack([X29, Xknn_n])   # X29 + kNN features (49-dim)
X_sk_full = np.hstack([X49, Xphys])  # X49 + physics interactions

# NaN fix
for arr in [X29, Xknn_n, X49, Xphys, X_sk_full]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:, c])
        if bad.any():
            arr[bad, c] = np.nanmedian(arr[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} Feature matrix sizes: X29={X29.shape}, X49={X49.shape}, Xphys={Xphys.shape}")
print(f"  cd/knn1_mean range: [{raw_phys[:,0].min():.3f}, {raw_phys[:,0].max():.3f}]")
print(f"  outlier_ratio range: [{outlier_ratio.min():.3f}, {outlier_ratio.max():.3f}]")

XGB_300 = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)
XGB_500 = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)

print(f"\n{T()} === SKEW LODO ===")
lodo_single(X29,         y_sk_z, "X29 LGB")
lodo_single(X49,         y_sk_z, "X29+kNN LGB")
lodo_single(X49,         y_sk_z, "X29+kNN XGB_300", XGBRegressor, XGB_300)
lodo_single(X_sk_full,   y_sk_z, "X49+phys LGB")
lodo_single(X_sk_full,   y_sk_z, "X49+phys XGB_300", XGBRegressor, XGB_300)
lodo_single(X_sk_full,   y_sk_z, "X49+phys XGB_500", XGBRegressor, XGB_500)
# kNN features alone (without X29)
X_knn_only = np.hstack([Xknn_n, Xphys])
lodo_single(X_knn_only,  y_sk_z, "kNN+phys only XGB_300", XGBRegressor, XGB_300)
# Just the rank(cd/knn) interaction features
lodo_single(np.hstack([X29, phys_rank]), y_sk_z, "X29+phys_rank XGB_300", XGBRegressor, XGB_300)

print(f"\n{T()} === POWER LODO (X29+kNN — check no regression) ===")
lodo_single(X29, y_pw_z, "X29 LGB (baseline)")
lodo_single(X49, y_pw_z, "X29+kNN LGB")

print(f"\n{T()} === WL LODO (X29+kNN) ===")
lodo_single(X29, y_wl_z, "X29 LGB (baseline)")
lodo_single(X49, y_wl_z, "X29+kNN LGB")

print(f"\n{T()} === DONE ===")
