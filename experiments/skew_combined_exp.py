"""
skew_combined_exp.py — Combine X29T (tight_path) + kNN density + interactions

Prior best: X29T+skew_inter XGB_300 = 0.2369
Hypothesis: Adding kNN density features should improve further.
X29T = X29 + tight_path (20-dim) → 49-dim (the existing X49)
X49 + kNN_density (20-dim) + interactions (36-dim) → test

Also tests: X49 + tight_path_×_cd interactions (better physics than what I tried)
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
print("Skew Combined: X29T + kNN Density")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df_cache = cache['X'], cache['Y'], cache['df']

with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp_cache = pickle.load(f)

with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_cache = pickle.load(f)

pids    = df_cache['placement_id'].values
designs = df_cache['design_name'].values
y_sk_z  = Y_cache[:, 0]
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
    s = '✓' if mean_mae < 0.10 else ('NEW BEST' if mean_mae < 0.237 else '')
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f} {s}")
    return mean_mae, maes

# ── X29 base ─────────────────────────────────────────────────────────
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

# ── Tight path features (X29T = X49 in this script) ───────────────────
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp_cache.get(pid)
    if v is not None:
        arr = np.array(v, dtype=np.float32)
        X_tight[i, :min(20, len(arr))] = arr[:20]
tp_std = X_tight.std(axis=0); tp_std[tp_std < 1e-9] = 1.0
X_tight_n = X_tight / tp_std
X29T = np.hstack([X29, X_tight_n])  # 49-dim (prior best)

# ── kNN density features (from ff_positions, per-placement) ───────────
print(f"{T()} Computing kNN density features...")
knn_feats = {}
for pid in np.unique(pids):
    pos_info = ff_cache.get(pid)
    if pos_info is None or pos_info.get('ff_norm') is None:
        knn_feats[pid] = np.zeros(20, np.float32)
        continue
    ff_xy = pos_info['ff_norm']
    if len(ff_xy) < 4:
        knn_feats[pid] = np.zeros(20, np.float32)
        continue
    tree = cKDTree(ff_xy)
    dists, _ = tree.query(ff_xy, k=min(9, len(ff_xy)))
    knn1 = dists[:, 1] if dists.shape[1] > 1 else np.zeros(len(ff_xy))
    knn4 = dists[:, min(4, dists.shape[1]-1)]
    knn8 = dists[:, min(8, dists.shape[1]-1)]
    centroid = ff_xy.mean(axis=0)
    cdist = np.linalg.norm(ff_xy - centroid, axis=1)
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
        cdist.std() / (cdist.mean() + 1e-8),
    ], dtype=np.float32)
    knn_feats[pid] = feats

Xknn_raw = np.array([knn_feats.get(pid, np.zeros(20)) for pid in pids], np.float32)
knn_std = Xknn_raw.std(axis=0); knn_std[knn_std < 1e-9] = 1.0
Xknn_n = Xknn_raw / knn_std

print(f"{T()} kNN feats built. cd/knn1_mean range: "
      f"[{(cd / (Xknn_raw[:,0]*1000+1e-4)).min():.2f}, {(cd / (Xknn_raw[:,0]*1000+1e-4)).max():.2f}]")

# ── Physics: cluster_dia × kNN interactions (key for skew) ────────────
knn1_mean = Xknn_raw[:, 0]
knn1_p90  = Xknn_raw[:, 2]
knn1_p99  = Xknn_raw[:, 4]
knn4_mean = Xknn_raw[:, 6]
cdist_p90 = Xknn_raw[:, 13]
cdist_p99 = Xknn_raw[:, 15]
outlier_ratio = Xknn_raw[:, 18]
tail_ratio = Xknn_raw[:, 17]

# Physical interactions: how well does cluster_dia cover the FF density?
raw_phys = np.column_stack([
    cd / (knn1_mean * 1000 + 1e-4),  # cluster_dia / mean nearest-neighbor
    cd / (knn1_p90 * 1000 + 1e-4),   # cluster_dia / p90 nearest-neighbor
    cd / (knn1_p99 * 1000 + 1e-4),   # cluster_dia / p99 nearest-neighbor (outlier FFs)
    cd / (knn4_mean * 1000 + 1e-4),  # cluster_dia / 4-NN mean
    bd / (knn1_mean * 1000 + 1e-4),  # buf_dist / mean knn
    cd / (cdist_p99 * 1000 + 1e-4),  # cluster / outlier centroid dist
    Xrank[:, 3] * outlier_ratio,     # rank(cd) × outlier_ratio
    Xrank[:, 3] * tail_ratio,        # rank(cd) × tail_ratio
    outlier_ratio,
    tail_ratio,
    Xknn_raw[:, 19],                 # knn1 CV (spatial uniformity)
    cd / (cdist_p90 * 1000 + 1e-4),
])

# Fix inf/nan
for c in range(raw_phys.shape[1]):
    bad = ~np.isfinite(raw_phys[:, c])
    if bad.any():
        raw_phys[bad, c] = np.nanmedian(raw_phys[~bad, c]) if (~bad).any() else 0.0
    raw_phys[:, c] = np.clip(raw_phys[:, c], -1e6, 1e6)

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

# ── Novel: tight_path × cluster_dia interactions ─────────────────────
# tp[5] = td_max (max tight path distance, in normalized die units)
# tp[7] = td_mean (mean tight path distance)
# tp[9] = td_cv (tight path CV)
# tp[4] = ad_cv (all path CV)
tp_ad_cv  = X_tight[:, 4]   # all-path CV (high = more spread)
tp_td_cv  = X_tight[:, 9]   # tight-path CV
tp_td_max = X_tight[:, 5].clip(1e-4)
tp_frac_td = X_tight[:, 15]  # fraction tight paths
tp_frac_nd = X_tight[:, 16]  # fraction negative-slack

# rank(cd) × tp features: within-placement knob variation × fixed placement geom
raw_tp_inter = np.column_stack([
    Xrank[:, 3] * tp_ad_cv,    # rank(cd) × all-path CV
    Xrank[:, 3] * tp_td_cv,    # rank(cd) × tight-path CV
    Xrank[:, 3] * tp_frac_nd,  # rank(cd) × frac negative paths
    Xrank[:, 3] * tp_td_max,   # rank(cd) × max tight path
    Xrank[:, 1] * tp_td_cv,    # rank(buf_dist) × tight-path CV
    tp_ad_cv * tp_frac_nd,     # combined spread×criticality
    tp_td_cv / (tp_frac_td + 0.01),  # variance per tight path density
])

for c in range(raw_tp_inter.shape[1]):
    bad = ~np.isfinite(raw_tp_inter[:, c])
    if bad.any():
        raw_tp_inter[bad, c] = np.nanmedian(raw_tp_inter[~bad, c]) if (~bad).any() else 0.0

tp_inter_rank = np.zeros_like(raw_tp_inter)
tp_inter_cent = np.zeros_like(raw_tp_inter)
g_ti = np.abs(raw_tp_inter).max(axis=0) + 1e-9
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(raw_tp_inter.shape[1]):
        v = raw_tp_inter[idxs, j]
        tp_inter_rank[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
        tp_inter_cent[idxs, j] = (v - v.mean()) / g_ti[j]

Xtp_inter = np.hstack([raw_tp_inter / g_ti, tp_inter_rank, tp_inter_cent])  # [n, 21]

# NaN fix all
for arr in [X29, X29T, Xknn_n, Xphys, Xtp_inter]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:, c])
        if bad.any():
            arr[bad, c] = np.nanmedian(arr[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} All features built:")
print(f"  X29={X29.shape[1]}, X29T={X29T.shape[1]}, kNN={Xknn_n.shape[1]}")
print(f"  Xphys={Xphys.shape[1]}, Xtp_inter={Xtp_inter.shape[1]}")

XGB_300 = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)
XGB_500 = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)
LGB_K = dict(n_estimators=300, num_leaves=15, learning_rate=0.05,
             min_child_samples=15, verbose=-1)

print(f"\n{T()} === SKEW LODO (prior best: X29T+inter XGB_300 = 0.2369) ===")
# Replicate prior best
lodo_single(X29,  y_sk_z, "X29 LGB (control)")
lodo_single(X29T, y_sk_z, "X29T LGB (prior baseline)")
lodo_single(X29T, y_sk_z, "X29T XGB_300 (prior best ~0.237)", XGBRegressor, XGB_300)

# Add kNN
lodo_single(np.hstack([X29T, Xknn_n]),         y_sk_z, "X29T+kNN LGB")
lodo_single(np.hstack([X29T, Xknn_n]),         y_sk_z, "X29T+kNN XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xphys]),          y_sk_z, "X29T+phys XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xknn_n, Xphys]), y_sk_z, "X29T+kNN+phys XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xknn_n, Xphys]), y_sk_z, "X29T+kNN+phys XGB_500", XGBRegressor, XGB_500)

# Add tight_path × cd interactions
lodo_single(np.hstack([X29T, Xtp_inter]),      y_sk_z, "X29T+tpinter XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xtp_inter, Xknn_n]), y_sk_z,
            "X29T+tpinter+kNN XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xtp_inter, Xknn_n, Xphys]), y_sk_z,
            "ALL XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X29T, Xtp_inter, Xknn_n, Xphys]), y_sk_z,
            "ALL XGB_500", XGBRegressor, XGB_500)

print(f"\n{T()} === DONE ===")
