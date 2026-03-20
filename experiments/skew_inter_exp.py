"""
skew_inter_exp.py — Novel skew-specific interaction features

Key physics: skew = max_path - min_path. CTS algorithm groups FFs within cluster_dia.
If cluster_dia < tight_path endpoint distance → can't equalize → high skew.

Novel features: cluster_dia / tight_path_dist percentiles
"""

import pickle, time, warnings
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Skew Interaction Feature Experiment")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df_cache = cache['X'], cache['Y'], cache['df']

with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp_cache = pickle.load(f)

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
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}")
    return mean_mae, maes

# ── Base X29 ──────────────────────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
place_cols = ['core_util', 'density', 'aspect_ratio']
Xraw = df_cache[knob_cols].values.astype(np.float32)
Xplc = df_cache[place_cols].values.astype(np.float32)
Xkz  = X_cache[:, 72:76]
raw_max = Xraw.max(axis=0) + 1e-6
Xrank = np.zeros((n, 4), np.float32)
Xcent = np.zeros((n, 4), np.float32)
Xrange = np.zeros((n, 4), np.float32)
Xmean = np.zeros((n, 4), np.float32)
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(4):
        v = Xraw[idxs, j]
        Xrank[idxs, j] = rank_within(v)
        Xcent[idxs, j] = (v - v.mean()) / raw_max[j]
        Xrange[idxs, j] = v.std() / raw_max[j]
        Xmean[idxs, j] = v.mean() / raw_max[j]
Xplc_n = Xplc / (Xplc.max(axis=0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; bd = Xraw[:, 1]; mw = Xraw[:, 0]
util = Xplc[:, 0]/100.0; density = Xplc[:, 1]; aspect = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*density, cd/(density+0.01),
                           cd*aspect, Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrange, Xmean])

# ── Tight path features (20-dim, per-placement) ────────────────────────
# Index mapping:
#   [0:5]  = dst(ad):  all-path  max, p90, mean, std, cv
#   [5:10] = dst(td):  tight(s<0.1) max, p90, mean, std, cv
#   [10:15]= dst(nd):  negative-slack max, p90, mean, std, cv
#   [15]   = len(td)/len(ad)
#   [16]   = len(nd)/len(ad)
#   [17]   = td_max/ad_max
#   [18]   = td_mean/ad_mean
#   [19]   = log1p(n_ff)
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp_cache.get(pid)
    if v is not None:
        arr = np.array(v, dtype=np.float32)
        X_tight[i, :min(20, len(arr))] = arr[:20]

tp_std = X_tight.std(axis=0); tp_std[tp_std < 1e-9] = 1.0
X_tight_n = X_tight / tp_std  # normalized

X49 = np.hstack([X29, X_tight_n])

# ── Novel: knob × tight-path interaction features ─────────────────────
# Physics:
#  cd / td_max  → if ratio >> 1, cluster can span entire tight path → low skew
#  cd / ad_p90  → cluster_dia vs typical long path
#  bd / td_mean → buffer stages per typical tight path (equalization quality)
#  cd / td_p90  → cluster can group 90th percentile of tight paths
#  (td_cv) × (1/cd) → path length variance × clustering difficulty

tp_ad_max  = X_tight[:, 0].clip(1e-4)   # max distance of all paths
tp_ad_p90  = X_tight[:, 1].clip(1e-4)   # p90 distance
tp_ad_mean = X_tight[:, 2].clip(1e-4)   # mean distance
tp_ad_cv   = X_tight[:, 4]              # cv of all paths
tp_td_max  = X_tight[:, 5].clip(1e-4)   # max tight path distance
tp_td_p90  = X_tight[:, 6].clip(1e-4)   # p90 tight path distance
tp_td_mean = X_tight[:, 7].clip(1e-4)   # mean tight path distance
tp_td_cv   = X_tight[:, 9]              # cv of tight paths
tp_nd_max  = X_tight[:, 10].clip(1e-4)  # max negative-slack path distance
tp_frac_td = X_tight[:, 15]             # fraction of tight paths
tp_frac_nd = X_tight[:, 16]             # fraction of negative paths

# Core interactions (all physically motivated)
cd_over_td_max  = cd / (tp_td_max  * 1000 + 1e-4)  # scale tp to die units
cd_over_td_p90  = cd / (tp_td_p90  * 1000 + 1e-4)
cd_over_ad_p90  = cd / (tp_ad_p90  * 1000 + 1e-4)
bd_over_td_mean = bd / (tp_td_mean * 1000 + 1e-4)
td_cv_times_inv_cd = tp_td_cv * (1.0 / (cd + 1.0))
cd_over_nd_max  = tp_nd_max * 1000 / (cd + 1e-4)
nd_coverage = tp_frac_nd * cd_over_nd_max
skew_risk = tp_td_cv * tp_frac_td * (1.0 / (cd + 1.0))

# Normalize each feature globally (not per-placement — these are cross-design)
raw_sk_inter = np.column_stack([
    cd_over_td_max,   # cluster spans tight path?
    cd_over_td_p90,   # cluster spans p90 tight path?
    cd_over_ad_p90,   # cluster spans p90 all paths?
    bd_over_td_mean,  # buffer density per tight path
    td_cv_times_inv_cd,  # path variance × clustering difficulty
    nd_coverage,      # negative-slack path / cluster radius
    skew_risk,        # combined skew risk score
    tp_td_cv,         # tight path cv (standalone)
    tp_frac_nd,       # fraction negative-slack paths
    tp_td_max * 1000, # max tight path distance (raw, µm units)
])

# Fix inf/nan
for c in range(raw_sk_inter.shape[1]):
    bad = ~np.isfinite(raw_sk_inter[:, c])
    if bad.any():
        raw_sk_inter[bad, c] = np.nanmedian(raw_sk_inter[~bad, c]) if (~bad).any() else 0.0
    raw_sk_inter[:, c] = np.clip(raw_sk_inter[:, c], -1e6, 1e6)

# Per-placement rank and centered versions
sk_inter_rank = np.zeros_like(raw_sk_inter)
sk_inter_cent = np.zeros_like(raw_sk_inter)
g_si = np.abs(raw_sk_inter).max(axis=0) + 1e-9
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(raw_sk_inter.shape[1]):
        v = raw_sk_inter[idxs, j]
        sk_inter_rank[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
        sk_inter_cent[idxs, j] = (v - v.mean()) / g_si[j]

Xsk_inter = np.hstack([raw_sk_inter / g_si, sk_inter_rank, sk_inter_cent])  # [n, 30]

# Clean
for arr in [X49, Xsk_inter]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:, c])
        if bad.any():
            arr[bad, c] = np.nanmedian(arr[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} Features: X29={X29.shape}, X49={X49.shape}, Xsk_inter={Xsk_inter.shape}")
print(f"  Sample cd/td_max: {cd_over_td_max[:5]}")
print(f"  Sample skew_risk: {skew_risk[:5]}")

XGB_300 = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)
XGB_500 = dict(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8,
               colsample_bytree=0.8, verbosity=0)

print(f"\n{T()} === SKEW EXPERIMENTS ===")
# Baselines
lodo_single(X29,                           y_sk_z, "X29 LGB baseline")
lodo_single(X49,                           y_sk_z, "X49 LGB baseline")
lodo_single(X49, y_sk_z, "X49 XGB_300", XGBRegressor, XGB_300)

# Novel: add interaction features
lodo_single(np.hstack([X29, Xsk_inter]), y_sk_z, "X29+skinter LGB")
lodo_single(np.hstack([X49, Xsk_inter]), y_sk_z, "X49+skinter LGB")
lodo_single(np.hstack([X29, Xsk_inter]), y_sk_z, "X29+skinter XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X49, Xsk_inter]), y_sk_z, "X49+skinter XGB_300", XGBRegressor, XGB_300)
lodo_single(np.hstack([X49, Xsk_inter]), y_sk_z, "X49+skinter XGB_500", XGBRegressor, XGB_500)

# Raw interaction only (no tight path in base)
lodo_single(np.hstack([X29, Xsk_inter[:, :10]]), y_sk_z, "X29+raw_inter_only XGB_300",
            XGBRegressor, XGB_300)

# Check: does tight alone add to X29?
lodo_single(np.hstack([X29, raw_sk_inter / g_si]), y_sk_z, "X29+raw_sk_only XGB_300",
            XGBRegressor, XGB_300)

print(f"\n{T()} === DONE ===")
