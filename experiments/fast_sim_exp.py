"""
fast_sim_exp.py — Fast targeted experiments using physics simulation features

Confirmed: X29 + sim_geometry features → WL = 0.0825 (from 0.0849 best, 0.0870 this run)
Goal: Push WL further, test corrected simulation, Wasserstein features, skew

Uses LGB_300 for WL (fast) instead of XGB_1000 — confirmed similar results.
"""

import os, pickle, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

BASE = '/home/rain/CTS-Task-Aware-Clustering'
SIM_CACHE = f'{BASE}/sim_features_cache.pkl'
FF_CACHE  = f'{BASE}/ff_positions_cache.pkl'

t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Fast Simulation Experiments")
print("=" * 70)

# Load everything
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df_cache = cache['X'], cache['Y'], cache['df']
with open(SIM_CACHE, 'rb') as f:
    sim_cache = pickle.load(f)
with open(FF_CACHE, 'rb') as f:
    ff_cache = pickle.load(f)
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp_cache = pickle.load(f)

pids    = df_cache['placement_id'].values
designs = df_cache['design_name'].values
y_pw_z  = Y_cache[:, 1]
y_wl_z  = Y_cache[:, 2]
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

def lodo_single(X, y, pids_, designs_, cls_kw, pw_cls=LGBMRegressor):
    """Single-task LODO evaluation."""
    dl = sorted(np.unique(designs_))
    maes = []
    for held in dl:
        tr = designs_ != held; te = designs_ == held
        sc = StandardScaler()
        m = pw_cls(**cls_kw)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        maes.append(rank_mae(pred, y[te], pids_[te]))
    return np.mean(maes), maes

# X29 baseline features
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
place_cols = ['core_util', 'density', 'aspect_ratio']
Xraw = df_cache[knob_cols].values.astype(np.float32)
Xplc = df_cache[place_cols].values.astype(np.float32)
Xkz = X_cache[:, 72:76]
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
Xplc_n = Xplc / Xplc.max(axis=0)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]
util = Xplc[:, 0]/100.0; density = Xplc[:, 1]; aspect = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*density, cd/(density+0.01),
                           cd*aspect, Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrange, Xmean])

# Tight path features
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp_cache.get(pid)
    if v is not None:
        arr = np.array(v, dtype=np.float32)
        X_tight[i, :min(20, len(arr))] = arr[:20]
tp_std = X_tight.std(axis=0); tp_std[tp_std < 1e-9] = 1.0
X_tight_n = X_tight / tp_std
X49 = np.hstack([X29, X_tight_n])

def _empty_sim():
    return {k: 0.0 for k in ['sim_n_clusters', 'sim_sum_hpwl', 'sim_mean_hpwl',
                               'sim_max_hpwl', 'sim_min_hpwl', 'sim_std_hpwl',
                               'sim_skew_proxy', 'sim_skew_cv', 'sim_inter_hpwl',
                               'sim_intra_wl', 'sim_total_wl', 'sim_fill_rate',
                               'sim_mean_cs', 'sim_frac_single']}

# Sim geometry features (no n_clusters — pure geometry)
sim_geo_keys = ['sim_sum_hpwl', 'sim_mean_hpwl', 'sim_std_hpwl', 'sim_max_hpwl',
                'sim_min_hpwl', 'sim_inter_hpwl', 'sim_intra_wl', 'sim_fill_rate']
run_ids = df_cache['run_id'].values
raw_geo = np.array([[sim_cache.get(r, _empty_sim()).get(k, 0.0) for k in sim_geo_keys]
                    for r in run_ids], dtype=np.float32)
g_max_geo = raw_geo.max(axis=0) + 1e-9
raw_geo_n = raw_geo / g_max_geo
rank_geo = np.zeros_like(raw_geo_n); cent_geo = np.zeros_like(raw_geo_n)
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(raw_geo.shape[1]):
        v = raw_geo[idxs, j]
        rank_geo[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
        cent_geo[idxs, j] = (v - v.mean()) / (g_max_geo[j])
Xsim_geo = np.hstack([raw_geo_n, rank_geo, cent_geo])  # [n, 24]

# Corrected n_clusters (max of spatial vs n_ff/cs)
n_ff_map = {pid: len(ff_cache[pid]['ff_norm'])
            for pid in np.unique(pids) if ff_cache.get(pid) and ff_cache[pid] is not None}
corr_ncl = np.array([
    max(sim_cache.get(r, _empty_sim())['sim_n_clusters'],
        n_ff_map.get(df_cache.iloc[i]['placement_id'], 3000) / df_cache.iloc[i]['cts_cluster_size'])
    for i, r in enumerate(run_ids)], dtype=np.float32)
g_cncl = corr_ncl.max() + 1e-9
rank_cncl = np.zeros(n, np.float32); cent_cncl = np.zeros(n, np.float32)
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    v = corr_ncl[idxs]
    rank_cncl[idxs] = rank_within(v) if v.max() > v.min() else 0.5
    cent_cncl[idxs] = (v - v.mean()) / g_cncl
Xsim_corr = np.column_stack([corr_ncl/g_cncl, rank_cncl, cent_cncl])  # [n, 3]

# Fix NaNs
for X in [X29, X49, Xsim_geo, Xsim_corr]:
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} Features built: X29={X29.shape} X49={X49.shape} Xsim_geo={Xsim_geo.shape}")

# Fast model configs
LGB_F = dict(n_estimators=300, num_leaves=20, learning_rate=0.03, min_child_samples=15, verbose=-1)
LGB_SK = dict(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=15, verbose=-1)
XGB_F = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, verbosity=0)

# Baselines (fast model, LGB_300)
print(f"\n{T()} === POWER EXPERIMENTS (LGB_300, fast) ===")
for name, X in [('X29', X29), ('X49', X49),
                ('X29+corr', np.hstack([X29, Xsim_corr])),
                ('X29+simgeo', np.hstack([X29, Xsim_geo])),
                ('X49+simgeo', np.hstack([X49, Xsim_geo]))]:
    mae, maes = lodo_single(X, y_pw_z, pids, designs, LGB_F)
    print(f"  power {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

print(f"\n{T()} === WL EXPERIMENTS (LGB_300 fast, z-score targets) ===")
for name, X in [('X29', X29), ('X49', X49),
                ('X29+simgeo', np.hstack([X29, Xsim_geo])),
                ('X49+simgeo', np.hstack([X49, Xsim_geo])),
                ('X29+simgeo+corr', np.hstack([X29, Xsim_geo, Xsim_corr]))]:
    mae, maes = lodo_single(X, y_wl_z, pids, designs, LGB_F)
    print(f"  wl    {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

# Also test XGB_300 for WL (found better than LGB for WL in session 5)
print(f"\n{T()} === WL with XGB_300_d4 (best config per session 5) ===")
for name, X in [('X29', X29), ('X49', X49),
                ('X29+simgeo', np.hstack([X29, Xsim_geo])),
                ('X49+simgeo', np.hstack([X49, Xsim_geo]))]:
    mae, maes = lodo_single(X, y_wl_z, pids, designs, XGB_F, XGBRegressor)
    print(f"  wl_xgb {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

print(f"\n{T()} === SKEW EXPERIMENTS (XGB_300) ===")
for name, X in [('X29', X29), ('X49', X49),
                ('X29+simgeo', np.hstack([X29, Xsim_geo])),
                ('X49+simgeo', np.hstack([X49, Xsim_geo]))]:
    mae, maes = lodo_single(X, y_sk_z, pids, designs, XGB_F, XGBRegressor)
    print(f"  sk_xgb {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

# Novel: Wasserstein-based placement similarity features
print(f"\n{T()} === WASSERSTEIN PLACEMENT SIMILARITY (novel) ===")
# Build FF distribution quantile features per placement
unique_pids = np.unique(pids)
wass_pid_feats = {}
for pid in unique_pids:
    pos = ff_cache.get(pid)
    if pos is None or pos.get('ff_norm') is None:
        wass_pid_feats[pid] = np.zeros(44)
        continue
    ff_xy = pos['ff_norm']
    q = np.linspace(0, 1, 20)
    qx = np.quantile(ff_xy[:, 0], q)  # 20 quantiles of x distribution
    qy = np.quantile(ff_xy[:, 1], q)  # 20 quantiles of y distribution
    # 4x4 density grid (16 features)
    xs = np.floor(ff_xy[:, 0] * 4).clip(0, 3).astype(int)
    ys = np.floor(ff_xy[:, 1] * 4).clip(0, 3).astype(int)
    grid = np.zeros(16)
    for i, j in zip(xs, ys):
        grid[i * 4 + j] += 1
    grid /= (grid.sum() + 1e-9)
    wass_pid_feats[pid] = np.concatenate([qx, qy, grid])  # 20+20+16=56? no, 20+20+4=44

Xwass = np.zeros((n, 56), np.float32)
for i, pid in enumerate(pids):
    v = wass_pid_feats.get(pid, np.zeros(56))
    Xwass[i, :len(v)] = v

# Check design ID accuracy
design_labels = np.array([sorted(np.unique(designs)).index(d) for d in designs])
idx_s = np.random.default_rng(42).choice(n, 1000, replace=False)
lr = LogisticRegression(max_iter=100, C=0.1)
acc = cross_val_score(lr, StandardScaler().fit_transform(Xwass[idx_s]),
                       design_labels[idx_s], cv=5).mean()
print(f"  Wass features design ID accuracy: {acc:.3f} (random=0.25)")

for name, X in [('X29+Wass_pw', np.hstack([X29, Xwass])),
                ('X29+Wass_wl', np.hstack([X29, Xwass]))]:
    task = 'pw' if 'pw' in name else 'wl'
    y = y_pw_z if task == 'pw' else y_wl_z
    mae, maes = lodo_single(X, y, pids, designs, LGB_F)
    print(f"  {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

# Novel: Interaction between sim geometry and placement geometry
# Physical insight: "large ff_hpwl with small cluster_dia → many clusters, each large → high WL"
print(f"\n{T()} === NOVEL: Sim × Placement Geometry Interactions ===")
# Interaction features:
# sim_sum_hpwl × (1/cluster_dia) → direct WL predictor
# sim_n_clusters × (1/cluster_size) → direct power predictor
# sim_inter_hpwl / sim_sum_hpwl → ratio of inter/intra routing (balance indicator)
cd_inv = 1.0 / (Xraw[:, 3] + 1e-6)  # 1/cluster_dia
cs_inv = 1.0 / (Xraw[:, 2] + 1e-6)  # 1/cluster_size
sim_n  = np.array([sim_cache.get(r, _empty_sim())['sim_n_clusters'] for r in run_ids])
sim_sh = np.array([sim_cache.get(r, _empty_sim())['sim_sum_hpwl'] for r in run_ids])
sim_ih = np.array([sim_cache.get(r, _empty_sim())['sim_inter_hpwl'] for r in run_ids])
sim_ia = np.array([sim_cache.get(r, _empty_sim())['sim_intra_wl'] for r in run_ids])

# Physics-motivated interaction features
g_sn = sim_n.max() + 1e-9; g_sh = sim_sh.max() + 1e-9; g_ih = sim_ih.max() + 1e-9
Xphys_inter = np.column_stack([
    sim_sh / g_sh * cd_inv / cd_inv.max(),    # sum_hpwl × (1/cd) — WL proxy
    sim_n / g_sn * cs_inv / cs_inv.max(),     # n_clusters × (1/cs) — power proxy
    sim_ih / (sim_sh + 1e-6),                 # inter/sum ratio (routing structure)
    sim_ia / (sim_sh * 1.5 + 1e-6),           # actual vs ideal routing factor
])

# Normalize inter within placement
Xphys_rank = np.zeros_like(Xphys_inter); Xphys_cent = np.zeros_like(Xphys_inter)
g_pi = np.abs(Xphys_inter).max(axis=0) + 1e-9
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    for j in range(Xphys_inter.shape[1]):
        v = Xphys_inter[idxs, j]
        Xphys_rank[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
        Xphys_cent[idxs, j] = (v - v.mean()) / (g_pi[j])

Xphys = np.hstack([Xphys_inter, Xphys_rank, Xphys_cent])  # [n, 12]

for name, X, y, task in [
    ('X29+phys_pw', np.hstack([X29, Xphys]), y_pw_z, 'pw'),
    ('X29+phys_wl', np.hstack([X29, Xphys]), y_wl_z, 'wl'),
    ('X49+phys_wl', np.hstack([X49, Xphys]), y_wl_z, 'wl'),
    ('X49+simgeo+phys_wl', np.hstack([X49, Xsim_geo, Xphys]), y_wl_z, 'wl'),
]:
    mae, maes = lodo_single(X, y, pids, designs, LGB_F)
    print(f"  {name}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f}")

print(f"\n{T()} === BEST CANDIDATE PER TASK ===")
# Power: X29 LGB_300 → 0.0656
# WL: X29+simgeo or X49+simgeo
# Skew: X29 or better

for (tname, X, y, cls, kw) in [
    ('POWER X29', X29, y_pw_z, LGBMRegressor, LGB_F),
    ('WL    X29+simgeo', np.hstack([X29, Xsim_geo]), y_wl_z, LGBMRegressor, LGB_F),
    ('WL    X49+simgeo', np.hstack([X49, Xsim_geo]), y_wl_z, LGBMRegressor, LGB_F),
    ('SKEW  X49+simgeo', np.hstack([X49, Xsim_geo]), y_sk_z, XGBRegressor, XGB_F),
]:
    mae, maes = lodo_single(X, y, pids, designs, kw, cls)
    s = 'PASS' if mae < 0.10 else 'FAIL'
    print(f"  {tname}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mae:.4f} {s}")

print("\n" + "=" * 70)
print("SUMMARY vs prior best: sk=0.2372✗  pw=0.0656✓  wl=0.0849✓")
print("=" * 70)
