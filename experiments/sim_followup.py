"""
sim_followup.py — Targeted follow-up experiments from physics simulation

Key findings from physics_sim_lodo.py:
1. Simulation alone HURTS power (0.0739 vs baseline 0.0662)
2. Simulation helps WL when combined: X29+Xsim = 0.0838 vs baseline 0.0870
3. SHA256 WL improved from 0.0790 → 0.0746 with simulation features

Strategy: Use task-specific feature sets:
- Power: X29 ONLY (simulation contaminates)
- WL: X29 + sim_geometry features (novel information: cluster shape)
- Skew: X29 + try sim_skew features

Novel approaches:
1. Sim geometry features (NOT n_clusters) for WL
2. Wasserstein kernel Nadaraya-Watson regression (similarity-weighted prediction)
3. Corrected simulation (explicit cluster_size binding constraint)
4. Two-stage: predict sim residuals, then add
"""

import os, sys, pickle, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
SIM_CACHE = f'{BASE}/sim_features_cache.pkl'
FF_CACHE  = f'{BASE}/ff_positions_cache.pkl'

t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"


# -----------------------------------------------------------------------
# Load existing caches
# -----------------------------------------------------------------------

print("=" * 70)
print("Simulation Follow-up: Task-Specific Feature Sets")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache = cache['X']
Y_cache = cache['Y']   # [5390, 3]: z-scored (sk, pw, wl) per placement
df_cache = cache['df']

with open(SIM_CACHE, 'rb') as f:
    sim_cache = pickle.load(f)

with open(FF_CACHE, 'rb') as f:
    ff_cache = pickle.load(f)

pids     = df_cache['placement_id'].values
designs  = df_cache['design_name'].values
y_sk_z   = Y_cache[:, 0]
y_pw_z   = Y_cache[:, 1]
y_wl_z   = Y_cache[:, 2]

print(f"{T()} Data loaded: {len(pids)} rows")


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def rank_within(vals):
    n = len(vals)
    return np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)


def zscore_to_rank_mae(pred, y_true, pids_):
    all_pred_r = np.zeros(len(pids_))
    all_true_r = np.zeros(len(pids_))
    for pid in np.unique(pids_):
        m = pids_ == pid
        idxs = np.where(m)[0]
        if len(idxs) > 1:
            all_pred_r[idxs] = rank_within(pred[idxs])
            all_true_r[idxs] = rank_within(y_true[idxs])
        else:
            all_pred_r[idxs] = 0.5
            all_true_r[idxs] = 0.5
    return mean_absolute_error(all_true_r, all_pred_r)


def _empty_sim():
    return {k: 0.0 for k in [
        'sim_n_clusters', 'sim_sum_hpwl', 'sim_mean_hpwl', 'sim_max_hpwl',
        'sim_min_hpwl', 'sim_std_hpwl', 'sim_skew_proxy', 'sim_skew_cv',
        'sim_inter_hpwl', 'sim_intra_wl', 'sim_total_wl', 'sim_fill_rate',
        'sim_mean_cs', 'sim_frac_single']}


def lodo(X_sk, X_pw, X_wl, y_sk, y_pw, y_wl, pids_, designs_,
         sk_cls, sk_kw, pw_cls, pw_kw, wl_cls, wl_kw, name=""):
    dl = sorted(np.unique(designs_))
    sk_m, pw_m, wl_m = [], [], []
    for held in dl:
        tr = designs_ != held
        te = designs_ == held
        results = {}
        for label, X, y, cls, kw in [
            ('sk', X_sk, y_sk, sk_cls, sk_kw),
            ('pw', X_pw, y_pw, pw_cls, pw_kw),
            ('wl', X_wl, y_wl, wl_cls, wl_kw),
        ]:
            sc = StandardScaler()
            m = cls(**kw)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            pred = m.predict(sc.transform(X[te]))
            mae = zscore_to_rank_mae(pred, y[te], pids_[te])
            results[label] = mae
        sk_m.append(results['sk']); pw_m.append(results['pw']); wl_m.append(results['wl'])
        print(f"  {held}: sk={results['sk']:.4f}  pw={results['pw']:.4f}  wl={results['wl']:.4f}")
    ms = np.mean(sk_m); mp = np.mean(pw_m); mw = np.mean(wl_m)
    sp = 'PASS' if mp < 0.10 else 'FAIL'
    sw = 'PASS' if mw < 0.10 else 'FAIL'
    ss = 'PASS' if ms < 0.10 else 'FAIL'
    print(f"  [{name}] MEAN: sk={ms:.4f}{ss}  pw={mp:.4f}{sp}  wl={mw:.4f}{sw}\n")
    return ms, mp, mw


# -----------------------------------------------------------------------
# Feature building
# -----------------------------------------------------------------------

knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
place_cols = ['core_util', 'density', 'aspect_ratio']
Xraw   = df_cache[knob_cols].values.astype(np.float32)
Xplc   = df_cache[place_cols].values.astype(np.float32)
Xkz    = X_cache[:, 72:76]
raw_max = Xraw.max(axis=0) + 1e-6

n = len(pids)
Xrank = np.zeros((n, 4), np.float32)
Xcent = np.zeros((n, 4), np.float32)
Xrange = np.zeros((n, 4), np.float32)
Xmean  = np.zeros((n, 4), np.float32)

for pid in np.unique(pids):
    m = pids == pid
    idxs = np.where(m)[0]
    for j in range(4):
        v = Xraw[idxs, j]
        Xrank[idxs, j]  = rank_within(v)
        Xcent[idxs, j]  = (v - v.mean()) / raw_max[j]
        Xrange[idxs, j] = v.std() / raw_max[j]
        Xmean[idxs, j]  = v.mean() / raw_max[j]

Xplc_norm = Xplc / Xplc.max(axis=0)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]
util = Xplc[:, 0] / 100.0; density = Xplc[:, 1]; aspect = Xplc[:, 2]

Xinter = np.column_stack([
    cd * util, mw * density, cd / (density + 0.01),
    cd * aspect, Xrank[:, 3] * util, Xrank[:, 2] * util
])

X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_norm, Xinter, Xrange, Xmean])

# -----------------------------------------------------------------------
# Simulation features (raw, rank, centered per placement)
# -----------------------------------------------------------------------

sim_keys = ['sim_n_clusters', 'sim_sum_hpwl', 'sim_total_wl',
            'sim_skew_proxy', 'sim_fill_rate', 'sim_skew_cv', 'sim_frac_single',
            'sim_mean_hpwl', 'sim_std_hpwl', 'sim_inter_hpwl', 'sim_intra_wl']
sim_geo_keys = ['sim_sum_hpwl', 'sim_mean_hpwl', 'sim_std_hpwl', 'sim_max_hpwl',
                'sim_min_hpwl', 'sim_inter_hpwl', 'sim_intra_wl', 'sim_fill_rate']
sim_count_keys = ['sim_n_clusters']

run_ids = df_cache['run_id'].values
raw_all = np.array([[sim_cache.get(r, _empty_sim()).get(k, 0.0)
                     for k in sim_keys] for r in run_ids], dtype=np.float32)
raw_geo = np.array([[sim_cache.get(r, _empty_sim()).get(k, 0.0)
                     for k in sim_geo_keys] for r in run_ids], dtype=np.float32)
raw_cnt = np.array([[sim_cache.get(r, _empty_sim()).get(k, 0.0)
                     for k in sim_count_keys] for r in run_ids], dtype=np.float32)

def _add_rank_cent(raw, keys_list):
    """Add per-placement rank and centered columns to raw features."""
    g_max = raw.max(axis=0) + 1e-9
    raw_n = raw / g_max
    rank_a = np.zeros_like(raw_n)
    cent_a = np.zeros_like(raw_n)
    for pid in np.unique(pids):
        m = pids == pid; idxs = np.where(m)[0]
        for j in range(raw.shape[1]):
            v = raw[idxs, j]
            rank_a[idxs, j] = rank_within(v) if v.max() > v.min() else 0.5
            cent_a[idxs, j] = (v - v.mean()) / (g_max[j])
    return np.hstack([raw_n, rank_a, cent_a])

Xsim     = _add_rank_cent(raw_all, sim_keys)     # all sim features × 3 = 33
Xsim_geo = _add_rank_cent(raw_geo, sim_geo_keys) # geometry-only × 3 = 24
Xsim_cnt = _add_rank_cent(raw_cnt, sim_count_keys) # n_clusters only × 3 = 3

# Fix NaN/Inf
for X in [X29, Xsim, Xsim_geo, Xsim_cnt]:
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} Features: X29={X29.shape}  Xsim={Xsim.shape}  Xsim_geo={Xsim_geo.shape}")

# -----------------------------------------------------------------------
# Tight path features (from existing cache)
# -----------------------------------------------------------------------

with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp_cache = pickle.load(f)

X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp_cache.get(pid)
    if v is not None:
        arr = np.array(v, dtype=np.float32)
        X_tight[i, :min(20, len(arr))] = arr[:20]

# Normalize tight features
tp_std = X_tight.std(axis=0)
tp_std[tp_std < 1e-9] = 1.0
X_tight_n = X_tight / tp_std

X49 = np.hstack([X29, X_tight_n])  # X29 + tight = X49

print(f"{T()} X49={X49.shape}")

# -----------------------------------------------------------------------
# Model configs
# -----------------------------------------------------------------------

LGB_SK = dict(n_estimators=300, num_leaves=15, learning_rate=0.05, min_child_samples=15, verbose=-1)
LGB_PW = dict(n_estimators=300, num_leaves=20, learning_rate=0.03, min_child_samples=15, verbose=-1)
XGB_WL = dict(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, verbosity=0)
XGB_SK = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
XGB_PW = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

# -----------------------------------------------------------------------
# EXPERIMENTS
# -----------------------------------------------------------------------

# --- EXP A: Task-specific feature sets (key insight from analysis) ---
# Power: X29 only (sim hurts power)
# WL: X29 + sim_geo (geometry features help WL, not count features)
# Skew: X29 + sim_all (exploration)
X_pw_A = X29
X_wl_A = np.hstack([X29, Xsim_geo])
X_sk_A = np.hstack([X29, Xsim])
print(f"\n{T()} === EXP A: Task-specific features (X29/X29+simgeo/X29+sim) ===")
lodo(X_sk_A, X_pw_A, X_wl_A, y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="task_specific")

# --- EXP B: X29+simgeo for WL vs X29 ---
print(f"\n{T()} === EXP B: WL focus — X29 vs X29+simgeo vs X29+sim ===")
for name, X_wl in [
    ('X29_only', X29),
    ('X29+simgeo', np.hstack([X29, Xsim_geo])),
    ('X29+sim_all', np.hstack([X29, Xsim])),
    ('X49+simgeo', np.hstack([X49, Xsim_geo])),
]:
    lodo(X29, X29, X_wl, y_sk_z, y_pw_z, y_wl_z, pids, designs,
         XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
         name=f"wl_{name}")

# --- EXP C: Wasserstein-kernel Nadaraya-Watson regression for WL ---
# The key idea: for each test placement, find the most similar training placements
# by Wasserstein distance of FF distributions, and use their knob-WL functions
print(f"\n{T()} === EXP C: Wasserstein-Kernel Regression (novel) ===")

def build_wass_features(ff_cache_in, df_in, pids_in):
    """
    For each placement pair, compute Wasserstein distance of FF distributions.
    Returns per-row features: wass similarity to 10 most similar training placements.

    For LODO: compute pairwise Wass distances between all placements,
    then for each test placement, find k-NN in FF distribution space.
    """
    # For each unique placement, compute 1D Wasserstein along x and y
    unique_pids = np.unique(pids_in)
    wass_feats = {}  # pid → (wass_x, wass_y) marginals

    for pid in unique_pids:
        pos = ff_cache_in.get(pid)
        if pos is None or pos.get('ff_norm') is None:
            wass_feats[pid] = None
            continue
        ff_xy = pos['ff_norm']  # [n_ff, 2] in [0,1]
        # Quantile features of FF distribution (5 quantiles for x and y)
        q = np.linspace(0, 1, 20)
        qx = np.quantile(ff_xy[:, 0], q)
        qy = np.quantile(ff_xy[:, 1], q)
        # Also: moments (mean, std, skew for x and y)
        mx, sx = ff_xy[:, 0].mean(), ff_xy[:, 0].std()
        my, sy = ff_xy[:, 1].mean(), ff_xy[:, 1].std()
        # FF density map (4x4 grid)
        xs = np.floor(ff_xy[:, 0] * 4).clip(0, 3).astype(int)
        ys = np.floor(ff_xy[:, 1] * 4).clip(0, 3).astype(int)
        grid = np.zeros(16)
        for i, j in zip(xs, ys):
            grid[i * 4 + j] += 1
        grid = grid / (grid.sum() + 1e-9)

        wass_feats[pid] = np.concatenate([qx, qy, [mx, sx, my, sy], grid])

    return wass_feats


print(f"  Computing Wasserstein features for all placements...")
wass_feats = build_wass_features(ff_cache, df_cache, pids)
n_ok = sum(1 for v in wass_feats.values() if v is not None)
print(f"  Wass features computed for {n_ok}/{len(wass_feats)} placements")

# Build Wass feature matrix (one vec per row, using the placement's wass features)
W_dim = 60  # 20+20+4+16
Xwass = np.zeros((n, W_dim), np.float32)
for i, pid in enumerate(pids):
    v = wass_feats.get(pid)
    if v is not None:
        Xwass[i, :len(v)] = v[:W_dim]

# Check if wass features are design-specific (if so, LODO won't work)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
design_labels = np.array([sorted(np.unique(designs)).index(d) for d in designs])
# Sample to speed up
idx_sample = np.random.default_rng(42).choice(len(designs), 1000, replace=False)
Xwass_s = StandardScaler().fit_transform(Xwass[idx_sample])
design_s = design_labels[idx_sample]
lr = LogisticRegression(max_iter=100)
acc = cross_val_score(lr, Xwass_s, design_s, cv=5).mean()
print(f"  Wass features design ID accuracy: {acc:.3f} (vs 0.25 random)")

if acc > 0.7:
    print("  WARNING: Wass features are design-specific! May hurt LODO.")
else:
    print("  Wass features are NOT design-specific. Safe for LODO.")

# Use Wass features with X29 for WL prediction
X_wl_wass = np.hstack([X29, Xwass])
print(f"\n{T()} === EXP C1: X29 + Wasserstein FF distribution features for WL ===")
lodo(X29, X29, X_wl_wass, y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="X29+Wass_WL")

# --- EXP D: Corrected simulation (cluster_size binding) ---
# Hypothesis: sim_n_clusters underestimates when cluster_size should be binding
# Correction: n_clusters_corrected = max(sim_n_clusters, n_ff/cluster_size * correction)
# For each run, compute corrected n_clusters
print(f"\n{T()} === EXP D: Corrected simulation (explicit n_ff/cs binding) ===")
n_ff_per_design = {}
for pid in np.unique(pids):
    pos = ff_cache.get(pid)
    if pos and pos.get('ff_norm') is not None:
        n_ff_per_design[pid] = len(pos['ff_norm'])

# Build corrected features
corrected_ncl = np.zeros(n, np.float32)
for i, (_, row) in enumerate(df_cache.iterrows()):
    pid = row['placement_id']
    cs = int(row['cts_cluster_size'])
    nff = n_ff_per_design.get(pid, 3000)
    sim_ncl = sim_cache.get(row['run_id'], _empty_sim())['sim_n_clusters']
    # Physical minimum clusters = ceil(n_ff / cluster_size)
    min_ncl = np.ceil(nff / cs)
    # Take the maximum (binding constraint)
    corrected_ncl[i] = max(sim_ncl, min_ncl)

# Build rank/centered of corrected_ncl
corrected_rank = np.zeros(n, np.float32)
corrected_cent = np.zeros(n, np.float32)
g_corrected = corrected_ncl.max() + 1e-9
for pid in np.unique(pids):
    m = pids == pid; idxs = np.where(m)[0]
    v = corrected_ncl[idxs]
    corrected_rank[idxs] = rank_within(v) if v.max() > v.min() else 0.5
    corrected_cent[idxs] = (v - v.mean()) / g_corrected

Xsim_corrected = np.column_stack([
    corrected_ncl / g_corrected,  # normalized corrected n_clusters
    corrected_rank,               # rank of corrected n_clusters (power proxy!)
    corrected_cent,               # centered corrected n_clusters
])

X_pw_D = np.hstack([X29, Xsim_corrected])  # corrected n_clusters for power

print(f"  Corrected sim shape: {Xsim_corrected.shape}")
# Check: does corrected n_clusters improve power?
print(f"  Testing corrected n_clusters oracle for power:")
corr_oracle_maes = []
for held in sorted(np.unique(designs)):
    te = designs == held
    # Oracle: rank(-corrected_ncl) within placement → power rank
    oracle_pw_pred = 1.0 - corrected_rank[te]  # power decreases as n_clusters increases
    oracle_pw_true = np.zeros(te.sum())
    for pid in np.unique(pids[te]):
        m = pids[te] == pid; idxs = np.where(m)[0]
        oracle_pw_true[idxs] = rank_within(y_pw_z[te][idxs])
    mae = mean_absolute_error(oracle_pw_true, 1.0 - corrected_rank[te])
    corr_oracle_maes.append(mae)
    print(f"    {held}: corrected oracle power MAE = {mae:.4f}")
print(f"  Mean corrected oracle power MAE: {np.mean(corr_oracle_maes):.4f}")

lodo(X29, X_pw_D, np.hstack([X29, Xsim_geo]), y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="corrected_sim_pw")

# --- EXP E: Best WL model (X49 with sim geometry) ---
print(f"\n{T()} === EXP E: Best WL attempt — X49 + all sim features ===")
lodo(X29, X29, np.hstack([X49, Xsim_geo]), y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="X49+simgeo_wl")

# --- EXP F: Sim geometric features for skew ---
# Test: sim_skew_proxy (max-min cluster HPWL) + sim_skew_cv + X29 for skew
X_sk_F = np.hstack([X29, Xsim_geo])  # geometry features may capture skew balance
print(f"\n{T()} === EXP F: Skew with sim geometry features ===")
lodo(X_sk_F, X29, X29, y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="sk_simgeo")

# --- EXP G: Combined best per task ---
print(f"\n{T()} === EXP G: Best per task (final submission candidate) ===")
X_sk_best = X29
X_pw_best = X29
X_wl_best = np.hstack([X49, Xsim_geo])
lodo(X_sk_best, X_pw_best, X_wl_best, y_sk_z, y_pw_z, y_wl_z, pids, designs,
     XGBRegressor, XGB_SK, LGBMRegressor, LGB_PW, XGBRegressor, XGB_WL,
     name="BEST_CANDIDATE")

print("\n" + "=" * 70)
print("SUMMARY vs best prior results (sk=0.2372, pw=0.0656, wl=0.0849)")
print("=" * 70)
