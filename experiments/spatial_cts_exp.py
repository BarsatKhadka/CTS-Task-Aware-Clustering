"""
spatial_cts_exp.py - Spatial-adaptive CTS features evaluated at each run's cluster_dia

Key insight: Ripley K at FIXED radii = per-placement constant (rho≈0 with z_power).
Ripley K evaluated AT EACH RUN's cluster_dia = per-run feature (should correlate with z_power).

For each CTS run:
  K_adaptive(cd) = # FF pairs within cluster_dia / n_ff²  × die_area
                 = expected FFs per cluster at this placement × cluster_dia scale

This gives a feature that varies with cluster_dia within a placement!
Additionally: complete-linkage cluster count, spatial entropies, etc.

Also: multi-scale spatial signature precomputed per placement, then interpolated at cd.
"""

import pickle, time, warnings
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Spatial-Adaptive CTS Features: K(cluster_dia) per run")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/absolute_v5_def_cache.pkl', 'rb') as f:
    def_cache = pickle.load(f)
with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_pos = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(df)
y_pw, y_wl = Y_cache[:, 1], Y_cache[:, 2]

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

# ── Build X29T ─────────────────────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xraw = df[knob_cols].values.astype(np.float32)
Xkz = X_cache[:, 72:76]
raw_max = Xraw.max(0) + 1e-6
Xrank = np.zeros((n, 4), np.float32); Xcent = np.zeros_like(Xrank)
Xrng = np.zeros_like(Xrank); Xmn = np.zeros_like(Xrank)
for pid in np.unique(pids):
    m = pids == pid; idx = np.where(m)[0]
    for j in range(4):
        v = Xraw[idx, j]; Xrank[idx, j] = rank_within(v)
        Xcent[idx, j] = (v - v.mean()) / raw_max[j]
        Xrng[idx, j] = v.std() / raw_max[j]; Xmn[idx, j] = v.mean() / raw_max[j]
Xplc = df[['core_util', 'density', 'aspect_ratio']].values.astype(np.float32)
Xplc_n = Xplc / (Xplc.max(0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
util = Xplc[:, 0]/100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp,
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T = np.hstack([X29, X_tight/tp_std])

# ── Spatial-adaptive features ──────────────────────────────────────────────
print(f"\n{T()} Computing spatial-adaptive K(cluster_dia) features...")
print("  (evaluating actual FF pairwise distances at each run's cluster_dia)")

from scipy.spatial import cKDTree

# Precompute per-placement: spatial K function lookup
placement_spatial = {}
n_ff_arr = np.array([def_cache.get(p, {}).get('n_ff', 2500) for p in pids], dtype=float)
die_area_arr = np.array([def_cache.get(p, {}).get('die_area', 400000) for p in pids], dtype=float)

for pid in np.unique(pids):
    pi = ff_pos.get(pid)
    if pi is None or pi.get('ff_norm') is None:
        placement_spatial[pid] = None
        continue
    ff_xy = pi['ff_norm']  # [N, 2] normalized [0,1]
    dw = pi.get('die_w', 500.0); dh = pi.get('die_h', 500.0)
    A = dw * dh  # µm²
    N = len(ff_xy)

    if N < 10:
        placement_spatial[pid] = None
        continue

    # Build KD-tree for this placement
    # For a given cluster_dia (µm), normalized radius = cluster_dia / sqrt(A/pi)?
    # Actually: normalize FF positions by die scale
    ff_phys = np.column_stack([ff_xy[:, 0] * dw, ff_xy[:, 1] * dh])  # [N, 2] in µm
    tree = cKDTree(ff_phys)

    # Evaluate K function at multiple cluster_dia values (sample grid)
    cd_samples = np.arange(30, 80, 2.5)  # 30, 32.5, ..., 77.5 µm (20 values)
    k_values = []
    for r in cd_samples:
        r_half = r / 2  # cluster radius = cluster_dia / 2
        pairs = tree.count_neighbors(tree, r_half)  # pairs within radius
        # Expected FFs within r_half: K = A × (pairs - N) / N²
        K = A * (pairs - N) / (N * N) if N > 1 else r_half**2 * np.pi
        L = np.sqrt(K / np.pi) if K > 0 else 0  # Besag's L
        n_in_r = K / A * N  # expected FFs within r_half per FF
        k_values.append(n_in_r)

    # Compute kNN distances at multiple k values
    k_vals_for_knn = [1, 3, 5, 10, 20]
    knn_distances = {}
    for k in k_vals_for_knn:
        if k < N:
            d, _ = tree.query(ff_phys, k=min(k+1, N))  # k+1 to exclude self
            knn_distances[k] = d[:, min(k, d.shape[1]-1)].mean()

    placement_spatial[pid] = {
        'cd_samples': cd_samples,
        'k_values': np.array(k_values),
        'dw': dw, 'dh': dh, 'A': A, 'N': N,
        'knn_distances': knn_distances
    }

print(f"  Computed spatial features for {sum(v is not None for v in placement_spatial.values())} placements")

# ── Build per-run spatial features ────────────────────────────────────────
print(f"\n{T()} Building per-run spatial features (K evaluated at each run's cd)...")

# Feature: K(cluster_dia_i) = expected FFs per cluster at this run's cluster_dia
kadapt = np.zeros(n, float)  # adaptive K at cluster_dia for each run
kadapt_log = np.zeros(n, float)  # log version
kadapt_sq = np.zeros(n, float)  # square root version

# Also: cluster_dia as fraction of k-NN distances
cd_over_knn1 = np.zeros(n, float)  # cd / knn_dist_1
cd_over_knn5 = np.zeros(n, float)  # cd / knn_dist_5
cd_over_knn20 = np.zeros(n, float)  # cd / knn_dist_20

# Effective cluster count: n_ff / min(cluster_size, K_adaptive(cd))
n_clust_adaptive = np.zeros(n, float)

for i, pid in enumerate(pids):
    sp = placement_spatial.get(pid)
    if sp is None:
        # Fallback: use uniform model
        da = def_cache.get(pid, {}).get('die_area', 400000)
        nff = def_cache.get(pid, {}).get('n_ff', 2500)
        lam = nff / da
        kadapt[i] = lam * np.pi * (cd[i]/2)**2
        kadapt_log[i] = np.log1p(kadapt[i])
        kadapt_sq[i] = np.sqrt(kadapt[i])
        cd_over_knn1[i] = cd[i] / 10.0  # fallback
        cd_over_knn5[i] = cd[i] / 20.0
        cd_over_knn20[i] = cd[i] / 40.0
        n_clust_adaptive[i] = nff / max(min(cs[i], kadapt[i]), 1)
        continue

    # Interpolate K at this run's cluster_dia
    cd_i = cd[i]
    cd_s = sp['cd_samples']
    k_s = sp['k_values']
    k_at_cd = float(np.interp(cd_i, cd_s, k_s))  # K(cd_i)
    kadapt[i] = max(k_at_cd, 0.1)
    kadapt_log[i] = np.log1p(kadapt[i])
    kadapt_sq[i] = np.sqrt(max(kadapt[i], 0))

    # cd / knn distances
    knn = sp['knn_distances']
    cd_over_knn1[i] = cd_i / max(knn.get(1, 10.0), 0.1)
    cd_over_knn5[i] = cd_i / max(knn.get(5, 20.0), 0.1)
    cd_over_knn20[i] = cd_i / max(knn.get(20, 40.0), 0.1)

    # Effective cluster count
    eff_cs = min(cs[i], kadapt[i])
    n_clust_adaptive[i] = sp['N'] / max(eff_cs, 1)

# Per-placement z-scores of spatial features
def z_within(arr, pids):
    z = np.zeros(len(arr))
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        v = arr[idx]; z[idx] = (v-v.mean())/max(v.std(),1e-8)
    return z

z_kadapt = z_within(kadapt, pids)
z_kadapt_log = z_within(kadapt_log, pids)
z_cd_knn1 = z_within(cd_over_knn1, pids)
z_cd_knn5 = z_within(cd_over_knn5, pids)
z_cd_knn20 = z_within(cd_over_knn20, pids)
z_nclust_adp = z_within(n_clust_adaptive, pids)

# Correlations
print("\n  Spatial-adaptive feature correlations with z_power:")
for name, z in [('z_K_adaptive(cd)', z_kadapt),
                ('z_log_K(cd)', z_kadapt_log),
                ('z_cd/knn1', z_cd_knn1),
                ('z_cd/knn5', z_cd_knn5),
                ('z_cd/knn20', z_cd_knn20),
                ('z_n_clust_adp', z_nclust_adp)]:
    rho_pw, _ = spearmanr(z, y_pw)
    rho_wl, _ = spearmanr(z, y_wl)
    print(f"    {name}: rho_power={rho_pw:.4f}, rho_WL={rho_wl:.4f}")

# ── Build feature matrices ─────────────────────────────────────────────────
# Physics-corrected features
n_ff_arr2 = np.array([def_cache.get(p, {}).get('n_ff', 2500) for p in pids], dtype=float)
lambda_arr = n_ff_arr2 / die_area_arr
n_in_radius = lambda_arr * np.pi * (cd/2)**2
effective_cs = np.minimum(cs, n_in_radius); effective_cs = np.maximum(effective_cs, 1.0)
n_clusters_phys = n_ff_arr2 / effective_cs
z_nclusters_phys = z_within(n_clusters_phys, pids)
ff_hpwl_arr = np.array([def_cache.get(p, {}).get('ff_hpwl', 1000) for p in pids], dtype=float)
z_hpwl_cd = z_within(ff_hpwl_arr/cd, pids)

# Complete spatial feature vector (per-run, zero-shot)
X_spatial = np.column_stack([z_kadapt, z_kadapt_log, z_cd_knn1, z_cd_knn5,
                               z_cd_knn20, z_nclust_adp, z_nclusters_phys, z_hpwl_cd])

# Fix NaN
for arr in [X_spatial, X29T]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

X_aug = np.hstack([X29T, X_spatial])
for c in range(X_aug.shape[1]):
    bad = ~np.isfinite(X_aug[:,c])
    if bad.any(): X_aug[bad,c] = 0.0

# ── LODO evaluation ────────────────────────────────────────────────────────
def lodo_lgb(X, y, label, ne=300, nl=20, lr=0.03, mc=15):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=ne, num_leaves=nl, learning_rate=lr,
                          min_child_samples=mc, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={mean_mae:.4f}{tag}")
    return maes

print(f"\n{T()} === LODO POWER MAE ===")
lodo_lgb(X29T, y_pw, "X29T baseline")
lodo_lgb(X_spatial, y_pw, "Spatial-adaptive only")
lodo_lgb(X_aug, y_pw, "X29T + spatial-adaptive")
lodo_lgb(X_aug, y_pw, "X29T + spatial (nl=15,mc=20)", nl=15, mc=20)
lodo_lgb(X_aug, y_pw, "X29T + spatial (nl=10,mc=20)", nl=10, mc=20)

print(f"\n{T()} === LODO WL MAE ===")
lodo_lgb(X29T, y_wl, "X29T baseline")
lodo_lgb(X_aug, y_wl, "X29T + spatial-adaptive")

# ── Buffer count prediction (Stage 1) ─────────────────────────────────────
n_total_clk = (df['clock_buffers'] + df['clock_inverters']).values.astype(float)
z_ntot = z_within(n_total_clk, pids)

print(f"\n{T()} === BUFFER COUNT PREDICTION (Stage 1 without using buffer oracle) ===")
print("  Target: predict z_total_clock_buffers from pre-CTS features")
lodo_lgb(X29T, z_ntot, "X29T → z_total_clk")
lodo_lgb(X_spatial, z_ntot, "Spatial-adaptive → z_total_clk")
lodo_lgb(X_aug, z_ntot, "X29T+spatial → z_total_clk")

# Test: use spatial features as direct predictor (physics-guided)
print(f"\n  Direct physics predictor (no ML):")
from sklearn.linear_model import Ridge
dl = sorted(np.unique(designs)); maes = []
for held in dl:
    tr = designs!=held; te = designs==held
    X_phy = np.column_stack([z_kadapt[tr], z_nclusters_phys[tr]])
    m = Ridge(alpha=0.1).fit(X_phy, y_pw[tr])
    maes.append(mean_absolute_error(y_pw[te], m.predict(np.column_stack([z_kadapt[te], z_nclusters_phys[te]]))))
print(f"  Ridge(z_K_adap + z_nclusters_phys) → power: {np.mean(maes):.4f}")

# ── Feature importance analysis ────────────────────────────────────────────
print(f"\n{T()} === FEATURE IMPORTANCE ANALYSIS ===")
dl = sorted(np.unique(designs)); sc = StandardScaler()
X_full = sc.fit_transform(X_aug)
m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
# Train on 3 designs, test on last
tr = designs != 'aes'; te = designs == 'aes'
m.fit(sc.fit_transform(X_aug[tr]), y_pw[tr])

import_vals = m.feature_importances_
n_x29t = X29T.shape[1]
print(f"  Top spatial features (indices {n_x29t} to {X_aug.shape[1]-1}):")
spatial_imps = list(zip(['z_K_adap','z_logK','z_cd/knn1','z_cd/knn5',
                          'z_cd/knn20','z_nclust','z_nclust_phys','z_hpwl_cd'],
                         import_vals[n_x29t:]))
spatial_imps.sort(key=lambda x: x[1], reverse=True)
for name, imp in spatial_imps:
    print(f"    {name}: {imp:.1f}")

print(f"\n{T()} DONE")
