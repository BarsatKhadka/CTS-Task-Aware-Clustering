"""
skew_grid_exp.py - Grid-based CTS simulation for skew prediction

Key physics:
  Skew = max(clock_arrival) - min(clock_arrival)
  Clock arrivals depend on distance from clock source → cluster centroid → FF.

Novel approach: For each CTS run's cluster_dia, approximate cluster centroids
using a grid (divide die into cluster_dia-sized cells). Compute:
  - std of centroid distances to clock source → skew proxy
  - max centroid distance - min centroid distance → skew lower bound
  - n_clusters (grid cells with any FF)

These features VARY with cluster_dia within a placement (unlike fixed kNN stats).
The grid approximation is O(N) per run — fast even for ethmac (N=10546).

Hypothesis: std_centroid_dist should correlate with z_skew because:
  - Larger spread of centroid distances = more clock path length variation = more skew
  - This varies with cluster_dia: larger cd → larger cells → fewer centroids → different spread
"""

import pickle, time, warnings, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Skew Grid Simulation: Clock Tree Centroid Spread as Skew Proxy")
print("=" * 70)
sys.stdout.flush()

# ── Load data ─────────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/sim_ff_cache.pkl', 'rb') as f:
    sim_cache = pickle.load(f)

pids    = df['placement_id'].values
designs = df['design_name'].values
n       = len(pids)
y_sk    = Y_cache[:, 0]   # z_skew
y_pw    = Y_cache[:, 1]   # z_power
y_wl    = Y_cache[:, 2]   # z_wl

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def z_within(v, pids_):
    out = np.zeros(len(v), np.float32)
    for pid in np.unique(pids_):
        idx = np.where(pids_ == pid)[0]
        vv  = v[idx].astype(float)
        s   = max(vv.std(), 1e-8)
        out[idx] = (vv - vv.mean()) / s
    return out

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

# ── Build X29T ─────────────────────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xraw = df[knob_cols].values.astype(np.float32)
Xkz  = X_cache[:, 72:76]
raw_max = Xraw.max(0) + 1e-6
Xrank = np.zeros((n, 4), np.float32); Xcent = np.zeros_like(Xrank)
Xrng  = np.zeros_like(Xrank);          Xmn   = np.zeros_like(Xrank)
for pid in np.unique(pids):
    m = pids == pid; idx = np.where(m)[0]
    for j in range(4):
        v = Xraw[idx, j]; Xrank[idx, j] = rank_within(v)
        Xcent[idx, j] = (v - v.mean()) / raw_max[j]
        Xrng[idx, j]  = v.std() / raw_max[j]; Xmn[idx, j] = v.mean() / raw_max[j]
Xplc   = df[['core_util', 'density', 'aspect_ratio']].values.astype(np.float32)
Xplc_n = Xplc / (Xplc.max(0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
util = Xplc[:, 0]/100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp,
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29    = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9] = 1.0
X29T = np.hstack([X29, X_tight/tp_std])

# ── Grid-based CTS simulation ───────────────────────────────────────────────
print(f"\n{T()} Computing grid-based cluster centroid features...")
sys.stdout.flush()

# Features for each run (5390 rows)
grid_std_dist   = np.zeros(n, np.float32)  # std of centroid distances to clock
grid_max_dist   = np.zeros(n, np.float32)  # max centroid distance
grid_min_dist   = np.zeros(n, np.float32)  # min centroid distance
grid_range_dist = np.zeros(n, np.float32)  # max - min = skew lower bound
grid_n_clust    = np.zeros(n, np.float32)  # number of grid clusters
grid_p90_dist   = np.zeros(n, np.float32)  # 90th percentile distance
grid_gini_dist  = np.zeros(n, np.float32)  # Gini coefficient of distances

missing = 0
for i, pid in enumerate(pids):
    sp = sim_cache.get(pid)
    if sp is None or len(sp['xy']) == 0:
        missing += 1
        continue

    xy  = sp['xy']        # [N, 2] in µm
    cx  = sp.get('cx') or sp['die_w'] / 2
    cy  = sp.get('cy') or 0.0
    clk = np.array([cx, cy], dtype=np.float32)
    cd_i = float(cd[i])  # cluster_dia for this run (µm)

    if cd_i < 1.0:
        continue

    # Grid-based clustering: assign each FF to a grid cell
    cell_x = (xy[:, 0] / cd_i).astype(np.int32)
    cell_y = (xy[:, 1] / cd_i).astype(np.int32)
    cells  = cell_x * 100000 + cell_y  # unique cell hash

    _, inv, counts = np.unique(cells, return_inverse=True, return_counts=True)
    n_cells = len(counts)

    if n_cells < 2:
        continue

    # Vectorized centroid computation
    centroids = np.zeros((n_cells, 2), np.float32)
    np.add.at(centroids[:, 0], inv, xy[:, 0])
    np.add.at(centroids[:, 1], inv, xy[:, 1])
    centroids /= counts[:, None]

    # Distance from each centroid to clock source
    dists = np.sqrt(((centroids - clk) ** 2).sum(axis=1))

    grid_std_dist[i]   = dists.std()
    grid_max_dist[i]   = dists.max()
    grid_min_dist[i]   = dists.min()
    grid_range_dist[i] = dists.max() - dists.min()
    grid_n_clust[i]    = n_cells
    grid_p90_dist[i]   = np.percentile(dists, 90)

    # Gini coefficient of distances (0=equal, 1=max inequality)
    d_sorted = np.sort(dists)
    n_d = len(d_sorted)
    if d_sorted.sum() > 0:
        gini = (2 * np.sum((np.arange(1, n_d+1)) * d_sorted) / (n_d * d_sorted.sum()) - (n_d+1)/n_d)
        grid_gini_dist[i] = max(0, gini)

print(f"  Missing: {missing}/{n}, computed: {n-missing}/{n}")
sys.stdout.flush()

# ── Correlations ───────────────────────────────────────────────────────────
print(f"\n{T()} Correlations with z_skew and z_power:")
for name, arr in [
    ('grid_std_dist',   grid_std_dist),
    ('grid_range_dist', grid_range_dist),
    ('grid_max_dist',   grid_max_dist),
    ('grid_p90_dist',   grid_p90_dist),
    ('grid_n_clust',    grid_n_clust),
    ('grid_gini',       grid_gini_dist),
]:
    rho_sk, _ = spearmanr(arr, y_sk)
    rho_pw, _ = spearmanr(arr, y_pw)
    rho_wl, _ = spearmanr(arr, y_wl)
    print(f"  {name}: rho_sk={rho_sk:.4f}  rho_pw={rho_pw:.4f}  rho_wl={rho_wl:.4f}")
sys.stdout.flush()

# ── Z-score within placement ─────────────────────────────────────────────
z_grid_std   = z_within(grid_std_dist,   pids)
z_grid_range = z_within(grid_range_dist, pids)
z_grid_max   = z_within(grid_max_dist,   pids)
z_grid_p90   = z_within(grid_p90_dist,   pids)
z_grid_nc    = z_within(grid_n_clust,    pids)
z_grid_gini  = z_within(grid_gini_dist,  pids)

print(f"\n{T()} Correlations of z_within versions with z_skew:")
for name, z in [
    ('z_grid_std',   z_grid_std),
    ('z_grid_range', z_grid_range),
    ('z_grid_max',   z_grid_max),
    ('z_grid_p90',   z_grid_p90),
    ('z_grid_nc',    z_grid_nc),
    ('z_grid_gini',  z_grid_gini),
]:
    rho_sk, _ = spearmanr(z, y_sk)
    rho_pw, _ = spearmanr(z, y_pw)
    print(f"  {name}: rho_sk={rho_sk:.4f}  rho_pw={rho_pw:.4f}")
sys.stdout.flush()

# ── LODO evaluation ────────────────────────────────────────────────────────
X_grid = np.column_stack([z_grid_std, z_grid_range, z_grid_max,
                           z_grid_p90, z_grid_nc, z_grid_gini])
for c in range(X_grid.shape[1]):
    bad = ~np.isfinite(X_grid[:, c]); X_grid[bad, c] = 0.0

X29T_grid = np.hstack([X29T, X_grid])
for c in range(X29T_grid.shape[1]):
    bad = ~np.isfinite(X29T_grid[:, c]); X29T_grid[bad, c] = 0.0

def lodo_rank(X, y, label, cls=LGBMRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m  = cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        maes.append(rank_mae(pred, y[te], pids[te]))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    sys.stdout.flush()
    return mean_mae, maes

print(f"\n{T()} === SKEW RANK MAE LODO (prior best ~0.237) ===")
lodo_rank(X29T,      y_sk, "X29T LGB (baseline)")
lodo_rank(X_grid,    y_sk, "Grid-only LGB")
lodo_rank(X29T_grid, y_sk, "X29T + grid LGB")

xgb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
               subsample=0.8, colsample_bytree=0.8, verbosity=0)
lodo_rank(X29T_grid, y_sk, "X29T + grid XGB_300", cls=XGBRegressor, kw=xgb_kw)
lodo_rank(X29T_grid, y_sk, "X29T + grid XGB_300_mc10",
          cls=XGBRegressor, kw={**xgb_kw, 'min_child_weight': 10})

# Also check rank targets (train on ranks, evaluate on ranks)
print(f"\n{T()} === SKEW (rank targets, rank eval) ===")
y_sk_rank = np.zeros(n, np.float32)
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    if len(idx) > 1:
        y_sk_rank[idx] = rank_within(y_sk[idx])
    else:
        y_sk_rank[idx] = 0.5

lodo_rank(X29T,      y_sk_rank, "X29T rank-targets LGB")
lodo_rank(X29T_grid, y_sk_rank, "X29T+grid rank-targets LGB")
lodo_rank(X29T_grid, y_sk_rank, "X29T+grid rank-targets XGB",
          cls=XGBRegressor, kw=xgb_kw)

# ── Check power/WL unchanged ────────────────────────────────────────────────
print(f"\n{T()} === POWER / WL (sanity check, rank MAE) ===")
lodo_rank(X29T,      y_pw, "X29T power (should be ~0.066)")
lodo_rank(X29T_grid, y_pw, "X29T+grid power")
lodo_rank(X29T,      y_wl, "X29T WL (should be ~0.087)")
lodo_rank(X29T_grid, y_wl, "X29T+grid WL")

# ── Feature importance of grid features ────────────────────────────────────
print(f"\n{T()} Feature importance of grid features for skew:")
tr = designs != 'aes'; te = designs == 'aes'
sc = StandardScaler()
m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                   min_child_samples=15, verbose=-1)
m.fit(sc.fit_transform(X29T_grid[tr]), y_sk[tr])
imps = m.feature_importances_
n_x29t = X29T.shape[1]
grid_names = ['z_grid_std', 'z_grid_range', 'z_grid_max', 'z_grid_p90', 'z_grid_nc', 'z_grid_gini']
print("  Top grid feature importances (held=aes):")
for name, imp in sorted(zip(grid_names, imps[n_x29t:]), key=lambda x: -x[1]):
    print(f"    {name}: {imp}")

print(f"\n{T()} DONE")
