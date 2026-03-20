"""
sim_physics_exp.py - Physics simulation of CTS outcomes using FF positions

Key insight: CTS groups FFs within cluster_dia radius. We can SIMULATE this
using actual FF positions (ff_positions_cache) to get exact cluster topology.
This gives physics-principled features that are design-agnostic.

For each placement × CTS run:
  1. Greedy clustering: connected components of FFs within cd_frac
  2. Constrained by cluster_size: split large components
  3. Compute: n_clusters, within-cluster WL, inter-cluster WL (MST proxy)
  4. Z-score per placement → direct comparison with power/WL z-scores

Goal: beat direct z-score MAE of 0.21 (power) and 0.24 (WL)
"""

import pickle, time, warnings
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Physics Simulation of CTS Outcomes")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_pos = pickle.load(f)

pids = df['placement_id'].values
designs = df['design_name'].values
n = len(pids)

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def lodo(X, y, label, cls=LGBMRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held; sc = StandardScaler()
        m = cls(**kw); m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te])); maes.append(mean_absolute_error(y[te], pred))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.05 else (' ~' if mean_mae < 0.10 else '')
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    return mean_mae

# ── Build X29T base features ──────────────────────────────────────────────
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
cd_raw = Xraw[:, 3]; cs_raw = Xraw[:, 2]; mw_raw = Xraw[:, 0]; bd_raw = Xraw[:, 1]
util = Xplc[:, 0]/100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd_raw*util, mw_raw*dens, cd_raw/(dens+0.01), cd_raw*asp,
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i, :20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std < 1e-9] = 1.0
X29T = np.hstack([X29, X_tight / tp_std])

# ── Physics simulation ────────────────────────────────────────────────────
print(f"{T()} Running CTS physics simulation for {len(np.unique(pids))} placements...")

def simulate_cts_run(ff_xy, cd_frac, cs_max, die_w, die_h):
    """
    Simulate one CTS run for a given placement and knob set.

    Returns:
      n_clusters: number of leaf clusters
      wl_within: total within-cluster wire length (µm, estimate)
      wl_inter: inter-cluster wire length estimate (µm, MST proxy)
      max_cluster_size: largest cluster
      mean_cluster_radius: mean radius of clusters
    """
    n_ff = len(ff_xy)
    if n_ff == 0:
        return 0, 0.0, 0.0, 0, 0.0

    # Build adjacency within cd_frac (in [0,1] die units)
    tree = cKDTree(ff_xy)
    # Find all pairs within cd_frac
    pairs = tree.query_pairs(cd_frac, output_type='ndarray')
    if len(pairs) == 0:
        # No clustering: each FF is its own cluster
        n_clusters = n_ff
        wl_within = 0.0
        # Inter-cluster = MST of all FFs (use nearest-neighbor heuristic)
        dist_nn, _ = tree.query(ff_xy, k=min(2, n_ff))
        wl_inter = dist_nn[:, 1].sum() * (die_w + die_h) / 2 if n_ff > 1 else 0.0
        return n_clusters, wl_within, wl_inter, 1, 0.0

    # Connected components
    row = pairs[:, 0]; col = pairs[:, 1]
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(n_ff, n_ff))
    adj = adj + adj.T  # symmetrize
    n_comp, labels = connected_components(adj, directed=False)

    # Apply cluster_size constraint: split oversized clusters
    cluster_info = {}
    for cid in range(n_comp):
        members = np.where(labels == cid)[0]
        cluster_info[cid] = members

    # Split oversized clusters (greedy spatial split)
    final_clusters = []
    for cid, members in cluster_info.items():
        if len(members) <= cs_max:
            final_clusters.append(members)
        else:
            # Greedy split: sort by x, assign greedily
            xy_m = ff_xy[members]
            order = np.argsort(xy_m[:, 0])
            for i in range(0, len(members), cs_max):
                final_clusters.append(members[order[i:i+cs_max]])

    n_clusters = len(final_clusters)

    # Compute within-cluster wire lengths (sum of distances to centroid × 2)
    wl_within = 0.0
    centroids = []
    radii = []
    for clust in final_clusters:
        xy_c = ff_xy[clust]
        centroid = xy_c.mean(0)
        centroids.append(centroid)
        dists = np.linalg.norm(xy_c - centroid, axis=1)
        wl_within += dists.sum()
        radii.append(dists.max() if len(dists) > 0 else 0.0)

    # Scale to µm
    scale = (die_w + die_h) / 2
    wl_within *= scale * 2  # × 2 for routing (clock goes down and back)

    # Inter-cluster WL: MST of centroids (approximate with sorted nearest-neighbor)
    if n_clusters > 1:
        ctrs = np.array(centroids)
        ctr_tree = cKDTree(ctrs)
        nn_dists, _ = ctr_tree.query(ctrs, k=min(2, n_clusters))
        wl_inter = nn_dists[:, 1].sum() * scale * 1.5  # 1.5 = Manhattan routing factor
    else:
        wl_inter = 0.0

    max_cs = max(len(c) for c in final_clusters) if final_clusters else 0
    mean_radius = np.mean(radii) * scale if radii else 0.0

    return n_clusters, wl_within, wl_inter, max_cs, mean_radius

# Precompute per-placement data
pid_ff = {}  # pid → ff_xy (in [0,1])
pid_die = {}  # pid → (die_w, die_h)
for pid in np.unique(pids):
    pi = ff_pos.get(pid)
    if pi is not None and pi.get('ff_norm') is not None and len(pi['ff_norm']) > 3:
        pid_ff[pid] = pi['ff_norm']
        pid_die[pid] = (pi.get('die_w', 500.0), pi.get('die_h', 500.0))
    else:
        pid_ff[pid] = None

print(f"{T()} Placements with FF positions: {sum(v is not None for v in pid_ff.values())}")

# Simulate for all rows
sim_feats = np.zeros((n, 10), np.float32)  # simulation feature vector per row

for i in range(n):
    pid = pids[i]
    ff_xy = pid_ff.get(pid)
    if ff_xy is None:
        continue
    die_w, die_h = pid_die[pid]
    cs = int(cs_raw[i])
    cd = cd_raw[i]
    # Convert cd to die-fraction: cd is in µm, die dimensions in µm
    cd_frac = cd / ((die_w + die_h) / 2)  # normalize by average die dimension

    n_cl, wl_w, wl_i, max_cs, mean_r = simulate_cts_run(ff_xy, cd_frac, cs, die_w, die_h)
    n_ff = len(ff_xy)
    wl_total = wl_w + wl_i

    sim_feats[i, 0] = n_cl                          # n_clusters
    sim_feats[i, 1] = n_cl / max(n_ff, 1)           # cluster fraction
    sim_feats[i, 2] = wl_w                           # within-cluster WL (µm)
    sim_feats[i, 3] = wl_i                           # inter-cluster WL (µm)
    sim_feats[i, 4] = wl_total                       # total sim WL (µm)
    sim_feats[i, 5] = wl_w / max(wl_total, 1.0)     # within fraction
    sim_feats[i, 6] = np.log1p(n_cl)                # log(n_clusters)
    sim_feats[i, 7] = np.log1p(wl_total)            # log(sim WL)
    sim_feats[i, 8] = max_cs / max(cs, 1)           # cluster fill fraction
    sim_feats[i, 9] = mean_r                         # mean cluster radius (µm)

    if (i+1) % 500 == 0: print(f"  {T()} Processed {i+1}/{n}")

print(f"{T()} Simulation complete")
print(f"  sim n_clusters range: [{sim_feats[:,0].min():.0f}, {sim_feats[:,0].max():.0f}]")
print(f"  sim WL range: [{sim_feats[:,4].min():.0f}, {sim_feats[:,4].max():.0f}] µm")

# Check correlation with actual targets
from scipy.stats import pearsonr
y_pw = Y_cache[:, 1]
y_wl = Y_cache[:, 2]
for j, name in enumerate(['n_cl','cl_frac','wl_within','wl_inter','wl_total','wl_frac','log_ncl','log_wl','cs_fill','mean_r']):
    rho_pw, _ = pearsonr(sim_feats[:,j], y_pw)
    rho_wl, _ = pearsonr(sim_feats[:,j], y_wl)
    print(f"  sim_feats[{j}] {name}: rho_power={rho_pw:.4f}, rho_wl={rho_wl:.4f}")

# ── Per-placement z-scored simulation features ────────────────────────────
# This is the key: z-score the simulation outputs WITHIN each placement
# so they directly predict the per-placement z-scored targets
sim_z = np.zeros_like(sim_feats)
sim_z_ok = np.zeros(n, dtype=bool)
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    if pid_ff.get(pid) is None: continue
    for j in range(10):
        v = sim_feats[idx, j]
        mu, sig = v.mean(), v.std()
        if sig > 1e-8:
            sim_z[idx, j] = (v - mu) / sig
            sim_z_ok[idx] = True
        else:
            sim_z[idx, j] = 0.0

print(f"\n{T()} Per-placement z-scored simulation features")
print(f"  Rows with valid sim z-scores: {sim_z_ok.sum()}")

# Correlation of z-scored sim features with z-scored targets
for j, name in enumerate(['n_cl','cl_frac','wl_within','wl_inter','wl_total','wl_frac','log_ncl','log_wl','cs_fill','mean_r']):
    mask = sim_z_ok
    rho_pw, _ = pearsonr(sim_z[mask,j], y_pw[mask])
    rho_wl, _ = pearsonr(sim_z[mask,j], y_wl[mask])
    print(f"  sim_z[{j}] {name}: rho_power={rho_pw:.4f}, rho_wl={rho_wl:.4f}")

# Global normalization for use as ML features
sf_std = sim_feats.std(0); sf_std[sf_std < 1e-9] = 1.0
sim_n = sim_feats / sf_std

# ── Combine with X29 and train ────────────────────────────────────────────
print(f"\n{T()} === POWER direct z-score MAE (baseline 0.2149) ===")
XGB_F = dict(n_estimators=500, max_depth=4, learning_rate=0.03,
             min_child_weight=10, subsample=0.8, colsample_bytree=0.8, verbosity=0)
LGB_F = dict(n_estimators=500, num_leaves=20, learning_rate=0.03,
             min_child_samples=15, verbose=-1)

lodo(X29, y_pw, "X29 LGB baseline")
lodo(np.hstack([X29, sim_n]), y_pw, "X29+sim LGB")
lodo(np.hstack([X29, sim_n]), y_pw, "X29+sim XGB_F", XGBRegressor, XGB_F)
lodo(X29T, y_pw, "X29T LGB")
lodo(np.hstack([X29T, sim_n]), y_pw, "X29T+sim LGB")
lodo(np.hstack([X29T, sim_n]), y_pw, "X29T+sim XGB_F", XGBRegressor, XGB_F)

# Also test using z-scored simulation directly as target proxy
# If z(sim_n_clusters) ≈ z(power), we can use it as a "physics baseline"
print(f"\n  Physics direct: using z(sim_n_clusters) as power predictor:")
mask = sim_z_ok
print(f"    sim_z(n_cl) vs y_pw: MAE={mean_absolute_error(y_pw[mask], sim_z[mask,0]):.4f}")
print(f"    sim_z(wl_total) vs y_pw: MAE={mean_absolute_error(y_pw[mask], sim_z[mask,4]):.4f}")
print(f"    -sim_z(log_ncl) vs y_pw: MAE={mean_absolute_error(y_pw[mask], -sim_z[mask,6]):.4f}")

print(f"\n{T()} === WIRELENGTH direct z-score MAE (baseline 0.2379) ===")
lodo(X29, y_wl, "X29 LGB baseline")
lodo(np.hstack([X29, sim_n]), y_wl, "X29+sim LGB")
lodo(np.hstack([X29, sim_n]), y_wl, "X29+sim XGB_F", XGBRegressor, XGB_F)
lodo(X29T, y_wl, "X29T LGB")
lodo(np.hstack([X29T, sim_n]), y_wl, "X29T+sim LGB")
lodo(np.hstack([X29T, sim_n]), y_wl, "X29T+sim XGB_F", XGBRegressor, XGB_F)

print(f"\n  Physics direct: using z(sim_wl) as WL predictor:")
mask = sim_z_ok
print(f"    sim_z(wl_total) vs y_wl: MAE={mean_absolute_error(y_wl[mask], sim_z[mask,4]):.4f}")
print(f"    sim_z(log_wl) vs y_wl: MAE={mean_absolute_error(y_wl[mask], sim_z[mask,7]):.4f}")
print(f"    sim_z(wl_inter) vs y_wl: MAE={mean_absolute_error(y_wl[mask], sim_z[mask,3]):.4f}")

print(f"\n{T()} DONE")
