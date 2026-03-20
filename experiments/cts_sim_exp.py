"""
cts_sim_exp.py — Physics simulation of CTS clustering using actual FF positions

Key hypothesis: By simulating the actual CTS clustering process using real FF positions
from DEF files, we can compute better per-run features (n_clusters_sim, centroid_hpwl_sim)
that account for spatial non-uniformity, outperforming the analytical uniform-density formula.

Novel features:
  z_n_clusters_sim   — simulated cluster count (kdtree density approximation)
  z_centroid_hpwl    — HPWL of cluster centroids (inter-cluster routing proxy for WL)
  z_intra_routing    — total intra-cluster routing length estimate
  z_cluster_entropy  — spatial entropy of cluster distribution

LODO evaluation, LGB baseline + X_best + sim features.
"""

import re, os, time, pickle, warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
DF_DIR = f'{BASE}/dataset_with_def/placement_files'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("CTS Simulation via Actual FF Positions")
print("=" * 70)

# ── Load base data ─────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)

pids = df['placement_id'].values
designs = df['design_name'].values
n = len(pids)
y_pw = Y_cache[:, 1]; y_wl = Y_cache[:, 2]

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def z_within(v, pids):
    out = np.zeros(len(v), np.float32)
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        vv = v[idx].astype(float)
        out[idx] = (vv - vv.mean()) / max(vv.std(), 1e-8)
    return out

# ── Build X_best (from previous session) ──────────────────────────────────
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

# Clock distance features (from clock_source_exp.py)
cd_arr = Xraw[:, 3]  # cluster_dia per run
cd_z_pp = z_within(cd_arr, pids)
z_log_cd = z_within(np.log(cd_arr), pids)
z_inv_cd = z_within(1.0/cd_arr, pids)
z_cd2 = z_within(cd_arr**2, pids)

# Parse clock source location from DEF
def get_clock_source(def_path):
    with open(def_path) as f:
        content = f.read()
    for pat in [r'USE CLOCK.*?PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)',
                r'- clk\s+\+\s+NET clk.*?PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)',
                r'- clk\b.*?PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)']:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            return float(m.group(1))/1000, float(m.group(2))/1000
    return None, None

# ── Parse FF positions from DEF files ────────────────────────────────────
print(f"{T()} Parsing FF positions from DEF files...")
SIM_CACHE_FILE = f'{BASE}/sim_ff_cache.pkl'

if os.path.exists(SIM_CACHE_FILE):
    print(f"  Loading cached FF positions...")
    with open(SIM_CACHE_FILE, 'rb') as f:
        ff_cache = pickle.load(f)
else:
    ff_cache = {}
    unique_pids = df[['placement_id', 'design_name']].drop_duplicates().values
    for i, (pid, design) in enumerate(unique_pids):
        def_path = f'{DF_DIR}/{pid}/{design}.def'
        if not os.path.exists(def_path):
            continue
        with open(def_path) as f:
            content = f.read()
        # Die area
        ma = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s+\(\s*(\d+)\s+(\d+)\s*\)', content)
        if not ma:
            continue
        x1,y1,x2,y2 = int(ma.group(1)),int(ma.group(2)),int(ma.group(3)),int(ma.group(4))
        die_w = (x2-x1)/1000; die_h = (y2-y1)/1000
        # FF positions (all DFF cell types)
        ff_pat = re.compile(r'sky130_fd_sc_hd__df\S*\s+\+.*?PLACED\s+\(\s*(\d+)\s+(\d+)\s*\)', re.DOTALL)
        xys = [(int(m.group(1))/1000, int(m.group(2))/1000) for m in ff_pat.finditer(content)]
        xy = np.array(xys, dtype=np.float32) if xys else np.zeros((0,2), np.float32)
        # Clock source
        cx, cy = get_clock_source(def_path)
        ff_cache[pid] = {'xy': xy, 'die_w': die_w, 'die_h': die_h,
                         'cx': cx, 'cy': cy}
        if (i+1) % 50 == 0:
            print(f"  Parsed {i+1}/{len(unique_pids)} placements... {T()}")
    with open(SIM_CACHE_FILE, 'wb') as f:
        pickle.dump(ff_cache, f)
    print(f"  Cached {len(ff_cache)} placements")

print(f"{T()} Done parsing. Computing simulation features...")

# ── Compute simulation features per run ──────────────────────────────────
# Raw arrays (will be z-scored within placement later)
sim_n_clusters = np.zeros(n, np.float32)
sim_centroid_hpwl = np.zeros(n, np.float32)
sim_intra_routing = np.zeros(n, np.float32)
sim_mean_dist_clk = np.zeros(n, np.float32)
sim_max_dist_clk = np.zeros(n, np.float32)
sim_p90_dist_clk = np.zeros(n, np.float32)

for i, pid in enumerate(pids):
    if pid not in ff_cache:
        continue
    data = ff_cache[pid]
    xy = data['xy']
    die_w = data['die_w']
    die_h = data['die_h']
    cx = data['cx']; cy = data['cy']

    if len(xy) == 0:
        continue

    cd_val = cd[i]  # cluster_dia in µm for this run

    # kdtree density-based n_clusters approximation
    # Build tree (will be rebuilt per run - cache not needed since fast)
    tree = cKDTree(xy)
    counts = tree.query_ball_point(xy, r=cd_val/2, return_length=True)
    counts = np.maximum(counts, 1)
    sim_n_clusters[i] = np.sum(1.0 / counts)

    # Grid-based cluster assignments for routing estimation
    # Each grid cell of size cd_val is a cluster
    gx = (xy[:, 0] // cd_val).astype(int)
    gy = (xy[:, 1] // cd_val).astype(int)
    cells = {}
    for k, (gxi, gyi) in enumerate(zip(gx, gy)):
        key = (gxi, gyi)
        if key not in cells:
            cells[key] = []
        cells[key].append(k)

    # Cluster centroids
    centroids = np.array([xy[idxs].mean(0) for idxs in cells.values()])

    # HPWL of cluster centroids (inter-cluster routing proxy)
    if len(centroids) > 1:
        hpwl = (centroids[:, 0].max() - centroids[:, 0].min() +
                centroids[:, 1].max() - centroids[:, 1].min())
    else:
        hpwl = 0.0
    sim_centroid_hpwl[i] = hpwl

    # Intra-cluster routing: sum of cluster diameters (each cluster spans ≤ cd_val)
    intra = 0.0
    for idxs in cells.values():
        cxy = xy[idxs]
        if len(cxy) > 1:
            intra += (cxy[:, 0].max() - cxy[:, 0].min() +
                      cxy[:, 1].max() - cxy[:, 1].min())
    sim_intra_routing[i] = intra

    # Clock distance features
    if cx is not None:
        dists = np.abs(xy[:, 0] - cx) + np.abs(xy[:, 1] - cy)  # Manhattan
        sim_mean_dist_clk[i] = dists.mean()
        sim_max_dist_clk[i] = dists.max()
        sim_p90_dist_clk[i] = np.percentile(dists, 90)

print(f"{T()} Simulation features computed.")

# Z-score all simulation features within placement
z_n_clust = z_within(sim_n_clusters, pids)
z_cent_hpwl = z_within(sim_centroid_hpwl, pids)
z_intra = z_within(sim_intra_routing, pids)
z_mdist = z_within(sim_mean_dist_clk, pids)
z_maxdist = z_within(sim_max_dist_clk, pids)
z_p90dist = z_within(sim_p90_dist_clk, pids)

# Key ratio features (the "clock distance / cluster_dia" idea from previous session)
raw_mdist = sim_mean_dist_clk; raw_mdist[raw_mdist==0] = 1.0
z_dist_cd = z_within(sim_mean_dist_clk / (cd_arr + 1e-6), pids)
z_mw_cd = z_within(mw / (cd_arr + 1e-6), pids)
z_bd_cd = z_within(bd / (cd_arr + 1e-6), pids)

# Correlation check
print(f"\nCorrelation with z_power:")
feats = [('z_n_clust', z_n_clust), ('z_cent_hpwl', z_cent_hpwl),
         ('z_intra', z_intra), ('z_mdist', z_mdist), ('z_maxdist', z_maxdist),
         ('z_p90dist', z_p90dist), ('z_dist_cd', z_dist_cd),
         ('z_mw_cd', z_mw_cd), ('z_bd_cd', z_bd_cd),
         ('z_log_cd', z_log_cd), ('z_inv_cd', z_inv_cd), ('z_cd2', z_cd2)]
for name, feat in feats:
    mask = np.isfinite(feat) & np.isfinite(y_pw)
    rho = np.corrcoef(feat[mask], y_pw[mask])[0, 1]
    rho_wl = np.corrcoef(feat[mask], y_wl[mask])[0, 1]
    print(f"  {name:20s}: power rho={rho:+.4f}, wl rho={rho_wl:+.4f}")

# ── Build feature matrices ─────────────────────────────────────────────────
# Current best from previous session: X29T + z_dist_cd + z_mw_cd + z_bd_cd + log/inv/sq cd
X_best = np.hstack([X29T, z_dist_cd.reshape(-1,1), z_mw_cd.reshape(-1,1),
                    z_bd_cd.reshape(-1,1), z_log_cd.reshape(-1,1),
                    z_inv_cd.reshape(-1,1), z_cd2.reshape(-1,1)])

X_with_sim = np.hstack([X_best, z_n_clust.reshape(-1,1), z_cent_hpwl.reshape(-1,1),
                         z_intra.reshape(-1,1)])

X_with_clk = np.hstack([X_best, z_n_clust.reshape(-1,1), z_cent_hpwl.reshape(-1,1),
                          z_intra.reshape(-1,1), z_mdist.reshape(-1,1),
                          z_maxdist.reshape(-1,1), z_p90dist.reshape(-1,1)])

X_sim_dist_cd = np.hstack([X_best, z_n_clust.reshape(-1,1), z_cent_hpwl.reshape(-1,1),
                             z_intra.reshape(-1,1),
                             z_within(sim_mean_dist_clk / (cd_arr + 1e-6), pids).reshape(-1,1),
                             z_within(sim_max_dist_clk / (cd_arr + 1e-6), pids).reshape(-1,1)])

# Clean NaN/inf
for arr in [X_best, X_with_sim, X_with_clk, X_sim_dist_cd]:
    for col in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:, col])
        if bad.any(): arr[bad, col] = 0.0

# ── LODO evaluation ────────────────────────────────────────────────────────
def lodo_lgb(X, y, label, n_leaves=20, lr=0.03, n_est=300):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = LGBMRegressor(n_estimators=n_est, num_leaves=n_leaves, learning_rate=lr,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    tag = ' ✓' if np.mean(maes) < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}{tag}")
    return maes

print(f"\n{T()} === POWER MAE ===")
lodo_lgb(X_best, y_pw, "X_best (baseline 0.2036)")
lodo_lgb(X_with_sim, y_pw, "X_best + sim(n_clust,hpwl,intra)")
lodo_lgb(X_with_clk, y_pw, "X_best + sim + clk_dists")
lodo_lgb(X_sim_dist_cd, y_pw, "X_best + sim + dist/cd features")

# Try n_clust alone + X_best
X_nc = np.hstack([X_best, z_n_clust.reshape(-1,1)])
lodo_lgb(X_nc, y_pw, "X_best + z_n_clust only")
X_ch = np.hstack([X_best, z_cent_hpwl.reshape(-1,1)])
lodo_lgb(X_ch, y_pw, "X_best + z_cent_hpwl only")

# Hyperparameter tuning on best
print(f"\n{T()} === POWER: Hyperparameter tuning ===")
lodo_lgb(X_with_sim, y_pw, "X_sim nl=31, lr=0.02", n_leaves=31, lr=0.02, n_est=400)
lodo_lgb(X_with_sim, y_pw, "X_sim nl=15, lr=0.04", n_leaves=15, lr=0.04, n_est=400)
lodo_lgb(X_with_sim, y_pw, "X_sim nl=20, lr=0.03, 500est", n_leaves=20, lr=0.03, n_est=500)

print(f"\n{T()} === WIRELENGTH MAE ===")
lodo_lgb(X_best, y_wl, "X_best WL (baseline ~0.2336)")
lodo_lgb(X_with_sim, y_wl, "X_best + sim WL")
lodo_lgb(X_with_clk, y_wl, "X_best + sim + clk_dists WL")

# WL: centroid_hpwl is key physics feature
X_wl_sim = np.hstack([X_best, z_cent_hpwl.reshape(-1,1), z_intra.reshape(-1,1)])
lodo_lgb(X_wl_sim, y_wl, "X_best + cent_hpwl + intra WL")

# Also try total_routing = centroid_hpwl + intra_routing
total_routing = sim_centroid_hpwl + sim_intra_routing
z_total_routing = z_within(total_routing, pids)
X_total = np.hstack([X_best, z_total_routing.reshape(-1,1)])
lodo_lgb(X_total, y_wl, "X_best + z_total_routing WL")

# WL: try scaling intra by cluster_dia
routing_per_cd = sim_intra_routing / (cd_arr + 1e-6)
z_routing_cd = z_within(routing_per_cd, pids)
X_rcd = np.hstack([X_best, z_total_routing.reshape(-1,1), z_routing_cd.reshape(-1,1)])
lodo_lgb(X_rcd, y_wl, "X_best + total_routing + routing/cd WL")

print(f"\n{T()} DONE")
