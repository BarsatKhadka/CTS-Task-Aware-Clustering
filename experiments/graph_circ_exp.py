"""
graph_circ_exp.py - Circuit-physics features from processed_graphs/*.pt

Key insight: processed_graphs have raw_areas (actual cell areas, not z-scored),
is_sequential/is_buffer flags, and per-cell toggle statistics. These encode
the ACTUAL physical properties of the circuit that drive power/WL.

Physics:
  Power ∝ Σ(α_i × C_i) where C_i ∝ cell_area, α_i = toggle rate
  WL    ∝ Steiner_tree(FF_positions) × CTS_efficiency(cluster_dia, cluster_size)

ICLR novelty: Circuit-identity features that capture placement-level physics
(cell areas, FF count, spatial HPWL) + CTS knob interactions → zero-shot
generalization to unseen designs.

For the 140 placements with graph files (out of 539 total):
  aes: ~31 placements, ethmac: ~47, picorv32: ~31, sha256: ~31
"""

import os, pickle, time, warnings
import numpy as np
import torch
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
GRAPH_DIR = f'{BASE}/processed_graphs'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Circuit-Physics Graph Features Experiment")
print("=" * 70)

# ── Load base cache ──────────────────────────────────────────────────────
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
    tag = ' ✓' if mean_mae < 0.05 else (' ~' if mean_mae < 0.07 else '')
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    return mean_mae

# ── Build X29 base ────────────────────────────────────────────────────────
def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

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
util = Xplc[:, 0] / 100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp, Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])

# Tight path features
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i, :20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std < 1e-9] = 1.0
X29T = np.hstack([X29, X_tight / tp_std])  # 49-dim

# ── Load graph files → circuit-physics features ───────────────────────────
print(f"{T()} Loading graph files...")

graph_feats = {}  # pid → 18-dim circuit feature vector

for fname in os.listdir(GRAPH_DIR):
    if not fname.endswith('.pt'): continue
    pid = fname.replace('.pt', '')
    try:
        d = torch.load(f'{GRAPH_DIR}/{fname}', weights_only=False)
        X_g = d['X'].numpy() if hasattr(d['X'], 'numpy') else d['X']  # [N, 18]
        ra = d['raw_areas'].numpy() if hasattr(d['raw_areas'], 'numpy') else d['raw_areas']  # [N]

        is_seq = X_g[:, 10] > 0.5   # is_sequential (binary, not z-scored)
        is_buf = X_g[:, 11] > 0.5   # is_buffer
        n_total = len(X_g)
        n_ff = is_seq.sum()
        n_buf = is_buf.sum()

        # Raw area statistics (NOT z-scored — actual cell areas)
        ff_areas = ra[is_seq] if n_ff > 0 else np.array([0.0])
        ff_area_total = ff_areas.sum()
        ff_area_mean = ff_areas.mean()
        ff_area_std = ff_areas.std() if n_ff > 1 else 0.0
        area_total = ra.sum()

        # Toggle distribution shape (z-scored WITHIN this graph — cols 12, 13)
        # So mean=0, std=1 by construction. But distribution SHAPE varies:
        toggle_raw = X_g[:, 12]   # z-scored toggle count
        sum_tog_raw = X_g[:, 13]  # z-scored sum_toggle
        # Distribution shape features
        toggle_skew = float(np.mean(toggle_raw**3))  # skewness proxy
        toggle_kurt = float(np.mean(toggle_raw**4))  # kurtosis proxy
        toggle_p90 = float(np.percentile(toggle_raw, 90))
        toggle_p99 = float(np.percentile(toggle_raw, 99))

        # Capacitance distribution (col 8 = total_pin_cap * 1000, z-scored)
        cap_raw = X_g[:, 8]  # z-scored
        cap_p90 = float(np.percentile(cap_raw, 90))
        cap_skew = float(np.mean(cap_raw**3))

        # Fan-out distribution (col 17 = log1p(fan_out), z-scored)
        fanout_p90 = float(np.percentile(X_g[:, 17], 90))

        # FF spatial features from positions (cols 0, 1 are z-scored within graph)
        xy = X_g[is_seq, :2] if n_ff > 3 else None  # z-scored positions
        if xy is not None:
            hpwl_norm = (xy[:, 0].max() - xy[:, 0].min() +
                         xy[:, 1].max() - xy[:, 1].min())  # in z-score units
            xy_std = np.sqrt(xy[:, 0].var() + xy[:, 1].var())
            xy_p90_span = (np.percentile(xy[:, 0], 90) - np.percentile(xy[:, 0], 10) +
                           np.percentile(xy[:, 1], 90) - np.percentile(xy[:, 1], 10))
        else:
            hpwl_norm = xy_std = xy_p90_span = 0.0

        # Circuit composition ratios
        ff_ratio = n_ff / max(n_total, 1)
        buf_ratio = n_buf / max(n_total, 1)
        ff_area_frac = ff_area_total / max(area_total, 1e-8)

        graph_feats[pid] = np.array([
            np.log1p(n_ff),           # 0: log(n_ff)
            np.log1p(n_buf),          # 1: log(n_buf)
            np.log1p(n_total),        # 2: log(n_total)
            ff_ratio,                 # 3: FF fraction
            buf_ratio,                # 4: buffer fraction
            np.log1p(ff_area_total),  # 5: log(total FF area)
            ff_area_mean,             # 6: mean FF area
            ff_area_std / (ff_area_mean + 1e-8),  # 7: FF area CV
            ff_area_frac,             # 8: FF area fraction
            np.log1p(area_total),     # 9: log(total cell area)
            toggle_skew,              # 10: toggle skewness
            toggle_kurt,              # 11: toggle kurtosis
            toggle_p90,               # 12: toggle p90
            toggle_p99,               # 13: toggle p99
            cap_p90,                  # 14: cap p90
            cap_skew,                 # 15: cap skewness
            hpwl_norm,                # 16: HPWL of FFs (z-score units)
            xy_p90_span,              # 17: p90 span of FF positions
        ], dtype=np.float32)
    except Exception as e:
        print(f"  Warning: {fname}: {e}")
        continue

print(f"{T()} Loaded {len(graph_feats)} graph files")

# ── Map graph features to data rows ─────────────────────────────────────
X_gf = np.zeros((n, 18), np.float32)
gf_mask = np.zeros(n, dtype=bool)  # which rows have graph data
for i, pid in enumerate(pids):
    if pid in graph_feats:
        X_gf[i] = graph_feats[pid]
        gf_mask[i] = True

n_with_graph = gf_mask.sum()
print(f"{T()} {n_with_graph}/{n} rows have graph features")
print(f"  Coverage by design:")
for d in ['aes', 'ethmac', 'picorv32', 'sha256']:
    dm = designs == d
    print(f"    {d}: {(gf_mask & dm).sum()}/{dm.sum()}")

# ── ff_positions features (all 539 placements) ───────────────────────────
print(f"\n{T()} Building ff_positions features...")
ff_geo = {}  # pid → [die_area, hpwl_um, ff_density, centroid_offset, n_ff_pos]
for pid in np.unique(pids):
    pi = ff_pos.get(pid)
    if pi is None or pi.get('ff_norm') is None:
        ff_geo[pid] = np.zeros(8, np.float32); continue
    xy = pi['ff_norm']  # [N_ff, 2] in [0,1]
    dw = pi.get('die_w', 1.0); dh = pi.get('die_h', 1.0)
    die_area = dw * dh
    hpwl_um = ((xy[:,0].max() - xy[:,0].min()) * dw +
                (xy[:,1].max() - xy[:,1].min()) * dh)
    n_ff_p = len(xy)
    density = n_ff_p / max(die_area, 1.0)
    cx, cy = xy.mean(0)
    cent_offset = np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2)
    p90_span = ((np.percentile(xy[:,0], 90) - np.percentile(xy[:,0], 10)) * dw +
                (np.percentile(xy[:,1], 90) - np.percentile(xy[:,1], 10)) * dh)
    # Quadrant imbalance (asymmetry measure)
    q1 = ((xy[:,0] < 0.5) & (xy[:,1] < 0.5)).mean()
    q2 = ((xy[:,0] >= 0.5) & (xy[:,1] < 0.5)).mean()
    q3 = ((xy[:,0] < 0.5) & (xy[:,1] >= 0.5)).mean()
    q4 = ((xy[:,0] >= 0.5) & (xy[:,1] >= 0.5)).mean()
    quad_asym = np.std([q1, q2, q3, q4])
    ff_geo[pid] = np.array([
        np.log1p(die_area),          # 0: log(die_area) µm²
        np.log1p(hpwl_um),           # 1: log(HPWL in µm)
        np.log1p(n_ff_p),            # 2: log(n_ff)
        density,                     # 3: FF density per µm²
        cent_offset,                 # 4: centroid offset from center
        np.log1p(p90_span),          # 5: log(p90 span in µm)
        quad_asym,                   # 6: quadrant asymmetry
        np.log(dw / max(dh, 1e-3)),  # 7: log(aspect ratio)
    ], np.float32)

X_geo = np.array([ff_geo.get(pid, np.zeros(8)) for pid in pids], np.float32)
geo_std = X_geo.std(0); geo_std[geo_std < 1e-9] = 1.0
X_geo_n = X_geo / geo_std

# ── Knob × Circuit interactions ─────────────────────────────────────────
# Physics: power ∝ n_ff × toggle_activity / cluster_size
# WL ∝ hpwl_um × cluster_dia_efficiency
hpwl_um = np.exp(X_geo[:, 1]) - 1  # recover HPWL in µm
log_nff = X_geo[:, 2]  # log(n_ff)
log_die = X_geo[:, 0]  # log(die_area)

# CTS knob × geometry interactions
Xknob_geo = np.column_stack([
    # WL physics: cluster_dia / die_scale, max_wire / HPWL
    cd / (hpwl_um / 1000 + 1e-4),          # cluster_dia / HPWL (coverage ratio)
    mw / (hpwl_um / 1000 + 1e-4),          # max_wire / HPWL
    cs / (log_nff + 1e-4),                  # cluster_size / log(n_ff)
    np.log1p(hpwl_um / (cd + 1e-4)),        # log(HPWL / cluster_dia)
    np.log1p(hpwl_um / (mw + 1e-4)),        # log(HPWL / max_wire)
    # Power physics: n_ff / cluster_size (buffer count proxy)
    np.exp(log_nff) / (cs + 1),             # n_ff / cluster_size
    cd * np.exp(log_nff) / 1e6,             # cluster_dia × n_ff (tree size proxy)
    log_die + np.log(cd + 1),               # log(die_area × cluster_dia)
])
for c in range(Xknob_geo.shape[1]):
    bad = ~np.isfinite(Xknob_geo[:, c])
    if bad.any(): Xknob_geo[bad, c] = np.nanmedian(Xknob_geo[~bad, c])
    Xknob_geo[:, c] = np.clip(Xknob_geo[:, c], -1e6, 1e6)
# Per-placement rank these interaction features
Xkg_rank = np.zeros_like(Xknob_geo)
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    for j in range(Xknob_geo.shape[1]):
        v = Xknob_geo[idx, j]
        if v.max() > v.min(): Xkg_rank[idx, j] = rank_within(v)
        else: Xkg_rank[idx, j] = 0.5

Xknob_geo_std = Xknob_geo.std(0); Xknob_geo_std[Xknob_geo_std < 1e-9] = 1.0
Xknob_geo_n = np.hstack([Xknob_geo / Xknob_geo_std, Xkg_rank])  # 16-dim

# ── Graph features: z-score globally ─────────────────────────────────────
gf_std = X_gf.std(0); gf_std[gf_std < 1e-9] = 1.0
X_gf_n = X_gf / gf_std

# ── Graph × knob interactions (only for rows with graph data) ─────────────
log_n_ff_gf = X_gf[:, 0]  # log(n_ff) from graph
log_area_gf = X_gf[:, 9]  # log(total area) from graph
Xgk = np.column_stack([
    np.exp(log_n_ff_gf) / (cs + 1),         # n_ff_gf / cluster_size
    cd * np.exp(log_n_ff_gf) / 1e6,          # cluster_dia × n_ff_gf
    log_area_gf * np.log1p(cd),              # log(area) × log(cluster_dia)
    log_area_gf * np.log1p(cs),              # log(area) × log(cluster_size)
    X_gf[:, 7] * Xrank[:, 3],               # FF area CV × rank(cluster_dia)
    X_gf[:, 10] * Xrank[:, 2],              # toggle skew × rank(cluster_size)
])
for c in range(Xgk.shape[1]):
    bad = ~np.isfinite(Xgk[:, c])
    if bad.any(): Xgk[bad, c] = np.nanmedian(Xgk[~bad, c])
    Xgk[:, c] = np.clip(Xgk[:, c], -1e6, 1e6)
Xgk_std = Xgk.std(0); Xgk_std[Xgk_std < 1e-9] = 1.0
Xgk_n = Xgk / Xgk_std

# ── Clean NaN ────────────────────────────────────────────────────────────
for arr in [X29, X29T, X_gf_n, X_geo_n, Xknob_geo_n, Xgk_n]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:, c])
        if bad.any(): arr[bad, c] = 0.0

print(f"{T()} Feature dims: X29={X29.shape[1]}, X29T={X29T.shape[1]}, "
      f"geo={X_geo_n.shape[1]}, knob_geo={Xknob_geo_n.shape[1]}, "
      f"gf={X_gf_n.shape[1]}, gk={Xgk_n.shape[1]}")

# ── Targets ──────────────────────────────────────────────────────────────
y_pw = Y_cache[:, 1]
y_wl = Y_cache[:, 2]
y_sk = Y_cache[:, 0]

XGB_F = dict(n_estimators=500, max_depth=4, learning_rate=0.03,
             min_child_weight=10, subsample=0.8, colsample_bytree=0.8, verbosity=0)
LGB_F = dict(n_estimators=500, num_leaves=20, learning_rate=0.03,
             min_child_samples=15, verbose=-1)

print(f"\n{T()} === POWER: per-placement z-score MAE (current best ~0.0662) ===")
lodo(X29, y_pw, "X29 LGB baseline")
lodo(np.hstack([X29, X_geo_n]), y_pw, "X29+geo LGB")
lodo(np.hstack([X29, Xknob_geo_n]), y_pw, "X29+kgeo LGB")
lodo(np.hstack([X29T, X_geo_n, Xknob_geo_n]), y_pw, "X29T+geo+kgeo LGB")
lodo(np.hstack([X29T, X_geo_n, Xknob_geo_n]), y_pw, "X29T+geo+kgeo XGB_F",
     XGBRegressor, XGB_F)
lodo(np.hstack([X29, X_gf_n]), y_pw, "X29+gf LGB")
lodo(np.hstack([X29, X_gf_n, Xgk_n]), y_pw, "X29+gf+gk LGB")
lodo(np.hstack([X29T, X_geo_n, X_gf_n, Xgk_n, Xknob_geo_n]), y_pw, "X29T+ALL LGB")
lodo(np.hstack([X29T, X_geo_n, X_gf_n, Xgk_n, Xknob_geo_n]), y_pw, "X29T+ALL XGB_F",
     XGBRegressor, XGB_F)

print(f"\n{T()} === WIRELENGTH: per-placement z-score MAE (current best ~0.0837) ===")
lodo(X29, y_wl, "X29 LGB baseline")
lodo(np.hstack([X29, X_geo_n]), y_wl, "X29+geo LGB")
lodo(np.hstack([X29, Xknob_geo_n]), y_wl, "X29+kgeo LGB")
lodo(np.hstack([X29T, X_geo_n, Xknob_geo_n]), y_wl, "X29T+geo+kgeo LGB")
lodo(np.hstack([X29T, X_geo_n, Xknob_geo_n]), y_wl, "X29T+geo+kgeo XGB_F",
     XGBRegressor, XGB_F)
lodo(np.hstack([X29, X_gf_n]), y_wl, "X29+gf LGB")
lodo(np.hstack([X29, X_gf_n, Xgk_n]), y_wl, "X29+gf+gk LGB")
lodo(np.hstack([X29T, X_geo_n, X_gf_n, Xgk_n, Xknob_geo_n]), y_wl, "X29T+ALL LGB")
lodo(np.hstack([X29T, X_geo_n, X_gf_n, Xgk_n, Xknob_geo_n]), y_wl, "X29T+ALL XGB_F",
     XGBRegressor, XGB_F)

print(f"\n{T()} === SKEW: rank target XGB (current best ~0.2527) ===")
Y_rank = np.zeros((n, 3), np.float32)
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    for j in range(3): Y_rank[idx, j] = rank_within(Y_cache[idx, j])

from sklearn.metrics import mean_absolute_error as mae_fn
def lodo_rank(X, yr, label, cls=XGBRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                  min_child_weight=15, subsample=0.8, colsample_bytree=0.8, verbosity=0)
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held; sc = StandardScaler()
        m = cls(**kw); m.fit(sc.fit_transform(X[tr]), yr[tr])
        pred = m.predict(sc.transform(X[te])); maes.append(mae_fn(yr[te], pred))
    mean_mae = np.mean(maes)
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}")
    return mean_mae

XGB_SK = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
              min_child_weight=15, subsample=0.8, colsample_bytree=0.8, verbosity=0)
lodo_rank(X29T, Y_rank[:, 0], "X29T rank XGB_SK baseline")
lodo_rank(np.hstack([X29T, X_geo_n]), Y_rank[:, 0], "X29T+geo rank XGB_SK")
lodo_rank(np.hstack([X29T, X_gf_n]), Y_rank[:, 0], "X29T+gf rank XGB_SK")
lodo_rank(np.hstack([X29T, X_geo_n, Xknob_geo_n]), Y_rank[:, 0], "X29T+geo+kgeo rank XGB_SK")

print(f"\n{T()} DONE")
