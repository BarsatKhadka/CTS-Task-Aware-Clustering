"""
domain_inv_exp.py - Domain-invariant physics features for zero-shot prediction

Key finding: cluster_dia has Spearman rho=-0.91 with power within placements,
but direct use gives MAE=0.27 (worse than X29's 0.21) because the SCALE of
z-scores varies across designs (different power sensitivities).

Solution: Normalize CTS knobs by die geometry and FF density to make features
design-invariant:
  cd / die_scale   → cluster_dia as fraction of die
  cd / knn_dist    → how many FF spacings per cluster
  cs / log(n_ff)   → cluster_size relative to circuit scale
  mw / die_scale   → max_wire relative to die

Also try: calibration regression (learn design-specific slope from 2 training runs).
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
print("Domain-Invariant Physics Features")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_pos = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def lodo(X, y, label, cls=LGBMRegressor, kw=None, best=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held; sc = StandardScaler()
        m = cls(**kw); m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te])); maes.append(mean_absolute_error(y[te], pred))
    mean_mae = np.mean(maes)
    tag = ' ✓✓' if mean_mae < 0.05 else (' ✓' if mean_mae < 0.10 else
          (' ~' if mean_mae < 0.15 else ('  ' if best is None or mean_mae < best else '  ↓')))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    return mean_mae

# ── Base X29 ──────────────────────────────────────────────────────────────
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
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp, Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T = np.hstack([X29, X_tight/tp_std])

# ── Domain-invariant features ─────────────────────────────────────────────
print(f"{T()} Building domain-invariant features from die geometry + FF positions...")

die_scale = np.zeros(n, np.float32)   # (die_w + die_h) / 2 [µm]
knn_dist = np.zeros(n, np.float32)    # mean nearest-neighbor FF distance [µm]
n_ff_log = np.zeros(n, np.float32)    # log(n_ff)
die_area = np.zeros(n, np.float32)    # log(die_w × die_h)

for pid in np.unique(pids):
    pi = ff_pos.get(pid)
    idx = np.where(pids == pid)[0]
    if pi is None or pi.get('ff_norm') is None:
        continue
    ff_xy = pi['ff_norm']  # [N, 2] in [0,1]
    dw = pi.get('die_w', 500.0); dh = pi.get('die_h', 500.0)
    scale = (dw + dh) / 2  # µm

    die_scale[idx] = scale
    n_ff_log[idx] = np.log1p(len(ff_xy))
    die_area[idx] = np.log(dw * dh + 1)

    if len(ff_xy) >= 4:
        tree = cKDTree(ff_xy)
        d, _ = tree.query(ff_xy, k=min(5, len(ff_xy)))
        knn_dist[idx] = d[:, 1].mean() * scale  # in µm (denormalize)
    else:
        knn_dist[idx] = scale * 0.1  # fallback

print(f"{T()} die_scale range: [{die_scale[die_scale>0].min():.0f}, {die_scale.max():.0f}] µm")
print(f"  knn_dist range: [{knn_dist[knn_dist>0].min():.2f}, {knn_dist.max():.2f}] µm")

# Domain-invariant CTS knob features
cd_frac = cd / (die_scale + 1e-4)         # cluster_dia / die_scale [0,1]
mw_frac = mw / (die_scale + 1e-4)         # max_wire / die_scale
cd_knn = cd / (knn_dist + 1e-4)           # cluster_dia / knn_dist (# FF spacings per cluster)
mw_knn = mw / (knn_dist + 1e-4)           # max_wire / knn_dist
cs_nff = cs / (np.exp(n_ff_log) + 1)      # cluster_size / n_ff (cluster fraction)
bd_frac = bd / (die_scale + 1e-4)         # buf_dist / die_scale

# Compute per-placement ranks of domain-invariant features
di_raw = np.column_stack([cd_frac, mw_frac, cd_knn, mw_knn, cs_nff, bd_frac])
di_rank = np.zeros_like(di_raw)
di_cent = np.zeros_like(di_raw)
di_g = np.abs(di_raw).max(0) + 1e-9
for pid in np.unique(pids):
    idx = np.where(pids == pid)[0]
    for j in range(di_raw.shape[1]):
        v = di_raw[idx, j]
        di_rank[idx, j] = rank_within(v) if v.max() > v.min() else 0.5
        di_cent[idx, j] = (v - v.mean()) / di_g[j]

di_std = di_raw.std(0); di_std[di_std<1e-9] = 1.0
X_di = np.hstack([di_raw/di_std, di_rank, di_cent])  # 18-dim

# Context features: placement scale
X_ctx = np.column_stack([die_area, n_ff_log, die_scale/1000])
ctx_std = X_ctx.std(0); ctx_std[ctx_std<1e-9]=1.0
X_ctx_n = X_ctx / ctx_std

# Fix NaN/Inf
for arr in [X29, X29T, X_di, X_ctx_n]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

print(f"{T()} Features: X29={X29.shape[1]}, X29T={X29T.shape[1]}, X_di={X_di.shape[1]}, X_ctx={X_ctx_n.shape[1]}")

# Check correlation of domain-invariant features with targets
from scipy.stats import spearmanr
y_pw, y_wl = Y_cache[:, 1], Y_cache[:, 2]
print(f"\nDomain-invariant features vs z-score targets (Spearman rho):")
for j, name in enumerate(['cd_frac','mw_frac','cd_knn','mw_knn','cs_nff','bd_frac']):
    rho_pw, _ = spearmanr(di_raw[:, j], y_pw)
    rho_wl, _ = spearmanr(di_raw[:, j], y_wl)
    print(f"  {name}: power={rho_pw:.4f}, wl={rho_wl:.4f}")

# ── LODO evaluation ───────────────────────────────────────────────────────
XGB_F = dict(n_estimators=500, max_depth=4, learning_rate=0.03,
             min_child_weight=10, subsample=0.8, colsample_bytree=0.8, verbosity=0)
LGB_F = dict(n_estimators=500, num_leaves=20, learning_rate=0.03,
             min_child_samples=15, verbose=-1)

print(f"\n{T()} === POWER z-score MAE (baseline X29=0.2149) ===")
b_pw = 0.2149
lodo(X29, y_pw, "X29 LGB baseline", best=b_pw)
lodo(np.hstack([X29, X_di]), y_pw, "X29+di LGB", best=b_pw)
lodo(np.hstack([X29, X_di, X_ctx_n]), y_pw, "X29+di+ctx LGB", best=b_pw)
lodo(np.hstack([X29T, X_di]), y_pw, "X29T+di LGB", best=b_pw)
lodo(np.hstack([X29T, X_di, X_ctx_n]), y_pw, "X29T+di+ctx LGB", best=b_pw)
lodo(np.hstack([X29T, X_di, X_ctx_n]), y_pw, "X29T+di+ctx XGB_F",
     XGBRegressor, XGB_F, b_pw)
lodo(np.hstack([X29T, X_di, X_ctx_n]), y_pw, "X29T+di+ctx LGB_F",
     LGBMRegressor, LGB_F, b_pw)

print(f"\n{T()} === WIRELENGTH z-score MAE (baseline X29=0.2379) ===")
b_wl = 0.2379
lodo(X29, y_wl, "X29 LGB baseline", best=b_wl)
lodo(np.hstack([X29, X_di]), y_wl, "X29+di LGB", best=b_wl)
lodo(np.hstack([X29, X_di, X_ctx_n]), y_wl, "X29+di+ctx LGB", best=b_wl)
lodo(np.hstack([X29T, X_di]), y_wl, "X29T+di LGB", best=b_wl)
lodo(np.hstack([X29T, X_di, X_ctx_n]), y_wl, "X29T+di+ctx LGB", best=b_wl)
lodo(np.hstack([X29T, X_di, X_ctx_n]), y_wl, "X29T+di+ctx XGB_F",
     XGBRegressor, XGB_F, b_wl)

# ── Few-shot calibration (oracle) ─────────────────────────────────────────
# If we allow just 2 runs of the new design, can we calibrate?
print(f"\n{T()} === Few-shot calibration (2 calibration runs) ===")
print("  For each test placement: use first 2 runs to calibrate per-design slope")

def lodo_fewshot(X, y, label, n_calib=2, cls=LGBMRegressor, kw=None):
    """
    Use n_calib runs from the first placement of held-out design to calibrate.
    Then apply corrected model to all test placements.
    """
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = cls(**kw); m.fit(sc.fit_transform(X[tr]), y[tr])

        # Few-shot calibration: use first n_calib runs from first placement
        te_pids = np.unique(pids[te])
        first_pid = te_pids[0]
        calib_idx = np.where(pids == first_pid)[0][:n_calib]
        all_te_idx = np.where(te)[0]

        # Compute bias correction from calibration runs
        pred_calib = m.predict(sc.transform(X[calib_idx]))
        true_calib = y[calib_idx]
        bias = (true_calib - pred_calib).mean()  # simple additive bias

        # Scale correction (slope adjustment)
        pred_te = m.predict(sc.transform(X[all_te_idx]))
        pred_te_corrected = pred_te + bias

        maes.append(mean_absolute_error(y[all_te_idx], pred_te_corrected))
    mean_mae = np.mean(maes)
    tag = ' ✓✓' if mean_mae < 0.05 else (' ✓' if mean_mae < 0.10 else '')
    print(f"  {label} (n_calib={n_calib}): "
          f"{maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    return mean_mae

X_best = np.hstack([X29T, X_di, X_ctx_n])
print(f"\n  Power:")
lodo_fewshot(X_best, y_pw, "X29T+di+ctx LGB", n_calib=1)
lodo_fewshot(X_best, y_pw, "X29T+di+ctx LGB", n_calib=2)
lodo_fewshot(X_best, y_pw, "X29T+di+ctx LGB", n_calib=5)

print(f"\n  WL:")
lodo_fewshot(X_best, y_wl, "X29T+di+ctx LGB", n_calib=1)
lodo_fewshot(X_best, y_wl, "X29T+di+ctx LGB", n_calib=2)
lodo_fewshot(X_best, y_wl, "X29T+di+ctx LGB", n_calib=5)

print(f"\n{T()} DONE")
