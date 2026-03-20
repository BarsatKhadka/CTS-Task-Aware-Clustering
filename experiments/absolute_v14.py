"""
absolute_v14.py — Zero-shot absolute prediction: Physics-Formula + Simulation

Goal: Predict ACTUAL power (W) and WL (µm) from single run, zero-shot.
No rank comparison, no per-placement normalization.

Key insight: CTS physics gives us:
  power ≈ n_buffers × C_buf × V² × f + WL × C_wire × V² × f × toggle
  WL    ≈ 1.2 × HPWL(FF bounding box) × f(cluster_dia/knn_dist, cluster_size)

Approach:
1. Compute physics quantities in ABSOLUTE units (µm, count)
   - n_ff from ff_positions
   - HPWL in µm from ff_positions × die_w/die_h
   - sim_n_clusters from sim_features_cache (rescaled to count)
   - sim_total_wl from sim_features_cache (rescaled to µm)
2. Use design-invariant ratios as targets:
   - log(power / (n_ff/cluster_size)) ← power per buffer stage
   - log(WL / HPWL) ← actual/ideal WL ratio
3. LGB/XGB LODO with MAPE metric

Prior best: v5 = power 37.8% MAPE, WL 21.2% MAPE
"""

import os, pickle, time, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Absolute v14: Zero-shot Physics-Formula + Simulation")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
Y_cache, df_cache = cache['Y'], cache['df']

with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_cache = pickle.load(f)

with open(f'{BASE}/sim_features_cache.pkl', 'rb') as f:
    sim_cache = pickle.load(f)

df_csv = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
# Match df_csv to df_cache ordering
df_csv = df_csv.set_index('run_id').reindex(df_cache['run_id'].values).reset_index()

pids    = df_cache['placement_id'].values
designs = df_cache['design_name'].values
run_ids = df_cache['run_id'].values
n = len(pids)

y_pw_abs = df_csv['power_total'].values.astype(np.float32)   # actual power (W)
y_wl_abs = df_csv['wirelength'].values.astype(np.float32)    # actual WL (µm)

print(f"{T()} Loaded: {n} rows")
print(f"  power range: [{y_pw_abs.min():.6f}, {y_pw_abs.max():.6f}] W")
print(f"  WL range: [{y_wl_abs.min():.0f}, {y_wl_abs.max():.0f}] µm")

# ── Physics quantities per run ─────────────────────────────────────────
def mape(y_true, y_pred):
    return 100.0 * np.mean(np.abs((y_pred - y_true) / (y_true + 1e-12)))

def empty_sim():
    return {k: 0.0 for k in ['sim_n_clusters','sim_sum_hpwl','sim_mean_hpwl',
                               'sim_max_hpwl','sim_std_hpwl','sim_skew_proxy',
                               'sim_inter_hpwl','sim_intra_wl','sim_total_wl',
                               'sim_fill_rate','sim_mean_cs','sim_frac_single']}

print(f"{T()} Building physics features...")
feats = []
n_ff_arr = np.zeros(n, np.float32)
hpwl_arr = np.zeros(n, np.float32)
die_area_arr = np.zeros(n, np.float32)

for i, (pid, rid) in enumerate(zip(pids, run_ids)):
    # Placement-level quantities
    pos = ff_cache.get(pid)
    if pos and pos.get('ff_norm') is not None:
        ff_xy = pos['ff_norm']  # [N, 2] in [0,1] die fraction
        die_w = pos.get('die_w', 500.0)
        die_h = pos.get('die_h', 500.0)
        n_ff = len(ff_xy)
        # HPWL in µm
        hpwl = (ff_xy[:, 0].max() - ff_xy[:, 0].min()) * die_w + \
               (ff_xy[:, 1].max() - ff_xy[:, 1].min()) * die_h
        die_area = die_w * die_h / 1e6  # in mm²
        die_scale = np.sqrt(die_w * die_h)  # for normalizing distances
    else:
        n_ff = 1000; hpwl = 500.0; die_area = 0.25; die_scale = 500.0

    n_ff_arr[i] = n_ff
    hpwl_arr[i] = hpwl
    die_area_arr[i] = die_area

    # CTS knobs
    cs = df_cache.iloc[i]['cts_cluster_size']
    cd = df_cache.iloc[i]['cts_cluster_dia']
    mw = df_cache.iloc[i]['cts_max_wire']
    bd = df_cache.iloc[i]['cts_buf_dist']

    # Simulation-based quantities (rescale from die-fraction to physical units)
    sim = sim_cache.get(rid, empty_sim())
    sim_n = sim['sim_n_clusters']           # cluster count (design-scale)
    sim_wl = sim['sim_total_wl'] * die_scale  # total WL in µm
    sim_sh = sim['sim_sum_hpwl'] * die_scale  # sum cluster HPWL in µm
    sim_ih = sim['sim_inter_hpwl'] * die_scale  # inter-cluster HPWL in µm

    # Derived physics quantities
    n_buf_proxy = n_ff / (cs + 1e-4)  # proxy for buffer count
    hpwl_per_ff = hpwl / (n_ff + 1)    # FF spread per FF (µm/FF)
    cd_vs_hpwl = cd / (hpwl / n_ff**0.5 + 1e-4)  # cluster vs sqrt(density)
    wl_per_ff = hpwl_per_ff * np.log1p(n_ff / cs)  # WL per FF proxy

    # Feature vector (all log-normalized for cross-design invariance)
    feat = [
        # Core physics features (log scale)
        np.log1p(n_ff),
        np.log1p(hpwl),
        np.log1p(die_area * 1e6),
        np.log1p(n_buf_proxy),
        np.log1p(hpwl_per_ff),
        # CTS knobs (raw and log)
        np.log1p(cs), np.log1p(cd), np.log1p(mw), np.log1p(bd),
        cs, cd, mw, bd,  # raw knobs
        # Key interactions (physics-motivated)
        np.log1p(n_ff / cs),         # log cluster count proxy
        np.log1p(cd / (hpwl_per_ff + 1e-4)),  # cluster spans FF spacing?
        np.log1p(mw / (hpwl_per_ff + 1e-4)),  # wire budget vs FF spacing
        np.log1p(n_ff * hpwl_per_ff**2),  # total routing area proxy
        hpwl / (mw + 1e-4),             # hpwl / wire budget
        n_ff / (cs * np.log1p(hpwl)),  # buffer density per log-WL
        # Simulation-based features (physics simulation outputs)
        np.log1p(sim_n),
        np.log1p(sim_wl),
        np.log1p(sim_sh),
        np.log1p(sim_ih),
        sim_n / (n_ff / cs + 1e-4),     # sim vs analytical cluster count ratio
        sim_wl / (hpwl + 1e-4),         # sim WL / geometric WL
        # Die-normalized features
        np.log1p(n_ff / die_area),       # FF density (per µm²)
        np.log1p(hpwl / np.sqrt(die_area * 1e6 + 1)),  # HPWL vs sqrt(die)
    ]
    feats.append(feat)

X_abs = np.array(feats, dtype=np.float32)

# Fix NaN/inf
for c in range(X_abs.shape[1]):
    bad = ~np.isfinite(X_abs[:, c])
    if bad.any():
        X_abs[bad, c] = np.nanmedian(X_abs[~bad, c]) if (~bad).any() else 0.0

print(f"{T()} X_abs shape: {X_abs.shape}")
print(f"  n_ff range: [{n_ff_arr.min():.0f}, {n_ff_arr.max():.0f}]")
print(f"  HPWL range: [{hpwl_arr.min():.0f}, {hpwl_arr.max():.0f}] µm")

# ── Novel targets: design-invariant ratios ─────────────────────────────
# Power: normalize by n_buf_proxy (physics: power ∝ n_buffers)
n_buf_arr = n_ff_arr / (df_cache['cts_cluster_size'].values + 1e-4)
y_pw_ratio = np.log(y_pw_abs / (n_buf_arr + 1e-10))  # log(power / n_buf)
y_wl_ratio = np.log(y_wl_abs / (hpwl_arr + 1e-4))   # log(WL / HPWL)

print(f"\n  Power ratio range: [{y_pw_ratio.min():.3f}, {y_pw_ratio.max():.3f}]")
print(f"  WL ratio range: [{y_wl_ratio.min():.3f}, {y_wl_ratio.max():.3f}]")
print(f"  WL/HPWL mean: {np.exp(y_wl_ratio).mean():.3f} (expect ~1.2)")

# Check cross-design variation of ratios
for d in sorted(np.unique(designs)):
    m = designs == d
    print(f"  {d}: pw_ratio=[{y_pw_ratio[m].min():.2f},{y_pw_ratio[m].max():.2f}]  "
          f"wl_ratio=[{y_wl_ratio[m].min():.2f},{y_wl_ratio[m].max():.2f}]")

# ── LODO evaluation ────────────────────────────────────────────────────
def lodo_absolute(X, y_ratio, normalizer, label, cls=LGBMRegressor, kw=None):
    """LODO: train on ratio, predict ratio, denormalize, compute MAPE."""
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=10, verbose=-1)
    dl = sorted(np.unique(designs))
    pw_mapes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = cls(**kw)
        m.fit(sc.fit_transform(X[tr]), y_ratio[tr])
        pred_ratio = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_ratio) * normalizer[te]
        true_abs = np.exp(y_ratio[te]) * normalizer[te]
        mp = mape(true_abs, pred_abs)
        pw_mapes.append(mp)
    mean_mape = np.mean(pw_mapes)
    folds_str = '/'.join([f'{x:.1f}%' for x in pw_mapes])
    s = '✓' if mean_mape < 15 else ('~' if mean_mape < 25 else '')
    print(f"  {label}: {folds_str}  mean={mean_mape:.1f}% {s}")
    return mean_mape, pw_mapes

# ── Experiments ───────────────────────────────────────────────────────
print(f"\n{T()} === POWER ABSOLUTE (target: log(power/n_buf)) ===")
print(f"  Prior best: 37.8% MAPE")

LGB_F = dict(n_estimators=300, num_leaves=20, learning_rate=0.03, min_child_samples=10, verbose=-1)
LGB_L = dict(n_estimators=1000, num_leaves=20, learning_rate=0.01, min_child_samples=10, verbose=-1)
XGB_F = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, verbosity=0)
RID = None  # Ridge

lodo_absolute(X_abs, y_pw_ratio, n_buf_arr, "LGB_300 X_abs", LGBMRegressor, LGB_F)
lodo_absolute(X_abs, y_pw_ratio, n_buf_arr, "XGB_300 X_abs", XGBRegressor, XGB_F)
lodo_absolute(X_abs, y_pw_ratio, n_buf_arr, "LGB_1000 X_abs", LGBMRegressor, LGB_L)

# Try different normalizer: n_ff (simpler)
lodo_absolute(X_abs, np.log(y_pw_abs / n_ff_arr), n_ff_arr,
              "LGB_300 X_abs [norm=n_ff]", LGBMRegressor, LGB_F)

# Ridge regression (most regularized, may generalize better)
def lodo_ridge(X, y_ratio, normalizer, label, alpha=1.0):
    dl = sorted(np.unique(designs)); mapes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = Ridge(alpha=alpha)
        m.fit(sc.fit_transform(X[tr]), y_ratio[tr])
        pred_ratio = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_ratio) * normalizer[te]
        true_abs = np.exp(y_ratio[te]) * normalizer[te]
        mapes.append(mape(true_abs, pred_abs))
    mean_m = np.mean(mapes)
    print(f"  {label}: {'/'.join([f'{x:.1f}%' for x in mapes])}  mean={mean_m:.1f}%")
    return mean_m

lodo_ridge(X_abs, y_pw_ratio, n_buf_arr, "Ridge(1.0) X_abs")
lodo_ridge(X_abs, y_pw_ratio, n_buf_arr, "Ridge(10) X_abs", alpha=10.0)

print(f"\n{T()} === WL ABSOLUTE (target: log(WL/HPWL)) ===")
print(f"  Prior best: 21.2% MAPE")

lodo_absolute(X_abs, y_wl_ratio, hpwl_arr, "LGB_300 X_abs", LGBMRegressor, LGB_F)
lodo_absolute(X_abs, y_wl_ratio, hpwl_arr, "XGB_300 X_abs", XGBRegressor, XGB_F)
lodo_absolute(X_abs, y_wl_ratio, hpwl_arr, "LGB_1000 X_abs", LGBMRegressor, LGB_L)

lodo_ridge(X_abs, y_wl_ratio, hpwl_arr, "Ridge(1.0) X_abs")
lodo_ridge(X_abs, y_wl_ratio, hpwl_arr, "Ridge(10) X_abs", alpha=10.0)

# Direct prediction (no ratio)
lodo_absolute(X_abs, np.log(y_wl_abs), np.ones(n),
              "LGB_300 log(WL) direct", LGBMRegressor, LGB_F)

print(f"\n{T()} === NOVEL: Simulation-based direct formula ===")
# Physics formula: power ≈ a × sim_n_clusters + b × sim_total_wl + c
# WL ≈ d × HPWL + e × sim_total_wl
# Test: does adding sim features to X_abs help?
# Build feature set focused on simulation
sim_n_arr = np.array([sim_cache.get(r, empty_sim())['sim_n_clusters'] for r in run_ids], np.float32)
sim_wl_arr = np.array([sim_cache.get(r, empty_sim())['sim_total_wl'] for r in run_ids], np.float32)
die_scales = np.array([ff_cache.get(pid, {}).get('die_w', 500) * ff_cache.get(pid, {}).get('die_h', 500) ** 0.5
                       if ff_cache.get(pid) else 500.0 for pid in pids], np.float32)
sim_wl_um = sim_wl_arr * die_scales

# Check direct correlation
for d in sorted(np.unique(designs)):
    m = designs == d
    corr_pw = np.corrcoef(sim_n_arr[m], y_pw_abs[m])[0, 1]
    corr_wl = np.corrcoef(sim_wl_um[m], y_wl_abs[m])[0, 1]
    print(f"  {d}: corr(sim_n, power)={corr_pw:.3f}  corr(sim_wl, WL)={corr_wl:.3f}")

# Build minimal simulation-based features
Xsim_min = np.column_stack([
    np.log1p(n_ff_arr), np.log1p(hpwl_arr),
    np.log1p(sim_n_arr), np.log1p(sim_wl_um),
    np.log1p(df_cache['cts_cluster_size'].values),
    np.log1p(df_cache['cts_cluster_dia'].values),
])
for c in range(Xsim_min.shape[1]):
    bad = ~np.isfinite(Xsim_min[:, c])
    if bad.any(): Xsim_min[bad, c] = 0.0

lodo_absolute(Xsim_min, y_pw_ratio, n_buf_arr, "LGB_300 Xsim_min [power]", LGBMRegressor, LGB_F)
lodo_absolute(Xsim_min, y_wl_ratio, hpwl_arr, "LGB_300 Xsim_min [WL]", LGBMRegressor, LGB_F)

# Combined: full X_abs + simulation
X_full = np.hstack([X_abs, np.log1p(sim_wl_um).reshape(-1, 1)])
lodo_absolute(X_full, y_pw_ratio, n_buf_arr, "LGB_300 X_abs+sim_wl [power]", LGBMRegressor, LGB_F)
lodo_absolute(X_full, y_wl_ratio, hpwl_arr, "LGB_300 X_abs+sim_wl [WL]", LGBMRegressor, LGB_F)

print(f"\n{T()} DONE")
