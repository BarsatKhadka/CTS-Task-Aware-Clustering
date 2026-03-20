"""
Save Best Overnight Models — Using configs discovered in overnight_best.py experiment.

Best configs found:
  Power:  LGB X29       (n=300, lr=0.03, num_leaves=20) -> pw_LODO=0.0656
  WL:     XGB X69-64dim (n=1000, lr=0.01, max_depth=6)  -> wl_LODO=0.0849  [NEW BEST]
  Skew:   XGB X49-49dim (n=300,  lr=0.03, max_depth=4)  -> sk_LODO=0.2372

2-source ensemble weights:
  Power: A1_X29_LGB(0.65) + A3_chain_inv(0.35) -> 0.0665 (OOF metric)
  WL:    A2_n1000_d6(0.80) + A5_best_seed(0.20) -> 0.0853 (OOF metric)
  (Ensemble does NOT improve fold-avg metrics due to OOF metric difference)

Final model: trains on ALL data with best single configs.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# DATA LOADING (same as overnight_best.py)
# ---------------------------------------------------------------------------

with open('cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)

X_cache = cache['X']
Y_cache = cache['Y']
df_cache = cache['df']

with open('tight_path_feats_cache.pkl', 'rb') as f:
    tight_path_cache = pickle.load(f)

pids = df_cache['placement_id'].values
designs = df_cache['design_name'].values
design_list = sorted(np.unique(designs))

# ---------------------------------------------------------------------------
# FEATURE BUILDING (verbatim from overnight_best.py)
# ---------------------------------------------------------------------------

def rank_within(vals):
    n = len(vals)
    return np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)

def zscore_pred_to_rank_mae(y_pred_z, y_true_z, pids_):
    all_pred_r = np.zeros(len(pids_), dtype=float)
    all_true_r = np.zeros(len(pids_), dtype=float)
    for pid in np.unique(pids_):
        m = pids_ == pid
        rows = np.where(m)[0]
        all_pred_r[rows] = rank_within(y_pred_z[rows])
        all_true_r[rows] = rank_within(y_true_z[rows])
    return mean_absolute_error(all_true_r, all_pred_r)

def build_rank_targets(Y, pids_):
    n = len(pids_)
    Y_rank = np.zeros((n, 3), np.float32)
    for pid in np.unique(pids_):
        mask = pids_ == pid
        rows = np.where(mask)[0]
        for j in range(3):
            Y_rank[rows, j] = rank_within(Y[rows, j])
    return Y_rank

def build_X29(df, X_c):
    pids_ = df['placement_id'].values
    n = len(pids_)
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    place_cols = ['core_util', 'density', 'aspect_ratio']
    Xraw = df[knob_cols].values.astype(np.float32)
    Xplace = df[place_cols].values.astype(np.float32)
    Xkz = X_c[:, 72:76]
    raw_max = Xraw.max(axis=0) + 1e-6
    Xrank = np.zeros((n, 4), np.float32)
    Xcentered = np.zeros((n, 4), np.float32)
    Xknob_range = np.zeros((n, 4), np.float32)
    Xknob_mean = np.zeros((n, 4), np.float32)
    for pid in np.unique(pids_):
        mask = pids_ == pid
        rows = np.where(mask)[0]
        for ki in range(4):
            z_vals = Xkz[rows, ki]
            Xrank[rows, ki] = rank_within(z_vals)
            Xcentered[rows, ki] = z_vals - z_vals.mean()
            Xknob_range[rows, ki] = Xraw[rows, ki].std()
            Xknob_mean[rows, ki] = Xraw[rows, ki].mean()
    Xplace_norm = Xplace.copy()
    Xplace_norm[:, 0] /= 100.0
    Xkp = np.column_stack([
        Xraw[:, 3] * Xplace[:, 0] / 100,
        Xraw[:, 0] * Xplace[:, 1],
        Xraw[:, 3] / np.maximum(Xplace[:, 1], 0.01),
        Xraw[:, 3] * Xplace[:, 2],
        Xrank[:, 3] * (Xplace[:, 0] / 100),
        Xrank[:, 2] * (Xplace[:, 0] / 100),
    ])
    return np.hstack([Xkz, Xrank, Xcentered, Xplace_norm,
                      Xknob_range / raw_max, Xknob_mean / raw_max, Xkp])

def build_tight_features(df):
    pids_ = df['placement_id'].values
    n = len(pids_)
    X_tight = np.zeros((n, 20), np.float32)
    for i, pid in enumerate(pids_):
        if pid in tight_path_cache:
            X_tight[i] = tight_path_cache[pid]
    return X_tight

def build_inverse_features(df, pids_):
    n = len(pids_)
    cd = df['cts_cluster_dia'].values.astype(np.float64)
    cs = df['cts_cluster_size'].values.astype(np.float64)
    inv_cd = 1.0 / cd
    log_cd = np.log(cd)
    inv_cs = 1.0 / cs
    log_cs = np.log(cs)
    raw_feats = np.column_stack([inv_cd, log_cd, inv_cs, log_cs]).astype(np.float32)
    rank_feats = np.zeros((n, 4), np.float32)
    cent_feats = np.zeros((n, 4), np.float32)
    for pid in np.unique(pids_):
        mask = pids_ == pid
        rows = np.where(mask)[0]
        for ki in range(4):
            v = raw_feats[rows, ki]
            rank_feats[rows, ki] = rank_within(v)
            cent_feats[rows, ki] = v - v.mean()
    raw_norm = raw_feats / (np.abs(raw_feats).max(axis=0, keepdims=True) + 1e-6)
    prod_cd_cs = (inv_cd * inv_cs).astype(np.float32).reshape(-1, 1)
    prod_log = (log_cd + log_cs).astype(np.float32).reshape(-1, 1)
    prod_cd_cs /= (np.abs(prod_cd_cs).max() + 1e-8)
    prod_log /= (np.abs(prod_log).max() + 1e-8)
    rank_inv_cd = rank_feats[:, 0:1]
    return np.hstack([raw_norm, rank_feats, cent_feats, prod_cd_cs, prod_log, rank_inv_cd])

print("Building features...")
X29 = build_X29(df_cache, X_cache)
X_tight = build_tight_features(df_cache)
X_inv = build_inverse_features(df_cache, pids)
X49 = np.hstack([X29, X_tight])
X69 = np.hstack([X29, X_tight, X_inv])
Y_rank = build_rank_targets(Y_cache, pids)

print(f"  X29:{X29.shape}, X49:{X49.shape}, X69:{X69.shape}")

# ---------------------------------------------------------------------------
# BEST CONFIGS
# ---------------------------------------------------------------------------

LGB_POWER_CFG = dict(n_estimators=300, learning_rate=0.03, num_leaves=20,
                     min_child_samples=15, n_jobs=4, verbose=-1)

XGB_WL_CFG = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
                  min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                  n_jobs=4, verbosity=0)

XGB_SK_CFG = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                  min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                  n_jobs=4, verbosity=0)

# ---------------------------------------------------------------------------
# LODO VERIFICATION (reproduce overnight results)
# ---------------------------------------------------------------------------

print("\nVerifying LODO results with best configs...")
pw_folds, wl_folds, sk_folds = [], [], []
pw_oofs = np.zeros(len(pids))
wl_oofs = np.zeros(len(pids))
sk_oofs = np.zeros(len(pids))

for held in design_list:
    tr = designs != held
    te = ~tr
    sc = StandardScaler()

    # Power: X29 LGB
    m_pw = lgb.LGBMRegressor(**LGB_POWER_CFG)
    m_pw.fit(X29[tr], Y_cache[tr, 1])
    pw_pred = m_pw.predict(X29[te])
    pw_oofs[te] = pw_pred
    pw_mae = zscore_pred_to_rank_mae(pw_pred, Y_cache[te, 1], pids[te])
    pw_folds.append(pw_mae)

    # WL: X69 XGB depth=6
    Xtr = sc.fit_transform(X69[tr]); Xte = sc.transform(X69[te])
    m_wl = xgb.XGBRegressor(**XGB_WL_CFG)
    m_wl.fit(Xtr, Y_cache[tr, 2])
    wl_pred = m_wl.predict(Xte)
    wl_oofs[te] = wl_pred
    wl_mae = zscore_pred_to_rank_mae(wl_pred, Y_cache[te, 2], pids[te])
    wl_folds.append(wl_mae)

    # Skew: X49 XGB
    Xtr49 = sc.fit_transform(X49[tr]); Xte49 = sc.transform(X49[te])
    m_sk = xgb.XGBRegressor(**XGB_SK_CFG)
    m_sk.fit(Xtr49, Y_rank[tr, 0])
    sk_pred = m_sk.predict(Xte49)
    sk_oofs[te] = sk_pred
    sk_mae = mean_absolute_error(Y_rank[te, 0], sk_pred)
    sk_folds.append(sk_mae)

    print(f"  {held:12s}  pw={pw_mae:.4f}  wl={wl_mae:.4f}  sk={sk_mae:.4f}")

pw_mean = np.mean(pw_folds)
wl_mean = np.mean(wl_folds)
sk_mean = np.mean(sk_folds)
print(f"\n  LODO: pw={pw_mean:.4f}  wl={wl_mean:.4f}  sk={sk_mean:.4f}")
print(f"  Power {'✓' if pw_mean < 0.1 else '✗'}  WL {'✓' if wl_mean < 0.1 else '✗'}  Skew {'✓' if sk_mean < 0.1 else '✗'}")

# ---------------------------------------------------------------------------
# TRAIN FINAL MODELS ON ALL DATA
# ---------------------------------------------------------------------------

print("\nTraining final models on all data...")

sc_wl_final = StandardScaler()
X69_scaled = sc_wl_final.fit_transform(X69)

final_pw = lgb.LGBMRegressor(**LGB_POWER_CFG)
final_pw.fit(X29, Y_cache[:, 1])

final_wl = xgb.XGBRegressor(**XGB_WL_CFG)
final_wl.fit(X69_scaled, Y_cache[:, 2])

sc_sk_final = StandardScaler()
X49_scaled = sc_sk_final.fit_transform(X49)
final_sk = xgb.XGBRegressor(**XGB_SK_CFG)
final_sk.fit(X49_scaled, Y_rank[:, 0])

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

model_bundle = {
    'power': {
        'model': final_pw,
        'feature_set': 'X29',
        'feature_dim': 29,
        'target': 'per_placement_zscore_power',
        'lodo_mae': pw_mean,
        'lodo_folds': dict(zip(design_list, pw_folds)),
        'config': LGB_POWER_CFG,
        'library': 'lightgbm',
    },
    'wirelength': {
        'model': final_wl,
        'scaler': sc_wl_final,
        'feature_set': 'X69',
        'feature_dim': 69,
        'target': 'per_placement_zscore_wirelength',
        'lodo_mae': wl_mean,
        'lodo_folds': dict(zip(design_list, wl_folds)),
        'config': XGB_WL_CFG,
        'library': 'xgboost',
    },
    'skew': {
        'model': final_sk,
        'scaler': sc_sk_final,
        'feature_set': 'X49',
        'feature_dim': 49,
        'target': 'per_placement_rank_skew',
        'lodo_mae': sk_mean,
        'lodo_folds': dict(zip(design_list, sk_folds)),
        'config': XGB_SK_CFG,
        'library': 'xgboost',
    },
    'build_X29_src': 'overnight_best.py::build_X29',
    'build_tight_src': 'overnight_best.py::build_tight_features',
    'build_inverse_src': 'overnight_best.py::build_inverse_features',
    'tight_path_cache_path': 'tight_path_feats_cache.pkl',
    'session': 'session6_overnight',
}

with open('best_model_v7.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"\nSaved best_model_v7.pkl")
print(f"  Power:      {pw_mean:.4f}  (best: 0.0656)")
print(f"  Wirelength: {wl_mean:.4f}  (best: 0.0858 -> NEW BEST!)")
print(f"  Skew:       {sk_mean:.4f}  (best: 0.2369)")
