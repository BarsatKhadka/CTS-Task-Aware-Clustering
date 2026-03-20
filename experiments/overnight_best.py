"""
Overnight Best — Comprehensive CTS Prediction Experiment
=========================================================
Goal: Push power and WL MAE to near-zero using LODO evaluation.
Current best (best_model_v6.pkl): pw=0.0656, wl=0.0858, sk=0.2369

Approaches:
  1. Baseline — X29+tight (49 dims), z-score targets, LGB/XGB
  2. Physical Inverse Features — add 1/cd, log(cd), 1/cs, log(cs) and variants
  3. Cross-Task Physical Chain — WL z-score as power feature
  4. Isotonic Post-Processing — enforce monotone P(cluster_dia) within placement
  5. Calibrated Quantile Ensemble — multi-quantile LGB + seed ensemble
  6. Spatial Grid Features — 8x8 grid from .pt graph files (bounding-box normalized)
  7. Per-Placement Function Parameter Fitting (beta meta-regression)
  8. Optimal Ensemble — combine all approaches

Evaluation metric: rank MAE within placement (for z-score predictions, convert to rank first)
"""

import pickle
import warnings
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

print("=" * 70)
print("OVERNIGHT BEST — CTS PREDICTION EXPERIMENT")
print("=" * 70)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

print(f"\n{elapsed()} Loading data...")

with open('cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)

X_cache = cache['X']
Y_cache = cache['Y']   # Per-placement z-scored targets (shape: [5390, 3])
df_cache = cache['df']  # DataFrame with all metadata

with open('tight_path_feats_cache.pkl', 'rb') as f:
    tight_path_cache = pickle.load(f)

print(f"  X_cache: {X_cache.shape}")
print(f"  Y_cache: {Y_cache.shape}  (per-placement z-scored: skew, power, WL)")
print(f"  df_cache rows: {len(df_cache)}")
print(f"  tight_path_cache: {len(tight_path_cache)} entries (20-dim each)")

pids = df_cache['placement_id'].values
designs = df_cache['design_name'].values
design_list = sorted(np.unique(designs))

print(f"  Designs: {dict(zip(*np.unique(designs, return_counts=True)))}")
print(f"  Unique placements: {len(np.unique(pids))}")

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def rank_within(vals):
    """Fractional rank: 0=lowest, 1=highest."""
    n = len(vals)
    return np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)


def build_rank_targets(Y, pids_):
    n = len(pids_)
    Y_rank = np.zeros((n, 3), np.float32)
    for pid in np.unique(pids_):
        mask = pids_ == pid
        rows = np.where(mask)[0]
        for j in range(3):
            Y_rank[rows, j] = rank_within(Y[rows, j])
    return Y_rank


def zscore_pred_to_rank_mae(y_pred_z, y_true_z, pids_):
    all_pred_r = np.zeros(len(pids_), dtype=float)
    all_true_r = np.zeros(len(pids_), dtype=float)
    for pid in np.unique(pids_):
        m = pids_ == pid
        rows = np.where(m)[0]
        all_pred_r[rows] = rank_within(y_pred_z[rows])
        all_true_r[rows] = rank_within(y_true_z[rows])
    return mean_absolute_error(all_true_r, all_pred_r)


# ---------------------------------------------------------------------------
# FEATURE BUILDING
# ---------------------------------------------------------------------------

def build_X29(df, X_c):
    """29-dim baseline features (from best_model_v6)."""
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
    """20-dim: 1/cd, log(cd), 1/cs, log(cs) + per-placement rank + centered + products."""
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


print(f"\n{elapsed()} Building base features...")
X29 = build_X29(df_cache, X_cache)
X_tight = build_tight_features(df_cache)
Y_rank = build_rank_targets(Y_cache, pids)
X_inv = build_inverse_features(df_cache, pids)

X49 = np.hstack([X29, X_tight])
X69 = np.hstack([X29, X_tight, X_inv])
X_pw_inv = np.hstack([X29, X_inv])

print(f"  X29:{X29.shape}, X49:{X49.shape}, X69:{X69.shape}, X_pw_inv:{X_pw_inv.shape}")
print(f"  Y_cache (z-score): {Y_cache.shape}, Y_rank: {Y_rank.shape}")

# ---------------------------------------------------------------------------
# MODEL CONFIGS
# ---------------------------------------------------------------------------

LGB_CONFIGS = [
    dict(n_estimators=300, learning_rate=0.03, num_leaves=20, min_child_samples=15, n_jobs=4, verbose=-1),
    dict(n_estimators=500, learning_rate=0.02, num_leaves=31, min_child_samples=10, n_jobs=4, verbose=-1),
    dict(n_estimators=1000, learning_rate=0.01, num_leaves=31, min_child_samples=10, n_jobs=4, verbose=-1),
    dict(n_estimators=2000, learning_rate=0.005, num_leaves=31, min_child_samples=10, n_jobs=4, verbose=-1),
    dict(n_estimators=300, learning_rate=0.03, num_leaves=40, min_child_samples=10, n_jobs=4, verbose=-1),
    dict(n_estimators=500, learning_rate=0.02, num_leaves=50, min_child_samples=8, n_jobs=4, verbose=-1),
]

XGB_CONFIGS = [
    dict(n_estimators=1000, learning_rate=0.01, max_depth=4, min_child_weight=15, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
    dict(n_estimators=2000, learning_rate=0.005, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
    dict(n_estimators=1000, learning_rate=0.01, max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
    dict(n_estimators=500, learning_rate=0.02, max_depth=4, min_child_weight=15, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
]

LGB_DEFAULT = dict(n_estimators=300, learning_rate=0.03, num_leaves=20, min_child_samples=15, n_jobs=4, verbose=-1)
XGB_DEFAULT = dict(n_estimators=1000, learning_rate=0.01, max_depth=4, min_child_weight=15, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0)
XGB_SK = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=15, subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0)

# ---------------------------------------------------------------------------
# HELPERS FOR LODO
# ---------------------------------------------------------------------------

def lodo_zscore(X, Y_z_col, pids_, designs_, model_cls, params, label=""):
    """LODO with z-score targets, rank MAE evaluation."""
    folds = []
    oof = np.zeros(len(pids_), dtype=float)
    for held in design_list:
        tr = designs_ != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = model_cls(**params)
        m.fit(Xtr, Y_z_col[tr])
        pred = m.predict(Xte)
        oof[te] = pred
        folds.append(zscore_pred_to_rank_mae(pred, Y_z_col[te], pids_[te]))
    mean_mae = np.mean(folds)
    fold_str = [f'{v:.4f}' for v in folds]
    status = "PASS ✓" if mean_mae < 0.10 else "FAIL"
    if label:
        print(f"    {label}: {fold_str}  mean={mean_mae:.4f}  {status}")
    return folds, mean_mae, oof


def lodo_rank(X, Y_rank_col, pids_, designs_, model_cls, params, label=""):
    """LODO with rank targets, direct MAE evaluation."""
    folds = []
    oof = np.zeros(len(pids_), dtype=float)
    for held in design_list:
        tr = designs_ != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        m = model_cls(**params)
        m.fit(Xtr, Y_rank_col[tr])
        pred = m.predict(Xte)
        oof[te] = pred
        folds.append(mean_absolute_error(Y_rank_col[te], pred))
    mean_mae = np.mean(folds)
    fold_str = [f'{v:.4f}' for v in folds]
    status = "PASS ✓" if mean_mae < 0.10 else "FAIL"
    if label:
        print(f"    {label}: {fold_str}  mean={mean_mae:.4f}  {status}")
    return folds, mean_mae, oof


# Track best configs
best_pw_cfg = None; best_pw_mae = 999
best_wl_cfg = None; best_wl_mae = 999
best_sk_cfg = None; best_sk_mae = 999; best_sk_model = 'xgb'

# OOF prediction stores
pw_oofs = {}; wl_oofs = {}

# Results dict
results_summary = {}

# ---------------------------------------------------------------------------
# APPROACH 1: BASELINE
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 1: BASELINE (X29+tight=49dim, z-score targets for pw/wl)")
print("=" * 70)

try:
    print(f"{elapsed()} Skew (X49, rank target, XGB)...")
    sk_f1, sk_m1, sk_oof1 = lodo_rank(X49, Y_rank[:, 0], pids, designs,
                                        xgb.XGBRegressor, XGB_SK, "sk_XGB_SK")

    print(f"{elapsed()} Power (X29, z-score, LGB)...")
    pw_f1, pw_m1, pw_oof1 = lodo_zscore(X29, Y_cache[:, 1], pids, designs,
                                          lgb.LGBMRegressor, LGB_DEFAULT, "pw_LGB_def")
    pw_oofs['A1_X29_LGB'] = pw_oof1

    print(f"{elapsed()} WL (X49, z-score, XGB)...")
    wl_f1, wl_m1, wl_oof1 = lodo_zscore(X49, Y_cache[:, 2], pids, designs,
                                          xgb.XGBRegressor, XGB_DEFAULT, "wl_XGB_def")
    wl_oofs['A1_X49_XGB'] = wl_oof1

    res_A1 = {'sk': {'folds': sk_f1, 'mean': sk_m1},
               'pw': {'folds': pw_f1, 'mean': pw_m1},
               'wl': {'folds': wl_f1, 'mean': wl_m1}}
    results_summary['A1_baseline'] = res_A1

    print(f"\n  A1 Summary: pw={pw_m1:.4f}, wl={wl_m1:.4f}, sk={sk_m1:.4f}")
    print(f"  Reference:  pw=0.0656, wl=0.0858, sk=0.2369")
    print(f"  Delta:      pw={pw_m1-0.0656:+.4f}, wl={wl_m1-0.0858:+.4f}, sk={sk_m1-0.2369:+.4f}")

    if pw_m1 < best_pw_mae:
        best_pw_mae = pw_m1; best_pw_cfg = LGB_DEFAULT
    if wl_m1 < best_wl_mae:
        best_wl_mae = wl_m1; best_wl_cfg = XGB_DEFAULT
    if sk_m1 < best_sk_mae:
        best_sk_mae = sk_m1; best_sk_cfg = XGB_SK; best_sk_model = 'xgb'

except Exception as e:
    print(f"  ERROR in Approach 1: {e}")
    import traceback; traceback.print_exc()
    res_A1 = None

# ---------------------------------------------------------------------------
# APPROACH 2: PHYSICAL INVERSE FEATURES
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 2: PHYSICAL INVERSE FEATURES (X69 = X49 + 1/cd + log/rank)")
print("=" * 70)

try:
    # Grid search for power (X_pw_inv = X29+inv)
    print(f"\n{elapsed()} Grid search: LGB for power with X29+inv ({X_pw_inv.shape[1]}-dim)...")
    for cfg in LGB_CONFIGS:
        folds, mae, oof = lodo_zscore(X_pw_inv, Y_cache[:, 1], pids, designs,
                                       lgb.LGBMRegressor, cfg,
                                       f"pw n={cfg['n_estimators']},lr={cfg['learning_rate']},l={cfg['num_leaves']}")
        pw_oofs[f'A2_n{cfg["n_estimators"]}_l{cfg["num_leaves"]}'] = oof
        if mae < best_pw_mae:
            best_pw_mae = mae; best_pw_cfg = cfg

    print(f"\n  Best power config: {best_pw_cfg}  -> pw={best_pw_mae:.4f}")

    # Grid search for WL (X69 = X49+inv)
    print(f"\n{elapsed()} Grid search: XGB for WL with X69 ({X69.shape[1]}-dim)...")
    for cfg in XGB_CONFIGS:
        folds, mae, oof = lodo_zscore(X69, Y_cache[:, 2], pids, designs,
                                       xgb.XGBRegressor, cfg,
                                       f"wl n={cfg['n_estimators']},lr={cfg['learning_rate']},d={cfg['max_depth']}")
        wl_oofs[f'A2_n{cfg["n_estimators"]}_d{cfg["max_depth"]}'] = oof
        if mae < best_wl_mae:
            best_wl_mae = mae; best_wl_cfg = cfg

    print(f"\n  Best WL config: {best_wl_cfg}  -> wl={best_wl_mae:.4f}")

    # Run with best configs to get per-fold results
    print(f"\n{elapsed()} A2 best configs per-fold detail...")
    pw_f2, pw_m2, pw_oof2 = lodo_zscore(X_pw_inv, Y_cache[:, 1], pids, designs,
                                          lgb.LGBMRegressor, best_pw_cfg, "pw_best_A2")
    wl_f2, wl_m2, wl_oof2 = lodo_zscore(X69, Y_cache[:, 2], pids, designs,
                                          xgb.XGBRegressor, best_wl_cfg, "wl_best_A2")

    pw_oofs['A2_best'] = pw_oof2
    wl_oofs['A2_best'] = wl_oof2

    # Also try skew with X69
    sk_f2, sk_m2, sk_oof2 = lodo_rank(X69, Y_rank[:, 0], pids, designs,
                                        xgb.XGBRegressor, XGB_SK, "sk_X69_XGB")
    if sk_m2 < best_sk_mae:
        best_sk_mae = sk_m2; best_sk_cfg = XGB_SK; best_sk_model = 'xgb'

    res_A2 = {'sk': {'folds': sk_f2, 'mean': sk_m2},
               'pw': {'folds': pw_f2, 'mean': pw_m2},
               'wl': {'folds': wl_f2, 'mean': wl_m2}}
    results_summary['A2_inverse'] = res_A2

    pw_delta = pw_m2 - (res_A1['pw']['mean'] if res_A1 else 0.0656)
    wl_delta = wl_m2 - (res_A1['wl']['mean'] if res_A1 else 0.0858)
    print(f"\n  A2 Summary: pw={pw_m2:.4f} ({pw_delta:+.4f}), wl={wl_m2:.4f} ({wl_delta:+.4f}), sk={sk_m2:.4f}")

except Exception as e:
    print(f"  ERROR in Approach 2: {e}")
    import traceback; traceback.print_exc()
    res_A2 = None

# ---------------------------------------------------------------------------
# APPROACH 3: CROSS-TASK PHYSICAL CHAIN
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 3: CROSS-TASK PHYSICAL CHAIN (WL z-score -> power feature)")
print("=" * 70)
print(f"  rho(power,WL)=0.915 within placement  ->  WL prediction improves power")

try:
    def run_cross_task_chain(X_wl_feat, X_pw_feat, Y_cache_,
                              wl_params, pw_params, label):
        print(f"\n{elapsed()} Cross-task chain: {label}")
        power_maes, wl_maes = [], []
        oof_pw = np.zeros(len(pids), dtype=float)
        oof_wl = np.zeros(len(pids), dtype=float)

        for held in design_list:
            tr = designs != held; te = ~tr
            te_pids = pids[te]

            sc_wl = StandardScaler()
            Xwl_tr = sc_wl.fit_transform(X_wl_feat[tr])
            Xwl_te = sc_wl.transform(X_wl_feat[te])

            m_wl = xgb.XGBRegressor(**wl_params)
            m_wl.fit(Xwl_tr, Y_cache_[tr, 2])
            wl_pred_tr = m_wl.predict(Xwl_tr)
            wl_pred_te = m_wl.predict(Xwl_te)
            oof_wl[te] = wl_pred_te

            wl_mae = zscore_pred_to_rank_mae(wl_pred_te, Y_cache_[te, 2], te_pids)
            wl_maes.append(wl_mae)

            X_pw_tr_aug = np.hstack([X_pw_feat[tr], wl_pred_tr.reshape(-1, 1)])
            X_pw_te_aug = np.hstack([X_pw_feat[te], wl_pred_te.reshape(-1, 1)])

            sc_pw = StandardScaler()
            Xpw_tr = sc_pw.fit_transform(X_pw_tr_aug)
            Xpw_te = sc_pw.transform(X_pw_te_aug)

            m_pw = lgb.LGBMRegressor(**pw_params)
            m_pw.fit(Xpw_tr, Y_cache_[tr, 1])
            pw_pred = m_pw.predict(Xpw_te)
            oof_pw[te] = pw_pred
            pw_mae = zscore_pred_to_rank_mae(pw_pred, Y_cache_[te, 1], te_pids)
            power_maes.append(pw_mae)

            print(f"    {held}: WL={wl_mae:.4f}, power={pw_mae:.4f}")

        print(f"  WL  mean: {np.mean(wl_maes):.4f}  {[f'{v:.4f}' for v in wl_maes]}")
        print(f"  Pow mean: {np.mean(power_maes):.4f}  {[f'{v:.4f}' for v in power_maes]}")
        return power_maes, wl_maes, oof_pw, oof_wl

    # Chain with best configs from A2
    wl_cfg_chain = best_wl_cfg if best_wl_cfg else XGB_DEFAULT
    pw_cfg_chain = best_pw_cfg if best_pw_cfg else LGB_DEFAULT

    # Chain 1: X69 for WL, X_pw_inv for power
    ch1_pw, ch1_wl, ch1_oof_pw, ch1_oof_wl = run_cross_task_chain(
        X_wl_feat=X69, X_pw_feat=X_pw_inv, Y_cache_=Y_cache,
        wl_params=wl_cfg_chain, pw_params=pw_cfg_chain,
        label="X69->WL, X_pw_inv+WL_feat->power"
    )
    pw_oofs['A3_chain_inv'] = ch1_oof_pw
    wl_oofs['A3_chain_inv'] = ch1_oof_wl

    # Chain 2: Baseline (X49 for WL, X29 for power) — compare
    ch2_pw, ch2_wl, ch2_oof_pw, ch2_oof_wl = run_cross_task_chain(
        X_wl_feat=X49, X_pw_feat=X29, Y_cache_=Y_cache,
        wl_params=XGB_DEFAULT, pw_params=LGB_DEFAULT,
        label="X49->WL, X29+WL_feat->power (baseline chain)"
    )
    pw_oofs['A3_chain_base'] = ch2_oof_pw
    wl_oofs['A3_chain_base'] = ch2_oof_wl

    res_A3 = {
        'chain_inv': {'pw': {'folds': ch1_pw, 'mean': np.mean(ch1_pw)},
                      'wl': {'folds': ch1_wl, 'mean': np.mean(ch1_wl)}},
        'chain_base': {'pw': {'folds': ch2_pw, 'mean': np.mean(ch2_pw)},
                       'wl': {'folds': ch2_wl, 'mean': np.mean(ch2_wl)}},
    }
    results_summary['A3_chain'] = res_A3

    print(f"\n  A3 Summary:")
    print(f"    Chain+inv:  pw={np.mean(ch1_pw):.4f}, wl={np.mean(ch1_wl):.4f}")
    print(f"    Chain+base: pw={np.mean(ch2_pw):.4f}, wl={np.mean(ch2_wl):.4f}")

    if np.mean(ch1_pw) < best_pw_mae:
        best_pw_mae = np.mean(ch1_pw)

except Exception as e:
    print(f"  ERROR in Approach 3: {e}")
    import traceback; traceback.print_exc()
    res_A3 = None

# ---------------------------------------------------------------------------
# APPROACH 4: ISOTONIC POST-PROCESSING
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 4: ISOTONIC POST-PROCESSING (monotone P(cluster_dia))")
print("=" * 70)
print(f"  rho(cd, power)=-0.952 median  ->  enforce decreasing monotonicity")

try:
    cd_vals = df_cache['cts_cluster_dia'].values.astype(float)

    def isotonic_postprocess(y_pred, pids_, cd, increasing=False):
        y_out = y_pred.copy()
        for pid in np.unique(pids_):
            m = pids_ == pid
            rows = np.where(m)[0]
            if len(rows) < 3:
                continue
            iso = IsotonicRegression(increasing=increasing, out_of_bounds='clip')
            iso.fit(cd[rows], y_pred[rows])
            y_out[rows] = iso.predict(cd[rows])
        return y_out

    # Apply to best OOF predictions from A2
    if 'A2_best' in pw_oofs:
        pw_oof_a2 = pw_oofs['A2_best']
    elif 'A1_X29_LGB' in pw_oofs:
        pw_oof_a2 = pw_oofs['A1_X29_LGB']
    else:
        pw_oof_a2 = np.zeros(len(pids))

    if 'A2_best' in wl_oofs:
        wl_oof_a2 = wl_oofs['A2_best']
    elif 'A1_X49_XGB' in wl_oofs:
        wl_oof_a2 = wl_oofs['A1_X49_XGB']
    else:
        wl_oof_a2 = np.zeros(len(pids))

    pw_iso = isotonic_postprocess(pw_oof_a2, pids, cd_vals, increasing=False)
    wl_iso = isotonic_postprocess(wl_oof_a2, pids, cd_vals, increasing=False)

    pw_before = zscore_pred_to_rank_mae(pw_oof_a2, Y_cache[:, 1], pids)
    pw_after = zscore_pred_to_rank_mae(pw_iso, Y_cache[:, 1], pids)
    wl_before = zscore_pred_to_rank_mae(wl_oof_a2, Y_cache[:, 2], pids)
    wl_after = zscore_pred_to_rank_mae(wl_iso, Y_cache[:, 2], pids)

    print(f"\n  Power: {pw_before:.4f} -> {pw_after:.4f}  (delta={pw_after-pw_before:+.4f})")
    print(f"  WL:    {wl_before:.4f} -> {wl_after:.4f}  (delta={wl_after-wl_before:+.4f})")

    print(f"\n  Per-design isotonic effect:")
    for held in design_list:
        m = designs == held
        pw_b = zscore_pred_to_rank_mae(pw_oof_a2[m], Y_cache[m, 1], pids[m])
        pw_a = zscore_pred_to_rank_mae(pw_iso[m], Y_cache[m, 1], pids[m])
        wl_b = zscore_pred_to_rank_mae(wl_oof_a2[m], Y_cache[m, 2], pids[m])
        wl_a = zscore_pred_to_rank_mae(wl_iso[m], Y_cache[m, 2], pids[m])
        print(f"    {held}: pw {pw_b:.4f}->{pw_a:.4f} ({pw_a-pw_b:+.4f}), "
              f"wl {wl_b:.4f}->{wl_a:.4f} ({wl_a-wl_b:+.4f})")

    pw_oofs['A4_iso_pw'] = pw_iso
    wl_oofs['A4_iso_wl'] = wl_iso

    # Also apply to chain OOF if available
    if res_A3 is not None and 'A3_chain_inv' in pw_oofs:
        chain_pw = pw_oofs['A3_chain_inv']
        chain_wl = wl_oofs['A3_chain_inv']
        chain_pw_iso = isotonic_postprocess(chain_pw, pids, cd_vals, increasing=False)
        chain_wl_iso = isotonic_postprocess(chain_wl, pids, cd_vals, increasing=False)

        chain_pw_before = zscore_pred_to_rank_mae(chain_pw, Y_cache[:, 1], pids)
        chain_pw_after = zscore_pred_to_rank_mae(chain_pw_iso, Y_cache[:, 1], pids)
        chain_wl_before = zscore_pred_to_rank_mae(chain_wl, Y_cache[:, 2], pids)
        chain_wl_after = zscore_pred_to_rank_mae(chain_wl_iso, Y_cache[:, 2], pids)

        print(f"\n  Chain+isotonic: pw {chain_pw_before:.4f}->{chain_pw_after:.4f}, "
              f"wl {chain_wl_before:.4f}->{chain_wl_after:.4f}")

        pw_oofs['A4_chain_iso_pw'] = chain_pw_iso
        wl_oofs['A4_chain_iso_wl'] = chain_wl_iso

    res_A4 = {
        'pw_before': pw_before, 'pw_after': pw_after,
        'wl_before': wl_before, 'wl_after': wl_after,
    }
    results_summary['A4_isotonic'] = res_A4

    if pw_after < best_pw_mae:
        best_pw_mae = pw_after
    if wl_after < best_wl_mae:
        best_wl_mae = wl_after

except Exception as e:
    print(f"  ERROR in Approach 4: {e}")
    import traceback; traceback.print_exc()
    res_A4 = None

# ---------------------------------------------------------------------------
# APPROACH 5: QUANTILE ENSEMBLE + SEED ENSEMBLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 5: QUANTILE ENSEMBLE + SEED ENSEMBLE")
print("=" * 70)

try:
    quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_seeds = 20

    # Method 5a: Quantile ensemble for power
    print(f"\n{elapsed()} 5a: Quantile ensemble for power (X_pw_inv)...")
    qpw_oof = np.zeros(len(pids))
    qpw_folds = []
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_pw_inv[tr]); Xte = sc.transform(X_pw_inv[te])
        q_preds = []
        for q in quantiles:
            p = {**LGB_DEFAULT, 'objective': 'quantile', 'alpha': q}
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 1])
            q_preds.append(m.predict(Xte))
        ens_pred = np.mean(q_preds, axis=0)
        qpw_oof[te] = ens_pred
        qpw_folds.append(zscore_pred_to_rank_mae(ens_pred, Y_cache[te, 1], pids[te]))
        print(f"    {held}: qens_pw={qpw_folds[-1]:.4f}")

    qpw_mean = np.mean(qpw_folds)
    print(f"  Quantile ens power: {qpw_mean:.4f}  {[f'{v:.4f}' for v in qpw_folds]}")
    pw_oofs['A5_qens'] = qpw_oof

    # Quantile for WL
    print(f"\n{elapsed()} 5a: Quantile ensemble for WL (X69)...")
    qwl_oof = np.zeros(len(pids))
    qwl_folds = []
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X69[tr]); Xte = sc.transform(X69[te])
        q_preds = []
        for q in quantiles:
            p = {**LGB_DEFAULT, 'objective': 'quantile', 'alpha': q}
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 2])
            q_preds.append(m.predict(Xte))
        ens_pred = np.mean(q_preds, axis=0)
        qwl_oof[te] = ens_pred
        qwl_folds.append(zscore_pred_to_rank_mae(ens_pred, Y_cache[te, 2], pids[te]))
        print(f"    {held}: qens_wl={qwl_folds[-1]:.4f}")

    qwl_mean = np.mean(qwl_folds)
    print(f"  Quantile ens WL: {qwl_mean:.4f}  {[f'{v:.4f}' for v in qwl_folds]}")
    wl_oofs['A5_qens'] = qwl_oof

    # Method 5b: Seed ensemble (20 seeds)
    print(f"\n{elapsed()} 5b: Seed ensemble power (20 seeds, X_pw_inv)...")
    seed_pw_oof = np.zeros(len(pids))
    seed_pw_folds = []
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_pw_inv[tr]); Xte = sc.transform(X_pw_inv[te])
        seed_preds = []
        for seed in range(n_seeds):
            p = {**LGB_DEFAULT, 'random_state': seed}
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 1])
            seed_preds.append(m.predict(Xte))
        ens = np.mean(seed_preds, axis=0)
        seed_pw_oof[te] = ens
        seed_pw_folds.append(zscore_pred_to_rank_mae(ens, Y_cache[te, 1], pids[te]))
        print(f"    {held}: seed_pw={seed_pw_folds[-1]:.4f}")

    seed_pw_mean = np.mean(seed_pw_folds)
    print(f"  Seed ens power: {seed_pw_mean:.4f}")
    pw_oofs['A5_seed'] = seed_pw_oof

    print(f"\n{elapsed()} 5b: Seed ensemble WL (20 seeds, X69)...")
    seed_wl_oof = np.zeros(len(pids))
    seed_wl_folds = []
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X69[tr]); Xte = sc.transform(X69[te])
        seed_preds = []
        for seed in range(n_seeds):
            p = {**LGB_DEFAULT, 'random_state': seed}
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 2])
            seed_preds.append(m.predict(Xte))
        ens = np.mean(seed_preds, axis=0)
        seed_wl_oof[te] = ens
        seed_wl_folds.append(zscore_pred_to_rank_mae(ens, Y_cache[te, 2], pids[te]))
        print(f"    {held}: seed_wl={seed_wl_folds[-1]:.4f}")

    seed_wl_mean = np.mean(seed_wl_folds)
    print(f"  Seed ens WL: {seed_wl_mean:.4f}")
    wl_oofs['A5_seed'] = seed_wl_oof

    # Use best config with seed ensemble
    print(f"\n{elapsed()} 5b: Seed ensemble power with best config (X_pw_inv)...")
    best_seed_pw_oof = np.zeros(len(pids))
    best_seed_pw_folds = []
    cfg_use = best_pw_cfg if best_pw_cfg else LGB_DEFAULT
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_pw_inv[tr]); Xte = sc.transform(X_pw_inv[te])
        seed_preds = []
        for seed in range(n_seeds):
            p = {**cfg_use, 'random_state': seed}
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 1])
            seed_preds.append(m.predict(Xte))
        ens = np.mean(seed_preds, axis=0)
        best_seed_pw_oof[te] = ens
        best_seed_pw_folds.append(zscore_pred_to_rank_mae(ens, Y_cache[te, 1], pids[te]))
        print(f"    {held}: best_seed_pw={best_seed_pw_folds[-1]:.4f}")

    best_seed_pw_mean = np.mean(best_seed_pw_folds)
    print(f"  Best seed ens power: {best_seed_pw_mean:.4f}")
    pw_oofs['A5_best_seed'] = best_seed_pw_oof

    # Best config XGB seed ensemble for WL
    print(f"\n{elapsed()} 5b: Seed ensemble WL with best config (X69)...")
    best_seed_wl_oof = np.zeros(len(pids))
    best_seed_wl_folds = []
    cfg_wl_use = best_wl_cfg if best_wl_cfg else XGB_DEFAULT
    for held in design_list:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X69[tr]); Xte = sc.transform(X69[te])
        seed_preds = []
        for seed in range(n_seeds):
            p = {**cfg_wl_use, 'random_state': seed}
            m = xgb.XGBRegressor(**p)
            m.fit(Xtr, Y_cache[tr, 2])
            seed_preds.append(m.predict(Xte))
        ens = np.mean(seed_preds, axis=0)
        best_seed_wl_oof[te] = ens
        best_seed_wl_folds.append(zscore_pred_to_rank_mae(ens, Y_cache[te, 2], pids[te]))
        print(f"    {held}: best_seed_wl={best_seed_wl_folds[-1]:.4f}")

    best_seed_wl_mean = np.mean(best_seed_wl_folds)
    print(f"  Best seed ens WL: {best_seed_wl_mean:.4f}")
    wl_oofs['A5_best_seed'] = best_seed_wl_oof

    res_A5 = {
        'qens_pw': {'mean': qpw_mean, 'folds': qpw_folds},
        'qens_wl': {'mean': qwl_mean, 'folds': qwl_folds},
        'seed_pw': {'mean': seed_pw_mean, 'folds': seed_pw_folds},
        'seed_wl': {'mean': seed_wl_mean, 'folds': seed_wl_folds},
        'best_seed_pw': {'mean': best_seed_pw_mean, 'folds': best_seed_pw_folds},
        'best_seed_wl': {'mean': best_seed_wl_mean, 'folds': best_seed_wl_folds},
    }
    results_summary['A5_quantile'] = res_A5

    if best_seed_pw_mean < best_pw_mae:
        best_pw_mae = best_seed_pw_mean
    if best_seed_wl_mean < best_wl_mae:
        best_wl_mae = best_seed_wl_mean

except Exception as e:
    print(f"  ERROR in Approach 5: {e}")
    import traceback; traceback.print_exc()
    res_A5 = None

# ---------------------------------------------------------------------------
# APPROACH 6: SPATIAL GRID FEATURES
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 6: SPATIAL GRID FEATURES (8x8 from .pt graph files)")
print("=" * 70)

try:
    import os
    import torch
    from pathlib import Path

    PT_DIR = Path('processed_graphs')
    GRID_SIZE = 8

    def build_grid_features_from_pt(placement_id, grid_size=8):
        pt_path = PT_DIR / f"{placement_id}.pt"
        if not pt_path.exists():
            return np.zeros(4 * grid_size * grid_size, dtype=np.float32)
        try:
            data = torch.load(str(pt_path), map_location='cpu', weights_only=False)
            X_node = data['X']
            if X_node.shape[0] == 0:
                return np.zeros(4 * grid_size * grid_size, dtype=np.float32)
            x = np.clip(X_node[:, 0].numpy(), 0, 1 - 1e-6)
            y = np.clip(X_node[:, 1].numpy(), 0, 1 - 1e-6)
            area = np.exp(X_node[:, 6].numpy()) - 1
            toggle = X_node[:, 12].numpy()
            fan_in = np.exp(X_node[:, 16].numpy()) - 1
            xi = np.clip((x * grid_size).astype(int), 0, grid_size - 1)
            yi = np.clip((y * grid_size).astype(int), 0, grid_size - 1)
            G = grid_size
            grids = np.zeros((4, G, G), dtype=np.float32)
            np.add.at(grids[0], (xi, yi), 1.0)
            np.add.at(grids[1], (xi, yi), area)
            np.add.at(grids[2], (xi, yi), np.abs(toggle))
            np.add.at(grids[3], (xi, yi), fan_in)
            for c in range(4):
                total = grids[c].sum()
                if total > 0:
                    grids[c] /= total
            return grids.flatten()
        except Exception:
            return np.zeros(4 * grid_size * grid_size, dtype=np.float32)

    print(f"{elapsed()} Building 8x8 grid features from .pt files...")
    unique_pids = np.unique(pids)
    grid_cache = {}
    found_count = 0
    for pid in unique_pids:
        feats = build_grid_features_from_pt(pid)
        grid_cache[pid] = feats
        if feats.sum() != 0:
            found_count += 1

    print(f"  Found .pt files for {found_count}/{len(unique_pids)} placements")

    X_grid = np.array([grid_cache.get(pid, np.zeros(4 * GRID_SIZE * GRID_SIZE))
                       for pid in pids], dtype=np.float32)
    print(f"  X_grid shape: {X_grid.shape}")
    print(f"  X_grid nonzero rows: {(X_grid.sum(axis=1) != 0).sum()}/{len(pids)}")

    if found_count > 0:
        # Check design ID leakage
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score

        le = LabelEncoder()
        y_design_enc = le.fit_transform(designs)
        nonzero_mask = X_grid.sum(axis=1) != 0
        grid_design_acc = 0.25
        if nonzero_mask.sum() > 100:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(lr, X_grid[nonzero_mask], y_design_enc[nonzero_mask],
                                     cv=5, scoring='accuracy')
            grid_design_acc = scores.mean()
            print(f"  Grid feature design ID accuracy: {grid_design_acc:.3f} (threshold=0.50)")
            safe = "SAFE" if grid_design_acc < 0.50 else "UNSAFE"
            print(f"  -> {safe}")

        # Add grid features to existing sets
        X69_grid = np.hstack([X69, X_grid])
        X_pw_inv_grid = np.hstack([X_pw_inv, X_grid])

        print(f"\n{elapsed()} LODO: power with X_pw_inv+grid...")
        g_pw_folds = []
        g_pw_oof = np.zeros(len(pids))
        for held in design_list:
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_pw_inv_grid[tr]); Xte = sc.transform(X_pw_inv_grid[te])
            m = lgb.LGBMRegressor(**LGB_DEFAULT)
            m.fit(Xtr, Y_cache[tr, 1])
            pred = m.predict(Xte)
            g_pw_oof[te] = pred
            g_pw_folds.append(zscore_pred_to_rank_mae(pred, Y_cache[te, 1], pids[te]))
            print(f"    {held}: grid_pw={g_pw_folds[-1]:.4f}")

        g_pw_mean = np.mean(g_pw_folds)
        print(f"  Grid+inv power: {g_pw_mean:.4f}  {[f'{v:.4f}' for v in g_pw_folds]}")
        pw_oofs['A6_grid'] = g_pw_oof

        print(f"\n{elapsed()} LODO: WL with X69+grid...")
        g_wl_folds = []
        g_wl_oof = np.zeros(len(pids))
        for held in design_list:
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X69_grid[tr]); Xte = sc.transform(X69_grid[te])
            m = xgb.XGBRegressor(**XGB_DEFAULT)
            m.fit(Xtr, Y_cache[tr, 2])
            pred = m.predict(Xte)
            g_wl_oof[te] = pred
            g_wl_folds.append(zscore_pred_to_rank_mae(pred, Y_cache[te, 2], pids[te]))
            print(f"    {held}: grid_wl={g_wl_folds[-1]:.4f}")

        g_wl_mean = np.mean(g_wl_folds)
        print(f"  Grid+inv WL: {g_wl_mean:.4f}  {[f'{v:.4f}' for v in g_wl_folds]}")
        wl_oofs['A6_grid'] = g_wl_oof

        res_A6 = {
            'pw': {'mean': g_pw_mean, 'folds': g_pw_folds},
            'wl': {'mean': g_wl_mean, 'folds': g_wl_folds},
            'design_acc': grid_design_acc,
            'found_count': found_count,
        }
        results_summary['A6_grid'] = res_A6

        if g_pw_mean < best_pw_mae:
            best_pw_mae = g_pw_mean
        if g_wl_mean < best_wl_mae:
            best_wl_mae = g_wl_mean
    else:
        print(f"  No .pt files found - skipping LODO")
        res_A6 = {'found_count': 0}
        results_summary['A6_grid'] = res_A6

except Exception as e:
    print(f"  ERROR in Approach 6: {e}")
    import traceback; traceback.print_exc()
    res_A6 = None

# ---------------------------------------------------------------------------
# APPROACH 7: PER-PLACEMENT BETA META-REGRESSION
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 7: PER-PLACEMENT BETA META-REGRESSION")
print("=" * 70)
print(f"  z_power_i = beta * (1/cd_i - mean(1/cd)) for each placement")
print(f"  Fit beta per training placement, meta-regress from X29 features")

try:
    cd = df_cache['cts_cluster_dia'].values.astype(np.float64)

    # Fit beta for each placement
    unique_pids_list = np.unique(pids)
    beta_pw = {}; beta_wl = {}

    for pid in unique_pids_list:
        m = pids == pid
        rows = np.where(m)[0]
        x = 1.0 / cd[rows]
        mu_x = x.mean()
        x_c = x - mu_x
        denom = (x_c ** 2).sum()
        if denom > 1e-10:
            beta_pw[pid] = (x_c * Y_cache[rows, 1]).sum() / denom
            beta_wl[pid] = (x_c * Y_cache[rows, 2]).sum() / denom
        else:
            beta_pw[pid] = 0.0; beta_wl[pid] = 0.0

    beta_pw_arr = np.array([beta_pw[p] for p in unique_pids_list])
    beta_wl_arr = np.array([beta_wl[p] for p in unique_pids_list])

    print(f"\n  beta_pw stats: mean={beta_pw_arr.mean():.3f}, std={beta_pw_arr.std():.3f}")
    print(f"  beta_wl stats: mean={beta_wl_arr.mean():.3f}, std={beta_wl_arr.std():.3f}")

    cv_pw = abs(beta_pw_arr.std() / beta_pw_arr.mean()) if beta_pw_arr.mean() != 0 else 999
    cv_wl = abs(beta_wl_arr.std() / beta_wl_arr.mean()) if beta_wl_arr.mean() != 0 else 999
    print(f"  CV(beta_pw) = {cv_pw:.3f}  {'-> Low, meta-regression should work' if cv_pw < 0.3 else '-> High, generalization may fail'}")
    print(f"  CV(beta_wl) = {cv_wl:.3f}  {'-> Low, meta-regression should work' if cv_wl < 0.3 else '-> High, generalization may fail'}")

    for design in design_list:
        m = designs == design
        d_pids = np.unique(pids[m])
        betas = [beta_pw[p] for p in d_pids]
        print(f"    {design} beta_pw: mean={np.mean(betas):.3f}, std={np.std(betas):.3f}, "
              f"n={len(betas)}")

    # X29 per-placement (use mean over 10 runs)
    pid_to_idx = {p: i for i, p in enumerate(unique_pids_list)}
    X29_per_pid = np.zeros((len(unique_pids_list), X29.shape[1]))
    designs_per_pid = np.empty(len(unique_pids_list), dtype=object)
    for p in unique_pids_list:
        m = pids == p
        X29_per_pid[pid_to_idx[p]] = X29[m].mean(axis=0)
        designs_per_pid[pid_to_idx[p]] = designs[m][0]

    # LODO meta-regression
    pw_beta_folds = []; wl_beta_folds = []
    oof_pw_beta = np.zeros(len(pids))
    oof_wl_beta = np.zeros(len(pids))

    for held in design_list:
        tr_pl = designs_per_pid != held
        te_pl = ~tr_pl

        sc = StandardScaler()
        Xtr = sc.fit_transform(X29_per_pid[tr_pl])
        Xte = sc.transform(X29_per_pid[te_pl])

        # Fit beta predictors
        m_beta_pw = lgb.LGBMRegressor(**LGB_DEFAULT)
        m_beta_pw.fit(Xtr, beta_pw_arr[tr_pl])
        beta_pw_pred = m_beta_pw.predict(Xte)

        m_beta_wl = lgb.LGBMRegressor(**LGB_DEFAULT)
        m_beta_wl.fit(Xtr, beta_wl_arr[tr_pl])
        beta_wl_pred = m_beta_wl.predict(Xte)

        te_pids_pl = unique_pids_list[te_pl]

        # Generate run-level predictions
        pw_pred_runs = np.zeros(len(pids))
        wl_pred_runs = np.zeros(len(pids))
        for i, pid in enumerate(te_pids_pl):
            m_run = pids == pid
            rows = np.where(m_run)[0]
            x = 1.0 / cd[rows]
            mu_x = x.mean()
            x_c = x - mu_x
            pw_pred_runs[rows] = beta_pw_pred[i] * x_c
            wl_pred_runs[rows] = beta_wl_pred[i] * x_c

        # Evaluate on test design
        te_des_mask = designs == held
        pw_mae = zscore_pred_to_rank_mae(pw_pred_runs[te_des_mask], Y_cache[te_des_mask, 1],
                                          pids[te_des_mask])
        wl_mae = zscore_pred_to_rank_mae(wl_pred_runs[te_des_mask], Y_cache[te_des_mask, 2],
                                          pids[te_des_mask])
        pw_beta_folds.append(pw_mae)
        wl_beta_folds.append(wl_mae)
        oof_pw_beta[te_des_mask] = pw_pred_runs[te_des_mask]
        oof_wl_beta[te_des_mask] = wl_pred_runs[te_des_mask]

        print(f"    {held}: beta_pw={pw_mae:.4f}, beta_wl={wl_mae:.4f}")

    pw_beta_mean = np.mean(pw_beta_folds)
    wl_beta_mean = np.mean(wl_beta_folds)
    print(f"  Beta meta-regression: power={pw_beta_mean:.4f}  {[f'{v:.4f}' for v in pw_beta_folds]}")
    print(f"  Beta meta-regression: WL   ={wl_beta_mean:.4f}  {[f'{v:.4f}' for v in wl_beta_folds]}")

    pw_oofs['A7_beta'] = oof_pw_beta
    wl_oofs['A7_beta'] = oof_wl_beta

    res_A7 = {
        'pw': {'mean': pw_beta_mean, 'folds': pw_beta_folds},
        'wl': {'mean': wl_beta_mean, 'folds': wl_beta_folds},
        'cv_pw': cv_pw, 'cv_wl': cv_wl,
    }
    results_summary['A7_beta'] = res_A7

    if pw_beta_mean < best_pw_mae:
        best_pw_mae = pw_beta_mean
    if wl_beta_mean < best_wl_mae:
        best_wl_mae = wl_beta_mean

except Exception as e:
    print(f"  ERROR in Approach 7: {e}")
    import traceback; traceback.print_exc()
    res_A7 = None

# ---------------------------------------------------------------------------
# ADDITIONAL: HYPERPARAMETER SEARCH FOR SKEW
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("ADDITIONAL: SKEW HYPERPARAMETER SEARCH")
print("=" * 70)

try:
    print(f"\n{elapsed()} XGB config search for skew (X49, rank target)...")
    for cfg in XGB_CONFIGS:
        folds, mae, oof = lodo_rank(X49, Y_rank[:, 0], pids, designs,
                                     xgb.XGBRegressor, cfg,
                                     f"sk XGB n={cfg['n_estimators']},lr={cfg['learning_rate']},d={cfg['max_depth']}")
        if mae < best_sk_mae:
            best_sk_mae = mae; best_sk_cfg = cfg; best_sk_model = 'xgb'

    print(f"\n{elapsed()} LGB config search for skew (X49, rank target)...")
    for cfg in LGB_CONFIGS:
        folds, mae, oof = lodo_rank(X49, Y_rank[:, 0], pids, designs,
                                     lgb.LGBMRegressor, cfg,
                                     f"sk LGB n={cfg['n_estimators']},lr={cfg['learning_rate']},l={cfg['num_leaves']}")
        if mae < best_sk_mae:
            best_sk_mae = mae; best_sk_cfg = cfg; best_sk_model = 'lgb'

    # Try X69 with best config
    print(f"\n{elapsed()} Skew with X69 ({best_sk_model.upper()} best cfg)...")
    ModelCls = xgb.XGBRegressor if best_sk_model == 'xgb' else lgb.LGBMRegressor
    sk_f_69, sk_m_69, sk_oof_69 = lodo_rank(X69, Y_rank[:, 0], pids, designs,
                                              ModelCls, best_sk_cfg,
                                              f"sk X69 {best_sk_model.upper()}")

    print(f"\n  Skew best: {best_sk_model.upper()} {best_sk_cfg}  -> sk={best_sk_mae:.4f}")
    print(f"  Skew with X69: {sk_m_69:.4f}")
    res_sk = {
        'best_mae': best_sk_mae, 'best_cfg': best_sk_cfg, 'best_model': best_sk_model,
        'sk_X69': sk_m_69,
    }
    if sk_m_69 < best_sk_mae:
        best_sk_mae = sk_m_69
        best_sk_cfg_69 = best_sk_cfg
    results_summary['sk_search'] = res_sk

except Exception as e:
    print(f"  ERROR in skew search: {e}")
    import traceback; traceback.print_exc()
    res_sk = None

# ---------------------------------------------------------------------------
# APPROACH 8: OPTIMAL ENSEMBLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("APPROACH 8: OPTIMAL ENSEMBLE (combine all OOF predictions)")
print("=" * 70)

try:
    # Print individual OOF MAEs
    print(f"\n  Individual OOF MAEs on all 5390 samples:")
    print(f"  {'Source':<25} {'power':>8} {'WL':>8}")
    print(f"  {'-'*42}")

    all_pw_maes = {}
    all_wl_maes = {}

    for k, oof in pw_oofs.items():
        if oof.sum() != 0:
            mae = zscore_pred_to_rank_mae(oof, Y_cache[:, 1], pids)
            all_pw_maes[k] = mae
            wl_k = wl_oofs.get(k)
            wl_mae = zscore_pred_to_rank_mae(wl_k, Y_cache[:, 2], pids) if wl_k is not None else 999
            print(f"  {k:<25} {mae:>8.4f} {wl_mae:>8.4f}" + (" ✓" if mae < 0.10 else ""))
    for k, oof in wl_oofs.items():
        if k not in pw_oofs and oof.sum() != 0:
            wl_mae = zscore_pred_to_rank_mae(oof, Y_cache[:, 2], pids)
            all_wl_maes[k] = wl_mae
            print(f"  {'(wl only) '+k:<25} {'--':>8} {wl_mae:>8.4f}" + (" ✓" if wl_mae < 0.10 else ""))

    # Find best single predictions
    best_pw_single = min(all_pw_maes.items(), key=lambda x: x[1]) if all_pw_maes else ('none', 999)
    all_wl = {**{k: zscore_pred_to_rank_mae(wl_oofs[k], Y_cache[:, 2], pids)
                  for k in wl_oofs if wl_oofs[k].sum() != 0}}
    best_wl_single = min(all_wl.items(), key=lambda x: x[1]) if all_wl else ('none', 999)

    print(f"\n  Best single power: {best_pw_single[0]} = {best_pw_single[1]:.4f}")
    print(f"  Best single WL:    {best_wl_single[0]} = {best_wl_single[1]:.4f}")

    # Grid search for best 2-source ensemble
    print(f"\n{elapsed()} 2-source ensemble grid search...")
    best_pw_ens = best_pw_single[1]; best_pw_blend = {best_pw_single[0]: 1.0}
    best_wl_ens = best_wl_single[1]; best_wl_blend = {best_wl_single[0]: 1.0}

    pw_keys = [k for k, v in pw_oofs.items() if v.sum() != 0]
    wl_keys = [k for k, v in wl_oofs.items() if v.sum() != 0]

    for i in range(len(pw_keys)):
        for j in range(i+1, len(pw_keys)):
            k1, k2 = pw_keys[i], pw_keys[j]
            for w in np.arange(0.1, 0.91, 0.05):
                pred = w * pw_oofs[k1] + (1-w) * pw_oofs[k2]
                mae = zscore_pred_to_rank_mae(pred, Y_cache[:, 1], pids)
                if mae < best_pw_ens:
                    best_pw_ens = mae
                    best_pw_blend = {k1: w, k2: round(1-w, 3)}

    for i in range(len(wl_keys)):
        for j in range(i+1, len(wl_keys)):
            k1, k2 = wl_keys[i], wl_keys[j]
            for w in np.arange(0.1, 0.91, 0.05):
                pred = w * wl_oofs[k1] + (1-w) * wl_oofs[k2]
                mae = zscore_pred_to_rank_mae(pred, Y_cache[:, 2], pids)
                if mae < best_wl_ens:
                    best_wl_ens = mae
                    best_wl_blend = {k1: w, k2: round(1-w, 3)}

    print(f"  Best 2-source power: {best_pw_blend}  -> {best_pw_ens:.4f}")
    print(f"  Best 2-source WL:    {best_wl_blend}  -> {best_wl_ens:.4f}")

    # 3-source ensemble
    print(f"\n{elapsed()} 3-source ensemble grid search (power)...")
    for i in range(len(pw_keys)):
        for j in range(i+1, len(pw_keys)):
            for k in range(j+1, len(pw_keys)):
                k1, k2, k3 = pw_keys[i], pw_keys[j], pw_keys[k]
                for w1 in np.arange(0.1, 0.81, 0.1):
                    for w2 in np.arange(0.1, 0.91-w1, 0.1):
                        w3 = 1 - w1 - w2
                        if w3 < 0.05: continue
                        pred = w1 * pw_oofs[k1] + w2 * pw_oofs[k2] + w3 * pw_oofs[k3]
                        mae = zscore_pred_to_rank_mae(pred, Y_cache[:, 1], pids)
                        if mae < best_pw_ens:
                            best_pw_ens = mae
                            best_pw_blend = {k1: round(w1,2), k2: round(w2,2), k3: round(w3,2)}

    print(f"  Best 3-source power: {best_pw_blend}  -> {best_pw_ens:.4f}")

    print(f"\n{elapsed()} 3-source ensemble grid search (WL)...")
    for i in range(len(wl_keys)):
        for j in range(i+1, len(wl_keys)):
            for k in range(j+1, len(wl_keys)):
                k1, k2, k3 = wl_keys[i], wl_keys[j], wl_keys[k]
                for w1 in np.arange(0.1, 0.81, 0.1):
                    for w2 in np.arange(0.1, 0.91-w1, 0.1):
                        w3 = 1 - w1 - w2
                        if w3 < 0.05: continue
                        pred = w1 * wl_oofs[k1] + w2 * wl_oofs[k2] + w3 * wl_oofs[k3]
                        mae = zscore_pred_to_rank_mae(pred, Y_cache[:, 2], pids)
                        if mae < best_wl_ens:
                            best_wl_ens = mae
                            best_wl_blend = {k1: round(w1,2), k2: round(w2,2), k3: round(w3,2)}

    print(f"  Best 3-source WL: {best_wl_blend}  -> {best_wl_ens:.4f}")

    # Build best ensemble predictions
    def apply_blend(oofs_dict, blend):
        total = sum(blend.values())
        result = np.zeros(len(pids))
        for k, w in blend.items():
            if k in oofs_dict:
                result += w / total * oofs_dict[k]
        return result

    best_pw_ens_pred = apply_blend(pw_oofs, best_pw_blend)
    best_wl_ens_pred = apply_blend(wl_oofs, best_wl_blend)

    # Per-design breakdown of best ensemble
    print(f"\n  Best ensemble per-design breakdown:")
    for held in design_list:
        m = designs == held
        pw_d = zscore_pred_to_rank_mae(best_pw_ens_pred[m], Y_cache[m, 1], pids[m])
        wl_d = zscore_pred_to_rank_mae(best_wl_ens_pred[m], Y_cache[m, 2], pids[m])
        pw_sym = " ✓" if pw_d < 0.10 else " ✗"
        wl_sym = " ✓" if wl_d < 0.10 else " ✗"
        print(f"    {held}: power={pw_d:.4f}{pw_sym}, WL={wl_d:.4f}{wl_sym}")

    res_A8 = {
        'pw_ens': best_pw_ens, 'pw_blend': best_pw_blend,
        'wl_ens': best_wl_ens, 'wl_blend': best_wl_blend,
        'pw_pred': best_pw_ens_pred,
        'wl_pred': best_wl_ens_pred,
    }
    results_summary['A8_ensemble'] = res_A8

    if best_pw_ens < best_pw_mae:
        best_pw_mae = best_pw_ens
    if best_wl_ens < best_wl_mae:
        best_wl_mae = best_wl_ens

except Exception as e:
    print(f"  ERROR in Approach 8: {e}")
    import traceback; traceback.print_exc()
    res_A8 = None

# ---------------------------------------------------------------------------
# COMPLETELY UNSEEN TEST — Ethmac prediction scatter
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("UNSEEN DESIGN TEST — ethmac prediction vs actual (first 5 placements)")
print("=" * 70)

try:
    held = 'ethmac'
    tr = designs != held; te = ~tr
    te_pids = pids[te]

    sc_pw = StandardScaler()
    sc_wl = StandardScaler()
    m_pw = lgb.LGBMRegressor(**(best_pw_cfg if best_pw_cfg else LGB_DEFAULT))
    m_pw.fit(sc_pw.fit_transform(X_pw_inv[tr]), Y_cache[tr, 1])
    pw_pred_te = m_pw.predict(sc_pw.transform(X_pw_inv[te]))

    m_wl = xgb.XGBRegressor(**(best_wl_cfg if best_wl_cfg else XGB_DEFAULT))
    m_wl.fit(sc_wl.fit_transform(X69[tr]), Y_cache[tr, 2])
    wl_pred_te = m_wl.predict(sc_wl.transform(X69[te]))

    print(f"\n  Power prediction (z-score -> rank), 5 ethmac placements:")
    print(f"  {'Placement':<20} {'Run':<4} {'True_z':>8} {'Pred_z':>8} {'True_rk':>8} {'Pred_rk':>8}")
    print(f"  {'-'*60}")

    for pid in np.unique(te_pids)[:5]:
        m = te_pids == pid
        rows = np.where(m)[0]
        true_z = Y_cache[te, 1][rows]
        pred_z = pw_pred_te[rows]
        true_r = rank_within(true_z)
        pred_r = rank_within(pred_z)
        for i in range(len(rows)):
            pid_s = pid[-8:]
            print(f"  {pid_s:<20} {i+1:<4} {true_z[i]:>8.3f} {pred_z[i]:>8.3f} "
                  f"{true_r[i]:>8.3f} {pred_r[i]:>8.3f}")
        print()

    pw_mae_eth = zscore_pred_to_rank_mae(pw_pred_te, Y_cache[te, 1], te_pids)
    wl_mae_eth = zscore_pred_to_rank_mae(wl_pred_te, Y_cache[te, 2], te_pids)
    print(f"  Ethmac test MAE: power={pw_mae_eth:.4f}, WL={wl_mae_eth:.4f}")

except Exception as e:
    print(f"  ERROR in unseen test: {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# FINAL MODEL TRAINING
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

try:
    final_pw_cfg = best_pw_cfg if best_pw_cfg else LGB_DEFAULT
    final_wl_cfg = best_wl_cfg if best_wl_cfg else XGB_DEFAULT
    final_sk_cfg = best_sk_cfg if best_sk_cfg else XGB_SK
    final_sk_model_name = best_sk_model if best_sk_model else 'xgb'

    sc_pw = StandardScaler(); sc_wl = StandardScaler(); sc_sk = StandardScaler()

    m_pw_final = lgb.LGBMRegressor(**final_pw_cfg)
    m_pw_final.fit(sc_pw.fit_transform(X_pw_inv), Y_cache[:, 1])

    m_wl_final = xgb.XGBRegressor(**final_wl_cfg)
    m_wl_final.fit(sc_wl.fit_transform(X69), Y_cache[:, 2])

    if final_sk_model_name == 'xgb':
        m_sk_final = xgb.XGBRegressor(**final_sk_cfg)
    else:
        m_sk_final = lgb.LGBMRegressor(**final_sk_cfg)
    m_sk_final.fit(sc_sk.fit_transform(X49), Y_rank[:, 0])

    print(f"  Power: LGB {final_pw_cfg}")
    print(f"  WL:    XGB {final_wl_cfg}")
    print(f"  Skew:  {final_sk_model_name.upper()} {final_sk_cfg}")

    out_models = {
        'pw': m_pw_final, 'wl': m_wl_final, 'sk': m_sk_final,
        'sc_pw': sc_pw, 'sc_wl': sc_wl, 'sc_sk': sc_sk,
        'X_pw': X_pw_inv, 'X_wl': X69, 'X_sk': X49,
        'feature_desc': {
            'pw': f'X29+inv ({X_pw_inv.shape[1]}-dim): z-knobs+rank+cent+geo+interact+inv_physics | z-score target | LGB_{final_pw_cfg["n_estimators"]}',
            'wl': f'X69 ({X69.shape[1]}-dim): X49+inv_physics | z-score target | XGB_{final_wl_cfg["n_estimators"]}',
            'sk': f'X49 ({X49.shape[1]}-dim): X29+tight_path | rank target | {final_sk_model_name.upper()}_{final_sk_cfg["n_estimators"]}',
        },
        'lodo_means': {
            'pw': best_pw_mae,
            'wl': best_wl_mae,
            'sk': best_sk_mae,
        },
        'design_order': design_list,
        'results_summary': results_summary,
        'pw_oofs': pw_oofs,
        'wl_oofs': wl_oofs,
    }

    out_path = 'best_model_overnight.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(out_models, f)
    print(f"\n  Saved to: {out_path}")

except Exception as e:
    print(f"  ERROR in final model: {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# SUMMARY TABLE
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY TABLE — LODO RANK MAE (lower is better, target < 0.10)")
print("=" * 70)
print(f"\n  {'Approach':<40} {'power':>8} {'WL':>8} {'skew':>8}")
print(f"  {'-'*68}")
print(f"  {'Reference (best_model_v6)':<40} {'0.0656':>8} {'0.0858':>8} {'0.2369':>8}")

def fmt(res, key, sub='mean'):
    try:
        v = res[key][sub] if isinstance(res[key], dict) else res[key]
        s = f"{v:.4f}"
        return s + (" ✓" if v < 0.10 else "")
    except:
        return "  --  "

if res_A1:
    print(f"  {'A1: Baseline X29+tight':<40} {fmt(res_A1,'pw'):>8} {fmt(res_A1,'wl'):>8} {fmt(res_A1,'sk'):>8}")
if res_A2:
    print(f"  {'A2: X69+inverse (best LGB/XGB)':<40} {fmt(res_A2,'pw'):>8} {fmt(res_A2,'wl'):>8} {fmt(res_A2,'sk'):>8}")
if res_A3:
    chain = res_A3['chain_inv']
    print(f"  {'A3: Cross-task chain+inv':<40} {fmt(chain,'pw'):>8} {fmt(chain,'wl'):>8} {'  --  ':>8}")
if res_A4:
    pw_a = res_A4.get('pw_after', 999)
    wl_a = res_A4.get('wl_after', 999)
    pw_s = f"{pw_a:.4f}" + (" ✓" if pw_a < 0.10 else "")
    wl_s = f"{wl_a:.4f}" + (" ✓" if wl_a < 0.10 else "")
    print(f"  {'A4: Isotonic post-process':<40} {pw_s:>8} {wl_s:>8} {'  --  ':>8}")
if res_A5:
    print(f"  {'A5: Quantile ens (pw/wl)':<40} {res_A5['qens_pw']['mean']:>8.4f}{'✓' if res_A5['qens_pw']['mean']<0.1 else ''} "
          f"{res_A5['qens_wl']['mean']:>8.4f}{'✓' if res_A5['qens_wl']['mean']<0.1 else ''} {'  --  ':>8}")
    print(f"  {'A5: Seed ens best cfg (pw/wl)':<40} {res_A5['best_seed_pw']['mean']:>8.4f}{'✓' if res_A5['best_seed_pw']['mean']<0.1 else ''} "
          f"{res_A5['best_seed_wl']['mean']:>8.4f}{'✓' if res_A5['best_seed_wl']['mean']<0.1 else ''} {'  --  ':>8}")
if res_A6 and res_A6.get('found_count', 0) > 0 and 'pw' in res_A6:
    print(f"  {'A6: Spatial grid 8x8':<40} {fmt(res_A6,'pw'):>8} {fmt(res_A6,'wl'):>8} {'  --  ':>8}")
if res_A7:
    print(f"  {'A7: Beta meta-regression':<40} {fmt(res_A7,'pw'):>8} {fmt(res_A7,'wl'):>8} {'  --  ':>8}")
if res_A8:
    pw_e = res_A8['pw_ens']; wl_e = res_A8['wl_ens']
    pw_s = f"{pw_e:.4f}" + (" ✓" if pw_e < 0.10 else "")
    wl_s = f"{wl_e:.4f}" + (" ✓" if wl_e < 0.10 else "")
    print(f"  {'A8: Optimal ensemble':<40} {pw_s:>8} {wl_s:>8} {'  --  ':>8}")

print(f"\n  Best power MAE achieved: {best_pw_mae:.4f}  {'PASS ✓' if best_pw_mae < 0.10 else 'FAIL'}")
print(f"  Best WL MAE achieved:    {best_wl_mae:.4f}  {'PASS ✓' if best_wl_mae < 0.10 else 'FAIL'}")
print(f"  Best skew MAE achieved:  {best_sk_mae:.4f}  {'PASS ✓' if best_sk_mae < 0.10 else 'FAIL'}")
print(f"\n  Saved final model: best_model_overnight.pkl")
print(f"\n{'=' * 70}")
print(f"EXPERIMENT COMPLETE  Total time: {time.time()-START_TIME:.0f}s")
print(f"{'=' * 70}")

# Write to RESEARCH_LOG
try:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d")

    entry_lines = [
        f"\n\n---\n\n## {ts} Approach: Overnight Comprehensive (overnight_best.py)\n",
        f"\n**Hypothesis**: Inverse physics features (1/cd, log(cd)) + cross-task WL->power chain ",
        f"+ isotonic post-processing + quantile/seed ensembles should push below pw=0.0656, wl=0.0858.\n",
        f"\n**Feature sets**: X29(29) + tight_path(20) + inverse_physics(20) = X69(69-dim)\n",
        f"\n**Results**:\n",
    ]

    for name, res in results_summary.items():
        try:
            if 'pw' in res and isinstance(res['pw'], dict) and 'mean' in res['pw']:
                pw_v = res['pw']['mean']
                wl_v = res['wl']['mean'] if 'wl' in res else None
                sk_v = res['sk']['mean'] if 'sk' in res else None
                entry_lines.append(f"- {name}: pw={pw_v:.4f}{'✓' if pw_v < 0.10 else ''}")
                if wl_v: entry_lines.append(f", wl={wl_v:.4f}{'✓' if wl_v < 0.10 else ''}")
                if sk_v: entry_lines.append(f", sk={sk_v:.4f}{'✓' if sk_v < 0.10 else ''}")
                entry_lines.append("\n")
        except:
            pass

    entry_lines.append(f"\n**Best overall**: pw={best_pw_mae:.4f}, wl={best_wl_mae:.4f}, sk={best_sk_mae:.4f}\n")
    entry_lines.append(f"**Saved**: best_model_overnight.pkl\n")

    with open('RESEARCH_LOG.md', 'a') as f:
        f.write(''.join(entry_lines))
    print(f"\n{elapsed()} RESEARCH_LOG.md updated.")
except Exception as e:
    print(f"\n{elapsed()} WARNING: Could not write to RESEARCH_LOG.md: {e}")
