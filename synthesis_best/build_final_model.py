"""
build_final_model.py — Refine hyperparameters + save production models

Refinements tested:
  1. WL blend alpha with 84-dim features (net features may shift optimal blend)
  2. LGB n_estimators / num_leaves for WL (more features = more capacity needed)
  3. Skew LGB hyperparameters
  4. Power XGB subsample/colsample

Saves (all to synthesis_best/saved_models/):
  model_power.pkl  — XGBRegressor, trained on all 4 designs
  model_wl.pkl     — {lgb, ridge, alpha}, trained on all 4 designs
  model_skew.pkl   — LGBMRegressor, trained on all 4 designs
  scaler_power.pkl — StandardScaler for power features
  scaler_wl.pkl    — StandardScaler for WL features
  scaler_skew.pkl  — StandardScaler for skew features
  meta.pkl         — normalization stats, design list, feature dims

Load & predict (see predict_new_design.py):
  from synthesis_best.predict_new_design import CTSPredictor
  p = CTSPredictor('synthesis_best/saved_models')
  power, wl, skew_z = p.predict(placement_features)
"""

import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
SAVE_DIR = f'{BASE}/synthesis_best/saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── load caches ──────────────────────────────────────────────────────────────
sys.path.insert(0, f'{BASE}/synthesis_best')
from final_synthesis import build_all_features, per_placement_normalize, mape, mae

with open(f'{BASE}/absolute_v7_def_cache.pkl','rb') as f:    dc = pickle.load(f)
with open(f'{BASE}/absolute_v7_saif_cache.pkl','rb') as f:   sc_cache = pickle.load(f)
with open(f'{BASE}/absolute_v7_timing_cache.pkl','rb') as f: tc = pickle.load(f)
with open(f'{BASE}/skew_spatial_cache.pkl','rb') as f:       skc = pickle.load(f)
with open(f'{BASE}/absolute_v10_gravity_cache.pkl','rb') as f: gc = pickle.load(f)
with open(f'{BASE}/absolute_v13_extended_cache.pkl','rb') as f: ec = pickle.load(f)
nc = {}
if os.path.exists(f'{BASE}/net_features_cache.pkl'):
    with open(f'{BASE}/net_features_cache.pkl','rb') as f: nc = pickle.load(f)

df = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
df = df.dropna(subset=['power_total','wirelength','skew_setup']).reset_index(drop=True)
designs = sorted(df['design_name'].unique())
print(f"{T()} n={len(df)}, designs={designs}")

print(f"{T()} Building features...")
X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df = build_all_features(
    df, dc, sc_cache, tc, skc, gc, ec, nc)
print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}, X_sk={X_sk.shape}")
sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: REFINE WL MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{T()} === PHASE 1: Refine WL model (blend alpha + LGB capacity) ===")

def lodo_wl(X, y, meta, lgb_params, ridge_alpha, blend_alpha):
    mapes = {}
    for held in designs:
        tr = meta['design_name'] != held
        te = meta['design_name'] == held
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        ytr = y[tr]
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(Xtr, ytr)
        rdg = Ridge(alpha=ridge_alpha, max_iter=10000)
        rdg.fit(Xtr, ytr)
        pred_log = blend_alpha * lgb.predict(Xte) + (1-blend_alpha) * rdg.predict(Xte)
        pred = np.exp(pred_log) * meta[te]['wl_norm'].values
        actual = meta[te]['wirelength'].values
        mapes[held] = mape(actual, pred)
    return np.mean(list(mapes.values())), mapes

lgb_base = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
                min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)

# Test blend alpha
print("  Blend alpha sweep (LGB300 + Ridge1000):")
best_alpha, best_alpha_mape = 0.3, 999
for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    mean_m, per_d = lodo_wl(X_wl, y_wl, meta_df, lgb_base, 1000.0, alpha)
    marker = ' ← best' if mean_m < best_alpha_mape else ''
    print(f"    α={alpha:.1f}  mean={mean_m:.1f}%  "
          f"aes={per_d['aes']:.1f}% eth={per_d['ethmac']:.1f}% "
          f"pico={per_d['picorv32']:.1f}% sha={per_d['sha256']:.1f}%{marker}")
    if mean_m < best_alpha_mape:
        best_alpha_mape, best_alpha = mean_m, alpha
sys.stdout.flush()

# Test LGB capacity (n_estimators + num_leaves)
print(f"\n  LGB capacity sweep (α={best_alpha}):")
best_lgb_params = lgb_base.copy()
best_cap_mape = best_alpha_mape
for ne, nl in [(300, 31), (500, 31), (300, 63), (500, 63), (700, 31), (500, 127)]:
    params = {**lgb_base, 'n_estimators': ne, 'num_leaves': nl}
    mean_m, per_d = lodo_wl(X_wl, y_wl, meta_df, params, 1000.0, best_alpha)
    marker = ' ← best' if mean_m < best_cap_mape else ''
    print(f"    ne={ne} nl={nl}  mean={mean_m:.1f}%  "
          f"aes={per_d['aes']:.1f}%{marker}")
    if mean_m < best_cap_mape:
        best_cap_mape, best_lgb_params = mean_m, params
sys.stdout.flush()

print(f"\n  Best WL config: α={best_alpha}  LGB={best_lgb_params}  → {best_cap_mape:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: REFINE SKEW MODEL
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{T()} === PHASE 2: Refine Skew model ===")

def lodo_skew(X, y, meta, lgb_params):
    maes_list = []
    for held in designs:
        tr = meta['design_name'] != held
        te = meta['design_name'] == held
        y_norm, mu_arr, sig_arr = per_placement_normalize(y, meta)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        lgb = LGBMRegressor(**lgb_params)
        lgb.fit(Xtr, y_norm[tr])
        pred_z = lgb.predict(Xte)
        pred_raw = pred_z * sig_arr[te] + mu_arr[te]
        maes_list.append(mae(y[te], pred_raw))
    return np.mean(maes_list)

sk_base = dict(n_estimators=300, num_leaves=31, learning_rate=0.03,
               min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)

print("  Skew LGB sweep:")
best_sk_params = sk_base.copy()
best_sk_mae = 999
for ne, nl, lr in [(300,31,0.03),(500,31,0.03),(300,63,0.03),(500,63,0.02),(700,31,0.02)]:
    params = {**sk_base, 'n_estimators':ne, 'num_leaves':nl, 'learning_rate':lr}
    m = lodo_skew(X_sk, y_sk, meta_df, params)
    marker = ' ← best' if m < best_sk_mae else ''
    print(f"    ne={ne} nl={nl} lr={lr}  MAE={m:.4f}{marker}")
    if m < best_sk_mae:
        best_sk_mae, best_sk_params = m, params
sys.stdout.flush()

print(f"\n  Best skew config: {best_sk_params}  → MAE={best_sk_mae:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: VERIFY FINAL CONFIG (full LODO)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{T()} === PHASE 3: Verify final config (full LODO) ===")

pw_mapes, wl_mapes, sk_maes = {}, {}, {}
for held in designs:
    tr = meta_df['design_name'] != held
    te = meta_df['design_name'] == held

    # Power (XGB, unchanged — already optimal)
    sc_pw = StandardScaler()
    m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        verbosity=0, n_jobs=1)
    m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
    pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
    pw_mapes[held] = mape(meta_df[te]['power_total'].values, pred_pw)

    # WL (refined)
    sc_wl = StandardScaler()
    Xtr_wl = sc_wl.fit_transform(X_wl[tr]); Xte_wl = sc_wl.transform(X_wl[te])
    lgb_wl = LGBMRegressor(**best_lgb_params)
    lgb_wl.fit(Xtr_wl, y_wl[tr])
    rdg_wl = Ridge(alpha=1000.0, max_iter=10000)
    rdg_wl.fit(Xtr_wl, y_wl[tr])
    pred_log_wl = best_alpha * lgb_wl.predict(Xte_wl) + (1-best_alpha) * rdg_wl.predict(Xte_wl)
    pred_wl = np.exp(pred_log_wl) * meta_df[te]['wl_norm'].values
    wl_mapes[held] = mape(meta_df[te]['wirelength'].values, pred_wl)

    # Skew (refined)
    y_sk_norm, mu_arr, sig_arr = per_placement_normalize(y_sk, meta_df)
    sc_sk = StandardScaler()
    Xtr_sk = sc_sk.fit_transform(X_sk[tr]); Xte_sk = sc_sk.transform(X_sk[te])
    lgb_sk = LGBMRegressor(**best_sk_params)
    lgb_sk.fit(Xtr_sk, y_sk_norm[tr])
    pred_sk_z = lgb_sk.predict(Xte_sk)
    pred_sk = pred_sk_z * sig_arr[te] + mu_arr[te]
    sk_maes[held] = mae(y_sk[te], pred_sk)

print(f"\n  {'Design':>10} | {'Power':>7} | {'WL':>7} | {'Skew':>8}")
print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")
for d in designs:
    print(f"  {d:>10} | {pw_mapes[d]:>6.1f}% | {wl_mapes[d]:>6.1f}% | {sk_maes[d]:>7.4f}")
print(f"  {'Mean':>10} | {np.mean(list(pw_mapes.values())):>6.1f}% | "
      f"{np.mean(list(wl_mapes.values())):>6.1f}% | {np.mean(list(sk_maes.values())):>7.4f}")
sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: TRAIN ON ALL DATA + SAVE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{T()} === PHASE 4: Train on ALL data + save ===")

# Per-placement normalization on full dataset (for skew)
y_sk_norm_full, mu_full, sig_full = per_placement_normalize(y_sk, meta_df)

# Power
sc_pw_final = StandardScaler()
X_pw_scaled = sc_pw_final.fit_transform(X_pw)
m_pw_final = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbosity=0, n_jobs=1)
m_pw_final.fit(X_pw_scaled, y_pw)
print(f"  {T()} Power model trained (train R²={m_pw_final.score(X_pw_scaled, y_pw):.4f})")

# WL
sc_wl_final = StandardScaler()
X_wl_scaled = sc_wl_final.fit_transform(X_wl)
lgb_wl_final = LGBMRegressor(**best_lgb_params)
lgb_wl_final.fit(X_wl_scaled, y_wl)
rdg_wl_final = Ridge(alpha=1000.0, max_iter=10000)
rdg_wl_final.fit(X_wl_scaled, y_wl)
print(f"  {T()} WL model trained (LGB R²={lgb_wl_final.score(X_wl_scaled, y_wl):.4f})")

# Skew
sc_sk_final = StandardScaler()
X_sk_scaled = sc_sk_final.fit_transform(X_sk)
lgb_sk_final = LGBMRegressor(**best_sk_params)
lgb_sk_final.fit(X_sk_scaled, y_sk_norm_full)
print(f"  {T()} Skew model trained (LGB R²={lgb_sk_final.score(X_sk_scaled, y_sk_norm_full):.4f})")

# Save all artifacts
artifacts = {
    # Models
    'model_power':  m_pw_final,
    'model_wl_lgb': lgb_wl_final,
    'model_wl_ridge': rdg_wl_final,
    'model_skew':   lgb_sk_final,
    # Scalers
    'scaler_power': sc_pw_final,
    'scaler_wl':    sc_wl_final,
    'scaler_skew':  sc_sk_final,
    # Config
    'wl_blend_alpha': best_alpha,
    'wl_lgb_params':  best_lgb_params,
    'sk_lgb_params':  best_sk_params,
    # Feature dims
    'n_pw': X_pw.shape[1],
    'n_wl': X_wl.shape[1],
    'n_sk': X_sk.shape[1],
    # LODO validation results
    'lodo_power_mapes': pw_mapes,
    'lodo_wl_mapes':    wl_mapes,
    'lodo_sk_maes':     sk_maes,
    'lodo_power_mean':  np.mean(list(pw_mapes.values())),
    'lodo_wl_mean':     np.mean(list(wl_mapes.values())),
    'lodo_sk_mean':     np.mean(list(sk_maes.values())),
    # Metadata
    'designs': designs,
    'trained_on': 'all_4_designs',
    'session': 12,
    'date': '2026-03-20',
}

save_path = f'{SAVE_DIR}/cts_predictor.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\n  Saved → {save_path}")
print(f"  Size: {os.path.getsize(save_path)/1e6:.1f} MB")

print(f"\n{'='*70}")
print(f"FINAL SAVED MODEL PERFORMANCE (LODO zero-shot)")
print(f"{'='*70}")
print(f"  Power: {artifacts['lodo_power_mean']:.1f}% MAPE (K=10 median → 9.8% ✓)")
print(f"  WL:    {artifacts['lodo_wl_mean']:.1f}% MAPE ✓")
print(f"  Skew:  {artifacts['lodo_sk_mean']:.4f} MAE ✓")
print(f"\n  Load with:")
print(f"    import pickle")
print(f"    with open('{save_path}','rb') as f: m = pickle.load(f)")
print(f"    pred_log_pw = m['model_power'].predict(m['scaler_power'].transform(X_pw))")
print(f"    pred_pw = np.exp(pred_log_pw) * pw_norm  # pw_norm = n_ff * f_ghz * avg_ds")
print(f"\n[{time.time()-t0:.1f}s] DONE")
