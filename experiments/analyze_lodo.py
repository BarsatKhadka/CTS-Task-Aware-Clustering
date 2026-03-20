"""
Quick LODO analysis script.
Runs in ~30s. No background processes.
"""
import numpy as np, pickle, pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('dataset_with_def/unified_manifest_normalized.csv').dropna()
with open('cache_v2_fixed.pkl', 'rb') as f:
    d = pickle.load(f)
X, Y_pp, df_cache = d['X'], d['Y'], d['df']

print(f"Cache: {X.shape}, Y std: {Y_pp.std(0).round(4)}")
print(f"Designs: {df_cache['design_name'].value_counts().to_dict()}")

# Min-max normalization
print("\nComputing min-max normalized targets...")
for col in ['skew_setup', 'power_total', 'wirelength']:
    df[f'mm_{col}'] = 0.0
    for pid, g in df.groupby('placement_id'):
        vals = g[col].values
        rng = max(vals.max() - vals.min(), abs(vals.mean()) * 0.001, 1e-6)
        df.loc[g.index, f'mm_{col}'] = (vals - vals.min()) / rng

# ===== TEST 1: Simple Ridge on 4 z-scored knobs =====
print("\n=== TEST 1: Ridge on 4 z-knobs ===")
knob_idx = [72, 73, 74, 75]  # z_mw, z_bd, z_cs, z_cd in cache
X_knobs = X[:, knob_idx]
designs_cache = df_cache['design_name'].values

for j, (target_col, y_arr) in enumerate([
    ('z_skew', Y_pp[:, 0]),
    ('z_power', Y_pp[:, 1]),
    ('z_wl', Y_pp[:, 2]),
]):
    fold_maes = []
    for test_d in sorted(df_cache['design_name'].unique()):
        tr = designs_cache != test_d
        te = ~tr
        m = Ridge(alpha=0.1)
        m.fit(X_knobs[tr], y_arr[tr])
        yp = m.predict(X_knobs[te])
        fold_maes.append(mean_absolute_error(y_arr[te], yp))
    print(f"  {target_col}: {[f'{m:.4f}' for m in fold_maes]}  mean={np.mean(fold_maes):.4f}")

# ===== TEST 2: LightGBM on 4 z-knobs (fewer trees) =====
print("\n=== TEST 2: LGB 200 trees on 4 z-knobs ===")
for j, (target_col, y_arr) in enumerate([
    ('z_skew', Y_pp[:, 0]),
    ('z_power', Y_pp[:, 1]),
    ('z_wl', Y_pp[:, 2]),
]):
    fold_maes = []
    for test_d in sorted(df_cache['design_name'].unique()):
        tr = designs_cache != test_d
        te = ~tr
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.1, num_leaves=15,
                               min_child_samples=20, n_jobs=1, verbose=-1)
        m.fit(X_knobs[tr], y_arr[tr])
        yp = m.predict(X_knobs[te])
        fold_maes.append(mean_absolute_error(y_arr[te], yp))
    print(f"  {target_col}: {[f'{m:.4f}' for m in fold_maes]}  mean={np.mean(fold_maes):.4f}")

# ===== TEST 3: LightGBM on knobs + placement features with strong regularization =====
print("\n=== TEST 3: LGB on knobs + placement (strong reg) ===")
X_full = X[:, list(range(72, 76)) + list(range(22, 58))]  # z-knobs + def features
print(f"  Feature dim: {X_full.shape[1]}")
for j, (target_col, y_arr) in enumerate([
    ('z_skew', Y_pp[:, 0]),
    ('z_power', Y_pp[:, 1]),
    ('z_wl', Y_pp[:, 2]),
]):
    fold_maes = []
    for test_d in sorted(df_cache['design_name'].unique()):
        tr = designs_cache != test_d
        te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_full[tr])
        Xte = sc.transform(X_full[te])
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               max_depth=4, min_child_samples=50, subsample=0.7,
                               colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
                               n_jobs=1, verbose=-1)
        m.fit(Xtr, y_arr[tr])
        yp = m.predict(Xte)
        fold_maes.append(mean_absolute_error(y_arr[te], yp))
    print(f"  {target_col}: {[f'{m:.4f}' for m in fold_maes]}  mean={np.mean(fold_maes):.4f}")

print("\nDone.")
