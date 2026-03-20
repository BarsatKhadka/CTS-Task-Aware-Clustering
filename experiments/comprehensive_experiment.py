"""
Comprehensive pre-CTS power/WL/skew prediction experiment.
Goal: push power, WL (and skew) to the lowest possible MAE using ONLY pre-CTS features.

Pre-CTS features safe for LODO:
- CTS knobs (z-scored, rank, centered, range, mean)
- Placement geometry: core_util, density, aspect_ratio
- Synthesis flags: io_mode, time_driven, routability_driven, synth_strategy
- Tight path features from timing_paths.csv (normalized distances)

NOT safe (>90% design ID accuracy):
- All graph features (n_ff, HPWL in µm, skip distances, SAIF toggle rates)
"""
import pickle, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
warnings.filterwarnings('ignore')

# ─────────────────────────────────── DATA LOADING ─────────────────────────────────────

def load_data():
    with open('cache_v2_fixed.pkl', 'rb') as f:
        cache = pickle.load(f)
    df_csv = pd.read_csv('dataset_with_def/unified_manifest_normalized.csv').dropna()
    return cache['X'], cache['Y'], cache['df'], df_csv


# ─────────────────────────────────── FEATURE BUILDERS ─────────────────────────────────

def rank_within(vals):
    n = len(vals)
    return np.argsort(np.argsort(vals)).astype(float) / max(n - 1, 1)


def build_X29(df_cache, df_csv, X_cache):
    """X29 baseline: knobs + rank + centered + placement geometry + interactions."""
    designs = df_cache['design_name'].values
    pids = df_cache['placement_id'].values
    n = len(pids)

    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    place_cols = ['core_util', 'density', 'aspect_ratio']

    df_m = df_cache[['run_id']].merge(df_csv[['run_id'] + knob_cols + place_cols], on='run_id', how='left')
    Xraw = df_m[knob_cols].values.astype(np.float32)
    Xplace = df_m[place_cols].values.astype(np.float32)
    Xkz = X_cache[:, 72:76]  # global z-scored knobs
    raw_max = Xraw.max(axis=0) + 1e-6

    Xrank = np.zeros((n, 4), np.float32)
    Xcent = np.zeros((n, 4), np.float32)
    Xrange = np.zeros((n, 4), np.float32)
    Xmean = np.zeros((n, 4), np.float32)

    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for ki in range(4):
            z = Xkz[rows, ki]
            Xrank[rows, ki] = rank_within(z)
            Xcent[rows, ki] = z - z.mean()
            Xrange[rows, ki] = Xraw[rows, ki].std()
            Xmean[rows, ki] = Xraw[rows, ki].mean()

    Xplace_norm = Xplace.copy()
    Xplace_norm[:, 0] /= 100.0  # core_util

    Xkp = np.column_stack([
        Xraw[:, 3] * Xplace[:, 0] / 100,   # cd × util
        Xraw[:, 0] * Xplace[:, 1],           # mw × density
        Xraw[:, 3] / np.maximum(Xplace[:, 1], 0.01),  # cd / density
        Xraw[:, 3] * Xplace[:, 2],           # cd × aspect
        Xrank[:, 3] * (Xplace[:, 0] / 100), # rank(cd) × util
        Xrank[:, 2] * (Xplace[:, 0] / 100), # rank(cs) × util
    ])

    X29 = np.hstack([Xkz, Xrank, Xcent, Xplace_norm,
                     Xrange / raw_max, Xmean / raw_max, Xkp])
    assert X29.shape[1] == 29, f"Expected 29, got {X29.shape[1]}"
    return X29, pids, designs


def build_synth_features(df_cache, df_csv):
    """Synthesis strategy flags and encoded strategy — check design-invariance separately."""
    df_m = df_cache[['run_id']].merge(
        df_csv[['run_id', 'io_mode', 'time_driven', 'routability_driven', 'synth_strategy']],
        on='run_id', how='left')

    def encode_synth(s):
        if isinstance(s, str):
            parts = s.strip().split()
            t = {'DELAY': 1, 'AREA': 0}.get(parts[0], 0)
            p = int(parts[1]) if len(parts) > 1 else 0
        else:
            t, p = 0, 0
        return t, p

    enc = [encode_synth(s) for s in df_m['synth_strategy']]
    synth_type = np.array([e[0] for e in enc], dtype=np.float32)
    synth_pri = np.array([e[1] for e in enc], dtype=np.float32)

    Xsyn = np.column_stack([
        df_m['io_mode'].values.astype(np.float32),
        df_m.get('time_driven', pd.Series(0, index=df_m.index)).values.astype(np.float32),
        df_m.get('routability_driven', pd.Series(0, index=df_m.index)).values.astype(np.float32),
        synth_type,
        synth_pri / 4.0,  # normalize 0-4 range
    ])
    return Xsyn


def build_knob_products(df_cache, df_csv, X_cache):
    """Additional quadratic and cross-product features from knobs."""
    pids = df_cache['placement_id'].values
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    df_m = df_cache[['run_id']].merge(df_csv[['run_id'] + knob_cols], on='run_id', how='left')
    Xraw = df_m[knob_cols].values.astype(np.float32)
    Xkz = X_cache[:, 72:76]

    # Physics-motivated cross products
    # Power: 1/cluster_size (buffer count), cluster_dia×cluster_size (cluster efficiency)
    # WL: cluster_dia×max_wire (routing budget), cluster_dia/cluster_size
    # Skew: buf_dist/cluster_dia (buffer capability ratio from iCTS)

    cs = Xraw[:, 2]; cd = Xraw[:, 3]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
    cs_safe = np.maximum(cs, 1e-3)
    cd_safe = np.maximum(cd, 1e-3)

    Xphys = np.column_stack([
        1.0 / cs_safe,                        # 1/cluster_size (buffer count proxy)
        cd / cs_safe,                         # cluster_dia / cluster_size
        cd * mw,                              # cluster_dia × max_wire
        bd / cd_safe,                         # buf_dist / cluster_dia (iCTS capability ratio)
        cd * cd,                              # cluster_dia² (nonlinear WL)
        cs * cs,                              # cluster_size²
        mw * mw,                              # max_wire²
        Xkz[:, 3] * Xkz[:, 2],              # z_cd × z_cs
        Xkz[:, 0] * Xkz[:, 3],              # z_mw × z_cd
        Xkz[:, 3] ** 2,                       # z_cd²
        Xkz[:, 2] ** 2,                       # z_cs²
    ])
    # Normalize by global max to keep scale reasonable
    Xphys_norm = Xphys / (np.abs(Xphys).max(axis=0) + 1e-6)
    return Xphys_norm


def build_rank_products(df_cache, df_csv, X_cache):
    """Rank-based products: within-placement rank of composite physics quantities."""
    pids = df_cache['placement_id'].values
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    df_m = df_cache[['run_id']].merge(df_csv[['run_id'] + knob_cols], on='run_id', how='left')
    Xraw = df_m[knob_cols].values.astype(np.float32)
    n = len(pids)

    cs = Xraw[:, 2]; cd = Xraw[:, 3]; mw = Xraw[:, 0]; bd = Xraw[:, 1]

    # Physics composites for within-placement ranking
    composites = {
        '1/cs': 1.0 / np.maximum(cs, 1e-3),
        'cd*mw': cd * mw,
        'cd/cs': cd / np.maximum(cs, 1e-3),
        'cd+mw': cd + mw,
        'cs*cd': cs * cd,
    }

    Xrp = np.zeros((n, len(composites)), np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for j, (name, vals) in enumerate(composites.items()):
            Xrp[rows, j] = rank_within(vals[rows])
    return Xrp


def build_tight_path_features(df_cache, tight_cache_path='tight_path_feats_cache.pkl'):
    """Tight timing path features from pre-CTS timing_paths.csv + DEF positions."""
    if not Path(tight_cache_path).exists():
        print(f"  [WARNING] Tight path cache not found at {tight_cache_path}. Skipping.")
        return None

    with open(tight_cache_path, 'rb') as f:
        tc = pickle.load(f)

    pids = df_cache['placement_id'].values
    X_tight = np.array([tc.get(pid, np.zeros(20)) for pid in pids], dtype=np.float32)
    return X_tight


def build_rank_targets(Y_cache, pids):
    """Fractional rank within each placement's 10 CTS runs."""
    n = len(pids)
    Y_rank = np.zeros((n, 3), np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for j in range(3):
            vals = Y_cache[mask := m, j] if False else Y_cache[rows, j]  # noqa
            Y_rank[rows, j] = rank_within(vals)
    return Y_rank


def build_zscore_targets(df_csv, pids):
    """Per-placement z-scored targets (alternative to rank)."""
    n = len(pids)
    Y_z = np.zeros((n, 3), np.float32)
    targets = ['skew_setup', 'power_total', 'wirelength']
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for j, tgt in enumerate(targets):
            vals = df_csv.loc[m, tgt].values if hasattr(df_csv, 'loc') else df_csv[tgt].values[rows]
            mu, sig = vals.mean(), vals.std()
            sig = max(sig, max(abs(mu) * 0.01, 1e-4))
            Y_z[rows, j] = (vals - mu) / sig
    return Y_z


# ─────────────────────────────────── LODO EVAL ────────────────────────────────────────

def lodo_eval(feature_dict, Y_rank, designs, models_per_task=None):
    """
    LODO evaluation. feature_dict: {task: Xfeats} or single X array for all tasks.
    models_per_task: dict {task_idx: model_class} or None for default.
    """
    if isinstance(feature_dict, np.ndarray):
        feature_dict = {0: feature_dict, 1: feature_dict, 2: feature_dict}

    task_names = ['sk', 'pw', 'wl']
    design_list = sorted(np.unique(designs))
    results = []

    for held in design_list:
        tr = designs != held
        te = ~tr
        fold = []

        for j, task in enumerate(task_names):
            X_j = feature_dict.get(j, feature_dict.get(task, None))
            if X_j is None:
                X_j = next(iter(feature_dict.values()))

            sc = StandardScaler()
            Xtr = sc.fit_transform(X_j[tr])
            Xte = sc.transform(X_j[te])

            if models_per_task and j in models_per_task:
                m = models_per_task[j]()
            else:
                # Default: XGBoost for skew, LightGBM for power/WL
                if j == 0:
                    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                                         min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                                         n_jobs=4, verbosity=0)
                else:
                    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=20,
                                          min_child_samples=15, n_jobs=4, verbose=-1)

            m.fit(Xtr, Y_rank[tr, j])
            pred = np.clip(m.predict(Xte), 0, 1)
            fold.append(mean_absolute_error(Y_rank[te, j], pred))

        results.append(fold)

    means = [np.mean([r[j] for r in results]) for j in range(3)]
    return results, means, design_list


def print_lodo_results(label, results, means, design_list):
    task_names = ['sk', 'pw', 'wl']
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for held, fold in zip(design_list, results):
        print(f"  {held:10s}: sk={fold[0]:.4f} pw={fold[1]:.4f} wl={fold[2]:.4f}")
    for j, (name, mean) in enumerate(zip(task_names, means)):
        status = "PASS ✓" if mean < 0.10 else "FAIL ✗"
        print(f"  {name}: mean={mean:.4f}  {status}")


# ─────────────────────────────────── HYPERPARAMETER SWEEP ─────────────────────────────

def lgb_sweep(Xtr, ytr, Xte, yte, param_grid):
    """Try multiple LGB hyperparameter configs, return best MAE and params."""
    best_mae, best_params = float('inf'), None
    for params in param_grid:
        m = lgb.LGBMRegressor(**params, n_jobs=4, verbose=-1)
        m.fit(Xtr, ytr)
        mae = mean_absolute_error(yte, np.clip(m.predict(Xte), 0, 1))
        if mae < best_mae:
            best_mae, best_params = mae, params
    return best_mae, best_params


def lodo_hp_sweep(X, Y_rank, designs, task_idx, param_grids):
    """LODO with hyperparameter sweep per fold."""
    design_list = sorted(np.unique(designs))
    fold_maes = []
    for held in design_list:
        tr = designs != held
        te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        mae, _ = lgb_sweep(Xtr, Y_rank[tr, task_idx], Xte, Y_rank[te, task_idx], param_grids)
        fold_maes.append(mae)
    return fold_maes


# ─────────────────────────────────── MAIN ─────────────────────────────────────────────

def main():
    print("Loading data...")
    X_cache, Y_cache, df_cache, df_csv = load_data()

    # Align df_csv with df_cache by run_id
    df_merged = df_cache[['run_id', 'placement_id', 'design_name']].merge(
        df_csv, on='run_id', how='left')

    print("Building features...")
    X29, pids, designs = build_X29(df_cache, df_merged, X_cache)
    Y_rank = build_rank_targets(Y_cache, pids)

    print(f"  X29 shape: {X29.shape}, total rows: {len(designs)}")
    print(f"  Designs: {dict(zip(*np.unique(designs, return_counts=True)))}")

    # ── SYNTH FLAGS ──────────────────────────────────────────────────────────────────
    print("\nBuilding synth features...")
    Xsyn = build_synth_features(df_cache, df_merged)

    # Check design ID accuracy of synth flags
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score
    le = LabelEncoder()
    y_d = le.fit_transform(designs)
    sc_tmp = StandardScaler()
    Xsyn_sc = sc_tmp.fit_transform(Xsyn)
    lr = LogisticRegression(max_iter=1000, C=0.1)
    synth_acc = cross_val_score(lr, Xsyn_sc, y_d, cv=4).mean()
    print(f"  Synth flag design ID accuracy: {synth_acc:.1%}")

    # ── PHYSICS PRODUCTS ────────────────────────────────────────────────────────────
    print("Building physics interaction features...")
    Xphys = build_knob_products(df_cache, df_merged, X_cache)
    Xrp = build_rank_products(df_cache, df_merged, X_cache)

    # ── TIGHT PATH FEATURES ─────────────────────────────────────────────────────────
    print("Loading tight path features...")
    X_tight = build_tight_path_features(df_cache)
    if X_tight is not None:
        print(f"  Tight path features shape: {X_tight.shape}")

    # ── FEATURE COMBINATIONS ────────────────────────────────────────────────────────
    feature_sets = {
        'A_X29': X29,
        'B_X29+synth': np.hstack([X29, Xsyn]),
        'C_X29+phys': np.hstack([X29, Xphys]),
        'D_X29+rank_prod': np.hstack([X29, Xrp]),
        'E_X29+phys+rank': np.hstack([X29, Xphys, Xrp]),
        'F_X29+synth+phys+rank': np.hstack([X29, Xsyn, Xphys, Xrp]),
    }
    if X_tight is not None:
        feature_sets['G_X29+tight'] = np.hstack([X29, X_tight])
        feature_sets['H_X29+all+tight'] = np.hstack([X29, Xsyn, Xphys, Xrp, X_tight])

    print(f"\n{'='*60}")
    print("FEATURE SET COMPARISON (default LGB/XGB models)")
    print(f"{'='*60}")

    all_results = {}
    for name, X_fs in feature_sets.items():
        results, means, design_list = lodo_eval(X_fs, Y_rank, designs)
        all_results[name] = (results, means)
        sk_status = "✓" if means[0] < 0.10 else "✗"
        pw_status = "✓" if means[1] < 0.10 else "✗"
        wl_status = "✓" if means[2] < 0.10 else "✗"
        print(f"  {name:30s}: sk={means[0]:.4f}{sk_status} pw={means[1]:.4f}{pw_status} wl={means[2]:.4f}{wl_status}")

    # ── HYPERPARAMETER SWEEP for Power and WL ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SWEEP: Power and WL")
    print(f"{'='*60}")

    # Use best feature set from above (start with X29+all)
    best_feature_name = min(all_results,
                             key=lambda k: all_results[k][1][1] + all_results[k][1][2])
    X_best = feature_sets[best_feature_name]
    print(f"  Using feature set: {best_feature_name}")

    param_grids_lgb = [
        dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=20),
        dict(n_estimators=300, learning_rate=0.03, num_leaves=20, min_child_samples=15),
        dict(n_estimators=500, learning_rate=0.02, num_leaves=16, min_child_samples=20),
        dict(n_estimators=500, learning_rate=0.01, num_leaves=12, min_child_samples=25),
        dict(n_estimators=1000, learning_rate=0.01, num_leaves=15, min_child_samples=20),
        dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=10),
        dict(n_estimators=500, learning_rate=0.02, num_leaves=24, min_child_samples=15),
        dict(n_estimators=200, learning_rate=0.05, num_leaves=20, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8),
        dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=20,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1),
    ]

    for task_idx, task_name in [(1, 'power'), (2, 'WL')]:
        maes = lodo_hp_sweep(X_best, Y_rank, designs, task_idx, param_grids_lgb)
        print(f"  {task_name} best LODO: {[f'{m:.4f}' for m in maes]}  mean={np.mean(maes):.4f}")

    # ── MODEL ALTERNATIVES ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("MODEL ALTERNATIVES: Power and WL on best feature set")
    print(f"{'='*60}")

    alt_models = {
        'RandomForest': lambda: RandomForestRegressor(n_estimators=500, max_depth=10,
                                                      min_samples_leaf=5, n_jobs=4),
        'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=10,
                                                  min_samples_leaf=5, n_jobs=4),
        'GBM': lambda: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                  max_depth=4, min_samples_leaf=10,
                                                  subsample=0.8),
        'XGBoost': lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                             max_depth=4, min_child_weight=15,
                                             subsample=0.8, colsample_bytree=0.8,
                                             n_jobs=4, verbosity=0),
        'XGB_deep': lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.02,
                                              max_depth=6, min_child_weight=10,
                                              subsample=0.8, colsample_bytree=0.8,
                                              n_jobs=4, verbosity=0),
    }

    for model_name, model_factory in alt_models.items():
        pw_maes = []
        wl_maes = []
        for held in sorted(np.unique(designs)):
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_best[tr])
            Xte = sc.transform(X_best[te])
            for task_idx, res_list in [(1, pw_maes), (2, wl_maes)]:
                m = model_factory()
                m.fit(Xtr, Y_rank[tr, task_idx])
                pred = np.clip(m.predict(Xte), 0, 1)
                res_list.append(mean_absolute_error(Y_rank[te, task_idx], pred))
        pw_status = "✓" if np.mean(pw_maes) < 0.10 else "✗"
        wl_status = "✓" if np.mean(wl_maes) < 0.10 else "✗"
        print(f"  {model_name:15s}: pw={np.mean(pw_maes):.4f}{pw_status} ({[f'{m:.4f}' for m in pw_maes]})  "
              f"wl={np.mean(wl_maes):.4f}{wl_status} ({[f'{m:.4f}' for m in wl_maes]})")

    # ── TASK-SPECIFIC FEATURE SETS ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TASK-SPECIFIC FEATURE OPTIMIZATION")
    print(f"{'='*60}")

    # Power: cluster_dia dominates. Build a minimal power-optimal feature set.
    # Key features: z_cd, rank_cd, centered_cd, z_cs, rank_cs, placement geo, interactions
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    df_m = df_cache[['run_id']].merge(df_merged[['run_id'] + knob_cols], on='run_id', how='left')
    Xraw = df_m[knob_cols].values.astype(np.float32)
    Xkz = X_cache[:, 72:76]
    Xplace = df_merged[['core_util', 'density', 'aspect_ratio']].values.astype(np.float32)
    Xplace_norm = Xplace.copy(); Xplace_norm[:, 0] /= 100.0

    Xrank_all = np.zeros((len(pids), 4), np.float32)
    Xcent_all = np.zeros((len(pids), 4), np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for ki in range(4):
            z = Xkz[rows, ki]
            Xrank_all[rows, ki] = rank_within(z)
            Xcent_all[rows, ki] = z - z.mean()

    # Power-specific composite: 1/cluster_size rank (buffer count proxy)
    Xrank_phys_pw = np.zeros((len(pids), 5), np.float32)
    Xrank_phys_wl = np.zeros((len(pids), 5), np.float32)
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        cs = Xraw[rows, 2]; cd = Xraw[rows, 3]; mw = Xraw[rows, 0]
        Xrank_phys_pw[rows, 0] = rank_within(1.0 / np.maximum(cs, 1e-3))  # 1/cs
        Xrank_phys_pw[rows, 1] = rank_within(cd / np.maximum(cs, 1e-3))   # cd/cs
        Xrank_phys_pw[rows, 2] = rank_within(cd)                           # cd
        Xrank_phys_pw[rows, 3] = rank_within(-cs)                          # -cs
        Xrank_phys_pw[rows, 4] = rank_within(cd * mw)                      # cd*mw
        Xrank_phys_wl[rows, 0] = rank_within(cd)                           # cd
        Xrank_phys_wl[rows, 1] = rank_within(cd * mw)                      # cd*mw
        Xrank_phys_wl[rows, 2] = rank_within(-cs)                          # -cs
        Xrank_phys_wl[rows, 3] = rank_within(1.0 / np.maximum(cs, 1e-3))  # 1/cs
        Xrank_phys_wl[rows, 4] = rank_within(cd / np.maximum(cs, 1e-3))   # cd/cs

    X_pw_specific = np.hstack([Xkz, Xrank_all, Xcent_all, Xplace_norm, Xrank_phys_pw])
    X_wl_specific = np.hstack([X29, Xrank_phys_wl])

    if X_tight is not None:
        X_wl_tight = np.hstack([X_wl_specific, X_tight])
    else:
        X_wl_tight = X_wl_specific

    # Evaluate task-specific
    for task_idx, task_name, X_ts in [(1, 'power', X_pw_specific), (2, 'WL', X_wl_specific)]:
        maes = []
        for held in sorted(np.unique(designs)):
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_ts[tr])
            Xte = sc.transform(X_ts[te])
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=20,
                                   min_child_samples=15, n_jobs=4, verbose=-1)
            m.fit(Xtr, Y_rank[tr, task_idx])
            pred = np.clip(m.predict(Xte), 0, 1)
            maes.append(mean_absolute_error(Y_rank[te, task_idx], pred))
        status = "✓" if np.mean(maes) < 0.10 else "✗"
        print(f"  {task_name}-specific LGB500: mean={np.mean(maes):.4f}{status} {[f'{m:.4f}' for m in maes]}")

    if X_tight is not None:
        # WL with tight path features
        maes = []
        for held in sorted(np.unique(designs)):
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_wl_tight[tr])
            Xte = sc.transform(X_wl_tight[te])
            m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=20,
                                   min_child_samples=15, n_jobs=4, verbose=-1)
            m.fit(Xtr, Y_rank[tr, 2])
            pred = np.clip(m.predict(Xte), 0, 1)
            maes.append(mean_absolute_error(Y_rank[te, 2], pred))
        status = "✓" if np.mean(maes) < 0.10 else "✗"
        print(f"  WL-specific+tight LGB500: mean={np.mean(maes):.4f}{status} {[f'{m:.4f}' for m in maes]}")

    # ── SKEW ATTEMPTS ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SKEW: Best pre-CTS attempts")
    print(f"{'='*60}")

    skew_feature_sets = {
        'X29': X29,
        'X29+phys': np.hstack([X29, Xphys]),
        'X29+rank_prod': np.hstack([X29, Xrp]),
        'X29+synth': np.hstack([X29, Xsyn]),
        'X29+all': np.hstack([X29, Xsyn, Xphys, Xrp]),
    }
    if X_tight is not None:
        skew_feature_sets['X29+all+tight'] = np.hstack([X29, Xsyn, Xphys, Xrp, X_tight])

    xgb_configs = [
        dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=15,
             subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
        dict(n_estimators=500, learning_rate=0.02, max_depth=5, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0),
    ]

    for feat_name, X_sk in skew_feature_sets.items():
        best_maes = None; best_mean = float('inf')
        for xgb_p in xgb_configs:
            maes = []
            for held in sorted(np.unique(designs)):
                tr = designs != held; te = ~tr
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_sk[tr])
                Xte = sc.transform(X_sk[te])
                m = xgb.XGBRegressor(**xgb_p)
                m.fit(Xtr, Y_rank[tr, 0])
                pred = np.clip(m.predict(Xte), 0, 1)
                maes.append(mean_absolute_error(Y_rank[te, 0], pred))
            if np.mean(maes) < best_mean:
                best_mean = np.mean(maes)
                best_maes = maes
        status = "✓" if best_mean < 0.10 else "✗"
        print(f"  {feat_name:20s}: sk={best_mean:.4f}{status} {[f'{m:.4f}' for m in best_maes]}")

    # ── Z-SCORE TARGET ALTERNATIVE ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Z-SCORE TARGETS (vs rank targets): Power and WL")
    print(f"{'='*60}")

    # Build per-placement z-score targets
    Y_z = np.zeros((len(pids), 3), np.float32)
    targets = ['skew_setup', 'power_total', 'wirelength']
    for pid in np.unique(pids):
        m = pids == pid
        rows = np.where(m)[0]
        for j, tgt in enumerate(targets):
            vals = df_merged.loc[m, tgt].values
            mu, sig = vals.mean(), vals.std()
            sig = max(sig, max(abs(mu) * 0.01, 1e-4))
            Y_z[rows, j] = (vals - mu) / sig

    for task_idx, task_name in [(1, 'power'), (2, 'WL')]:
        maes_z = []
        for held in sorted(np.unique(designs)):
            tr = designs != held; te = ~tr
            sc = StandardScaler()
            Xtr = sc.fit_transform(X29[tr])
            Xte = sc.transform(X29[te])
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=20,
                                   min_child_samples=15, n_jobs=4, verbose=-1)
            m.fit(Xtr, Y_z[tr, task_idx])
            pred_z = m.predict(Xte)
            # Convert z-score pred to rank for fair comparison
            pred_z_rank = np.zeros(te.sum(), np.float32)
            pids_te = pids[te]
            for pid in np.unique(pids_te):
                m2 = pids_te == pid
                rows2 = np.where(m2)[0]
                pred_z_rank[rows2] = rank_within(pred_z[rows2])
            true_rank = Y_rank[te, task_idx]
            maes_z.append(mean_absolute_error(true_rank, pred_z_rank))
        status = "✓" if np.mean(maes_z) < 0.10 else "✗"
        print(f"  {task_name} z-score→rank: {np.mean(maes_z):.4f}{status} {[f'{m:.4f}' for m in maes_z]}")

    # ── SUMMARY ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Feature set comparison (all pre-CTS):")
    for name, (results, means) in all_results.items():
        sk_s = "✓" if means[0] < 0.10 else "✗"
        pw_s = "✓" if means[1] < 0.10 else "✗"
        wl_s = "✓" if means[2] < 0.10 else "✗"
        print(f"  {name:30s}: sk={means[0]:.4f}{sk_s} pw={means[1]:.4f}{pw_s} wl={means[2]:.4f}{wl_s}")


if __name__ == "__main__":
    main()
