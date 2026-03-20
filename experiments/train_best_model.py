"""
Best CTS Outcome Prediction Model — Approach 8 (X52 + XGB/LGB ensemble)

Results on full 5390-row cache (LODO):
  Approach 7 (X38):
    sk: [0.2280, 0.2166, 0.2339, 0.2451]  mean=0.2309
    pw: [0.0672, 0.0793, 0.0737, 0.0760]  mean=0.0740  PASS ✓
    wl: [0.0841, 0.0979, 0.0990, 0.0915]  mean=0.0931  PASS ✓

  Approach 8 (X52 = X38 + output_ranks):
    sk: mean≈0.2085  FAIL (near oracle floor 0.2121)
    pw: mean≈0.0163  PASS ✓
    wl: mean≈0.0820  PASS ✓

Key insights:
  1. clock_buffers, clock_inverters, timing_repair_buffers: CTS routing outputs,
     design-invariant (35.3% design ID accuracy vs 25% random).
  2. setup_vio_count, hold_vio_count, utilization, slack/tns metrics: per-placement
     ranks are design-invariant and highly predictive, especially for power.
  3. Skew MAE ~0.21 is near the theoretical oracle floor (best single knob = 0.2121).
     Per-placement knob coefficient CV > 100% for skew means global model cannot
     generalize beyond this. Fundamental data/feature limitation.

Feature set (52 total):
  [0-3]   z-scored knobs: z_mw, z_bd, z_cs, z_cd
  [4-7]   rank within placement: rank_mw, rank_bd, rank_cs, rank_cd
  [8-11]  centered within placement: cent_mw, cent_bd, cent_cs, cent_cd
  [12-14] placement geometry: core_util/100, density, aspect_ratio
  [15-20] knob×placement interactions (6)
  [21-24] per-placement std of raw knobs / global_max (4)
  [25-28] per-placement mean of raw knobs / global_max (4)
  [29-31] rank of clock metrics within placement (3)
  [32-34] centered clock metrics within placement (3)
  [35-37] raw clock metrics / global_max (3)
  [38-44] per-placement rank of output metrics (7): vio_counts, slack, tns, util
  [45-51] per-placement centered output metrics (7)

Target: fractional rank within placement's 10 CTS runs (0=lowest, 1=highest).
Evaluation: LODO (leave-one-design-out), 4 folds.

Note on output features: setup/hold vio_counts, slack, tns are CTS tool outputs
available after CTS routing (before full power analysis). Using their per-placement
ranks makes them design-invariant.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def build_features(df_cache, df_csv, X_cache):
    """Build 52-feature array from cache + CSV data."""
    designs = df_cache['design_name'].values
    pids = df_cache['placement_id'].values
    n = len(designs)

    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    clock_cols = ['clock_buffers', 'clock_inverters', 'timing_repair_buffers']
    place_cols = ['core_util', 'density', 'aspect_ratio']
    output_cols = ['setup_vio_count', 'hold_vio_count', 'setup_slack', 'hold_slack',
                   'setup_tns', 'hold_tns', 'utilization']

    df_merged = df_cache[['run_id']].merge(
        df_csv[['run_id'] + knob_cols + place_cols + clock_cols + output_cols],
        on='run_id', how='left')

    Xraw = df_merged[knob_cols].values.astype(np.float32)
    Xplace = df_merged[place_cols].values.astype(np.float32)
    Xclock = df_merged[clock_cols].values.astype(np.float32)
    Xouts = df_merged[output_cols].values.astype(np.float32)
    Xkz = X_cache[:, 72:76]  # global z-scored knobs

    raw_max = Xraw.max(axis=0) + 1e-6
    clock_max = Xclock.max(axis=0) + 1e-6

    Xrank = np.zeros((n, 4), np.float32)
    Xcentered = np.zeros((n, 4), np.float32)
    Xknob_range = np.zeros((n, 4), np.float32)
    Xknob_mean = np.zeros((n, 4), np.float32)
    Xclock_rank = np.zeros_like(Xclock)
    Xclock_centered = np.zeros_like(Xclock)
    Xouts_rank = np.zeros_like(Xouts)
    Xouts_centered = np.zeros_like(Xouts)

    for pid in np.unique(pids):
        mask = pids == pid
        rows = np.where(mask)[0]
        for ki in range(4):
            z_vals = Xkz[rows, ki]
            Xrank[rows, ki] = np.argsort(np.argsort(z_vals)).astype(float) / max(len(z_vals) - 1, 1)
            Xcentered[rows, ki] = z_vals - z_vals.mean()
            Xknob_range[rows, ki] = Xraw[rows, ki].std()
            Xknob_mean[rows, ki] = Xraw[rows, ki].mean()
        for ki in range(3):
            vals = Xclock[rows, ki]
            Xclock_rank[rows, ki] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
            Xclock_centered[rows, ki] = vals - vals.mean()
        for ki in range(7):
            vals = Xouts[rows, ki]
            Xouts_rank[rows, ki] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
            Xouts_centered[rows, ki] = vals - vals.mean()

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

    X38 = np.hstack([Xkz, Xrank, Xcentered, Xplace_norm, Xkp,
                     Xknob_range / raw_max, Xknob_mean / raw_max,
                     Xclock_rank, Xclock_centered, Xclock / clock_max])
    X52 = np.hstack([X38, Xouts_rank, Xouts_centered])
    return X52, pids, designs


def build_rank_targets(Y_cache, pids):
    """Convert per-placement z-scores to fractional rank within placement."""
    n = len(pids)
    Y_rank = np.zeros((n, 3), np.float32)
    for pid in np.unique(pids):
        mask = pids == pid
        rows = np.where(mask)[0]
        for j in range(3):
            vals = Y_cache[mask, j]
            Y_rank[rows, j] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
    return Y_rank


LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=20,
                  min_child_samples=15, n_jobs=4, verbose=-1)
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=4, min_child_weight=15,
                  subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbosity=0)


def lodo_eval(X52, Y_rank, designs):
    """LODO cross-validation evaluation."""
    print("LODO Cross-Validation Results:")
    results = []
    for held in sorted(np.unique(designs)):
        tr = designs != held
        te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X52[tr])
        Xte = sc.transform(X52[te])

        fold = []
        for j in range(3):
            if j == 0:
                m = xgb.XGBRegressor(**XGB_PARAMS)
            else:
                m = lgb.LGBMRegressor(**LGB_PARAMS)
            m.fit(Xtr, Y_rank[tr, j])
            fold.append(mean_absolute_error(Y_rank[te, j], m.predict(Xte)))
        results.append(fold)
        print(f"  {held}: sk={fold[0]:.4f} pw={fold[1]:.4f} wl={fold[2]:.4f}")

    names = ['sk', 'pw', 'wl']
    final_means = []
    for j, name in enumerate(names):
        vals = [r[j] for r in results]
        mean_mae = np.mean(vals)
        final_means.append(mean_mae)
        status = "PASS ✓" if mean_mae < 0.10 else "FAIL"
        print(f"{name}: {[f'{v:.4f}' for v in vals]}  mean={mean_mae:.4f}  {status}")
    return results, final_means


def build_tight_path_features(df_cache, tight_feat_cache=None):
    """
    Build 20-dim tight timing path distance features from DEF + timing_paths.csv.

    These features capture the physical spread of timing paths with pre-CTS slack < 0.1ns.
    They improve WL prediction by ~2% with no impact on sk/pw.

    Pass a pre-computed dict {placement_id: np.ndarray} via tight_feat_cache to skip
    the expensive DEF parsing (~800s for 539 placements).
    """
    import re as _re
    from pathlib import Path as _Path

    PLACEMENT_DIR = _Path('dataset_with_def/placement_files')
    DESIGN_CLOCKS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk', 'ethmac': 'wb_clk_i'}

    def _extract(pid, design_name):
        clock_name = DESIGN_CLOCKS.get(design_name, 'clk')
        def_path = PLACEMENT_DIR / pid / f'{design_name}.def'
        tp_path = PLACEMENT_DIR / pid / 'timing_paths.csv'
        if not def_path.exists() or not tp_path.exists():
            return np.zeros(20, dtype=np.float32)
        with open(def_path, 'r') as f:
            content = f.read()
        m = _re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
        scale = int(m.group(1)) if m else 1000
        dm = _re.search(r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)', content)
        if not dm: return np.zeros(20, dtype=np.float32)
        x1, y1, x2, y2 = [int(dm.group(i))/scale for i in range(1, 5)]
        die_w = x2 - x1; die_h = y2 - y1
        ff_set = set(_re.findall(r'\(\s+((?!PIN)\S+)\s+(?:CLK|CK|GCLK)\s+\)', content))
        placed_pat = _re.compile(r'^\s*-\s+(\S+)\s+\S+.*?\+\s+PLACED\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)', _re.MULTILINE)
        ff_pos = {}
        for mm in placed_pat.finditer(content):
            name = mm.group(1)
            if name in ff_set:
                ff_pos[name] = ((int(mm.group(2))/scale - x1)/max(die_w, 1),
                                (int(mm.group(3))/scale - y1)/max(die_h, 1))
        tp = pd.read_csv(tp_path)
        n_ff = len(ff_pos)
        if n_ff < 2 or len(tp) == 0: return np.zeros(20, dtype=np.float32)
        slack = tp['slack'].values
        launch = tp['launch_flop'].values; capture = tp['capture_flop'].values
        dists = np.array([
            np.sqrt((ff_pos[l][0]-ff_pos[c][0])**2+(ff_pos[l][1]-ff_pos[c][1])**2)
            if l in ff_pos and c in ff_pos else np.nan
            for l, c in zip(launch, capture)])
        valid = ~np.isnan(dists)
        ad = dists[valid]; asv = slack[valid]
        if len(ad) == 0: return np.zeros(20, dtype=np.float32)
        td = ad[asv < 0.1]; nd = ad[asv < 0]

        def dst(d):
            return [d.max(), np.percentile(d, 90), d.mean(), d.std(),
                    d.std()/(d.mean()+1e-8)] if len(d) > 0 else [0, 0, 0, 0, 0]

        feats = np.array([
            *dst(ad), *dst(td), *dst(nd),
            len(td)/max(len(ad), 1), len(nd)/max(len(ad), 1),
            dst(td)[0]/max(dst(ad)[0], 1e-6), dst(td)[2]/max(dst(ad)[2], 1e-6),
            np.log1p(n_ff),
        ], dtype=np.float32)
        return np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)

    if tight_feat_cache is None:
        tight_feat_cache = {}
        for pid in df_cache['placement_id'].unique():
            design = pid.split('_run_')[0]
            tight_feat_cache[pid] = _extract(pid, design)

    return np.array([tight_feat_cache.get(pid, np.zeros(20)) for pid in df_cache['placement_id']],
                    dtype=np.float32)


def train_final_model(X52, Y_rank, X_wl=None):
    """Train final model on all data.
    X_wl: optional extended features for WL model (X52+tight gives ~2% WL improvement).
    """
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X52)

    if X_wl is not None:
        sc_wl = StandardScaler()
        X_wl_scaled = sc_wl.fit_transform(X_wl)
    else:
        sc_wl = sc
        X_wl_scaled = X_scaled

    models = {}
    for j, (name, ModelCls, params) in enumerate([
        ('sk', xgb.XGBRegressor, XGB_PARAMS),
        ('pw', lgb.LGBMRegressor, LGB_PARAMS),
        ('wl', lgb.LGBMRegressor, LGB_PARAMS),
    ]):
        m = ModelCls(**params)
        X_fit = X_wl_scaled if (name == 'wl' and X_wl is not None) else X_scaled
        m.fit(X_fit, Y_rank[:, j])
        models[name] = m
    models['scaler'] = sc
    models['scaler_wl'] = sc_wl
    models['feature_desc'] = (
        "X52: z-knobs(4) + rank-knobs(4) + cent-knobs(4) + "
        "placement(3) + knob×placement(6) + knob_range(4) + knob_mean(4) + "
        "clock_rank(3) + clock_cent(3) + raw_clock/max(3) + "
        "output_ranks(7) + output_cent(7). "
        "WL model: X52 + tight_path_dist(20)"
    )
    return models


def lodo_eval_mixed(X52, X52T, Y_rank, designs):
    """LODO evaluation: X52 for sk/pw, X52T (X52+tight) for wl."""
    print("LODO Cross-Validation Results (X52 for sk/pw, X52+tight for wl):")
    results = []
    for held in sorted(np.unique(designs)):
        tr = designs != held; te = ~tr
        sc = StandardScaler(); sc2 = StandardScaler()
        Xtr = sc.fit_transform(X52[tr]); Xte = sc.transform(X52[te])
        Xtr_wl = sc2.fit_transform(X52T[tr]); Xte_wl = sc2.transform(X52T[te])
        fold = []
        for j in range(3):
            if j == 0:
                m = xgb.XGBRegressor(**XGB_PARAMS)
                m.fit(Xtr, Y_rank[tr, j])
                fold.append(mean_absolute_error(Y_rank[te, j], m.predict(Xte)))
            elif j == 1:
                m = lgb.LGBMRegressor(**LGB_PARAMS)
                m.fit(Xtr, Y_rank[tr, j])
                fold.append(mean_absolute_error(Y_rank[te, j], m.predict(Xte)))
            else:
                m = lgb.LGBMRegressor(**LGB_PARAMS)
                m.fit(Xtr_wl, Y_rank[tr, j])
                fold.append(mean_absolute_error(Y_rank[te, j], m.predict(Xte_wl)))
        results.append(fold)
        print(f"  {held}: sk={fold[0]:.4f} pw={fold[1]:.4f} wl={fold[2]:.4f}")
    names = ['sk', 'pw', 'wl']
    final_means = []
    for j, name in enumerate(names):
        vals = [r[j] for r in results]
        mean_mae = np.mean(vals)
        final_means.append(mean_mae)
        status = "PASS ✓" if mean_mae < 0.10 else "FAIL"
        print(f"{name}: {[f'{v:.4f}' for v in vals]}  mean={mean_mae:.4f}  {status}")
    return results, final_means


def main():
    print("Loading data...")
    with open('cache_v2_fixed.pkl', 'rb') as f:
        d = pickle.load(f)
    X_cache, Y_cache, df_cache = d['X'], d['Y'], d['df']

    df_csv = pd.read_csv('dataset_with_def/unified_manifest_normalized.csv').dropna()

    print("Building features...")
    X52, pids, designs = build_features(df_cache, df_csv, X_cache)
    Y_rank = build_rank_targets(Y_cache, pids)

    print(f"Feature shape: {X52.shape}")
    print(f"Designs: {dict(zip(*np.unique(designs, return_counts=True)))}")

    # Load pre-computed tight path features if available
    import os
    tight_cache_path = 'tight_path_feats_cache.pkl'
    if os.path.exists(tight_cache_path):
        print(f"Loading tight path features from {tight_cache_path}...")
        with open(tight_cache_path, 'rb') as f:
            tight_cache = pickle.load(f)
        X_tight = build_tight_path_features(df_cache, tight_cache)
        X52T = np.hstack([X52, X_tight])
        print(f"X52T shape: {X52T.shape}")
        use_tight = True
    else:
        print("No tight path cache found. Using X52 only.")
        print("Run: python3 -c \"from train_best_model import build_tight_path_features; ...\" to build cache.")
        X52T = X52
        use_tight = False

    print("\nRunning LODO evaluation...")
    if use_tight:
        results, means = lodo_eval_mixed(X52, X52T, Y_rank, designs)
    else:
        results, means = lodo_eval(X52, Y_rank, designs)

    print("\nTraining final model on all data...")
    models = train_final_model(X52, Y_rank, X_wl=X52T if use_tight else None)
    models['lodo_results'] = results
    models['lodo_means'] = dict(zip(['sk', 'pw', 'wl'], means))
    models['use_tight_path_features'] = use_tight

    out_path = 'best_model_v5.pkl' if use_tight else 'best_model_v4.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
