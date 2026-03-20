"""
Zero-shot absolute power + WL predictor.

Strategy: Calibrate across designs using truth from DEF + SAIF + graph features + CTS knobs.
Key insight: No single source has the full truth. Combine:
  - DEF geometry: n_ff, die_area, FF bounding box HPWL, FF density
  - SAIF activity: mean signal probability (T1-normalized, cross-design comparable)
  - X_cache graph features: skip distances, capacitances (per-placement z-scored but relative)
  - CTS knobs: cluster_dia, cluster_size, max_wire, buf_dist
  - Physics interactions: n_ff/cluster_size, hpwl×cluster_dia, etc.

Evaluation: LODO MAPE on log(power) and log(WL).
"""

import re
import os
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
CACHE_FILE = f'{BASE}/absolute_v3_cache.pkl'

# -----------------------------------------------------------------------
# PARSING
# -----------------------------------------------------------------------

def parse_def(def_path):
    """Extract n_ff, die geometry, FF positions from DEF file."""
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception as e:
        return None

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content)
    if not die_m:
        return None
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_w = x1 - x0
    die_h = y1 - y0
    die_area = die_w * die_h

    # Flip-flop cells: sky130 sequential cells (dfxtp, dfrtp, dfbbp, etc.)
    ff_pattern = r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    ff_xy = []
    for m in re.finditer(ff_pattern, content):
        celltype, x, y = m.groups()
        ff_xy.append((float(x) / units, float(y) / units))

    if not ff_xy:
        return None

    xs = np.array([p[0] for p in ff_xy])
    ys = np.array([p[1] for p in ff_xy])
    n_ff = len(ff_xy)
    ff_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
    ff_bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1.0
    ff_density = n_ff / die_area
    ff_spacing = np.sqrt(ff_bbox_area / max(n_ff, 1))  # avg spacing between FFs

    # Centroid (normalized)
    cx = xs.mean() / (die_w + 1e-6)
    cy = ys.mean() / (die_h + 1e-6)

    # Std dev of positions
    ff_x_std = xs.std() / (die_w + 1e-6)
    ff_y_std = ys.std() / (die_h + 1e-6)

    # 2D grid entropy (8x8)
    gx = np.clip(((xs - x0) / die_w * 8).astype(int), 0, 7)
    gy = np.clip(((ys - y0) / die_h * 8).astype(int), 0, 7)
    grid = np.zeros((8, 8))
    for i, j in zip(gx, gy):
        grid[i, j] += 1
    grid /= grid.sum() + 1e-9
    grid_ent = -np.sum(grid[grid > 0] * np.log(grid[grid > 0] + 1e-9)) / np.log(64)

    return {
        'n_ff': n_ff,
        'die_area': die_area,
        'die_w': die_w,
        'die_h': die_h,
        'ff_hpwl': ff_hpwl,
        'ff_bbox_area': ff_bbox_area,
        'ff_density': ff_density,
        'ff_spacing': ff_spacing,
        'ff_cx': cx,
        'ff_cy': cy,
        'ff_x_std': ff_x_std,
        'ff_y_std': ff_y_std,
        'grid_ent': grid_ent,
        'die_aspect': die_w / (die_h + 1e-6),
        'log_n_ff': np.log1p(n_ff),
        'log_die_area': np.log1p(die_area),
        'log_ff_hpwl': np.log1p(ff_hpwl),
        'log_ff_spacing': np.log1p(ff_spacing),
        'log_ff_density': np.log1p(ff_density),
        'sqrt_nff_area': np.sqrt(n_ff * die_area),          # ~WL predictor
        'log_sqrt_nff_area': np.log1p(np.sqrt(n_ff * die_area)),
    }


def parse_saif(saif_path):
    """Extract activity metrics from SAIF file."""
    try:
        with open(saif_path) as f:
            lines = f.readlines()
    except Exception as e:
        return None

    duration = None
    total_tc = 0
    total_t1 = 0
    total_t0 = 0
    n_nets = 0
    tc_vals = []

    for line in lines:
        if '(DURATION' in line:
            m = re.search(r'[\d.]+', line)
            if m:
                duration = float(m.group())
        if '(TC' in line:
            m = re.search(r'\(TC\s+(\d+)\)', line)
            if m:
                tc = int(m.group(1))
                total_tc += tc
                tc_vals.append(tc)
                n_nets += 1
        if '(T1' in line:
            m = re.search(r'\(T1\s+(\d+)\)', line)
            if m:
                total_t1 += int(m.group(1))
        if '(T0' in line:
            m = re.search(r'\(T0\s+(\d+)\)', line)
            if m:
                total_t0 += int(m.group(1))

    if not duration or n_nets == 0:
        return None

    # Mean signal probability: T1_sum / (n_nets * DURATION) - cross-design comparable!
    mean_sig_prob = total_t1 / (n_nets * duration)
    mean_tc_per_net = total_tc / n_nets
    # Normalized toggle rate: TC / DURATION × T_clk_proxy
    # Since we don't know T_clk, use TC/n_nets as a relative toggle density
    tc_arr = np.array(tc_vals)
    tc_std = tc_arr.std()
    tc_p90 = np.percentile(tc_arr, 90)
    tc_p50 = np.median(tc_arr)

    return {
        'n_nets': n_nets,
        'mean_sig_prob': mean_sig_prob,        # signal_high fraction (cross-design safe)
        'mean_tc_per_net': mean_tc_per_net,    # avg toggles per net (NOT normalized by DURATION)
        'log_mean_tc': np.log1p(mean_tc_per_net),
        'tc_p90_frac': tc_p90 / (total_tc / n_nets + 1),    # relative heavy switchers
        'tc_std_frac': tc_std / (mean_tc_per_net + 1),      # spread of toggle rates
        'log_n_nets': np.log1p(n_nets),
        'nets_per_ff_proxy': n_nets,                          # raw, to be combined with n_ff
    }


# -----------------------------------------------------------------------
# FEATURE CACHE
# -----------------------------------------------------------------------

def build_or_load_cache(df):
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
        print(f"Loaded cache: {len(cache['def_feats'])} DEF, {len(cache['saif_feats'])} SAIF entries")
        return cache

    print("Building cache from DEF + SAIF files...")
    unique_pids = df['placement_id'].unique()

    def_feats = {}
    saif_feats = {}

    for i, pid in enumerate(unique_pids):
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)

        df_f = parse_def(def_path)
        sf_f = parse_saif(saif_path)

        if df_f:
            def_feats[pid] = df_f
        if sf_f:
            saif_feats[pid] = sf_f

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(unique_pids)}")

    cache = {'def_feats': def_feats, 'saif_feats': saif_feats}
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)
    print(f"Cache saved: {len(def_feats)} DEF, {len(saif_feats)} SAIF entries")
    return cache


# -----------------------------------------------------------------------
# FEATURE MATRIX BUILDER
# -----------------------------------------------------------------------

def build_features(df, cache):
    """
    Build absolute feature matrix for all rows.
    Features from DEF + SAIF + CTS knobs + physics interactions.
    Target: log(power_total) and log(wirelength).
    """
    def_feats = cache['def_feats']
    saif_feats = cache['saif_feats']

    rows = []
    targets_pw = []
    targets_wl = []
    meta = []

    for _, row in df.iterrows():
        pid = row['placement_id']
        df_f = def_feats.get(pid, {})
        sf_f = saif_feats.get(pid, {})

        if not df_f or not sf_f:
            continue

        # CTS knobs
        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        n_ff = df_f['n_ff']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        n_nets = sf_f['n_nets']

        # ---- Physics interactions ----
        # WL: dominated by n_ff × avg_routing_distance
        # avg_routing_distance ∝ sqrt(die_area/n_ff) = ff_spacing
        # cluster_dia reduces local routing by grouping FFs
        # cluster_size affects routing depth
        wl_physics1 = np.log1p(n_ff * ff_spacing)             # base WL predictor
        wl_physics2 = np.log1p(ff_hpwl / (cd + 1))           # HPWL reduced by cluster grouping
        wl_physics3 = np.log1p(n_ff / cs)                     # n_cluster_groups

        # Power: P_clock ∝ WL × C_wire + n_bufs × C_buf + P_logic
        # P_logic ∝ n_ff × cap_per_ff × toggle_rate
        # mean_sig_prob captures toggle-like activity without needing simulation time
        pw_physics1 = np.log1p(n_ff * sf_f['mean_sig_prob'])  # logic power proxy
        pw_physics2 = np.log1p(n_ff / cs)                      # buffer count → clock power
        pw_physics3 = np.log1p(n_nets * sf_f['mean_sig_prob']) # total net activity
        pw_physics4 = sf_f['log_mean_tc']                      # per-net toggle rate
        pw_physics5 = np.log1p(n_ff * ff_hpwl / (cd + 1))    # clock tree WL proxy × activity

        # Nets-to-FF ratio (captures combinational complexity per FF)
        nets_per_ff = np.log1p(n_nets / max(n_ff, 1))

        feat = [
            # DEF geometry
            df_f['log_n_ff'],
            df_f['log_die_area'],
            df_f['log_ff_hpwl'],
            df_f['log_ff_spacing'],
            df_f['log_ff_density'],
            df_f['log_sqrt_nff_area'],
            df_f['ff_cx'],
            df_f['ff_cy'],
            df_f['ff_x_std'],
            df_f['ff_y_std'],
            df_f['grid_ent'],
            df_f['die_aspect'],

            # SAIF activity
            sf_f['mean_sig_prob'],
            sf_f['log_mean_tc'],
            sf_f['tc_p90_frac'],
            sf_f['tc_std_frac'],
            sf_f['log_n_nets'],
            nets_per_ff,

            # CTS knobs (raw)
            np.log1p(cd),
            np.log1p(cs),
            np.log1p(mw),
            np.log1p(bd),

            # Physics interactions
            wl_physics1,
            wl_physics2,
            wl_physics3,
            pw_physics1,
            pw_physics2,
            pw_physics3,
            pw_physics4,
            pw_physics5,

            # Cross interactions
            np.log1p(cd * n_ff),             # cluster_dia × scale
            np.log1p(cs * ff_spacing),        # cluster_size × density
            np.log1p(mw * ff_hpwl),           # max_wire × HPWL
            np.log1p(bd / (ff_spacing + 1)),   # buf_dist / avg_FF_spacing
            np.log1p(n_ff * sf_f['mean_sig_prob'] / (cs + 1)),  # switching per cluster
        ]

        rows.append(feat)
        targets_pw.append(np.log(row['power_total']))
        targets_wl.append(np.log(row['wirelength']))
        meta.append({
            'placement_id': pid,
            'design_name': row['design_name'],
            'power_total': row['power_total'],
            'wirelength': row['wirelength'],
        })

    X = np.array(rows, dtype=np.float64)
    y_pw = np.array(targets_pw, dtype=np.float64)
    y_wl = np.array(targets_wl, dtype=np.float64)
    meta_df = pd.DataFrame(meta)

    # Replace NaN/inf with column medians
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            median = np.nanmedian(X[~bad, col]) if (~bad).any() else 0.0
            X[bad, col] = median

    print(f"Feature matrix: {X.shape}  targets: {len(y_pw)}")
    print(f"Power log range: [{y_pw.min():.3f}, {y_pw.max():.3f}]")
    print(f"WL log range: [{y_wl.min():.3f}, {y_wl.max():.3f}]")
    return X, y_pw, y_wl, meta_df


# -----------------------------------------------------------------------
# LODO EVALUATION
# -----------------------------------------------------------------------

def mape(y_true, y_pred_log):
    """MAPE where predictions are in log space."""
    y_pred = np.exp(y_pred_log)
    y_true = np.array(y_true)
    return np.mean(np.abs(y_pred - y_true) / (y_true + 1e-10)) * 100


def lodo_eval(X, y_pw, y_wl, meta_df, model_cls, model_kwargs, name=""):
    """Leave-One-Design-Out evaluation."""
    designs = meta_df['design_name'].unique()
    results = {}

    for held_out in designs:
        train_mask = meta_df['design_name'] != held_out
        test_mask = meta_df['design_name'] == held_out

        X_tr = X[train_mask]
        X_te = X[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Power
        m_pw = model_cls(**model_kwargs)
        m_pw.fit(X_tr_s, y_pw[train_mask])
        pred_pw = m_pw.predict(X_te_s)
        mape_pw = mape(meta_df[test_mask]['power_total'].values, pred_pw)

        # WL
        m_wl = model_cls(**model_kwargs)
        m_wl.fit(X_tr_s, y_wl[train_mask])
        pred_wl = m_wl.predict(X_te_s)
        mape_wl = mape(meta_df[test_mask]['wirelength'].values, pred_wl)

        results[held_out] = {'mape_pw': mape_pw, 'mape_wl': mape_wl}
        print(f"  {held_out}: power_MAPE={mape_pw:.1f}%  WL_MAPE={mape_wl:.1f}%")

    mean_pw = np.mean([v['mape_pw'] for v in results.values()])
    mean_wl = np.mean([v['mape_wl'] for v in results.values()])
    print(f"  [{name}] Mean MAPE → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return results


def lodo_per_fold_detail(X, y_pw, y_wl, meta_df, model_cls, model_kwargs):
    """Detailed LODO: per-placement and per-knob breakdown."""
    designs = meta_df['design_name'].unique()

    for held_out in designs:
        train_mask = meta_df['design_name'] != held_out
        test_mask = meta_df['design_name'] == held_out

        X_tr = X[train_mask]
        X_te = X[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        m_pw = model_cls(**model_kwargs)
        m_pw.fit(X_tr_s, y_pw[train_mask])
        m_wl = model_cls(**model_kwargs)
        m_wl.fit(X_tr_s, y_wl[train_mask])

        pred_pw = np.exp(m_pw.predict(X_te_s))
        pred_wl = np.exp(m_wl.predict(X_te_s))

        true_pw = meta_df[test_mask]['power_total'].values
        true_wl = meta_df[test_mask]['wirelength'].values

        print(f"\n=== Held-out: {held_out} ===")
        print(f"  Power:  true=[{true_pw.min():.4f},{true_pw.max():.4f}]  "
              f"pred=[{pred_pw.min():.4f},{pred_pw.max():.4f}]")
        print(f"  WL:  true=[{true_wl.min():.0f},{true_wl.max():.0f}]  "
              f"pred=[{pred_wl.min():.0f},{pred_wl.max():.0f}]")

        mape_pw = np.mean(np.abs(pred_pw - true_pw) / (true_pw + 1e-10)) * 100
        mape_wl = np.mean(np.abs(pred_wl - true_wl) / (true_wl + 1e-10)) * 100
        print(f"  MAPE: power={mape_pw:.1f}%  WL={mape_wl:.1f}%")

        # Per-placement breakdown (top 5 worst)
        test_meta = meta_df[test_mask].reset_index(drop=True)
        pids = test_meta['placement_id'].unique()
        errors_pw = []
        for pid in pids:
            pmask = test_meta['placement_id'] == pid
            e = np.mean(np.abs(pred_pw[pmask] - true_pw[pmask]) / true_pw[pmask]) * 100
            errors_pw.append((pid, e))
        errors_pw.sort(key=lambda x: x[1], reverse=True)
        print(f"  Worst placements (power MAPE%):")
        for pid, e in errors_pw[:3]:
            print(f"    {pid}: {e:.1f}%")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Zero-Shot Absolute Predictor (DEF+SAIF+knobs)")
    print("=" * 60)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Total rows: {len(df)}, designs: {df['design_name'].nunique()}")

    # Build / load cache
    t0 = time.time()
    cache = build_or_load_cache(df)
    print(f"Cache ready in {time.time()-t0:.1f}s")

    # Build feature matrix
    X, y_pw, y_wl, meta_df = build_features(df, cache)

    # Feature names for inspection
    feat_names = [
        'log_n_ff', 'log_die_area', 'log_ff_hpwl', 'log_ff_spacing', 'log_ff_density',
        'log_sqrt_nff_area', 'ff_cx', 'ff_cy', 'ff_x_std', 'ff_y_std', 'grid_ent', 'die_aspect',
        'mean_sig_prob', 'log_mean_tc', 'tc_p90_frac', 'tc_std_frac', 'log_n_nets', 'nets_per_ff',
        'log_cd', 'log_cs', 'log_mw', 'log_bd',
        'wl_phys1', 'wl_phys2', 'wl_phys3',
        'pw_phys1', 'pw_phys2', 'pw_phys3', 'pw_phys4', 'pw_phys5',
        'cd_x_nff', 'cs_x_spacing', 'mw_x_hpwl', 'bd_div_spacing', 'sw_per_cluster',
    ]
    print(f"\nFeature count: {X.shape[1]}")

    print("\n--- Approach 1: Ridge Regression (baseline) ---")
    lodo_eval(X, y_pw, y_wl, meta_df,
              Ridge, {'alpha': 1.0}, name="Ridge")

    print("--- Approach 2: LightGBM (100 trees) ---")
    lodo_eval(X, y_pw, y_wl, meta_df,
              LGBMRegressor,
              {'n_estimators': 100, 'num_leaves': 15, 'learning_rate': 0.05,
               'min_child_samples': 5, 'random_state': 42, 'verbose': -1},
              name="LGB_100")

    print("--- Approach 3: LightGBM (300 trees, deeper) ---")
    lodo_eval(X, y_pw, y_wl, meta_df,
              LGBMRegressor,
              {'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.03,
               'min_child_samples': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
               'random_state': 42, 'verbose': -1},
              name="LGB_300")

    print("--- Approach 4: XGBoost depth=4 ---")
    lodo_eval(X, y_pw, y_wl, meta_df,
              XGBRegressor,
              {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
               'subsample': 0.8, 'colsample_bytree': 0.8,
               'random_state': 42, 'verbosity': 0},
              name="XGB_d4")

    # Best model detailed breakdown
    print("\n--- Detailed per-design breakdown (LGB_300) ---")
    lodo_per_fold_detail(X, y_pw, y_wl, meta_df,
                         LGBMRegressor,
                         {'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.03,
                          'min_child_samples': 5, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
                          'random_state': 42, 'verbose': -1})

    # Feature importance for WL model (train on all, check importance)
    print("\n--- Feature importances (full-data WL model) ---")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    m = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                      min_child_samples=5, random_state=42, verbose=-1)
    m.fit(X_s, y_wl)
    imps = sorted(zip(feat_names, m.feature_importances_), key=lambda x: -x[1])
    for name, imp in imps[:15]:
        print(f"  {name}: {imp}")

    print("\n--- Feature importances (full-data POWER model) ---")
    m2 = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                       min_child_samples=5, random_state=42, verbose=-1)
    m2.fit(X_s, y_pw)
    imps2 = sorted(zip(feat_names, m2.feature_importances_), key=lambda x: -x[1])
    for name, imp in imps2[:15]:
        print(f"  {name}: {imp}")


if __name__ == '__main__':
    main()
