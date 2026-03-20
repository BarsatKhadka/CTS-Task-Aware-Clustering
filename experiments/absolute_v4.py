"""
Zero-shot absolute power + WL predictor v4.

HIGH-DIMENSIONAL approach: combine ALL available information sources.
- DEF: n_ff, active_cells, drive strengths, HPWL, die_area, cell type breakdown
- SAIF: relative toggle activity (mean_TC/max_TC), n_nets, TC distribution
- CSV: synth_strategy (CRITICAL for power!), aspect_ratio, core_util, density,
       time_driven, routability_driven
- Clock period (user-provided): scales power linearly
- CTS knobs: cluster_dia, cluster_size, max_wire, buf_dist
- Physics interactions: n_active × f_clk × rel_act, n_ff × f_clk, etc.

Key insights from v3 analysis:
- SHA256 power anomaly explained by: SAIF duration bug + synth_strategy
- mean_TC/max_TC is duration-independent relative activity metric
- f_clk directly scales absolute power
- synth_strategy: AREA→smaller cells→lower power, DELAY→larger→higher power
"""

import re
import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
DEF_CACHE = f'{BASE}/absolute_v4_def_cache.pkl'
SAIF_CACHE = f'{BASE}/absolute_v4_saif_cache.pkl'

# User-provided clock periods (ns → ps)
T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


# -----------------------------------------------------------------------
# DEF PARSER (rich features)
# -----------------------------------------------------------------------

def parse_def_rich(def_path):
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
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

    # Cell type extraction
    cell_type_matches = re.findall(r'sky130_fd_sc_hd__(\w+)', content)
    ct = Counter(cell_type_matches)
    n_total_cells = len(cell_type_matches)

    # Classify cells
    n_tap = sum(v for k, v in ct.items() if any(x in k for x in ['tap', 'decap', 'fill', 'phy']))
    n_active = n_total_cells - n_tap
    n_ff = sum(v for k, v in ct.items() if k.startswith('df') or k.startswith('ff') or 'latch' in k)
    n_buf = sum(v for k, v in ct.items() if k.startswith('buf'))
    n_inv = sum(v for k, v in ct.items() if k.startswith('inv'))
    n_comb = n_active - n_ff - n_buf - n_inv

    # Drive strengths (active cells only)
    active_ds = []
    for k, v in ct.items():
        if not any(x in k for x in ['tap', 'decap', 'fill', 'phy']):
            m = re.search(r'_(\d+)$', k)
            if m:
                active_ds.extend([int(m.group(1))] * v)

    avg_active_ds = np.mean(active_ds) if active_ds else 1.0
    std_active_ds = np.std(active_ds) if len(active_ds) > 1 else 0.0
    p90_active_ds = np.percentile(active_ds, 90) if active_ds else 1.0
    frac_drive4_plus = sum(1 for d in active_ds if d >= 4) / (len(active_ds) + 1)

    # FF positions
    ff_pattern = r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    ff_xy = [(float(x) / units, float(y) / units) for _, x, y in re.findall(ff_pattern, content)]

    if not ff_xy:
        return None

    xs = np.array([p[0] for p in ff_xy])
    ys = np.array([p[1] for p in ff_xy])
    ff_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
    ff_bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1.0
    ff_density = len(ff_xy) / die_area
    ff_spacing = np.sqrt(ff_bbox_area / max(len(ff_xy), 1))

    return {
        # Geometry
        'die_area': die_area, 'die_w': die_w, 'die_h': die_h,
        'die_aspect': die_w / (die_h + 1e-6),
        'ff_hpwl': ff_hpwl, 'ff_bbox_area': ff_bbox_area,
        'ff_density': ff_density, 'ff_spacing': ff_spacing,
        'ff_cx': xs.mean() / (die_w + 1e-6), 'ff_cy': ys.mean() / (die_h + 1e-6),
        'ff_x_std': xs.std() / (die_w + 1e-6), 'ff_y_std': ys.std() / (die_h + 1e-6),
        # Cell counts
        'n_ff': len(ff_xy), 'n_active': n_active, 'n_total': n_total_cells,
        'n_tap': n_tap, 'n_buf': n_buf, 'n_inv': n_inv, 'n_comb': n_comb,
        'frac_active': n_active / (n_total_cells + 1),
        'frac_ff': n_ff / (n_active + 1),
        'frac_buf_inv': (n_buf + n_inv) / (n_active + 1),
        # Drive strengths
        'avg_active_ds': avg_active_ds, 'std_active_ds': std_active_ds,
        'p90_active_ds': p90_active_ds, 'frac_drive4plus': frac_drive4_plus,
        # Power proxies
        'cap_proxy': n_active * avg_active_ds,       # C ∝ n_cells × drive_strength
        'cap_proxy_ff': len(ff_xy) * avg_active_ds,  # FF contribution to cap
    }


# -----------------------------------------------------------------------
# SAIF PARSER (rich features)
# -----------------------------------------------------------------------

def parse_saif_rich(saif_path):
    try:
        with open(saif_path) as f:
            lines = f.readlines()
    except Exception:
        return None

    duration = None
    total_tc = 0
    total_t1 = 0
    total_t0 = 0
    n_nets = 0
    max_tc = 0
    tc_vals = []

    for line in lines:
        if '(DURATION' in line:
            m = re.search(r'[\d.]+', line)
            if m:
                duration = float(m.group())
        m = re.search(r'\(TC\s+(\d+)\)', line)
        if m:
            tc = int(m.group(1))
            tc_vals.append(tc)
            n_nets += 1
            total_tc += tc
            max_tc = max(max_tc, tc)
        m2 = re.search(r'\(T1\s+(\d+)\)', line)
        if m2:
            total_t1 += int(m2.group(1))
        m3 = re.search(r'\(T0\s+(\d+)\)', line)
        if m3:
            total_t0 += int(m3.group(1))

    if not duration or n_nets == 0 or max_tc == 0:
        return None

    tc_arr = np.array(tc_vals)
    mean_tc = total_tc / n_nets
    # Duration-independent activity: how active is average net vs most active net
    rel_act = mean_tc / max_tc   # 0-1, DURATION-independent
    # Duration-normalized signal probability (DURATION-independent for cross-design comparison)
    mean_sig_prob = total_t1 / (n_nets * duration) if duration > 0 else 0

    return {
        'n_nets': n_nets, 'max_tc': max_tc,
        'total_tc': total_tc, 'mean_tc': mean_tc,
        'rel_act': rel_act,           # DURATION-independent relative activity
        'mean_sig_prob': mean_sig_prob,
        'tc_p50': np.percentile(tc_arr, 50),
        'tc_p90': np.percentile(tc_arr, 90),
        'tc_std': tc_arr.std(),
        'tc_std_norm': tc_arr.std() / (mean_tc + 1),
        'frac_zero': (tc_arr == 0).mean(),   # fraction of idle nets
        'frac_high_act': (tc_arr > mean_tc * 2).mean(),
        'log_total_tc': np.log1p(total_tc),
        'log_max_tc': np.log1p(max_tc),
        'log_n_nets': np.log1p(n_nets),
        'nets_per_maxTC': n_nets / (max_tc + 1),  # design "complexity" vs activity
    }


# -----------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------

def build_caches(df):
    pids = df['placement_id'].unique()

    if os.path.exists(DEF_CACHE) and os.path.exists(SAIF_CACHE):
        with open(DEF_CACHE, 'rb') as f:
            def_cache = pickle.load(f)
        with open(SAIF_CACHE, 'rb') as f:
            saif_cache = pickle.load(f)
        print(f"Loaded caches: {len(def_cache)} DEF, {len(saif_cache)} SAIF")
        return def_cache, saif_cache

    print(f"Building caches for {len(pids)} placements...")
    def_cache, saif_cache = {}, {}

    for i, pid in enumerate(pids):
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)

        df_f = parse_def_rich(def_path)
        sf_f = parse_saif_rich(saif_path)
        if df_f:
            def_cache[pid] = df_f
        if sf_f:
            saif_cache[pid] = sf_f

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pids)}")

    with open(DEF_CACHE, 'wb') as f:
        pickle.dump(def_cache, f)
    with open(SAIF_CACHE, 'wb') as f:
        pickle.dump(saif_cache, f)
    print(f"Saved caches: {len(def_cache)} DEF, {len(saif_cache)} SAIF")
    return def_cache, saif_cache


# -----------------------------------------------------------------------
# SYNTH STRATEGY ENCODING
# -----------------------------------------------------------------------

SYNTH_STRATS = ['AREA 0', 'AREA 1', 'AREA 2', 'DELAY 0', 'DELAY 1',
                'DELAY 2', 'DELAY 3', 'DELAY 4']

def encode_synth_strategy(s):
    """Encode synth_strategy → [synth_type (0=AREA,1=DELAY), synth_level]"""
    if pd.isna(s):
        return 0.5, 2.0
    parts = str(s).strip().split()
    synth_type = 0.0 if 'AREA' in str(s).upper() else 1.0
    try:
        synth_level = float(parts[-1])
    except (ValueError, IndexError):
        synth_level = 2.0
    return synth_type, synth_level


# -----------------------------------------------------------------------
# FEATURE BUILDER
# -----------------------------------------------------------------------

def build_features_v4(df, def_cache, saif_cache):
    rows, y_pw, y_wl, meta = [], [], [], []

    for _, row in df.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = def_cache.get(pid)
        sf_f = saif_cache.get(pid)
        if not df_f or not sf_f:
            continue
        if pd.isna(row['power_total']) or pd.isna(row['wirelength']):
            continue

        # CTS knobs
        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # Clock frequency (from user-provided periods)
        t_clk = T_CLK_NS.get(design, 7.0)   # ns
        f_clk_ghz = 1.0 / t_clk              # GHz (easier numerical range)

        # Synth strategy
        synth_type, synth_level = encode_synth_strategy(row.get('synth_strategy', 'AREA 2'))

        # Per-placement CSV features
        aspect_ratio = float(row.get('aspect_ratio', 1.0))
        core_util = float(row.get('core_util', 55.0))
        density = float(row.get('density', 0.5))
        time_driven = float(row.get('time_driven', 0))
        routability = float(row.get('routability_driven', 0))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        cap_proxy = df_f['cap_proxy']
        rel_act = sf_f['rel_act']
        n_nets = sf_f['n_nets']

        # ---- Physics-informed features ----
        # Power physics (key combined features):
        # P ≈ k × cap_proxy × f_clk × rel_act × (1 - synth_area_penalty)
        pw_phys1 = cap_proxy * f_clk_ghz * rel_act          # main power proxy
        pw_phys2 = cap_proxy * f_clk_ghz * (1 - synth_type * 0.3)  # synth effect
        pw_phys3 = n_ff * f_clk_ghz                          # FF clock power
        pw_phys4 = sf_f['frac_zero'] * n_nets                # idle nets (less power)
        pw_phys5 = (n_ff / cs) * f_clk_ghz                  # buffer power

        # WL physics:
        wl_phys1 = np.sqrt(n_ff * die_area)                  # dominant WL predictor
        wl_phys2 = n_ff * ff_spacing / (cd + 1)             # cluster grouping
        wl_phys3 = n_ff / cs                                  # cluster count
        wl_phys4 = mw * n_ff / die_area                      # wire budget
        wl_phys5 = ff_hpwl * cs / (cd + 1)                  # cluster topology

        feat = [
            # DEF geometry (log-scaled)
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            np.log1p(df_f['ff_density']), np.log1p(np.sqrt(n_ff * die_area)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            df_f['die_aspect'], aspect_ratio,

            # DEF cell features
            np.log1p(n_active), np.log1p(df_f['n_total']),
            df_f['avg_active_ds'], df_f['std_active_ds'], df_f['p90_active_ds'],
            df_f['frac_drive4plus'], df_f['frac_active'], df_f['frac_ff'],
            df_f['frac_buf_inv'], np.log1p(cap_proxy),

            # SAIF activity (duration-independent)
            sf_f['rel_act'], sf_f['mean_sig_prob'],
            sf_f['tc_std_norm'], sf_f['frac_zero'], sf_f['frac_high_act'],
            sf_f['log_n_nets'], sf_f['nets_per_maxTC'],

            # Clock + frequency
            f_clk_ghz, t_clk,

            # CSV per-placement
            synth_type, synth_level,
            core_util / 100.0, density, time_driven, routability,

            # CTS knobs (raw + log)
            cd, cs, mw, bd,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),

            # Physics interactions (log)
            np.log1p(pw_phys1), np.log1p(pw_phys2), np.log1p(pw_phys3),
            np.log1p(pw_phys4), np.log1p(pw_phys5),
            np.log1p(wl_phys1), np.log1p(wl_phys2), np.log1p(wl_phys3),
            np.log1p(wl_phys4), np.log1p(wl_phys5),

            # Cross-interactions
            np.log1p(cap_proxy * f_clk_ghz),           # power scale
            np.log1p(n_ff * f_clk_ghz),                # FF frequency product
            synth_type * synth_level,                   # synthesis aggressiveness
            synth_type * f_clk_ghz,                     # DELAY synth at high freq → more power
            (1 - synth_type) * cap_proxy,               # AREA synth with more cells
            rel_act * (1 - sf_f['frac_zero']),          # true switching fraction
            np.log1p(n_active * rel_act * f_clk_ghz),  # full power proxy
            np.log1p(cd * n_ff / die_area),             # cluster density
            np.log1p(cs * ff_spacing),                  # cluster-spacing product
            np.log1p(mw * ff_hpwl),                     # wire budget × tree size
            core_util * density,                         # layout compactness
        ]

        rows.append(feat)
        y_pw.append(np.log(row['power_total']))
        y_wl.append(np.log(row['wirelength']))
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': row['power_total'], 'wirelength': row['wirelength']})

    X = np.array(rows, dtype=np.float64)
    y_pw = np.array(y_pw)
    y_wl = np.array(y_wl)
    meta_df = pd.DataFrame(meta)

    # Fix NaN/inf
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            med = np.nanmedian(X[~bad, col]) if (~bad).any() else 0.0
            X[bad, col] = med

    print(f"Feature matrix: {X.shape}  targets: {len(y_pw)}")
    print(f"  Power range: [{np.exp(y_pw.min()):.4f}, {np.exp(y_pw.max()):.4f}] W")
    print(f"  WL range: [{np.exp(y_wl.min()):.0f}, {np.exp(y_wl.max()):.0f}] µm")
    return X, y_pw, y_wl, meta_df


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def mape(y_true, y_pred_log):
    y_pred = np.exp(np.clip(y_pred_log, -20, 20))
    return np.mean(np.abs(y_pred - y_true) / (y_true + 1e-12)) * 100


def lodo_eval(X, y_pw, y_wl, meta_df, model_cls, kwargs, name=""):
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        m_pw = model_cls(**kwargs)
        m_pw.fit(X_tr, y_pw[tr])
        mpw = mape(meta_df[te]['power_total'].values, m_pw.predict(X_te))

        m_wl = model_cls(**kwargs)
        m_wl.fit(X_tr, y_wl[tr])
        mwl = mape(meta_df[te]['wirelength'].values, m_wl.predict(X_te))

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"  {held}: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")

    print(f"  [{name}] mean → power={np.mean(pw_mapes):.1f}%  WL={np.mean(wl_mapes):.1f}%\n")
    return np.mean(pw_mapes), np.mean(wl_mapes)


def lodo_detailed(X, y_pw, y_wl, meta_df, model_cls, kwargs):
    """Full detail: per-design predictions and power decomposition."""
    designs = meta_df['design_name'].unique()

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        m_pw = model_cls(**kwargs)
        m_pw.fit(X_tr, y_pw[tr])
        m_wl = model_cls(**kwargs)
        m_wl.fit(X_tr, y_wl[tr])

        pred_pw = np.exp(np.clip(m_pw.predict(X_te), -20, 20))
        pred_wl = np.exp(np.clip(m_wl.predict(X_te), -20, 20))
        true_pw = meta_df[te]['power_total'].values
        true_wl = meta_df[te]['wirelength'].values

        mpw = mape(true_pw, np.log(pred_pw))
        mwl = mape(true_wl, np.log(pred_wl))
        print(f"\n=== {held} (LODO) ===")
        print(f"  Power: true=[{true_pw.min():.4f},{true_pw.max():.4f}] "
              f"pred=[{pred_pw.min():.4f},{pred_pw.max():.4f}]  MAPE={mpw:.1f}%")
        print(f"  WL:    true=[{true_wl.min():.0f},{true_wl.max():.0f}] "
              f"pred=[{pred_wl.min():.0f},{pred_wl.max():.0f}]  MAPE={mwl:.1f}%")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Zero-Shot Absolute Predictor v4 — High-Dimensional ML")
    print("=" * 65)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Rows: {len(df)}, designs: {df['design_name'].nunique()}")

    # Use existing v3 caches for DEF (same parser essentially) if available
    # otherwise build fresh ones
    def_cache, saif_cache = build_caches(df)

    X, y_pw, y_wl, meta_df = build_features_v4(df, def_cache, saif_cache)
    print(f"Feature dim: {X.shape[1]}")

    print("\n--- Ridge (baseline) ---")
    lodo_eval(X, y_pw, y_wl, meta_df, Ridge, {'alpha': 1.0}, name="Ridge")

    print("--- LightGBM 200 trees, 31 leaves ---")
    lgb_kw = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                  min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
                  random_state=42, verbose=-1)
    lodo_eval(X, y_pw, y_wl, meta_df, LGBMRegressor, lgb_kw, name="LGB_200")

    print("--- LightGBM 500 trees, 63 leaves ---")
    lgb_kw2 = dict(n_estimators=500, num_leaves=63, learning_rate=0.02,
                   min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                   random_state=42, verbose=-1)
    lodo_eval(X, y_pw, y_wl, meta_df, LGBMRegressor, lgb_kw2, name="LGB_500")

    print("--- XGBoost depth=5 ---")
    xgb_kw = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    lodo_eval(X, y_pw, y_wl, meta_df, XGBRegressor, xgb_kw, name="XGB_d5")

    print("--- Detailed breakdown (LGB_500) ---")
    lodo_detailed(X, y_pw, y_wl, meta_df, LGBMRegressor, lgb_kw2)

    # Feature importance on full data
    print("\n--- Feature importances (WL, full data) ---")
    feat_names = [
        'log_n_ff', 'log_die_area', 'log_ff_hpwl', 'log_ff_spacing', 'log_ff_density',
        'log_sqrt_nff_area', 'ff_cx', 'ff_cy', 'ff_x_std', 'ff_y_std', 'die_aspect', 'aspect_ratio',
        'log_n_active', 'log_n_total', 'avg_ds', 'std_ds', 'p90_ds', 'frac_ds4p',
        'frac_active', 'frac_ff', 'frac_buf_inv', 'log_cap_proxy',
        'rel_act', 'mean_sig_prob', 'tc_std_norm', 'frac_zero', 'frac_high_act',
        'log_n_nets', 'nets_per_maxTC',
        'f_clk_ghz', 't_clk_ns',
        'synth_type', 'synth_level', 'core_util', 'density', 'time_driven', 'routability',
        'cd', 'cs', 'mw', 'bd', 'log_cd', 'log_cs', 'log_mw', 'log_bd',
        'pw_phys1', 'pw_phys2', 'pw_phys3', 'pw_phys4', 'pw_phys5',
        'wl_phys1', 'wl_phys2', 'wl_phys3', 'wl_phys4', 'wl_phys5',
        'log_cap_f', 'log_n_ff_f', 'synth_agg', 'synth_delay_f',
        'area_cap', 'true_switch', 'log_act_pow', 'log_cd_dens',
        'log_cs_sp', 'log_mw_hpwl', 'util_density',
    ]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    m_wl = LGBMRegressor(**lgb_kw2)
    m_wl.fit(X_s, y_wl)
    imps = sorted(zip(feat_names[:X.shape[1]], m_wl.feature_importances_), key=lambda x: -x[1])
    for n, imp in imps[:20]:
        print(f"  {n}: {imp}")

    print("\n--- Feature importances (power, full data) ---")
    m_pw = LGBMRegressor(**lgb_kw2)
    m_pw.fit(X_s, y_pw)
    imps2 = sorted(zip(feat_names[:X.shape[1]], m_pw.feature_importances_), key=lambda x: -x[1])
    for n, imp in imps2[:20]:
        print(f"  {n}: {imp}")


if __name__ == '__main__':
    main()
