"""
Zero-shot absolute predictor v6: ADAPTIVE RATIO REGRESSION.

Key improvements over v5:
1. Extended SAIF TC percentiles (p10,p25,p50,p75,p90 / max_tc) — duration-independent
   → tc_std_norm separates SHA256 (1.54) from AES (4.30) but we need richer tail info
2. Adaptive two-component power normalizer:
   n_eff = n_ff + alpha * n_comb * rel_act
   → optimized alpha minimizes cross-design log-ratio variance on training designs
3. WL normalizer: also try HPWL-based (sqrt(n_ff * HPWL)) for designs with high AES wl_ratio
4. More interaction features: tc_cv, frac_zero*comb_per_ff, xor*tc_cv, etc.
"""

import re
import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize_scalar
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
DEF_CACHE_V6  = f'{BASE}/absolute_v6_def_cache.pkl'
SAIF_CACHE_V6 = f'{BASE}/absolute_v6_saif_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


# -----------------------------------------------------------------------
# DEF PARSER (same as v5)
# -----------------------------------------------------------------------

def parse_def_v6(def_path):
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
    die_w, die_h = x1 - x0, y1 - y0
    die_area = die_w * die_h

    ct = Counter(re.findall(r'sky130_fd_sc_hd__(\w+)', content))
    n_total = sum(ct.values())

    filler_keys = ['tap', 'decap', 'fill', 'phy']
    n_tap = sum(v for k, v in ct.items() if any(x in k for x in filler_keys))
    n_active = n_total - n_tap
    n_ff = sum(v for k, v in ct.items() if k.startswith('df') or k.startswith('ff'))
    n_buf = sum(v for k, v in ct.items() if k.startswith('buf'))
    n_inv = sum(v for k, v in ct.items() if k.startswith('inv'))
    n_xor_xnor = sum(v for k, v in ct.items() if k.startswith('xor') or k.startswith('xnor'))
    n_mux = sum(v for k, v in ct.items() if k.startswith('mux'))
    n_and_or = sum(v for k, v in ct.items() if k.startswith('and') or k.startswith('or'))
    n_nand_nor = sum(v for k, v in ct.items() if k.startswith('nand') or k.startswith('nor'))
    n_comb = n_active - n_ff - n_buf - n_inv

    # AOI/OAI multi-input gates (adder structures in datapaths)
    n_aoi_oai = sum(v for k, v in ct.items() if k.startswith('a2') or k.startswith('o2')
                    or k.startswith('a3') or k.startswith('o3') or k.startswith('a4'))

    active_ds = []
    for k, v in ct.items():
        if not any(x in k for x in filler_keys):
            m = re.search(r'_(\d+)$', k)
            if m:
                active_ds.extend([int(m.group(1))] * v)

    avg_ds = np.mean(active_ds) if active_ds else 1.0
    std_ds = np.std(active_ds) if len(active_ds) > 1 else 0.0
    p90_ds = np.percentile(active_ds, 90) if active_ds else 1.0
    frac_ds4plus = sum(1 for d in active_ds if d >= 4) / (len(active_ds) + 1)

    ff_pattern = r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    ff_xy = [(float(x) / units, float(y) / units) for _, x, y in re.findall(ff_pattern, content)]
    if not ff_xy:
        return None

    xs = np.array([p[0] for p in ff_xy])
    ys = np.array([p[1] for p in ff_xy])
    ff_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
    ff_bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1.0
    ff_spacing = np.sqrt(ff_bbox_area / max(len(ff_xy), 1))

    return {
        'die_area': die_area, 'die_w': die_w, 'die_h': die_h,
        'die_aspect': die_w / (die_h + 1e-6),
        'ff_hpwl': ff_hpwl, 'ff_spacing': ff_spacing,
        'ff_density': len(ff_xy) / die_area,
        'ff_cx': xs.mean() / die_w, 'ff_cy': ys.mean() / die_h,
        'ff_x_std': xs.std() / die_w, 'ff_y_std': ys.std() / die_h,
        'n_ff': len(ff_xy), 'n_active': n_active, 'n_total': n_total, 'n_tap': n_tap,
        'n_buf': n_buf, 'n_inv': n_inv, 'n_comb': max(n_comb, 0),
        'n_xor_xnor': n_xor_xnor, 'n_mux': n_mux,
        'n_and_or': n_and_or, 'n_nand_nor': n_nand_nor, 'n_aoi_oai': n_aoi_oai,
        'frac_xor': n_xor_xnor / (n_active + 1),
        'frac_mux': n_mux / (n_active + 1),
        'frac_and_or': n_and_or / (n_active + 1),
        'frac_nand_nor': n_nand_nor / (n_active + 1),
        'frac_aoi_oai': n_aoi_oai / (n_active + 1),
        'frac_ff_active': n_ff / (n_active + 1),
        'frac_buf_inv': (n_buf + n_inv) / (n_active + 1),
        'comb_per_ff': n_comb / (n_ff + 1),
        'avg_ds': avg_ds, 'std_ds': std_ds, 'p90_ds': p90_ds,
        'frac_ds4plus': frac_ds4plus,
        'cap_proxy': n_active * avg_ds,
        'ff_cap_proxy': len(ff_xy) * avg_ds,
    }


# -----------------------------------------------------------------------
# SAIF PARSER v6: extended TC distribution
# -----------------------------------------------------------------------

def parse_saif_v6(saif_path):
    try:
        with open(saif_path) as f:
            lines = f.readlines()
    except Exception:
        return None

    total_tc = 0
    total_t1 = 0
    n_nets = 0
    max_tc = 0
    tc_vals = []
    duration = None

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

    if n_nets == 0 or max_tc == 0:
        return None

    tc_arr = np.array(tc_vals, dtype=float)
    mean_tc = total_tc / n_nets
    rel_act = mean_tc / max_tc   # duration-independent

    # TC percentiles normalized by max_tc (duration-independent shape features)
    p10, p25, p50, p75, p90 = np.percentile(tc_arr, [10, 25, 50, 75, 90])
    tc_cv = tc_arr.std() / (mean_tc + 1)      # coefficient of variation
    tc_iqr_norm = (p75 - p25) / (max_tc + 1)  # IQR normalized
    tc_gini = _gini(tc_arr)                    # Gini of TC distribution

    mean_sig_prob = total_t1 / (n_nets * duration) if duration and duration > 0 else 0.0

    return {
        'n_nets': n_nets, 'max_tc': max_tc, 'mean_tc': mean_tc,
        'rel_act': rel_act,
        'mean_sig_prob': mean_sig_prob,
        'tc_std_norm': tc_arr.std() / (mean_tc + 1),
        'tc_cv': tc_cv,
        'tc_iqr_norm': tc_iqr_norm,
        'tc_gini': tc_gini,
        'frac_zero': (tc_arr == 0).mean(),
        'frac_high_act': (tc_arr > mean_tc * 2).mean(),
        'log_n_nets': np.log1p(n_nets),
        # Normalized percentiles (duration-independent)
        'tc_p10_norm': p10 / (max_tc + 1),
        'tc_p25_norm': p25 / (max_tc + 1),
        'tc_p50_norm': p50 / (max_tc + 1),
        'tc_p75_norm': p75 / (max_tc + 1),
        'tc_p90_norm': p90 / (max_tc + 1),
        # Ratios between percentiles (distribution shape, size-invariant)
        'tc_p90_p10_ratio': (p90 + 1) / (p10 + 1),  # skewness proxy
        'tc_p75_p25_ratio': (p75 + 1) / (p25 + 1),  # IQR ratio
        'tc_median_mean_ratio': (p50 + 1) / (mean_tc + 1),  # skew indicator
    }


def _gini(arr):
    """Gini coefficient of array (measures inequality of TC distribution)."""
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cumsum = np.cumsum(arr)
    return (2 * np.sum((np.arange(1, n + 1)) * arr) / (n * cumsum[-1]) - (n + 1) / n)


# -----------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------

def build_caches(df):
    pids = df['placement_id'].unique()
    if os.path.exists(DEF_CACHE_V6) and os.path.exists(SAIF_CACHE_V6):
        with open(DEF_CACHE_V6, 'rb') as f:
            dc = pickle.load(f)
        with open(SAIF_CACHE_V6, 'rb') as f:
            sc = pickle.load(f)
        print(f"Loaded caches: {len(dc)} DEF, {len(sc)} SAIF")
        return dc, sc

    print(f"Building caches for {len(pids)} placements...")
    dc, sc = {}, {}
    for i, pid in enumerate(pids):
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        df_f = parse_def_v6(def_path)
        sf_f = parse_saif_v6(saif_path)
        if df_f:
            dc[pid] = df_f
        if sf_f:
            sc[pid] = sf_f
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pids)}")

    with open(DEF_CACHE_V6, 'wb') as f:
        pickle.dump(dc, f)
    with open(SAIF_CACHE_V6, 'wb') as f:
        pickle.dump(sc, f)
    print(f"Caches saved: {len(dc)} DEF, {len(sc)} SAIF")
    return dc, sc


# -----------------------------------------------------------------------
# SYNTH STRATEGY
# -----------------------------------------------------------------------

def encode_synth(s):
    if pd.isna(s):
        return 0.5, 2.0, 0.5
    s = str(s).upper()
    synth_is_delay = 1.0 if 'DELAY' in s else 0.0
    try:
        level = float(s.split()[-1])
    except Exception:
        level = 2.0
    synth_agg = synth_is_delay * level / 4.0
    return synth_is_delay, level, synth_agg


# -----------------------------------------------------------------------
# ADAPTIVE NORMALIZER: find alpha that minimizes cross-design variation
# -----------------------------------------------------------------------

def find_adaptive_alpha(meta_rows, target='power'):
    """
    Power normalizer: n_eff * f_ghz * avg_ds  where n_eff = n_ff + alpha * n_comb * rel_act
    Find alpha that minimizes variance of log(target / normalizer) across designs
    (i.e., minimizes how much design-type matters after normalization).
    """
    designs = list(set(r['design'] for r in meta_rows))

    def cross_design_var(alpha):
        design_means = []
        for d in designs:
            rows_d = [r for r in meta_rows if r['design'] == d]
            ratios = []
            for r in rows_d:
                if target == 'power':
                    n_eff = r['n_ff'] + max(alpha, 0) * r['n_comb'] * r['rel_act']
                    norm = n_eff * r['f_ghz'] * r['avg_ds']
                else:
                    # WL: try sqrt(n_ff + alpha*n_comb) * sqrt(die_area)
                    n_eff = r['n_ff'] + max(alpha, 0) * r['n_comb']
                    norm = np.sqrt(n_eff * r['die_area'])
                if norm > 0:
                    ratios.append(np.log(r['target'] / norm))
            if ratios:
                design_means.append(np.mean(ratios))
        return np.var(design_means)

    result = minimize_scalar(cross_design_var, bounds=(0.0, 5.0), method='bounded')
    return max(result.x, 0.0)


# -----------------------------------------------------------------------
# FEATURE BUILDER
# -----------------------------------------------------------------------

def build_features_v6(df, dc, sc, alpha_pw=0.0, alpha_wl=0.0, verbose=True):
    """Build features. alpha_pw/wl control n_comb contribution to normalizer.
    Default 0.0 = same as v5 (proven stable). Can be set per LODO fold."""
    if verbose:
        print(f"  Normalizer alpha: power={alpha_pw:.3f}  WL={alpha_wl:.3f}")

    rows, y_pw_ratio, y_wl_ratio, meta = [], [], [], []

    for _, row in df.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf_f = sc.get(pid)
        if not df_f or not sf_f:
            continue
        pw = row['power_total']
        wl = row['wirelength']
        if not np.isfinite(pw) or not np.isfinite(wl) or pw <= 0 or wl <= 0:
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk

        synth_delay, synth_level, synth_agg = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        n_comb = df_f['n_comb']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']

        rel_act = sf_f['rel_act']
        tc_cv = sf_f['tc_cv']
        tc_gini = sf_f['tc_gini']
        frac_zero = sf_f['frac_zero']
        tc_p50_norm = sf_f['tc_p50_norm']
        tc_p90_norm = sf_f['tc_p90_norm']
        tc_p10_norm = sf_f['tc_p10_norm']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # ---- ADAPTIVE NORMALIZERS ----
        n_eff_pw = n_ff + alpha_pw * n_comb * rel_act
        pw_normalizer = n_eff_pw * f_ghz * avg_ds
        n_eff_wl = n_ff + alpha_wl * n_comb
        wl_normalizer = np.sqrt(n_eff_wl * die_area)

        if pw_normalizer < 1e-10 or wl_normalizer < 1e-10:
            continue

        log_pw_ratio = np.log(pw / pw_normalizer)
        log_wl_ratio = np.log(wl / wl_normalizer)

        # ---- FEATURE VECTOR ----
        # tc_cv × frac_xor is the KEY new feature:
        #   AES:     tc_cv=4.30, frac_xor=0.081 → 0.348 (high variance XOR → high power)
        #   SHA256:  tc_cv=1.54, frac_xor=0.059 → 0.091 (uniform XOR → lower power)
        #   ETH:     tc_cv=4.43, frac_xor=0.002 → 0.009
        #   PicoRV:  tc_cv=3.41, frac_xor=0.007 → 0.024
        feat = [
            # DEF geometry (same as v5)
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],

            # DEF cell composition (same as v5 + frac_aoi_oai)
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'], df_f['frac_aoi_oai'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            # Drive strengths (same as v5)
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),

            # SAIF (v5 features + 3 new: tc_cv, tc_gini, tc_p90_norm)
            rel_act, sf_f['tc_std_norm'], frac_zero, sf_f['frac_high_act'],
            sf_f['mean_sig_prob'], sf_f['log_n_nets'], sf_f['n_nets'] / (n_ff + 1),
            tc_cv, tc_gini, tc_p90_norm,           # NEW: richer TC distribution

            # Clock + synthesis (same as v5)
            f_ghz, t_clk,
            synth_delay, synth_level, synth_agg,
            core_util, density,

            # CTS knobs (same as v5)
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,

            # v5 physics interactions
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            synth_delay * avg_ds,
            synth_agg * f_ghz,
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),

            # v5 cross-design alignment
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),

            # NEW KEY INTERACTIONS
            tc_cv * frac_xor,              # HIGH for AES, LOW for SHA256 — breaks degeneracy
            tc_cv * comb_per_ff,           # variance × circuit depth
            tc_gini * comb_per_ff,         # inequality × complexity
            frac_zero * comb_per_ff,       # dead combinational logic
            tc_p90_norm * frac_xor,        # heavy tail + XOR = AES signature
        ]

        rows.append(feat)
        y_pw_ratio.append(log_pw_ratio)
        y_wl_ratio.append(log_wl_ratio)
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl,
                     'pw_norm': pw_normalizer, 'wl_norm': wl_normalizer})

    X = np.array(rows, dtype=np.float64)
    y_pw_r = np.array(y_pw_ratio)
    y_wl_r = np.array(y_wl_ratio)
    meta_df = pd.DataFrame(meta)

    # Fix NaN/Inf
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    if verbose:
        print(f"Feature matrix: {X.shape}")
        print(f"Power ratio: exp[{y_pw_r.min():.2f},{y_pw_r.max():.2f}] → [{np.exp(y_pw_r.min()):.3e},{np.exp(y_pw_r.max()):.3e}]")
        print(f"WL ratio:    exp[{y_wl_r.min():.2f},{y_wl_r.max():.2f}] → [{np.exp(y_wl_r.min()):.3f},{np.exp(y_wl_r.max()):.3f}]")
    return X, y_pw_r, y_wl_r, meta_df


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def mape(y_true, y_pred_abs):
    return np.mean(np.abs(y_pred_abs - y_true) / (y_true + 1e-12)) * 100


def lodo_eval(X, y_pw_r, y_wl_r, meta_df, model_cls, kwargs, name=""):
    """Standard LODO evaluation."""
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr])
        X_te_s = scaler.transform(X[te])

        m_pw = model_cls(**kwargs)
        m_pw.fit(X_tr_s, y_pw_r[tr])
        pred_pw_abs = np.exp(m_pw.predict(X_te_s)) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw_abs)

        m_wl = model_cls(**kwargs)
        m_wl.fit(X_tr_s, y_wl_r[tr])
        pred_wl_abs = np.exp(m_wl.predict(X_te_s)) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl_abs)

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%")

    mean_pw = np.mean(pw_mapes)
    mean_wl = np.mean(wl_mapes)
    print(f"  [{name}] mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return mean_pw, mean_wl


def check_normalizer_quality(meta_df):
    print("\n=== Normalizer quality (lower cv_between_designs = better) ===")
    for design in meta_df['design_name'].unique():
        d = meta_df[meta_df['design_name'] == design]
        pw_r = d['power_total'] / d['pw_norm']
        wl_r = d['wirelength'] / d['wl_norm']
        print(f"  {design}: mean_pw_ratio={pw_r.mean():.3e}  mean_wl_ratio={wl_r.mean():.3f}")
    # Cross-design variance
    design_pw = [meta_df[meta_df['design_name'] == d]['power_total'].values /
                 meta_df[meta_df['design_name'] == d]['pw_norm'].values
                 for d in meta_df['design_name'].unique()]
    design_means = [np.log(np.mean(x)) for x in design_pw]
    print(f"  log(pw_ratio) across designs: mean={np.mean(design_means):.3f} std={np.std(design_means):.3f}")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Zero-Shot Absolute Predictor v6 — Adaptive Ratio Regression")
    print("=" * 65)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Rows: {len(df)}, designs: {df['design_name'].nunique()}")

    dc, sc = build_caches(df)
    X, y_pw_r, y_wl_r, meta_df = build_features_v6(df, dc, sc)

    check_normalizer_quality(meta_df)

    print("\n--- Ridge (baseline) ---")
    lodo_eval(X, y_pw_r, y_wl_r, meta_df, Ridge, {'alpha': 1.0}, name="Ridge")

    print("--- LightGBM 200 ---")
    lgb_kw = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                  min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
                  random_state=42, verbose=-1)
    lodo_eval(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw, name="LGB_200")

    print("--- LightGBM 500 ---")
    lgb_kw2 = dict(n_estimators=500, num_leaves=63, learning_rate=0.02,
                   min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                   random_state=42, verbose=-1)
    lodo_eval(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw2, name="LGB_500")

    print("--- XGBoost depth=4 ---")
    xgb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    lodo_eval(X, y_pw_r, y_wl_r, meta_df, XGBRegressor, xgb_kw, name="XGB_d4")

    # ---- Per-fold adaptive alpha ----
    print("--- LightGBM 500 (heavy reg) ---")
    lgb_kw3 = dict(n_estimators=500, num_leaves=15, learning_rate=0.02,
                   min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0,
                   random_state=42, verbose=-1)
    lodo_eval(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw3, name="LGB_heavy")


if __name__ == '__main__':
    main()
