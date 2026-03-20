"""
Zero-shot absolute predictor v5: RATIO REGRESSION.

Key insight: Normalize power/WL by physics-derived baselines that remove
most cross-design variation, then predict the residual ratio with ML.

  y_pw = log(power / (n_ff × f_clk × avg_drive))   → within 15-50% across designs
  y_wl = log(WL / sqrt(n_ff × die_area))             → within 2x across designs

The residual ratio is much easier to predict with structural features because
the bulk of design-type variation is already removed by the normalizer.

Features: DEF cell composition (frac_xor, frac_mux, comb/ff ratio),
SAIF relative activity, synth_strategy, CTS knobs, f_clk.
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
DEF_CACHE_V5 = f'{BASE}/absolute_v5_def_cache.pkl'
SAIF_CACHE_V5 = f'{BASE}/absolute_v5_saif_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


# -----------------------------------------------------------------------
# RICH DEF PARSER: cell composition + geometry
# -----------------------------------------------------------------------

def parse_def_v5(def_path):
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

    # Drive strengths of active cells
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

    # FF positions
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
        'n_buf': n_buf, 'n_inv': n_inv, 'n_comb': n_comb,
        # CELL TYPE COMPOSITION (key new features!)
        'n_xor_xnor': n_xor_xnor, 'n_mux': n_mux,
        'n_and_or': n_and_or, 'n_nand_nor': n_nand_nor,
        'frac_xor': n_xor_xnor / (n_active + 1),
        'frac_mux': n_mux / (n_active + 1),
        'frac_and_or': n_and_or / (n_active + 1),
        'frac_nand_nor': n_nand_nor / (n_active + 1),
        'frac_ff_active': n_ff / (n_active + 1),
        'frac_buf_inv': (n_buf + n_inv) / (n_active + 1),
        'comb_per_ff': n_comb / (n_ff + 1),
        # Drive strengths
        'avg_ds': avg_ds, 'std_ds': std_ds, 'p90_ds': p90_ds,
        'frac_ds4plus': frac_ds4plus,
        # Power proxies
        'cap_proxy': n_active * avg_ds,
        'ff_cap_proxy': len(ff_xy) * avg_ds,
    }


# -----------------------------------------------------------------------
# SAIF PARSER (same as v4)
# -----------------------------------------------------------------------

def parse_saif_v5(saif_path):
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

    tc_arr = np.array(tc_vals)
    mean_tc = total_tc / n_nets
    rel_act = mean_tc / max_tc   # duration-independent relative activity
    mean_sig_prob = total_t1 / (n_nets * duration) if duration and duration > 0 else 0

    return {
        'n_nets': n_nets, 'max_tc': max_tc, 'mean_tc': mean_tc,
        'rel_act': rel_act,
        'mean_sig_prob': mean_sig_prob,
        'tc_std_norm': tc_arr.std() / (mean_tc + 1),
        'frac_zero': (tc_arr == 0).mean(),
        'frac_high_act': (tc_arr > mean_tc * 2).mean(),
        'log_n_nets': np.log1p(n_nets),
    }


# -----------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------

def build_caches(df):
    pids = df['placement_id'].unique()
    if os.path.exists(DEF_CACHE_V5) and os.path.exists(SAIF_CACHE_V5):
        with open(DEF_CACHE_V5, 'rb') as f:
            dc = pickle.load(f)
        with open(SAIF_CACHE_V5, 'rb') as f:
            sc = pickle.load(f)
        print(f"Loaded caches: {len(dc)} DEF, {len(sc)} SAIF")
        return dc, sc

    print(f"Building caches for {len(pids)} placements...")
    dc, sc = {}, {}
    for i, pid in enumerate(pids):
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        df_f = parse_def_v5(def_path)
        sf_f = parse_saif_v5(saif_path)
        if df_f:
            dc[pid] = df_f
        if sf_f:
            sc[pid] = sf_f
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pids)}")

    with open(DEF_CACHE_V5, 'wb') as f:
        pickle.dump(dc, f)
    with open(SAIF_CACHE_V5, 'wb') as f:
        pickle.dump(sc, f)
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
    synth_agg = synth_is_delay * level / 4.0   # 0-1 aggressiveness
    return synth_is_delay, level, synth_agg


# -----------------------------------------------------------------------
# FEATURE BUILDER with RATIO TARGETS
# -----------------------------------------------------------------------

def build_features_v5(df, dc, sc):
    rows, y_pw_ratio, y_wl_ratio, y_pw_abs, y_wl_abs, meta = [], [], [], [], [], []

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
        aspect_ratio = float(row.get('aspect_ratio', 1.0))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        rel_act = sf_f['rel_act']
        n_nets = sf_f['n_nets']

        # ---- RATIO TARGETS ----
        # Power normalizer: n_ff × f_clk × avg_drive captures most cross-design variation
        pw_normalizer = n_ff * f_ghz * avg_ds
        # WL normalizer: sqrt(n_ff × die_area) captures routing tree scale
        wl_normalizer = np.sqrt(n_ff * die_area)

        # Prevent div by zero
        if pw_normalizer < 1e-10 or wl_normalizer < 1e-10:
            continue

        log_pw_ratio = np.log(pw / pw_normalizer)   # target to predict
        log_wl_ratio = np.log(wl / wl_normalizer)   # target to predict

        # ---- FEATURES ----
        feat = [
            # DEF geometry (log scale)
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], aspect_ratio,
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],

            # DEF cell composition (KEY FEATURES)
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            # Drive strengths
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),

            # SAIF (duration-independent)
            rel_act, sf_f['mean_sig_prob'],
            sf_f['tc_std_norm'], sf_f['frac_zero'], sf_f['frac_high_act'],
            sf_f['log_n_nets'],
            n_nets / (n_ff + 1),   # nets per FF

            # Clock + synthesis (CRITICAL for power)
            f_ghz, t_clk,
            synth_delay, synth_level, synth_agg,
            core_util, density,

            # CTS knobs
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,

            # Physics interactions for residual
            frac_xor * comb_per_ff,                    # XOR-intensive combinational
            rel_act * frac_xor,                         # XOR + activity
            rel_act * (1 - df_f['frac_ff_active']),    # comb activity
            synth_delay * avg_ds,                       # DELAY synth → bigger cells
            synth_agg * f_ghz,                          # high-speed synthesis at high freq
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),                        # cluster count proxy
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),       # full activity proxy

            # Cross-design alignment features
            np.log1p(frac_xor * n_active),              # absolute XOR cell count proxy
            np.log1p(frac_mux * n_active),              # absolute MUX count proxy
            np.log1p(comb_per_ff * n_ff),               # total combinational cells
        ]

        rows.append(feat)
        y_pw_ratio.append(log_pw_ratio)
        y_wl_ratio.append(log_wl_ratio)
        y_pw_abs.append(pw)
        y_wl_abs.append(wl)
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl,
                     'pw_norm': pw_normalizer, 'wl_norm': wl_normalizer})

    X = np.array(rows, dtype=np.float64)
    y_pw_r = np.array(y_pw_ratio)
    y_wl_r = np.array(y_wl_ratio)
    meta_df = pd.DataFrame(meta)

    # Fix NaN
    for c in range(X.shape[1]):
        bad = ~np.isfinite(X[:, c])
        if bad.any():
            X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    print(f"Feature matrix: {X.shape}")
    print(f"Power ratio range: exp[{y_pw_r.min():.2f},{y_pw_r.max():.2f}] = [{np.exp(y_pw_r.min()):.3e},{np.exp(y_pw_r.max()):.3e}]")
    print(f"WL ratio range: exp[{y_wl_r.min():.2f},{y_wl_r.max():.2f}] = [{np.exp(y_wl_r.min()):.3f},{np.exp(y_wl_r.max()):.3f}]")
    return X, y_pw_r, y_wl_r, meta_df


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def mape(y_true, y_pred_log):
    y_pred = np.exp(np.clip(y_pred_log, -20, 20))
    return np.mean(np.abs(y_pred - y_true) / (y_true + 1e-12)) * 100


def lodo_ratio_eval(X, y_pw_r, y_wl_r, meta_df, model_cls, kwargs, name=""):
    """LODO: predict ratio, denormalize, compute MAPE."""
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        # Power: predict log(pw / pw_norm), denormalize with pw_norm
        m_pw = model_cls(**kwargs)
        m_pw.fit(X_tr, y_pw_r[tr])
        pred_pw_ratio = m_pw.predict(X_te)
        pred_pw_abs = np.exp(pred_pw_ratio) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, np.log(pred_pw_abs + 1e-20))

        # WL: predict log(wl / wl_norm), denormalize with wl_norm
        m_wl = model_cls(**kwargs)
        m_wl.fit(X_tr, y_wl_r[tr])
        pred_wl_ratio = m_wl.predict(X_te)
        pred_wl_abs = np.exp(pred_wl_ratio) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, np.log(pred_wl_abs + 1e-20))

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"  {held}: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")

    print(f"  [{name}] mean → power={np.mean(pw_mapes):.1f}%  WL={np.mean(wl_mapes):.1f}%\n")
    return np.mean(pw_mapes), np.mean(wl_mapes)


def check_normalizer_quality(meta_df):
    """Check how well normalizers reduce cross-design variation."""
    print("\n=== Normalizer Quality Check ===")
    print("  Power normalizer: n_ff × f_clk × avg_ds")
    print("  WL normalizer: sqrt(n_ff × die_area)")
    for design in meta_df['design_name'].unique():
        d = meta_df[meta_df['design_name'] == design]
        # log(power / pw_norm) should be similar across designs
        print(f"  {design}: mean(power/pw_norm)={d['power_total'].values.mean()/d['pw_norm'].values.mean():.3e}  "
              f"cv_wl={(d['wirelength']/d['wl_norm']).std()/(d['wirelength']/d['wl_norm']).mean():.3f}")


def lodo_detailed(X, y_pw_r, y_wl_r, meta_df, model_cls, kwargs):
    for held in meta_df['design_name'].unique():
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])

        m_pw = model_cls(**kwargs)
        m_pw.fit(X_tr, y_pw_r[tr])
        m_wl = model_cls(**kwargs)
        m_wl.fit(X_tr, y_wl_r[tr])

        pred_pw = np.exp(m_pw.predict(X_te)) * meta_df[te]['pw_norm'].values
        pred_wl = np.exp(m_wl.predict(X_te)) * meta_df[te]['wl_norm'].values
        true_pw = meta_df[te]['power_total'].values
        true_wl = meta_df[te]['wirelength'].values

        mpw = mape(true_pw, np.log(pred_pw + 1e-20))
        mwl = mape(true_wl, np.log(pred_wl + 1e-20))
        print(f"\n=== {held} ===")
        print(f"  Power: true=[{true_pw.min():.4f},{true_pw.max():.4f}]  "
              f"pred=[{pred_pw.min():.4f},{pred_pw.max():.4f}]  MAPE={mpw:.1f}%")
        print(f"  WL:    true=[{true_wl.min():.0f},{true_wl.max():.0f}]  "
              f"pred=[{pred_wl.min():.0f},{pred_wl.max():.0f}]  MAPE={mwl:.1f}%")


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Zero-Shot Absolute Predictor v5 — Ratio Regression")
    print("=" * 65)

    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Rows: {len(df)}, designs: {df['design_name'].nunique()}")

    dc, sc = build_caches(df)
    X, y_pw_r, y_wl_r, meta_df = build_features_v5(df, dc, sc)

    check_normalizer_quality(meta_df)

    print("\n--- Ridge (baseline) ---")
    lodo_ratio_eval(X, y_pw_r, y_wl_r, meta_df, Ridge, {'alpha': 1.0}, name="Ridge")

    print("--- LightGBM 200 trees ---")
    lgb_kw = dict(n_estimators=200, num_leaves=31, learning_rate=0.05,
                  min_child_samples=5, reg_alpha=0.1, reg_lambda=0.1,
                  random_state=42, verbose=-1)
    lodo_ratio_eval(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw, name="LGB_200")

    print("--- LightGBM 500 trees ---")
    lgb_kw2 = dict(n_estimators=500, num_leaves=63, learning_rate=0.02,
                   min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                   random_state=42, verbose=-1)
    lodo_ratio_eval(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw2, name="LGB_500")

    print("--- XGBoost depth=4 ---")
    xgb_kw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    lodo_ratio_eval(X, y_pw_r, y_wl_r, meta_df, XGBRegressor, xgb_kw, name="XGB_d4")

    print("--- Detailed breakdown (LGB_500) ---")
    lodo_detailed(X, y_pw_r, y_wl_r, meta_df, LGBMRegressor, lgb_kw2)


if __name__ == '__main__':
    main()

def lodo_with_active_normalizer(X, y_pw_r, y_wl_r, meta_df, model_cls, kwargs, name=""):
    """Try n_active-based normalizer variant."""
    pass  # placeholder
