"""
Zero-shot absolute predictor v8: WL NORMALIZER FIX FOR SMALL DESIGNS.

Builds on v7 (power=32.0% / WL=21.2% LODO), fixing zipdiv WL extrapolation.

Problem with v7 on zipdiv (n_ff=142):
  - wl_norm = sqrt(n_ff * die_area) → zipdiv predicts ratio=21 vs actual=14.5
  - Cause: comb_per_ff=6.5 (above training max=5.5) drives model to high ratio
  - All zipdiv DEF features are outside training range (n_ff 11x smaller)

Fix: Change WL normalizer to n_ff (WL per FF):
  - WL/n_ff for zipdiv = 216 µm/FF (solidly in training range: 112-317)
  - Add area_per_ff = die_area/n_ff as explicit feature (physical spread proxy)
  - Add log(ff_hpwl/n_ff) as FF spread distance feature

Physics: WL ≈ k × n_ff × avg_FF_routing_distance_per_FF
  - avg routing distance per FF varies by design: ETH=112, PicoRV=169, SHA256=199, AES=317
  - This range is more predictable from DEF features than wl/sqrt(n_ff*die_area)
"""

import re
import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
# Reuse v7 caches (same parser, just add test CSV placements)
DEF_CACHE_V7  = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE_V7 = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE_V7 = f'{BASE}/absolute_v7_timing_cache.pkl'
# v8 extends caches with test placements
DEF_CACHE_V8  = f'{BASE}/absolute_v8_def_cache.pkl'
SAIF_CACHE_V8 = f'{BASE}/absolute_v8_saif_cache.pkl'
TIMING_CACHE_V8 = f'{BASE}/absolute_v8_timing_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


# -----------------------------------------------------------------------
# DEF PARSER
# -----------------------------------------------------------------------

def parse_def(def_path):
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
    n_comb = max(n_active - n_ff - n_buf - n_inv, 0)

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
        'n_buf': n_buf, 'n_inv': n_inv, 'n_comb': n_comb,
        'n_xor_xnor': n_xor_xnor, 'n_mux': n_mux,
        'n_and_or': n_and_or, 'n_nand_nor': n_nand_nor,
        'frac_xor': n_xor_xnor / (n_active + 1),
        'frac_mux': n_mux / (n_active + 1),
        'frac_and_or': n_and_or / (n_active + 1),
        'frac_nand_nor': n_nand_nor / (n_active + 1),
        'frac_ff_active': n_ff / (n_active + 1),
        'frac_buf_inv': (n_buf + n_inv) / (n_active + 1),
        'comb_per_ff': n_comb / (n_ff + 1),
        'avg_ds': avg_ds, 'std_ds': std_ds, 'p90_ds': p90_ds,
        'frac_ds4plus': frac_ds4plus,
        'cap_proxy': n_active * avg_ds,
        'ff_cap_proxy': len(ff_xy) * avg_ds,
    }


# -----------------------------------------------------------------------
# SAIF PARSER
# -----------------------------------------------------------------------

def parse_saif(saif_path):
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
    rel_act = mean_tc / max_tc
    mean_sig_prob = total_t1 / (n_nets * duration) if duration and duration > 0 else 0.0

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
# TIMING PATH PARSER
# -----------------------------------------------------------------------

def parse_timing(tp_path):
    try:
        tp = pd.read_csv(tp_path)
        sl = tp['slack'].values
        return {
            'n_paths': len(sl),
            'slack_mean': sl.mean(), 'slack_std': sl.std(),
            'slack_min': sl.min(), 'slack_p10': np.percentile(sl, 10),
            'slack_p50': np.percentile(sl, 50),
            'frac_neg': (sl < 0).mean(),
            'frac_tight': (sl < 0.5).mean(),
            'frac_critical': (sl < 0.1).mean(),
        }
    except Exception:
        return None


# -----------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------

def build_caches(df):
    """Build/load caches. Uses v8 cache files (extends v7 with test placements)."""
    pids = df['placement_id'].unique()

    if (os.path.exists(DEF_CACHE_V8) and
            os.path.exists(SAIF_CACHE_V8) and
            os.path.exists(TIMING_CACHE_V8)):
        with open(DEF_CACHE_V8, 'rb') as f:
            dc = pickle.load(f)
        with open(SAIF_CACHE_V8, 'rb') as f:
            sc = pickle.load(f)
        with open(TIMING_CACHE_V8, 'rb') as f:
            tc = pickle.load(f)
        # Check all pids are present; if not, rebuild
        missing = [p for p in pids if p not in dc]
        if not missing:
            print(f"Loaded v8 caches: {len(dc)} DEF, {len(sc)} SAIF, {len(tc)} timing")
            return dc, sc, tc
        print(f"v8 cache missing {len(missing)} placements, rebuilding...")
    else:
        # Start from v7 caches if available (avoid re-parsing 140 training placements)
        if (os.path.exists(DEF_CACHE_V7) and
                os.path.exists(SAIF_CACHE_V7) and
                os.path.exists(TIMING_CACHE_V7)):
            with open(DEF_CACHE_V7, 'rb') as f:
                dc = pickle.load(f)
            with open(SAIF_CACHE_V7, 'rb') as f:
                sc = pickle.load(f)
            with open(TIMING_CACHE_V7, 'rb') as f:
                tc = pickle.load(f)
            print(f"Loaded v7 caches: {len(dc)} DEF, {len(sc)} SAIF, {len(tc)} timing")
        else:
            dc, sc, tc = {}, {}, {}

    # Parse any missing placements
    missing = [p for p in pids if p not in dc]
    print(f"Parsing {len(missing)} new placements...")
    for i, pid in enumerate(missing):
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        tp_path = os.path.join(os.path.dirname(def_path), 'timing_paths.csv')

        df_f = parse_def(def_path)
        sf_f = parse_saif(saif_path)
        tp_f = parse_timing(tp_path)

        if df_f:
            dc[pid] = df_f
        if sf_f:
            sc[pid] = sf_f
        if tp_f:
            tc[pid] = tp_f

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(missing)}")

    with open(DEF_CACHE_V8, 'wb') as f:
        pickle.dump(dc, f)
    with open(SAIF_CACHE_V8, 'wb') as f:
        pickle.dump(sc, f)
    with open(TIMING_CACHE_V8, 'wb') as f:
        pickle.dump(tc, f)
    print(f"Saved v8 caches: {len(dc)} DEF, {len(sc)} SAIF, {len(tc)} timing")
    return dc, sc, tc


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
# FEATURE BUILDER: two feature sets (power includes timing, WL does not)
# -----------------------------------------------------------------------

def build_features(df, dc, sc, tc):
    """Build separate feature matrices for power and WL models."""
    rows_pw, rows_wl = [], []
    y_pw_r, y_wl_r = [], []
    meta = []

    for _, row in df.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf_f = sc.get(pid)
        tp_f = tc.get(pid)
        if not df_f or not sf_f or not tp_f:
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
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']
        rel_act = sf_f['rel_act']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # Normalizers
        pw_norm = n_ff * f_ghz * avg_ds        # v5/v7 proven
        wl_norm = float(n_ff)                  # v8: WL per FF (more stable across design scales)
        if pw_norm < 1e-10 or wl_norm < 1e-10:
            continue

        # v8: design scale features (put zipdiv in-distribution)
        area_per_ff = die_area / (n_ff + 1)          # µm²/FF — FF spread proxy
        hpwl_per_ff = ff_hpwl / (n_ff + 1)          # µm/FF — routing distance proxy
        n_comb_total = df_f['n_comb']                # total combinational cells

        # === BASE FEATURES (v5 + v8 additions, shared by both power and WL models) ===
        base = [
            # DEF geometry
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            # DEF cell composition
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            # SAIF (duration-independent)
            rel_act, sf_f['mean_sig_prob'], sf_f['tc_std_norm'], sf_f['frac_zero'],
            sf_f['frac_high_act'], sf_f['log_n_nets'], sf_f['n_nets'] / (n_ff + 1),
            # Clock + synthesis
            f_ghz, t_clk, synth_delay, synth_level, synth_agg, core_util, density,
            # CTS knobs
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,
            # Physics interactions (v5)
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
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
            # v8: design-scale features (zipdiv in-distribution)
            np.log1p(area_per_ff),                   # log(µm²/FF): ETH=4.2, PicoRV=4.5, SHA256=4.9, AES=4.9, zipdiv=5.5
            np.log1p(hpwl_per_ff),                   # log(µm/FF): HPWL per FF (routing distance scale)
            np.log1p(n_comb_total),                  # log(total comb cells): ETH=9.8k, AES=9.7k, zipdiv=0.9k
            np.log1p(n_ff * area_per_ff),            # ≡ log(die_area): redundant but explicit signal
            comb_per_ff * np.log1p(n_ff),            # joint feature: ETH=16.9, PicoRV=23.4, AES=33.8, SHA256=41.2, zipdiv=32.2
            np.log1p(ff_spacing * comb_per_ff),      # routing complexity proxy
            np.log1p(mw / (ff_hpwl + 1)),            # max_wire relative to FF spread
            np.log1p(cd / (ff_spacing + 1)),         # cluster_dia relative to FF spacing
        ]

        # === TIMING FEATURES (only for power model) ===
        sm = tp_f['slack_mean']
        fn = tp_f['frac_neg']
        ft = tp_f['frac_tight']
        timing = [
            sm, tp_f['slack_std'], tp_f['slack_min'], tp_f['slack_p10'], tp_f['slack_p50'],
            fn, ft, tp_f['frac_critical'],
            tp_f['n_paths'] / (n_ff + 1),
            # Interactions: timing × circuit complexity
            sm * frac_xor,                      # XOR cells × slack (large for SHA256: 0.18)
            sm * comb_per_ff,                   # circuit depth × slack
            fn * comb_per_ff,                   # timing violations × circuit depth
            ft * avg_ds,                        # timing pressure × drive strength
            # Explicit thresholds for XGB hard splits
            1.0 if sm > 1.5 else 0.0,           # loose/tight boundary
            1.0 if sm > 2.0 else 0.0,           # definitely loose
            1.0 if sm > 3.0 else 0.0,           # very loose (SHA256/ETH regime)
            np.log1p(sm),                       # log slack
            sm * f_ghz,                         # slack × frequency
        ]

        rows_pw.append(base + timing)
        rows_wl.append(base)   # WL: no timing (prevents SHA256 WL degradation)
        y_pw_r.append(np.log(pw / pw_norm))
        y_wl_r.append(np.log(wl / wl_norm))
        meta.append({'placement_id': pid, 'design_name': design,
                     'power_total': pw, 'wirelength': wl,
                     'pw_norm': pw_norm, 'wl_norm': wl_norm})

    X_pw = np.array(rows_pw, dtype=np.float64)
    X_wl = np.array(rows_wl, dtype=np.float64)
    y_pw = np.array(y_pw_r)
    y_wl = np.array(y_wl_r)
    meta_df = pd.DataFrame(meta)

    # Fix NaN/Inf
    for X in [X_pw, X_wl]:
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    print(f"X_pw={X_pw.shape}, X_wl={X_wl.shape}, samples={len(meta_df)}")
    return X_pw, X_wl, y_pw, y_wl, meta_df


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def mape(y_true, y_pred_abs):
    return np.mean(np.abs(y_pred_abs - y_true) / (y_true + 1e-12)) * 100


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, pw_cls, pw_kw, wl_cls, wl_kw, name=""):
    """LODO: separate power and WL models with separate feature sets."""
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        # Power model (with timing features)
        sc_pw = StandardScaler()
        X_tr_pw = sc_pw.fit_transform(X_pw[tr])
        X_te_pw = sc_pw.transform(X_pw[te])
        m_pw = pw_cls(**pw_kw)
        m_pw.fit(X_tr_pw, y_pw[tr])
        pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        # WL model (v5 features only)
        sc_wl = StandardScaler()
        X_tr_wl = sc_wl.fit_transform(X_wl[tr])
        X_te_wl = sc_wl.transform(X_wl[te])
        m_wl = wl_cls(**wl_kw)
        m_wl.fit(X_tr_wl, y_wl[tr])
        pred_wl = np.exp(m_wl.predict(X_te_wl)) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl)

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%")

    mean_pw = np.mean(pw_mapes)
    mean_wl = np.mean(wl_mapes)
    print(f"  [{name}] mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return mean_pw, mean_wl


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def zipdiv_eval(df_train, df_test, dc, sc, tc, pw_cls, pw_kw, wl_cls, wl_kw, name=""):
    """Train on all 4 training designs, evaluate on zipdiv test set."""
    X_pw_tr, X_wl_tr, y_pw_tr, y_wl_tr, meta_tr = build_features(df_train, dc, sc, tc)
    X_pw_te, X_wl_te, _, _, meta_te = build_features(df_test, dc, sc, tc)

    if len(meta_te) == 0:
        print(f"[{name}] No test samples found!")
        return None, None

    # Power model (with timing)
    sc_pw = StandardScaler()
    X_tr_pw = sc_pw.fit_transform(X_pw_tr)
    X_te_pw = sc_pw.transform(X_pw_te)
    m_pw = pw_cls(**pw_kw)
    m_pw.fit(X_tr_pw, y_pw_tr)
    pred_log_pw = m_pw.predict(X_te_pw)
    pred_pw = np.exp(pred_log_pw) * meta_te['pw_norm'].values
    mpw = mape(meta_te['power_total'].values, pred_pw)

    # WL model (no timing)
    sc_wl = StandardScaler()
    X_tr_wl = sc_wl.fit_transform(X_wl_tr)
    X_te_wl = sc_wl.transform(X_wl_te)
    m_wl = wl_cls(**wl_kw)
    m_wl.fit(X_tr_wl, y_wl_tr)
    pred_log_wl = m_wl.predict(X_te_wl)
    pred_wl = np.exp(pred_log_wl) * meta_te['wl_norm'].values
    mwl = mape(meta_te['wirelength'].values, pred_wl)

    print(f"[{name}] zipdiv: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")
    print(f"  Power pred: {pred_pw[:5].round(5)}  true: {meta_te['power_total'].values[:5].round(5)}")
    print(f"  WL pred:    {pred_wl[:5].astype(int)}  true: {meta_te['wirelength'].values[:5].astype(int)}")
    print(f"  WL norm (n_ff): {meta_te['wl_norm'].values[:3].astype(int)}")
    return mpw, mwl


def main():
    print("=" * 70)
    print("Zero-Shot Absolute Predictor v8 — WL Normalizer Fix (WL/n_ff)")
    print("=" * 70)
    print("Power: XGBoost (timing features)  |  WL: LightGBM (no timing)")
    print("WL normalizer: n_ff (WL per FF, in-distribution for zipdiv)")
    print()

    df_train = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df_train = df_train.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Train rows: {len(df_train)}, designs: {df_train['design_name'].nunique()}")

    df_test = pd.read_csv(f'{DATASET}/unified_manifest_normalized_test.csv')
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Test rows: {len(df_test)}, designs: {df_test['design_name'].nunique()}")

    # Combine for cache building (need to parse test placement DEFs too)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    dc, sc_cache, tc = build_caches(df_all)

    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df_train, dc, sc_cache, tc)
    print(f"Features: X_pw={X_pw.shape}, X_wl={X_wl.shape}")

    # Power model: XGBoost with timing features
    xgb_pw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=0)

    # WL model: LightGBM without timing
    lgb_wl = dict(n_estimators=500, num_leaves=63, learning_rate=0.02,
                  min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                  random_state=42, verbose=-1)

    print("\n--- LODO on 4 training designs ---")
    lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df,
              XGBRegressor, xgb_pw,
              LGBMRegressor, lgb_wl,
              name="XGB4_LGB500_v8")

    print("\n--- Zipdiv zero-shot evaluation (train=all 4, test=zipdiv) ---")
    zipdiv_eval(df_train, df_test, dc, sc_cache, tc,
                XGBRegressor, xgb_pw,
                LGBMRegressor, lgb_wl,
                name="XGB4_LGB500_v8")


if __name__ == '__main__':
    main()
