"""
Zero-shot absolute predictor v9: GRAPH-STRUCTURAL MESSAGE PASSING FEATURES.

Builds on v7 (power=32.0% / WL=21.2% LODO).

Key insight from user: handcrafted global DEF features (n_ff, comb_per_ff, die_area)
fail for zipdiv because they're all outside training distribution (n_ff 11x smaller).
The fix: LOCAL graph-structural features that generalize across design scales.

Two types of structural features (computable from DEF + timing_paths.csv for ANY design):

1. kNN FF Proximity (1-hop spatial SGC approximation):
   - For each FF: distance to k=4,8,16 nearest FF neighbors
   - Stats (mean, std, p25, p75, p90, max) across all FFs
   - Physical meaning: local FF density → routing distance per FF
   - Why better: kNN distance for zipdiv is only 1.4x above AES (vs n_ff being 11x below)

2. Timing Path Degree (skip graph structural feature):
   - For each FF: number of timing paths involving it (launch + capture)
   - Stats: mean, std, max, p90, coefficient-of-variation, Gini
   - AES mean=4.0 AND zipdiv mean=4.0 → SAME! Both in-distribution
   - CV differs: AES CV=7.9 (complex hub topology), zipdiv CV=1.2 (simple)

3. Ridge regression for WL (ensemble with LGB):
   - Tree models extrapolate by holding last leaf constant → bad for scale extrapolation
   - Ridge extrapolates LINEARLY → better when design scale is outside training
   - Blend: WL_pred = 0.5*Ridge + 0.5*LGB

These changes address the fundamental generalization problem:
- Local features → design-scale invariant
- Diverse models → robust extrapolation
"""

import re
import os
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.spatial import cKDTree

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'

DEF_CACHE_V7  = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE_V7 = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE_V7 = f'{BASE}/absolute_v7_timing_cache.pkl'

GRAPH_CACHE_V9 = f'{BASE}/absolute_v9_graph_cache.pkl'  # kNN + timing degree features

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}


# -----------------------------------------------------------------------
# DEF PARSER (unchanged from v7)
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
        # Store FF positions for graph feature computation
        'ff_xs': xs.tolist(), 'ff_ys': ys.tolist(),
    }


# -----------------------------------------------------------------------
# SAIF PARSER (unchanged from v7)
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
# TIMING PATH PARSER (unchanged from v7)
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
# GRAPH STRUCTURAL FEATURES (NEW IN V9)
# FF kNN proximity + timing path degree
# -----------------------------------------------------------------------

def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def compute_graph_features(df_f, tp_path, n_ff):
    """
    Compute graph-structural features from FF positions and timing paths.

    These features are LOCAL (per-FF neighborhood), not global design stats.
    They generalize better across design scales than global features like n_ff, die_area.

    Returns: dict with kNN distance stats and timing degree stats.
    """
    feats = {}

    # === 1. FF kNN proximity (spatial SGC approximation) ===
    # Build kNN graph from FF positions
    ff_xs = np.array(df_f.get('ff_xs', []))
    ff_ys = np.array(df_f.get('ff_ys', []))

    if len(ff_xs) >= 5:
        pts = np.column_stack([ff_xs, ff_ys])
        die_area = df_f['die_area']

        # Normalize positions to [0,1]² for kNN
        die_scale = np.sqrt(die_area)
        pts_norm = pts / die_scale

        tree = cKDTree(pts_norm)

        for k in [4, 8, 16]:
            k_eff = min(k + 1, len(ff_xs))  # +1 to exclude self
            dists, _ = tree.query(pts_norm, k=k_eff)
            # Exclude self (first neighbor, dist=0)
            if k_eff > 1:
                knn_dists = dists[:, 1:]  # exclude self
            else:
                knn_dists = dists
            mean_dist = knn_dists.mean(axis=1)  # per-FF mean distance to k neighbors

            # Stats across all FFs (in normalized units × die_scale = µm)
            feats[f'knn{k}_mean'] = float(mean_dist.mean())
            feats[f'knn{k}_std'] = float(mean_dist.std())
            feats[f'knn{k}_p25'] = float(np.percentile(mean_dist, 25))
            feats[f'knn{k}_p75'] = float(np.percentile(mean_dist, 75))
            feats[f'knn{k}_p90'] = float(np.percentile(mean_dist, 90))
            feats[f'knn{k}_max'] = float(mean_dist.max())
            feats[f'knn{k}_gini'] = float(_gini(mean_dist))
            # CV (coefficient of variation): scale-independent measure of FF clustering
            feats[f'knn{k}_cv'] = float(mean_dist.std() / (mean_dist.mean() + 1e-9))
    else:
        for k in [4, 8, 16]:
            for stat in ['mean', 'std', 'p25', 'p75', 'p90', 'max', 'gini', 'cv']:
                feats[f'knn{k}_{stat}'] = 0.0

    # === 2. Timing path degree (skip graph structural feature) ===
    try:
        tp = pd.read_csv(tp_path)
        all_ffs = list(tp['launch_flop']) + list(tp['capture_flop'])
        degree_counter = Counter(all_ffs)
        degrees = np.array(list(degree_counter.values()), dtype=float)

        # FFs with 0 timing paths
        n_ff_total = max(n_ff, 1)
        n_ff_in_paths = len(degree_counter)

        feats['tp_degree_mean'] = float(degrees.mean())
        feats['tp_degree_std'] = float(degrees.std())
        feats['tp_degree_max'] = float(degrees.max())
        feats['tp_degree_p90'] = float(np.percentile(degrees, 90))
        feats['tp_degree_gini'] = float(_gini(degrees))
        # CV: AES=7.9 (complex hub), zipdiv=1.2 (simple) → very discriminating
        feats['tp_degree_cv'] = float(degrees.std() / (degrees.mean() + 1e-9))
        feats['tp_frac_involved'] = float(n_ff_in_paths / n_ff_total)
        feats['tp_paths_per_ff'] = float(len(tp) / n_ff_total)
        # High-degree FFs (critical hubs): fraction with degree > 2×mean
        feats['tp_frac_hub'] = float((degrees > 2 * degrees.mean()).mean())
    except Exception:
        for stat in ['tp_degree_mean', 'tp_degree_std', 'tp_degree_max', 'tp_degree_p90',
                     'tp_degree_gini', 'tp_degree_cv', 'tp_frac_involved', 'tp_paths_per_ff',
                     'tp_frac_hub']:
            feats[stat] = 0.0

    return feats


# -----------------------------------------------------------------------
# CACHE (extended to include graph structural features)
# -----------------------------------------------------------------------

def build_caches(df):
    """Build/load caches. Starts from v7 caches if available."""
    pids = df['placement_id'].unique()

    # Load v7 caches
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
    if missing:
        print(f"Parsing {len(missing)} new placements for DEF/SAIF/timing...")
        for pid in missing:
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

    # Load or build graph features cache (v9-specific)
    if os.path.exists(GRAPH_CACHE_V9):
        with open(GRAPH_CACHE_V9, 'rb') as f:
            gc = pickle.load(f)
        missing_g = [p for p in pids if p not in gc]
        if missing_g:
            print(f"Computing graph features for {len(missing_g)} placements...")
    else:
        gc = {}
        missing_g = list(pids)
        print(f"Building graph feature cache for {len(missing_g)} placements...")

    for i, pid in enumerate(missing_g):
        if pid not in dc:
            continue
        row = df[df['placement_id'] == pid].iloc[0]
        def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
        tp_path = os.path.join(os.path.dirname(def_path), 'timing_paths.csv')
        n_ff = dc[pid].get('n_ff', 1)
        gc[pid] = compute_graph_features(dc[pid], tp_path, n_ff)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(missing_g)}")

    with open(GRAPH_CACHE_V9, 'wb') as f:
        pickle.dump(gc, f)
    print(f"Graph cache: {len(gc)} placements")

    return dc, sc, tc, gc


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
# FEATURE BUILDER
# -----------------------------------------------------------------------

def build_features(df, dc, sc, tc, gc):
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
        gf = gc.get(pid, {})
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

        # Normalizers: v7-proven
        pw_norm = n_ff * f_ghz * avg_ds
        wl_norm = np.sqrt(n_ff * die_area)
        if pw_norm < 1e-10 or wl_norm < 1e-10:
            continue

        # v9: design scale features
        area_per_ff = die_area / (n_ff + 1)
        hpwl_per_ff = ff_hpwl / (n_ff + 1)
        n_comb_total = df_f['n_comb']

        # === BASE FEATURES (v5/v7 + v9 additions) ===
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
            # v9: design-scale features
            np.log1p(area_per_ff),
            np.log1p(hpwl_per_ff),
            np.log1p(n_comb_total),
            comb_per_ff * np.log1p(n_ff),      # joint: zipdiv=32.2, in [16.9, 41.2]
            np.log1p(ff_spacing * comb_per_ff),
            np.log1p(mw / (ff_hpwl + 1)),
            np.log1p(cd / (ff_spacing + 1)),
            # v9: kNN graph structural features (message-passing approx, LOCAL topology)
            gf.get('knn4_mean', 0.0), gf.get('knn4_std', 0.0), gf.get('knn4_cv', 0.0),
            gf.get('knn8_mean', 0.0), gf.get('knn8_std', 0.0), gf.get('knn8_cv', 0.0),
            gf.get('knn16_mean', 0.0), gf.get('knn16_std', 0.0), gf.get('knn16_cv', 0.0),
            gf.get('knn4_p90', 0.0), gf.get('knn8_p90', 0.0), gf.get('knn16_p90', 0.0),
            gf.get('knn4_gini', 0.0), gf.get('knn8_gini', 0.0), gf.get('knn16_gini', 0.0),
            # v9: timing path degree structural features (skip graph)
            gf.get('tp_degree_mean', 0.0),
            gf.get('tp_degree_std', 0.0),
            gf.get('tp_degree_cv', 0.0),    # AES=7.9 vs zipdiv=1.2 → KEY discriminator
            gf.get('tp_degree_gini', 0.0),
            gf.get('tp_degree_max', 0.0),
            gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0),
            gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]

        # === TIMING FEATURES (only for power model — v7 proven) ===
        sm = tp_f['slack_mean']
        fn = tp_f['frac_neg']
        ft = tp_f['frac_tight']
        timing = [
            sm, tp_f['slack_std'], tp_f['slack_min'], tp_f['slack_p10'], tp_f['slack_p50'],
            fn, ft, tp_f['frac_critical'],
            tp_f['n_paths'] / (n_ff + 1),
            sm * frac_xor,
            sm * comb_per_ff,
            fn * comb_per_ff,
            ft * avg_ds,
            1.0 if sm > 1.5 else 0.0,
            1.0 if sm > 2.0 else 0.0,
            1.0 if sm > 3.0 else 0.0,
            np.log1p(sm),
            sm * f_ghz,
        ]

        rows_pw.append(base + timing)
        rows_wl.append(base)
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


def run_wl_ensemble(X_tr, y_tr, X_te, alpha=0.5):
    """Ensemble of LGB + Ridge for WL. Ridge extrapolates linearly (better for OOD scale)."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)

    lgb = LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.02,
                        min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                        random_state=42, verbose=-1)
    lgb.fit(Xtr_s, y_tr)
    pred_lgb = lgb.predict(Xte_s)

    ridge = Ridge(alpha=1.0, max_iter=10000)
    ridge.fit(Xtr_s, y_tr)
    pred_ridge = ridge.predict(Xte_s)

    return alpha * pred_lgb + (1 - alpha) * pred_ridge, pred_lgb, pred_ridge


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, pw_cls, pw_kw, wl_alpha=0.5, name=""):
    """LODO with ensemble WL model."""
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        # Power model (XGBoost + timing features)
        sc_pw = StandardScaler()
        X_tr_pw = sc_pw.fit_transform(X_pw[tr])
        X_te_pw = sc_pw.transform(X_pw[te])
        m_pw = pw_cls(**pw_kw)
        m_pw.fit(X_tr_pw, y_pw[tr])
        pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        # WL ensemble (LGB + Ridge)
        pred_wl_ens, _, _ = run_wl_ensemble(X_wl[tr], y_wl[tr], X_wl[te], alpha=wl_alpha)
        pred_wl = np.exp(pred_wl_ens) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl)

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%")

    mean_pw = np.mean(pw_mapes)
    mean_wl = np.mean(wl_mapes)
    print(f"  [{name}] mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return mean_pw, mean_wl


def zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, pw_cls, pw_kw, wl_alpha=0.5, name=""):
    """Train on all 4, evaluate on zipdiv."""
    X_pw_tr, X_wl_tr, y_pw_tr, y_wl_tr, meta_tr = build_features(df_train, dc, sc_cache, tc, gc)
    X_pw_te, X_wl_te, _, _, meta_te = build_features(df_test, dc, sc_cache, tc, gc)

    if len(meta_te) == 0:
        print(f"[{name}] No test samples!")
        return None, None

    # Power
    sc_pw = StandardScaler()
    X_tr_pw = sc_pw.fit_transform(X_pw_tr)
    X_te_pw = sc_pw.transform(X_pw_te)
    m_pw = pw_cls(**pw_kw)
    m_pw.fit(X_tr_pw, y_pw_tr)
    pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_te['pw_norm'].values
    mpw = mape(meta_te['power_total'].values, pred_pw)

    # WL ensemble
    pred_log_wl, pred_lgb, pred_ridge = run_wl_ensemble(X_wl_tr, y_wl_tr, X_wl_te, alpha=wl_alpha)
    wl_norm_v = meta_te['wl_norm'].values
    pred_wl = np.exp(pred_log_wl) * wl_norm_v
    mwl = mape(meta_te['wirelength'].values, pred_wl)
    mwl_lgb = mape(meta_te['wirelength'].values, np.exp(pred_lgb) * wl_norm_v)
    mwl_ridge = mape(meta_te['wirelength'].values, np.exp(pred_ridge) * wl_norm_v)

    print(f"[{name}] zipdiv: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")
    print(f"  WL breakdown: LGB={mwl_lgb:.1f}%  Ridge={mwl_ridge:.1f}%  Ensemble={mwl:.1f}%")
    print(f"  WL pred (ens): {pred_wl[:5].astype(int)}  true: {meta_te['wirelength'].values[:5].astype(int)}")
    print(f"  Power pred: {pred_pw[:5].round(5)}  true: {meta_te['power_total'].values[:5].round(5)}")
    return mpw, mwl


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Zero-Shot Absolute Predictor v9 — Graph-Structural + Ensemble")
    print("=" * 70)
    print("Power: XGBoost (timing+graph features)")
    print("WL:    LGB + Ridge ensemble (graph features, no timing)")
    print()

    df_train = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df_train = df_train.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Train rows: {len(df_train)}, designs: {df_train['design_name'].nunique()}")

    df_test = pd.read_csv(f'{DATASET}/unified_manifest_normalized_test.csv')
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"Test rows: {len(df_test)}, designs: {df_test['design_name'].nunique()}")

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    dc, sc_cache, tc, gc = build_caches(df_all)

    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df_train, dc, sc_cache, tc, gc)
    print(f"Features: X_pw={X_pw.shape}, X_wl={X_wl.shape}")
    print()

    xgb_pw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8,
                  random_state=42, verbosity=0)

    print("--- LODO on 4 training designs (LGB+Ridge ensemble, alpha=0.5) ---")
    lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw, wl_alpha=0.5,
              name="XGB+Ensemble")

    print("--- LODO: LGB only (baseline) ---")
    lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw, wl_alpha=1.0,
              name="XGB+LGBonly")

    print("\n--- Zipdiv zero-shot (LGB+Ridge ensemble) ---")
    zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, XGBRegressor, xgb_pw, wl_alpha=0.5,
                name="XGB+Ensemble")

    print("\n--- Zipdiv zero-shot (Ridge only for WL) ---")
    zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, XGBRegressor, xgb_pw, wl_alpha=0.0,
                name="XGB+RidgeOnly")


if __name__ == '__main__':
    main()
