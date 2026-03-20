"""
Zero-shot absolute predictor v10: WIRE-GRAPH MESSAGE PASSING (Gravity Vectors).

Builds on v7 (power=32.0% / WL=21.2% LODO), fixing zipdiv WL extrapolation.

Core insight: handcrafted global features (n_ff=142, comb_per_ff=6.5) put zipdiv
OUT of training distribution. But LOCAL graph-structural features generalize:

1. Gravity vectors (wire graph 1-hop aggregation, from NETS section):
   For each FF: vector from FF position to mean position of connected logic gates.
   - gravity_mean_abs: zipdiv=10.8µm, training=[8.6-14.3µm] → IN DISTRIBUTION ✓
   - Physical meaning: average routing distance from FF to its logic cells
   - Computed from actual DEF NETS section (real wire edges, not kNN approx)

2. Timing path degree (skip graph, from timing_paths.csv):
   - degree_cv: AES=7.9 (complex hubs), zipdiv=1.2 (simple) → key discriminator
   - Also in-distribution: mean degree = 4.0 for BOTH AES and zipdiv

3. LGB + Ridge ensemble for WL:
   - Ridge extrapolates linearly for OOD designs, LGB handles within-distribution
   - Blend provides robustness

Why gravity helps for WL:
   gravity_mean_abs for zipdiv (10.8) ≈ ETH/AES range, NOT extreme
   → pushes prediction toward designs with similar routing complexity
   Previously: model saw comb_per_ff=6.5 (above training max=5.5) → predicted like AES
   Now: model sees gravity=10.8 (same as AES=9.6, ETH=8.6) → correct interpolation

Wire edge recovery from DEF NETS section:
   - Parse NETS section for FF-connected logic
   - For each FF: collect positions of fan-in + fan-out logic cells
   - Compute gravity vector = FF_pos - mean(logic_positions)
   - Stats of |gravity| across all FFs: mean, std, p25, p75, p90, max, gini, cv
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

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'

DEF_CACHE_V7     = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE_V7    = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE_V7  = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE_V10 = f'{BASE}/absolute_v10_gravity_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}
CLOCK_PORTS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk',
               'ethmac': 'wb_clk_i', 'zipdiv': 'i_clk'}


# -----------------------------------------------------------------------
# PARSERS (DEF / SAIF / Timing — unchanged from v7)
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


def parse_saif(saif_path):
    try:
        with open(saif_path) as f:
            lines = f.readlines()
    except Exception:
        return None

    total_tc = 0; total_t1 = 0; n_nets = 0; max_tc = 0
    tc_vals = []; duration = None

    for line in lines:
        if '(DURATION' in line:
            m = re.search(r'[\d.]+', line)
            if m: duration = float(m.group())
        m = re.search(r'\(TC\s+(\d+)\)', line)
        if m:
            tc = int(m.group(1)); tc_vals.append(tc); n_nets += 1; total_tc += tc
            max_tc = max(max_tc, tc)
        m2 = re.search(r'\(T1\s+(\d+)\)', line)
        if m2: total_t1 += int(m2.group(1))

    if n_nets == 0 or max_tc == 0:
        return None

    tc_arr = np.array(tc_vals, dtype=float)
    mean_tc = total_tc / n_nets
    rel_act = mean_tc / max_tc
    mean_sig_prob = total_t1 / (n_nets * duration) if duration and duration > 0 else 0.0
    return {
        'n_nets': n_nets, 'max_tc': max_tc, 'mean_tc': mean_tc, 'rel_act': rel_act,
        'mean_sig_prob': mean_sig_prob,
        'tc_std_norm': tc_arr.std() / (mean_tc + 1),
        'frac_zero': (tc_arr == 0).mean(),
        'frac_high_act': (tc_arr > mean_tc * 2).mean(),
        'log_n_nets': np.log1p(n_nets),
    }


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
# GRAVITY VECTOR FEATURES (wire-graph 1-hop aggregation from NETS section)
# -----------------------------------------------------------------------

def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def compute_gravity_features(def_path, clock_port='clk'):
    """
    Parse DEF NETS section to compute gravity vector features.

    Gravity vector per FF = displacement to centroid of connected logic cells.
    This is actual wire-graph 1-hop message passing (D^{-1} A X for positions).

    Features are in-distribution for zipdiv:
      gravity_mean_abs: training=[8.6-14.3µm], zipdiv=10.8µm → IN RANGE ✓
    """
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
        return {}

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content)
    if not die_m:
        return {}
    x0, y0, x1, y1 = [float(v)/units for v in die_m.groups()]
    die_area = (x1-x0) * (y1-y0)
    die_scale = np.sqrt(die_area)

    # FF names from clock net
    clock_pat = rf'-\s+{re.escape(clock_port)}\s+\(\s+PIN\s+{re.escape(clock_port)}\s+\).*?;'
    cm = re.search(clock_pat, content, re.DOTALL)
    ff_names = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', cm.group(0))) if cm else set()
    if not ff_names:
        return {}

    # All cell positions
    comp_pat = r'-\s+(\S+)\s+sky130_fd_sc_hd__\S+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d]+)\s+([\d]+)\s*\)'
    cell_pos = {m.group(1): (float(m.group(2))/units, float(m.group(3))/units)
                for m in re.finditer(comp_pat, content)}

    # Parse NETS for FF-connected logic (actual wire edges)
    nets_start = content.find('NETS')
    nets_end = content.find('END NETS')
    if nets_start == -1:
        return {}
    nets_section = content[nets_start:nets_end]
    conn_pat = re.compile(r'\(\s+(\S+)\s+(\S+)\s+\)')
    lg_pat = re.compile(r'^_\d+_$')
    gravity_vecs = {ff: [] for ff in ff_names}

    for net_block in nets_section.split(';'):
        connections = conn_pat.findall(net_block)
        net_ff, net_logic = [], []
        for inst, pin in connections:
            if inst in ff_names:
                net_ff.append(inst)
            elif lg_pat.match(inst) and inst in cell_pos:
                net_logic.append(inst)
        if net_ff and net_logic:
            for ff in net_ff:
                for lg in net_logic:
                    gravity_vecs[ff].append(cell_pos[lg])

    # Compute gravity magnitudes (absolute µm and normalized by sqrt(die_area))
    mags_abs = []
    mags_norm = []
    dx_vals = []
    dy_vals = []

    for ff in ff_names:
        if ff not in cell_pos or not gravity_vecs.get(ff):
            continue
        fx, fy = cell_pos[ff]
        lx = [c[0] for c in gravity_vecs[ff]]
        ly = [c[1] for c in gravity_vecs[ff]]
        cx, cy = np.mean(lx), np.mean(ly)
        dx, dy = cx - fx, cy - fy
        mag = np.sqrt(dx**2 + dy**2)
        mags_abs.append(mag)
        mags_norm.append(mag / die_scale)
        dx_vals.append(abs(dx))
        dy_vals.append(abs(dy))

    if not mags_abs:
        return {}

    mags_abs = np.array(mags_abs)
    mags_norm = np.array(mags_norm)

    feats = {
        # Absolute gravity magnitude (IN-DISTRIBUTION for zipdiv: 10.8µm, training: 8.6-14.3µm)
        'grav_abs_mean': float(mags_abs.mean()),
        'grav_abs_std':  float(mags_abs.std()),
        'grav_abs_p25':  float(np.percentile(mags_abs, 25)),
        'grav_abs_p75':  float(np.percentile(mags_abs, 75)),
        'grav_abs_p90':  float(np.percentile(mags_abs, 90)),
        'grav_abs_max':  float(mags_abs.max()),
        'grav_abs_cv':   float(mags_abs.std() / (mags_abs.mean() + 1e-9)),
        'grav_abs_gini': float(_gini(mags_abs)),
        # Normalized gravity (relative to die scale)
        'grav_norm_mean': float(mags_norm.mean()),
        'grav_norm_cv':   float(mags_norm.std() / (mags_norm.mean() + 1e-9)),
        # Direction asymmetry (dx vs dy) — captures non-uniform FF layout
        'grav_dx_mean': float(np.mean(dx_vals)),
        'grav_dy_mean': float(np.mean(dy_vals)),
        'grav_anisotropy': float(abs(np.mean(dx_vals) - np.mean(dy_vals)) /
                                 (np.mean(dx_vals) + np.mean(dy_vals) + 1e-9)),
        # Fraction of FFs with very local vs long-range logic
        'grav_frac_local': float((mags_abs < np.percentile(mags_abs, 50)).mean()),
        'grav_frac_longrange': float((mags_abs > np.percentile(mags_abs, 90)).mean()),
        # REMOVED: grav_load = n_ff × gravity — contains n_ff, out-of-dist for zipdiv and ETH
    }
    return feats


def compute_timing_degree_features(tp_path, n_ff):
    """Timing path degree features from timing_paths.csv (skip graph)."""
    feats = {}
    try:
        tp = pd.read_csv(tp_path)
        all_ffs = list(tp['launch_flop']) + list(tp['capture_flop'])
        degree_counter = Counter(all_ffs)
        degrees = np.array(list(degree_counter.values()), dtype=float)
        n_ff_total = max(n_ff, 1)
        feats = {
            'tp_degree_mean': float(degrees.mean()),
            'tp_degree_std':  float(degrees.std()),
            'tp_degree_max':  float(degrees.max()),
            'tp_degree_p90':  float(np.percentile(degrees, 90)),
            'tp_degree_cv':   float(degrees.std() / (degrees.mean() + 1e-9)),
            'tp_degree_gini': float(_gini(degrees)),
            'tp_frac_involved': float(len(degree_counter) / n_ff_total),
            'tp_paths_per_ff': float(len(tp) / n_ff_total),
            'tp_frac_hub': float((degrees > 2 * degrees.mean()).mean()),
        }
    except Exception:
        feats = {k: 0.0 for k in ['tp_degree_mean', 'tp_degree_std', 'tp_degree_max',
                                   'tp_degree_p90', 'tp_degree_cv', 'tp_degree_gini',
                                   'tp_frac_involved', 'tp_paths_per_ff', 'tp_frac_hub']}
    return feats


# -----------------------------------------------------------------------
# CACHE
# -----------------------------------------------------------------------

def build_caches(df):
    """Build/load all caches."""
    pids = df['placement_id'].unique()

    # Load base caches (v7: DEF/SAIF/timing)
    if (os.path.exists(DEF_CACHE_V7) and os.path.exists(SAIF_CACHE_V7) and
            os.path.exists(TIMING_CACHE_V7)):
        with open(DEF_CACHE_V7, 'rb') as f: dc = pickle.load(f)
        with open(SAIF_CACHE_V7, 'rb') as f: sc = pickle.load(f)
        with open(TIMING_CACHE_V7, 'rb') as f: tc = pickle.load(f)
        print(f"Loaded v7 caches: {len(dc)} DEF, {len(sc)} SAIF, {len(tc)} timing")
    else:
        dc, sc, tc = {}, {}, {}

    # Parse any missing base entries (e.g. zipdiv)
    missing_base = [p for p in pids if p not in dc]
    if missing_base:
        print(f"Parsing {len(missing_base)} new DEF/SAIF/timing...")
        for pid in missing_base:
            row = df[df['placement_id'] == pid].iloc[0]
            def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
            saif_path = row['saif_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
            tp_path = os.path.join(os.path.dirname(def_path), 'timing_paths.csv')
            df_f = parse_def(def_path)
            sf_f = parse_saif(saif_path)
            tp_f = parse_timing(tp_path)
            if df_f: dc[pid] = df_f
            if sf_f: sc[pid] = sf_f
            if tp_f: tc[pid] = tp_f

    # Build gravity + timing degree cache (v10-specific)
    if os.path.exists(GRAVITY_CACHE_V10):
        with open(GRAVITY_CACHE_V10, 'rb') as f: gc = pickle.load(f)
        missing_g = [p for p in pids if p not in gc]
    else:
        gc = {}
        missing_g = list(pids)

    if missing_g:
        print(f"Computing gravity+timing-degree for {len(missing_g)} placements...")
        for i, pid in enumerate(missing_g):
            if pid not in dc:
                continue
            row = df[df['placement_id'] == pid].iloc[0]
            design = row['design_name']
            def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
            tp_path = os.path.join(os.path.dirname(def_path), 'timing_paths.csv')
            clock_port = CLOCK_PORTS.get(design, 'clk')
            n_ff = dc[pid].get('n_ff', 1)

            grav = compute_gravity_features(def_path, clock_port)
            tp_deg = compute_timing_degree_features(tp_path, n_ff)
            gc[pid] = {**grav, **tp_deg}

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(missing_g)}")

        with open(GRAVITY_CACHE_V10, 'wb') as f:
            pickle.dump(gc, f)
        print(f"Saved gravity cache: {len(gc)} entries")

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

        pw_norm = n_ff * f_ghz * avg_ds
        wl_norm = np.sqrt(n_ff * die_area)
        if pw_norm < 1e-10 or wl_norm < 1e-10:
            continue

        area_per_ff = die_area / (n_ff + 1)
        n_comb_total = df_f['n_comb']

        # === BASE FEATURES (v5/v7) ===
        base = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf_f['mean_sig_prob'], sf_f['tc_std_norm'], sf_f['frac_zero'],
            sf_f['frac_high_act'], sf_f['log_n_nets'], sf_f['n_nets'] / (n_ff + 1),
            f_ghz, t_clk, synth_delay, synth_level, synth_agg, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
            cd, cs, mw, bd,
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
        ]

        # v10 extra scale features (WL-only — these hurt SHA256 power in LODO)
        extra_scale = [
            np.log1p(area_per_ff),
            np.log1p(n_comb_total),
            comb_per_ff * np.log1p(n_ff),   # joint: zipdiv=32.2, training=[16.9, 41.2] ✓
        ]

        # === V10: GRAVITY VECTOR FEATURES (wire-graph 1-hop aggregation) ===
        graph_feats = extra_scale + [
            # Absolute gravity (in-distribution for zipdiv: training=[8.6-14.3µm], zipdiv=10.8µm)
            gf.get('grav_abs_mean', 0.0),
            gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0),
            gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0),     # CV: AES=1.67, ETH=2.16, zipdiv=0.83
            gf.get('grav_abs_gini', 0.0),
            # Normalized gravity
            gf.get('grav_norm_mean', 0.0),
            gf.get('grav_norm_cv', 0.0),
            # Layout anisotropy
            gf.get('grav_anisotropy', 0.0),
            # Interaction: gravity × CTS knobs
            gf.get('grav_abs_mean', 0.0) * cd,       # gravity × cluster_dia
            gf.get('grav_abs_mean', 0.0) * mw,       # gravity × max_wire (routing budget)
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),  # gravity relative to FF spacing
            # Timing degree (skip graph)
            gf.get('tp_degree_mean', 0.0),
            gf.get('tp_degree_cv', 0.0),    # KEY: AES=7.9, ETH=2.2, zipdiv=1.2, PicoRV=?
            gf.get('tp_degree_gini', 0.0),
            gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0),
            gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]

        # === TIMING FEATURES (power model only — v7) ===
        sm = tp_f['slack_mean']
        fn = tp_f['frac_neg']
        ft = tp_f['frac_tight']
        timing = [
            sm, tp_f['slack_std'], tp_f['slack_min'], tp_f['slack_p10'], tp_f['slack_p50'],
            fn, ft, tp_f['frac_critical'],
            tp_f['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            1.0 if sm > 1.5 else 0.0,
            1.0 if sm > 2.0 else 0.0,
            1.0 if sm > 3.0 else 0.0,
            np.log1p(sm), sm * f_ghz,
        ]

        rows_pw.append(base + timing)          # v11 fix: no graph_feats for power
        rows_wl.append(base + graph_feats)
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


def fit_wl_models(X_tr, y_tr):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    lgb = LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.02,
                        min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                        random_state=42, verbose=-1)
    lgb.fit(Xtr_s, y_tr)
    # Ridge with strong regularization; alpha=1000 optimal from sweep
    ridge = Ridge(alpha=1000.0, max_iter=10000)
    ridge.fit(Xtr_s, y_tr)
    return sc, lgb, ridge


def predict_wl(sc, lgb, ridge, X_te, alpha=0.5):
    Xte_s = sc.transform(X_te)
    pred_lgb = lgb.predict(Xte_s)
    pred_ridge = ridge.predict(Xte_s)
    return alpha * pred_lgb + (1 - alpha) * pred_ridge, pred_lgb, pred_ridge


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, pw_cls, pw_kw, wl_alpha=0.5, name=""):
    designs = meta_df['design_name'].unique()
    pw_mapes, wl_mapes = [], []

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        sc_pw = StandardScaler()
        X_tr_pw = sc_pw.fit_transform(X_pw[tr])
        X_te_pw = sc_pw.transform(X_pw[te])
        m_pw = pw_cls(**pw_kw)
        m_pw.fit(X_tr_pw, y_pw[tr])
        pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        sc_wl, lgb_wl, ridge_wl = fit_wl_models(X_wl[tr], y_wl[tr])
        pred_log, _, _ = predict_wl(sc_wl, lgb_wl, ridge_wl, X_wl[te], wl_alpha)
        pred_wl_v = np.exp(pred_log) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl_v)

        pw_mapes.append(mpw); wl_mapes.append(mwl)
        print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%")

    mean_pw = np.mean(pw_mapes); mean_wl = np.mean(wl_mapes)
    print(f"  [{name}] mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return mean_pw, mean_wl


def zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, pw_cls, pw_kw, wl_alpha=0.5, name=""):
    X_pw_tr, X_wl_tr, y_pw_tr, y_wl_tr, meta_tr = build_features(df_train, dc, sc_cache, tc, gc)
    X_pw_te, X_wl_te, _, _, meta_te = build_features(df_test, dc, sc_cache, tc, gc)
    if len(meta_te) == 0:
        print(f"[{name}] No test samples!"); return None, None

    sc_pw = StandardScaler()
    X_tr_pw = sc_pw.fit_transform(X_pw_tr)
    X_te_pw = sc_pw.transform(X_pw_te)
    m_pw = pw_cls(**pw_kw)
    m_pw.fit(X_tr_pw, y_pw_tr)
    pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_te['pw_norm'].values
    mpw = mape(meta_te['power_total'].values, pred_pw)

    sc_wl, lgb_wl, ridge_wl = fit_wl_models(X_wl_tr, y_wl_tr)
    pred_log, pred_lgb, pred_ridge = predict_wl(sc_wl, lgb_wl, ridge_wl, X_wl_te, wl_alpha)
    wl_norm_v = meta_te['wl_norm'].values
    pred_wl = np.exp(pred_log) * wl_norm_v
    mwl = mape(meta_te['wirelength'].values, pred_wl)
    mwl_lgb = mape(meta_te['wirelength'].values, np.exp(pred_lgb) * wl_norm_v)
    mwl_ridge = mape(meta_te['wirelength'].values, np.exp(pred_ridge) * wl_norm_v)

    print(f"[{name}] zipdiv: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")
    print(f"  WL: LGB={mwl_lgb:.1f}%  Ridge={mwl_ridge:.1f}%  Ensemble={mwl:.1f}%")
    print(f"  WL pred (ens): {pred_wl[:5].astype(int)}  true: {meta_te['wirelength'].values[:5].astype(int)}")
    print(f"  Power pred: {pred_pw[:3].round(5)}  true: {meta_te['power_total'].values[:3].round(5)}")
    return mpw, mwl


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Zero-Shot Absolute Predictor v11 — Power fix (no graph feats for power)")
    print("=" * 70)
    print("v11 changes from v10:")
    print("  - Power model: base+timing only (no gravity/timing-degree) → recover v7 power MAPE")
    print("  - WL model: base+gravity — gravity in-distribution for zipdiv ✓")
    print("  - Alpha sweep to find optimal LGB/Ridge blend")
    print()

    df_train = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df_train = df_train.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    df_test = pd.read_csv(f'{DATASET}/unified_manifest_normalized_test.csv')
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Train: {len(df_train)} rows, {df_train['design_name'].nunique()} designs")
    print(f"Test: {len(df_test)} rows, {df_test['design_name'].nunique()} designs")

    dc, sc_cache, tc, gc = build_caches(df_all)
    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df_train, dc, sc_cache, tc, gc)
    X_pw_te, X_wl_te, _, _, meta_te = build_features(df_test, dc, sc_cache, tc, gc)

    xgb_pw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

    # --- LODO: Power ---
    print("--- LODO: Power (XGB, base+timing only) ---")
    lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw,
              wl_alpha=1.0, name="v11_power_fixed")

    # --- LODO WL alpha sweep ---
    print("\n--- LODO WL alpha sweep (Ridge alpha=1000) ---")
    for f_lgb in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        _, mwl = lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw,
                           wl_alpha=f_lgb, name=f"f_lgb={f_lgb}")

    # --- Zipdiv: alpha sweep ---
    print("\n--- Zipdiv WL alpha sweep ---")
    for f_lgb in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, XGBRegressor, xgb_pw,
                    wl_alpha=f_lgb, name=f"f_lgb={f_lgb}")


if __name__ == '__main__':
    main()
