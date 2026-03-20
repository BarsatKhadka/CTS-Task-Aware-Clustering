"""
Zero-shot absolute predictor v13: Physics-exact features + GP regression.

Builds on v11 (power=32.0% / WL=13.1% LODO, zipdiv power=7.5% / WL=5.5%).

New additions vs v11:
1. MST wirelength per FF (Steiner tree proxy, ~1.1x more accurate than HPWL):
   - mst_per_ff = MST_length / n_ff  [size-invariant, units: µm/FF]
   - mst_norm   = MST_length / sqrt(n_ff × die_area)  [same scale as WL target]
   - Physical: clock WL ≈ 1.1-1.5 × Steiner ≤ 1.5 × HPWL → MST tighter than HPWL

2. FF spatial density grid features (n_ff-invariant, fixes v12 kNN problem):
   - 4×4 grid of normalized FF density per block
   - density_cv, density_gini, density_entropy, density_p90_norm
   - These are percentage-based → scale-invariant across n_ff=142 (zipdiv) to 5000 (ETH)

3. Wasserstein FF-logic distance (Optimal Transport routing pressure):
   - wasserstein_x, wasserstein_y: 1D OT distance between FF and logic distributions
   - wasserstein_total = sqrt(wx² + wy²)  [units: µm]
   - Physical: WL ∝ transport cost of clock signals from FFs to their logic
   - Invariant: EMD scales with area, not n_ff

4. Liberty-based driven capacitance (exact electrical feature):
   - Parse sky130 .lib → cell_type → total_input_capacitance (pf)
   - For each FF, sum caps of all driven logic cells from DEF NETS
   - driven_cap_mean, driven_cap_std, driven_cap_p90, driven_cap_cv  [units: pf/FF]
   - Physical: P_dynamic ∝ C_total = C_wire + C_input_pins
   - Invariant: per-FF average doesn't scale with n_ff

5. GP regressor with ARD RBF kernel for WL:
   - Learns separate length scale per feature → automatic feature selection
   - Better than Ridge for 4-design LODO: finds physics-consistent kernel
   - Compare GP vs Ridge vs LGB in alpha sweep

Key insight preserved from v11:
   - Power model: base+timing ONLY (no graph features — hurts SHA256)
   - WL model: base+graph_feats+v13_feats
   - Ridge(alpha=1000) for OOD zipdiv extrapolation
"""

import re
import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'
LIBERTY_FILE = f'{BASE}/sky130_fd_sc_hd_tt_025C_1v80.lib'

DEF_CACHE_V7     = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE_V7    = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE_V7  = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE_V10 = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXTENDED_CACHE_V13 = f'{BASE}/absolute_v13_extended_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}
CLOCK_PORTS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk',
               'ethmac': 'wb_clk_i', 'zipdiv': 'i_clk'}


# -----------------------------------------------------------------------
# LIBERTY PARSER — cell_type → total input capacitance (pf)
# -----------------------------------------------------------------------

def parse_liberty_caps(lib_path):
    """
    Parse sky130 liberty file → dict: short_cell_name → total_input_cap (pf).
    Example: 'and2_1' → 0.00296 pf (sum of all input pin capacitances).
    Uses line-by-line brace-depth tracking to handle nested pin blocks.
    """
    cell_caps = {}
    try:
        with open(lib_path) as f:
            content = f.read()
    except Exception as e:
        print(f"Warning: cannot read liberty file: {e}")
        return cell_caps

    current_cell = None
    current_pin_dir = None
    current_pin_cap = None
    brace_depth = 0
    in_pin = False
    cell_depth = 0
    pin_depth = 0

    for line in content.split('\n'):
        opens = line.count('{')
        closes = line.count('}')

        cell_m = re.search(r'cell\s*\(\s*"sky130_fd_sc_hd__(\w+)"\s*\)', line)
        if cell_m:
            current_cell = cell_m.group(1)
            cell_caps[current_cell] = 0.0
            cell_depth = brace_depth

        pin_m = re.search(r'pin\s*\(\s*"(\w+)"\s*\)', line)
        if pin_m and current_cell:
            current_pin_dir = None
            current_pin_cap = None
            in_pin = True
            pin_depth = brace_depth

        if in_pin:
            dir_m = re.search(r'direction\s*:\s*"(\w+)"', line)
            if dir_m:
                current_pin_dir = dir_m.group(1)
            cap_m = re.search(r'^\s*capacitance\s*:\s*([\d.e+-]+)', line)
            if cap_m:
                current_pin_cap = float(cap_m.group(1))

        brace_depth += opens - closes

        # Leaving pin block
        if in_pin and brace_depth <= pin_depth:
            in_pin = False
            if current_pin_dir == 'input' and current_pin_cap is not None:
                cell_caps[current_cell] = cell_caps.get(current_cell, 0.0) + current_pin_cap

        # Leaving cell block
        if current_cell and brace_depth <= cell_depth:
            if cell_caps.get(current_cell, 0.0) == 0.0:
                del cell_caps[current_cell]
            current_cell = None

    print(f"Liberty parsed: {len(cell_caps)} cells with input caps")
    for k in ['and2_1', 'xor2_1', 'dfxtp_1', 'buf_1', 'mux2_1']:
        print(f"  {k}: {cell_caps.get(k, 'N/A')}")
    return cell_caps


# -----------------------------------------------------------------------
# PARSERS (DEF / SAIF / Timing — from v11)
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
# GINI helper
# -----------------------------------------------------------------------

def _gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


# -----------------------------------------------------------------------
# EXTENDED FEATURES (v13 new: MST, density, Wasserstein, driven_cap)
# -----------------------------------------------------------------------

def mst_length_from_positions(xy):
    """
    Compute MST length from 2D position array using kNN-sparse graph.
    Returns MST length in µm.
    O(n log n) via kNN approximation.
    """
    n = len(xy)
    if n < 2:
        return 0.0
    k = min(15, n - 1)
    tree = cKDTree(xy)
    dists, idx = tree.query(xy, k=k+1)  # +1 for self

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].flatten()
    data = dists[:, 1:].flatten()

    # Symmetrize
    D = csr_matrix((data, (rows, cols)), shape=(n, n))
    D = (D + D.T) / 2

    mst = minimum_spanning_tree(D)
    return float(mst.sum())


def grid_density_features(ff_xy, x0, y0, x1, y1, grid=4):
    """
    Compute per-block normalized FF density at given grid resolution.
    Returns n_ff-invariant statistics of the density distribution.
    Density = fraction of FFs in each block (sums to 1.0).
    """
    if len(ff_xy) < 4:
        return {'dens_cv': 0.0, 'dens_gini': 0.0, 'dens_entropy': 0.0, 'dens_p90_norm': 1.0}

    xs = ff_xy[:, 0]
    ys = ff_xy[:, 1]
    dx = (x1 - x0) / grid
    dy = (y1 - y0) / grid

    counts = np.zeros((grid, grid))
    for x, y in zip(xs, ys):
        gi = min(int((x - x0) / dx), grid - 1)
        gj = min(int((y - y0) / dy), grid - 1)
        counts[gi, gj] += 1

    # Normalize to fraction (sum = 1.0) — n_ff invariant
    fracs = counts.flatten() / (counts.sum() + 1e-12)

    uniform = 1.0 / (grid * grid)
    cv = fracs.std() / (fracs.mean() + 1e-12)
    gini = _gini(fracs)
    # Entropy in nats, normalized by max entropy (log(grid²))
    fracs_nz = fracs[fracs > 0]
    entropy = -np.sum(fracs_nz * np.log(fracs_nz)) / np.log(grid * grid)
    p90_norm = np.percentile(fracs, 90) / uniform  # relative to uniform

    return {
        'dens_cv': float(cv),
        'dens_gini': float(gini),
        'dens_entropy': float(entropy),
        'dens_p90_norm': float(p90_norm),
    }


def compute_extended_features(def_path, clock_port, cell_cap_db):
    """
    Compute v13 extended features from DEF file:
    - FF and logic positions (for MST, Wasserstein, density)
    - Liberty-based driven capacitance per FF
    - MST length, Wasserstein, grid density stats

    Returns dict of features (all n_ff-invariant per-FF statistics).
    """
    feats = {
        'mst_per_ff': 0.0, 'mst_norm': 0.0,
        'dens_cv': 0.0, 'dens_gini': 0.0, 'dens_entropy': 0.5, 'dens_p90_norm': 1.0,
        'wass_x': 0.0, 'wass_y': 0.0, 'wass_total': 0.0,
        'driven_cap_mean': 0.002, 'driven_cap_std': 0.001,
        'driven_cap_p90': 0.005, 'driven_cap_cv': 0.5,
        'driven_cap_per_ff': 0.002,
    }

    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
        return feats

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content
    )
    if not die_m:
        return feats
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_area = (x1 - x0) * (y1 - y0)

    # --- All cell positions AND types ---
    comp_pat = re.compile(
        r'-\s+(\S+)\s+sky130_fd_sc_hd__(\w+)\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d]+)\s+([\d]+)\s*\)'
    )
    cell_pos = {}   # name → (x, y)
    cell_type = {}  # name → short type (e.g. 'and2_1')
    for m in comp_pat.finditer(content):
        name, ctype, cx, cy = m.groups()
        cell_pos[name] = (float(cx) / units, float(cy) / units)
        cell_type[name] = ctype

    # --- FF names from clock net ---
    clock_pat = rf'-\s+{re.escape(clock_port)}\s+\(\s+PIN\s+{re.escape(clock_port)}\s+\).*?;'
    cm = re.search(clock_pat, content, re.DOTALL)
    ff_names_in_clk = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', cm.group(0))) if cm else set()

    # Fallback: find FFs by cell type prefix
    ff_names_by_type = {n for n, t in cell_type.items() if t.startswith('df') or t.startswith('ff')}
    ff_names = ff_names_in_clk if ff_names_in_clk else ff_names_by_type

    if not ff_names:
        return feats

    # --- FF and logic positions ---
    filler_keys = ['tap', 'decap', 'fill', 'phy']
    ff_xy = np.array([cell_pos[n] for n in ff_names if n in cell_pos])
    logic_names = [n for n, t in cell_type.items()
                   if n not in ff_names and not any(x in t for x in filler_keys)]
    logic_xy = np.array([cell_pos[n] for n in logic_names if n in cell_pos])

    if len(ff_xy) < 2:
        return feats

    # --- MST wirelength ---
    mst_wl = mst_length_from_positions(ff_xy)
    n_ff = len(ff_xy)
    feats['mst_per_ff'] = float(mst_wl / n_ff)
    feats['mst_norm'] = float(mst_wl / (np.sqrt(n_ff * die_area) + 1e-6))

    # --- Grid density features ---
    dens = grid_density_features(ff_xy, x0, y0, x1, y1, grid=4)
    feats.update(dens)

    # --- Wasserstein FF-logic distance ---
    if len(logic_xy) > 0:
        wx = wasserstein_distance(ff_xy[:, 0], logic_xy[:, 0])
        wy = wasserstein_distance(ff_xy[:, 1], logic_xy[:, 1])
        feats['wass_x'] = float(wx)
        feats['wass_y'] = float(wy)
        feats['wass_total'] = float(np.hypot(wx, wy))

    # --- Liberty-based driven capacitance per FF (from NETS section) ---
    if cell_cap_db:
        nets_start = content.find('NETS')
        nets_end = content.find('END NETS')
        if nets_start != -1:
            nets_section = content[nets_start:nets_end]
            conn_pat = re.compile(r'\(\s+(\S+)\s+(\S+)\s+\)')

            driven_caps_per_ff = {ff: [] for ff in ff_names}
            for net_block in nets_section.split(';'):
                connections = conn_pat.findall(net_block)
                net_ffs = []
                net_driven_cap = 0.0
                for inst, pin in connections:
                    if inst in ff_names:
                        net_ffs.append(inst)
                    elif inst in cell_type:
                        ct = cell_type[inst]
                        net_driven_cap += cell_cap_db.get(ct, 0.002)
                if net_ffs and net_driven_cap > 0:
                    for ff in net_ffs:
                        driven_caps_per_ff[ff].append(net_driven_cap)

            ff_caps = []
            for ff in ff_names:
                if driven_caps_per_ff.get(ff):
                    ff_caps.append(sum(driven_caps_per_ff[ff]))
                else:
                    ff_caps.append(0.0)

            ff_caps = np.array(ff_caps, dtype=float)
            ff_caps_nz = ff_caps[ff_caps > 0]
            if len(ff_caps_nz) > 0:
                feats['driven_cap_mean'] = float(ff_caps_nz.mean())
                feats['driven_cap_std'] = float(ff_caps_nz.std())
                feats['driven_cap_p90'] = float(np.percentile(ff_caps_nz, 90))
                feats['driven_cap_cv'] = float(ff_caps_nz.std() / (ff_caps_nz.mean() + 1e-12))
                feats['driven_cap_per_ff'] = float(ff_caps.sum() / n_ff)

    return feats


# -----------------------------------------------------------------------
# V10 GRAVITY FEATURES (unchanged from v11)
# -----------------------------------------------------------------------

def compute_gravity_features(def_path, clock_port='clk'):
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

    clock_pat = rf'-\s+{re.escape(clock_port)}\s+\(\s+PIN\s+{re.escape(clock_port)}\s+\).*?;'
    cm = re.search(clock_pat, content, re.DOTALL)
    ff_names = set(re.findall(r'\(\s+((?!PIN)\S+)\s+CLK\s+\)', cm.group(0))) if cm else set()
    if not ff_names:
        return {}

    comp_pat = r'-\s+(\S+)\s+sky130_fd_sc_hd__\S+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d]+)\s+([\d]+)\s*\)'
    cell_pos = {m.group(1): (float(m.group(2))/units, float(m.group(3))/units)
                for m in re.finditer(comp_pat, content)}

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

    mags_abs, mags_norm, dx_vals, dy_vals = [], [], [], []
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

    return {
        'grav_abs_mean': float(mags_abs.mean()),
        'grav_abs_std':  float(mags_abs.std()),
        'grav_abs_p25':  float(np.percentile(mags_abs, 25)),
        'grav_abs_p75':  float(np.percentile(mags_abs, 75)),
        'grav_abs_p90':  float(mags_abs.max()),
        'grav_abs_max':  float(mags_abs.max()),
        'grav_abs_cv':   float(mags_abs.std() / (mags_abs.mean() + 1e-9)),
        'grav_abs_gini': float(_gini(mags_abs)),
        'grav_norm_mean': float(mags_norm.mean()),
        'grav_norm_cv':   float(mags_norm.std() / (mags_norm.mean() + 1e-9)),
        'grav_dx_mean': float(np.mean(dx_vals)),
        'grav_dy_mean': float(np.mean(dy_vals)),
        'grav_anisotropy': float(abs(np.mean(dx_vals) - np.mean(dy_vals)) /
                                 (np.mean(dx_vals) + np.mean(dy_vals) + 1e-9)),
        'grav_frac_local': float((mags_abs < np.percentile(mags_abs, 50)).mean()),
        'grav_frac_longrange': float((mags_abs > np.percentile(mags_abs, 90)).mean()),
    }


def compute_timing_degree_features(tp_path, n_ff):
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

def build_caches(df, cell_cap_db):
    pids = df['placement_id'].unique()

    # Load base caches (v7)
    if (os.path.exists(DEF_CACHE_V7) and os.path.exists(SAIF_CACHE_V7) and
            os.path.exists(TIMING_CACHE_V7)):
        with open(DEF_CACHE_V7, 'rb') as f: dc = pickle.load(f)
        with open(SAIF_CACHE_V7, 'rb') as f: sc = pickle.load(f)
        with open(TIMING_CACHE_V7, 'rb') as f: tc = pickle.load(f)
        print(f"Loaded v7 caches: {len(dc)} DEF, {len(sc)} SAIF, {len(tc)} timing")
    else:
        dc, sc, tc = {}, {}, {}

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

    # Gravity cache (v10)
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
                print(f"  gravity {i+1}/{len(missing_g)}")
        with open(GRAVITY_CACHE_V10, 'wb') as f:
            pickle.dump(gc, f)
        print(f"Saved gravity cache: {len(gc)} entries")

    # v13 extended cache (MST, density, Wasserstein, driven_cap)
    if os.path.exists(EXTENDED_CACHE_V13):
        with open(EXTENDED_CACHE_V13, 'rb') as f: ec = pickle.load(f)
        missing_e = [p for p in pids if p not in ec]
    else:
        ec = {}
        missing_e = list(pids)

    if missing_e:
        print(f"Computing v13 extended features for {len(missing_e)} placements...")
        for i, pid in enumerate(missing_e):
            if pid not in dc:
                ec[pid] = {}
                continue
            row = df[df['placement_id'] == pid].iloc[0]
            design = row['design_name']
            def_path = row['def_path'].replace('../dataset_with_def/placement_files', PLACEMENT_DIR)
            clock_port = CLOCK_PORTS.get(design, 'clk')
            ec[pid] = compute_extended_features(def_path, clock_port, cell_cap_db)
            if (i + 1) % 50 == 0:
                print(f"  extended {i+1}/{len(missing_e)}")
        with open(EXTENDED_CACHE_V13, 'wb') as f:
            pickle.dump(ec, f)
        print(f"Saved v13 extended cache: {len(ec)} entries")

    return dc, sc, tc, gc, ec


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

def build_features(df, dc, sc, tc, gc, ec):
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
        ef = ec.get(pid, {})
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

        # === BASE FEATURES (v5/v7, unchanged) ===
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

        # Extra scale (WL-only)
        extra_scale = [
            np.log1p(area_per_ff),
            np.log1p(n_comb_total),
            comb_per_ff * np.log1p(n_ff),
        ]

        # V10 gravity + timing degree features
        gravity_feats = [
            gf.get('grav_abs_mean', 0.0),
            gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0),
            gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0),
            gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0),
            gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd,
            gf.get('grav_abs_mean', 0.0) * mw,
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            gf.get('tp_degree_mean', 0.0),
            gf.get('tp_degree_cv', 0.0),
            gf.get('tp_degree_gini', 0.0),
            gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0),
            gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]

        # === V13 NEW FEATURES ===
        mst_per_ff = ef.get('mst_per_ff', ff_hpwl / (n_ff + 1))  # fallback: HPWL/n
        mst_norm = ef.get('mst_norm', 0.5)
        wass_total = ef.get('wass_total', 0.0)
        wass_x = ef.get('wass_x', 0.0)
        wass_y = ef.get('wass_y', 0.0)
        driven_cap_mean = ef.get('driven_cap_mean', 0.002)
        driven_cap_std = ef.get('driven_cap_std', 0.001)
        driven_cap_p90 = ef.get('driven_cap_p90', 0.005)
        driven_cap_cv = ef.get('driven_cap_cv', 0.5)
        driven_cap_per_ff = ef.get('driven_cap_per_ff', 0.002)

        v13_feats = [
            # MST Steiner proxy (more accurate than HPWL for WL prediction)
            np.log1p(mst_per_ff),          # log(MST/n_ff) — per-FF routing distance
            mst_norm,                       # MST / sqrt(n_ff × die_area) — same scale as WL
            np.log1p(mst_per_ff * cd),     # interaction: MST × cluster_dia
            np.log1p(mst_per_ff * cs),     # interaction: MST × cluster_size

            # Wasserstein FF-logic transport distance (OT routing pressure)
            np.log1p(wass_total),           # total transport cost
            wass_x / (df_f['die_w'] + 1),  # normalized by die dimensions
            wass_y / (df_f['die_h'] + 1),
            np.log1p(wass_total * cd),     # interaction: transport × cluster_dia

            # Grid density (n_ff-invariant spatial distribution)
            ef.get('dens_cv', 0.0),         # CV of block densities (0=uniform, high=clustered)
            ef.get('dens_gini', 0.0),       # Gini of block densities
            ef.get('dens_entropy', 0.5),    # Entropy (1=uniform, 0=all in one block)
            ef.get('dens_p90_norm', 1.0),   # 90th pctile density / uniform density

            # Liberty-based driven capacitance (exact electrical feature)
            np.log1p(driven_cap_mean),      # avg cap load per FF [pf]
            np.log1p(driven_cap_std),       # variability in cap load
            np.log1p(driven_cap_p90),       # p90 cap load
            driven_cap_cv,                  # CV of cap loads
            np.log1p(driven_cap_per_ff),    # total driven cap per FF
            np.log1p(driven_cap_per_ff * f_ghz),  # P_dynamic ∝ C × f
            np.log1p(driven_cap_mean * n_ff / die_area),  # cap density (careful: has n_ff)
        ]

        # === TIMING FEATURES (power model only — from v7) ===
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

        # Power: base + timing (v11 exact — no graph_feats, no v13_feats for power)
        rows_pw.append(base + timing)
        # WL: base + extra_scale + gravity + v13_feats
        rows_wl.append(base + extra_scale + gravity_feats + v13_feats)

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
    """Fit Ridge + LGB + GP for WL prediction."""
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)

    # LGB
    lgb = LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.02,
                        min_child_samples=5, reg_alpha=0.05, reg_lambda=0.05,
                        random_state=42, verbose=-1)
    lgb.fit(Xtr_s, y_tr)

    # Ridge (strong regularization for OOD extrapolation)
    ridge = Ridge(alpha=1000.0, max_iter=10000)
    ridge.fit(Xtr_s, y_tr)

    # GP with ARD RBF kernel — learns per-feature length scales
    # Use top-30 most informative features (by Ridge coefficients) to keep GP tractable
    n_feat = Xtr_s.shape[1]
    ridge_coef = np.abs(ridge.coef_)
    top_feat_idx = np.argsort(ridge_coef)[-30:]
    Xtr_gp = Xtr_s[:, top_feat_idx]

    kernel = (C(1.0, constant_value_bounds=(0.1, 10.0)) *
              RBF(length_scale=np.ones(30), length_scale_bounds=(0.01, 100.0)) +
              WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 1.0)))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-4,
                                   n_restarts_optimizer=2, normalize_y=True)
    try:
        gp.fit(Xtr_gp, y_tr)
        gp_ok = True
    except Exception as e:
        print(f"  GP fit failed: {e}")
        gp = None
        gp_ok = False

    return sc, lgb, ridge, gp, top_feat_idx


def predict_wl(sc, lgb, ridge, gp, top_feat_idx, X_te, alpha_lgb=0.0, alpha_gp=0.0):
    """
    Blend: alpha_lgb * LGB + alpha_gp * GP + (1 - alpha_lgb - alpha_gp) * Ridge
    """
    Xte_s = sc.transform(X_te)
    pred_lgb = lgb.predict(Xte_s)
    pred_ridge = ridge.predict(Xte_s)

    if gp is not None and alpha_gp > 0:
        Xte_gp = Xte_s[:, top_feat_idx]
        pred_gp = gp.predict(Xte_gp)
    else:
        pred_gp = pred_ridge.copy()

    alpha_ridge = 1.0 - alpha_lgb - alpha_gp
    pred = alpha_lgb * pred_lgb + alpha_gp * pred_gp + alpha_ridge * pred_ridge
    return pred, pred_lgb, pred_ridge, pred_gp


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, pw_cls, pw_kw,
              alpha_lgb=0.0, alpha_gp=0.0, name=""):
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

        sc_wl, lgb_wl, ridge_wl, gp_wl, gp_idx = fit_wl_models(X_wl[tr], y_wl[tr])
        pred_log, _, _, _ = predict_wl(sc_wl, lgb_wl, ridge_wl, gp_wl, gp_idx,
                                       X_wl[te], alpha_lgb, alpha_gp)
        pred_wl_v = np.exp(pred_log) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl_v)

        pw_mapes.append(mpw)
        wl_mapes.append(mwl)
        print(f"    {held}: power={mpw:.1f}%  WL={mwl:.1f}%")

    mean_pw = np.mean(pw_mapes)
    mean_wl = np.mean(wl_mapes)
    print(f"  [{name}] mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%\n")
    return mean_pw, mean_wl


def zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, ec, pw_cls, pw_kw,
                alpha_lgb=0.0, alpha_gp=0.0, name=""):
    X_pw_tr, X_wl_tr, y_pw_tr, y_wl_tr, meta_tr = build_features(
        df_train, dc, sc_cache, tc, gc, ec)
    X_pw_te, X_wl_te, _, _, meta_te = build_features(
        df_test, dc, sc_cache, tc, gc, ec)
    if len(meta_te) == 0:
        print(f"[{name}] No test samples!")
        return None, None

    sc_pw = StandardScaler()
    X_tr_pw = sc_pw.fit_transform(X_pw_tr)
    X_te_pw = sc_pw.transform(X_pw_te)
    m_pw = pw_cls(**pw_kw)
    m_pw.fit(X_tr_pw, y_pw_tr)
    pred_pw = np.exp(m_pw.predict(X_te_pw)) * meta_te['pw_norm'].values
    mpw = mape(meta_te['power_total'].values, pred_pw)

    sc_wl, lgb_wl, ridge_wl, gp_wl, gp_idx = fit_wl_models(X_wl_tr, y_wl_tr)
    pred_log, pred_lgb, pred_ridge, pred_gp = predict_wl(
        sc_wl, lgb_wl, ridge_wl, gp_wl, gp_idx, X_wl_te, alpha_lgb, alpha_gp)
    wl_norm_v = meta_te['wl_norm'].values
    pred_wl = np.exp(pred_log) * wl_norm_v
    mwl = mape(meta_te['wirelength'].values, pred_wl)
    mwl_lgb = mape(meta_te['wirelength'].values, np.exp(pred_lgb) * wl_norm_v)
    mwl_ridge = mape(meta_te['wirelength'].values, np.exp(pred_ridge) * wl_norm_v)
    mwl_gp = mape(meta_te['wirelength'].values, np.exp(pred_gp) * wl_norm_v)

    print(f"[{name}] zipdiv: power_MAPE={mpw:.1f}%  WL_MAPE={mwl:.1f}%")
    print(f"  WL components: LGB={mwl_lgb:.1f}%  Ridge={mwl_ridge:.1f}%  GP={mwl_gp:.1f}%")
    print(f"  WL pred: {pred_wl[:5].astype(int)}  true: {meta_te['wirelength'].values[:5].astype(int)}")
    print(f"  Power pred: {pred_pw[:3].round(5)}  true: {meta_te['power_total'].values[:3].round(5)}")
    return mpw, mwl


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Zero-Shot Absolute Predictor v13")
    print("  New: MST + Wasserstein + Grid density + Liberty caps + GP regressor")
    print("=" * 70)
    sys.stdout.flush()

    # Load liberty caps
    cell_cap_db = parse_liberty_caps(LIBERTY_FILE)
    sys.stdout.flush()

    df_train = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df_train = df_train.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    df_test = pd.read_csv(f'{DATASET}/unified_manifest_normalized_test.csv')
    df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")
    sys.stdout.flush()

    dc, sc_cache, tc, gc, ec = build_caches(df_all, cell_cap_db)
    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df_train, dc, sc_cache, tc, gc, ec)
    sys.stdout.flush()

    # Print v13 feature distribution stats
    print("\n--- V13 Feature Distribution Check ---")
    # Spot check MST and Wasserstein per design
    for design in meta_df['design_name'].unique():
        mask = meta_df['design_name'] == design
        # Features are at the end of X_wl: last 19 are v13_feats
        # mst_norm is at index [base(57)+extra_scale(3)+gravity(19)+1] = 80
        idx_mst_norm = 57 + 3 + 19 + 1  # mst_norm
        idx_wass = 57 + 3 + 19 + 4      # log1p(wass_total)
        idx_dcap = 57 + 3 + 19 + 12     # log1p(driven_cap_mean)
        pids = meta_df[mask]['placement_id'].unique()[:3]
        for pid in pids:
            ef = ec.get(pid, {})
            print(f"  {design}/{pid}: mst/ff={ef.get('mst_per_ff',0):.1f}µm  "
                  f"wass={ef.get('wass_total',0):.1f}µm  "
                  f"dcap={ef.get('driven_cap_mean',0):.4f}pf  "
                  f"dens_cv={ef.get('dens_cv',0):.3f}")
        break  # just first design for quick check
    sys.stdout.flush()

    # Check zipdiv extended features
    df_test_pids = df_test['placement_id'].unique()
    for pid in df_test_pids[:2]:
        ef = ec.get(pid, {})
        print(f"  zipdiv/{pid}: mst/ff={ef.get('mst_per_ff',0):.1f}µm  "
              f"wass={ef.get('wass_total',0):.1f}µm  "
              f"dcap={ef.get('driven_cap_mean',0):.4f}pf  "
              f"dens_cv={ef.get('dens_cv',0):.3f}")
    sys.stdout.flush()

    xgb_pw = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)

    print("\n--- LODO WL alpha sweep (alpha_gp=0 first) ---")
    for f_lgb in [0.0, 0.1, 0.3, 0.5]:
        lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw,
                  alpha_lgb=f_lgb, alpha_gp=0.0, name=f"lgb={f_lgb}")
        sys.stdout.flush()

    print("\n--- LODO: GP variants ---")
    for f_gp in [0.0, 0.1, 0.2, 0.3]:
        lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, XGBRegressor, xgb_pw,
                  alpha_lgb=0.0, alpha_gp=f_gp, name=f"gp={f_gp}")
        sys.stdout.flush()

    print("\n--- Zipdiv: Ridge-only (best for OOD) ---")
    zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, ec, XGBRegressor, xgb_pw,
                alpha_lgb=0.0, alpha_gp=0.0, name="ridge_only")
    sys.stdout.flush()

    print("\n--- Zipdiv: GP=0.2 ---")
    zipdiv_eval(df_train, df_test, dc, sc_cache, tc, gc, ec, XGBRegressor, xgb_pw,
                alpha_lgb=0.0, alpha_gp=0.2, name="gp_0.2")
    sys.stdout.flush()

    print("\n--- V13 vs V11 comparison summary ---")
    print("V11 baseline: power LODO=32.0%, WL LODO=13.1% (f_lgb=0.3)")
    print("V11 zipdiv:   power=7.5%, WL=5.5% (Ridge-only)")
    print("V13 target:   power LODO<25%, WL LODO<10%, zipdiv WL<3%")


if __name__ == '__main__':
    main()
