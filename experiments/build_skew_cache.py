"""
build_skew_cache.py — Build skew spatial feature cache from DEF + timing_paths.csv

For each placement:
  1. Parse DEF: FF name → (x, y) position map
  2. Parse timing_paths.csv: worst-K launch-capture FF pairs
  3. Compute spatial features of critical paths

Output: skew_spatial_cache.pkl  {placement_id: feature_dict}
"""

import re, os, glob, sys, time, pickle
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from collections import Counter

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
PLACEMENT_DIR = f'{BASE}/dataset_with_def/placement_files'
OUT_CACHE = f'{BASE}/skew_spatial_cache.pkl'

CLOCK_PORTS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk',
               'ethmac': 'wb_clk_i', 'zipdiv': 'i_clk'}


def parse_def_ff_positions(def_path):
    """Return dict: ff_name -> (x_norm, y_norm) and die dimensions."""
    try:
        with open(def_path) as f:
            content = f.read()
    except Exception:
        return None, None, None, None

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)', content)
    if not die_m:
        return None, None, None, None
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_w, die_h = x1 - x0, y1 - y0

    # Parse named FF instances: - NAME CELLTYPE + PLACED (x y) N ;
    ff_pattern = r'-\s+(\S+)\s+sky130_fd_sc_hd__df\w+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    ff_positions = {}
    for name, x, y in re.findall(ff_pattern, content):
        ff_positions[name] = (float(x) / units, float(y) / units)

    return ff_positions, die_w, die_h, (x0, y0)


def compute_skew_features(ff_positions, die_w, die_h, origin, timing_df, top_k=50):
    """
    Compute spatial features of critical timing paths.

    Returns dict of features.
    """
    if ff_positions is None or len(ff_positions) == 0:
        return None

    # All FF positions
    all_xs = np.array([p[0] for p in ff_positions.values()])
    all_ys = np.array([p[1] for p in ff_positions.values()])
    all_cx = all_xs.mean()
    all_cy = all_ys.mean()

    # Normalize by die dimensions
    dw = max(die_w, 1.0)
    dh = max(die_h, 1.0)

    # Get worst-K paths (most negative / smallest slack)
    top_paths = timing_df.nsmallest(top_k, 'slack')

    # Collect critical FF positions
    crit_launch_pos, crit_capture_pos, path_dists = [], [], []
    launch_counts = Counter()

    for _, row in top_paths.iterrows():
        lf = str(row['launch_flop']).strip()
        cf = str(row['capture_flop']).strip()
        if lf in ff_positions and cf in ff_positions:
            lp = ff_positions[lf]
            cp = ff_positions[cf]
            crit_launch_pos.append(lp)
            crit_capture_pos.append(cp)
            dist = np.sqrt((lp[0]-cp[0])**2 + (lp[1]-cp[1])**2)
            path_dists.append(dist)
            launch_counts[lf] += 1

    if not path_dists:
        # Fallback: use aggregate stats only
        return {
            'crit_mean_dist': 0.0, 'crit_max_dist': 0.0, 'crit_p90_dist': 0.0,
            'crit_ff_hpwl': 0.0, 'crit_cx_offset': 0.0, 'crit_cy_offset': 0.0,
            'crit_x_std': 0.0, 'crit_y_std': 0.0,
            'crit_frac_boundary': 0.0, 'crit_star_degree': 0.0,
            'crit_chain_frac': 0.0, 'n_unique_launch': 0.0,
            'crit_asymmetry': 0.0, 'crit_eccentricity': 0.0,
            'crit_density_ratio': 0.0,
        }

    path_dists = np.array(path_dists)
    all_crit = np.array(crit_launch_pos + crit_capture_pos)
    crit_xs, crit_ys = all_crit[:, 0], all_crit[:, 1]

    # Critical FF centroid vs all FF centroid
    crit_cx = crit_xs.mean()
    crit_cy = crit_ys.mean()
    cx_offset = abs(crit_cx - all_cx) / dw
    cy_offset = abs(crit_cy - all_cy) / dh

    # HPWL of critical FFs
    crit_hpwl = ((crit_xs.max() - crit_xs.min()) + (crit_ys.max() - crit_ys.min()))
    crit_hpwl_norm = crit_hpwl / (dw + dh)

    # Spread of critical FFs
    crit_x_std = crit_xs.std() / dw
    crit_y_std = crit_ys.std() / dh

    # Eccentricity (ratio of x to y spread)
    crit_eccentricity = max(crit_xs.std(), 1.0) / max(crit_ys.std(), 1.0)
    if crit_eccentricity > 1.0:
        crit_eccentricity = 1.0 / crit_eccentricity  # always <= 1, close to 1 = round

    # Fraction of critical FFs near die boundary (within 10%)
    boundary_margin = 0.1
    near_boundary = (
        (crit_xs < (all_xs.min() + boundary_margin * dw)) |
        (crit_xs > (all_xs.max() - boundary_margin * dw)) |
        (crit_ys < (all_ys.min() + boundary_margin * dh)) |
        (crit_ys > (all_ys.max() - boundary_margin * dh))
    )
    frac_boundary = near_boundary.mean()

    # Star pattern: if top launch FF appears in many paths
    top_launch_count = max(launch_counts.values()) if launch_counts else 0
    star_degree = top_launch_count / max(len(path_dists), 1)

    # Chain pattern: many unique launch FFs (chain) vs few (star)
    n_unique_launch = len(launch_counts)
    chain_frac = n_unique_launch / max(len(path_dists), 1)

    # Asymmetry: imbalance between left/right and top/bottom critical FF distribution
    left_frac = (crit_xs < (all_xs.min() + 0.5 * dw)).mean()
    top_frac = (crit_ys > (all_ys.min() + 0.5 * dh)).mean()
    asymmetry = abs(left_frac - 0.5) + abs(top_frac - 0.5)

    # Density ratio: critical FF density vs all FF density
    if len(all_crit) > 4 and len(all_xs) > 4:
        try:
            crit_area = crit_hpwl ** 2 + 1.0
            all_area = ((all_xs.max()-all_xs.min()) * (all_ys.max()-all_ys.min())) + 1.0
            density_ratio = (len(all_crit) / crit_area) / (len(all_xs) / all_area)
        except Exception:
            density_ratio = 1.0
    else:
        density_ratio = 1.0

    return {
        'crit_mean_dist': path_dists.mean() / (dw + dh),
        'crit_max_dist': path_dists.max() / (dw + dh),
        'crit_p90_dist': np.percentile(path_dists, 90) / (dw + dh),
        'crit_ff_hpwl': crit_hpwl_norm,
        'crit_cx_offset': cx_offset,
        'crit_cy_offset': cy_offset,
        'crit_x_std': crit_x_std,
        'crit_y_std': crit_y_std,
        'crit_frac_boundary': frac_boundary,
        'crit_star_degree': star_degree,
        'crit_chain_frac': chain_frac,
        'n_unique_launch': n_unique_launch / 50.0,  # normalize by top_k
        'crit_asymmetry': asymmetry,
        'crit_eccentricity': crit_eccentricity,
        'crit_density_ratio': min(density_ratio, 10.0),
        # Raw distances for interaction features
        'crit_max_dist_um': path_dists.max(),
        'crit_mean_dist_um': path_dists.mean(),
    }


def build_cache():
    df = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
    all_pids = df['placement_id'].unique()
    print(f"{T()} Building skew spatial cache for {len(all_pids)} placements...")
    sys.stdout.flush()

    cache = {}
    n_ok, n_fail = 0, 0

    for pid in sorted(all_pids):
        design = pid.split('_run_')[0]
        place_dir = f'{PLACEMENT_DIR}/{pid}'
        def_path = f'{place_dir}/{design}.def'
        timing_path = f'{place_dir}/timing_paths.csv'

        if not os.path.exists(def_path) or not os.path.exists(timing_path):
            n_fail += 1
            continue

        ff_positions, die_w, die_h, origin = parse_def_ff_positions(def_path)
        if ff_positions is None or len(ff_positions) == 0:
            n_fail += 1
            continue

        try:
            timing_df = pd.read_csv(timing_path)
            feats = compute_skew_features(ff_positions, die_w, die_h, origin, timing_df)
            if feats is not None:
                cache[pid] = feats
                n_ok += 1
            else:
                n_fail += 1
        except Exception as e:
            n_fail += 1

        if (n_ok + n_fail) % 50 == 0:
            print(f"  {T()} {n_ok+n_fail}/{len(all_pids)} done ({n_ok} ok, {n_fail} fail)")
            sys.stdout.flush()

    print(f"{T()} Done: {n_ok} ok, {n_fail} fail. Saving to {OUT_CACHE}")
    with open(OUT_CACHE, 'wb') as f:
        pickle.dump(cache, f)

    # Show sample
    k = list(cache.keys())[0]
    print(f"  Sample ({k}):")
    for key, val in cache[k].items():
        print(f"    {key}: {val:.4f}")

    return cache


if __name__ == '__main__':
    build_cache()
    print(f"{T()} DONE")
