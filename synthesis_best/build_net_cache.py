"""
build_net_cache.py — Parse DEF NETS section to compute:
  1. net_hpwl_sum: sum of per-net HPWL (for wire cap estimate, T1-B)
  2. rsmt_total: sum of RSMT estimates per net (for WL normalization, T2-E)
     rsmt_per_net = hpwl * (1 + 0.1 * log(max(pin_count, 1))) -- Rent correction
  3. rudy_score: RUDY congestion proxy (T2-F)
  4. net_degree_stats: mean/p90/max fanout

Output: net_features_cache.pkl {placement_id: feature_dict}
"""

import re, os, glob, sys, time, pickle
import numpy as np
import pandas as pd

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
PLACEMENT_DIR = f'{BASE}/dataset_with_def/placement_files'
OUT_CACHE = f'{BASE}/net_features_cache.pkl'


def parse_def_nets(def_path):
    """Parse DEF NETS section to get per-net pin coordinates."""
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

    # Parse all component positions for cross-referencing (simplified: just named instances)
    comp_pos = {}
    comp_pattern = r'-\s+(\S+)\s+sky130_fd_sc_hd__\S+\s+\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)'
    for name, x, y in re.findall(comp_pattern, content):
        comp_pos[name] = (float(x) / units, float(y) / units)

    if not comp_pos:
        return None

    # Parse NETS section: extract each net and its connections
    nets_section = re.search(r'\bNETS\s+\d+\s*;(.*?)\bEND NETS\b', content, re.DOTALL)
    if not nets_section:
        return None

    nets_text = nets_section.group(1)

    # Parse each net: - NETNAME ( INST PIN ) ... + ROUTED ... ;
    # We just need the instance references to get positions
    net_blocks = re.split(r'\n\s*-\s+', nets_text)

    net_hpwls = []
    net_degrees = []
    rudy_grid = np.zeros((16, 16))  # 16×16 RUDY grid

    grid_w = die_w / 16
    grid_h = die_h / 16

    for block in net_blocks[1:]:  # skip first empty
        # Get all (inst pin) pairs
        inst_refs = re.findall(r'\(\s*(\S+)\s+\S+\s*\)', block)
        # Filter to known components (skip ports/pins)
        positions = [comp_pos[inst] for inst in inst_refs if inst in comp_pos]

        if len(positions) < 2:
            continue

        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])

        hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
        net_hpwls.append(hpwl)
        net_degrees.append(len(positions))

        # RUDY: increment grid cells covered by this net's bounding box
        gx0 = int((xs.min() - x0) / grid_w)
        gx1 = int((xs.max() - x0) / grid_w)
        gy0 = int((ys.min() - y0) / grid_h)
        gy1 = int((ys.max() - y0) / grid_h)
        gx0 = max(0, min(gx0, 15)); gx1 = max(0, min(gx1, 15))
        gy0 = max(0, min(gy0, 15)); gy1 = max(0, min(gy1, 15))
        area = max((gx1-gx0+1) * (gy1-gy0+1), 1)
        rudy_grid[gy0:gy1+1, gx0:gx1+1] += 1.0 / area

    if not net_hpwls:
        return None

    net_hpwls = np.array(net_hpwls)
    net_degrees = np.array(net_degrees)

    # RSMT estimate: HPWL × Rent correction factor
    rent_corr = 1.0 + 0.1 * np.log1p(net_degrees - 1)
    rsmt_est = net_hpwls * rent_corr

    # Sky130 wire cap: ~0.2 fF/um (M2 layer estimate)
    cap_per_um = 0.2e-15  # F/um
    wire_cap_total = net_hpwls.sum() * cap_per_um  # total estimated wire cap in F

    # RUDY stats
    rudy_flat = rudy_grid.flatten()
    rudy_mean = rudy_flat.mean()
    rudy_max = rudy_flat.max()
    rudy_p90 = np.percentile(rudy_flat, 90)
    rudy_cv = rudy_flat.std() / (rudy_flat.mean() + 1e-6)

    return {
        'net_hpwl_sum': net_hpwls.sum(),           # total HPWL across all nets (um)
        'net_hpwl_mean': net_hpwls.mean(),
        'net_hpwl_p90': np.percentile(net_hpwls, 90),
        'net_hpwl_max': net_hpwls.max(),
        'rsmt_total': rsmt_est.sum(),               # total RSMT estimate (um)
        'rsmt_mean': rsmt_est.mean(),
        'wire_cap_total': wire_cap_total,           # estimated total wire cap (F)
        'log_wire_cap': np.log1p(wire_cap_total * 1e15),  # in fF
        'n_nets_parsed': len(net_hpwls),
        'net_degree_mean': net_degrees.mean(),
        'net_degree_p90': np.percentile(net_degrees, 90),
        'net_degree_max': float(net_degrees.max()),
        'frac_high_fanout': (net_degrees > 10).mean(),
        'rudy_mean': rudy_mean,
        'rudy_max': rudy_max,
        'rudy_p90': rudy_p90,
        'rudy_cv': rudy_cv,
        'die_w': die_w, 'die_h': die_h,
    }


def build_cache():
    df = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
    all_pids = df['placement_id'].unique()
    print(f"{T()} Building net features cache for {len(all_pids)} placements...")
    sys.stdout.flush()

    cache = {}
    n_ok, n_fail = 0, 0

    for pid in sorted(all_pids):
        design = pid.split('_run_')[0]
        place_dir = f'{PLACEMENT_DIR}/{pid}'
        def_path = f'{place_dir}/{design}.def'

        if not os.path.exists(def_path):
            n_fail += 1
            continue

        result = parse_def_nets(def_path)
        if result is not None:
            cache[pid] = result
            n_ok += 1
        else:
            n_fail += 1

        if (n_ok + n_fail) % 50 == 0:
            print(f"  {T()} {n_ok+n_fail}/{len(all_pids)} done ({n_ok} ok, {n_fail} fail)")
            sys.stdout.flush()

    print(f"{T()} Done: {n_ok} ok, {n_fail} fail. Saving to {OUT_CACHE}")
    with open(OUT_CACHE, 'wb') as f:
        pickle.dump(cache, f)

    # Show sample
    k = list(cache.keys())[0]
    print(f"\n  Sample ({k}):")
    for key, val in cache[k].items():
        if isinstance(val, float):
            print(f"    {key}: {val:.6g}")
        else:
            print(f"    {key}: {val}")

    # Design-level wire cap stats
    print(f"\n  Wire cap by design:")
    for design in ['aes','ethmac','picorv32','sha256']:
        wcs = [v['wire_cap_total']*1e12 for pid,v in cache.items() if design in pid]
        if wcs:
            print(f"    {design}: wire_cap=[{min(wcs):.2f},{max(wcs):.2f}] pF mean={np.mean(wcs):.2f} pF")

    return cache


if __name__ == '__main__':
    build_cache()
    print(f"\n{T()} DONE")
